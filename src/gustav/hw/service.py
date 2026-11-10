"""Hardware detection and planning as the daemon uses them.

Holds the policy the planner itself deliberately does not: remembering the machine's
fingerprint, noticing when it changes, and refusing to swap the base model behind your
back. Adapters are tied to the base they were trained on, so a base change becomes a
pending approval and nothing else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from gustav.config.schema import Config, HardwareConfig
from gustav.core.events import Events
from gustav.hw.detect import Detector, NvmlDetector, default_detector
from gustav.hw.ladder import Ladder, ModelSpec, load_ladder
from gustav.hw.memory import inference_estimate, training_estimate
from gustav.hw.planner import Plan, Planner, PlannerError
from gustav.hw.preflight import check_vram, vram_budget
from gustav.hw.speed import Measurements
from gustav.hw.types import Hardware
from gustav.paths import Paths
from gustav.store.db import Database

META_HARDWARE = "hardware"
META_FINGERPRINT = "hardware_fingerprint"
META_PLAN = "hardware_plan"
META_PENDING_PLAN = "hardware_plan_pending"
META_ACTIVE_BASE = "active_base_model"

APPROVAL_BASE_CHANGE = "base_change"


@dataclass
class PlanOutcome:
    plan: Plan | None
    #: "config" (pinned in config.yaml or a profile), "planner", or "cached".
    source: str
    changed: bool = False
    fingerprint_changed: bool = False
    approval_id: int | None = None
    messages: list[str] = field(default_factory=list)
    error: str | None = None

    def summary(self) -> str:
        if self.error:
            return f"no usable plan: {self.error}"
        assert self.plan is not None
        bits = [self.plan.summary()]
        if self.fingerprint_changed:
            bits.append("hardware changed")
        if self.approval_id:
            bits.append(f"base-model change waiting for approval #{self.approval_id}")
        return "; ".join(bits)


class HardwareService:
    def __init__(
        self,
        config: Config,
        paths: Paths,
        db: Database,
        events: Events,
        *,
        detector: Detector | None = None,
        ladder: Ladder | None = None,
    ):
        self.config = config
        self.paths = paths
        self.db = db
        self.events = events
        self.detector = detector or default_detector(paths.data_dir)
        self.ladder = ladder or load_ladder(config.hardware.ladder or None)
        self.measurements = Measurements(db)
        self._hardware: Hardware | None = None

    # --- detection ------------------------------------------------------
    def detect(self, *, refresh: bool = False) -> Hardware:
        if self._hardware is None or refresh:
            self._hardware = self.detector.detect()
            self.db.set_meta(META_HARDWARE, self._hardware.to_dict())
        return self._hardware

    @property
    def hardware(self) -> Hardware:
        return self.detect()

    def gpu_temperatures(self) -> dict[int, float]:
        if isinstance(self.detector, NvmlDetector):
            return self.detector.detect_temperatures()
        return {
            g.index: g.temperature_c for g in self.hardware.gpus if g.temperature_c is not None
        }

    def hottest_gpu_c(self) -> float | None:
        temps = self.gpu_temperatures()
        return max(temps.values()) if temps else None

    def gpu_pressure(self) -> tuple[int, float | None]:
        """(VRAM held by other processes, how busy the card is) — a cheap NVML poll.

        Used by the scheduler to stay out of the way of whatever else is on the GPU.
        """
        if isinstance(self.detector, NvmlDetector):
            try:
                fresh = self.detector.detect()
            except Exception:  # noqa: BLE001 - never let a poll break the tick
                return 0, None
        else:
            fresh = self.hardware
        plan = self.stored_plan()
        indices = (plan.inference_gpus if plan and plan.inference_gpus else None)
        gpus = [
            g for g in fresh.gpus if indices is None or g.index in indices
        ] or list(fresh.gpus)
        if not gpus:
            return 0, None
        foreign = max(g.foreign_mb for g in gpus)
        utilisations = [g.utilization_pct for g in gpus if g.utilization_pct is not None]
        return foreign, (max(utilisations) if utilisations else None)

    # --- the ladder, refined by what is actually on disk ----------------
    def _ladder_with_local_configs(self) -> Ladder:
        """Prefer a downloaded model's own ``config.json`` over the ladder's numbers."""
        overrides: list[ModelSpec] = []
        models_dir = self.paths.models
        if not models_dir.exists():
            return self.ladder
        for spec in self.ladder.models:
            candidate = models_dir / spec.id / "config.json"
            if candidate.exists():
                try:
                    overrides.append(ModelSpec.from_hf_config(candidate, spec_id=spec.id))
                except Exception as exc:  # noqa: BLE001 - a bad config.json must not stop startup
                    self.events.emit(
                        "hardware.config_unreadable",
                        f"could not read {candidate}: {exc}",
                        level="warning",
                    )
        return self.ladder.with_overrides(overrides) if overrides else self.ladder

    # --- plans ----------------------------------------------------------
    def stored_plan(self) -> Plan | None:
        raw = self.db.get_meta(META_PLAN)
        return Plan.from_dict(raw) if isinstance(raw, dict) else None

    def pending_plan(self) -> Plan | None:
        raw = self.db.get_meta(META_PENDING_PLAN)
        return Plan.from_dict(raw) if isinstance(raw, dict) else None

    @property
    def active_base_model(self) -> str | None:
        value = self.db.get_meta(META_ACTIVE_BASE)
        return value if isinstance(value, str) else None

    def compute_plan(
        self, *, balance: float | None = None, overrides: dict[str, Any] | None = None
    ) -> Plan:
        """Run the planner without storing anything. Used by ``ai plan-hardware``."""
        hw_config = self._hardware_config(balance=balance, overrides=overrides)
        planner = Planner(
            self.detect(),
            hw_config,
            ladder=self._ladder_with_local_configs(),
            measurements=self.measurements,
        )
        return planner.plan()

    def _hardware_config(
        self, *, balance: float | None = None, overrides: dict[str, Any] | None = None
    ) -> HardwareConfig:
        data = self.config.hardware.model_dump()
        if balance is not None:
            data["balance"] = balance
        data.update(overrides or {})
        return HardwareConfig.model_validate(data)

    def ensure_plan(self, *, force: bool = False, balance: float | None = None) -> PlanOutcome:
        """Make sure there is a usable plan, and say what changed.

        Called at startup and whenever the hardware fingerprint changes.
        """
        hardware = self.detect(refresh=force)
        fingerprint = hardware.fingerprint()
        previous = self.db.get_meta(META_FINGERPRINT)
        fingerprint_changed = bool(previous) and previous != fingerprint

        if fingerprint_changed:
            self.events.emit(
                "hardware.changed",
                f"the hardware fingerprint changed ({previous} -> {fingerprint}); re-planning",
                level="warning",
                before=previous,
                after=fingerprint,
                gpus=[g.describe() for g in hardware.gpus],
            )

        pinned = self.config.hardware.plan
        if pinned:
            outcome = self._use_pinned_plan(pinned, hardware, fingerprint_changed)
        else:
            stored = self.stored_plan()
            needs_plan = (
                force
                or stored is None
                or (fingerprint_changed and self.config.hardware.replan_on_hardware_change)
                or stored.balance != self.config.hardware.balance
            )
            if not needs_plan and stored is not None:
                outcome = PlanOutcome(plan=stored, source="cached")
            else:
                outcome = self._run_planner(hardware, fingerprint_changed)

        self.db.set_meta(META_FINGERPRINT, fingerprint)
        if outcome.plan is not None:
            self._check_plan_against_hardware(outcome)
            # The fit check fills in the VRAM figures for a pinned or cached plan, so
            # store it again: `ai status` reads the stored copy.
            self.db.set_meta(META_PLAN, outcome.plan.to_dict())
        return outcome

    def _use_pinned_plan(
        self, pinned: dict[str, Any], hardware: Hardware, fingerprint_changed: bool
    ) -> PlanOutcome:
        data = {
            "balance": self.config.hardware.balance,
            "min_tokens_per_second": self.config.hardware.min_tokens_per_second,
            "fingerprint": hardware.fingerprint(),
            "estimated_tokens_per_second": 0.0,
            "speed_measured": False,
            "vram_inference_mb": 0.0,
            "vram_training_mb": 0.0,
            "ram_required_mb": 0.0,
            "profile": self.config.hardware.profile,
            **pinned,
        }
        try:
            plan = Plan.from_dict(data)
        except TypeError as exc:
            return PlanOutcome(
                plan=None, source="config", error=f"the pinned plan is incomplete: {exc}"
            )
        plan.explanation = [
            f"Plan taken from {'profile ' + plan.profile if plan.profile != 'auto' else 'config.yaml'}"
            f" rather than computed. The planner will not override it."
        ]
        self.db.set_meta(META_PLAN, plan.to_dict())
        return PlanOutcome(
            plan=plan, source="config", changed=True, fingerprint_changed=fingerprint_changed
        )

    def _run_planner(self, hardware: Hardware, fingerprint_changed: bool) -> PlanOutcome:
        try:
            plan = self.compute_plan()
        except PlannerError as exc:
            self.events.emit("hardware.no_plan", str(exc).splitlines()[0], level="error")
            return PlanOutcome(
                plan=self.stored_plan(), source="cached", error=str(exc),
                fingerprint_changed=fingerprint_changed,
            )

        active_base = self.active_base_model
        approval_id: int | None = None
        messages: list[str] = []

        if active_base and plan.base_model != active_base:
            # Adapters belong to a base model. Never swap it automatically.
            approval_id = self._request_base_change(active_base, plan)
            messages.append(
                f"the planner would prefer {plan.base_model} over the active {active_base}; "
                f"approve with 'ai approve-base-change {approval_id}'"
            )
            self.db.set_meta(META_PENDING_PLAN, plan.to_dict())
            kept = self._plan_for_fixed_base(active_base, approval_id)
            if kept is None:
                return PlanOutcome(
                    plan=self.stored_plan(), source="cached",
                    error=(
                        f"the active base model {active_base} has no workable "
                        f"configuration on this hardware; approve #{approval_id} to move to "
                        f"{plan.base_model}, or adjust hardware.* and re-run 'ai plan-hardware'"
                    ),
                    approval_id=approval_id, messages=messages,
                    fingerprint_changed=fingerprint_changed,
                )
            plan = kept
        elif not active_base:
            # First plan on this machine: there are no adapters to invalidate.
            self.db.set_meta(META_ACTIVE_BASE, plan.base_model)

        self.db.set_meta(META_PLAN, plan.to_dict())
        if self.config.hardware.profile == "auto":
            plan.write(self.paths.generated_profile)
        self.events.emit(
            "hardware.planned",
            plan.summary(),
            plan=plan.base_model,
            quantization=plan.quantization,
            tokens_per_second=plan.estimated_tokens_per_second,
            fingerprint=plan.fingerprint,
        )
        return PlanOutcome(
            plan=plan, source="planner", changed=True,
            fingerprint_changed=fingerprint_changed, approval_id=approval_id, messages=messages,
        )

    def _plan_for_fixed_base(self, active_base: str, approval_id: int) -> Plan | None:
        """A plan that keeps the base model the adapters were trained on.

        If `balance` is low enough that even the current base misses the adjusted speed
        floor, we drop the speed requirement rather than abandon the base: a working plan
        for the weights we actually have beats no plan at all. Only the floor is relaxed —
        every other knob stays as `balance` set it — and the warning says so.
        """
        try:
            plan = self.compute_plan(overrides={"prefer_model": active_base})
            plan.warnings.append(
                f"kept the active base model {active_base}; approval #{approval_id} "
                f"would move to the planner's preferred model"
            )
            return plan
        except PlannerError:
            pass
        try:
            plan = self.compute_plan(
                overrides={"prefer_model": active_base, "min_tokens_per_second": 0.0}
            )
        except PlannerError:
            return None
        # Report the configured floor, not the relaxed one we planned with.
        plan.min_tokens_per_second = self.config.hardware.min_tokens_per_second
        plan.warnings.append(
            f"kept the active base model {active_base}, but it cannot meet the speed floor "
            f"that balance {self.config.hardware.balance} implies; the floor was relaxed to "
            f"the configured {self.config.hardware.min_tokens_per_second:.0f} tok/s. "
            f"Approve #{approval_id} to switch to a model that can."
        )
        return plan

    def _check_plan_against_hardware(self, outcome: PlanOutcome) -> None:
        """A plan that was pinned or cached may no longer fit. Say so loudly."""
        plan = outcome.plan
        if plan is None or not self.hardware.has_gpu:
            return
        ladder = self._ladder_with_local_configs()
        try:
            model = ladder.model(plan.base_model)
            quant = ladder.quant(plan.quantization)
        except KeyError as exc:
            outcome.messages.append(str(exc))
            return
        index = plan.inference_gpus[0] if plan.inference_gpus else self.hardware.gpus[0].index
        try:
            gpu = self.hardware.gpu(index)
        except KeyError:
            outcome.messages.append(
                f"the plan serves on GPU{index}, which this machine does not have"
            )
            return
        budget = vram_budget(gpu, reserve_fraction=self.config.hardware.vram_reserve_fraction)
        estimate = inference_estimate(
            model, quant, seq_len=plan.max_seq_len, kv_dtype=plan.kv_cache_dtype, adapter_count=1
        )
        result = check_vram(estimate, budget)
        plan.vram_inference_mb = estimate.total_mb
        if not result.ok:
            message = f"the active plan does not fit this machine: {result.blocking[0]}"
            outcome.messages.append(message)
            self.events.emit("hardware.plan_too_big", message, level="error")
        if plan.training_allowed:
            train = training_estimate(
                model, quant, seq_len=plan.train_seq_len, lora_rank=plan.lora_rank,
                lora_targets=model.lora_targets, optimizer=plan.optimizer,
            )
            plan.vram_training_mb = train.total_mb

    # --- approvals ------------------------------------------------------
    def _request_base_change(self, active_base: str, plan: Plan) -> int:
        existing = self.db.query_one(
            "SELECT id FROM approvals WHERE kind = ? AND status = 'pending'"
            " AND subject = ? ORDER BY id DESC LIMIT 1",
            (APPROVAL_BASE_CHANGE, f"{active_base} -> {plan.base_model}"),
        )
        if existing:
            return int(existing["id"])
        approval_id = self.db.insert(
            "approvals",
            {
                "kind": APPROVAL_BASE_CHANGE,
                "subject": f"{active_base} -> {plan.base_model}",
                "rationale": "\n".join(plan.explanation),
                "payload": json.dumps(plan.to_dict()),
            },
        )
        self.events.emit(
            "hardware.base_change_proposed",
            f"base model change {active_base} -> {plan.base_model} needs approval "
            f"(#{approval_id})",
            level="warning",
            approval=approval_id,
        )
        return approval_id

    def approve_base_change(self, approval_id: int) -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM approvals WHERE id = ?", (approval_id,))
        if row is None:
            raise KeyError(f"no approval #{approval_id}")
        if row["kind"] != APPROVAL_BASE_CHANGE:
            raise ValueError(
                f"approval #{approval_id} is a {row['kind']}, not a base-model change"
            )
        if row["status"] != "pending":
            return {"approved": False, "reason": f"approval #{approval_id} is {row['status']}"}

        plan = Plan.from_dict(json.loads(row["payload"]))
        # A base change invalidates every adapter trained on the old base, so section 9
        # requires a verified backup first. The daemon wraps this call in the guard; the
        # snapshot id is recorded on the approval so it can be found later.
        self.events.emit(
            "hardware.base_change_approved",
            f"base model is now {plan.base_model} (approval #{approval_id}); "
            f"adapters trained on the previous base no longer apply",
            level="warning",
            approval=approval_id,
            previous_base=self.active_base_model,
        )
        self.db.set_meta(META_ACTIVE_BASE, plan.base_model)
        self.db.set_meta(META_PLAN, plan.to_dict())
        self.db.delete_meta(META_PENDING_PLAN)
        self.db.execute(
            "UPDATE approvals SET status = 'approved', decided_at = datetime('now')"
            " WHERE id = ?",
            (approval_id,),
        )
        if self.config.hardware.profile == "auto":
            plan.write(self.paths.generated_profile)
        return {"approved": True, "plan": plan.to_dict()}

    def reject_approval(self, approval_id: int, note: str = "") -> dict[str, Any]:
        self.db.execute(
            "UPDATE approvals SET status = 'rejected', decided_at = datetime('now'),"
            " note = ? WHERE id = ? AND status = 'pending'",
            (note or None, approval_id),
        )
        self.db.delete_meta(META_PENDING_PLAN)
        return {"rejected": True, "id": approval_id}

    # --- reporting ------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        hardware = self.hardware
        plan = self.stored_plan()
        return {
            "fingerprint": hardware.fingerprint(),
            "source": hardware.source,
            "gpus": [g.describe() for g in hardware.gpus],
            "host": {
                "cpu_count": hardware.host.cpu_count,
                "ram_total_mb": hardware.host.ram_total_mb,
                "disk_free_mb": hardware.host.disk_free_mb,
            },
            "budgets": [
                vram_budget(
                    g, reserve_fraction=self.config.hardware.vram_reserve_fraction
                ).describe()
                for g in hardware.gpus
            ],
            "temperatures_c": self.gpu_temperatures(),
            "active_base_model": self.active_base_model,
            "plan": plan.to_dict() if plan else None,
            "pending_plan": (self.pending_plan().to_dict() if self.pending_plan() else None),
        }
