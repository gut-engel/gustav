"""The auto planner.

Given the detected hardware and one knob — ``balance``, from 0.0 (as fast as possible)
to 1.0 (as capable as possible) — it picks the largest model and quantisation that

1. passes the memory pre-flight check for inference, including the KV cache,
2. passes it for QLoRA training too, if training is required on this machine, and
3. still meets the minimum speed.

It only ever *recommends* a base model. Adapters are tied to the base they were trained
on, so changing it is never automatic: the daemon turns a base-model change into a
pending approval (``ai approve-base-change``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from gustav.hw.detect import flash_attention_available
from gustav.hw.ladder import Ladder, ModelSpec, QuantSpec, load_ladder
from gustav.hw.memory import (
    MemoryEstimate,
    inference_estimate,
    kv_dtype_bytes,
    ram_estimate,
    training_estimate,
    weights_mb,
)
from gustav.hw.preflight import (
    PreflightResult,
    VramBudget,
    check_disk,
    check_ram,
    check_vram,
    vram_budget,
)
from gustav.hw.speed import Measurements, SpeedEstimate
from gustav.hw.types import Hardware

#: Training rarely needs the full inference context; long sequences cost quadratically.
DEFAULT_TRAIN_SEQ_LEN = 1024
#: Effective batch size we aim for through gradient accumulation.
TARGET_EFFECTIVE_BATCH = 16
#: Section 8: expansion must leave this fraction of VRAM free at peak.
EXPANSION_HEADROOM_FRACTION = 0.20


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class ExpansionBudget:
    """What self-expansion is allowed to spend (section 8), derived from the hardware."""

    max_adapters: int
    max_lora_rank: int
    max_total_adapter_params: int
    max_adapter_vram_mb: int
    warm_cache_mb: int
    cold_storage_mb: int
    reserved_headroom_fraction: float = EXPANSION_HEADROOM_FRACTION

    def describe(self) -> list[str]:
        plural = "adapter" if self.max_adapters == 1 else "adapters"
        return [
            f"up to {self.max_adapters} {plural}, LoRA rank <= {self.max_lora_rank}",
            f"at most {self.max_total_adapter_params / 1e6:.0f}M adapter parameters "
            f"({self.max_adapter_vram_mb} MiB of VRAM)",
            f"warm (pinned RAM) cache {self.warm_cache_mb / 1024:.0f} GiB, "
            f"cold (disk) {self.cold_storage_mb / 1024:.0f} GiB",
            f"{self.reserved_headroom_fraction:.0%} of VRAM kept free at peak",
        ]


@dataclass
class Candidate:
    model: ModelSpec
    quant: QuantSpec
    seq_len: int
    kv_dtype: str
    inference: PreflightResult
    speed: SpeedEstimate
    training: PreflightResult | None = None
    train_seq_len: int = DEFAULT_TRAIN_SEQ_LEN
    feasible: bool = False
    #: The speed this candidate had to beat, after `balance` adjusted the floor.
    effective_floor: float = 0.0
    rejected_because: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.model.id}@{self.quant.id}"

    @property
    def selection_key(self) -> tuple[int, int, float]:
        """Capability first, then the preferred quantisation, then speed.

        This is the rule from the spec: take the *largest* model that fits and is fast
        enough. `balance` does its work by raising the speed floor (see
        ``Planner.effective_floor``), not by trading capability away point by point —
        which would quietly pick a tiny model on a big card.
        """
        return (self.model.rung, self.quant.preference, self.speed.tokens_per_second)

    def summary(self) -> str:
        verdict = "ok" if self.feasible else "; ".join(self.rejected_because) or "rejected"
        return (
            f"{self.key:<24} seq {self.seq_len:<5} "
            f"{self.inference.estimate.total_mb / 1024:5.2f} GiB  "
            f"{self.speed.tokens_per_second:6.0f} tok/s  rung {self.model.rung}  {verdict}"
        )


@dataclass
class Plan:
    """The hardware plan. Written to ``profiles/auto-generated.yaml``."""

    base_model: str
    repo: str
    quantization: str
    max_seq_len: int
    kv_cache_dtype: str
    compute_dtype: str
    attention: str
    lora_rank: int
    train_seq_len: int
    batch_size: int
    grad_accum: int
    gradient_checkpointing: bool
    optimizer: str
    speculative_decoding: bool
    draft_model: str | None
    inference_gpus: list[int]
    training_gpus: list[int]
    training_allowed: bool
    training_strategy: str
    estimated_tokens_per_second: float
    speed_measured: bool
    vram_inference_mb: float
    vram_training_mb: float
    ram_required_mb: float
    expansion: ExpansionBudget
    balance: float
    min_tokens_per_second: float
    fingerprint: str
    profile: str = "auto"
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    explanation: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.base_model} @ {self.quantization}, seq {self.max_seq_len}, "
            f"LoRA r{self.lora_rank}, ~{self.estimated_tokens_per_second:.0f} tok/s"
            + ("" if self.speed_measured else " (est.)")
            + (", training on " + str(self.training_gpus) if self.training_allowed else
               ", inference only")
        )

    def to_yaml(self) -> str:
        header = [
            "# Generated by `ai plan-hardware`. Edit freely: anything you set here wins,",
            "# and the planner will not overwrite the file while hardware.profile points at it.",
            "#",
            *[f"# {line}" for line in self.explanation],
        ]
        if self.rejected:
            header += ["#", "# Not chosen:", *[f"#   {line}" for line in self.rejected]]
        body = {
            "balance": self.balance,
            "min_tokens_per_second": self.min_tokens_per_second,
            "plan": {
                k: v for k, v in self.to_dict().items()
                if k not in {"explanation", "rejected", "warnings", "expansion",
                             "balance", "min_tokens_per_second"}
            },
        }
        body["plan"]["expansion"] = asdict(self.expansion)
        return "\n".join(header) + "\n" + yaml.safe_dump(body, sort_keys=False)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Plan":
        payload = dict(data)
        expansion = payload.pop("expansion", None)
        known = {f for f in cls.__dataclass_fields__}
        payload = {k: v for k, v in payload.items() if k in known}
        payload["expansion"] = (
            ExpansionBudget(**expansion) if isinstance(expansion, dict)
            else ExpansionBudget(4, 16, 0, 0, 0, 0)
        )
        return cls(**payload)


class PlannerError(RuntimeError):
    """No configuration fits this machine; the message says what to change."""


class Planner:
    def __init__(
        self,
        hardware: Hardware,
        config: Any,
        *,
        ladder: Ladder | None = None,
        measurements: Measurements | None = None,
    ):
        self.hardware = hardware
        self.config = config
        self.ladder = ladder or load_ladder(getattr(config, "ladder", None) or None)
        self.measurements = measurements or Measurements(None)

    # --- GPU assignment -------------------------------------------------
    def _gpu_split(self) -> tuple[list[int], list[int]]:
        """Which cards serve and which train.

        With two or more GPUs they are separated, so the model keeps answering while it
        trains. With one, they are the same card and sleep pauses to answer.
        """
        available = [g.index for g in self.hardware.gpus]
        inference = [i for i in (self.config.inference_gpus or []) if i in available]
        training = [i for i in (self.config.training_gpus or []) if i in available]
        if inference or training:
            return inference or available[:1], training or available
        if len(available) >= 2:
            return available[:1], available[1:]
        return available, available

    def _budget(self, gpu_index: int) -> VramBudget | None:
        if not self.hardware.has_gpu:
            return None
        return vram_budget(
            self.hardware.gpu(gpu_index),
            reserve_fraction=self.config.vram_reserve_fraction,
        )

    # --- candidate evaluation -------------------------------------------
    def _seq_len_preference(self) -> list[int]:
        options = sorted(self.ladder.seq_len_options, reverse=True)
        if self.config.max_seq_len:
            return [s for s in options if s <= self.config.max_seq_len] or [options[-1]]
        # Low balance favours a short, fast context; high balance a long one.
        ceiling = [1024, 2048, 4096, 8192][min(int(round(self.config.balance * 3)), 3)]
        return [s for s in options if s <= ceiling] or [options[-1]]

    def _allowed(self, model: ModelSpec) -> tuple[bool, str]:
        if self.config.deny_models and model.id in self.config.deny_models:
            return False, "denied in config"
        if self.config.allow_models and model.id not in self.config.allow_models:
            return False, "not in hardware.allow_models"
        if self.config.prefer_model and model.id != self.config.prefer_model:
            return False, f"hardware.prefer_model is {self.config.prefer_model}"
        return True, ""

    def evaluate(self, model: ModelSpec, quant: QuantSpec) -> Candidate | None:
        inference_gpus, training_gpus = self._gpu_split()
        inf_budget = self._budget(inference_gpus[0]) if inference_gpus else None
        gpu = self.hardware.gpu(inference_gpus[0]) if inference_gpus else None

        allowed, why = self._allowed(model)
        if not allowed:
            candidate = self._barren(model, quant, inf_budget, why)
            return candidate

        if quant.requires_bf16 and gpu is not None and not gpu.supports_bf16:
            return self._barren(
                model, quant, inf_budget,
                f"{gpu.name} is {gpu.architecture} and has no bfloat16; use fp16 or a 4-bit load",
            )

        best: Candidate | None = None
        for seq_len in self._seq_len_preference():
            if seq_len > model.max_position:
                continue
            kv_dtype = self._kv_dtype(gpu)
            estimate = inference_estimate(
                model, quant, seq_len=seq_len, kv_dtype=kv_dtype, adapter_count=1,
            )
            inference = check_vram(
                estimate, inf_budget,
                suggestions=[f"max_seq_len {seq_len // 2}"] if seq_len > 512 else None,
            )
            speed = self.measurements.speed(
                model, quant, gpu, context_tokens=min(seq_len, 2048),
            )
            candidate = Candidate(
                model=model, quant=quant, seq_len=seq_len, kv_dtype=kv_dtype,
                inference=inference, speed=speed,
            )
            if not inference.ok:
                candidate.rejected_because = list(inference.blocking)
                best = best or candidate
                continue
            candidate.effective_floor = self.effective_floor
            if speed.tokens_per_second < self.effective_floor:
                candidate.rejected_because = [
                    f"{speed.tokens_per_second:.0f} tok/s is below the "
                    f"{self.effective_floor:.0f} tok/s floor that balance "
                    f"{self.config.balance} demands"
                ]
                best = best or candidate
                continue

            self._evaluate_training(candidate, model, quant, training_gpus)
            if candidate.training is not None and not candidate.training.ok:
                if self.config.require_training and self.config.allow_training:
                    candidate.rejected_because = [
                        f"training does not fit: {candidate.training.blocking[0]}"
                    ]
                    best = best or candidate
                    continue

            candidate.feasible = True
            return candidate

        return best

    def _evaluate_training(
        self, candidate: Candidate, model: ModelSpec, quant: QuantSpec, training_gpus: list[int]
    ) -> None:
        if not self.config.allow_training or not training_gpus or not quant.trainable:
            return
        train_seq = min(
            self.config.train_seq_len or DEFAULT_TRAIN_SEQ_LEN, candidate.seq_len, model.max_position
        )
        candidate.train_seq_len = train_seq
        rank = self._lora_rank(model)
        estimate = training_estimate(
            model, quant, seq_len=train_seq, lora_rank=rank,
            lora_targets=model.lora_targets, optimizer=self._optimizer(),
        )
        budgets = [self._budget(i) for i in training_gpus]
        budgets = [b for b in budgets if b is not None]
        if not budgets:
            return
        single = min(budgets, key=lambda b: b.usable_mb)
        result = check_vram(
            estimate, single,
            suggestions=[f"train_seq_len {train_seq // 2}", "a smaller model",
                         "training on another machine"],
        )
        if not result.ok and len(budgets) > 1:
            # Sharded training across the training GPUs (FSDP/ZeRO, stage 5).
            pooled = VramBudget(
                gpu_index=-1,
                total_mb=sum(b.total_mb for b in budgets),
                reserved_mb=sum(b.reserved_mb for b in budgets),
                foreign_mb=sum(b.foreign_mb for b in budgets),
                usable_mb=int(sum(b.usable_mb for b in budgets) * 0.9),
                reserve_fraction=self.config.vram_reserve_fraction,
            )
            sharded = check_vram(estimate, pooled)
            if sharded.ok:
                sharded.warnings.append(
                    f"needs sharded training across GPUs {training_gpus} (FSDP or DeepSpeed ZeRO)"
                )
                result = sharded
        candidate.training = result

    def _barren(
        self, model: ModelSpec, quant: QuantSpec, budget: VramBudget | None, why: str
    ) -> Candidate:
        estimate = inference_estimate(model, quant, seq_len=1024)
        result = PreflightResult(ok=False, estimate=estimate, budget=budget, blocking=[why])
        return Candidate(
            model=model, quant=quant, seq_len=1024, kv_dtype="fp16",
            inference=result, speed=SpeedEstimate(0, 0, False, why),
            rejected_because=[why],
        )

    # --- knobs ----------------------------------------------------------
    def _kv_dtype(self, gpu) -> str:
        configured = getattr(self.config, "kv_cache_dtype", "auto")
        if configured and configured != "auto":
            return configured
        # Quantised KV buys context at a small quality cost; only worth it when
        # we are leaning towards capability over speed.
        if gpu is not None and gpu.supports_fp8 and self.config.balance >= 0.75:
            return "fp8"
        return "fp16"

    def _lora_rank(self, model: ModelSpec) -> int:
        if self.config.balance >= 0.75:
            rank = 32
        elif self.config.balance <= 0.25:
            rank = 8
        else:
            rank = 16
        return min(rank, 64)

    def _optimizer(self) -> str:
        return "paged_adamw_8bit"

    def _attention(self, gpu) -> str:
        if gpu is None:
            return "sdpa"
        if gpu.attention_backend == "flash_attention_2" and not flash_attention_available():
            return "sdpa"
        return gpu.attention_backend

    @property
    def effective_floor(self) -> float:
        """The speed a candidate must beat, after `balance` has had its say.

        At balance 1.0 the only requirement is the configured floor, so the biggest
        model that fits wins. At balance 0.0 a candidate must be three times faster
        than the floor, which pushes the choice down the ladder to a smaller model.
        """
        multiplier = 1.0 + (1.0 - _clamp(self.config.balance, 0.0, 1.0)) * 2.0
        return round(max(self.config.min_tokens_per_second, 0.0) * multiplier, 1)

    # --- expansion budget -----------------------------------------------
    def _expansion_budget(self, candidate: Candidate) -> ExpansionBudget:
        inference_gpus, _ = self._gpu_split()
        budget = self._budget(inference_gpus[0]) if inference_gpus else None
        usable = budget.usable_mb if budget else 0
        peak = candidate.inference.estimate.total_mb
        free = max(usable * (1 - EXPANSION_HEADROOM_FRACTION) - peak, 0)

        max_rank = {True: 64, False: 32}[self.config.balance >= 0.75] if usable else 16
        if self.config.balance <= 0.25:
            max_rank = 16
        adapter_mb = max(candidate.model.lora_params(max_rank) * 2 / (1024 * 1024), 1.0)
        by_vram = int(free // adapter_mb)
        by_balance = int(round(2 + self.config.balance * 10))
        max_adapters = max(min(by_vram, by_balance), 1)

        ram_total = self.hardware.host.ram_total_mb
        warm = int(min(max(ram_total - self.config.ram_reserve_mb, 0) * 0.25,
                       16384 * max(self.config.balance, 0.25)))
        cold = int(min(self.hardware.host.disk_free_mb * 0.10, 200 * 1024))
        return ExpansionBudget(
            max_adapters=max_adapters,
            max_lora_rank=max_rank,
            max_total_adapter_params=int(candidate.model.lora_params(max_rank) * max_adapters),
            max_adapter_vram_mb=int(adapter_mb * max_adapters),
            warm_cache_mb=warm,
            cold_storage_mb=cold,
        )

    # --- speculative decoding -------------------------------------------
    def _draft(self, candidate: Candidate) -> tuple[bool, str | None, str]:
        setting = getattr(self.config, "speculative_decoding", "auto")
        if setting == "off":
            return False, None, "speculative decoding is off in config"
        drafts = [
            d for d in self.ladder.draft_models()
            if d.total_params * 5 < candidate.model.total_params
        ]
        if not drafts:
            return False, None, "no draft model small enough to help"
        draft = drafts[0]
        quant = self.ladder.quant(
            "4bit-nf4" if "4bit-nf4" in draft.quantizations else draft.quantizations[0]
        )
        quantized, unquantized = weights_mb(draft, quant)
        extra = quantized + unquantized
        headroom = candidate.inference.headroom_mb
        if extra > headroom:
            return False, None, (
                f"the {draft.label} draft model needs {extra / 1024:.1f} GiB but only "
                f"{headroom / 1024:.1f} GiB is spare"
            )
        if setting == "on":
            return True, draft.id, f"speculative decoding with {draft.label} (forced on)"
        if self.config.balance <= 0.5:
            return True, draft.id, (
                f"speculative decoding with {draft.label}: balance {self.config.balance} "
                f"favours speed, and it fits in the spare {headroom / 1024:.1f} GiB"
            )
        return False, None, (
            f"speculative decoding left off: balance {self.config.balance} favours capability, "
            f"and the spare VRAM is better spent on adapters"
        )

    # --- the plan -------------------------------------------------------
    def plan(self) -> Plan:
        candidates: list[Candidate] = []
        for model, quant in self.ladder.candidates():
            candidate = self.evaluate(model, quant)
            if candidate is not None:
                candidates.append(candidate)

        feasible = [c for c in candidates if c.feasible]
        if not feasible:
            raise PlannerError(self._nothing_fits(candidates))

        chosen = max(feasible, key=lambda c: c.selection_key)
        return self._build_plan(chosen, candidates)

    def _build_plan(self, chosen: Candidate, candidates: list[Candidate]) -> Plan:
        inference_gpus, training_gpus = self._gpu_split()
        gpu = self.hardware.gpu(inference_gpus[0]) if inference_gpus else None
        speculative, draft_id, draft_reason = self._draft(chosen)
        expansion = self._expansion_budget(chosen)
        rank = self._lora_rank(chosen.model)
        training_ok = bool(
            self.config.allow_training and chosen.training is not None and chosen.training.ok
        )
        strategy = "none"
        if training_ok:
            strategy = (
                "sharded" if any("sharded" in w for w in chosen.training.warnings) else "single-gpu"
            )

        ram_needed = ram_estimate(
            chosen.model, chosen.quant, warm_cache_mb=expansion.warm_cache_mb
        )
        warnings = list(chosen.inference.warnings)
        ram_ok, ram_message = check_ram(
            ram_needed, self.hardware, reserve_mb=self.config.ram_reserve_mb
        )
        if not ram_ok:
            warnings.append(ram_message)
        disk_needed = (sum(weights_mb(chosen.model, chosen.quant)) * 3) + 5120
        disk_ok, disk_message = check_disk(
            disk_needed, self.hardware, reserve_mb=self.config.disk_reserve_mb
        )
        if not disk_ok:
            warnings.append(disk_message)
        if chosen.training is not None:
            warnings.extend(chosen.training.warnings)
        if expansion.max_adapters <= 1:
            warnings.append(
                f"there is room for only one resident adapter at this size, so "
                f"self-expansion has almost nothing to work with; a lower "
                f"hardware.balance picks a smaller model and frees VRAM for several"
            )
        if not chosen.model.verified:
            warnings.append(
                f"{chosen.model.label}'s architecture came from ladder.yaml, not from a "
                f"downloaded config.json; sizes will be exact once the weights are present"
            )

        plan = Plan(
            base_model=chosen.model.id,
            repo=chosen.model.repo,
            quantization=chosen.quant.id,
            max_seq_len=chosen.seq_len,
            kv_cache_dtype=chosen.kv_dtype,
            compute_dtype=(
                gpu.preferred_dtype if gpu and chosen.quant.compute_dtype == "auto"
                else chosen.quant.compute_dtype
            ),
            attention=self._attention(gpu),
            lora_rank=rank,
            train_seq_len=chosen.train_seq_len,
            batch_size=1,
            grad_accum=max(TARGET_EFFECTIVE_BATCH // max(len(training_gpus) or 1, 1), 1),
            gradient_checkpointing=True,
            optimizer=self._optimizer(),
            speculative_decoding=speculative,
            draft_model=draft_id,
            inference_gpus=inference_gpus,
            training_gpus=training_gpus if training_ok else [],
            training_allowed=training_ok,
            training_strategy=strategy,
            estimated_tokens_per_second=chosen.speed.tokens_per_second,
            speed_measured=chosen.speed.measured,
            vram_inference_mb=chosen.inference.estimate.total_mb,
            vram_training_mb=(
                chosen.training.estimate.total_mb if chosen.training is not None else 0.0
            ),
            ram_required_mb=ram_needed,
            expansion=expansion,
            balance=self.config.balance,
            min_tokens_per_second=self.config.min_tokens_per_second,
            fingerprint=self.hardware.fingerprint(),
            profile="auto",
            warnings=warnings,
        )
        plan.explanation = self._explain(plan, chosen, draft_reason, expansion)
        plan.rejected = [
            c.summary() for c in sorted(candidates, key=lambda c: -c.model.rung)
            if not c.feasible
        ]
        return plan

    def _explain(
        self, plan: Plan, chosen: Candidate, draft_reason: str, expansion: ExpansionBudget
    ) -> list[str]:
        gpu = self.hardware.gpu(plan.inference_gpus[0]) if plan.inference_gpus else None
        lines = [
            f"Chose {chosen.model.describe()} in {chosen.quant.label}.",
            f"It is the largest model on the ladder that fits and is fast enough: "
            f"balance {plan.balance} turns the {plan.min_tokens_per_second:.0f} tok/s floor "
            f"into an effective {self.effective_floor:.0f} tok/s requirement "
            f"(0.0 = fastest, 1.0 = most capable).",
        ]
        if gpu:
            lines.append(gpu.describe())
            if gpu.supports_flash_attention_2 and plan.attention != "flash_attention_2":
                lines.append(
                    "This GPU supports FlashAttention-2 but the flash-attn package is not "
                    "installed, so PyTorch SDPA is used instead; install it to get a little "
                    "more speed on long prompts."
                )
            budget = self._budget(gpu.index)
            if budget:
                lines.append(budget.describe())
        lines.append(
            f"Inference at seq {plan.max_seq_len} needs "
            f"{plan.vram_inference_mb / 1024:.2f} GiB, leaving "
            f"{chosen.inference.headroom_mb / 1024:.2f} GiB spare."
        )
        if plan.training_allowed:
            lines.append(
                f"QLoRA training at seq {plan.train_seq_len}, rank {plan.lora_rank} needs "
                f"{plan.vram_training_mb / 1024:.2f} GiB, so this machine can train itself"
                + (f" ({plan.training_strategy} on GPUs {plan.training_gpus})"
                   if len(plan.training_gpus) > 1 else "")
                + "."
            )
        elif not self.config.allow_training:
            lines.append("Training is disabled in config: this machine serves only.")
        else:
            why = (
                chosen.training.blocking[0] if chosen.training and chosen.training.blocking
                else "no training GPU is available"
            )
            lines.append(
                f"Training does NOT fit ({why}). Train on another machine and copy the "
                f"adapters over, or lower train_seq_len."
            )
        if len(plan.inference_gpus) and plan.inference_gpus == plan.training_gpus:
            lines.append(
                "One GPU does both, so sleep pauses to answer you and resumes afterwards."
            )
        elif plan.training_allowed and plan.training_gpus:
            lines.append(
                f"GPU{plan.inference_gpus[0]} serves while GPUs {plan.training_gpus} train, "
                f"so chat stays up during sleep."
            )
        lines.append(draft_reason + ".")
        lines.append(
            f"Speed: {chosen.speed.describe()}."
            + ("" if chosen.speed.measured else
               " Estimated from memory bandwidth; stage 3 replaces this with a measurement.")
        )
        lines.append("Expansion budget: " + "; ".join(expansion.describe()) + ".")
        return lines

    def _nothing_fits(self, candidates: list[Candidate]) -> str:
        lines = ["no model on the ladder fits this machine:"]
        if not self.hardware.has_gpu:
            lines.append("  no GPU was detected at all")
        for candidate in sorted(candidates, key=lambda c: c.model.rung)[:6]:
            lines.append(f"  {candidate.summary()}")
        lines.append(
            "  try: lower hardware.min_tokens_per_second, lower hardware.vram_reserve_fraction, "
            "set hardware.require_training: false, or add a smaller model to the ladder"
        )
        return "\n".join(lines)
