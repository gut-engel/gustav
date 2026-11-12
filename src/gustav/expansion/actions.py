"""Applying a proposal — the only place self-expansion changes anything.

Every action goes through the weight-change guard, so every one of them has a verified
snapshot behind it before it starts and is restored from that snapshot if it fails. The
snapshot id is written to ``expansion_log``, which is what makes `ai expansion --revert`
a single operation rather than an archaeology exercise.

The registry counts as weights here. It is not a tensor, but it decides which adapters
load and how requests are routed, so a half-written registry is as bad as a half-written
checkpoint — it is written to a staging file and moved into place with one `os.replace`.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from gustav.backup.guard import WeightChangeGuard
from gustav.config.schema import Config
from gustav.core.events import Events
from gustav.expansion.proposals import Proposal
from gustav.inference.adapters import AdapterStore
from gustav.inference.registry import ModuleSpec, Registry, load_registry
from gustav.paths import Paths
from gustav.store.db import Database


@dataclass
class Applied:
    proposal: Proposal
    ok: bool
    detail: str = ""
    snapshot_id: str | None = None
    log_id: int | None = None
    changes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        verb = "applied" if self.ok else "failed"
        return f"{verb}: {self.proposal.kind} {self.proposal.subject} — {self.detail}"


class Expander:
    """Applies proposals, records them, and knows how to take one back."""

    def __init__(
        self,
        *,
        paths: Paths,
        config: Config,
        db: Database,
        events: Events,
        guard: WeightChangeGuard,
        adapters: AdapterStore | None = None,
        logger: Any = None,
    ):
        from gustav.core.logs import get_logger

        self.paths = paths
        self.config = config
        self.db = db
        self.events = events
        self.guard = guard
        self.adapters = adapters or AdapterStore(paths.adapters)
        self.log = logger or get_logger("expansion")

    # --- the log --------------------------------------------------------
    def record(
        self, proposal: Proposal, *, run_id: int | None, status: str,
        approval_id: int | None = None, snapshot_id: str | None = None, detail: str = "",
    ) -> int:
        return self.db.insert(
            "expansion_log",
            {
                "run_id": run_id,
                "kind": proposal.kind,
                "subject": proposal.subject,
                "rationale": proposal.rationale,
                "evidence": json.dumps(proposal.evidence, default=str),
                "cost": json.dumps(proposal.cost, default=str),
                "payload": json.dumps(proposal.payload, default=str),
                "status": status,
                "approval_id": approval_id,
                "snapshot_id": snapshot_id,
                "detail": detail[:2000],
            },
        )

    def _finish(self, log_id: int | None, status: str, *, snapshot_id: str | None = None,
                detail: str = "") -> None:
        if log_id is None:
            return
        self.db.execute(
            "UPDATE expansion_log SET status = ?, decided_at = datetime('now'),"
            " snapshot_id = COALESCE(?, snapshot_id), detail = ? WHERE id = ?",
            (status, snapshot_id, detail[:2000], log_id),
        )

    def file_approval(self, proposal: Proposal, *, run_id: int | None) -> int:
        """Ask, rather than do. Returns the approval id."""
        approval_id = self.db.insert(
            "approvals",
            {
                "kind": "expansion",
                "subject": f"{proposal.kind}: {proposal.subject}",
                "rationale": proposal.rationale,
                "payload": json.dumps(
                    {
                        "kind": proposal.kind,
                        "subject": proposal.subject,
                        "payload": proposal.payload,
                        "cost": proposal.cost,
                        "evidence": proposal.evidence,
                    },
                    default=str,
                ),
            },
        )
        log_id = self.record(
            proposal, run_id=run_id, status="pending", approval_id=approval_id
        )
        self.db.execute(
            "UPDATE approvals SET note = ? WHERE id = ?",
            (f"expansion_log={log_id}", approval_id),
        )
        self.events.emit(
            "expansion.proposed",
            f"{proposal.kind} {proposal.subject} needs your approval (#{approval_id}): "
            f"{proposal.rationale}",
            level="warning",
            approval=approval_id,
            action=proposal.kind,
            subject=proposal.subject,
        )
        return approval_id

    # --- applying -------------------------------------------------------
    def apply(
        self, proposal: Proposal, *, run_id: int | None = None, dry_run: bool = False,
        log_id: int | None = None, progress: Any = None,
    ) -> Applied:
        if not proposal.applicable:
            # A refusal is a result, not a non-event: it is recorded here whoever
            # called, so "it wanted to grow and the budget said no" is in the log.
            if log_id is None:
                log_id = self.record(
                    proposal, run_id=run_id, status="refused",
                    detail=proposal.refused or "",
                )
            else:
                self._finish(log_id, "refused", detail=proposal.refused or "")
            return Applied(proposal, ok=False, detail=proposal.refused or "refused",
                           log_id=log_id)

        handler = {
            "new_module": self._new_module,
            "raise_rank": self._raise_rank,
            "prune_module": self._prune_module,
            "merge_to_base": self._merge_to_base,
        }.get(proposal.kind)
        if handler is None:
            return Applied(proposal, ok=False, detail=f"unknown action {proposal.kind}")

        if log_id is None:
            log_id = self.record(proposal, run_id=run_id, status="applying")

        sources = {
            "new_module": ("registry",),
            "raise_rank": ("registry",),
            "prune_module": ("adapters", "registry"),
            "merge_to_base": ("adapters", "registry", "base_model"),
        }[proposal.kind]
        include_base = proposal.kind == "merge_to_base"

        try:
            with self.guard.change(
                f"expansion: {proposal.kind} {proposal.subject}",
                sources=sources,
                include_base=include_base,
                run_id=run_id,
                step="expansion",
                dry_run=dry_run,
                meta={"kind": proposal.kind, "subject": proposal.subject},
            ) as change:
                detail = handler(proposal, change, progress)
                snapshot_id = change.snapshot_id
                changes = list(change.promoted)
        except Exception as exc:  # noqa: BLE001 - one proposal failing is not the night
            self.log.exception("expansion %s %s failed", proposal.kind, proposal.subject)
            self._finish(log_id, "failed", detail=f"{type(exc).__name__}: {exc}")
            self.events.emit(
                "expansion.failed",
                f"{proposal.kind} {proposal.subject} failed: {exc}",
                level="error",
                action=proposal.kind,
                subject=proposal.subject,
            )
            return Applied(proposal, ok=False, detail=str(exc)[:400], log_id=log_id)

        self._finish(log_id, "applied", snapshot_id=snapshot_id, detail=detail)
        self.events.emit(
            "expansion.applied",
            f"{proposal.kind} {proposal.subject}: {detail}",
            action=proposal.kind,
            subject=proposal.subject,
            snapshot=snapshot_id,
        )
        return Applied(proposal, ok=True, detail=detail, snapshot_id=snapshot_id,
                       log_id=log_id, changes=changes)

    # --- the four actions -----------------------------------------------
    def _registry(self) -> Registry:
        return load_registry(self.paths.registry)

    def _write_registry(self, registry: Registry, change: Any) -> None:
        staged = Path(change.staging) / "registry.yaml"
        staged.parent.mkdir(parents=True, exist_ok=True)
        registry.write(staged)
        change.promote(staged, self.paths.registry)

    def _new_module(self, proposal: Proposal, change: Any, progress: Any) -> str:
        payload = proposal.payload
        spec = ModuleSpec(
            name=payload["name"],
            description=payload.get("description", ""),
            keywords=tuple(payload.get("keywords") or ()),
            priority=int(payload.get("priority", 1)),
            lora_rank=int(payload.get("lora_rank") or 0),
            notes=("added by self-expansion",),
        )
        registry = self._registry().add(spec)
        self._write_registry(registry, change)
        # No weights yet, deliberately: the module exists so that tonight's routing and
        # tomorrow's consolidation can start collecting material for it. The first
        # adapter is trained by the normal training step, and promoted by the normal
        # gate.
        return (
            f"registered module {spec.name!r} with {len(spec.keywords)} keywords; "
            f"it has no adapter until a night has material for it"
        )

    def _raise_rank(self, proposal: Proposal, change: Any, progress: Any) -> str:
        module = proposal.payload["module"]
        target = int(proposal.payload["lora_rank"])
        registry = self._registry()
        spec = registry.get(module)
        updated = ModuleSpec(**{**spec.__dict__, "lora_rank": target})
        self._write_registry(registry.replace(updated), change)
        return (
            f"{module}'s LoRA rank is now {target} (was "
            f"{proposal.payload.get('previous') or 'the plan default'}); the next night "
            f"trains at the new rank and the gate decides whether it helped"
        )

    def _prune_module(self, proposal: Proposal, change: Any, progress: Any) -> str:
        module = proposal.payload["module"]
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        destination = self.paths.data_dir / "archive" / f"{module}-{stamp}"
        moved = self.adapters.archive(module, destination)
        self._write_registry(self._registry().remove(module), change)
        return f"archived {module} to {moved} and removed it from the registry"

    def _merge_to_base(self, proposal: Proposal, change: Any, progress: Any) -> str:
        from gustav.expansion.merge import merge_adapter_into_base
        from gustav.hw.service import META_PLAN

        module = proposal.payload["module"]
        active = self.adapters.active(module)
        if active is None:
            raise RuntimeError(f"{module} has no active adapter to merge")
        plan = self.db.get_meta(META_PLAN) or {}
        base_id = plan.get("base_model")
        if not base_id:
            raise RuntimeError("there is no hardware plan, so the base model is unknown")
        base_dir = self.paths.models / base_id
        if not (base_dir / "config.json").exists():
            raise RuntimeError(f"the base checkpoint is not at {base_dir}")

        staging = Path(change.staging) / "merged"
        report = merge_adapter_into_base(
            base_dir=base_dir,
            adapter_dir=active.path,
            output_dir=staging,
            dtype=str(plan.get("compute_dtype") or "bfloat16"),
            device="cpu",
            progress=progress,
        )
        change.promote(staging, base_dir)

        # The adapter is now part of the base, so a delta from the *old* base is no
        # longer meaningful. Archive it rather than leave it loadable.
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        destination = self.paths.data_dir / "archive" / f"{module}-merged-{stamp}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.adapters.module_dir(module)), str(destination))

        # And so is every *other* adapter: they were trained as a delta from weights
        # that no longer exist. The model id does not change (merging changes no
        # shapes), so the loader's base-model check cannot catch this on its own — each
        # one is stamped instead, and the harness refuses to load a stamped adapter
        # until it has been retrained. Refusing is the point: quietly applying a delta
        # to the wrong weights is the kind of error that shows up as the model getting
        # mysteriously worse.
        stale = []
        for other in self.adapters.modules():
            if other == module:
                continue
            for version in self.adapters.versions(other):
                meta = dict(version.meta)
                meta["stale_base"] = (
                    f"{module}@{active.version} was merged into {base_id} on {stamp}; "
                    f"this adapter is a delta from the base as it was before that"
                )
                self.adapters.write_meta(other, version.version, meta)
                stale.append(f"{other}@{version.version}")
        if stale:
            self.events.emit(
                "expansion.adapters_stale",
                f"{len(stale)} adapter(s) are now deltas from a base that changed and "
                f"will not be loaded until they are retrained: {', '.join(stale)}",
                level="warning",
                adapters=stale,
            )

        return (
            f"merged {module}@{active.version} into {base_id} "
            f"({report['size_mb']:.0f} MiB written) and archived the adapter to "
            f"{destination}"
            + (
                f"; {len(stale)} other adapter version(s) are now stale and will be "
                f"skipped until retrained"
                if stale
                else ""
            )
        )

    # --- taking it back -------------------------------------------------
    def revert_last(self, *, only: str | None = None) -> dict[str, Any]:
        """Undo the most recent applied expansion by restoring its snapshot."""
        row = self.db.query_one(
            "SELECT * FROM expansion_log WHERE status = 'applied'"
            " AND snapshot_id IS NOT NULL ORDER BY id DESC LIMIT 1"
        )
        if row is None:
            raise KeyError("no applied expansion has a snapshot to revert to")
        snapshot_id = row["snapshot_id"]
        result = self.guard.restorer.restore(snapshot_id, only=(only,) if only else ())
        self._finish(
            int(row["id"]),
            "reverted",
            detail=f"restored from {snapshot_id} (undo snapshot {result.undo_snapshot_id})",
        )
        self.events.emit(
            "expansion.reverted",
            f"reverted {row['kind']} {row['subject']} from snapshot {snapshot_id}",
            level="warning",
            action=row["kind"],
            subject=row["subject"],
            snapshot=snapshot_id,
        )
        return {
            "id": int(row["id"]),
            "kind": row["kind"],
            "subject": row["subject"],
            "snapshot": snapshot_id,
            "undo_snapshot": result.undo_snapshot_id,
            "restored": list(result.restored),
        }

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM expansion_log ORDER BY id DESC LIMIT ?", (limit,)
        )
        out = []
        for row in rows:
            item = dict(row)
            for field_name in ("evidence", "cost", "payload"):
                try:
                    item[field_name] = json.loads(item[field_name] or "{}")
                except (ValueError, TypeError):
                    item[field_name] = {}
            out.append(item)
        return out
