"""The one way weights are allowed to change.

Section 9 of the spec: *no operation that adds, changes, merges or removes weights may
start without a verified backup; if the backup fails, the operation is refused.* This is
that rule, expressed as a context manager:

    with guard.change("promote coding v004", sources=("adapters",)) as change:
        ...  # write the new weights
        change.promote(staging, target)

On entry it takes a snapshot and verifies it, or raises ``BackupRequired`` and the
operation never happens. On the way out it records the change as done. If the block
raises, the touched sources are restored from that snapshot before the error propagates.

Every change is logged in ``weight_changes`` with its snapshot id, so an entry still
``in_progress`` when the daemon starts is an interrupted operation — which is what crash
recovery looks for and undoes.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from gustav.backup.atomic import atomic_symlink, promote, staged
from gustav.backup.restore import RestoreResult, Restorer
from gustav.backup.snapshot import BackupError, Snapshot, SnapshotStore

META_LAST_BACKUP = "last_backup"


class BackupRequired(RuntimeError):
    """A weight change was refused because no verified backup could be taken."""

    def __init__(self, operation: str, cause: BaseException | str):
        super().__init__(
            f"refusing to {operation}: no verified backup could be taken ({cause}). "
            f"Nothing was changed."
        )
        self.operation = operation
        self.cause = cause


@dataclass
class Change:
    """Handed to the body of a guarded operation."""

    operation: str
    snapshot: Snapshot | None
    staging: Path
    sources: tuple[str, ...] = ()
    record_id: int | None = None
    promoted: list[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def snapshot_id(self) -> str | None:
        return self.snapshot.id if self.snapshot else None

    def promote(self, staging: Path, target: Path) -> None:
        """Make something staged live, atomically."""
        if self.dry_run:
            self.promoted.append(f"(dry run) {target}")
            return
        promote(staging, target)
        self.promoted.append(str(target))

    def activate(self, link: Path, target: str | Path) -> None:
        """Swap an ``active`` symlink, atomically."""
        if self.dry_run:
            self.promoted.append(f"(dry run) {link} -> {target}")
            return
        atomic_symlink(link, target)
        self.promoted.append(f"{link} -> {target}")


class WeightChangeGuard:
    def __init__(
        self,
        store: SnapshotStore,
        *,
        restorer: Restorer | None = None,
        db=None,
        events=None,
        enabled: bool = True,
        logger=None,
    ):
        from gustav.core.logs import get_logger

        self.store = store
        self.restorer = restorer or Restorer(store, events=events)
        self.db = db
        self.events = events
        self.enabled = enabled
        self.log = logger or get_logger("guard")

    # --- the guard ------------------------------------------------------
    @contextmanager
    def change(
        self,
        operation: str,
        *,
        sources: tuple[str, ...] = ("adapters",),
        include_base: bool = False,
        run_id: int | None = None,
        step: str | None = None,
        dry_run: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> Iterator[Change]:
        if not self.enabled:
            # Only ever set deliberately, and said out loud: the whole point of this
            # module is that it is not optional.
            self.log.warning(
                "backups are disabled in config, so %r is proceeding unprotected", operation
            )
        snapshot: Snapshot | None = None
        record_id: int | None = None

        if self.enabled and not dry_run:
            try:
                snapshot = self.store.create(
                    operation,
                    include_base=include_base,
                    verify=True,
                    meta={"run_id": run_id, "step": step, **(meta or {})},
                )
            except BaseException as exc:  # noqa: BLE001 - disk full, permissions, anything
                self._record_refusal(operation, exc, run_id=run_id, step=step)
                raise BackupRequired(operation, exc) from exc
            if self.db is not None:
                self.db.set_meta(
                    META_LAST_BACKUP,
                    {
                        "id": snapshot.id,
                        "operation": operation,
                        "created_at": snapshot.created_at,
                        "size_mb": snapshot.size_mb,
                        "verified": snapshot.verified,
                    },
                )
        elif self.enabled and dry_run:
            snapshot = self.store.create(operation, include_base=include_base, dry_run=True)

        record_id = self._record_start(
            operation, snapshot, sources=sources, run_id=run_id, step=step, dry_run=dry_run
        )
        change = Change(
            operation=operation, snapshot=snapshot, staging=self.store.paths.tmp,
            sources=sources, record_id=record_id, dry_run=dry_run,
        )

        with staged(self.store.paths.tmp, _slug(operation)) as staging:
            change.staging = staging
            try:
                yield change
            except BaseException as exc:  # noqa: BLE001
                self._on_failure(change, exc)
                raise
            else:
                self._record_finish(record_id, "done", detail=", ".join(change.promoted))
                if self.events is not None and not dry_run:
                    self.events.emit(
                        "weights.changed",
                        f"{operation} completed (snapshot {change.snapshot_id})",
                        operation=operation,
                        snapshot=change.snapshot_id,
                        promoted=change.promoted,
                    )

    # --- rolling back ---------------------------------------------------
    def _on_failure(self, change: Change, exc: BaseException) -> None:
        if change.dry_run or change.snapshot is None:
            self._record_finish(change.record_id, "failed", detail=str(exc)[:500])
            return
        self.log.error("%s failed (%s); rolling back", change.operation, exc)
        try:
            self.restorer.restore(
                change.snapshot.id, only=change.sources, snapshot_first=False
            )
            status = "rolled_back"
            detail = f"{exc}; restored from {change.snapshot.id}"
        except BaseException as restore_error:  # noqa: BLE001
            status = "failed"
            detail = f"{exc}; THE ROLLBACK ALSO FAILED: {restore_error}"
            self.log.critical(
                "rollback of %s from %s failed: %s",
                change.operation, change.snapshot.id, restore_error,
            )
        self._record_finish(change.record_id, status, detail=detail[:500])
        if self.events is not None:
            self.events.emit(
                "weights.rolled_back" if status == "rolled_back" else "weights.rollback_failed",
                f"{change.operation} failed: {detail}".split("\n")[0],
                level="error",
                operation=change.operation,
                snapshot=change.snapshot.id,
            )

    def rollback_interrupted(self) -> list[dict[str, Any]]:
        """Undo every change that was still in progress. Called at startup."""
        if self.db is None:
            return []
        rows = self.db.query(
            "SELECT * FROM weight_changes WHERE status = 'in_progress' ORDER BY id"
        )
        undone: list[dict[str, Any]] = []
        for row in rows:
            snapshot_id = row["snapshot_id"]
            sources = tuple((row["sources"] or "").split(",")) if row["sources"] else ()
            outcome: dict[str, Any] = {
                "id": row["id"], "operation": row["operation"], "snapshot": snapshot_id,
            }
            if not snapshot_id:
                self._record_finish(row["id"], "failed", detail="no snapshot to restore from")
                outcome["restored"] = False
                outcome["detail"] = "no snapshot was recorded"
                undone.append(outcome)
                continue
            try:
                result = self.restorer.restore(
                    snapshot_id, only=tuple(s for s in sources if s), snapshot_first=False
                )
                self._record_finish(
                    row["id"], "rolled_back",
                    detail=f"interrupted; restored {', '.join(result.restored)}",
                )
                outcome["restored"] = True
                outcome["sources"] = result.restored
            except BaseException as exc:  # noqa: BLE001
                self._record_finish(row["id"], "failed", detail=f"interrupted; {exc}"[:500])
                outcome["restored"] = False
                outcome["detail"] = str(exc)
            undone.append(outcome)
        return undone

    def rollback_step(self, run_id: int, step: str) -> RestoreResult | None:
        """The rollback handler crash recovery calls for a weight-changing step."""
        if self.db is None:
            return None
        row = self.db.query_one(
            "SELECT * FROM weight_changes WHERE run_id = ? AND step = ?"
            " ORDER BY id DESC LIMIT 1",
            (run_id, step),
        )
        if row is None or not row["snapshot_id"]:
            return None
        sources = tuple(s for s in (row["sources"] or "").split(",") if s)
        result = self.restorer.restore(
            row["snapshot_id"], only=sources, snapshot_first=False
        )
        self._record_finish(row["id"], "rolled_back", detail="rolled back by crash recovery")
        return result

    # --- the log --------------------------------------------------------
    def _record_start(
        self,
        operation: str,
        snapshot: Snapshot | None,
        *,
        sources: tuple[str, ...],
        run_id: int | None,
        step: str | None,
        dry_run: bool,
    ) -> int | None:
        if self.db is None or dry_run:
            return None
        return self.db.insert(
            "weight_changes",
            {
                "operation": operation,
                "snapshot_id": snapshot.id if snapshot else None,
                "run_id": run_id,
                "step": step,
                "sources": ",".join(sources),
            },
        )

    def _record_finish(self, record_id: int | None, status: str, *, detail: str = "") -> None:
        if self.db is None or record_id is None:
            return
        self.db.execute(
            "UPDATE weight_changes SET status = ?, finished_at = datetime('now'),"
            " detail = ? WHERE id = ?",
            (status, detail or None, record_id),
        )

    def _record_refusal(
        self, operation: str, exc: BaseException, *, run_id: int | None, step: str | None
    ) -> None:
        self.log.error("refusing %s: the backup failed (%s)", operation, exc)
        if self.db is not None:
            self.db.insert(
                "weight_changes",
                {
                    "operation": operation,
                    "status": "refused",
                    "run_id": run_id,
                    "step": step,
                    "finished_at": None,
                    "detail": f"backup failed: {exc}"[:500],
                },
            )
        if self.events is not None:
            self.events.emit(
                "weights.refused",
                f"refused to {operation}: no verified backup ({exc})".split("\n")[0],
                level="error",
                operation=operation,
            )

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        if self.db is None:
            return []
        return [
            dict(row)
            for row in self.db.query(
                "SELECT * FROM weight_changes ORDER BY id DESC LIMIT ?", (limit,)
            )
        ]


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text)[:40] or "change"
