"""Restoring from a snapshot — and undoing the restore.

A restore is itself a change, so it takes a snapshot of the current state first. That is
what makes "I restored the wrong one" recoverable instead of final.

Restores are staged and swapped: the replacement is assembled beside the target and
moved into place, so a crash mid-restore leaves either the old tree or the new one, never
a mixture.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gustav.backup.atomic import promote, remove
from gustav.backup.manifest import MANIFEST_NAME
from gustav.backup.snapshot import BackupError, Snapshot, SnapshotStore


@dataclass
class RestoreResult:
    snapshot_id: str
    restored: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    #: The snapshot of the pre-restore state, so this can be undone.
    undo_snapshot_id: str | None = None
    dry_run: bool = False

    def describe(self) -> list[str]:
        verb = "would restore" if self.dry_run else "restored"
        lines = [f"{verb} from {self.snapshot_id}: {', '.join(self.restored) or 'nothing'}"]
        if self.skipped:
            lines.append(f"  skipped (not in the snapshot): {', '.join(self.skipped)}")
        if self.undo_snapshot_id:
            lines.append(
                f"  the previous state is snapshot {self.undo_snapshot_id} — "
                f"'ai backup restore {self.undo_snapshot_id}' undoes this"
            )
        return lines


class Restorer:
    def __init__(self, store: SnapshotStore, *, events=None, logger=None):
        from gustav.core.logs import get_logger

        self.store = store
        self.paths = store.paths
        self.events = events
        self.log = logger or get_logger("restore")

    def targets(self) -> dict[str, Path]:
        """Where each source in a snapshot belongs on disk."""
        paths = self.paths
        return {
            "adapters": paths.adapters,
            "registry": paths.registry,
            "database": paths.db,
            "config": paths.root / "config.yaml",
            "profiles": paths.profiles,
            "router": paths.data_dir / "router",
            "draft_model": paths.data_dir / "draft",
            "benchmarks": paths.data_dir / "benchmarks",
            "base_model": paths.models,
        }

    def restore(
        self,
        snapshot_id: str,
        *,
        only: tuple[str, ...] = (),
        module: str | None = None,
        snapshot_first: bool = True,
        dry_run: bool = False,
    ) -> RestoreResult:
        """Put a snapshot's contents back.

        ``only`` restores named sources; ``module`` restores a single adapter module.
        """
        snapshot = self.store.get(snapshot_id)
        if not snapshot.complete:
            raise BackupError(
                f"snapshot {snapshot_id} is incomplete and cannot be restored from"
            )
        verification = self.store.verify(snapshot_id)
        if not verification.ok:
            raise BackupError(
                f"refusing to restore from {snapshot_id}: it does not verify "
                + "; ".join(p.describe() for p in verification.problems[:3])
            )

        result = RestoreResult(snapshot_id=snapshot_id, dry_run=dry_run)
        wanted = self._selection(snapshot, only=only, module=module)
        if not wanted:
            raise BackupError(
                f"nothing to restore: {snapshot_id} has no "
                + (f"module {module!r}" if module else f"source(s) {', '.join(only)}")
            )

        if dry_run:
            result.restored = [name for name, _, _ in wanted]
            return result

        if snapshot_first:
            # A restore is a change like any other, so it is recoverable too.
            undo = self.store.create(
                f"pre-restore-{snapshot_id[:15]}",
                meta={"restoring": snapshot_id, "sources": [n for n, _, _ in wanted]},
            )
            result.undo_snapshot_id = undo.id

        for name, source, target in wanted:
            self._replace(source, target)
            result.restored.append(name)
        result.skipped = [
            name for name in self._available(snapshot) if name not in result.restored
        ]
        if self.events is not None:
            self.events.emit(
                "backup.restored",
                f"restored {', '.join(result.restored)} from {snapshot_id}",
                level="warning",
                snapshot=snapshot_id,
                undo=result.undo_snapshot_id,
            )
        return result

    def _available(self, snapshot: Snapshot) -> list[str]:
        return sorted(
            child.name
            for child in snapshot.path.iterdir()
            if child.is_dir() and child.name != MANIFEST_NAME
        )

    def _selection(
        self, snapshot: Snapshot, *, only: tuple[str, ...], module: str | None
    ) -> list[tuple[str, Path, Path]]:
        targets = self.targets()
        if module:
            source = snapshot.path / "adapters" / module
            if not source.exists():
                return []
            return [(f"adapters/{module}", source, self.paths.adapters / module)]

        chosen: list[tuple[str, Path, Path]] = []
        for name in self._available(snapshot):
            if only and name not in only:
                continue
            target = targets.get(name)
            if target is None:
                continue
            source = snapshot.path / name
            # Single-file sources were copied into a directory of that name.
            if not target.is_dir() and target.name:
                inner = source / target.name
                if inner.exists():
                    source = inner
            chosen.append((name, source, target))
        return chosen

    def _replace(self, source: Path, target: Path) -> None:
        """Copy beside the target, then swap it in. Never a partially written tree."""
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.restoring")
        remove(staging)
        if source.is_dir():
            shutil.copytree(source, staging, symlinks=True)
        else:
            shutil.copy2(source, staging)
        try:
            promote(staging, target)
        except OSError:
            remove(staging)
            raise
