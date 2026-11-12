"""Taking, verifying and listing snapshots.

A snapshot is a directory ``backups/<timestamp>-<operation>/`` holding copies of
everything that matters, plus a manifest of SHA-256 hashes. The manifest is written
*last* and atomically, so a directory without one is by definition incomplete — which is
how an interrupted backup is told apart from a finished one.

What goes in (section 9): all adapters and their metadata, the module registry, the
SQLite store, the router and draft model, the config and the hardware plan, and the
held-out benchmark definitions. The base checkpoint is only included for operations that
could change it, because it is several gigabytes.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from gustav.backup.manifest import (
    MANIFEST_NAME,
    Manifest,
    Problem,
    build_manifest,
    should_skip,
)
from gustav.paths import Paths

#: Written while a snapshot is being built, removed when it is complete.
INCOMPLETE_MARKER = ".incomplete"


class BackupError(RuntimeError):
    pass


class NotEnoughSpace(BackupError):
    def __init__(self, needed_mb: float, free_mb: float, location: Path):
        super().__init__(
            f"the backup needs about {needed_mb / 1024:.2f} GiB but {location} has only "
            f"{free_mb / 1024:.2f} GiB free; refusing rather than filling the disk"
        )
        self.needed_mb = needed_mb
        self.free_mb = free_mb


@dataclass(frozen=True)
class Source:
    """One thing to copy into a snapshot."""

    name: str
    path: Path
    #: A source that must exist; a missing one aborts the snapshot.
    required: bool = False
    #: SQLite databases are copied through the backup API, not as files.
    database: bool = False


@dataclass
class Snapshot:
    id: str
    path: Path
    operation: str
    created_at: str
    size_bytes: int = 0
    file_count: int = 0
    verified: bool = False
    complete: bool = True
    manifest: Manifest | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / (1024 * 1024), 1)

    def describe(self) -> str:
        state = "verified" if self.verified else ("complete" if self.complete else "INCOMPLETE")
        return (
            f"{self.id:<34} {self.operation:<18} {self.size_mb:>9.1f} MiB  "
            f"{self.file_count:>4} files  {state}"
        )


@dataclass
class SnapshotPlan:
    """What a snapshot would contain, for `--dry-run`."""

    operation: str
    sources: list[tuple[str, Path, float]] = field(default_factory=list)
    estimated_mb: float = 0.0
    free_mb: float = 0.0
    location: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> list[str]:
        lines = [f"snapshot for {self.operation!r} would go to {self.location}"]
        for name, path, size in self.sources:
            lines.append(f"  {name:<22} {size:>9.1f} MiB  {path}")
        lines.append(f"  {'total':<22} {self.estimated_mb:>9.1f} MiB")
        lines.append(f"  free at the destination: {self.free_mb / 1024:.2f} GiB")
        lines.extend(f"  warning: {w}" for w in self.warnings)
        return lines


@dataclass
class VerifyResult:
    snapshot: Snapshot
    problems: list[Problem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def describe(self) -> list[str]:
        if self.ok:
            return [
                f"{self.snapshot.id}: {self.snapshot.file_count} files, "
                f"{self.snapshot.size_mb:.1f} MiB, all hashes match"
            ]
        return [f"{self.snapshot.id}: {len(self.problems)} problem(s)"] + [
            f"  {p.describe()}" for p in self.problems
        ]


def directory_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for current, _, files in os.walk(path, followlinks=False):
        for name in files:
            candidate = Path(current) / name
            if candidate.is_file() and not candidate.is_symlink():
                total += candidate.stat().st_size
    return round(total / (1024 * 1024), 4)


class SnapshotStore:
    def __init__(
        self,
        paths: Paths,
        *,
        location: Path | str | None = None,
        db=None,
        events=None,
        min_free_mb: int = 5120,
        warn_free_mb: int = 20480,
        logger=None,
    ):
        from gustav.core.logs import get_logger

        self.paths = paths
        self.location = Path(location) if location else paths.root / "backups"
        self.db = db
        self.events = events
        self.min_free_mb = min_free_mb
        self.warn_free_mb = warn_free_mb
        self.log = logger or get_logger("backup")

    # --- what to copy ---------------------------------------------------
    def sources(self, *, include_base: bool = False) -> list[Source]:
        paths = self.paths
        found = [
            Source("adapters", paths.adapters),
            Source("tools", paths.tools),
            Source("registry", paths.registry),
            Source("database", paths.db, database=True),
            Source("config", paths.root / "config.yaml"),
            Source("profiles", paths.profiles),
            Source("router", paths.data_dir / "router"),
            Source("draft_model", paths.data_dir / "draft"),
            Source("benchmarks", paths.data_dir / "benchmarks"),
        ]
        if include_base:
            # Several gigabytes: only for operations that could change the base weights.
            found.append(Source("base_model", paths.models))
        return found

    # --- planning -------------------------------------------------------
    def plan(self, operation: str, *, include_base: bool = False) -> SnapshotPlan:
        plan = SnapshotPlan(operation=operation, location=self.location)
        for source in self.sources(include_base=include_base):
            if not source.path.exists():
                continue
            size = (
                directory_size_mb(source.path)
                if source.path.is_dir()
                else round(source.path.stat().st_size / (1024 * 1024), 4)
            )
            plan.sources.append((source.name, source.path, size))
        plan.estimated_mb = round(sum(size for _, _, size in plan.sources) * 1.05, 4)
        plan.free_mb = self.free_mb()
        if plan.free_mb - plan.estimated_mb < self.min_free_mb:
            plan.warnings.append(
                f"this would leave less than {self.min_free_mb / 1024:.1f} GiB free; "
                f"the snapshot will be refused"
            )
        elif plan.free_mb < self.warn_free_mb:
            plan.warnings.append(
                f"the backup location is getting full ({plan.free_mb / 1024:.1f} GiB free)"
            )
        return plan

    def free_mb(self) -> float:
        target = self.location
        while not target.exists() and target != target.parent:
            target = target.parent
        return round(shutil.disk_usage(target).free / (1024 * 1024), 1)

    # --- creating -------------------------------------------------------
    def create(
        self,
        operation: str,
        *,
        include_base: bool = False,
        meta: dict[str, Any] | None = None,
        verify: bool = True,
        dry_run: bool = False,
    ) -> Snapshot:
        """Copy, hash, and verify. Raises rather than returning an unusable snapshot."""
        plan = self.plan(operation, include_base=include_base)
        if plan.free_mb - plan.estimated_mb < self.min_free_mb:
            raise NotEnoughSpace(plan.estimated_mb, plan.free_mb, self.location)
        if dry_run:
            return Snapshot(
                id=self._next_id(operation), path=self.location / "(dry-run)",
                operation=operation, created_at=datetime.now().isoformat(timespec="seconds"),
                size_bytes=int(plan.estimated_mb * 1024 * 1024), complete=False,
                meta={"dry_run": True, "plan": [s[0] for s in plan.sources]},
            )

        snapshot_id = self._next_id(operation)
        root = self.location / snapshot_id
        root.mkdir(parents=True, exist_ok=False)
        (root / INCOMPLETE_MARKER).write_text("in progress\n", encoding="utf-8")

        copied: dict[str, str] = {}
        try:
            for source in self.sources(include_base=include_base):
                if not source.path.exists():
                    if source.required:
                        raise BackupError(f"{source.name} is missing at {source.path}")
                    continue
                destination = root / source.name
                if source.database:
                    self._copy_database(source.path, destination / source.path.name)
                elif source.path.is_dir():
                    shutil.copytree(
                        source.path, destination, symlinks=True,
                        ignore=_ignore_noise, dirs_exist_ok=False,
                    )
                else:
                    destination.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source.path, destination / source.path.name)
                copied[source.name] = str(source.path)

            manifest = build_manifest(
                root, snapshot_id=snapshot_id, operation=operation, sources=copied,
                meta={"include_base": include_base, **(meta or {})},
            )
            (root / INCOMPLETE_MARKER).unlink(missing_ok=True)
            manifest.files = [f for f in manifest.files if f.path != INCOMPLETE_MARKER]
            manifest.write(root)
        except BaseException:
            # A half-written snapshot is worse than none: it could be restored from.
            shutil.rmtree(root, ignore_errors=True)
            raise

        snapshot = self._load(root)
        if verify:
            result = self.verify(snapshot.id)
            if not result.ok:
                shutil.rmtree(root, ignore_errors=True)
                raise BackupError(
                    f"the snapshot failed verification immediately after being written: "
                    + "; ".join(p.describe() for p in result.problems[:3])
                )
            snapshot.verified = True
        self._emit(
            "backup.created",
            f"snapshot {snapshot.id} for {operation}: {snapshot.file_count} files, "
            f"{snapshot.size_mb:.1f} MiB"
            + (", verified" if snapshot.verified else ""),
            snapshot=snapshot.id,
            operation=operation,
            size_mb=snapshot.size_mb,
        )
        return snapshot

    def _copy_database(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.db is not None and Path(self.db.path) == source:
            self.db.backup_to(destination)
            return
        from gustav.store.db import Database

        other = Database(source)
        try:
            other.backup_to(destination)
        finally:
            other.close()

    def _next_id(self, operation: str) -> str:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in operation)[:32]
        candidate = f"{stamp}-{safe}"
        suffix = 1
        while (self.location / candidate).exists():
            suffix += 1
            candidate = f"{stamp}-{safe}-{suffix}"
        return candidate

    # --- reading --------------------------------------------------------
    def _load(self, root: Path) -> Snapshot:
        complete = (root / MANIFEST_NAME).exists() and not (root / INCOMPLETE_MARKER).exists()
        manifest = Manifest.read(root) if complete else None
        created = manifest.created_at if manifest else ""
        if not created:
            created = datetime.fromtimestamp(root.stat().st_mtime).isoformat(timespec="seconds")
        return Snapshot(
            id=root.name,
            path=root,
            operation=manifest.operation if manifest else root.name.split("-", 1)[-1],
            created_at=created,
            size_bytes=manifest.total_bytes if manifest else int(
                directory_size_mb(root) * 1024 * 1024
            ),
            file_count=manifest.file_count if manifest else 0,
            complete=complete,
            manifest=manifest,
            meta=manifest.meta if manifest else {},
        )

    def list(self, *, include_incomplete: bool = True) -> list[Snapshot]:
        if not self.location.exists():
            return []
        found = [
            self._load(child)
            for child in sorted(self.location.iterdir())
            if child.is_dir() and not child.name.startswith(".")
        ]
        if not include_incomplete:
            found = [s for s in found if s.complete]
        return found

    def get(self, snapshot_id: str) -> Snapshot:
        root = self.location / snapshot_id
        if not root.is_dir():
            available = [s.id for s in self.list()]
            raise BackupError(
                f"no snapshot {snapshot_id!r}"
                + (f"; available: {', '.join(available[-5:])}" if available else "")
            )
        return self._load(root)

    def latest(self, *, verified_only: bool = False) -> Snapshot | None:
        candidates = [s for s in self.list() if s.complete]
        if verified_only:
            candidates = [s for s in candidates if self.verify(s.id).ok]
        return candidates[-1] if candidates else None

    def verify(self, snapshot_id: str, *, check_extra: bool = False) -> VerifyResult:
        snapshot = self.get(snapshot_id)
        if snapshot.manifest is None:
            return VerifyResult(
                snapshot,
                [Problem(MANIFEST_NAME, "missing", "the snapshot was never completed")],
            )
        problems = snapshot.manifest.verify(snapshot.path, check_extra=check_extra)
        snapshot.verified = not problems
        return VerifyResult(snapshot, problems)

    def delete(self, snapshot_id: str) -> None:
        snapshot = self.get(snapshot_id)
        shutil.rmtree(snapshot.path, ignore_errors=True)
        self._emit("backup.deleted", f"snapshot {snapshot_id} deleted", snapshot=snapshot_id)

    def _emit(self, kind: str, message: str, **data: Any) -> None:
        if self.events is not None:
            self.events.emit(kind, message, **data)
        else:
            self.log.info(message)


def _ignore_noise(directory: str, names: list[str]) -> set[str]:
    return {
        name for name in names if should_skip(Path(name)) or name in {"__pycache__", ".git"}
    }
