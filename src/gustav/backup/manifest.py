"""Snapshot manifests: a SHA-256 of every file, so a backup can be proven intact.

A backup nobody has verified is a hope, not a backup. Every snapshot carries a manifest
listing each file's size and hash; verification re-hashes and compares. Symlinks are
recorded as symlinks rather than followed, because ``adapters/<module>/active`` *is* the
promotion mechanism — resolving it would quietly turn a pointer into a copy.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

CHUNK = 1024 * 1024
MANIFEST_NAME = "manifest.json"
#: Never worth copying into a snapshot.
SKIP_NAMES = frozenset({"__pycache__", ".pytest_cache", ".git"})
SKIP_SUFFIXES = (".pyc", ".tmp", ".part", ".swap")


def hash_file(path: Path, *, chunk: int = CHUNK) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def should_skip(path: Path) -> bool:
    if any(part in SKIP_NAMES for part in path.parts):
        return True
    return path.name.endswith(SKIP_SUFFIXES)


@dataclass(frozen=True)
class FileEntry:
    #: Path relative to the snapshot root.
    path: str
    size: int
    sha256: str = ""
    symlink_to: str | None = None
    mode: int = 0o644

    @property
    def is_symlink(self) -> bool:
        return self.symlink_to is not None


@dataclass
class Problem:
    path: str
    kind: str  # missing | size | hash | symlink | extra
    detail: str = ""

    def describe(self) -> str:
        return f"{self.path}: {self.kind}" + (f" ({self.detail})" if self.detail else "")


@dataclass
class Manifest:
    id: str
    operation: str
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    files: list[FileEntry] = field(default_factory=list)
    #: What was backed up, by logical name: "adapters", "database", "config", ...
    sources: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def file_count(self) -> int:
        return len(self.files)

    def to_json(self) -> str:
        payload = {
            "id": self.id,
            "operation": self.operation,
            "created_at": self.created_at,
            "sources": self.sources,
            "meta": self.meta,
            "total_bytes": self.total_bytes,
            "file_count": self.file_count,
            "files": [asdict(f) for f in self.files],
        }
        return json.dumps(payload, indent=2, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        data = json.loads(text)
        return cls(
            id=data["id"],
            operation=data["operation"],
            created_at=data.get("created_at", ""),
            files=[FileEntry(**entry) for entry in data.get("files", [])],
            sources=data.get("sources", {}),
            meta=data.get("meta", {}),
        )

    def write(self, root: Path) -> Path:
        """Write the manifest last and atomically: its presence means "complete"."""
        target = root / MANIFEST_NAME
        temporary = target.with_suffix(".json.part")
        temporary.write_text(self.to_json(), encoding="utf-8")
        os.replace(temporary, target)
        return target

    @classmethod
    def read(cls, root: Path) -> "Manifest":
        return cls.from_json((root / MANIFEST_NAME).read_text(encoding="utf-8"))

    def verify(self, root: Path, *, check_extra: bool = False) -> list[Problem]:
        """Re-hash everything and report every difference."""
        problems: list[Problem] = []
        for entry in self.files:
            target = root / entry.path
            if entry.is_symlink:
                if not target.is_symlink():
                    problems.append(Problem(entry.path, "symlink", "no longer a symlink"))
                elif os.readlink(target) != entry.symlink_to:
                    problems.append(
                        Problem(
                            entry.path, "symlink",
                            f"points at {os.readlink(target)}, expected {entry.symlink_to}",
                        )
                    )
                continue
            if not target.exists():
                problems.append(Problem(entry.path, "missing"))
                continue
            actual_size = target.stat().st_size
            if actual_size != entry.size:
                problems.append(
                    Problem(entry.path, "size", f"{actual_size} bytes, expected {entry.size}")
                )
                continue
            if entry.sha256 and hash_file(target) != entry.sha256:
                problems.append(Problem(entry.path, "hash", "contents changed"))
        if check_extra:
            known = {f.path for f in self.files} | {MANIFEST_NAME}
            for found in walk(root):
                relative = str(found.relative_to(root))
                if relative not in known:
                    problems.append(Problem(relative, "extra", "not in the manifest"))
        return problems


def walk(root: Path) -> Iterable[Path]:
    """Every file and symlink under ``root``, skipping noise. Symlinks are not followed."""
    if not root.exists():
        return
    for current, directories, files in os.walk(root, followlinks=False):
        here = Path(current)
        directories[:] = [d for d in directories if d not in SKIP_NAMES]
        # A symlinked directory is an entry in its own right, not something to descend.
        for name in list(directories):
            candidate = here / name
            if candidate.is_symlink():
                directories.remove(name)
                yield candidate
        for name in files:
            candidate = here / name
            if not should_skip(candidate.relative_to(root)):
                yield candidate


def entry_for(path: Path, relative: str) -> FileEntry:
    if path.is_symlink():
        return FileEntry(
            path=relative, size=0, symlink_to=os.readlink(path), mode=0o777
        )
    stat = path.stat()
    return FileEntry(
        path=relative,
        size=stat.st_size,
        sha256=hash_file(path),
        mode=stat.st_mode & 0o777,
    )


def build_manifest(
    root: Path, *, snapshot_id: str, operation: str, sources: dict[str, str] | None = None,
    meta: dict[str, Any] | None = None,
) -> Manifest:
    """Hash everything already copied into ``root``."""
    files = [entry_for(path, str(path.relative_to(root))) for path in sorted(walk(root))]
    return Manifest(
        id=snapshot_id,
        operation=operation,
        files=files,
        sources=sources or {},
        meta=meta or {},
    )
