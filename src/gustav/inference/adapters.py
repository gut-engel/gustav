"""The adapter store on disk.

``adapters/<module>/v001/``, ``v002/``, ... with an ``active`` symlink pointing at the
version in use. Promotion is a symlink swap via ``os.replace``, which is atomic: a crash
at any instant leaves the previous version active and loadable. Stage 4 wraps these
operations in a guard that takes a verified backup first; the mechanism lives here.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

VERSION_RE = re.compile(r"^v(\d{3,})$")
ACTIVE_LINK = "active"
META_FILE = "meta.json"
#: Files PEFT writes for an adapter. Used to tell a real adapter from a stray directory.
ADAPTER_MARKERS = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")


class AdapterError(RuntimeError):
    pass


@dataclass
class AdapterVersion:
    module: str
    version: str
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def number(self) -> int:
        match = VERSION_RE.match(self.version)
        return int(match.group(1)) if match else -1

    @property
    def complete(self) -> bool:
        """True if the directory actually holds a loadable adapter."""
        return any((self.path / marker).exists() for marker in ADAPTER_MARKERS)

    @property
    def base_model(self) -> str | None:
        return self.meta.get("base_model")

    def describe(self) -> str:
        score = self.meta.get("benchmark_score")
        suffix = f", score {score}" if score is not None else ""
        return f"{self.module}@{self.version}{suffix}"


class AdapterStore:
    """Reads and writes ``adapters/``. Knows nothing about PEFT or torch."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    # --- reading --------------------------------------------------------
    def modules(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def module_dir(self, module: str) -> Path:
        return self.root / module

    def versions(self, module: str) -> list[AdapterVersion]:
        directory = self.module_dir(module)
        if not directory.exists():
            return []
        found = [
            self._version(module, child.name)
            for child in directory.iterdir()
            if child.is_dir() and not child.is_symlink() and VERSION_RE.match(child.name)
        ]
        return sorted(found, key=lambda v: v.number)

    def _version(self, module: str, version: str) -> AdapterVersion:
        path = self.module_dir(module) / version
        return AdapterVersion(module=module, version=version, path=path, meta=self.read_meta(path))

    def get(self, module: str, version: str) -> AdapterVersion:
        candidate = self._version(module, version)
        if not candidate.path.is_dir():
            raise AdapterError(f"no adapter {module}@{version} in {self.root}")
        return candidate

    def active(self, module: str) -> AdapterVersion | None:
        link = self.module_dir(module) / ACTIVE_LINK
        if not link.exists():
            return None
        target = link.resolve()
        if not target.is_dir():
            return None
        return self._version(module, target.name)

    def active_versions(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for module in self.modules():
            current = self.active(module)
            if current is not None:
                out[module] = current.version
        return out

    def latest(self, module: str) -> AdapterVersion | None:
        versions = self.versions(module)
        return versions[-1] if versions else None

    def previous(self, module: str, version: str | None = None) -> AdapterVersion | None:
        """The version before the given one (or before the active one)."""
        reference = version or (self.active(module).version if self.active(module) else None)
        if reference is None:
            return None
        versions = self.versions(module)
        numbers = [v.version for v in versions]
        if reference not in numbers:
            return None
        index = numbers.index(reference)
        return versions[index - 1] if index > 0 else None

    # --- metadata -------------------------------------------------------
    def read_meta(self, path: Path) -> dict[str, Any]:
        file = path / META_FILE
        try:
            return json.loads(file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, NotADirectoryError):
            return {}

    def write_meta(self, module: str, version: str, meta: dict[str, Any]) -> Path:
        path = self.module_dir(module) / version
        path.mkdir(parents=True, exist_ok=True)
        file = path / META_FILE
        payload = {"module": module, "version": version, **meta}
        payload.setdefault("written_at", datetime.now().isoformat(timespec="seconds"))
        # Write-then-rename so a crash cannot leave a half-written metadata file.
        temporary = file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, file)
        return file

    # --- writing --------------------------------------------------------
    def next_version(self, module: str) -> str:
        versions = self.versions(module)
        return f"v{(versions[-1].number + 1) if versions else 1:03d}"

    def reserve(self, module: str, version: str | None = None) -> AdapterVersion:
        """Create an empty version directory to train into."""
        version = version or self.next_version(module)
        path = self.module_dir(module) / version
        if path.exists():
            raise AdapterError(f"{module}@{version} already exists at {path}")
        path.mkdir(parents=True)
        return AdapterVersion(module=module, version=version, path=path)

    def activate(self, module: str, version: str) -> AdapterVersion:
        """Point ``active`` at a version, atomically.

        The symlink is created under a temporary name and moved into place with
        ``os.replace``, so there is no moment at which ``active`` is missing.
        """
        target = self.get(module, version)
        if not target.complete:
            raise AdapterError(
                f"{module}@{version} has no adapter weights in {target.path}; refusing to "
                f"activate it"
            )
        link = self.module_dir(module) / ACTIVE_LINK
        temporary = self.module_dir(module) / f".{ACTIVE_LINK}.swap"
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(version)
        os.replace(temporary, link)
        return target

    def rollback(self, module: str) -> AdapterVersion:
        """Activate the version before the active one."""
        earlier = self.previous(module)
        if earlier is None:
            raise AdapterError(
                f"{module} has no earlier version to roll back to "
                f"(versions: {[v.version for v in self.versions(module)] or 'none'})"
            )
        return self.activate(module, earlier.version)

    def discard(self, module: str, version: str) -> None:
        """Delete a version that was never activated (a failed candidate)."""
        current = self.active(module)
        if current is not None and current.version == version:
            raise AdapterError(f"refusing to delete {module}@{version}: it is active")
        shutil.rmtree(self.module_dir(module) / version, ignore_errors=True)

    def archive(self, module: str, destination: Path) -> Path:
        """Move an unused module out of the way instead of deleting it (section 8)."""
        source = self.module_dir(module)
        if not source.exists():
            raise AdapterError(f"no module {module} to archive")
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / module
        if target.exists():
            target = destination / f"{module}-{datetime.now():%Y%m%d%H%M%S}"
        shutil.move(str(source), str(target))
        return target

    # --- reporting ------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for module in self.modules():
            current = self.active(module)
            out[module] = {
                "active": current.version if current else None,
                "versions": [v.version for v in self.versions(module)],
                "base_model": current.base_model if current else None,
                "meta": current.meta if current else {},
            }
        return out
