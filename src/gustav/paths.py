"""Filesystem layout.

Every path the system uses is derived from a project root plus a data directory,
so tests can run against a throwaway tree and the daemon never writes outside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir: Path

    @classmethod
    def create(cls, root: Path | str, data_dir: Path | str = "var") -> "Paths":
        root = Path(root).expanduser().resolve()
        data = Path(data_dir).expanduser()
        if not data.is_absolute():
            data = root / data
        return cls(root=root, data_dir=data.resolve())

    # --- data dir -------------------------------------------------------
    @property
    def logs(self) -> Path:
        return self.data_dir / "logs"

    @property
    def log_file(self) -> Path:
        return self.logs / "gustav.log"

    @property
    def db(self) -> Path:
        return self.data_dir / "gustav.db"

    @property
    def lock_file(self) -> Path:
        return self.data_dir / "gustav.pid"

    @property
    def socket(self) -> Path:
        return self.data_dir / "gustav.sock"

    @property
    def runs(self) -> Path:
        """Per-sleep-run working directories (checkpoints, scratch)."""
        return self.data_dir / "runs"

    @property
    def reports(self) -> Path:
        return self.data_dir / "reports"

    @property
    def doc_cache(self) -> Path:
        return self.data_dir / "doc_cache"

    @property
    def tmp(self) -> Path:
        """Staging area for atomic promotions; same filesystem as adapters."""
        return self.data_dir / "tmp"

    # --- project dir ----------------------------------------------------
    @property
    def profiles(self) -> Path:
        return self.root / "profiles"

    @property
    def generated_profile(self) -> Path:
        return self.profiles / "auto-generated.yaml"

    @property
    def registry(self) -> Path:
        return self.root / "modules" / "registry.yaml"

    @property
    def adapters(self) -> Path:
        return self.root / "adapters"

    @property
    def models(self) -> Path:
        return self.root / "models"

    def run_dir(self, run_id: int) -> Path:
        return self.runs / f"run-{run_id:06d}"

    def ensure(self) -> "Paths":
        for p in (
            self.data_dir,
            self.logs,
            self.runs,
            self.reports,
            self.doc_cache,
            self.tmp,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self
