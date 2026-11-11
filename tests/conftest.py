from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from gustav.config.schema import Config
from gustav.core.clock import FakeClock
from gustav.core.events import Events
from gustav.core.journal import Journal
from gustav.paths import Paths
from gustav.sleep.pipeline import default_pipeline
from gustav.store.db import open_db

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every step of the night, simulated. The pipeline machinery (resume, retry, watchdog,
#: preemption) is what most tests are about, and the placeholder handler is what makes
#: those behaviours provokable on demand. Tests of the real steps opt in by calling the
#: step directly or by overriding `sleep.simulated_steps`.
ALL_STEPS = [step.spec.name for step in default_pipeline(Config()).steps]


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A throwaway project tree with a minimal config.yaml."""
    root = tmp_path / "project"
    (root / "profiles").mkdir(parents=True)
    (root / "modules").mkdir(parents=True)
    (root / "config.yaml").write_text(
        "instance:\n  data_dir: var\n"
        "logging:\n  console: false\n  level: DEBUG\n"
        "schedule:\n  enabled: false\n  tick_seconds: 0.1\n  min_idle_seconds: 0\n"
        "  cooldown_minutes: 0\n  min_pending_items: 0\n"
        # The busy-GPU guards are tested explicitly in test_scheduler.py; here they
        # would make every test depend on what this machine's GPU is doing.
        "  max_foreign_vram_mb: 0\n  max_foreign_gpu_percent: 0\n"
        "supervisor:\n  heartbeat_timeout_seconds: 5\n  step_timeout_seconds: 30\n"
        "  max_attempts: 2\n  retry_backoff_seconds: 0.05\n  memory_limit_mb: 0\n"
        "sleep:\n  simulated_step_seconds: 0.02\n"
        f"  simulated_steps: [{', '.join(ALL_STEPS)}]\n"
        # Deterministic and GPU-free: the tests that need real weights are marked `gpu`.
        "inference:\n  responder: echo\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def paths(project: Path) -> Paths:
    return Paths.create(project, "var").ensure()


@pytest.fixture
def config() -> Config:
    return Config.model_validate(
        {
            "logging": {"console": False},
            "schedule": {"tick_seconds": 0.05, "min_idle_seconds": 0, "cooldown_minutes": 0,
                         "min_pending_items": 0},
            "supervisor": {
                "heartbeat_timeout_seconds": 5,
                "step_timeout_seconds": 30,
                "max_attempts": 2,
                "retry_backoff_seconds": 0.01,
                "memory_limit_mb": 0,
            },
            "sleep": {"simulated_step_seconds": 0.02, "simulated_steps": ALL_STEPS},
        }
    )


@pytest.fixture
def db(paths: Paths):
    database = open_db(paths.db)
    yield database
    database.close()


@pytest.fixture
def journal(db) -> Journal:
    return Journal(db)


@pytest.fixture
def events(db) -> Events:
    return Events(db)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(datetime(2026, 3, 1, 12, 0, 0))


@pytest.fixture(autouse=True)
def _clean_fault_env():
    saved = os.environ.pop("GUSTAV_FAULT", None)
    yield
    if saved is not None:
        os.environ["GUSTAV_FAULT"] = saved
    else:
        os.environ.pop("GUSTAV_FAULT", None)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def python_exe() -> str:
    venv = REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else shutil.which("python3") or "python3"
