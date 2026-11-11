"""Backup services, built inside a step's child process."""

from __future__ import annotations

from gustav.backup.guard import WeightChangeGuard
from gustav.backup.restore import Restorer
from gustav.backup.snapshot import SnapshotStore
from gustav.sleep.stages._base import StepEnv


def snapshot_store(env: StepEnv) -> SnapshotStore:
    cfg = env.config.backup
    return SnapshotStore(
        env.paths,
        location=cfg.location,
        db=env.db,
        events=env.events,
        min_free_mb=cfg.min_free_mb,
        warn_free_mb=cfg.warn_free_mb,
    )


def weight_guard(env: StepEnv) -> WeightChangeGuard:
    """The one choke point for weight changes (section 9), as a step sees it."""
    store = snapshot_store(env)
    return WeightChangeGuard(
        store,
        restorer=Restorer(store, events=env.events),
        db=env.db,
        events=env.events,
        enabled=env.config.backup.enabled,
    )
