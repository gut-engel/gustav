"""The first step of every night: a verified backup.

Section 9 of the specification is a rule, not a feature — *no operation that adds,
changes, merges or removes weights may start without a verified backup; if the backup
fails, the operation is refused.* The guard enforces that per operation, but the night
also takes one snapshot up front, for two reasons:

* a failure here is **critical**, so the pipeline stops before any step has had the
  chance to change anything, with the night's reason recorded in the journal;
* the snapshot id is written where `report`, `gate` and `ai runs` can find it, so
  "undo last night" is one command rather than an archaeology exercise.
"""

from __future__ import annotations

from typing import Any

from gustav.backup.snapshot import BackupError
from gustav.sleep.stages._backup import snapshot_store
from gustav.sleep.stages._base import StepEnv, sleep_step

META_RUN_SNAPSHOT = "sleep.run_snapshot"


@sleep_step
def run_step(env: StepEnv) -> dict[str, Any]:
    operation = f"sleep run {env.run_id}"
    store = snapshot_store(env)

    if not env.config.backup.enabled:
        env.log(
            "backups are disabled in config: tonight runs without a safety net",
            level="warning",
        )
        env.events.emit(
            "sleep.snapshot.skipped",
            "backups are disabled in config, so no pre-sleep snapshot was taken",
            level="warning",
            run_id=env.run_id,
        )
        return {"skipped": True, "reason": "backup.enabled is false"}

    plan = store.plan(operation)
    env.progress(
        f"snapshotting {len(plan.sources)} sources (~{plan.estimated_mb:.1f} MiB)",
        sources=len(plan.sources),
    )
    for warning in plan.warnings:
        env.log(warning, level="warning")

    if env.dry_run:
        return {
            "dry_run": True,
            "sources": [name for name, _, _ in plan.sources],
            "estimated_mb": plan.estimated_mb,
            "free_mb": plan.free_mb,
            "warnings": plan.warnings,
        }

    try:
        snapshot = store.create(operation, meta={"run_id": env.run_id, "trigger": "sleep"})
    except BackupError as exc:
        # Deliberately fatal: this step is `critical`, so the night ends here rather
        # than proceeding towards a training step with nothing to roll back to.
        env.log(f"the night's backup failed, so nothing will run: {exc}", level="error")
        raise

    env.db.set_meta(
        META_RUN_SNAPSHOT,
        {"run_id": env.run_id, "snapshot": snapshot.id, "created_at": snapshot.created_at},
    )
    env.write_json("snapshot.json", {"id": snapshot.id, "path": str(snapshot.path),
                                     "size_mb": snapshot.size_mb,
                                     "files": snapshot.file_count,
                                     "verified": snapshot.verified})
    env.events.emit(
        "sleep.snapshot",
        f"snapshot {snapshot.id} taken and verified ({snapshot.size_mb:.1f} MiB)",
        run_id=env.run_id,
        snapshot=snapshot.id,
    )
    return {
        "snapshot": snapshot.id,
        "size_mb": snapshot.size_mb,
        "files": snapshot.file_count,
        "verified": snapshot.verified,
    }
