"""The step journal: what makes sleep resumable.

A sleep run is a sequence of named steps recorded in SQLite *before* they run. Each
step is idempotent and keyed ``(run_id, name)``, so a pause, a crash, or a reboot
resumes at the step that was interrupted instead of restarting the night — or
corrupting something half-written.

``on_crash`` says what recovery should do with a step that was running when the
process died:

* ``resume``   — set it back to pending and run it again from its checkpoint (default)
* ``restart``  — discard the checkpoint and run it from scratch
* ``rollback`` — run the step's rollback handler and fail the run (weight changes)
* ``skip``     — give up on this step but continue the run
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from gustav.store.db import Database


class CrashPolicy(StrEnum):
    RESUME = "resume"
    RESTART = "restart"
    ROLLBACK = "rollback"
    SKIP = "skip"


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    PREEMPTED = "preempted"


class RunStatus(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    ABORTED = "aborted"


#: A run in one of these states still has work to do and should be resumed.
OPEN_RUN_STATES = frozenset({RunStatus.RUNNING, RunStatus.PAUSED, RunStatus.INTERRUPTED})
#: A step in one of these states is finished for this run and must not run again.
TERMINAL_STEP_STATES = frozenset({StepStatus.DONE, StepStatus.SKIPPED})


@dataclass(frozen=True)
class StepSpec:
    """Declaration of a pipeline step. The handler is looked up by name by the runner."""

    name: str
    on_crash: CrashPolicy = CrashPolicy.RESUME
    #: Wall-clock limit for one attempt; falls back to the supervisor default.
    timeout_seconds: float | None = None
    #: Attempts before the step is given up on; falls back to the supervisor default.
    max_attempts: int | None = None
    #: Needs exclusive use of the GPU (inference is unloaded first).
    requires_gpu: bool = False
    #: A failure here fails the whole run rather than being skipped.
    critical: bool = False
    description: str = ""


@dataclass
class StepRecord:
    name: str
    position: int
    status: StepStatus
    attempts: int
    on_crash: CrashPolicy
    checkpoint: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL_STEP_STATES


@dataclass
class RunRecord:
    id: int
    kind: str
    trigger: str
    status: RunStatus
    dry_run: bool
    run_dir: Path | None
    started_at: str | None = None
    finished_at: str | None = None
    reason: str | None = None
    steps: list[StepRecord] = field(default_factory=list)

    def step(self, name: str) -> StepRecord | None:
        return next((s for s in self.steps if s.name == name), None)

    @property
    def next_step(self) -> StepRecord | None:
        return next((s for s in self.steps if not s.finished), None)

    @property
    def done_count(self) -> int:
        return sum(1 for s in self.steps if s.status is StepStatus.DONE)


class Journal:
    """Reads and writes the sleep journal. All methods are synchronous and fast."""

    def __init__(self, db: Database):
        self.db = db

    # --- runs -----------------------------------------------------------
    def create_run(
        self,
        *,
        trigger: str,
        steps: list[StepSpec],
        kind: str = "sleep",
        dry_run: bool = False,
        run_dir: Path | None = None,
        pid: int | None = None,
    ) -> RunRecord:
        with self.db.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO sleep_runs (kind, trigger, status, dry_run, run_dir, pid)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (kind, trigger, RunStatus.RUNNING.value, int(dry_run),
                 str(run_dir) if run_dir else None, pid),
            )
            run_id = int(cur.lastrowid or 0)
            for position, spec in enumerate(steps):
                conn.execute(
                    "INSERT INTO sleep_steps (run_id, name, position, status, on_crash)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (run_id, spec.name, position, StepStatus.PENDING.value,
                     spec.on_crash.value),
                )
        return self.load_run(run_id)

    def load_run(self, run_id: int) -> RunRecord:
        row = self.db.query_one("SELECT * FROM sleep_runs WHERE id = ?", (run_id,))
        if row is None:
            raise KeyError(f"no sleep run {run_id}")
        steps = [
            StepRecord(
                name=r["name"],
                position=r["position"],
                status=StepStatus(r["status"]),
                attempts=r["attempts"],
                on_crash=CrashPolicy(r["on_crash"]),
                checkpoint=_loads(r["checkpoint"]),
                result=_loads(r["result"]),
                error=r["error"],
                started_at=r["started_at"],
                finished_at=r["finished_at"],
            )
            for r in self.db.query(
                "SELECT * FROM sleep_steps WHERE run_id = ? ORDER BY position", (run_id,)
            )
        ]
        return RunRecord(
            id=row["id"],
            kind=row["kind"],
            trigger=row["trigger"],
            status=RunStatus(row["status"]),
            dry_run=bool(row["dry_run"]),
            run_dir=Path(row["run_dir"]) if row["run_dir"] else None,
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            reason=row["reason"],
            steps=steps,
        )

    def open_runs(self) -> list[RunRecord]:
        marks = ", ".join("?" for _ in OPEN_RUN_STATES)
        rows = self.db.query(
            f"SELECT id FROM sleep_runs WHERE status IN ({marks}) ORDER BY id",
            [s.value for s in OPEN_RUN_STATES],
        )
        return [self.load_run(r["id"]) for r in rows]

    def latest_run(self) -> RunRecord | None:
        row = self.db.query_one("SELECT id FROM sleep_runs ORDER BY id DESC LIMIT 1")
        return self.load_run(row["id"]) if row else None

    def last_completed_run(self) -> RunRecord | None:
        row = self.db.query_one(
            "SELECT id FROM sleep_runs WHERE status IN (?, ?) ORDER BY id DESC LIMIT 1",
            (RunStatus.COMPLETED.value, RunStatus.FAILED.value),
        )
        return self.load_run(row["id"]) if row else None

    def set_run_status(
        self,
        run_id: int,
        status: RunStatus,
        *,
        reason: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        finished = status not in OPEN_RUN_STATES
        self.db.execute(
            "UPDATE sleep_runs SET status = ?, reason = COALESCE(?, reason),"
            " summary = COALESCE(?, summary),"
            " finished_at = CASE WHEN ? THEN datetime('now') ELSE finished_at END"
            " WHERE id = ?",
            (
                status.value,
                reason,
                json.dumps(summary) if summary is not None else None,
                int(finished),
                run_id,
            ),
        )

    def set_run_pid(self, run_id: int, pid: int | None) -> None:
        self.db.execute("UPDATE sleep_runs SET pid = ? WHERE id = ?", (pid, run_id))

    # --- steps ----------------------------------------------------------
    def begin_step(self, run_id: int, name: str) -> int:
        """Mark a step running and bump its attempt counter. Returns the attempt number."""
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE sleep_steps SET status = ?, attempts = attempts + 1,"
                " started_at = COALESCE(started_at, datetime('now')),"
                " heartbeat_at = datetime('now'), error = NULL"
                " WHERE run_id = ? AND name = ?",
                (StepStatus.RUNNING.value, run_id, name),
            )
            row = conn.execute(
                "SELECT attempts FROM sleep_steps WHERE run_id = ? AND name = ?",
                (run_id, name),
            ).fetchone()
        if row is None:
            raise KeyError(f"no step {name!r} in run {run_id}")
        return int(row[0])

    def heartbeat(self, run_id: int, name: str) -> None:
        self.db.execute(
            "UPDATE sleep_steps SET heartbeat_at = datetime('now')"
            " WHERE run_id = ? AND name = ?",
            (run_id, name),
        )

    def set_checkpoint(self, run_id: int, name: str, checkpoint: dict[str, Any]) -> None:
        self.db.execute(
            "UPDATE sleep_steps SET checkpoint = ?, heartbeat_at = datetime('now')"
            " WHERE run_id = ? AND name = ?",
            (json.dumps(checkpoint), run_id, name),
        )

    def finish_step(
        self,
        run_id: int,
        name: str,
        status: StepStatus,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        checkpoint: dict[str, Any] | None = None,
        clear_checkpoint: bool = False,
    ) -> None:
        self.db.execute(
            "UPDATE sleep_steps SET status = ?, finished_at = datetime('now'),"
            " result = COALESCE(?, result), error = ?,"
            " checkpoint = CASE WHEN ? THEN NULL ELSE COALESCE(?, checkpoint) END"
            " WHERE run_id = ? AND name = ?",
            (
                status.value,
                json.dumps(result) if result is not None else None,
                error,
                int(clear_checkpoint),
                json.dumps(checkpoint) if checkpoint is not None else None,
                run_id,
                name,
            ),
        )

    def reset_step(self, run_id: int, name: str, *, clear_checkpoint: bool = False) -> None:
        """Put a step back to pending so it will be attempted again."""
        self.db.execute(
            "UPDATE sleep_steps SET status = ?, finished_at = NULL,"
            " checkpoint = CASE WHEN ? THEN NULL ELSE checkpoint END"
            " WHERE run_id = ? AND name = ?",
            (StepStatus.PENDING.value, int(clear_checkpoint), run_id, name),
        )

    def running_steps(self, run_id: int) -> list[StepRecord]:
        run = self.load_run(run_id)
        return [s for s in run.steps if s.status is StepStatus.RUNNING]

    # --- pending work ---------------------------------------------------
    def pending_work_count(self) -> int:
        return int(
            self.db.scalar(
                "SELECT COALESCE(SUM(weight), 0) FROM pending_work"
                " WHERE consumed_run_id IS NULL",
                default=0,
            )
        )

    def add_pending_work(self, kind: str, *, weight: int = 1, ref: str | None = None) -> int:
        return self.db.insert("pending_work", {"kind": kind, "weight": weight, "ref": ref})

    def consume_pending_work(self, run_id: int) -> int:
        cur = self.db.execute(
            "UPDATE pending_work SET consumed_run_id = ? WHERE consumed_run_id IS NULL",
            (run_id,),
        )
        return cur.rowcount


def _loads(raw: Any) -> dict[str, Any] | None:
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else {"value": value}
