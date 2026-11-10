"""The sleep pipeline and its runner.

The runner is the only thing that executes steps, and it does so strictly through the
journal: a step is recorded, run in a child process, and recorded again. Stage 1 ships
the real step *names* and crash policies with placeholder handlers, so the hard parts
(resume, retry, preemption, watchdog) are proven before real work hangs off them.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gustav.config.schema import Config
from gustav.core.clock import Clock, RealClock
from gustav.core.events import Events
from gustav.core.gpu_lease import GpuLease
from gustav.core.journal import (
    CrashPolicy,
    Journal,
    RunRecord,
    RunStatus,
    StepSpec,
    StepStatus,
)
from gustav.core.logs import get_logger
from gustav.core.supervisor import ChildSpec, Outcome, Supervisor

PLACEHOLDER = "gustav.sleep.stages.placeholder:run_step"


@dataclass(frozen=True)
class Step:
    spec: StepSpec
    target: str = PLACEHOLDER


@dataclass(frozen=True)
class Pipeline:
    name: str
    steps: tuple[Step, ...]

    @property
    def specs(self) -> list[StepSpec]:
        return [s.spec for s in self.steps]

    def target_for(self, name: str) -> str:
        for step in self.steps:
            if step.spec.name == name:
                return step.target
        raise KeyError(name)

    def spec_for(self, name: str) -> StepSpec:
        for step in self.steps:
            if step.spec.name == name:
                return step.spec
        raise KeyError(name)


def default_pipeline(config: Config) -> Pipeline:
    """The night's work, in order. Section 7 of the spec.

    ``on_crash`` is the interesting column: anything that can touch weights is
    ``rollback`` (undo and stop) or ``restart`` (cheap, so just redo it cleanly);
    long grinding steps are ``resume`` so a reboot does not throw away the night.
    """
    g = True  # needs the GPU
    steps: list[Step] = [
        Step(StepSpec("snapshot", CrashPolicy.RESTART, critical=True,
                      description="verified backup before anything may change weights")),
        Step(StepSpec("consolidation", CrashPolicy.RESUME, requires_gpu=g,
                      description="extract durable facts and corrections from the day")),
        Step(StepSpec("reflect", CrashPolicy.RESUME, requires_gpu=g,
                      description="name the topics where sandbox results show weakness")),
        Step(StepSpec("fetch_docs", CrashPolicy.RESUME,
                      description="fetch allowlisted documentation for those topics")),
        Step(StepSpec("study", CrashPolicy.RESUME, requires_gpu=g,
                      description="turn documentation into cited Q&A and tasks")),
        Step(StepSpec("practice", CrashPolicy.RESUME, requires_gpu=g,
                      description="solve tasks and let the compiler and tests judge")),
        Step(StepSpec("dreaming", CrashPolicy.RESUME, requires_gpu=g,
                      description="verifiable self-practice with rejection sampling")),
        Step(StepSpec("training", CrashPolicy.RESUME, requires_gpu=g,
                      description="train a new adapter version per module")),
        Step(StepSpec("gate", CrashPolicy.RESTART, requires_gpu=g, critical=True,
                      description="held-out benchmark and old-skills regression gate")),
        Step(StepSpec("expansion", CrashPolicy.ROLLBACK, requires_gpu=g,
                      description="grow capacity only if measurements demand it")),
        Step(StepSpec("retest", CrashPolicy.RESTART, requires_gpu=g,
                      description="re-run the original failures and update the skills table")),
        Step(StepSpec("speed", CrashPolicy.RESTART, requires_gpu=g,
                      description="draft-model distillation and router training")),
        Step(StepSpec("report", CrashPolicy.RESTART,
                      description="write the HTML report for this run")),
    ]
    disabled = set(config.sleep.disabled_steps)
    return Pipeline(name="sleep", steps=tuple(s for s in steps if s.spec.name not in disabled))


@dataclass
class RunOutcome:
    status: RunStatus
    reason: str
    completed: list[str]
    failed: list[str]
    skipped: list[str]
    preempted_at: str | None = None


class PipelineRunner:
    """Executes a pipeline, one step at a time, resuming from the journal."""

    def __init__(
        self,
        *,
        config: Config,
        journal: Journal,
        supervisor: Supervisor,
        events: Events,
        lease: GpuLease | None = None,
        clock: Clock | None = None,
        logger=None,
    ):
        self.config = config
        self.journal = journal
        self.supervisor = supervisor
        self.events = events
        self.lease = lease
        self.clock = clock or RealClock()
        self.log = logger or get_logger("sleep")
        self.current_step: str | None = None

    async def run(
        self,
        run: RunRecord,
        pipeline: Pipeline,
        *,
        stop_event: asyncio.Event | None = None,
        deadline: float | None = None,
    ) -> RunOutcome:
        completed: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []
        run_dir = run.run_dir or Path.cwd()
        run_dir.mkdir(parents=True, exist_ok=True)

        for record in run.steps:
            if record.status is StepStatus.DONE:
                completed.append(record.name)
                continue
            if record.status is StepStatus.SKIPPED:
                skipped.append(record.name)
                continue
            if stop_event is not None and stop_event.is_set():
                return self._pause(run, record.name, "a user request needs the GPU",
                                   completed, failed, skipped)
            if deadline is not None and self.clock.monotonic() > deadline:
                return self._pause(run, record.name, "sleep exceeded its maximum duration",
                                   completed, failed, skipped)

            spec = pipeline.spec_for(record.name)
            outcome = await self._run_step(
                run, spec, pipeline.target_for(record.name), stop_event=stop_event
            )
            if outcome is StepStatus.DONE:
                completed.append(record.name)
            elif outcome is StepStatus.SKIPPED:
                skipped.append(record.name)
            elif outcome is StepStatus.PREEMPTED:
                return self._pause(run, record.name, "preempted", completed, failed, skipped)
            else:
                failed.append(record.name)
                if spec.critical:
                    self.journal.set_run_status(
                        run.id, RunStatus.FAILED, reason=f"critical step {spec.name} failed"
                    )
                    self.events.emit(
                        "sleep.failed",
                        f"critical step {spec.name} failed; nothing was promoted",
                        level="error",
                        run_id=run.id,
                    )
                    return RunOutcome(
                        RunStatus.FAILED, f"critical step {spec.name} failed",
                        completed, failed, skipped,
                    )

        self.journal.set_run_status(
            run.id,
            RunStatus.COMPLETED,
            reason="all steps finished",
            summary={"completed": completed, "failed": failed, "skipped": skipped},
        )
        self.events.emit(
            "sleep.completed",
            f"sleep run {run.id} finished: {len(completed)} done, "
            f"{len(failed)} failed, {len(skipped)} skipped",
            run_id=run.id,
        )
        return RunOutcome(RunStatus.COMPLETED, "finished", completed, failed, skipped)

    # --- one step -------------------------------------------------------
    async def _run_step(
        self,
        run: RunRecord,
        spec: StepSpec,
        target: str,
        *,
        stop_event: asyncio.Event | None,
    ) -> StepStatus:
        sup = self.config.supervisor
        max_attempts = spec.max_attempts or sup.max_attempts
        backoff = sup.retry_backoff_seconds
        self.current_step = spec.name

        for _ in range(max_attempts + 1):
            attempt = self.journal.begin_step(run.id, spec.name)
            if attempt > max_attempts:
                self.journal.finish_step(
                    run.id,
                    spec.name,
                    StepStatus.FAILED,
                    error=f"gave up after {max_attempts} attempts",
                )
                self.events.emit(
                    "sleep.step.gave_up",
                    f"step {spec.name} failed {max_attempts}x; skipped for tonight",
                    level="warning",
                    run_id=run.id,
                    step=spec.name,
                )
                return StepStatus.FAILED

            record = self.journal.load_run(run.id).step(spec.name)
            checkpoint = record.checkpoint if record else None
            self.events.emit(
                "sleep.step.start",
                f"step {spec.name} (attempt {attempt})",
                run_id=run.id,
                step=spec.name,
                attempt=attempt,
                resumed=bool(checkpoint),
            )

            lease_holder = f"sleep:{spec.name}"
            acquired = False
            try:
                if spec.requires_gpu and self.lease is not None:
                    await self.lease.acquire(
                        lease_holder, reason=f"sleep step {spec.name}", timeout=None
                    )
                    acquired = True

                child = ChildSpec(
                    name=f"{run.id}:{spec.name}",
                    target=target,
                    kwargs={
                        "run_id": run.id,
                        "step": spec.name,
                        "run_dir": str(run.run_dir) if run.run_dir else None,
                        "checkpoint": checkpoint,
                        "attempt": attempt,
                        "dry_run": run.dry_run,
                        "simulated_seconds": self.config.sleep.simulated_step_seconds,
                    },
                    timeout_seconds=spec.timeout_seconds or sup.step_timeout_seconds,
                    heartbeat_timeout_seconds=sup.heartbeat_timeout_seconds,
                    memory_limit_mb=sup.memory_limit_mb,
                    stop_grace_seconds=sup.kill_grace_seconds * 2,
                    kill_grace_seconds=sup.kill_grace_seconds,
                )
                result = await self.supervisor.run(
                    child,
                    stop_event=stop_event,
                    on_message=lambda msg: self._on_child_message(run.id, spec.name, msg),
                )
            finally:
                if acquired and self.lease is not None:
                    await self.lease.release(lease_holder)

            if result.outcome is Outcome.OK:
                self.journal.finish_step(
                    run.id, spec.name, StepStatus.DONE,
                    result=result.payload, clear_checkpoint=True,
                )
                self.events.emit(
                    "sleep.step.done",
                    f"step {spec.name} done in {result.duration_seconds:.1f}s",
                    run_id=run.id, step=spec.name,
                    duration=round(result.duration_seconds, 2),
                )
                return StepStatus.DONE

            if result.outcome is Outcome.PREEMPTED:
                self.journal.finish_step(
                    run.id, spec.name, StepStatus.PREEMPTED,
                    checkpoint=result.checkpoint, error=result.error,
                )
                self.events.emit(
                    "sleep.step.preempted",
                    f"step {spec.name} checkpointed and stopped on request",
                    run_id=run.id, step=spec.name,
                )
                return StepStatus.PREEMPTED

            self.journal.finish_step(
                run.id, spec.name, StepStatus.FAILED,
                error=f"{result.outcome}: {result.error}", checkpoint=result.checkpoint,
            )
            self.events.emit(
                "sleep.step.failed",
                f"step {spec.name} {result.outcome}: {result.error}",
                level="warning",
                run_id=run.id, step=spec.name, attempt=attempt,
                outcome=str(result.outcome),
            )
            if not result.outcome.retryable:
                return StepStatus.FAILED
            if attempt >= max_attempts:
                continue  # one more loop turn reports the give-up
            await self.clock.sleep(backoff)
            backoff *= sup.retry_backoff_factor

        return StepStatus.FAILED

    def _on_child_message(self, run_id: int, step: str, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "checkpoint":
            # Written immediately: a kill -9 one instruction later must still resume.
            self.journal.set_checkpoint(run_id, step, msg.get("data") or {})
        elif kind == "heartbeat":
            self.journal.heartbeat(run_id, step)
        elif kind == "log":
            self.log.log(
                {"warning": 30, "error": 40}.get(str(msg.get("level")), 20),
                f"[{step}] {msg.get('message')}",
            )

    def _pause(
        self,
        run: RunRecord,
        step: str,
        reason: str,
        completed: list[str],
        failed: list[str],
        skipped: list[str],
    ) -> RunOutcome:
        self.journal.set_run_status(run.id, RunStatus.PAUSED, reason=reason)
        self.events.emit(
            "sleep.paused", f"sleep paused at {step}: {reason}", run_id=run.id, step=step
        )
        return RunOutcome(RunStatus.PAUSED, reason, completed, failed, skipped, preempted_at=step)
