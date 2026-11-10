"""Startup crash recovery.

The daemon may have been killed at any instant — mid-step, mid-promotion, mid-backup.
On startup we read the journal, decide per step what its ``on_crash`` policy demands,
and leave the system in a state that is either clean or explicitly resumable. Only
then does the daemon start serving.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field

from gustav.core.events import Events
from gustav.core.journal import CrashPolicy, Journal, RunStatus, StepStatus
from gustav.core.logs import get_logger
from gustav.paths import Paths

#: A rollback handler undoes a half-finished weight-changing step. Stage 4 registers
#: the real ones (restore from the snapshot the step recorded); stage 1 has none.
RollbackHandler = Callable[[int, str], None]
#: A health check returns (ok, message). A failure puts the daemon in safe mode.
HealthCheck = Callable[[], tuple[bool, str]]


@dataclass
class RecoveryReport:
    interrupted_runs: list[int] = field(default_factory=list)
    resumable_runs: list[int] = field(default_factory=list)
    reset_steps: list[str] = field(default_factory=list)
    restarted_steps: list[str] = field(default_factory=list)
    rolled_back_steps: list[str] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)
    failed_runs: list[int] = field(default_factory=list)
    cleaned_staging: list[str] = field(default_factory=list)
    safe_mode: bool = False
    messages: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (
            self.interrupted_runs
            or self.rolled_back_steps
            or self.cleaned_staging
            or self.safe_mode
        )

    def summary(self) -> str:
        if self.clean:
            return "clean startup"
        bits = []
        if self.interrupted_runs:
            bits.append(f"{len(self.interrupted_runs)} interrupted run(s)")
        if self.resumable_runs:
            bits.append(f"resumable: {self.resumable_runs}")
        if self.rolled_back_steps:
            bits.append(f"rolled back: {', '.join(self.rolled_back_steps)}")
        if self.restarted_steps:
            bits.append(f"restarting: {', '.join(self.restarted_steps)}")
        if self.cleaned_staging:
            bits.append(f"removed {len(self.cleaned_staging)} staging dir(s)")
        if self.safe_mode:
            bits.append("entering SAFE MODE")
        return "; ".join(bits)


class Recovery:
    def __init__(
        self,
        *,
        journal: Journal,
        events: Events,
        paths: Paths,
        rollbacks: dict[str, RollbackHandler] | None = None,
        health_checks: list[HealthCheck] | None = None,
        logger=None,
    ):
        self.journal = journal
        self.events = events
        self.paths = paths
        self.rollbacks = rollbacks or {}
        self.health_checks = health_checks or []
        self.log = logger or get_logger("recovery")

    def run(self) -> RecoveryReport:
        report = RecoveryReport()
        self._recover_runs(report)
        self._clean_staging(report)
        self._health(report)
        self.events.emit(
            "daemon.recovery",
            report.summary(),
            level="warning" if not report.clean else "info",
            interrupted=report.interrupted_runs,
            resumable=report.resumable_runs,
            safe_mode=report.safe_mode,
        )
        return report

    # --- journal --------------------------------------------------------
    def _recover_runs(self, report: RecoveryReport) -> None:
        for run in self.journal.open_runs():
            was_running = run.status is RunStatus.RUNNING
            if was_running:
                report.interrupted_runs.append(run.id)
                self.journal.set_run_status(
                    run.id, RunStatus.INTERRUPTED, reason="daemon stopped unexpectedly"
                )
                self.journal.set_run_pid(run.id, None)

            aborted = False
            for step in run.steps:
                label = f"{run.id}:{step.name}"
                if step.status is StepStatus.RUNNING:
                    if step.on_crash is CrashPolicy.ROLLBACK:
                        self._rollback(run.id, step.name, report)
                        self.journal.finish_step(
                            run.id, step.name, StepStatus.FAILED,
                            error="interrupted; rolled back", clear_checkpoint=True,
                        )
                        aborted = True
                    elif step.on_crash is CrashPolicy.SKIP:
                        self.journal.finish_step(
                            run.id, step.name, StepStatus.SKIPPED,
                            error="interrupted; skipped by policy",
                        )
                        report.skipped_steps.append(label)
                    elif step.on_crash is CrashPolicy.RESTART:
                        self._clear_step_dir(run.id, step.name)
                        self.journal.reset_step(run.id, step.name, clear_checkpoint=True)
                        report.restarted_steps.append(label)
                    else:  # RESUME
                        self.journal.reset_step(run.id, step.name)
                        report.reset_steps.append(label)
                elif step.status is StepStatus.PREEMPTED:
                    # Was stopped on purpose; pick it up where it left off.
                    self.journal.reset_step(run.id, step.name)
                    report.reset_steps.append(label)

            if aborted:
                self.journal.set_run_status(
                    run.id, RunStatus.FAILED,
                    reason="a weight-changing step was interrupted and rolled back",
                )
                report.failed_runs.append(run.id)
                continue

            refreshed = self.journal.load_run(run.id)
            if refreshed.next_step is not None:
                report.resumable_runs.append(run.id)
            else:
                self.journal.set_run_status(
                    refreshed.id, RunStatus.COMPLETED, reason="all steps already finished"
                )

    def _rollback(self, run_id: int, step: str, report: RecoveryReport) -> None:
        handler = self.rollbacks.get(step)
        if handler is None:
            report.messages.append(
                f"step {step} needs rollback but no handler is registered (stage 1 has none)"
            )
        else:
            try:
                handler(run_id, step)
            except Exception as exc:  # noqa: BLE001 - never block startup
                report.messages.append(f"rollback of {step} failed: {exc}")
                self.log.exception("rollback of %s failed", step)
        report.rolled_back_steps.append(f"{run_id}:{step}")

    def _clear_step_dir(self, run_id: int, step: str) -> None:
        directory = self.paths.run_dir(run_id) / step
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)

    # --- staging --------------------------------------------------------
    def _clean_staging(self, report: RecoveryReport) -> None:
        """Remove half-written staging directories.

        Atomic promotion stages into ``var/tmp`` and then renames, so anything left
        here is by definition not live and safe to delete.
        """
        if not self.paths.tmp.exists():
            return
        for entry in self.paths.tmp.iterdir():
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            except OSError as exc:
                report.messages.append(f"could not remove staging {entry}: {exc}")
                continue
            report.cleaned_staging.append(entry.name)

    # --- health ---------------------------------------------------------
    def _health(self, report: RecoveryReport) -> None:
        for check in self.health_checks:
            try:
                ok, message = check()
            except Exception as exc:  # noqa: BLE001
                ok, message = False, f"health check raised {type(exc).__name__}: {exc}"
            if not ok:
                report.safe_mode = True
                report.messages.append(message)
                self.log.error("health check failed: %s", message)
