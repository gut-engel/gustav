"""The one always-on process.

``python daemon.py`` starts this. It owns the state machine, the scheduler, the
control plane, the chat responder and the sleep pipeline; everything heavy runs in
supervised children. There is no cron, no systemd timer and no external scheduler:
if the daemon is running, the system is complete.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from gustav.config.load import load_config
from gustav.config.schema import Config
from gustav.core.clock import Clock, RealClock
from gustav.core.events import Events, Telemetry
from gustav.core.gpu_lease import GpuLease
from gustav.core.journal import Journal, RunRecord, RunStatus, StepStatus
from gustav.core.lockfile import PidLock
from gustav.core.logs import get_logger, setup_logging
from gustav.core.recovery import Recovery, RecoveryReport
from gustav.core.scheduler import Observation, Scheduler, SleepDecision
from gustav.core.state import State, StateMachine
from gustav.core.supervisor import Supervisor
from gustav.hw.service import HardwareService, PlanOutcome
from gustav.paths import Paths
from gustav.sleep.pipeline import PipelineRunner, RunOutcome, default_pipeline
from gustav.store.db import Database, open_db
from gustav.wake.responder import EchoResponder, Reply, Responder

META_LAST_SLEEP = "last_sleep_finished_at"
META_LEARNING_PAUSED = "learning_paused"

#: How long a pause request waits for the sleep pipeline to checkpoint and stop.
PAUSE_TIMEOUT_SECONDS = 120.0


class Daemon:
    def __init__(
        self,
        config: Config,
        paths: Paths,
        *,
        clock: Clock | None = None,
        responder: Responder | None = None,
        db: Database | None = None,
        detector=None,
    ):
        self.config = config
        self.paths = paths.ensure()
        self.clock = clock or RealClock()
        self.log = get_logger("daemon")

        self.db = db or open_db(paths.db)
        self.events = Events(self.db, get_logger("events"))
        self.telemetry = Telemetry(self.db)
        self.journal = Journal(self.db)
        self.lock = PidLock(paths.lock_file)
        self.lease = GpuLease("gpu0", logger=get_logger("gpu"))
        self.supervisor = Supervisor(logger=get_logger("supervisor"))
        self.scheduler = Scheduler(config.schedule, self.clock)
        self.hw = HardwareService(
            config, self.paths, self.db, self.events, detector=detector
        )
        self.responder: Responder = responder or EchoResponder()
        self.state = StateMachine(
            State.STARTING, clock=self.clock, on_transition=self._record_transition
        )
        self.runner = PipelineRunner(
            config=config,
            journal=self.journal,
            supervisor=self.supervisor,
            events=self.events,
            lease=self.lease,
            clock=self.clock,
            logger=get_logger("sleep"),
        )

        self.started_at = self.clock.now()
        self.last_activity_at = self.clock.now()
        self.recovery_report: RecoveryReport | None = None
        self.plan_outcome: PlanOutcome | None = None
        self.last_decision: SleepDecision | None = None
        self.last_run_outcome: RunOutcome | None = None

        self._sleep_task: asyncio.Task[None] | None = None
        self._stop_sleep = asyncio.Event()
        self._tasks: list[asyncio.Task[Any]] = []
        self._servers: list[Any] = []
        self._shutdown = asyncio.Event()
        self._manual_trigger: str | None = None
        self._manual_dry_run = False
        self._chat_lock = asyncio.Lock()
        self._current_run_id: int | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        self.lock.acquire(
            {
                "socket": str(self.socket_path),
                "data_dir": str(self.paths.data_dir),
                "version": _version(),
            }
        )
        self.events.emit(
            "daemon.start",
            f"daemon starting (pid {os.getpid()})",
            pid=os.getpid(),
            data_dir=str(self.paths.data_dir),
        )

        self.recovery_report = Recovery(
            journal=self.journal,
            events=self.events,
            paths=self.paths,
            health_checks=[self.responder.health],
            logger=get_logger("recovery"),
        ).run()

        self._plan_hardware(reason="startup")

        if self.recovery_report.safe_mode:
            self.state.transition(State.SAFE_MODE, "startup health check failed")
        else:
            await self.responder.load()
            await self.lease.acquire(
                "inference", evictor=self._evict_inference, reason="serving", timeout=None
            )
            self.state.transition(State.WAKE, "startup complete")

        await self._start_servers()
        self._tasks.append(asyncio.create_task(self._scheduler_loop(), name="scheduler"))
        self._tasks.append(asyncio.create_task(self._maintenance_loop(), name="maintenance"))
        self.log.info("gustav is awake; control socket at %s", self.socket_path)

    async def serve_forever(self) -> None:
        await self._shutdown.wait()

    async def shutdown(self, reason: str = "shutdown requested") -> None:
        """Stop accepting work, checkpoint anything running, release the GPU."""
        if self.state.state in (State.STOPPING, State.STOPPED):
            return
        self.log.info("shutting down: %s", reason)
        self.state.transition(State.STOPPING, reason)
        self.events.emit("daemon.stopping", reason)

        # 1. stop taking requests
        for server in self._servers:
            server.should_exit = True

        # 2. let a running sleep checkpoint and stop
        await self._pause_sleep(f"daemon shutdown: {reason}", wait=True)

        # 3. stop our own loops
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()

        # 4. release the GPU and close up
        with contextlib.suppress(Exception):
            await self.responder.unload()
        with contextlib.suppress(Exception):
            await self.lease.release("inference")

        self.state.transition(State.STOPPED, reason)
        self.events.emit("daemon.stopped", reason)
        for path in (self.socket_path,):
            with contextlib.suppress(FileNotFoundError, OSError):
                path.unlink()
        self.lock.release()
        self.db.close()
        self._shutdown.set()

    def request_shutdown(self, reason: str) -> None:
        asyncio.ensure_future(self.shutdown(reason))

    # ------------------------------------------------------------------
    # servers
    # ------------------------------------------------------------------
    @property
    def socket_path(self) -> Path:
        configured = self.config.instance.socket
        return Path(configured) if configured else self.paths.socket

    async def _start_servers(self) -> None:
        import uvicorn

        from gustav.server.app import create_app

        app = create_app(self)
        level = self.config.logging.level.lower()

        class _Server(uvicorn.Server):
            # The daemon owns its signal handling; uvicorn must not steal SIGTERM.
            def install_signal_handlers(self) -> None:
                return None

        path = self.socket_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_socket():
            with contextlib.suppress(FileNotFoundError, OSError):
                path.unlink()
        uds_server = _Server(
            uvicorn.Config(app, uds=str(path), log_level=level, access_log=False)
        )
        self._servers.append(uds_server)
        self._tasks.append(asyncio.create_task(uds_server.serve(), name="uds-server"))

        if self.config.instance.http_enabled:
            tcp_server = _Server(
                uvicorn.Config(
                    app,
                    host=self.config.instance.http_host,
                    port=self.config.instance.http_port,
                    log_level=level,
                    access_log=False,
                )
            )
            self._servers.append(tcp_server)
            self._tasks.append(asyncio.create_task(tcp_server.serve(), name="tcp-server"))

        # Wait until the socket is actually accepting connections.
        for _ in range(200):
            if path.exists() and _socket_ready(path):
                break
            await asyncio.sleep(0.02)

    # ------------------------------------------------------------------
    # chat (wake phase)
    # ------------------------------------------------------------------
    async def chat(self, message: str, *, session: str | None = None) -> dict[str, Any]:
        if self.state.state in (State.STOPPING, State.STOPPED):
            raise RuntimeError("daemon is shutting down")

        self.last_activity_at = self.clock.now()

        # A message during sleep pauses sleep: checkpoint, hand the GPU back, answer.
        paused = False
        if self.state.state is State.SLEEP:
            paused = await self._pause_sleep("a user message arrived", wait=True)

        async with self._chat_lock:
            if not self.responder.loaded:
                await self.lease.acquire(
                    "inference", evictor=self._evict_inference, reason="serving", timeout=None
                )
                await self.responder.load()
            session_id = self._touch_session(session)
            history = self._history(session_id)
            started = self.clock.monotonic()
            reply: Reply = await self.responder.respond(
                session_id=session_id, message=message, history=history
            )
            latency_ms = int((self.clock.monotonic() - started) * 1000)

        self._record_turn(session_id, "user", message)
        turn_id = self._record_turn(
            session_id,
            "assistant",
            reply.text,
            adapter_versions=reply.adapter_versions,
            base_model=reply.base_model,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            latency_ms=latency_ms,
            tokens_per_second=reply.tokens_per_second,
        )
        self.journal.add_pending_work("turn", ref=str(turn_id))
        self.last_activity_at = self.clock.now()
        return {
            "reply": reply.text,
            "session": session_id,
            "turn_id": turn_id,
            "state": str(self.state.state),
            "sleep_paused": paused,
            "adapters": reply.adapter_versions,
            "latency_ms": latency_ms,
        }

    def _touch_session(self, external_id: str | None) -> int:
        key = external_id or "default"
        row = self.db.query_one("SELECT id FROM sessions WHERE external_id = ?", (key,))
        if row is None:
            return self.db.insert("sessions", {"external_id": key, "source": "api"})
        self.db.execute(
            "UPDATE sessions SET last_activity_at = datetime('now') WHERE id = ?", (row["id"],)
        )
        return int(row["id"])

    def _history(self, session_id: int) -> list[dict[str, str]]:
        rows = self.db.query(
            "SELECT role, content FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, self.config.wake.max_history_turns),
        )
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def _record_turn(self, session_id: int, role: str, content: str, **extra: Any) -> int:
        values: dict[str, Any] = {"session_id": session_id, "role": role, "content": content}
        adapter_versions = extra.pop("adapter_versions", None)
        if adapter_versions:
            values["adapter_versions"] = adapter_versions
        values.update({k: v for k, v in extra.items() if v is not None})
        return self.db.insert("turns", values)

    # ------------------------------------------------------------------
    # sleep
    # ------------------------------------------------------------------
    async def request_sleep(
        self, *, dry_run: bool = False, force: bool = False
    ) -> dict[str, Any]:
        if self.state.state is State.SLEEP:
            return {"accepted": False, "reason": "already asleep", "run_id": self._current_run_id}
        obs = self._observe(manual="manual")
        decision = self.scheduler.decide(obs)
        if not decision.should_sleep and not force:
            return {"accepted": False, "reason": decision.reason}
        self._manual_dry_run = dry_run
        self._manual_trigger = "manual"
        run_id = await self._enter_sleep("manual", dry_run=dry_run)
        return {"accepted": True, "reason": "sleeping now", "run_id": run_id,
                "dry_run": dry_run}

    async def request_wake(self) -> dict[str, Any]:
        if self.state.state is not State.SLEEP:
            return {"accepted": False, "reason": f"not asleep (state {self.state.state})"}
        paused = await self._pause_sleep("woken by request", wait=True)
        return {"accepted": paused, "reason": "sleep paused and checkpointed"}

    async def _enter_sleep(self, trigger: str, *, dry_run: bool = False) -> int:
        self.state.transition(State.SLEEP_PENDING, f"trigger: {trigger}")
        run = self._open_or_create_run(trigger, dry_run=dry_run)
        self._current_run_id = run.id
        self._stop_sleep = asyncio.Event()
        self.state.transition(State.SLEEP, f"run {run.id} ({trigger})")
        self.events.emit(
            "sleep.started",
            f"sleep run {run.id} started ({trigger})",
            run_id=run.id,
            trigger=trigger,
            dry_run=dry_run,
            resumed=run.done_count > 0,
        )
        self._sleep_task = asyncio.create_task(self._sleep_body(run), name=f"sleep-{run.id}")
        return run.id

    def _open_or_create_run(self, trigger: str, *, dry_run: bool) -> RunRecord:
        for run in self.journal.open_runs():
            if run.next_step is not None:
                self.journal.set_run_status(run.id, RunStatus.RUNNING, reason=f"resumed ({trigger})")
                self.journal.set_run_pid(run.id, os.getpid())
                return self.journal.load_run(run.id)
        pipeline = default_pipeline(self.config)
        run = self.journal.create_run(
            trigger=trigger, steps=pipeline.specs, dry_run=dry_run, pid=os.getpid()
        )
        run_dir = self.paths.run_dir(run.id)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.db.execute("UPDATE sleep_runs SET run_dir = ? WHERE id = ?", (str(run_dir), run.id))
        self.journal.consume_pending_work(run.id)
        return self.journal.load_run(run.id)

    async def _sleep_body(self, run: RunRecord) -> None:
        deadline = self.clock.monotonic() + self.config.schedule.max_duration_minutes * 60
        try:
            # Training wants the card: hand it over before any GPU step starts.
            outcome = await self.runner.run(
                run, default_pipeline(self.config), stop_event=self._stop_sleep, deadline=deadline
            )
            self.last_run_outcome = outcome
            if outcome.status in (RunStatus.COMPLETED, RunStatus.FAILED):
                self.db.set_meta(META_LAST_SLEEP, self.clock.now().isoformat())
        except asyncio.CancelledError:
            self.journal.set_run_status(run.id, RunStatus.PAUSED, reason="daemon stopping")
            raise
        except Exception as exc:  # noqa: BLE001 - a broken sleep must not kill chat
            self.log.exception("sleep run %s crashed", run.id)
            self.journal.set_run_status(
                run.id, RunStatus.FAILED, reason=f"{type(exc).__name__}: {exc}"
            )
            self.events.emit(
                "sleep.crashed", f"sleep run {run.id} crashed: {exc}", level="error", run_id=run.id
            )
            self.db.set_meta(META_LAST_SLEEP, self.clock.now().isoformat())
        finally:
            self.journal.set_run_pid(run.id, None)
            self._current_run_id = None
            if self.state.state is State.SLEEP:
                self.state.transition(State.WAKE, "sleep finished")
            await self._reload_for_serving()

    async def _pause_sleep(self, reason: str, *, wait: bool) -> bool:
        """Ask a running sleep to checkpoint and stop. True if a run was paused."""
        task = self._sleep_task
        if task is None or task.done():
            return False
        self.log.info("pausing sleep: %s", reason)
        self._stop_sleep.set()
        await self.lease.request_return(reason)
        if not wait:
            return True
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=PAUSE_TIMEOUT_SECONDS)
        except (TimeoutError, asyncio.TimeoutError):
            self.log.error("sleep did not stop within %.0fs; cancelling", PAUSE_TIMEOUT_SECONDS)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            self.log.exception("sleep task failed while pausing")
        finally:
            self._sleep_task = None
        return True

    async def _reload_for_serving(self) -> None:
        if self.state.state in (State.STOPPING, State.STOPPED):
            return
        await self.lease.acquire(
            "inference", evictor=self._evict_inference, reason="serving", timeout=None
        )
        if not self.responder.loaded:
            await self.responder.load()

    async def _evict_inference(self) -> None:
        """The GPU lease calls this when a sleep step needs the card."""
        self.log.info("unloading inference to free the GPU")
        await self.responder.unload()
        await self.lease.release("inference")

    # ------------------------------------------------------------------
    # loops
    # ------------------------------------------------------------------
    def _plan_hardware(self, *, reason: str, force: bool = False) -> PlanOutcome:
        """Detect the hardware and make sure a usable plan exists.

        Runs at startup and on the maintenance timer. A base-model change never
        happens here: it becomes a pending approval.
        """
        try:
            outcome = self.hw.ensure_plan(force=force)
        except Exception as exc:  # noqa: BLE001 - never block startup on the planner
            self.log.exception("hardware planning failed")
            self.events.emit(
                "hardware.planner_failed", f"{type(exc).__name__}: {exc}", level="error"
            )
            return PlanOutcome(plan=None, source="planner", error=str(exc))

        self.plan_outcome = outcome
        if outcome.error:
            self.events.emit("hardware.plan_problem", outcome.error, level="error")
        for message in outcome.messages:
            self.events.emit("hardware.notice", message, level="warning")
        if outcome.changed and outcome.plan is not None:
            self.log.info("hardware plan (%s): %s", reason, outcome.plan.summary())
            for line in outcome.plan.explanation:
                self.log.info("  %s", line)
        return outcome

    async def _scheduler_loop(self) -> None:
        tick = max(self.config.schedule.tick_seconds, 0.05)
        while True:
            try:
                await asyncio.sleep(tick)
                if self.state.state is not State.WAKE:
                    continue
                obs = self._observe()
                decision = self.scheduler.decide(obs)
                self.last_decision = decision
                if decision.should_sleep:
                    self.events.emit(
                        "sleep.trigger",
                        f"trigger {decision.trigger}: {decision.reason}",
                        trigger=decision.trigger,
                    )
                    await self._enter_sleep(decision.trigger or "auto")
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                self.log.exception("scheduler tick failed")
                await asyncio.sleep(tick)

    async def _maintenance_loop(self) -> None:
        interval = max(self.config.maintenance.health_check_minutes * 60, 5)
        recheck_every = max(self.config.maintenance.hardware_recheck_minutes * 60, interval)
        last_hardware_check = self.clock.monotonic()
        while True:
            try:
                await asyncio.sleep(interval)
                ok, message = self.responder.health()
                self.telemetry.record("daemon.alive", 1.0, state=str(self.state.state))
                if self.clock.monotonic() - last_hardware_check >= recheck_every:
                    last_hardware_check = self.clock.monotonic()
                    self._plan_hardware(reason="periodic re-check", force=True)
                if not ok and self.state.state is State.WAKE:
                    self.events.emit("daemon.unhealthy", message, level="error")
                    self.state.transition(State.SAFE_MODE, message)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                self.log.exception("maintenance tick failed")

    def _observe(self, *, manual: str | None = None) -> Observation:
        last_sleep = self.db.get_meta(META_LAST_SLEEP)
        resumable = any(r.next_step is not None for r in self.journal.open_runs())
        return Observation(
            state=self.state.state,
            now=self.clock.now(),
            last_activity_at=self.last_activity_at,
            pending_items=self.journal.pending_work_count(),
            learning_paused=self.learning_paused,
            last_sleep_finished_at=_parse_dt(last_sleep),
            manual_trigger=manual,
            gpu_temp_c=self._gpu_temperature(),
            resumable_run=resumable,
        )

    def _gpu_temperature(self) -> float | None:
        """Hottest GPU, for the scheduler's thermal guard. Never raises."""
        try:
            return self.hw.hottest_gpu_c()
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # control surface
    # ------------------------------------------------------------------
    @property
    def learning_paused(self) -> bool:
        return bool(self.db.get_meta(META_LEARNING_PAUSED, False))

    def set_learning_paused(self, paused: bool) -> dict[str, Any]:
        self.db.set_meta(META_LEARNING_PAUSED, paused)
        self.events.emit(
            "learning.paused" if paused else "learning.resumed",
            "learning paused by operator" if paused else "learning resumed by operator",
        )
        return {"learning_paused": paused}

    def status(self) -> dict[str, Any]:
        run = self.journal.latest_run()
        obs = self._observe()
        open_runs = self.journal.open_runs()
        failed_steps = [
            {"run": r.id, "step": s.name, "error": s.error}
            for r in ([run] if run else [])
            for s in r.steps
            if s.status is StepStatus.FAILED
        ]
        approvals = self.db.query(
            "SELECT id, kind, subject, created_at FROM approvals WHERE status = 'pending'"
            " ORDER BY id"
        )
        return {
            "state": str(self.state.state),
            "pid": os.getpid(),
            "version": _version(),
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "uptime_seconds": round((self.clock.now() - self.started_at).total_seconds(), 1),
            "learning_paused": self.learning_paused,
            "last_activity_at": self.last_activity_at.isoformat(timespec="seconds"),
            "model": self.responder.describe(),
            "gpu": {
                "holder": self.lease.holder,
                "return_requested": self.lease.return_requested,
                "reason": self.lease.info().reason,
            },
            "sleep": {
                "current_run": self._current_run_id,
                "last_run": _run_summary(run),
                "resumable_runs": [r.id for r in open_runs if r.next_step is not None],
                "last_finished_at": self.db.get_meta(META_LAST_SLEEP),
                "next": self.scheduler.describe_next(obs),
                "decision": (
                    {"trigger": self.last_decision.trigger, "reason": self.last_decision.reason}
                    if self.last_decision
                    else None
                ),
                "windows": self.config.schedule.windows,
            },
            "pending_items": obs.pending_items,
            "pending_threshold": self.config.schedule.pending_threshold,
            "failed_steps": failed_steps,
            "pending_approvals": [dict(a) for a in approvals],
            "recovery": (
                {
                    "summary": self.recovery_report.summary(),
                    "messages": self.recovery_report.messages,
                }
                if self.recovery_report
                else None
            ),
            "hardware": self._hardware_status(),
            # Filled in by a later stage; declared now so `ai status` has a stable shape.
            "last_backup": self.db.get_meta("last_backup"),
        }

    def _hardware_status(self) -> dict[str, Any]:
        plan = self.hw.stored_plan()
        outcome = self.plan_outcome
        return {
            "fingerprint": self.hw.hardware.fingerprint(),
            "gpus": [g.describe() for g in self.hw.hardware.gpus],
            "plan": plan.summary() if plan else None,
            "plan_source": outcome.source if outcome else None,
            "base_model": self.hw.active_base_model,
            "vram_inference_mb": plan.vram_inference_mb if plan else None,
            "vram_training_mb": plan.vram_training_mb if plan else None,
            "expansion_budget": (
                {
                    "max_adapters": plan.expansion.max_adapters,
                    "max_lora_rank": plan.expansion.max_lora_rank,
                    "max_adapter_vram_mb": plan.expansion.max_adapter_vram_mb,
                }
                if plan
                else None
            ),
            "temperatures_c": self._gpu_temperatures_quiet(),
            "warnings": plan.warnings if plan else [],
            "problem": outcome.error if outcome else None,
        }

    def _gpu_temperatures_quiet(self) -> dict[int, float]:
        try:
            return self.hw.gpu_temperatures()
        except Exception:  # noqa: BLE001
            return {}

    def _record_transition(self, transition) -> None:
        self.db.insert(
            "state_history",
            {
                "from_state": str(transition.source) if transition.source else None,
                "to_state": str(transition.target),
                "reason": transition.reason,
            },
        )
        self.log.info(
            "state %s -> %s (%s)",
            transition.source or "-",
            transition.target,
            transition.reason,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run_summary(run: RunRecord | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "id": run.id,
        "trigger": run.trigger,
        "status": str(run.status),
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "reason": run.reason,
        "steps": [
            {"name": s.name, "status": str(s.status), "attempts": s.attempts, "error": s.error}
            for s in run.steps
        ],
    }


def _parse_dt(raw: Any) -> datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _socket_ready(path: Path) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            sock.connect(str(path))
        return True
    except OSError:
        return False


def _version() -> str:
    from gustav import __version__

    return __version__


async def run_daemon(config: Config, paths: Paths) -> int:
    daemon = Daemon(config, paths)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, daemon.request_shutdown, f"received {sig.name}")
    await daemon.start()
    await daemon.serve_forever()
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="daemon.py", description="Run the gustav daemon (wake/sleep cycle)."
    )
    parser.add_argument("-c", "--config", help="path to config.yaml")
    parser.add_argument("--data-dir", help="override instance.data_dir")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument(
        "--no-console-log", action="store_true", help="log to file only"
    )
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {}
    if args.data_dir:
        overrides["instance"] = {"data_dir": args.data_dir}
    logging_overrides: dict[str, Any] = {}
    if args.log_level:
        logging_overrides["level"] = args.log_level
    if args.no_console_log:
        logging_overrides["console"] = False
    if logging_overrides:
        overrides["logging"] = logging_overrides

    config, paths = load_config(args.config, overrides=overrides)
    paths.ensure()
    log_file = Path(config.logging.file) if config.logging.file else paths.log_file
    setup_logging(config.logging, log_file, force=True)

    try:
        return asyncio.run(run_daemon(config, paths))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
