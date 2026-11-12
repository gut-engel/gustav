"""The one always-on process.

``python daemon.py`` starts this. It owns the state machine, the scheduler, the
control plane, the chat responder and the sleep pipeline; everything heavy runs in
supervised children. There is no cron, no systemd timer and no external scheduler:
if the daemon is running, the system is complete.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from gustav.backup.guard import WeightChangeGuard
from gustav.backup.restore import Restorer
from gustav.backup.retention import RetentionPolicy, plan_retention
from gustav.backup.snapshot import BackupError, SnapshotStore
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
from gustav.core.supervisor import ChildSpec, Supervisor
from gustav.hw.service import HardwareService, PlanOutcome
from gustav.paths import Paths
from gustav.sleep.pipeline import PipelineRunner, RunOutcome, default_pipeline
from gustav.store.db import Database, open_db
from gustav.wake.commands import CommandHandler, CommandResult
from gustav.wake.implicit import detect as detect_implicit
from gustav.wake.responder import EchoResponder, Reply, Responder
from gustav.wake import secrets as secret_scanner

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
        config_file: Path | None = None,
    ):
        self.config = config
        self.paths = paths.ensure()
        self._config_file = Path(config_file) if config_file else None
        self.clock = clock or RealClock()
        self.log = get_logger("daemon")

        self.db = db or open_db(paths.db)
        self.events = Events(self.db, get_logger("events"))
        self.telemetry = Telemetry(self.db)
        self.journal = Journal(self.db)
        self.lock = PidLock(paths.lock_file)
        self.backups = SnapshotStore(
            self.paths,
            location=config.backup.location,
            db=self.db,
            events=self.events,
            min_free_mb=config.backup.min_free_mb,
            warn_free_mb=config.backup.warn_free_mb,
            logger=get_logger("backup"),
        )
        self.guard = WeightChangeGuard(
            self.backups,
            restorer=Restorer(self.backups, events=self.events),
            db=self.db,
            events=self.events,
            enabled=config.backup.enabled,
            logger=get_logger("guard"),
        )
        self.lease = GpuLease("gpu0", logger=get_logger("gpu"))
        self.supervisor = Supervisor(logger=get_logger("supervisor"))
        self.scheduler = Scheduler(config.schedule, self.clock)
        self.hw = HardwareService(
            config, self.paths, self.db, self.events, detector=detector
        )
        self.responder: Responder = responder or self._default_responder()
        self.commands = CommandHandler(self.db, self.journal, self.events)
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
            paths=self.paths,
            logger=get_logger("sleep"),
        )

        self.started_at = self.clock.now()
        self.last_activity_at = self.clock.now()
        self.recovery_report: RecoveryReport | None = None
        self.plan_outcome: PlanOutcome | None = None
        #: Why the model is not loaded, if it is not. Shown by `ai status`.
        self.load_error: str | None = None
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

    def _default_responder(self) -> Responder:
        """The real harness when the ML stack is installed, the echo otherwise.

        ``inference.responder`` (or ``GUSTAV_RESPONDER``) forces the choice: ``echo``
        runs the whole daemon — scheduler, journal, recovery, the control plane —
        without a GPU or any weights, which is how most of the test suite stays fast.
        """
        import importlib.util

        choice = (
            os.environ.get("GUSTAV_RESPONDER")
            or (self.config.inference or {}).get("responder")
            or "auto"
        ).lower()
        if choice == "echo":
            return EchoResponder()
        if choice not in ("auto", "harness"):
            raise ValueError(
                f"inference.responder must be auto, harness or echo, not {choice!r}"
            )
        if importlib.util.find_spec("torch") is None:
            if choice == "harness":
                raise RuntimeError(
                    "inference.responder is 'harness' but torch is not installed; "
                    'install the ML extras with: pip install -e ".[ml]"'
                )
            self.log.warning(
                "torch is not installed, so chat is an echo; "
                'install the ML extras with: pip install -e ".[ml]"'
            )
            return EchoResponder()
        from gustav.inference.harness import Harness

        return Harness(
            config=self.config,
            paths=self.paths,
            hardware=self.hw,
            events=self.events,
            telemetry=self.telemetry,
            db=self.db,
            logger=get_logger("harness"),
        )

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

        # A project that has no registry or source map yet gets the shipped defaults,
        # once. An absent config.yaml is left absent: every key has a default, so an
        # empty project is already a valid one, and writing a file nobody asked for
        # would be the opposite of helpful.
        from gustav.data import install_defaults

        installed = install_defaults(self.paths.root, only=("registry.yaml", "sources.yaml"))
        for path in installed.created:
            self.log.info("wrote the default %s", path)
            self.events.emit(
                "startup.defaults",
                f"this project had no {path.name}, so the shipped default was written "
                f"to {path}",
                path=str(path),
            )

        # An interrupted weight change is undone before anything else runs.
        undone = self.guard.rollback_interrupted()
        for item in undone:
            self.events.emit(
                "weights.recovered" if item.get("restored") else "weights.recovery_failed",
                f"interrupted {item['operation']}: "
                + (
                    f"restored from {item['snapshot']}"
                    if item.get("restored")
                    else f"could NOT be rolled back ({item.get('detail')})"
                ),
                level="warning" if item.get("restored") else "error",
                **{k: v for k, v in item.items() if k != "detail"},
            )

        self.recovery_report = Recovery(
            journal=self.journal,
            events=self.events,
            paths=self.paths,
            rollbacks={
                name: self._rollback_step
                for name in ("expansion", "training", "gate", "snapshot")
            },
            health_checks=[self.responder.health],
            logger=get_logger("recovery"),
        ).run()

        self._plan_hardware(reason="startup")

        if self.recovery_report.safe_mode:
            self.state.transition(State.SAFE_MODE, "startup health check failed")
        else:
            await self.lease.acquire(
                "inference", evictor=self._evict_inference, reason="serving", timeout=None
            )
            if await self._try_load_model("startup"):
                self.state.transition(State.WAKE, "startup complete")
            else:
                # The control plane still has to come up: `ai status` is how you find
                # out what went wrong, so dying here would be the worst thing to do.
                self.state.transition(State.SAFE_MODE, self.load_error or "the model would not load")

        await self._start_servers()
        self._tasks.append(asyncio.create_task(self._scheduler_loop(), name="scheduler"))
        self._tasks.append(asyncio.create_task(self._maintenance_loop(), name="maintenance"))
        self.log.info("gustav is awake; control socket at %s", self.socket_path)

    def _rollback_step(self, run_id: int, step: str) -> None:
        """Undo a weight-changing step that was interrupted (crash policy `rollback`)."""
        result = self.guard.rollback_step(run_id, step)
        if result is None:
            self.log.info(
                "step %s of run %s had not changed any weights; nothing to roll back",
                step, run_id,
            )
            return
        self.log.warning(
            "rolled %s of run %s back from snapshot %s", step, run_id, result.snapshot_id
        )

    async def _try_load_model(self, reason: str) -> bool:
        """Load the model, or record why not. Never raises."""
        try:
            await self.responder.load()
        except Exception as exc:  # noqa: BLE001 - a bad model must not stop the daemon
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.log.error("the model did not load (%s): %s", reason, exc)
            self.events.emit(
                "inference.load_failed",
                f"the model did not load: {exc}".split("\n")[0],
                level="error",
                detail=self.load_error,
            )
            return False
        self.load_error = None
        return True

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
        """One turn. Commands are handled here; everything else reaches the model."""
        session_id, command = self._before_model(session, message)
        if command is not None:
            return self._command_response(session_id, command)

        paused = await self._prepare_to_answer()
        stored, findings = self._store_text(message)
        async with self._chat_lock:
            await self._ensure_loaded()
            history = self._history(session_id)
            signal = self._implicit_signal(session_id, message)
            started = self.clock.monotonic()
            reply: Reply = await self.responder.respond(
                session_id=session_id, message=message, history=history
            )
            latency_ms = int((self.clock.monotonic() - started) * 1000)

        turn_id = self._log_exchange(
            session_id, message, stored, findings, reply, latency_ms, signal
        )
        self.last_activity_at = self.clock.now()
        return {
            "reply": reply.text,
            "session": session_id,
            "turn_id": turn_id,
            "state": str(self.state.state),
            "sleep_paused": paused,
            "adapters": reply.adapter_versions,
            "latency_ms": latency_ms,
            "tokens_per_second": reply.tokens_per_second,
            "redacted": [f.kind for f in findings],
            "meta": reply.meta,
        }

    async def chat_stream(self, message: str, *, session: str | None = None):
        """The same turn, streamed. Yields dicts the server turns into SSE events."""
        session_id, command = self._before_model(session, message)
        if command is not None:
            payload = self._command_response(session_id, command)
            yield {"type": "command", **payload}
            yield {"type": "done", "session": session_id, "turn_id": None}
            return

        paused = await self._prepare_to_answer()
        stored, findings = self._store_text(message)
        yield {
            "type": "start",
            "session": session_id,
            "sleep_paused": paused,
            "redacted": [f.kind for f in findings],
        }

        pieces: list[str] = []
        async with self._chat_lock:
            await self._ensure_loaded()
            history = self._history(session_id)
            signal = self._implicit_signal(session_id, message)
            started = self.clock.monotonic()
            streamer = getattr(self.responder, "stream", None)
            if streamer is None:
                reply = await self.responder.respond(
                    session_id=session_id, message=message, history=history
                )
                pieces.append(reply.text)
                yield {"type": "chunk", "text": reply.text}
            else:
                async for chunk in streamer(
                    session_id=session_id, message=message, history=history
                ):
                    if chunk.text:
                        pieces.append(chunk.text)
                        yield {"type": "chunk", "text": chunk.text}
                reply = self._reply_from_stream(pieces)
            latency_ms = int((self.clock.monotonic() - started) * 1000)

        turn_id = self._log_exchange(
            session_id, message, stored, findings, reply, latency_ms, signal
        )
        self.last_activity_at = self.clock.now()
        yield {
            "type": "done",
            "session": session_id,
            "turn_id": turn_id,
            "latency_ms": latency_ms,
            "adapters": reply.adapter_versions,
            "tokens_per_second": reply.tokens_per_second,
            "meta": reply.meta,
        }

    def _reply_from_stream(self, pieces: list[str]) -> Reply:
        """Build the record for a streamed answer from the backend's own numbers."""
        text = "".join(pieces)
        getter = getattr(self.responder, "last_generation", None)
        generation = getter() if callable(getter) else None
        if generation is None:
            return Reply(text=text)
        if isinstance(generation, Reply):
            return generation
        return Reply(
            text=text or generation.text,
            adapter_versions=generation.adapters,
            base_model=generation.base_model,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            tokens_per_second=generation.tokens_per_second,
            meta={
                "finish_reason": generation.finish_reason,
                "time_to_first_token_ms": generation.time_to_first_token_ms,
                "cached_prompt_tokens": generation.meta.get("cached_prompt_tokens", 0),
            },
        )

    # --- the pieces both paths share ------------------------------------
    def _before_model(
        self, session: str | None, message: str
    ) -> tuple[int, CommandResult | None]:
        if self.state.state in (State.STOPPING, State.STOPPED):
            raise RuntimeError("daemon is shutting down")
        self.last_activity_at = self.clock.now()
        session_id = self._touch_session(session)
        return session_id, self.commands.handle(session_id, message)

    def _command_response(self, session_id: int, command: CommandResult) -> dict[str, Any]:
        if command.clears_history:
            self.db.execute(
                "UPDATE sessions SET external_id = external_id || '-' || ?"
                " WHERE id = ?",
                (self.clock.now().strftime("%H%M%S"), session_id),
            )
        self.last_activity_at = self.clock.now()
        return {
            "reply": command.reply,
            "session": session_id,
            "turn_id": None,
            "state": str(self.state.state),
            "sleep_paused": False,
            "command": command.action,
            "adapters": {},
            "latency_ms": 0,
        }

    async def _prepare_to_answer(self) -> bool:
        """A message during sleep pauses it: checkpoint, hand the GPU back, answer."""
        if self.state.state is State.SLEEP:
            return await self._pause_sleep("a user message arrived", wait=True)
        return False

    async def _ensure_loaded(self) -> None:
        if self.responder.loaded:
            return
        await self.lease.acquire(
            "inference", evictor=self._evict_inference, reason="serving", timeout=None
        )
        if not await self._try_load_model("a chat request"):
            raise RuntimeError(
                f"the model is not available, so I cannot answer: {self.load_error}"
            )
        if self.state.state is State.SAFE_MODE:
            self.state.transition(State.WAKE, "the model loaded on retry")

    def _store_text(self, message: str) -> tuple[str, list]:
        """What goes on disk. Secrets are replaced unless the operator opted out."""
        findings = secret_scanner.scan(message)
        if not findings:
            return message, []
        if not self.config.wake.redact_secrets:
            self.events.emit(
                "wake.secret_detected",
                f"{secret_scanner.summarise(findings)} detected; the turn will never be "
                f"trained on (redaction is off, so the text is stored as-is)",
                level="warning",
                kinds=sorted({f.kind for f in findings}),
            )
            return message, findings
        redacted, _ = secret_scanner.redact(message)
        self.events.emit(
            "wake.secret_redacted",
            f"{secret_scanner.summarise(findings)} detected: stored redacted and marked "
            f"never-train",
            level="warning",
            kinds=sorted({f.kind for f in findings}),
        )
        return redacted, findings

    def _implicit_signal(self, session_id: int, message: str):
        previous = self.db.query_one(
            "SELECT content FROM turns WHERE session_id = ? AND role = 'user'"
            " ORDER BY id DESC LIMIT 1",
            (session_id,),
        )
        return detect_implicit(
            message, previous_user=previous["content"] if previous else None
        )

    def _log_exchange(
        self,
        session_id: int,
        message: str,
        stored: str,
        findings: list,
        reply: Reply,
        latency_ms: int,
        signal,
    ) -> int:
        trainable = 0 if findings else 1
        reason = (
            f"secret:{secret_scanner.summarise(findings)}" if findings else None
        )
        user_turn = self._record_turn(
            session_id, "user", stored,
            trainable=trainable, redacted=1 if findings else 0, exclude_reason=reason,
        )
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
            trainable=trainable,
            exclude_reason=reason,
        )
        if trainable:
            self.journal.add_pending_work("turn", ref=str(turn_id))
        if signal is not None and signal.negative:
            # A candidate for review, never training data on its own.
            self.db.insert(
                "feedback",
                {
                    "turn_id": user_turn,
                    "kind": "implicit",
                    "payload": f"{signal.kind}: {signal.reason}",
                    "source": "detector",
                    "reviewed": 0,
                },
            )
            self.events.emit(
                "wake.implicit_feedback",
                f"{signal.kind} signal ({signal.confidence:.0%}): {signal.reason}",
                level="debug",
                turn=user_turn,
            )
        return turn_id

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
    # backups
    # ------------------------------------------------------------------
    def protected_snapshots(self) -> set[str]:
        """Snapshots retention must never delete: those a pending approval refers to."""
        rows = self.db.query(
            "SELECT snapshot_id FROM approvals WHERE status = 'pending'"
            " AND snapshot_id IS NOT NULL"
        )
        protected = {row["snapshot_id"] for row in rows}
        rows = self.db.query(
            "SELECT snapshot_id FROM weight_changes WHERE status = 'in_progress'"
            " AND snapshot_id IS NOT NULL"
        )
        return protected | {row["snapshot_id"] for row in rows}

    def retention_policy(self) -> RetentionPolicy:
        return RetentionPolicy(
            keep_last=self.config.backup.keep_last,
            daily_days=self.config.backup.daily_days,
            weekly_weeks=self.config.backup.weekly_weeks,
        )

    def apply_retention(self, *, dry_run: bool = False) -> dict[str, Any]:
        """Delete what the policy no longer needs. Never the newest, never a pending one."""
        decision = plan_retention(
            self.backups.list(),
            self.retention_policy(),
            protected=self.protected_snapshots(),
            now=self.clock.now(),
        )
        free_mb = self.backups.free_mb()
        if free_mb < self.config.backup.warn_free_mb:
            # Warn rather than deleting beyond the policy: losing backups to free space
            # is the very failure backups exist to prevent.
            self.events.emit(
                "backup.low_space",
                f"the backup location has {free_mb / 1024:.1f} GiB free "
                f"(warning below {self.config.backup.warn_free_mb / 1024:.1f} GiB)",
                level="warning",
                free_mb=free_mb,
            )
        deleted: list[str] = []
        if not dry_run:
            for snapshot in decision.delete:
                try:
                    self.backups.delete(snapshot.id)
                    deleted.append(snapshot.id)
                except BackupError as exc:
                    self.log.warning("could not delete snapshot %s: %s", snapshot.id, exc)
        return {
            "kept": [s.id for s in decision.keep],
            "deleted": deleted if not dry_run else [s.id for s in decision.delete],
            "reasons": decision.reasons,
            "policy": self.retention_policy().describe(),
            "free_mb": free_mb,
            "dry_run": dry_run,
        }

    def list_backups(self) -> dict[str, Any]:
        protected = self.protected_snapshots()
        return {
            "location": str(self.backups.location),
            "free_mb": self.backups.free_mb(),
            "policy": self.retention_policy().describe(),
            "snapshots": [
                {
                    "id": s.id, "operation": s.operation, "created_at": s.created_at,
                    "size_mb": s.size_mb, "file_count": s.file_count,
                    "complete": s.complete, "protected": s.id in protected,
                }
                for s in self.backups.list()
            ],
            "changes": self.guard.history(limit=10),
        }

    def create_backup(self, operation: str = "manual", *, dry_run: bool = False,
                      include_base: bool = False) -> dict[str, Any]:
        if dry_run:
            plan = self.backups.plan(operation, include_base=include_base)
            return {
                "dry_run": True,
                "operation": operation,
                "estimated_mb": plan.estimated_mb,
                "free_mb": plan.free_mb,
                "sources": [{"name": n, "path": str(p), "size_mb": s} for n, p, s in plan.sources],
                "warnings": plan.warnings,
                "lines": plan.describe(),
            }
        snapshot = self.backups.create(operation, include_base=include_base)
        self.db.set_meta(
            "last_backup",
            {
                "id": snapshot.id, "operation": operation, "created_at": snapshot.created_at,
                "size_mb": snapshot.size_mb, "verified": snapshot.verified,
            },
        )
        return {
            "dry_run": False, "id": snapshot.id, "size_mb": snapshot.size_mb,
            "file_count": snapshot.file_count, "verified": snapshot.verified,
            "path": str(snapshot.path),
        }

    # ------------------------------------------------------------------
    # tools (section 11)
    # ------------------------------------------------------------------
    def _tool_store(self):
        from gustav.tools.store import ToolStore

        return ToolStore(
            self.db, self.paths.tools, max_tools=self.config.tools.max_tools
        )

    def tools_status(self, *, limit: int = 20) -> dict[str, Any]:
        """Every tool, what it may do, and whether anyone has read it."""
        store = self._tool_store()
        from gustav.tools.policy import policy_for

        tools = []
        for record in store.all():
            version = None
            with contextlib.suppress(Exception):
                version = store.version(record.name, record.active_version)
            tools.append(
                {
                    "name": record.name,
                    "description": record.description,
                    "status": record.status,
                    "active_version": record.active_version,
                    "versions": record.versions,
                    "capabilities": list(version.capabilities) if version else [],
                    "usage": store.usage(record.name),
                    "updated_at": record.updated_at,
                }
            )
        return {
            "enabled": self.config.tools.enabled,
            "authoring": self.config.tools.allow_authoring,
            "require_approval": self.config.tools.require_approval,
            "allow_capabilities": list(self.config.tools.allow_capabilities),
            "limits": policy_for(["pure"], self.config.tools).describe(),
            "tools": tools,
            "pending": [
                {
                    "name": version.name,
                    "version": version.version,
                    "description": version.description,
                    "capabilities": list(version.capabilities),
                    "created_at": version.created_at,
                }
                for version in store.pending()
            ],
            "requests": store.open_requests(limit=limit),
            "calls": store.calls(limit=limit),
        }

    def tool_detail(self, name: str, version: str | None = None) -> dict[str, Any]:
        """Everything a reviewer needs: the source, the schema, the checks, the usage."""
        store = self._tool_store()
        record = store.record(name)
        if record is None:
            raise KeyError(f"there is no tool called {name!r}")
        chosen = store.version(name, version)
        from gustav.tools.policy import policy_for

        return {
            "name": record.name,
            "description": record.description or chosen.description,
            "status": record.status,
            "active_version": record.active_version,
            "version": chosen.version,
            "version_status": chosen.status,
            "author": chosen.author,
            "created_at": chosen.created_at,
            "reviewed_at": chosen.reviewed_at,
            "reviewed_by": chosen.reviewed_by,
            "note": chosen.note,
            "source": chosen.source,
            "parameters": chosen.parameters,
            "capabilities": list(chosen.capabilities),
            "examples": list(chosen.examples),
            "checks": chosen.checks,
            "limits": policy_for(chosen.capabilities, self.config.tools).describe(),
            "versions": [
                {"version": v.version, "status": v.status, "created_at": v.created_at}
                for v in store.versions(name)
            ],
            "usage": store.usage(name),
            "calls": store.calls(name, limit=10),
            "path": str(chosen.path) if chosen.path else None,
        }

    def review_tool(
        self,
        name: str,
        *,
        approve: bool,
        version: str | None = None,
        by: str = "user",
        note: str = "",
    ) -> dict[str, Any]:
        """Approve or reject a version. The only way a tool becomes callable."""
        store = self._tool_store()
        decided = (
            store.approve(name, version, by=by, note=note)
            if approve
            else store.reject(name, version, by=by, note=note)
        )
        self.events.emit(
            "tool.approved" if approve else "tool.rejected",
            f"{decided.name}@{decided.version} was "
            + ("approved" if approve else "rejected")
            + f" by {by}" + (f": {note}" if note else ""),
            level="warning",
            tool=decided.name,
            version=decided.version,
            by=by,
        )
        return {
            "name": decided.name,
            "version": decided.version,
            "status": decided.status,
            "approved": approve,
            "note": note or None,
        }

    def retire_tool(self, name: str, *, note: str = "") -> dict[str, Any]:
        store = self._tool_store()
        store.retire(name, note=note)
        self.events.emit(
            "tool.retired", f"{name} was retired", level="warning", tool=name
        )
        return {"name": name, "status": "retired"}

    def rollback_tool(self, name: str, *, by: str = "user") -> dict[str, Any]:
        decided = self._tool_store().rollback(name, by=by)
        self.events.emit(
            "tool.rolled_back",
            f"{name} is back to {decided.version}",
            level="warning",
            tool=name,
            version=decided.version,
        )
        return {"name": name, "version": decided.version, "status": decided.status}

    def request_tool(self, text: str, *, source: str = "user") -> dict[str, Any]:
        """Ask for a tool. The night writes it; you review it."""
        if not text.strip():
            raise ValueError("a tool request needs a description of what it should do")
        request_id = self._tool_store().add_request(text, source=source)
        self.journal.add_pending_work("tool_request", weight=5, ref=str(request_id))
        self.events.emit(
            "tool.requested", f"a tool was requested: {text[:160]}", request=request_id
        )
        return {"id": request_id, "text": text.strip(), "status": "open"}

    async def run_tool(
        self, name: str, arguments: dict[str, Any], *, source: str = "cli"
    ) -> dict[str, Any]:
        """Call a tool by hand — which is how you try one out before approving it.

        An unapproved version can be run this way *on purpose*: reading the source and
        then seeing what it does is the review. It is never reachable by the model
        until it has been approved.
        """
        from gustav.tools.policy import policy_for
        from gustav.tools.runner import ToolRunner

        store = self._tool_store()
        version = store.version(name)
        runner = ToolRunner(workspace_root=self.paths.tmp)
        policy = policy_for(version.capabilities, self.config.tools)
        result = await asyncio.to_thread(
            runner.run, version.source, arguments, policy
        )
        store.record_call(
            name,
            version=version.version,
            arguments=arguments,
            result=result.result,
            ok=result.ok,
            error="" if result.ok else result.summary,
            duration_ms=result.duration_ms,
            source=source,
        )
        return {
            "name": name,
            "version": version.version,
            "status": version.status,
            "ok": result.ok,
            "result": result.result,
            "error": None if result.ok else result.summary,
            "detail": result.detail or None,
            "stdout": result.stdout[:2000] or None,
            "duration_ms": result.duration_ms,
        }

    def delete_tool(self, name: str) -> dict[str, Any]:
        removed = self._tool_store().delete(name)
        self.events.emit(
            "tool.deleted",
            f"{name} was deleted" + (f" from {removed}" if removed else ""),
            level="warning",
            tool=name,
        )
        return {"name": name, "deleted": True, "path": str(removed) if removed else None}

    # ------------------------------------------------------------------
    # self-expansion (section 8)
    # ------------------------------------------------------------------
    def _expander(self):
        from gustav.expansion.actions import Expander
        from gustav.inference.adapters import AdapterStore

        return Expander(
            paths=self.paths,
            config=self.config,
            db=self.db,
            events=self.events,
            guard=self.guard,
            adapters=AdapterStore(self.paths.adapters),
            logger=get_logger("expansion"),
        )

    def expansion_status(self, limit: int = 20) -> dict[str, Any]:
        """What it has proposed, done, and been refused — newest first."""
        from gustav.expansion.triggers import measure
        from gustav.inference.adapters import AdapterStore
        from gustav.inference.registry import load_registry

        plan = self.hw.stored_plan()
        registry = load_registry(self.paths.registry)
        triggers = measure(
            self.db, registry, self.config.expansion, adapters=AdapterStore(self.paths.adapters)
        )
        return {
            "enabled": self.config.expansion.enabled,
            "budget": plan.expansion.describe() if plan else [],
            "modules": registry.names,
            "triggers": [
                {"kind": t.kind, "subject": t.subject, "strength": round(t.strength, 3),
                 "rationale": t.rationale}
                for t in triggers
            ],
            "history": self._expander().history(limit),
        }

    async def approve_expansion(self, approval_id: int) -> dict[str, Any]:
        """Accept a proposed structural change and apply it now.

        The apply path is the same one the night uses, guard and all. A base merge
        rewrites the checkpoint the running model was loaded from, so the model is
        unloaded first and reloaded afterwards rather than left pointing at weights
        that no longer exist.
        """
        from gustav.expansion.proposals import Proposal

        row = self.db.query_one(
            "SELECT * FROM approvals WHERE id = ? AND kind = 'expansion'", (approval_id,)
        )
        if row is None:
            raise KeyError(f"there is no pending expansion approval #{approval_id}")
        if row["status"] != "pending":
            raise ValueError(
                f"approval #{approval_id} was already {row['status']}"
            )
        payload = json.loads(row["payload"] or "{}")
        proposal = Proposal(
            kind=payload.get("kind", ""),
            subject=payload.get("subject", ""),
            rationale=row["rationale"] or "",
            evidence=payload.get("evidence") or {},
            cost=payload.get("cost") or {},
            payload=payload.get("payload") or {},
            needs_approval=True,
        )
        log_row = self.db.query_one(
            "SELECT id FROM expansion_log WHERE approval_id = ? ORDER BY id DESC LIMIT 1",
            (approval_id,),
        )
        expander = self._expander()
        heavy = proposal.kind == "merge_to_base"
        if heavy:
            await self._pause_sleep(f"merging {proposal.subject} into the base", wait=True)
            await self.release_gpu(f"merging {proposal.subject} into the base")

        applied = expander.apply(
            proposal,
            log_id=int(log_row["id"]) if log_row else None,
            progress=lambda message: self.log.info("expansion: %s", message),
        )
        self.db.execute(
            "UPDATE approvals SET status = ?, decided_at = datetime('now'), snapshot_id = ?"
            " WHERE id = ?",
            ("approved" if applied.ok else "pending", applied.snapshot_id, approval_id),
        )
        if heavy:
            await self._reload_for_serving()
        if applied.ok:
            # The registry decides routing and which adapters exist, so the serving
            # side has to be told rather than finding out on the next restart.
            await self._reload_registry()
        return {
            "approved": applied.ok,
            "kind": proposal.kind,
            "subject": proposal.subject,
            "detail": applied.detail,
            "snapshot": applied.snapshot_id,
        }

    def reject_expansion(self, approval_id: int, note: str = "") -> dict[str, Any]:
        row = self.db.query_one(
            "SELECT * FROM approvals WHERE id = ? AND kind = 'expansion'", (approval_id,)
        )
        if row is None:
            raise KeyError(f"there is no expansion approval #{approval_id}")
        self.db.execute(
            "UPDATE approvals SET status = 'rejected', decided_at = datetime('now'),"
            " note = ? WHERE id = ?",
            (note or None, approval_id),
        )
        self.db.execute(
            "UPDATE expansion_log SET status = 'rejected', decided_at = datetime('now'),"
            " detail = ? WHERE approval_id = ?",
            (note or "rejected", approval_id),
        )
        self.events.emit(
            "expansion.rejected",
            f"expansion #{approval_id} ({row['subject']}) was rejected",
            approval=approval_id,
        )
        return {"rejected": True, "id": approval_id, "subject": row["subject"]}

    async def revert_expansion(self) -> dict[str, Any]:
        """Undo the most recent applied expansion, from its own snapshot."""
        await self._pause_sleep("reverting an expansion", wait=True)
        result = self._expander().revert_last()
        await self._reload_registry()
        if "base_model" in (result.get("restored") or []):
            await self.release_gpu("the base checkpoint was restored")
            await self._reload_for_serving()
        return result

    async def _reload_registry(self) -> None:
        """Re-read modules/registry.yaml into the live router and adapter store."""
        reload = getattr(self.responder, "reload_registry", None)
        if reload is None:
            return
        try:
            result = reload()
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:  # noqa: BLE001 - a bad registry must not kill the daemon
            self.log.error("could not reload the registry: %s", exc)

    # ------------------------------------------------------------------
    # self-optimisation (section 10)
    # ------------------------------------------------------------------
    @property
    def config_file(self) -> Path:
        """The configuration file this daemon was started from.

        It matters for exactly one thing: knowing which `tunable` values *you* pinned,
        which self-optimisation may never touch. Guessing the filename would mean
        guessing wrong for anyone who runs `ai start -c other.yaml`.
        """
        return self._config_file or (self.paths.root / "config.yaml")

    def _tuner(self):
        from gustav.config.load import pinned_tunables
        from gustav.optimize.tuner import Tuner

        return Tuner(
            config=self.config.optimisation,
            tunable=self.config.tunable,
            paths=self.paths,
            db=self.db,
            events=self.events,
            pinned=pinned_tunables(self.config_file),
            logger=get_logger("optimise"),
        )

    def optimisation_status(self, limit: int = 20) -> dict[str, Any]:
        from gustav.config.load import pinned_tunables, read_learned_tunable
        from gustav.optimize.router_fit import RouterModel

        tuner = self._tuner()
        router = RouterModel.load(self.paths.data_dir / "router")
        return {
            "enabled": self.config.optimisation.enabled,
            "settings": self.config.optimisation.settings,
            "pinned": sorted(pinned_tunables(self.config_file)),
            "in_use": {
                setting: getattr(self.config.tunable, setting, None)
                for setting in self.config.optimisation.settings
            },
            "learned": read_learned_tunable(self.paths.data_dir),
            "next_trials": [
                {"setting": setting, "candidate": repr(value)}
                for setting, value in tuner.candidates()
            ],
            "history": tuner.history(limit),
            "router": (
                {
                    "trained_at": router.trained_at,
                    "threshold": router.threshold,
                    "agreement": router.agreement,
                    "modules": len(router.modules),
                    "material": sum(router.documents.values()),
                }
                if router is not None
                else None
            ),
        }

    async def revert_optimisation(self) -> dict[str, Any]:
        """Undo the last kept tunable change and reload with the old value."""
        result = self._tuner().revert_last()
        # Re-read the configuration so the running daemon uses what is now on disk.
        reloaded, _paths = load_config(self.config_file)
        self.config.tunable = reloaded.tunable
        await self.release_gpu(f"reverting {result['setting']}")
        await self._reload_for_serving()
        return result

    # ------------------------------------------------------------------
    # what it learned
    # ------------------------------------------------------------------
    def facts(self, *, module: str | None = None, limit: int = 100,
              subject: str | None = None) -> dict[str, Any]:
        """Durable statements the night extracted, newest and most confident first."""
        from gustav.learn.facts import FactStore

        store = FactStore(self.db)
        if subject:
            return {"subject": subject, "history": store.history(subject)}
        return {
            "facts": store.active(module=module, limit=limit),
            "total": store.count(),
            "retracted": store.count("retracted"),
        }

    def skills(self) -> dict[str, Any]:
        """The measured skill table: what it can do, how often it was asked."""
        from gustav.learn.skills import SkillStore

        store = SkillStore(self.db)
        return {
            "skills": [
                {
                    "module": s.module, "topic": s.topic, "trials": s.trials,
                    "passes": s.passes, "score": round(s.score, 3),
                    "demand": round(s.demand, 3), "weakness": round(s.weakness, 3),
                    "studied_at": s.studied_at,
                }
                for s in sorted(store.all(), key=lambda s: (-s.weakness, s.module, s.topic))
            ],
            "summary": store.summary(),
        }

    def write_report(self, run_id: int | None = None) -> dict[str, Any]:
        """Render a run's report. Regenerated from the store, so any past run works."""
        from gustav.report.html import write_report as render

        if run_id is None:
            row = self.db.query_one("SELECT id FROM sleep_runs ORDER BY id DESC LIMIT 1")
            if row is None:
                raise KeyError("there are no sleep runs yet, so there is nothing to report")
            run_id = int(row["id"])
        path = render(self.db, run_id, self.paths.reports, instance=self.config.instance.name)
        self.db.set_meta("sleep.last_report", str(path))
        return {"run_id": run_id, "path": str(path), "bytes": path.stat().st_size}

    def verify_backup(self, snapshot_id: str) -> dict[str, Any]:
        result = self.backups.verify(snapshot_id, check_extra=True)
        return {
            "id": result.snapshot.id,
            "ok": result.ok,
            "file_count": result.snapshot.file_count,
            "size_mb": result.snapshot.size_mb,
            "problems": [{"path": p.path, "kind": p.kind, "detail": p.detail}
                         for p in result.problems],
            "lines": result.describe(),
        }

    async def restore_backup(
        self,
        snapshot_id: str,
        *,
        only: tuple[str, ...] = (),
        module: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Restore, having first snapshotted the current state so this is undoable."""
        restorer = Restorer(self.backups, events=self.events)
        if not dry_run:
            # The model holds adapters open; put it down before swapping files underneath.
            await self.release_gpu(f"restoring snapshot {snapshot_id}")
        result = restorer.restore(
            snapshot_id, only=only, module=module, dry_run=dry_run
        )
        payload = {
            "snapshot": result.snapshot_id,
            "restored": result.restored,
            "skipped": result.skipped,
            "undo_snapshot": result.undo_snapshot_id,
            "dry_run": result.dry_run,
            "lines": result.describe(),
        }
        if not dry_run:
            # The store on disk changed under us; re-read it.
            self.db.set_meta("last_restore", payload)
        return payload

    def approve_base_change(self, approval_id: int) -> dict[str, Any]:
        """Accept a base-model change — guarded, because it invalidates every adapter.

        The base checkpoint itself is included in the snapshot: this is one of the two
        operations section 9 names for that, since it is the one that could lose it.
        """
        with self.guard.change(
            f"base-model change (approval {approval_id})",
            sources=("adapters", "registry", "database", "profiles"),
            include_base=self.config.backup.include_base_model,
            meta={"approval": approval_id},
        ) as change:
            result = self.hw.approve_base_change(approval_id)
            if result.get("approved"):
                self.db.execute(
                    "UPDATE approvals SET snapshot_id = ? WHERE id = ?",
                    (change.snapshot_id, approval_id),
                )
                change.promoted.append(f"base model -> {result['plan']['base_model']}")
        result["snapshot"] = change.snapshot_id
        return result

    # ------------------------------------------------------------------
    # benchmark
    # ------------------------------------------------------------------
    async def benchmark(self, *, max_tokens: int = 96, repeats: int = 1) -> dict[str, Any]:
        """Measure this machine, and let the planner use measurements from now on."""
        from gustav.hw.speed import Measurements
        from gustav.inference.bench import run_benchmark

        if not hasattr(self.responder, "backend"):
            raise RuntimeError("this responder cannot be benchmarked (no model behind it)")
        await self._ensure_loaded()
        async with self._chat_lock:
            result = await run_benchmark(
                self.responder,
                max_tokens=max_tokens,
                repeats=repeats,
                measurements=Measurements(self.db),
            )
        self.events.emit("bench.completed", result.summary(), **{
            "tokens_per_second": result.tokens_per_second,
            "vram_peak_mb": result.vram_peak_mb,
        })
        # The plan's speed numbers were estimates until now.
        self._plan_hardware(reason="a benchmark measured this machine", force=True)
        return result.to_dict()

    # ------------------------------------------------------------------
    # model weights
    # ------------------------------------------------------------------
    def list_models(self) -> dict[str, Any]:
        from gustav.inference.models import list_models

        ladder = self.hw._ladder_with_local_configs()
        local = list_models(self.paths.models, ladder)
        plan = self.hw.stored_plan()
        return {
            "models_dir": str(self.paths.models),
            "models": [
                {
                    "id": m.id, "size_mb": m.size_mb, "complete": m.complete,
                    "missing": m.missing, "repo": m.repo, "params_b": m.params_b,
                    "active": bool(plan and plan.base_model == m.id),
                }
                for m in local
            ],
            "ladder": [
                {"id": s.id, "repo": s.repo, "params_b": s.params_b, "rung": s.rung}
                for s in ladder.rungs()
            ],
            "plan_model": plan.base_model if plan else None,
        }

    async def pull_model(self, model_id: str) -> dict[str, Any]:
        """Fetch weights in a supervised child process. The only network access here."""
        ladder = self.hw._ladder_with_local_configs()
        try:
            spec = ladder.model(model_id)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        destination = self.paths.models / spec.id
        self.events.emit(
            "model.pull.started", f"fetching {spec.repo} into {destination}", model=spec.id
        )
        result = await self.supervisor.run(
            ChildSpec(
                name=f"pull:{spec.id}",
                target="gustav.inference.models:pull_child",
                kwargs={"repo": spec.repo, "destination": str(destination)},
                timeout_seconds=7200.0,
                heartbeat_timeout_seconds=300.0,
                memory_limit_mb=0,
            )
        )
        if not result.ok:
            self.events.emit(
                "model.pull.failed",
                f"could not fetch {spec.repo}: {result.error}",
                level="error",
                model=spec.id,
            )
            return {"ok": False, "model": spec.id, "error": result.error,
                    "outcome": str(result.outcome)}
        payload = result.payload or {}
        self.events.emit(
            "model.pull.done",
            f"{spec.repo} fetched ({payload.get('size_mb', 0) / 1024:.2f} GiB)",
            model=spec.id,
        )
        # A newly downloaded model makes the plan's numbers exact, so re-plan.
        self._plan_hardware(reason="new weights on disk", force=True)
        return {"ok": True, "model": spec.id, **payload}

    # ------------------------------------------------------------------
    # adapter control (used by /adapter/*)
    # ------------------------------------------------------------------
    def adapter_status(self) -> dict[str, Any]:
        getter = getattr(self.responder, "adapter_status", None)
        if callable(getter):
            return getter()
        return {"modules": {}, "active_in_model": {}, "note": "this responder has no adapters"}

    async def rollback_adapter(self, module: str) -> dict[str, Any]:
        """Go back a version. Guarded: a verified snapshot exists before anything moves."""
        rollback = getattr(self.responder, "rollback_adapter", None)
        if not callable(rollback):
            raise RuntimeError("this responder has no adapters to roll back")
        with self.guard.change(
            f"rollback {module}", sources=("adapters",), meta={"module": module}
        ) as change:
            version = await rollback(module)
            change.promoted.append(f"{module} -> {version.version}")
        return {
            "module": module, "version": version.version, "snapshot": change.snapshot_id,
        }

    async def activate_adapter(self, module: str, version: str) -> dict[str, Any]:
        """Hot-swap a version in. Guarded, like every other change to weights."""
        activate = getattr(self.responder, "activate_adapter", None)
        if not callable(activate):
            raise RuntimeError("this responder has no adapters to activate")
        with self.guard.change(
            f"activate {module}@{version}",
            sources=("adapters",),
            meta={"module": module, "version": version},
        ) as change:
            target = await activate(module, version)
            change.promoted.append(f"{module} -> {target.version}")
        return {
            "module": module, "version": target.version, "snapshot": change.snapshot_id,
        }

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
            # The night may have changed the registry (a new module, a raised rank, one
            # archived), and the serving router holds a parsed copy of it.
            await self._reload_registry()
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
        """One timer, several jobs, each on its own interval.

        The tick itself is deliberately cheap: a 30-second sleep and a couple of
        comparisons. Anything that costs real work has its own, much longer interval.
        """
        tick = 30.0
        health_every = max(self.config.maintenance.health_check_minutes * 60, tick)
        recheck_every = max(self.config.maintenance.hardware_recheck_minutes * 60, health_every)
        retention_every = max(self.config.maintenance.backup_retention_minutes * 60, health_every)
        last_health = last_hardware = last_retention = self.clock.monotonic()
        while True:
            try:
                await asyncio.sleep(tick)
                now = self.clock.monotonic()

                await self._unload_if_idle()

                if now - last_health >= health_every:
                    last_health = now
                    ok, message = self.responder.health()
                    self.telemetry.record("daemon.alive", 1.0, state=str(self.state.state))
                    if not ok and self.state.state is State.WAKE:
                        self.events.emit("daemon.unhealthy", message, level="error")
                        self.state.transition(State.SAFE_MODE, message)

                if now - last_hardware >= recheck_every:
                    last_hardware = now
                    self._plan_hardware(reason="periodic re-check", force=True)

                if now - last_retention >= retention_every:
                    last_retention = now
                    self.apply_retention()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                self.log.exception("maintenance tick failed")

    async def _unload_if_idle(self) -> None:
        """Give the VRAM back when nobody is talking to us.

        The model is several gigabytes; holding it while you are doing something else
        with the GPU is rude. The next message reloads it, which costs a second or two.
        """
        minutes = self.config.wake.idle_unload_minutes
        if not minutes or not self.responder.loaded:
            return
        if self.state.state is not State.WAKE:
            return
        idle_seconds = (self.clock.now() - self.last_activity_at).total_seconds()
        if idle_seconds < minutes * 60:
            return
        async with self._chat_lock:
            if not self.responder.loaded:
                return
            await self.responder.unload()
        self.events.emit(
            "inference.idle_unload",
            f"released the GPU after {idle_seconds / 60:.0f} min idle; "
            f"the next message reloads the model",
            idle_minutes=round(idle_seconds / 60, 1),
        )

    async def release_gpu(self, reason: str = "requested") -> dict[str, Any]:
        """Unload the model now. Chat still works; it reloads on the next message."""
        was_loaded = self.responder.loaded
        paused = False
        if self.state.state is State.SLEEP:
            paused = await self._pause_sleep(f"GPU release: {reason}", wait=True)
        async with self._chat_lock:
            if self.responder.loaded:
                await self.responder.unload()
        if was_loaded or paused:
            self.events.emit("inference.released", f"released the GPU ({reason})")
        return {
            "released": was_loaded,
            "sleep_paused": paused,
            "note": "the next message reloads the model",
        }

    def _observe(self, *, manual: str | None = None) -> Observation:
        last_sleep = self.db.get_meta(META_LAST_SLEEP)
        resumable = any(r.next_step is not None for r in self.journal.open_runs())
        foreign_mb, utilisation = self._gpu_pressure_quiet()
        return Observation(
            state=self.state.state,
            now=self.clock.now(),
            last_activity_at=self.last_activity_at,
            pending_items=self.journal.pending_work_count(),
            learning_paused=self.learning_paused,
            last_sleep_finished_at=_parse_dt(last_sleep),
            manual_trigger=manual,
            gpu_temp_c=self._gpu_temperature(),
            foreign_vram_mb=foreign_mb,
            gpu_utilization_pct=utilisation,
            resumable_run=resumable,
        )

    def _gpu_pressure_quiet(self) -> tuple[int, float | None]:
        """What else is using the GPU. Never raises; the scheduler only needs a hint.

        Our own inference process is excluded, but a *loaded* model of ours still
        occupies VRAM, so the reading is taken as "someone else's" usage only.
        """
        try:
            return self.hw.gpu_pressure()
        except Exception:  # noqa: BLE001
            return 0, None

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
            "load_error": self.load_error,
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


async def run_daemon(
    config: Config, paths: Paths, *, config_file: str | None = None
) -> int:
    daemon = Daemon(config, paths, config_file=Path(config_file) if config_file else None)
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
        return asyncio.run(run_daemon(config, paths, config_file=args.config))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
