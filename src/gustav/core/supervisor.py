"""Supervised child processes.

Every heavy or risky job (training, sandbox runs, doc fetching) runs in a separate
process, never a thread. A CUDA OOM, a segfault, or a hang in a child must never take
the chat down, so the parent only ever observes the child through a pipe, a heartbeat
deadline, and a wall-clock timeout — and escalates SIGTERM -> SIGKILL when either
deadline passes.

Children are started with the ``spawn`` context (no forked CUDA state) and the target
is given as an importable ``"module:function"`` string.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import resource
import signal
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import import_module
from typing import Any

#: Exit code a child uses when it stopped cleanly because it was asked to.
EXIT_PREEMPTED = 17

_OOM_MARKERS = ("MemoryError", "OutOfMemoryError", "CUDA out of memory", "Cannot allocate memory")


class Outcome(StrEnum):
    OK = "ok"
    FAILED = "failed"
    TIMEOUT = "timeout"
    HANG = "hang"
    OOM = "oom"
    CRASHED = "crashed"
    PREEMPTED = "preempted"

    @property
    def retryable(self) -> bool:
        """Worth another attempt with the same settings.

        ``OOM`` deliberately is not: re-running an identical job that just ran out of
        memory wastes the night. A step that can shrink its own footprint (lower
        sequence length, smaller batch) retries internally and reports ``FAILED``.
        """
        return self in {Outcome.FAILED, Outcome.TIMEOUT, Outcome.HANG, Outcome.CRASHED}


@dataclass(frozen=True)
class ChildSpec:
    name: str
    target: str  # "package.module:function"
    kwargs: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 1800.0
    heartbeat_timeout_seconds: float = 60.0
    memory_limit_mb: int = 0  # 0 = no RLIMIT_AS
    cpu_seconds: int = 0  # 0 = no RLIMIT_CPU
    nice: int = 5
    #: After a stop request, how long the child may take to checkpoint and exit
    #: before it is terminated anyway. A sleep step must give the GPU back promptly.
    stop_grace_seconds: float = 20.0
    #: Grace between SIGTERM and SIGKILL once we have decided the child must die.
    kill_grace_seconds: float = 5.0
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ChildResult:
    outcome: Outcome
    payload: dict[str, Any] | None = None
    error: str | None = None
    exit_code: int | None = None
    duration_seconds: float = 0.0
    heartbeats: int = 0
    checkpoint: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.outcome is Outcome.OK


# ---------------------------------------------------------------------------
# Child side
# ---------------------------------------------------------------------------

_stop_requested = threading.Event()


def _on_stop_signal(signum: int, frame: object) -> None:  # pragma: no cover - child side
    _stop_requested.set()


class ChildContext:
    """What a child's target function uses to talk back to the daemon."""

    def __init__(self, conn: Any, name: str):
        self._conn = conn
        self.name = name

    def _send(self, message: dict[str, Any]) -> None:
        try:
            self._conn.send(message)
        except (BrokenPipeError, OSError):  # parent went away; keep going quietly
            pass

    def heartbeat(self, progress: str | None = None, **data: Any) -> None:
        """Prove liveness. A child that stops doing this is killed by the watchdog."""
        self._send({"type": "heartbeat", "progress": progress, "data": data or None})

    def checkpoint(self, data: dict[str, Any]) -> None:
        """Record resumable progress. Written to the journal by the parent immediately."""
        self._send({"type": "checkpoint", "data": data})

    def log(self, message: str, level: str = "info") -> None:
        self._send({"type": "log", "level": level, "message": message})

    def should_stop(self) -> bool:
        """True once the daemon has asked this child to wind up (user message, shutdown)."""
        return _stop_requested.is_set()

    def raise_if_stopping(self) -> None:
        if self.should_stop():
            raise Preempted(f"{self.name} asked to stop")


class Preempted(RuntimeError):
    """Raised inside a child that has been asked to stop at a safe point."""


def _apply_limits(spec_limits: dict[str, int]) -> None:  # pragma: no cover - child side
    mem_mb = spec_limits.get("memory_limit_mb", 0)
    if mem_mb:
        limit = mem_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ValueError, OSError):
            pass
    cpu_s = spec_limits.get("cpu_seconds", 0)
    if cpu_s:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
        except (ValueError, OSError):
            pass
    nice = spec_limits.get("nice", 0)
    if nice:
        try:
            os.nice(nice)
        except OSError:
            pass


def _child_main(conn: Any, payload: dict[str, Any]) -> None:  # pragma: no cover - child side
    """Entry point in the spawned process."""
    name = payload["name"]
    signal.signal(signal.SIGUSR1, _on_stop_signal)
    signal.signal(signal.SIGTERM, _on_stop_signal)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for key, value in (payload.get("env") or {}).items():
        os.environ[key] = value
    _apply_limits(payload.get("limits") or {})

    ctx = ChildContext(conn, name)
    code = 0
    try:
        module_name, _, func_name = payload["target"].partition(":")
        if not func_name:
            raise ValueError(f"target {payload['target']!r} must be 'module:function'")
        func: Callable[..., Any] = getattr(import_module(module_name), func_name)
        ctx.heartbeat("started")
        result = func(ctx, **(payload.get("kwargs") or {}))
        if not isinstance(result, (dict, type(None))):
            result = {"value": result}
        ctx._send({"type": "result", "payload": result})
    except Preempted as exc:
        ctx._send({"type": "preempted", "error": str(exc)})
        code = EXIT_PREEMPTED
    except BaseException as exc:  # noqa: BLE001 - report everything upward
        ctx._send(
            {
                "type": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=12),
            }
        )
        code = 1
    finally:
        try:
            conn.close()
        except OSError:
            pass
    os._exit(code)


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


class Supervisor:
    """Runs one child at a time per call, with a watchdog."""

    def __init__(self, *, logger=None, poll_interval: float = 0.1):
        self.log = logger
        self.poll_interval = poll_interval
        self._mp = mp.get_context("spawn")

    def _debug(self, message: str) -> None:
        if self.log:
            self.log.debug(message)

    async def run(
        self,
        spec: ChildSpec,
        *,
        stop_event: Any | None = None,
        on_message: Callable[[dict[str, Any]], None] | None = None,
    ) -> ChildResult:
        """Run ``spec`` to completion and return what happened.

        ``stop_event`` is an ``asyncio.Event``/``threading.Event``-like object; when it
        is set the child gets SIGUSR1 and is expected to checkpoint and exit cleanly.
        """
        import asyncio

        parent_conn, child_conn = self._mp.Pipe(duplex=False)
        payload = {
            "name": spec.name,
            "target": spec.target,
            "kwargs": spec.kwargs,
            "env": spec.env,
            "limits": {
                "memory_limit_mb": spec.memory_limit_mb,
                "cpu_seconds": spec.cpu_seconds,
                "nice": spec.nice,
            },
        }
        proc = self._mp.Process(
            target=_child_main, args=(child_conn, payload), name=f"gustav-{spec.name}", daemon=False
        )
        started = time.monotonic()
        proc.start()
        child_conn.close()  # only the child writes

        messages: queue.Queue[dict[str, Any]] = queue.Queue()
        reader = threading.Thread(
            target=_drain, args=(parent_conn, messages), name=f"reader-{spec.name}", daemon=True
        )
        reader.start()

        result = ChildResult(outcome=Outcome.FAILED)
        heartbeats = 0
        last_beat = started
        signalled = False
        stop_deadline: float | None = None
        traceback_text: str | None = None

        try:
            while True:
                drained = False
                while True:
                    try:
                        msg = messages.get_nowait()
                    except queue.Empty:
                        break
                    drained = True
                    kind = msg.get("type")
                    if kind == "heartbeat":
                        heartbeats += 1
                        last_beat = time.monotonic()
                    elif kind == "checkpoint":
                        last_beat = time.monotonic()
                        result.checkpoint = msg.get("data")
                    elif kind == "result":
                        result.outcome = Outcome.OK
                        result.payload = msg.get("payload")
                    elif kind == "preempted":
                        result.outcome = Outcome.PREEMPTED
                        result.error = msg.get("error")
                    elif kind == "error":
                        result.outcome = Outcome.FAILED
                        result.error = msg.get("error")
                        traceback_text = msg.get("traceback")
                    elif kind == "eof":
                        pass
                    if on_message:
                        on_message(msg)

                if not proc.is_alive() and messages.empty() and not drained:
                    break

                now = time.monotonic()
                # Only signal once the child has spoken: a SIGUSR1 that arrives
                # before it installs its handler would kill it outright.
                if (
                    stop_event is not None
                    and not signalled
                    and heartbeats >= 1
                    and _is_set(stop_event)
                ):
                    self._debug(f"asking child {spec.name} to stop")
                    _signal(proc, signal.SIGUSR1)
                    signalled = True
                    last_beat = now
                    stop_deadline = now + spec.stop_grace_seconds

                if stop_deadline is not None and now > stop_deadline:
                    # It was asked to stop and did not. Take the GPU back by force;
                    # its last checkpoint is already in the journal.
                    self._debug(f"child {spec.name} ignored the stop request; killing")
                    result.outcome = Outcome.PREEMPTED
                    result.error = (
                        f"did not stop within {spec.stop_grace_seconds:.0f}s of being asked; killed"
                    )
                    await self._terminate(proc, spec.kill_grace_seconds)
                    break

                if spec.timeout_seconds and now - started > spec.timeout_seconds:
                    self._debug(f"child {spec.name} exceeded {spec.timeout_seconds}s")
                    result.outcome = Outcome.TIMEOUT
                    result.error = f"timed out after {spec.timeout_seconds:.0f}s"
                    await self._terminate(proc, spec.kill_grace_seconds)
                    break

                if (
                    spec.heartbeat_timeout_seconds
                    and now - last_beat > spec.heartbeat_timeout_seconds
                ):
                    self._debug(f"child {spec.name} missed its heartbeat")
                    result.outcome = Outcome.HANG
                    result.error = (
                        f"no heartbeat for {spec.heartbeat_timeout_seconds:.0f}s"
                    )
                    await self._terminate(proc, spec.kill_grace_seconds)
                    break

                await asyncio.sleep(self.poll_interval)
        finally:
            if proc.is_alive():
                await self._terminate(proc, spec.kill_grace_seconds)
            proc.join(timeout=5)
            try:
                parent_conn.close()
            except OSError:
                pass

        result.exit_code = proc.exitcode
        result.duration_seconds = time.monotonic() - started
        result.heartbeats = heartbeats

        # Classify what the exit code tells us that the pipe did not.
        if result.outcome is Outcome.FAILED and result.error is None:
            if proc.exitcode == EXIT_PREEMPTED:
                result.outcome = Outcome.PREEMPTED
                result.error = "child stopped on request"
            elif proc.exitcode == 0:
                result.outcome = Outcome.OK
            elif proc.exitcode is not None and proc.exitcode < 0:
                sig = -proc.exitcode
                if signalled:
                    result.outcome = Outcome.PREEMPTED
                    result.error = f"died while stopping (signal {sig})"
                else:
                    # SIGKILL could be the OOM killer or anything else external; we
                    # only call it OOM when the child itself reported a memory error.
                    result.outcome = Outcome.CRASHED
                    result.error = f"killed by signal {sig} ({signal.Signals(sig).name})"
            else:
                result.outcome = Outcome.CRASHED
                result.error = f"exited with code {proc.exitcode}"

        if result.error and any(m in result.error for m in _OOM_MARKERS):
            result.outcome = Outcome.OOM
        if traceback_text and result.error:
            result.error = f"{result.error}\n{traceback_text}"
        return result

    async def _terminate(self, proc: Any, grace: float = 5.0) -> None:
        import asyncio

        if not proc.is_alive():
            return
        proc.terminate()
        deadline = time.monotonic() + max(grace, 0.0)
        while proc.is_alive() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if proc.is_alive():
            self._debug(f"SIGKILL for pid {proc.pid}")
            proc.kill()
            deadline = time.monotonic() + 5.0
            while proc.is_alive() and time.monotonic() < deadline:
                await asyncio.sleep(0.05)


def _drain(conn: Any, sink: "queue.Queue[dict[str, Any]]") -> None:
    """Read framed messages off the pipe until the child closes it."""
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        except Exception:  # noqa: BLE001 - unpickling garbage must not kill the reader
            break
        if isinstance(msg, dict):
            sink.put(msg)
    sink.put({"type": "eof"})


def _is_set(event: Any) -> bool:
    try:
        return bool(event.is_set())
    except AttributeError:
        return bool(event)


def _signal(proc: Any, sig: int) -> None:
    pid = proc.pid
    if not pid or not proc.is_alive():
        return
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass
