"""Child-process targets used to exercise the supervisor and watchdog."""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Any

from gustav.core.supervisor import ChildContext


def ok_child(ctx: ChildContext, *, beats: int = 3, value: int = 1) -> dict[str, Any]:
    for i in range(beats):
        ctx.heartbeat(f"beat {i + 1}/{beats}")
        time.sleep(0.01)
    return {"value": value, "beats": beats, "pid": os.getpid()}


def failing_child(ctx: ChildContext, *, message: str = "boom") -> dict[str, Any]:
    ctx.heartbeat("about to fail")
    raise RuntimeError(message)


def hanging_child(ctx: ChildContext) -> dict[str, Any]:
    """Alive but silent: the heartbeat watchdog must notice and kill it."""
    ctx.heartbeat("last words")
    while True:
        time.sleep(0.2)


def busy_child(ctx: ChildContext, *, seconds: float = 30.0) -> dict[str, Any]:
    """Heart-beats dutifully but never finishes: the wall-clock timeout must fire."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        ctx.heartbeat("working")
        time.sleep(0.05)
    return {"done": True}


def oom_child(ctx: ChildContext) -> dict[str, Any]:
    """Allocates until the memory limit kills it."""
    ctx.heartbeat("allocating")
    blocks = []
    while True:
        blocks.append(bytearray(32 * 1024 * 1024))
        ctx.heartbeat(f"allocated {len(blocks) * 32} MiB")


def crashing_child(ctx: ChildContext) -> dict[str, Any]:
    """Dies the way a segfault or the OOM killer would: no message, no cleanup."""
    ctx.heartbeat("goodbye")
    os.kill(os.getpid(), signal.SIGKILL)
    return {}


def checkpointing_child(
    ctx: ChildContext,
    *,
    units: int = 4,
    start: int = 0,
    marker_dir: str | None = None,
    stall: bool = False,
) -> dict[str, Any]:
    """Does unit work, checkpoints after each, and stops cleanly when asked."""
    done = start
    for unit in range(start, units):
        ctx.heartbeat(f"unit {unit + 1}/{units}")
        if marker_dir:
            Path(marker_dir).mkdir(parents=True, exist_ok=True)
            (Path(marker_dir) / f"unit-{unit}").write_text("done\n", encoding="utf-8")
        ctx.checkpoint({"unit": unit + 1})
        done = unit + 1
        if stall:
            while not ctx.should_stop():
                ctx.heartbeat("stalling")
                time.sleep(0.05)
        ctx.raise_if_stopping()
        time.sleep(0.01)
    return {"units": done}


def stoppable_child(ctx: ChildContext, *, checkpoint_every: float = 0.05) -> dict[str, Any]:
    """Runs until asked to stop, then checkpoints and exits cleanly (never half-written)."""
    ticks = 0
    while not ctx.should_stop():
        ticks += 1
        ctx.heartbeat(f"tick {ticks}")
        ctx.checkpoint({"ticks": ticks})
        time.sleep(checkpoint_every)
    ctx.checkpoint({"ticks": ticks, "final": True})
    ctx.raise_if_stopping()
    return {"ticks": ticks}


def ignoring_child(ctx: ChildContext) -> dict[str, Any]:
    """Refuses to stop: heart-beats, ignores the stop request. Must be killed."""
    while True:
        ctx.heartbeat("not listening")
        time.sleep(0.05)
