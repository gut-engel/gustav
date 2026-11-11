"""Stage-1 placeholder step handler.

Runs in a child process and does simulated, checkpointed work so the daemon's hard
parts — resume, retry, watchdog, preemption, crash recovery — can be exercised for
real before stage 5 replaces these bodies with consolidation, study, practice and
training.

Fault injection for tests: set ``GUSTAV_FAULT`` to a comma-separated list of
``step=mode`` pairs, optionally suffixed ``:once`` so only the first attempt is
affected. Modes:

``fail``   raise an exception                     ``crash``  die without a word (SIGKILL)
``hang``   stop heartbeating but stay alive       ``oom``    allocate until it dies
``pause``  heartbeat forever until asked to stop  ``slow``   take 10x as long
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Any

from gustav.core.supervisor import ChildContext

#: Simulated work units per step; the unit index is the checkpoint.
UNITS = 4


def _faults() -> dict[str, str]:
    raw = os.environ.get("GUSTAV_FAULT", "")
    out: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        step, _, mode = item.partition("=")
        out[step.strip()] = mode.strip()
    return out


def _fault_for(step: str, run_dir: Path | None, attempt: int) -> str | None:
    mode = _faults().get(step)
    if not mode:
        return None
    once = mode.endswith(":once")
    mode = mode.removesuffix(":once")
    if once and attempt > 1:
        return None
    return mode


def _apply_fault(ctx: ChildContext, mode: str) -> None:
    ctx.log(f"fault injection: {mode}", level="warning")
    if mode == "fail":
        raise RuntimeError(f"injected failure ({mode})")
    if mode == "crash":
        os.kill(os.getpid(), signal.SIGKILL)
    if mode == "hang":
        while True:  # no heartbeat: the watchdog must notice
            time.sleep(0.2)
    if mode == "oom":
        blocks = []
        while True:
            blocks.append(bytearray(64 * 1024 * 1024))
    if mode == "pause":
        while not ctx.should_stop():
            ctx.heartbeat("paused by fault injection")
            time.sleep(0.2)
        ctx.raise_if_stopping()


def run_step(
    ctx: ChildContext,
    *,
    run_id: int,
    step: str,
    run_dir: str | None,
    checkpoint: dict[str, Any] | None = None,
    attempt: int = 1,
    dry_run: bool = False,
    simulated_seconds: float = 0.5,
    **_ignored: Any,
) -> dict[str, Any]:
    directory = Path(run_dir) if run_dir else None
    if directory is not None:
        (directory / step).mkdir(parents=True, exist_ok=True)

    fault = _fault_for(step, directory, attempt)
    if fault in {"crash", "fail", "hang", "oom"}:
        _apply_fault(ctx, fault)

    start_unit = int((checkpoint or {}).get("unit", 0))
    per_unit = max(simulated_seconds, 0.0) / UNITS
    if fault == "slow":
        per_unit *= 10

    for unit in range(start_unit, UNITS):
        if fault == "pause" and unit == UNITS // 2:
            _apply_fault(ctx, "pause")
        ctx.heartbeat(f"{step} {unit + 1}/{UNITS}", unit=unit + 1)
        if per_unit:
            time.sleep(per_unit)
        if directory is not None and not dry_run:
            (directory / step / f"unit-{unit:02d}.done").write_text("ok\n", encoding="utf-8")
        # Checkpoint AFTER the work, so a crash re-does at most one unit.
        ctx.checkpoint({"unit": unit + 1, "step": step})
        ctx.raise_if_stopping()

    return {
        "step": step,
        "units": UNITS,
        "resumed_from": start_unit,
        "dry_run": dry_run,
        "placeholder": True,
    }
