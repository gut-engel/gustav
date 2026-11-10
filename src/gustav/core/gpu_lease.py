"""Exclusive GPU ownership.

Inference and training must never share a single GPU, so exactly one holder owns the
lease at a time. Two ways the lease changes hands:

* The holder registered an **evictor** (inference does: unload the model, free VRAM).
  A new claimant triggers it and the holder releases.
* The holder did not (a training child does not). The claimant sets the return request
  and waits; the daemon turns that into a stop signal, the child checkpoints at a step
  boundary and exits, and the step releases the lease.

With separate inference and training GPUs the daemon simply creates two leases and
nobody ever has to wait.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

Evictor = Callable[[], Awaitable[None] | None]


class GpuBusy(TimeoutError):
    def __init__(self, holder: str | None, timeout: float):
        super().__init__(
            f"GPU is held by {holder!r} and was not released within {timeout:.0f}s"
        )
        self.holder = holder


@dataclass(frozen=True)
class LeaseInfo:
    holder: str | None
    reason: str
    held_for: float
    return_requested: bool
    waiters: int


class GpuLease:
    def __init__(self, name: str = "gpu0", *, logger=None):
        self.name = name
        self.log = logger
        self._cond = asyncio.Condition()
        self._holder: str | None = None
        self._reason: str = ""
        self._since: float = 0.0
        self._evictor: Evictor | None = None
        self._return_requested = False
        self._waiters = 0

    # --- observation ----------------------------------------------------
    @property
    def holder(self) -> str | None:
        return self._holder

    @property
    def return_requested(self) -> bool:
        """True when somebody else needs the GPU and the holder should wind up."""
        return self._return_requested

    def info(self) -> LeaseInfo:
        return LeaseInfo(
            holder=self._holder,
            reason=self._reason,
            held_for=(time.monotonic() - self._since) if self._holder else 0.0,
            return_requested=self._return_requested,
            waiters=self._waiters,
        )

    def is_held_by(self, holder: str) -> bool:
        return self._holder == holder

    # --- acquire / release ----------------------------------------------
    async def acquire(
        self,
        holder: str,
        *,
        evictor: Evictor | None = None,
        reason: str = "",
        timeout: float | None = 300.0,
    ) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            evict_now: Evictor | None = None
            async with self._cond:
                if self._holder is None:
                    self._take(holder, evictor, reason)
                    return
                if self._holder == holder:
                    return  # re-entrant for the same holder
                self._return_requested = True
                if self._evictor is not None:
                    evict_now, self._evictor = self._evictor, None
                self._waiters += 1

            try:
                if evict_now is not None:
                    self._debug(f"evicting {self._holder!r} for {holder!r}")
                    outcome = evict_now()
                    if inspect.isawaitable(outcome):
                        await outcome
                remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
                async with self._cond:
                    if self._holder is None:
                        self._take(holder, evictor, reason)
                        return
                    try:
                        await asyncio.wait_for(
                            self._cond.wait_for(lambda: self._holder is None), remaining
                        )
                    except (TimeoutError, asyncio.TimeoutError) as exc:
                        raise GpuBusy(self._holder, timeout or 0.0) from exc
                    self._take(holder, evictor, reason)
                    return
            finally:
                async with self._cond:
                    self._waiters = max(self._waiters - 1, 0)

    def _take(self, holder: str, evictor: Evictor | None, reason: str) -> None:
        self._holder = holder
        self._evictor = evictor
        self._reason = reason
        self._since = time.monotonic()
        self._return_requested = False
        self._debug(f"{self.name} -> {holder!r} ({reason})")

    async def release(self, holder: str) -> None:
        async with self._cond:
            if self._holder != holder:
                return
            self._debug(f"{self.name} released by {holder!r}")
            self._holder = None
            self._evictor = None
            self._reason = ""
            self._return_requested = False
            self._cond.notify_all()

    async def request_return(self, reason: str = "") -> bool:
        """Ask the current holder to give the GPU back. True if there was a holder."""
        evict_now: Evictor | None = None
        async with self._cond:
            if self._holder is None:
                return False
            self._return_requested = True
            self._reason = reason or self._reason
            if self._evictor is not None:
                evict_now, self._evictor = self._evictor, None
        if evict_now is not None:
            outcome = evict_now()
            if inspect.isawaitable(outcome):
                await outcome
        return True

    def hold(self, holder: str, **kwargs):
        lease = self

        class _Hold:
            async def __aenter__(self):
                await lease.acquire(holder, **kwargs)
                return lease

            async def __aexit__(self, *exc):
                await lease.release(holder)

        return _Hold()

    def _debug(self, message: str) -> None:
        if self.log:
            self.log.debug(message)
