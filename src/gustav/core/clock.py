"""Time, injected.

Every component that cares about time takes a ``Clock``, so the scheduler can be
tested across midnight boundaries without waiting for midnight.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime:
        """Local wall-clock time (naive, local zone) — what sleep windows are expressed in."""

    def monotonic(self) -> float:
        """Monotonic seconds, for durations and timeouts."""

    async def sleep(self, seconds: float) -> None: ...


class RealClock:
    def now(self) -> datetime:
        return datetime.now()

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class FakeClock:
    """Manually advanced clock for tests."""

    def __init__(self, start: datetime | None = None, *, monotonic: float = 1000.0):
        self._now = start or datetime(2026, 1, 1, 12, 0, 0)
        self._mono = monotonic
        self.slept: list[float] = []

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._mono += seconds

    def set(self, moment: datetime) -> None:
        delta = (moment - self._now).total_seconds()
        self._now = moment
        self._mono += max(delta, 0.0)

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.advance(seconds)
        await asyncio.sleep(0)
