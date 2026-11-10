"""The built-in scheduler: no cron, no systemd timer.

``decide()`` is a pure function of observable facts, which is what makes the four
triggers (window, idle, pending work, manual) testable without waiting for midnight.
The daemon calls it on a timer and acts on the decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from gustav.config.schema import ScheduleConfig, TimeWindow
from gustav.core.clock import Clock, RealClock
from gustav.core.state import State


@dataclass(frozen=True)
class SleepDecision:
    should_sleep: bool
    trigger: str | None
    reason: str

    def __bool__(self) -> bool:
        return self.should_sleep


NO = SleepDecision(False, None, "")


@dataclass(frozen=True)
class Observation:
    """Everything the decision depends on, gathered by the daemon each tick."""

    state: State
    now: datetime
    #: Last user interaction, or daemon start if there has not been one yet.
    last_activity_at: datetime
    pending_items: int = 0
    learning_paused: bool = False
    last_sleep_finished_at: datetime | None = None
    manual_trigger: str | None = None
    gpu_temp_c: float | None = None
    #: A run that was interrupted and still has steps to finish.
    resumable_run: bool = False


class Scheduler:
    def __init__(self, config: ScheduleConfig, clock: Clock | None = None):
        self.config = config
        self.clock = clock or RealClock()
        self._windows: list[TimeWindow] = config.parsed_windows()

    # --- windows --------------------------------------------------------
    def in_window(self, moment: datetime) -> bool:
        if not self._windows:
            return False
        return any(w.contains(moment.time()) for w in self._windows)

    def next_window_start(self, moment: datetime) -> datetime | None:
        """When the next sleep window opens (or now, if one is open)."""
        if not self._windows:
            return None
        if self.in_window(moment):
            return moment
        candidates = []
        for window in self._windows:
            start = moment.replace(
                hour=window.start.hour, minute=window.start.minute, second=0, microsecond=0
            )
            if start <= moment:
                start += timedelta(days=1)
            candidates.append(start)
        return min(candidates)

    # --- the decision ---------------------------------------------------
    def decide(self, obs: Observation) -> SleepDecision:
        cfg = self.config
        idle_seconds = max((obs.now - obs.last_activity_at).total_seconds(), 0.0)

        if obs.manual_trigger:
            if obs.learning_paused:
                return SleepDecision(
                    False, None, "learning is paused ('ai resume-learning' to allow sleep)"
                )
            if obs.state is not State.WAKE:
                return SleepDecision(False, None, f"cannot sleep from state {obs.state}")
            return SleepDecision(True, obs.manual_trigger, "requested manually")

        if not cfg.enabled:
            return SleepDecision(False, None, "sleep is disabled in config")
        if obs.learning_paused:
            return SleepDecision(False, None, "learning is paused")
        if obs.state is not State.WAKE:
            return SleepDecision(False, None, f"state is {obs.state}")

        if obs.gpu_temp_c is not None and obs.gpu_temp_c > cfg.max_gpu_temp_c:
            return SleepDecision(
                False, None, f"GPU is {obs.gpu_temp_c:.0f}C, above the {cfg.max_gpu_temp_c}C limit"
            )

        if obs.last_sleep_finished_at is not None:
            since = (obs.now - obs.last_sleep_finished_at).total_seconds()
            if since < cfg.cooldown_minutes * 60:
                mins = (cfg.cooldown_minutes * 60 - since) / 60
                return SleepDecision(False, None, f"cooldown: {mins:.0f} min until next sleep")

        if idle_seconds < cfg.min_idle_seconds:
            return SleepDecision(
                False, None, f"user active {idle_seconds:.0f}s ago (min idle {cfg.min_idle_seconds}s)"
            )

        in_window = self.in_window(obs.now)
        enough_pending = obs.pending_items >= cfg.pending_threshold
        has_work = obs.pending_items >= cfg.min_pending_items or obs.resumable_run

        # An interrupted run outranks everything: finish what was started.
        if obs.resumable_run and (in_window or enough_pending):
            return SleepDecision(True, "resume", "an interrupted sleep run has unfinished steps")

        if in_window:
            if not has_work:
                return SleepDecision(
                    False, None, f"in sleep window but only {obs.pending_items} pending items"
                )
            return SleepDecision(
                True, "window", f"inside sleep window, {obs.pending_items} items pending"
            )

        if cfg.pending_trigger_enabled and enough_pending:
            return SleepDecision(
                True,
                "pending",
                f"{obs.pending_items} pending items (threshold {cfg.pending_threshold})",
            )

        if cfg.idle_trigger_enabled and idle_seconds >= cfg.idle_minutes * 60:
            if enough_pending:
                return SleepDecision(
                    True,
                    "idle",
                    f"idle {idle_seconds / 60:.0f} min and {obs.pending_items} items pending",
                )
            return SleepDecision(
                False,
                None,
                f"idle {idle_seconds / 60:.0f} min but outside the window with "
                f"{obs.pending_items} items (needs {cfg.pending_threshold})",
            )

        return SleepDecision(False, None, "no trigger met")

    # --- for `ai status` ------------------------------------------------
    def describe_next(self, obs: Observation) -> str:
        if not self.config.enabled:
            return "sleep disabled"
        if obs.learning_paused:
            return "paused"
        nxt = self.next_window_start(obs.now)
        if nxt is None:
            return "no window configured (idle/pending triggers only)"
        if self.in_window(obs.now):
            return "window open now"
        return nxt.isoformat(timespec="minutes")
