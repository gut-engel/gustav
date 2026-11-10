from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from gustav.config.schema import ScheduleConfig
from gustav.core.scheduler import Observation, Scheduler
from gustav.core.state import State


def sched(**kwargs) -> Scheduler:
    defaults = dict(windows=["00:00-06:00"], min_idle_seconds=120, cooldown_minutes=60,
                    idle_minutes=45, pending_threshold=50, min_pending_items=1)
    defaults.update(kwargs)
    return Scheduler(ScheduleConfig(**defaults))


def obs(now: datetime, *, idle_minutes: float = 10, pending: int = 10, **kwargs) -> Observation:
    base = dict(
        state=State.WAKE,
        now=now,
        last_activity_at=now - timedelta(minutes=idle_minutes),
        pending_items=pending,
    )
    base.update(kwargs)
    return Observation(**base)


NIGHT = datetime(2026, 3, 1, 2, 0)
DAY = datetime(2026, 3, 1, 14, 0)


# --- window trigger -------------------------------------------------------
def test_window_trigger_fires_at_night():
    decision = sched().decide(obs(NIGHT))
    assert decision.should_sleep and decision.trigger == "window"


def test_window_trigger_does_not_fire_during_the_day():
    assert not sched().decide(obs(DAY))


def test_wrapping_window():
    s = sched(windows=["22:00-02:00"])
    assert s.decide(obs(datetime(2026, 3, 1, 23, 30))).trigger == "window"
    assert s.decide(obs(datetime(2026, 3, 1, 1, 30))).trigger == "window"
    assert not s.decide(obs(datetime(2026, 3, 1, 3, 0)))


def test_window_with_nothing_to_do_does_not_sleep():
    decision = sched(min_pending_items=5).decide(obs(NIGHT, pending=0))
    assert not decision.should_sleep
    assert "pending" in decision.reason


# --- idle trigger ---------------------------------------------------------
def test_idle_trigger_needs_pending_work_outside_the_window():
    s = sched(idle_minutes=45)
    assert not s.decide(obs(DAY, idle_minutes=60, pending=10))
    decision = s.decide(obs(DAY, idle_minutes=60, pending=60))
    assert decision.should_sleep and decision.trigger == "pending"


def test_idle_trigger_inside_window_is_the_window_trigger():
    decision = sched().decide(obs(NIGHT, idle_minutes=60))
    assert decision.trigger == "window"


def test_idle_explains_why_it_held_back():
    decision = sched(idle_minutes=45).decide(obs(DAY, idle_minutes=90, pending=3))
    assert not decision.should_sleep
    assert "idle 90 min" in decision.reason


# --- pending trigger ------------------------------------------------------
def test_pending_threshold_trigger_during_the_day():
    decision = sched(pending_threshold=20).decide(obs(DAY, pending=25, idle_minutes=5))
    assert decision.should_sleep and decision.trigger == "pending"


def test_pending_below_threshold_during_the_day_does_nothing():
    assert not sched(pending_threshold=20).decide(obs(DAY, pending=19, idle_minutes=5))


# --- manual ---------------------------------------------------------------
def test_manual_overrides_window_and_cooldown():
    decision = sched().decide(
        obs(DAY, pending=0, idle_minutes=0, manual_trigger="manual",
            last_sleep_finished_at=DAY - timedelta(minutes=1))
    )
    assert decision.should_sleep and decision.trigger == "manual"


def test_manual_refused_when_learning_paused():
    decision = sched().decide(obs(NIGHT, manual_trigger="manual", learning_paused=True))
    assert not decision.should_sleep
    assert "paused" in decision.reason


def test_manual_refused_when_not_awake():
    decision = sched().decide(obs(NIGHT, manual_trigger="manual", state=State.SLEEP))
    assert not decision.should_sleep


# --- guards ---------------------------------------------------------------
def test_min_idle_protects_a_live_conversation():
    decision = sched(min_idle_seconds=300).decide(obs(NIGHT, idle_minutes=1))
    assert not decision.should_sleep
    assert "user active" in decision.reason


def test_cooldown_blocks_back_to_back_sleeps():
    decision = sched(cooldown_minutes=60).decide(
        obs(NIGHT, last_sleep_finished_at=NIGHT - timedelta(minutes=10))
    )
    assert not decision.should_sleep
    assert "cooldown" in decision.reason


def test_hot_gpu_defers_sleep():
    decision = sched().decide(obs(NIGHT, gpu_temp_c=95.0))
    assert not decision.should_sleep
    assert "95C" in decision.reason


def test_learning_paused_blocks_automatic_sleep():
    assert not sched().decide(obs(NIGHT, learning_paused=True))


def test_disabled_schedule_never_sleeps():
    assert not sched(enabled=False).decide(obs(NIGHT))


def test_not_awake_never_sleeps():
    for state in (State.SLEEP, State.SAFE_MODE, State.STARTING, State.ERROR):
        assert not sched().decide(obs(NIGHT, state=state))


# --- resume ---------------------------------------------------------------
def test_interrupted_run_is_resumed_first():
    decision = sched().decide(obs(NIGHT, resumable_run=True, pending=0))
    assert decision.should_sleep and decision.trigger == "resume"


def test_interrupted_run_waits_for_a_window_if_there_is_little_to_do():
    assert not sched().decide(obs(DAY, resumable_run=True, pending=0, idle_minutes=5))


# --- next-sleep estimate --------------------------------------------------
def test_next_window_start():
    s = sched()
    assert s.next_window_start(DAY) == datetime(2026, 3, 2, 0, 0)
    assert s.next_window_start(NIGHT) == NIGHT  # a window is open now


def test_describe_next():
    s = sched()
    assert s.describe_next(obs(NIGHT)) == "window open now"
    assert s.describe_next(obs(DAY)).startswith("2026-03-02")
    assert "paused" == s.describe_next(obs(DAY, learning_paused=True))


def test_no_window_configured_falls_back_to_other_triggers():
    s = sched(windows=[])
    assert s.next_window_start(DAY) is None
    assert not s.decide(obs(DAY, pending=10, idle_minutes=90))
    assert s.decide(obs(DAY, pending=99, idle_minutes=90)).trigger == "pending"
