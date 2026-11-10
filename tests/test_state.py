from __future__ import annotations

import pytest

from gustav.core.clock import FakeClock
from gustav.core.state import SERVING_STATES, InvalidTransition, State, StateMachine


def machine() -> StateMachine:
    return StateMachine(clock=FakeClock())


def test_happy_path_cycle():
    sm = machine()
    assert sm.state is State.STARTING
    sm.transition(State.WAKE, "started")
    sm.transition(State.SLEEP_PENDING, "window")
    sm.transition(State.SLEEP, "run 1")
    sm.transition(State.WAKE, "done")
    assert [t.target for t in sm.history] == [
        State.STARTING, State.WAKE, State.SLEEP_PENDING, State.SLEEP, State.WAKE,
    ]


def test_illegal_transition_raises():
    sm = machine()
    with pytest.raises(InvalidTransition):
        sm.transition(State.SLEEP, "cannot sleep before waking")
    assert sm.state is State.STARTING


def test_sleep_cannot_be_entered_from_wake_directly():
    sm = machine()
    sm.transition(State.WAKE, "up")
    with pytest.raises(InvalidTransition):
        sm.transition(State.SLEEP, "skipping SLEEP_PENDING")


def test_sleep_pending_can_be_cancelled_back_to_wake():
    sm = machine()
    sm.transition(State.WAKE, "up")
    sm.transition(State.SLEEP_PENDING, "window")
    sm.transition(State.WAKE, "user showed up")
    assert sm.state is State.WAKE


def test_reentering_current_state_is_a_noop():
    sm = machine()
    sm.transition(State.WAKE, "up")
    before = len(sm.history)
    sm.transition(State.WAKE, "again")
    assert len(sm.history) == before


def test_safe_mode_is_a_serving_state():
    sm = machine()
    sm.transition(State.SAFE_MODE, "adapter failed to load")
    assert sm.serving
    assert sm.state in SERVING_STATES
    sm.transition(State.WAKE, "operator fixed it")


def test_stopped_is_terminal():
    sm = machine()
    sm.transition(State.WAKE, "up")
    sm.transition(State.STOPPING, "sigterm")
    sm.transition(State.STOPPED, "done")
    for target in State:
        if target is State.STOPPED:
            continue
        with pytest.raises(InvalidTransition):
            sm.transition(target, "after the end")


def test_listeners_see_every_transition():
    seen = []
    sm = StateMachine(clock=FakeClock(), on_transition=seen.append)
    sm.transition(State.WAKE, "a")
    sm.transition(State.SLEEP_PENDING, "b")
    assert [(t.source, t.target) for t in seen] == [
        (State.STARTING, State.WAKE),
        (State.WAKE, State.SLEEP_PENDING),
    ]


def test_every_state_can_reach_stopping_or_is_terminal():
    """Graceful shutdown must be possible from wherever we are."""
    for state in State:
        if state in (State.STOPPING, State.STOPPED):
            continue
        sm = StateMachine(state, clock=FakeClock())
        assert sm.can(State.STOPPING), f"{state} cannot shut down"
