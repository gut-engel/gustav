"""The daemon's state machine.

STARTING -> WAKE -> SLEEP_PENDING -> SLEEP -> WAKE, plus ERROR and SAFE_MODE.
Transitions are explicit and validated: an illegal transition is a bug, and raising
here is how we find it instead of ending up in a state nobody designed for.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class State(StrEnum):
    STARTING = "starting"
    WAKE = "wake"
    SLEEP_PENDING = "sleep_pending"
    SLEEP = "sleep"
    ERROR = "error"
    SAFE_MODE = "safe_mode"
    STOPPING = "stopping"
    STOPPED = "stopped"


#: Where each state may go. SAFE_MODE serves chat with known-good weights, so it is a
#: serving state like WAKE, but it never schedules a sleep until the operator clears it.
TRANSITIONS: dict[State, frozenset[State]] = {
    State.STARTING: frozenset({State.WAKE, State.SAFE_MODE, State.ERROR, State.STOPPING}),
    State.WAKE: frozenset(
        {State.SLEEP_PENDING, State.SAFE_MODE, State.ERROR, State.STOPPING}
    ),
    State.SLEEP_PENDING: frozenset(
        {State.SLEEP, State.WAKE, State.SAFE_MODE, State.ERROR, State.STOPPING}
    ),
    State.SLEEP: frozenset({State.WAKE, State.SAFE_MODE, State.ERROR, State.STOPPING}),
    State.ERROR: frozenset({State.WAKE, State.SAFE_MODE, State.STOPPING}),
    State.SAFE_MODE: frozenset({State.WAKE, State.ERROR, State.STOPPING}),
    State.STOPPING: frozenset({State.STOPPED}),
    State.STOPPED: frozenset(),
}

#: States in which the daemon answers chat requests.
SERVING_STATES = frozenset({State.WAKE, State.SLEEP_PENDING, State.SAFE_MODE})


class InvalidTransition(RuntimeError):
    def __init__(self, source: State, target: State):
        super().__init__(f"illegal state transition {source} -> {target}")
        self.source = source
        self.target = target


@dataclass(frozen=True)
class Transition:
    at: datetime
    source: State | None
    target: State
    reason: str


Listener = Callable[[Transition], None]


class StateMachine:
    def __init__(
        self,
        initial: State = State.STARTING,
        *,
        clock=None,
        on_transition: Listener | None = None,
    ):
        from gustav.core.clock import RealClock

        self._clock = clock or RealClock()
        self._state = initial
        self._listeners: list[Listener] = []
        if on_transition:
            self._listeners.append(on_transition)
        self.history: list[Transition] = [
            Transition(self._clock.now(), None, initial, "initial")
        ]

    @property
    def state(self) -> State:
        return self._state

    @property
    def serving(self) -> bool:
        return self._state in SERVING_STATES

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def can(self, target: State) -> bool:
        return target in TRANSITIONS[self._state]

    def transition(self, target: State, reason: str = "") -> Transition:
        if target == self._state:
            # Idempotent: re-entering the current state is a no-op, not an error.
            return self.history[-1]
        if not self.can(target):
            raise InvalidTransition(self._state, target)
        tr = Transition(self._clock.now(), self._state, target, reason)
        self._state = target
        self.history.append(tr)
        for listener in list(self._listeners):
            listener(tr)
        return tr

    def __repr__(self) -> str:
        return f"<StateMachine {self._state}>"
