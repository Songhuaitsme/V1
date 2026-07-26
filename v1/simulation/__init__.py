"""v1.0 discrete-event simulation primitives."""

from .state_machine import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    StateTransition,
    StateTransitionError,
    TaskStateMachine,
)
from .event_engine import DomainEvent, EventEngine


__all__ = [
    "ALLOWED_TRANSITIONS",
    "DomainEvent",
    "EventEngine",
    "TERMINAL_STATES",
    "StateTransition",
    "StateTransitionError",
    "TaskStateMachine",
]
