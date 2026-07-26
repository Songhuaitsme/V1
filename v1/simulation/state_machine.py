"""Deterministic v1.0 task lifecycle transitions."""

from dataclasses import dataclass, replace
from typing import Dict, Optional, Set

from v1.domain.models import TaskRuntime, TaskSpec, TaskState
from v1.domain.units import finite_number


TERMINAL_STATES = {
    TaskState.COMPLETED,
    TaskState.REJECTED,
    TaskState.EXPIRED,
    TaskState.FAILED,
}

ALLOWED_TRANSITIONS = {
    TaskState.ARRIVED: {TaskState.QUEUED, TaskState.REJECTED},
    TaskState.QUEUED: {
        TaskState.RESERVED,
        TaskState.PENDING_UNCOMMITTED,
        TaskState.EXPIRED,
    },
    TaskState.PENDING_UNCOMMITTED: {TaskState.QUEUED, TaskState.EXPIRED},
    TaskState.RESERVED: {
        TaskState.TRANSMITTING,
        TaskState.RUNNING,
        TaskState.FAILED,
    },
    TaskState.TRANSMITTING: {TaskState.RUNNING, TaskState.FAILED},
    TaskState.RUNNING: {TaskState.COMPLETED, TaskState.FAILED},
    TaskState.COMPLETED: set(),
    TaskState.REJECTED: set(),
    TaskState.EXPIRED: set(),
    TaskState.FAILED: set(),
}

TERMINAL_REASON_BY_STATE = {
    TaskState.COMPLETED: {"COMPLETED"},
    TaskState.REJECTED: {
        "INVALID_TASK",
        "STATICALLY_UNSERVICEABLE",
        "SCHEDULER_QUEUE_CAPACITY",
    },
    TaskState.EXPIRED: {"ABSOLUTE_START_DEADLINE"},
    TaskState.FAILED: {
        "TRANSMISSION_FAILURE",
        "EXECUTION_FAILURE",
        "RESERVATION_BROKEN",
    },
}


class StateTransitionError(ValueError):
    pass


@dataclass(frozen=True)
class StateTransition:
    task_id: str
    event_time_sim: float
    previous_state: TaskState
    new_state: TaskState
    state_version: int
    terminal_reason: Optional[str] = None
    reservation_id: Optional[str] = None


class TaskStateMachine:
    def __init__(self):
        self._specs: Dict[str, TaskSpec] = {}
        self._runtimes: Dict[str, TaskRuntime] = {}

    def register(self, task_spec: TaskSpec) -> TaskRuntime:
        if task_spec.task_id in self._specs:
            raise StateTransitionError(f"duplicate task_id: {task_spec.task_id}")
        runtime = TaskRuntime(
            task_id=task_spec.task_id,
            state=TaskState.ARRIVED,
            last_state_change_sim=task_spec.arrival_time_sim,
        )
        self._specs[task_spec.task_id] = task_spec
        self._runtimes[task_spec.task_id] = runtime
        return replace(runtime)

    def task_spec(self, task_id: str) -> TaskSpec:
        try:
            return self._specs[task_id]
        except KeyError:
            raise StateTransitionError(f"unknown task_id: {task_id}")

    def runtime(self, task_id: str) -> TaskRuntime:
        try:
            return replace(self._runtimes[task_id])
        except KeyError:
            raise StateTransitionError(f"unknown task_id: {task_id}")

    def transition(
        self,
        task_id: str,
        new_state: TaskState,
        event_time_sim: float,
        terminal_reason: Optional[str] = None,
        reservation_id: Optional[str] = None,
    ) -> Optional[StateTransition]:
        if not isinstance(new_state, TaskState):
            raise StateTransitionError("new_state must be a TaskState")
        try:
            runtime = self._runtimes[task_id]
        except KeyError:
            raise StateTransitionError(f"unknown task_id: {task_id}")
        event_time = finite_number("event_time_sim", event_time_sim)
        if event_time < runtime.last_state_change_sim - 1e-12:
            raise StateTransitionError("task state cannot move backwards in time")
        if runtime.state is new_state:
            return None
        if new_state not in ALLOWED_TRANSITIONS[runtime.state]:
            raise StateTransitionError(
                f"illegal transition {runtime.state.value}->{new_state.value}"
            )

        if new_state in TERMINAL_STATES:
            allowed_reasons: Set[str] = TERMINAL_REASON_BY_STATE[new_state]
            if terminal_reason not in allowed_reasons:
                raise StateTransitionError(
                    f"{new_state.value} requires one of {sorted(allowed_reasons)}"
                )
        elif terminal_reason is not None:
            raise StateTransitionError("non-terminal transition cannot set terminal_reason")
        if (
            reservation_id is not None
            and runtime.reservation_id not in (None, reservation_id)
        ):
            raise StateTransitionError("reservation_id is immutable once assigned")

        previous = runtime.state
        runtime.state = new_state
        runtime.state_version += 1
        runtime.last_state_change_sim = event_time
        if reservation_id is not None:
            runtime.reservation_id = reservation_id
        if terminal_reason is not None:
            runtime.terminal_reason = terminal_reason
        return StateTransition(
            task_id=task_id,
            event_time_sim=event_time,
            previous_state=previous,
            new_state=new_state,
            state_version=runtime.state_version,
            terminal_reason=runtime.terminal_reason,
            reservation_id=runtime.reservation_id,
        )

    def count_by_state(self) -> Dict[TaskState, int]:
        counts = {state: 0 for state in TaskState}
        for runtime in self._runtimes.values():
            counts[runtime.state] += 1
        return counts

    def increment_pending_attempts(self, task_id: str) -> int:
        try:
            runtime = self._runtimes[task_id]
        except KeyError:
            raise StateTransitionError(f"unknown task_id: {task_id}")
        runtime.pending_attempts += 1
        return runtime.pending_attempts

    def increment_commit_attempts(self, task_id: str) -> int:
        try:
            runtime = self._runtimes[task_id]
        except KeyError:
            raise StateTransitionError(f"unknown task_id: {task_id}")
        runtime.commit_attempts_current_decision += 1
        return runtime.commit_attempts_current_decision

    def reset_commit_attempts(self, task_id: str) -> None:
        try:
            self._runtimes[task_id].commit_attempts_current_decision = 0
        except KeyError:
            raise StateTransitionError(f"unknown task_id: {task_id}")

    @property
    def task_count(self) -> int:
        return len(self._specs)

    @property
    def task_ids(self):
        return tuple(self._specs.keys())
