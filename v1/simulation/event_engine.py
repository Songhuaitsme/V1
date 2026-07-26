"""Reservation-driven discrete-event lifecycle engine for v1.0."""

from dataclasses import dataclass
import heapq
from typing import Dict, List, Optional, Tuple

from v1.domain.models import TaskSpec, TaskState
from v1.domain.reservations import ReleaseStatus, Reservation
from v1.domain.units import finite_number
from v1.scheduler.resource_calendar import ReservationCalendar

from .state_machine import StateTransition, StateTransitionError, TaskStateMachine


@dataclass(frozen=True)
class DomainEvent:
    event_type: str
    event_time_sim: float
    task_id: str
    reservation_id: Optional[str]
    previous_state: Optional[TaskState]
    new_state: Optional[TaskState]
    detail: Optional[str] = None


@dataclass(frozen=True, order=True)
class _ScheduledEvent:
    event_time_sim: float
    priority: int
    task_id: str
    event_type: str
    reservation_id: str


class EventEngine:
    """Advance reservation lifecycle events in deterministic timestamp batches."""

    _PRIORITY = {
        "compute_end": 0,
        "compute_start": 2,
        "transmission_start": 3,
    }

    def __init__(
        self,
        calendar: ReservationCalendar,
        state_machine: Optional[TaskStateMachine] = None,
        initial_time_sim: float = 0.0,
    ):
        self.calendar = calendar
        self.state_machine = state_machine or TaskStateMachine()
        self.current_time_sim = finite_number("initial_time_sim", initial_time_sim)
        self._event_heap: List[_ScheduledEvent] = []
        self._reservations: Dict[str, Reservation] = {}
        self._terminal_task_ids = set()

    def register_task(self, task_spec: TaskSpec, enqueue: bool = True) -> List[DomainEvent]:
        self.state_machine.register(task_spec)
        if not enqueue:
            return []
        transition = self.state_machine.transition(
            task_spec.task_id,
            TaskState.QUEUED,
            task_spec.arrival_time_sim,
        )
        return [self._domain_event("TASK_QUEUED", transition)]

    def register_reservation(self, reservation: Reservation) -> List[DomainEvent]:
        if reservation.reservation_id in self._reservations:
            return []
        if reservation.committed_at_sim < self.current_time_sim - 1e-12:
            raise StateTransitionError("cannot register a reservation in the past")
        if not self.calendar.verify_reservation(reservation.reservation_id):
            raise StateTransitionError("reservation is not fully present in calendar")
        runtime = self.state_machine.runtime(reservation.task_id)
        if runtime.state is not TaskState.QUEUED:
            raise StateTransitionError("only Queued tasks can receive reservations")

        transition = self.state_machine.transition(
            reservation.task_id,
            TaskState.RESERVED,
            reservation.committed_at_sim,
            reservation_id=reservation.reservation_id,
        )
        self._reservations[reservation.reservation_id] = reservation
        if reservation.transmission_interval_sim is not None:
            self._push(
                reservation.transmission_interval_sim.start_sim,
                "transmission_start",
                reservation,
            )
        self._push(
            reservation.compute_interval_sim.start_sim,
            "compute_start",
            reservation,
        )
        self._push(
            reservation.compute_interval_sim.end_sim,
            "compute_end",
            reservation,
        )
        events = [self._domain_event("RESERVATION_COMMITTED", transition)]
        if reservation.committed_at_sim <= self.current_time_sim + 1e-12:
            events.extend(self._process_timestamp(self.current_time_sim))
        return events

    def _push(self, time_sim: float, event_type: str, reservation: Reservation) -> None:
        heapq.heappush(
            self._event_heap,
            _ScheduledEvent(
                event_time_sim=time_sim,
                priority=self._PRIORITY[event_type],
                task_id=reservation.task_id,
                event_type=event_type,
                reservation_id=reservation.reservation_id,
            ),
        )

    @staticmethod
    def _domain_event(event_type: str, transition: StateTransition) -> DomainEvent:
        return DomainEvent(
            event_type=event_type,
            event_time_sim=transition.event_time_sim,
            task_id=transition.task_id,
            reservation_id=transition.reservation_id,
            previous_state=transition.previous_state,
            new_state=transition.new_state,
            detail=transition.terminal_reason,
        )

    def advance_to(self, target_time_sim: float) -> List[DomainEvent]:
        target = finite_number("target_time_sim", target_time_sim)
        if target < self.current_time_sim - 1e-12:
            raise StateTransitionError("event engine cannot move backwards")
        emitted = []
        while self._event_heap and self._event_heap[0].event_time_sim <= target + 1e-12:
            timestamp = self._event_heap[0].event_time_sim
            emitted.extend(self._process_timestamp(timestamp))
        self.current_time_sim = target
        return emitted

    def _process_timestamp(self, timestamp: float) -> List[DomainEvent]:
        scheduled = []
        while (
            self._event_heap
            and abs(self._event_heap[0].event_time_sim - timestamp) <= 1e-12
        ):
            scheduled.append(heapq.heappop(self._event_heap))
        scheduled.sort()
        emitted = []
        for item in scheduled:
            if item.task_id in self._terminal_task_ids:
                continue
            emitted.extend(self._process_event(item))
        self.current_time_sim = max(self.current_time_sim, timestamp)
        return emitted

    def _process_event(self, item: _ScheduledEvent) -> List[DomainEvent]:
        reservation = self._reservations[item.reservation_id]
        if item.event_type in ("transmission_start", "compute_start"):
            if not self.calendar.verify_reservation(item.reservation_id):
                return self._fail_now(
                    reservation.task_id,
                    item.event_time_sim,
                    "RESERVATION_BROKEN",
                )

        if item.event_type == "transmission_start":
            transition = self.state_machine.transition(
                reservation.task_id,
                TaskState.TRANSMITTING,
                item.event_time_sim,
                reservation_id=reservation.reservation_id,
            )
            return [self._domain_event("TRANSMISSION_STARTED", transition)]

        if item.event_type == "compute_start":
            transition = self.state_machine.transition(
                reservation.task_id,
                TaskState.RUNNING,
                item.event_time_sim,
                reservation_id=reservation.reservation_id,
            )
            return [self._domain_event("COMPUTE_STARTED", transition)]

        if item.event_type == "compute_end":
            transition = self.state_machine.transition(
                reservation.task_id,
                TaskState.COMPLETED,
                item.event_time_sim,
                terminal_reason="COMPLETED",
                reservation_id=reservation.reservation_id,
            )
            release = self.calendar.release_on_normal_completion(
                reservation.reservation_id,
                at_time_sim=item.event_time_sim,
            )
            if release.status is not ReleaseStatus.RELEASED:
                raise StateTransitionError(
                    f"completion release failed: {release.status.value}"
                )
            self._terminal_task_ids.add(reservation.task_id)
            return [self._domain_event("TASK_COMPLETED", transition)]

        raise StateTransitionError(f"unknown scheduled event: {item.event_type}")

    def fail_task(
        self,
        task_id: str,
        event_time_sim: float,
        reason: str,
    ) -> List[DomainEvent]:
        self.advance_to(event_time_sim)
        return self._fail_now(task_id, event_time_sim, reason)

    def _fail_now(
        self,
        task_id: str,
        event_time_sim: float,
        reason: str,
    ) -> List[DomainEvent]:
        runtime = self.state_machine.runtime(task_id)
        if runtime.state in {
            TaskState.COMPLETED,
            TaskState.REJECTED,
            TaskState.EXPIRED,
            TaskState.FAILED,
        }:
            return []
        if runtime.state not in {
            TaskState.RESERVED,
            TaskState.TRANSMITTING,
            TaskState.RUNNING,
        }:
            raise StateTransitionError("only committed tasks can fail in WP-2")
        transition = self.state_machine.transition(
            task_id,
            TaskState.FAILED,
            event_time_sim,
            terminal_reason=reason,
            reservation_id=runtime.reservation_id,
        )
        if runtime.reservation_id is not None:
            self.calendar.release_on_failure(runtime.reservation_id)
        self._terminal_task_ids.add(task_id)
        return [self._domain_event("TASK_FAILED", transition)]

    @property
    def completed_count(self) -> int:
        return self.state_machine.count_by_state()[TaskState.COMPLETED]

    @property
    def next_event_time_sim(self) -> Optional[float]:
        return None if not self._event_heap else self._event_heap[0].event_time_sim
