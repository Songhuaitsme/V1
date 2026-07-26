"""Candidate-to-calendar commit coordination with bounded conflict attempts."""

from dataclasses import dataclass
from enum import Enum

from v1.domain.candidates import Candidate
from v1.domain.reservations import CommitStatus
from v1.simulation.event_engine import DomainEvent, EventEngine
from v1.simulation.state_machine import TaskStateMachine

from .resource_calendar import CommitResult, ReservationCalendar


class CommitDecisionStatus(str, Enum):
    COMMITTED = "COMMITTED"
    RETRY_WITH_NEW_SNAPSHOT = "RETRY_WITH_NEW_SNAPSHOT"
    ATTEMPT_LIMIT_REACHED = "ATTEMPT_LIMIT_REACHED"
    INFEASIBLE = "INFEASIBLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True)
class CommitDecisionResult:
    status: CommitDecisionStatus
    calendar_result: CommitResult
    commit_attempts: int
    events: tuple


class ReservationManager:
    def __init__(
        self,
        calendar: ReservationCalendar,
        state_machine: TaskStateMachine,
        event_engine: EventEngine,
        max_commit_attempts_per_decision: int,
    ):
        if (
            isinstance(max_commit_attempts_per_decision, bool)
            or not isinstance(max_commit_attempts_per_decision, int)
            or max_commit_attempts_per_decision <= 0
        ):
            raise ValueError("max_commit_attempts_per_decision must be positive")
        self.calendar = calendar
        self.state_machine = state_machine
        self.event_engine = event_engine
        self.max_commit_attempts = max_commit_attempts_per_decision

    def commit_selected(self, candidate: Candidate) -> CommitDecisionResult:
        request = candidate.to_reservation_request()
        result = self.calendar.try_commit(
            request,
            expected_version=candidate.reservation_snapshot_version,
        )
        if result.status is CommitStatus.COMMITTED:
            events = tuple(self.event_engine.register_reservation(result.reservation))
            attempts = self.state_machine.runtime(candidate.task_id).commit_attempts_current_decision
            return CommitDecisionResult(
                CommitDecisionStatus.COMMITTED,
                result,
                attempts,
                events,
            )
        if result.status is CommitStatus.CONFLICT:
            attempts = self.state_machine.increment_commit_attempts(candidate.task_id)
            status = (
                CommitDecisionStatus.ATTEMPT_LIMIT_REACHED
                if attempts >= self.max_commit_attempts
                else CommitDecisionStatus.RETRY_WITH_NEW_SNAPSHOT
            )
            return CommitDecisionResult(status, result, attempts, ())
        attempts = self.state_machine.runtime(candidate.task_id).commit_attempts_current_decision
        if result.status in {
            CommitStatus.CPU_INFEASIBLE,
            CommitStatus.BANDWIDTH_INFEASIBLE,
            CommitStatus.CANDIDATE_STALE,
        }:
            return CommitDecisionResult(
                CommitDecisionStatus.INFEASIBLE,
                result,
                attempts,
                (),
            )
        return CommitDecisionResult(
            CommitDecisionStatus.INTERNAL_ERROR,
            result,
            attempts,
            (),
        )
