"""Runtime invariant scanning and frozen quality-gate decisions."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Tuple

from v1.domain.models import TaskState
from v1.scheduler.resource_calendar import ReservationCalendar


@dataclass(frozen=True)
class InvariantViolation:
    invariant_id: str
    detail: str


def scan_scheduler_invariants(scheduler, cycle_result=None) -> Tuple[InvariantViolation, ...]:
    violations = []
    counts = scheduler.state_machine.count_by_state()
    if sum(counts.values()) != scheduler.state_machine.task_count:
        violations.append(InvariantViolation("I-14", "state counts do not conserve tasks"))
    snapshot = scheduler.calendar.snapshot()
    active_ids = {
        reservation.reservation_id
        for reservation in scheduler.calendar.active_reservations()
    }
    allocation_ids = {
        allocation.reservation_id
        for allocation in (
            snapshot.cpu_calendar_view + snapshot.link_calendar_view
        )
    }
    if allocation_ids != active_ids:
        violations.append(InvariantViolation("I-03", "CPU/BW allocation set is partially committed"))
    for allocation in snapshot.cpu_calendar_view:
        capacity = scheduler.calendar.node_capacity(allocation.resource_id)
        peak = ReservationCalendar.peak_usage(
            (
                item for item in snapshot.cpu_calendar_view
                if item.resource_id == allocation.resource_id
            ),
            allocation.interval_sim,
        )
        if capacity is None or peak > capacity + 1e-12:
            violations.append(InvariantViolation("I-01", f"CPU overcapacity at {allocation.resource_id}"))
    for allocation in snapshot.link_calendar_view:
        capacity = scheduler.calendar.link_capacity(allocation.resource_id)
        peak = ReservationCalendar.peak_usage(
            (
                item for item in snapshot.link_calendar_view
                if item.resource_id == allocation.resource_id
            ),
            allocation.interval_sim,
        )
        if capacity is None or peak > capacity + 1e-12:
            violations.append(InvariantViolation("I-02", f"link overcapacity at {allocation.resource_id}"))
    for reservation in scheduler.calendar.active_reservations():
        if not scheduler.calendar.verify_reservation(reservation.reservation_id):
            violations.append(InvariantViolation("I-04", f"broken reservation {reservation.reservation_id}"))
        if reservation.transmission_interval_sim is not None and not math.isclose(
            reservation.transmission_interval_sim.end_sim,
            reservation.compute_interval_sim.start_sim,
            abs_tol=1e-12,
        ):
            violations.append(InvariantViolation("I-05", f"non-JIT reservation {reservation.reservation_id}"))
        if reservation.transmission_interval_sim is not None and (
            reservation.transmission_interval_sim.start_sim
            < reservation.committed_at_sim - 1e-12
        ):
            violations.append(InvariantViolation("I-06", f"transmission predates decision {reservation.reservation_id}"))
        task = scheduler.state_machine.task_spec(reservation.task_id)
        if not math.isclose(
            reservation.compute_interval_sim.duration_sim,
            task.execution_duration_sim,
            abs_tol=1e-12,
        ):
            violations.append(InvariantViolation("I-07", f"execution duration mismatch {task.task_id}"))
        if reservation.compute_interval_sim.start_sim > task.absolute_latest_start_sim + 1e-12:
            violations.append(InvariantViolation("I-08", f"absolute SLA violation {task.task_id}"))
    for task_id in scheduler.state_machine.task_ids:
        runtime = scheduler.state_machine.runtime(task_id)
        if runtime.state is TaskState.PENDING_UNCOMMITTED and runtime.reservation_id is not None:
            violations.append(InvariantViolation("I-10", f"Pending task owns reservation {task_id}"))
        if runtime.state is TaskState.COMPLETED:
            reservation = scheduler.calendar.get_reservation(runtime.reservation_id)
            if reservation is None or not math.isclose(
                runtime.last_state_change_sim,
                reservation.compute_interval_sim.end_sim,
                abs_tol=1e-12,
            ):
                violations.append(InvariantViolation("I-09", f"invalid completion instant {task_id}"))
        candidate = scheduler.committed_candidate(task_id)
        if candidate is not None:
            expected_start_delay = (
                candidate.scheduler_queue_delay_sim
                + candidate.earliest_feasibility_lead_sim
                + candidate.active_wait_sim
            )
            if not math.isclose(candidate.start_delay_sim, expected_start_delay, abs_tol=1e-12):
                violations.append(InvariantViolation("I-18", f"delay decomposition mismatch {task_id}"))
    if cycle_result is not None:
        if cycle_result.time_sim != scheduler.event_engine.current_time_sim:
            violations.append(InvariantViolation("I-19", "cycle snapshot is not at batch end"))
        for decision in cycle_result.decisions:
            if (
                decision.candidate_mode == "complete"
                and decision.enumerated_time_slots != decision.theoretical_time_slots
            ):
                violations.append(InvariantViolation("I-16", f"incomplete enumeration {decision.task_id}"))
    return tuple(violations)


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"


@dataclass(frozen=True)
class QualityGateResult:
    status: GateStatus
    reasons: Tuple[str, ...]


def evaluate_quality_gate(
    *,
    seed_count: int,
    cost_mean_relative_change: float,
    cost_ci_upper: float,
    green_mean_change: float,
    green_ci_lower: float,
    completion_ci_lower: float,
    load_mean_change: float,
    physical_violation_count: int = 0,
) -> QualityGateResult:
    if seed_count < 10:
        return QualityGateResult(
            GateStatus.DIAGNOSTIC_ONLY,
            ("formal paired experiment requires at least 10 seeds",),
        )
    reasons = []
    if completion_ci_lower < -0.005:
        reasons.append("completion non-inferiority failed")
    if load_mean_change > 0.02:
        reasons.append("load guardrail failed")
    if physical_violation_count != 0:
        reasons.append("physical invariant violation")
    cost_improves = cost_mean_relative_change <= -0.05 and cost_ci_upper < 0.0
    cost_guards = cost_ci_upper <= 0.01
    green_improves = green_mean_change >= 0.03 and green_ci_lower > 0.0
    green_guards = green_ci_lower >= -0.01
    if not ((cost_improves and green_guards) or (green_improves and cost_guards)):
        reasons.append("neither objective improves while the other passes its guardrail")
    return QualityGateResult(
        GateStatus.PASS if not reasons else GateStatus.FAIL,
        tuple(reasons),
    )
