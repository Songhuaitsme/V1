"""Online per-task v1.0 scheduler orchestration."""

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Iterable, Optional, Tuple

from v1.domain.candidates import CandidateGenerationStatus
from v1.domain.models import TaskSpec
from v1.simulation.event_engine import DomainEvent, EventEngine
from v1.simulation.state_machine import StateTransition, TaskStateMachine

from .candidate_generator import CandidateGenerator
from .policies import CandidateStreamSelection, EarliestFeasiblePolicy
from .queue_manager import TaskQueueManager
from .reservation_manager import CommitDecisionStatus, ReservationManager
from .resource_calendar import ReservationCalendar


@dataclass(frozen=True)
class SchedulingDecision:
    task_id: str
    status: str
    candidate_count: int
    selected_candidate_id: Optional[str] = None
    reservation_id: Optional[str] = None
    reason: str = ""
    decision_id: str = ""
    queue_order_key: tuple = ()
    reservation_snapshot_version: Optional[int] = None
    forecast_version: str = ""
    candidate_mode: str = "complete"
    theoretical_time_slots: int = 0
    enumerated_time_slots: int = 0
    candidate_set_hash: str = ""
    earliest_counterfactual_candidate_id: Optional[str] = None
    commit_status: str = ""
    active_wait_sim: Optional[float] = None
    estimated_wait_benefit_vector: tuple = ()
    benefit_positive: Optional[bool] = None


@dataclass(frozen=True)
class SchedulingCycleResult:
    time_sim: float
    domain_events: Tuple[DomainEvent, ...]
    state_transitions: Tuple[StateTransition, ...]
    decisions: Tuple[SchedulingDecision, ...]
    calendar_version: int
    state_counts: dict


class V1Scheduler:
    def __init__(
        self,
        calendar: ReservationCalendar,
        candidate_generator: CandidateGenerator,
        max_queue_length: int,
        max_tasks_per_cycle: int,
        max_commit_attempts_per_decision: int,
        policy=None,
        metrics_ledger=None,
    ):
        if candidate_generator.calendar is not calendar:
            raise ValueError(
                "candidate_generator and V1Scheduler must share one reservation calendar"
            )
        if (
            isinstance(max_tasks_per_cycle, bool)
            or not isinstance(max_tasks_per_cycle, int)
            or max_tasks_per_cycle < 0
        ):
            raise ValueError("max_tasks_per_cycle must be a non-negative integer")
        self.calendar = calendar
        self.candidate_generator = candidate_generator
        self.state_machine = TaskStateMachine()
        self.event_engine = EventEngine(calendar, self.state_machine)
        self.queue_manager = TaskQueueManager(self.state_machine, max_queue_length)
        self.reservation_manager = ReservationManager(
            calendar,
            self.state_machine,
            self.event_engine,
            max_commit_attempts_per_decision,
        )
        self.max_tasks_per_cycle = max_tasks_per_cycle
        self.policy = policy or EarliestFeasiblePolicy()
        self.metrics_ledger = metrics_ledger
        self._committed_candidates = {}
        # Training can attach a lightweight profiler without changing the
        # scheduler's public construction contract or formal scheduling logic.
        self.profiler = None

    def submit_arrivals(self, tasks: Iterable[TaskSpec]) -> Tuple[StateTransition, ...]:
        transitions = []
        for task in sorted(tasks, key=lambda item: (item.arrival_time_sim, item.task_id)):
            transitions.append(self.queue_manager.enqueue_new(
                task,
                statically_serviceable=(
                    self.candidate_generator.is_statically_serviceable(task)
                ),
            ))
        return tuple(transitions)

    def run_cycle(
        self,
        now_sim: float,
        forecast_version: str = "perfect-v1",
        forecast_covered_until_sim: Optional[float] = None,
        metric_evaluator=None,
        arrivals: Iterable[TaskSpec] = (),
        physical_event_types: Iterable[str] = (),
    ) -> SchedulingCycleResult:
        domain_events = list(self.event_engine.advance_to(now_sim))
        if self.metrics_ledger is not None:
            for event in domain_events:
                if event.event_type != "TASK_COMPLETED":
                    continue
                reservation = self.calendar.get_reservation(event.reservation_id)
                if reservation is None:
                    raise RuntimeError(
                        "completed event references an unknown reservation"
                    )
                self.metrics_ledger.record_completed_reservation(reservation)
        if metric_evaluator is None and self.metrics_ledger is not None:
            metric_evaluator = (
                self.metrics_ledger.accounting.candidate_metric_evaluator(
                    self.calendar.snapshot()
                )
            )
        physical_events = list(physical_event_types)
        for event in domain_events:
            if event.event_type == "TASK_COMPLETED":
                physical_events.append("RESERVATION_RELEASED")
            if event.event_type == "COMPUTE_STARTED":
                physical_events.append("BANDWIDTH_INTERVAL_ENDED")
        transitions = list(
            self.queue_manager.reactivate_pending(now_sim, physical_events)
        )
        transitions.extend(self.submit_arrivals(arrivals))
        decisions = []
        for task in self.queue_manager.eligible_tasks(
            self.max_tasks_per_cycle,
            now_sim=now_sim,
        ):
            decision, commit_events = self._schedule_one(
                task,
                now_sim,
                forecast_version,
                forecast_covered_until_sim,
                metric_evaluator,
            )
            domain_events.extend(commit_events)
            decisions.append(decision)
            if decision.status == "PENDING":
                transitions.append(self.queue_manager.mark_pending(task.task_id, now_sim))
        transitions.extend(
            self.queue_manager.expire_due_tasks_after_boundary_opportunity(now_sim)
        )
        return SchedulingCycleResult(
            time_sim=now_sim,
            domain_events=tuple(domain_events),
            state_transitions=tuple(item for item in transitions if item is not None),
            decisions=tuple(decisions),
            calendar_version=self.calendar.version,
            state_counts=self.state_machine.count_by_state(),
        )

    def finalize_metrics_after_full_settlement(self, accounting_interval=None):
        if self.metrics_ledger is None:
            raise RuntimeError("V1Scheduler has no metrics ledger")
        nonterminal = {
            state.value: count
            for state, count in self.state_machine.count_by_state().items()
            if state.value in {
                "Arrived",
                "Queued",
                "PendingUncommitted",
                "Reserved",
                "Transmitting",
                "Running",
            }
            and count
        }
        if nonterminal:
            raise RuntimeError(
                f"cannot finalize before full settlement: {nonterminal}"
            )
        return self.metrics_ledger.finalize_after_full_settlement(
            accounting_interval=accounting_interval
        )

    def _schedule_one(
        self,
        task: TaskSpec,
        now_sim: float,
        forecast_version: str,
        forecast_covered_until_sim: Optional[float],
        metric_evaluator,
    ) -> Tuple[SchedulingDecision, Tuple[DomainEvent, ...]]:
        self.state_machine.reset_commit_attempts(task.task_id)
        while True:
            snapshot = self.calendar.snapshot()
            prepare_started = time.perf_counter()
            result = self.candidate_generator.prepare_complete_stream(
                task,
                now_sim,
                reservation_snapshot=snapshot,
                forecast_version=forecast_version,
                forecast_covered_until_sim=forecast_covered_until_sim,
                metric_evaluator=metric_evaluator,
            )
            if self.profiler is not None:
                self.profiler.add(
                    "candidate_prepare_seconds",
                    time.perf_counter() - prepare_started,
                )
                self.profiler.increment(
                    "prepare_theoretical_slot_count",
                    result.theoretical_slot_count,
                )
            if result.status is not CandidateGenerationStatus.OK:
                return self._decision_record(
                    task.task_id,
                    "PENDING",
                    0,
                    task=task,
                    result=result,
                    snapshot_version=snapshot.reservation_version,
                    forecast_version=forecast_version,
                    reason=result.status.value,
                ), ()
            if hasattr(self.policy, "select_complete_stream"):
                selection = self.policy.select_complete_stream(
                    result,
                    task=task,
                )
            elif hasattr(self.policy, "select_stream"):
                selection = self.policy.select_stream(
                    result.iter_candidates(),
                    task=task,
                    context=result.context,
                )
            else:
                # Compatibility for external policies. Built-in v1 policies all
                # implement select_stream and therefore never materialize here.
                candidates = tuple(result.iter_candidates())
                selected = self.policy.select(candidates, task=task)
                earliest = min(
                    candidates,
                    key=lambda item: (
                        item.compute_start_sim,
                        item.target_node,
                        item.path.path_id,
                        item.candidate_id,
                    ),
                )
                digest = hashlib.sha256()
                for candidate in candidates:
                    digest.update(candidate.candidate_id.encode("utf-8"))
                    digest.update(b"\0")
                selection = CandidateStreamSelection(
                    selected,
                    earliest,
                    len(candidates),
                    digest.hexdigest(),
                    result.context,
                )
            selected = selection.selected_candidate
            commit = self.reservation_manager.commit_selected(selected)
            if commit.status is CommitDecisionStatus.COMMITTED:
                reservation = commit.calendar_result.reservation
                self._committed_candidates[task.task_id] = selected
                return self._decision_record(
                    task.task_id,
                    "RESERVED",
                    selection.candidate_count,
                    selected.candidate_id,
                    reservation.reservation_id,
                    task=task,
                    result=result,
                    selection=selection,
                    snapshot_version=snapshot.reservation_version,
                    forecast_version=forecast_version,
                    commit_status=commit.status.value,
                ), tuple(commit.events)
            if commit.status is CommitDecisionStatus.RETRY_WITH_NEW_SNAPSHOT:
                continue
            if commit.status is CommitDecisionStatus.ATTEMPT_LIMIT_REACHED:
                return self._decision_record(
                    task.task_id,
                    "QUEUED_CONFLICT_LIMIT",
                    selection.candidate_count,
                    selected.candidate_id,
                    task=task,
                    result=result,
                    selection=selection,
                    snapshot_version=snapshot.reservation_version,
                    forecast_version=forecast_version,
                    reason=commit.status.value,
                    commit_status=commit.status.value,
                ), ()
            return self._decision_record(
                task.task_id,
                "PENDING",
                selection.candidate_count,
                selected.candidate_id,
                task=task,
                result=result,
                selection=selection,
                snapshot_version=snapshot.reservation_version,
                forecast_version=forecast_version,
                reason=commit.calendar_result.status.value,
                commit_status=commit.status.value,
            ), ()

    def committed_candidate(self, task_id: str):
        return self._committed_candidates.get(task_id)

    @staticmethod
    def _decision_record(
        task_id,
        status,
        candidate_count,
        selected_candidate_id=None,
        reservation_id=None,
        *,
        task,
        result,
        selection=None,
        snapshot_version,
        forecast_version,
        reason="",
        commit_status="",
    ):
        empty_hash = hashlib.sha256().hexdigest()
        candidate_set_hash = (
            empty_hash if selection is None else selection.candidate_set_hash
        )
        selected = None if selection is None else selection.selected_candidate
        earliest_candidate = (
            None if selection is None else selection.earliest_candidate
        )
        earliest = (
            None if earliest_candidate is None else earliest_candidate.candidate_id
        )
        if selected is None or earliest_candidate is None:
            active_wait_sim = None
            wait_benefit_vector = ()
            benefit_positive = None
        else:
            active_wait_sim = selected.compute_start_sim - earliest_candidate.compute_start_sim
            wait_benefit_vector = (
                ("economic_cost_yuan_saved", earliest_candidate.estimated_candidate_marginal_system_cost_yuan - selected.estimated_candidate_marginal_system_cost_yuan),
                ("green_coverage_gain", selected.estimated_green_coverage - earliest_candidate.estimated_green_coverage),
                ("green_absorption_gain", selected.estimated_green_absorption_delta - earliest_candidate.estimated_green_absorption_delta),
                ("capacity_margin_gain", selected.capacity_margin - earliest_candidate.capacity_margin),
            )
            benefit_positive = (
                active_wait_sim > 1e-12
                and any(value > 1e-12 for _, value in wait_benefit_vector)
            )
        decision_payload = (
            task_id,
            result.candidate_mode.value,
            snapshot_version,
            forecast_version,
            candidate_set_hash,
        )
        decision_id = "decision-" + hashlib.sha256(
            json.dumps(decision_payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        preferred = task.absolute_preferred_start_sim
        audit_queue_key = (
            task.absolute_latest_start_sim,
            {"Hard": 0, "Soft": 1, "Flexible": 2}[task.sla_type.value],
            preferred,
            task.arrival_time_sim,
            task.task_id,
        )
        # Complete mode visits every declared node/path/grid slot before physical
        # filtering; feasible_candidate_count is tracked separately.
        enumerated_slots = result.theoretical_slot_count
        return SchedulingDecision(
            task_id,
            status,
            candidate_count,
            selected_candidate_id,
            reservation_id,
            reason,
            decision_id,
            audit_queue_key,
            snapshot_version,
            forecast_version,
            result.candidate_mode.value,
            result.theoretical_slot_count,
            enumerated_slots,
            candidate_set_hash,
            earliest,
            commit_status,
            active_wait_sim,
            wait_benefit_vector,
            benefit_positive,
        )
