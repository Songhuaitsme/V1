"""Three-phase v1.0 frozen-policy evaluation runner."""

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Iterable, Mapping, Optional, Tuple

from v1.accounting.energy import AccountingReport
from v1.audit_v1 import scan_scheduler_invariants
from v1.domain.models import TaskSpec, TaskState
from v1.domain.reservations import TimeInterval
from v1.domain.units import TimeConverter, finite_number
from v1.scheduler.v1_scheduler import SchedulingCycleResult, V1Scheduler

from .metrics import SeedMetrics, TaskOutcome, build_seed_metrics


class EvaluationStatus(str, Enum):
    VALID = "VALID"
    INVALID_INCOMPLETE_SETTLEMENT = "INVALID_INCOMPLETE_SETTLEMENT"


@dataclass(frozen=True)
class EvaluationMetadata:
    requirements_version: str
    algorithm_version: str
    task_schema_version: str
    candidate_schema_version: str
    model_schema_version: str
    metric_schema_version: str
    aggregation_schema_version: str
    seed: int
    candidate_mode: str
    arrival_cutoff_sim: float
    evaluation_start_sim: float
    final_settlement_time_sim: float
    evaluation_safety_cap: int
    percentile_method: str = "linear"
    code_hash: str = hashlib.sha256(b"").hexdigest()
    model_hash: str = hashlib.sha256(b"").hexdigest()
    config_hash: str = hashlib.sha256(b"").hexdigest()
    topology_hash: str = hashlib.sha256(b"").hexdigest()
    task_trace_hash: str = hashlib.sha256(b"").hexdigest()
    exogenous_trace_hash: str = hashlib.sha256(b"").hexdigest()
    dependency_lock_hash: str = hashlib.sha256(b"").hexdigest()
    tariff_mode: str = "exogenous"
    gamma_per_second: float = 1.0
    ci_method: str = "paired_t"
    ci_resample_count: int = 0
    ci_random_seed: int = 0

    def __post_init__(self):
        versions = (
            self.requirements_version, self.algorithm_version,
            self.task_schema_version, self.candidate_schema_version,
            self.model_schema_version, self.metric_schema_version,
            self.aggregation_schema_version,
        )
        if any(version != "1.0" for version in versions):
            raise ValueError('all formal schema versions must equal "1.0"')
        for field in (
            "code_hash", "model_hash", "config_hash", "topology_hash",
            "task_trace_hash", "exogenous_trace_hash", "dependency_lock_hash",
        ):
            value = getattr(self, field)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
        gamma = finite_number("gamma_per_second", self.gamma_per_second)
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma_per_second must be in (0,1]")


@dataclass(frozen=True)
class TaskEvaluationRecord:
    task_id: str
    final_state: str
    terminal_reason: Optional[str]
    arrival_time_sim: float
    decision_time_sim: Optional[float]
    transmission_start_sim: Optional[float]
    transmission_end_sim: Optional[float]
    compute_start_sim: Optional[float]
    compute_end_sim: Optional[float]
    target_node: Optional[str]
    path_id: Optional[str]
    scheduler_queue_delay_sim: Optional[float]
    earliest_feasibility_lead_sim: Optional[float]
    active_wait_sim: Optional[float]
    reservation_lead_sim: Optional[float]
    start_delay_sim: Optional[float]
    completion_delay_sim: Optional[float]
    cpu_work_cpu_hours: float
    task_energy_mwh: Optional[float]
    task_attributed_cost_yuan: Optional[float]
    task_attributed_green_energy_mwh: Optional[float]
    candidate_marginal_system_cost_yuan: Optional[float]
    candidate_marginal_green_energy_mwh: Optional[float]
    estimated_vs_realized_errors: Optional[Mapping[str, float]]


@dataclass(frozen=True)
class EvaluationReport:
    status: EvaluationStatus
    metadata: EvaluationMetadata
    metrics: Optional[SeedMetrics]
    unsettled_task_ids: Tuple[str, ...]
    cycle_results: Tuple[SchedulingCycleResult, ...]
    phase_batch_counts: Mapping[str, int]
    task_records: Tuple[TaskEvaluationRecord, ...]
    decision_records: tuple
    accounting_report: Optional[AccountingReport]


class EvaluationRunner:
    NONTERMINAL = {
        TaskState.ARRIVED,
        TaskState.QUEUED,
        TaskState.PENDING_UNCOMMITTED,
        TaskState.RESERVED,
        TaskState.TRANSMITTING,
        TaskState.RUNNING,
    }
    UNADMITTED = {
        TaskState.ARRIVED,
        TaskState.QUEUED,
        TaskState.PENDING_UNCOMMITTED,
    }
    ACCEPTED_ACTIVE = {
        TaskState.RESERVED,
        TaskState.TRANSMITTING,
        TaskState.RUNNING,
    }

    def __init__(
        self,
        scheduler: V1Scheduler,
        time_converter: TimeConverter,
        evaluation_safety_cap: int,
        metadata_context: Optional[Mapping[str, object]] = None,
    ):
        if scheduler.metrics_ledger is None:
            raise ValueError("formal evaluation requires a metrics ledger")
        if (
            isinstance(evaluation_safety_cap, bool)
            or not isinstance(evaluation_safety_cap, int)
            or evaluation_safety_cap <= 0
        ):
            raise ValueError("evaluation_safety_cap must be a positive integer")
        self.scheduler = scheduler
        self.time_converter = time_converter
        self.safety_cap = evaluation_safety_cap
        self.metadata_context = dict(metadata_context or {})

    def run_frozen_policy(
        self,
        tasks: Iterable[TaskSpec],
        *,
        arrival_cutoff_sim: float,
        evaluation_start_sim: float = 0.0,
        seed: int = 0,
    ) -> EvaluationReport:
        start = finite_number("evaluation_start_sim", evaluation_start_sim)
        cutoff = finite_number("arrival_cutoff_sim", arrival_cutoff_sim)
        if cutoff < start:
            raise ValueError("arrival cutoff cannot precede evaluation start")
        trace = tuple(sorted(
            (
                task for task in tasks
                if start <= task.arrival_time_sim < cutoff
            ),
            key=lambda item: (item.arrival_time_sim, item.task_id),
        ))
        pending_arrivals = list(trace)
        results = []
        accepted = set()
        phase_counts = {"arrival": 0, "admission_settlement": 0, "execution_drain": 0}
        current = start
        last_run_time = None

        def run_batch(time_sim, arrivals=(), phase="arrival"):
            nonlocal current, last_run_time
            if len(results) >= self.safety_cap:
                return False
            result = self.scheduler.run_cycle(time_sim, arrivals=arrivals)
            violations = scan_scheduler_invariants(self.scheduler, result)
            if violations:
                detail = "; ".join(
                    f"{item.invariant_id}:{item.detail}" for item in violations
                )
                raise RuntimeError(f"v1.0 invariant gate failed: {detail}")
            results.append(result)
            phase_counts[phase] += 1
            current = time_sim
            last_run_time = time_sim
            for decision in result.decisions:
                if decision.status == "RESERVED":
                    accepted.add(decision.task_id)
            return True

        # Phase 1: arrivals before the frozen cutoff plus all physical events.
        while current < cutoff or pending_arrivals:
            next_arrival = (
                pending_arrivals[0].arrival_time_sim
                if pending_arrivals else None
            )
            next_event = self.scheduler.event_engine.next_event_time_sim
            choices = [cutoff]
            if next_arrival is not None and next_arrival < cutoff:
                choices.append(next_arrival)
            if next_event is not None and next_event <= cutoff:
                choices.append(next_event)
            next_time = min(value for value in choices if value >= current - 1e-12)
            arrivals_now = []
            while (
                pending_arrivals
                and abs(pending_arrivals[0].arrival_time_sim - next_time) <= 1e-12
                and next_time < cutoff
            ):
                arrivals_now.append(pending_arrivals.pop(0))
            if not run_batch(next_time, arrivals_now, "arrival"):
                return self._invalid(start, cutoff, seed, current, results, phase_counts)
            if next_time >= cutoff - 1e-12:
                break

        if last_run_time is None or last_run_time < cutoff - 1e-12:
            if not run_batch(cutoff, (), "arrival"):
                return self._invalid(start, cutoff, seed, current, results, phase_counts)

        # Phase 2: settle every arrived task's admission outcome.
        while self._task_ids_in_states(self.UNADMITTED):
            deadlines = [
                self.scheduler.state_machine.task_spec(task_id).absolute_latest_start_sim
                for task_id in self._task_ids_in_states(self.UNADMITTED)
                if self.scheduler.state_machine.runtime(task_id).state
                in {TaskState.QUEUED, TaskState.PENDING_UNCOMMITTED}
            ]
            next_event = self.scheduler.event_engine.next_event_time_sim
            choices = [value for value in deadlines if value >= current - 1e-12]
            if next_event is not None and next_event >= current - 1e-12:
                choices.append(next_event)
            if not choices:
                return self._invalid(start, cutoff, seed, current, results, phase_counts)
            next_time = min(choices)
            if not run_batch(next_time, (), "admission_settlement"):
                return self._invalid(start, cutoff, seed, current, results, phase_counts)

        # Phase 3: no arrivals/admission work; drain accepted reservations.
        while self._task_ids_in_states(self.ACCEPTED_ACTIVE):
            next_event = self.scheduler.event_engine.next_event_time_sim
            if next_event is None or next_event < current - 1e-12:
                return self._invalid(start, cutoff, seed, current, results, phase_counts)
            if not run_batch(next_event, (), "execution_drain"):
                return self._invalid(start, cutoff, seed, current, results, phase_counts)

        unsettled = self._task_ids_in_states(self.NONTERMINAL)
        if unsettled:
            return self._invalid(start, cutoff, seed, current, results, phase_counts)
        interval_end = current
        if interval_end <= start:
            interval_end = cutoff
        accounting = self.scheduler.finalize_metrics_after_full_settlement(
            accounting_interval=TimeInterval(start, interval_end)
            if interval_end > start else None
        )
        outcomes = tuple(
            TaskOutcome(
                task,
                self.scheduler.state_machine.runtime(task.task_id).state,
                self.calendar_reservation_for(task.task_id),
            )
            for task in trace
        )
        decision_records = tuple(
            decision
            for cycle_result in results
            for decision in cycle_result.decisions
        )
        metrics = build_seed_metrics(
            outcomes,
            accepted,
            self.scheduler.state_machine.count_by_state(),
            accounting,
            self.time_converter,
            decision_records,
        )
        task_records = self._task_records(trace, accounting)
        return EvaluationReport(
            EvaluationStatus.VALID,
            self._metadata(start, cutoff, seed, current),
            metrics,
            (),
            tuple(results),
            phase_counts,
            task_records,
            decision_records,
            accounting,
        )

    def calendar_reservation_for(self, task_id: str):
        runtime = self.scheduler.state_machine.runtime(task_id)
        if runtime.reservation_id is None:
            return None
        return self.scheduler.calendar.get_reservation(runtime.reservation_id)

    def _task_ids_in_states(self, states):
        return tuple(
            task_id
            for task_id in self.scheduler.state_machine.task_ids
            if self.scheduler.state_machine.runtime(task_id).state in states
        )

    def _metadata(self, start, cutoff, seed, final_time):
        values = dict(
            requirements_version="1.0",
            algorithm_version="1.0",
            task_schema_version="1.0",
            candidate_schema_version="1.0",
            model_schema_version="1.0",
            metric_schema_version="1.0",
            aggregation_schema_version="1.0",
            seed=seed,
            candidate_mode=self.scheduler.candidate_generator.candidate_mode.value,
            arrival_cutoff_sim=cutoff,
            evaluation_start_sim=start,
            final_settlement_time_sim=final_time,
            evaluation_safety_cap=self.safety_cap,
        )
        values.update(self.metadata_context)
        return EvaluationMetadata(**values)

    def _invalid(self, start, cutoff, seed, current, results, phase_counts):
        unsettled = self._task_ids_in_states(self.NONTERMINAL)
        return EvaluationReport(
            EvaluationStatus.INVALID_INCOMPLETE_SETTLEMENT,
            self._metadata(start, cutoff, seed, current),
            None,
            unsettled,
            tuple(results),
            phase_counts,
            (),
            tuple(
                decision
                for cycle_result in results
                for decision in cycle_result.decisions
            ),
            None,
        )

    def _task_records(self, tasks, accounting):
        realized_by_task = {
            record.task_id: record for record in accounting.task_records
        }
        records = []
        for task in sorted(tasks, key=lambda item: item.task_id):
            runtime = self.scheduler.state_machine.runtime(task.task_id)
            reservation = self.calendar_reservation_for(task.task_id)
            candidate = self.scheduler.committed_candidate(task.task_id)
            realized = realized_by_task.get(task.task_id)
            if candidate is None:
                timeline = (None,) * 13
                marginal_cost = None
                marginal_green = None
            else:
                timeline = (
                    candidate.decision_time_sim,
                    None if candidate.path.is_local else candidate.transmission_start_sim,
                    None if candidate.path.is_local else candidate.transmission_end_sim,
                    candidate.compute_start_sim,
                    candidate.compute_end_sim,
                    candidate.target_node,
                    candidate.path.path_id,
                    candidate.scheduler_queue_delay_sim,
                    candidate.earliest_feasibility_lead_sim,
                    candidate.active_wait_sim,
                    candidate.reservation_lead_sim,
                    candidate.start_delay_sim,
                    (
                        candidate.compute_end_sim - task.arrival_time_sim
                        if runtime.state is TaskState.COMPLETED else None
                    ),
                )
                marginal_cost = candidate.estimated_candidate_marginal_system_cost_yuan
                marginal_green = candidate.estimated_candidate_marginal_green_energy_mwh
            errors = None
            if candidate is not None and realized is not None:
                errors = {
                    "cost_yuan": (
                        realized.task_attributed_cost_yuan
                        - candidate.estimated_candidate_marginal_system_cost_yuan
                    ),
                    "green_coverage": (
                        realized.green_coverage
                        - candidate.estimated_green_coverage
                    ),
                    "green_energy_mwh": (
                        realized.task_attributed_green_energy_mwh
                        - candidate.estimated_candidate_marginal_green_energy_mwh
                    ),
                }
            records.append(TaskEvaluationRecord(
                task.task_id,
                runtime.state.value,
                runtime.terminal_reason,
                task.arrival_time_sim,
                *timeline,
                task.cpu_work_cpu_hours(self.time_converter),
                None if realized is None else realized.task_energy_mwh,
                None if realized is None else realized.task_attributed_cost_yuan,
                None if realized is None else realized.task_attributed_green_energy_mwh,
                marginal_cost,
                marginal_green,
                errors,
            ))
        return tuple(records)
