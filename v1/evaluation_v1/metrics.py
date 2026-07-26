"""Strict v1.0 per-seed metrics with explicit undefined semantics."""

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Optional, Tuple

from v1.accounting.energy import AccountingReport
from v1.domain.models import MetricValue, SlaType, TaskSpec, TaskState
from v1.domain.reservations import Reservation
from v1.domain.units import TimeConverter


def ratio_metric(numerator: float, denominator: float, reason: str) -> MetricValue:
    if denominator == 0.0:
        return MetricValue.not_applicable(reason)
    return MetricValue.valid(
        numerator / denominator,
        numerator=numerator,
        denominator=denominator,
    )


def linear_percentile(values: Iterable[float], percentile: float) -> MetricValue:
    items = sorted(float(value) for value in values)
    if not items:
        return MetricValue.not_applicable("empty sample")
    if not 0.0 <= percentile <= 100.0:
        return MetricValue.invalid("percentile outside [0,100]")
    position = (len(items) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        value = items[lower]
    else:
        fraction = position - lower
        value = items[lower] * (1.0 - fraction) + items[upper] * fraction
    return MetricValue.valid(value)


@dataclass(frozen=True)
class ActiveWaitMetrics:
    count: int
    mean_active_wait_sim: MetricValue
    p95_active_wait_sim: MetricValue
    positive_benefit_rate: MetricValue


def summarize_active_wait(records: Iterable[object]) -> ActiveWaitMetrics:
    """Return conditional active-wait metrics with explicit empty semantics."""

    waits = []
    positives = 0
    for record in records:
        if isinstance(record, Mapping):
            wait = record.get("active_wait_sim")
            positive = record.get("benefit_positive")
        else:
            wait = getattr(record, "active_wait_sim", None)
            positive = getattr(record, "benefit_positive", None)
        if wait is None or float(wait) <= 0.0:
            continue
        value = float(wait)
        if not math.isfinite(value):
            invalid = MetricValue.invalid("non-finite active wait")
            return ActiveWaitMetrics(0, invalid, invalid, invalid)
        waits.append(value)
        positives += int(positive is True)
    if not waits:
        not_applicable = MetricValue.not_applicable("no active-wait tasks")
        return ActiveWaitMetrics(0, not_applicable, not_applicable, not_applicable)
    return ActiveWaitMetrics(
        len(waits),
        MetricValue.valid(sum(waits) / len(waits)),
        linear_percentile(waits, 95.0),
        ratio_metric(positives, len(waits), "no active-wait tasks"),
    )


@dataclass(frozen=True)
class TaskOutcome:
    task: TaskSpec
    final_state: TaskState
    reservation: Optional[Reservation]

    @property
    def start_delay_sim(self) -> Optional[float]:
        if self.reservation is None:
            return None
        return (
            self.reservation.compute_interval_sim.start_sim
            - self.task.arrival_time_sim
        )


@dataclass(frozen=True)
class SlaMetrics:
    sla_type: SlaType
    count: int
    preferred_on_time_rate: MetricValue
    acceptable_tardy_rate: MetricValue
    expired_rate: MetricValue
    start_delay_p50_sim: MetricValue
    start_delay_p95_sim: MetricValue
    preferred_start_tardiness_p50: MetricValue
    preferred_start_tardiness_p95: MetricValue


def summarize_sla(outcomes: Iterable[TaskOutcome]) -> Mapping[SlaType, SlaMetrics]:
    items = tuple(outcomes)
    result = {}
    for sla_type in SlaType:
        group = [item for item in items if item.task.sla_type is sla_type]
        count = len(group)
        expired = sum(item.final_state is TaskState.EXPIRED for item in group)
        start_delays = [
            item.start_delay_sim
            for item in group
            if item.start_delay_sim is not None
        ]
        if sla_type is SlaType.HARD:
            preferred = MetricValue.not_applicable("Hard has no preferred start limit")
            tardy = MetricValue.not_applicable("Hard has no tardiness region")
            tardiness_p50 = MetricValue.not_applicable("Hard tardiness is not applicable")
            tardiness_p95 = MetricValue.not_applicable("Hard tardiness is not applicable")
        else:
            on_time = 0
            acceptable_tardy = 0
            tardiness_values = []
            for item in group:
                delay = item.start_delay_sim
                if delay is None:
                    continue
                if delay <= item.task.preferred_start_limit_sim + 1e-12:
                    on_time += 1
                    tardiness_values.append(0.0)
                else:
                    acceptable_tardy += 1
                    tardiness_values.append(
                        (delay - item.task.preferred_start_limit_sim)
                        / (
                            item.task.latest_start_limit_sim
                            - item.task.preferred_start_limit_sim
                        )
                    )
            preferred = ratio_metric(on_time, count, "empty SLA subgroup")
            tardy = ratio_metric(acceptable_tardy, count, "empty SLA subgroup")
            tardiness_p50 = linear_percentile(tardiness_values, 50.0)
            tardiness_p95 = linear_percentile(tardiness_values, 95.0)
        result[sla_type] = SlaMetrics(
            sla_type,
            count,
            preferred,
            tardy,
            ratio_metric(expired, count, "empty SLA subgroup"),
            linear_percentile(start_delays, 50.0),
            linear_percentile(start_delays, 95.0),
            tardiness_p50,
            tardiness_p95,
        )
    return result


@dataclass(frozen=True)
class SeedMetrics:
    arrival_count: int
    reserved_ever_count: int
    rejected_count: int
    expired_count: int
    completed_count: int
    failed_count: int
    final_state_counts: Mapping[TaskState, int]
    acceptance_rate: MetricValue
    completion_rate: MetricValue
    reservation_reliability: MetricValue
    total_economic_cost_yuan: float
    completed_cpu_hours: float
    arrived_requested_cpu_hours: float
    cost_yuan_per_completed_cpu_hour: MetricValue
    cost_yuan_per_arrived_cpu_hour: MetricValue
    completed_task_green_coverage: MetricValue
    system_green_absorption_rate: MetricValue
    sla_metrics: Mapping[SlaType, SlaMetrics]
    active_wait_metrics: ActiveWaitMetrics


def build_seed_metrics(
    outcomes: Iterable[TaskOutcome],
    reserved_ever_task_ids: Iterable[str],
    state_counts: Mapping[TaskState, int],
    accounting_report: AccountingReport,
    time_converter: TimeConverter,
    active_wait_records: Iterable[object] = (),
) -> SeedMetrics:
    items = tuple(outcomes)
    accepted = set(reserved_ever_task_ids)
    arrival_count = len(items)
    completed = [item for item in items if item.final_state is TaskState.COMPLETED]
    completed_cpu_hours = sum(
        item.task.cpu_work_cpu_hours(time_converter) for item in completed
    )
    arrived_cpu_hours = sum(
        item.task.cpu_work_cpu_hours(time_converter) for item in items
    )
    total_cost = accounting_report.total_task_attributed_cost_yuan
    completed_energy = sum(
        record.task_energy_mwh for record in accounting_report.task_records
    )
    completed_green = accounting_report.total_task_attributed_green_energy_mwh
    return SeedMetrics(
        arrival_count,
        len(accepted),
        state_counts[TaskState.REJECTED],
        state_counts[TaskState.EXPIRED],
        state_counts[TaskState.COMPLETED],
        state_counts[TaskState.FAILED],
        dict(state_counts),
        ratio_metric(len(accepted), arrival_count, "zero arrivals"),
        ratio_metric(len(completed), arrival_count, "zero arrivals"),
        ratio_metric(len(completed), len(accepted), "zero accepted reservations"),
        total_cost,
        completed_cpu_hours,
        arrived_cpu_hours,
        ratio_metric(total_cost, completed_cpu_hours, "zero completed CPU hours"),
        ratio_metric(total_cost, arrived_cpu_hours, "zero arrived CPU hours"),
        ratio_metric(completed_green, completed_energy, "zero completed task energy"),
        accounting_report.system_green_absorption_rate,
        summarize_sla(items),
        summarize_active_wait(active_wait_records),
    )
