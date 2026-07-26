"""Complete v1.0 node-time-path candidate enumeration.

The compatibility ``generate_complete`` API materializes the result for tests
and small diagnostics.  Training and formal runtime can use
``prepare_complete_stream`` to scan the complete grid and then consume
candidates without retaining the complete candidate object set in memory.
"""

from dataclasses import dataclass
import math
import time
from typing import Callable, Dict, Iterable, Iterator, NamedTuple, Optional

import numpy as np

from v1.domain.candidates import (
    CANDIDATE_SCHEMA_VERSION,
    Candidate,
    CandidateGenerationStatus,
    CandidateMode,
    CandidateSetResult,
    deterministic_candidate_id_v1_fields,
)
from v1.domain.models import SlaType, TaskSpec
from v1.domain.reservations import TimeInterval
from v1.domain.units import finite_number, positive_finite

from .path_provider import StaticPathProvider
from .resource_calendar import ReservationCalendar, ReservationSnapshot
from .transmission import TransmissionModel


MetricEvaluator = Callable[..., Dict[str, float]]
PREPARE_FEASIBILITY_CHUNK_SIZE = 65536


def _grid_tick_ceil(value: float, cycle: float, tolerance: float) -> int:
    quotient = value / cycle
    nearest = round(quotient)
    if abs(quotient - nearest) <= tolerance:
        return int(nearest)
    return int(math.ceil(quotient))


def _grid_tick_floor(value: float, cycle: float, tolerance: float) -> int:
    quotient = value / cycle
    nearest = round(quotient)
    if abs(quotient - nearest) <= tolerance:
        return int(nearest)
    return int(math.floor(quotient))


def _time_grid_tick_bounds(earliest, latest, cycle, tolerance):
    return (
        _grid_tick_ceil(earliest, cycle, tolerance),
        _grid_tick_floor(latest, cycle, tolerance),
    )


def complete_time_grid(
    earliest_start_sim: float,
    latest_start_sim: float,
    scheduling_cycle_sim: float,
    tolerance: float = 1e-9,
) -> tuple:
    earliest = finite_number("earliest_start_sim", earliest_start_sim)
    latest = finite_number("latest_start_sim", latest_start_sim)
    cycle = positive_finite("scheduling_cycle_sim", scheduling_cycle_sim)
    tolerance = positive_finite("time_tolerance", tolerance)
    first_tick, last_tick = _time_grid_tick_bounds(
        earliest, latest, cycle, tolerance
    )
    if first_tick > last_tick:
        return ()
    return tuple(tick * cycle for tick in range(first_tick, last_tick + 1))


@dataclass(frozen=True)
class CandidateGenerationContext:
    task: TaskSpec
    decision_time_sim: float
    reservation_snapshot: ReservationSnapshot
    forecast_version: str
    forecast_covered_until_sim: Optional[float]
    earliest_compute_start_sim: float


class CandidateRecord(NamedTuple):
    """Complete candidate identity/physics before expensive metric realization."""

    candidate_id: str
    target_node: str
    path: object
    transmission_start_sim: float
    compute_start_sim: float
    compute_end_sim: float
    preferred_start_tardiness_ratio: float
    preferred_start_tardiness_applicable: bool
    projected_node_utilization: float
    projected_path_peak_utilization: float
    capacity_margin: float


class CandidateFeatureRecordChunk(NamedTuple):
    features: object
    target_node: str
    path: object
    transmission_starts: object
    starts: object
    ends: object
    tardiness: object
    tardiness_applicable: object
    node_utilization: object
    path_utilization: object
    capacity_margin: object


@dataclass(frozen=True)
class CompleteCandidateStream:
    generator: "CandidateGenerator"
    context: CandidateGenerationContext
    status: CandidateGenerationStatus
    theoretical_slot_count: int
    feasible_candidate_count: int
    reason: str = ""
    metric_evaluator: Optional[MetricEvaluator] = None

    @property
    def candidate_mode(self):
        return CandidateMode.COMPLETE

    @property
    def earliest_compute_start_sim(self):
        return self.context.earliest_compute_start_sim

    def iter_candidates(self) -> Iterator[Candidate]:
        if self.status is not CandidateGenerationStatus.OK:
            return iter(())
        return self.generator.iter_context_candidates(
            self.context,
            metric_evaluator=self.metric_evaluator,
        )

    def iter_candidate_records(self) -> Iterator[CandidateRecord]:
        if self.status is not CandidateGenerationStatus.OK:
            return iter(())
        return self.generator.iter_context_candidate_records(self.context)

    def materialize_record(self, record: CandidateRecord) -> Candidate:
        return self.generator.materialize_context_candidate(
            self.context,
            record,
            metric_evaluator=self.metric_evaluator,
        )


class CandidateGenerator:
    def __init__(
        self,
        compute_nodes: Iterable[str],
        scheduling_cycle_sim: float,
        path_provider: StaticPathProvider,
        transmission_model: TransmissionModel,
        calendar: ReservationCalendar,
        time_tolerance: float = 1e-9,
    ):
        self.compute_nodes = tuple(str(node) for node in compute_nodes)
        if not self.compute_nodes:
            raise ValueError("compute_nodes cannot be empty")
        self.scheduling_cycle_sim = positive_finite(
            "scheduling_cycle_sim", scheduling_cycle_sim
        )
        self.path_provider = path_provider
        self.transmission_model = transmission_model
        self.calendar = calendar
        self.time_tolerance = positive_finite("time_tolerance", time_tolerance)
        self.profiler = None

    def is_statically_serviceable(self, task: TaskSpec) -> bool:
        for target_node in self.compute_nodes:
            capacity = self.calendar.node_capacity(target_node)
            if capacity is None or task.cpu_demand > capacity + self.time_tolerance:
                continue
            if self.path_provider.candidate_paths(
                task.source_node, target_node, task.bandwidth_demand_mbps
            ):
                return True
        return False

    def _declared_path_grids(self, task, decision_time):
        for target_node in self.compute_nodes:
            paths = self.path_provider.candidate_paths(
                task.source_node,
                target_node,
                task.bandwidth_demand_mbps,
            )
            for path in paths:
                duration = self.transmission_model.duration(task, path)
                if not duration.static_path_feasible:
                    continue
                first, last = _time_grid_tick_bounds(
                    decision_time + duration.total_sim,
                    task.absolute_latest_start_sim,
                    self.scheduling_cycle_sim,
                    self.time_tolerance,
                )
                yield target_node, path, duration, first, last

    def theoretical_slot_count(self, task, decision_time_sim):
        decision_time = finite_number("decision_time_sim", decision_time_sim)
        return sum(
            max(0, last - first + 1)
            for _, _, _, first, last in self._declared_path_grids(task, decision_time)
        )

    def _feasible_item(
        self,
        context,
        target_node,
        path,
        duration,
        compute_start,
        *,
        resources_unallocated=None,
        node_capacity=None,
    ):
        task = context.task
        compute_end = compute_start + task.execution_duration_sim
        forecast_limit = context.forecast_covered_until_sim
        if forecast_limit is not None and compute_end > forecast_limit + self.time_tolerance:
            return None, "forecast"
        schedule_started = time.perf_counter()
        if path.is_local:
            transmission_start = compute_start
            transmission_interval = None
        else:
            transmission_start = compute_start - duration.total_sim
            if transmission_start < context.decision_time_sim - 1e-12:
                return None, "physical"
            transmission_interval = TimeInterval(
                transmission_start, compute_start
            )
        if self.profiler is not None:
            self.profiler.add(
                "candidate_transmission_schedule_seconds",
                time.perf_counter() - schedule_started,
            )
        if resources_unallocated is None:
            resources_unallocated = self.calendar.resources_unallocated(
                context.reservation_snapshot,
                target_node,
                path,
            )
        if resources_unallocated:
            capacity = (
                self.calendar.node_capacity(target_node)
                if node_capacity is None else node_capacity
            )
            if capacity is None or task.cpu_demand > capacity + self.time_tolerance:
                return None, "physical"
            node_util = task.cpu_demand / capacity
            path_util = (
                0.0
                if path.is_local
                else task.bandwidth_demand_mbps / path.static_bottleneck_mbps
            )
            cpu_result = None
            path_result = None
        else:
            node_util = None
            path_util = None
        cpu_started = time.perf_counter()
        if not resources_unallocated:
            compute_interval = TimeInterval(compute_start, compute_end)
            cpu_result = self.calendar.cpu_feasible(
                context.reservation_snapshot,
                target_node,
                compute_interval,
                task.cpu_demand,
            )
        if self.profiler is not None:
            self.profiler.add(
                "candidate_cpu_feasibility_seconds",
                time.perf_counter() - cpu_started,
            )
        if cpu_result is not None and not cpu_result.feasible:
            return None, "physical"
        path_started = time.perf_counter()
        if not resources_unallocated:
            path_result = self.calendar.path_feasible(
                context.reservation_snapshot,
                path,
                transmission_interval,
                task.bandwidth_demand_mbps,
            )
        if self.profiler is not None:
            self.profiler.add(
                "candidate_path_feasibility_seconds",
                time.perf_counter() - path_started,
            )
        item_started = time.perf_counter()
        if (
            (path_result is not None and not path_result.feasible)
            or compute_start < task.arrival_time_sim
            or compute_start > task.absolute_latest_start_sim
        ):
            if self.profiler is not None:
                self.profiler.add(
                    "candidate_sla_and_item_seconds",
                    time.perf_counter() - item_started,
                )
            return None, "physical"
        if task.sla_type is SlaType.HARD:
            tardiness_value = 0.0
            tardiness_applicable = False
        else:
            start_delay = compute_start - task.arrival_time_sim
            denominator = (
                task.latest_start_limit_sim - task.preferred_start_limit_sim
            )
            tardiness_value = min(
                1.0,
                max(
                    0.0,
                    (start_delay - task.preferred_start_limit_sim) / denominator,
                ),
            )
            tardiness_applicable = True
        if not resources_unallocated:
            node_util = cpu_result.projected_peak / cpu_result.capacity
            path_util = (
                0.0
                if path.is_local or path_result.capacity <= 0.0
                else path_result.projected_peak / path_result.capacity
            )
        result = {
            "target_node": target_node,
            "path": path,
            "transmission_start": transmission_start,
            "compute_start": compute_start,
            "compute_end": compute_end,
            "tardiness": tardiness_value,
            "tardiness_applicable": tardiness_applicable,
            "node_util": node_util,
            "path_util": path_util,
            "capacity_margin": min(
                1.0 - node_util,
                1.0 - path_util if not path.is_local else 1.0,
            ),
        }
        if self.profiler is not None:
            self.profiler.add(
                "candidate_sla_and_item_seconds",
                time.perf_counter() - item_started,
            )
        return result, None

    def prepare_complete_stream(
        self,
        task: TaskSpec,
        decision_time_sim: float,
        reservation_snapshot: Optional[ReservationSnapshot] = None,
        forecast_version: str = "perfect-v1",
        forecast_covered_until_sim: Optional[float] = None,
        metric_evaluator: Optional[MetricEvaluator] = None,
    ) -> CompleteCandidateStream:
        decision_time = finite_number("decision_time_sim", decision_time_sim)
        snapshot = reservation_snapshot or self.calendar.snapshot()
        if not isinstance(forecast_version, str) or not forecast_version:
            raise ValueError("forecast_version must be non-empty")
        forecast_limit = (
            None
            if forecast_covered_until_sim is None
            else finite_number(
                "forecast_covered_until_sim", forecast_covered_until_sim
            )
        )
        base_context = CandidateGenerationContext(
            task,
            decision_time,
            snapshot,
            forecast_version,
            forecast_limit,
            decision_time,
        )
        if decision_time > task.absolute_latest_start_sim + self.time_tolerance:
            return CompleteCandidateStream(
                self,
                base_context,
                CandidateGenerationStatus.EXPIRED_BEFORE_DECISION,
                0,
                0,
                "decision time is after absolute latest start",
                metric_evaluator,
            )

        theoretical_slots = 0
        forecast_rejected = 0
        feasible_count = 0
        earliest = None
        for target_node, path, duration, first, last in self._declared_path_grids(
            task, decision_time
        ):
            if first > last:
                continue
            theoretical_slots += last - first + 1
            forecast_last = last
            if forecast_limit is not None:
                forecast_last = min(
                    last,
                    _grid_tick_floor(
                        forecast_limit - task.execution_duration_sim,
                        self.scheduling_cycle_sim,
                        self.time_tolerance,
                    ),
                )
                forecast_rejected += max(0, last - max(first - 1, forecast_last))
            capacity = self.calendar.node_capacity(target_node)
            resources_unallocated = self.calendar.resources_unallocated(
                snapshot,
                target_node,
                path,
            )
            if (
                resources_unallocated
                and capacity is not None
                and task.cpu_demand <= capacity + self.time_tolerance
            ):
                grid_count = max(0, forecast_last - first + 1)
                feasible_count += grid_count
                if grid_count and (
                    earliest is None
                    or first * self.scheduling_cycle_sim < earliest
                ):
                    earliest = first * self.scheduling_cycle_sim
                continue
            feasibility_started = time.perf_counter()
            eligible_last = min(last, forecast_last)
            for chunk_first in range(
                first,
                eligible_last + 1,
                PREPARE_FEASIBILITY_CHUNK_SIZE,
            ):
                chunk_last = min(
                    eligible_last + 1,
                    chunk_first + PREPARE_FEASIBILITY_CHUNK_SIZE,
                )
                ticks = np.arange(
                    chunk_first, chunk_last, dtype=np.int64
                )
                starts = (
                    ticks.astype(np.float64)
                    * self.scheduling_cycle_sim
                )
                ends = starts + task.execution_duration_sim
                transmission_starts = (
                    starts
                    if path.is_local
                    else starts - duration.total_sim
                )
                temporal_feasible = (
                    (
                        transmission_starts
                        >= decision_time - 1e-12
                    )
                    & (starts >= task.arrival_time_sim)
                    & (
                        starts
                        <= task.absolute_latest_start_sim
                    )
                )

                cpu_started = time.perf_counter()
                cpu = self.calendar.cpu_feasible_many(
                    snapshot,
                    target_node,
                    starts,
                    ends,
                    task.cpu_demand,
                )
                if self.profiler is not None:
                    self.profiler.add(
                        "candidate_cpu_feasibility_seconds",
                        time.perf_counter() - cpu_started,
                    )

                path_started = time.perf_counter()
                path_result = self.calendar.path_feasible_many(
                    snapshot,
                    path,
                    transmission_starts,
                    starts,
                    task.bandwidth_demand_mbps,
                )
                if self.profiler is not None:
                    self.profiler.add(
                        "candidate_path_feasibility_seconds",
                        time.perf_counter() - path_started,
                    )

                feasible = (
                    temporal_feasible
                    & cpu["feasible"]
                    & path_result["feasible"]
                )
                positions = np.flatnonzero(feasible)
                feasible_count += int(positions.size)
                if positions.size:
                    grid_earliest = float(starts[positions[0]])
                    if (
                        earliest is None
                        or grid_earliest < earliest
                    ):
                        earliest = grid_earliest
            if self.profiler is not None:
                self.profiler.add(
                    "candidate_feasibility_prepare_seconds",
                    time.perf_counter() - feasibility_started,
                )
                self.profiler.increment(
                    "prepare_feasibility_check_count", last - first + 1
                )

        if feasible_count == 0:
            status = (
                CandidateGenerationStatus.FORECAST_NOT_COVERED
                if forecast_rejected > 0
                else CandidateGenerationStatus.EMPTY_PHYSICAL
            )
            return CompleteCandidateStream(
                self,
                base_context,
                status,
                theoretical_slots,
                0,
                "forecast does not cover complete execution intervals"
                if forecast_rejected > 0
                else "complete physical enumeration is empty",
                metric_evaluator,
            )

        context = CandidateGenerationContext(
            task,
            decision_time,
            snapshot,
            forecast_version,
            forecast_limit,
            earliest,
        )
        return CompleteCandidateStream(
            self,
            context,
            CandidateGenerationStatus.OK,
            theoretical_slots,
            feasible_count,
            "",
            metric_evaluator,
        )

    def iter_context_candidate_records(
        self,
        context,
        *,
        include_candidate_id=True,
    ):
        task = context.task
        for target_node, path, duration, first, last in self._declared_path_grids(
            task, context.decision_time_sim
        ):
            resources_unallocated = self.calendar.resources_unallocated(
                context.reservation_snapshot,
                target_node,
                path,
            )
            capacity = self.calendar.node_capacity(target_node)
            if resources_unallocated:
                # With no allocations on either the target node or the path,
                # interval feasibility is invariant over the whole declared
                # grid.  Avoid rebuilding TimeInterval objects and result
                # dictionaries for every tick; retain every boundary check and
                # the canonical record/ID order of the general path.
                if (
                    capacity is None
                    or task.cpu_demand > capacity + self.time_tolerance
                ):
                    if self.profiler is not None:
                        self.profiler.increment(
                            "stream_feasibility_check_count",
                            max(0, last - first + 1),
                        )
                    continue
                node_util = task.cpu_demand / capacity
                path_util = (
                    0.0
                    if path.is_local
                    else (
                        task.bandwidth_demand_mbps
                        / path.static_bottleneck_mbps
                    )
                )
                capacity_margin = min(
                    1.0 - node_util,
                    1.0 - path_util if not path.is_local else 1.0,
                )
                forecast_limit = context.forecast_covered_until_sim
                for tick in range(first, last + 1):
                    feasibility_started = (
                        time.perf_counter()
                        if self.profiler is not None else None
                    )
                    compute_start = tick * self.scheduling_cycle_sim
                    compute_end = (
                        compute_start + task.execution_duration_sim
                    )
                    transmission_start = (
                        compute_start
                        if path.is_local
                        else compute_start - duration.total_sim
                    )
                    feasible = not (
                        (
                            forecast_limit is not None
                            and compute_end
                            > forecast_limit + self.time_tolerance
                        )
                        or transmission_start
                        < context.decision_time_sim - 1e-12
                        or compute_start < task.arrival_time_sim
                        or compute_start
                        > task.absolute_latest_start_sim
                    )
                    if task.sla_type is SlaType.HARD:
                        tardiness_value = 0.0
                        tardiness_applicable = False
                    else:
                        start_delay = (
                            compute_start - task.arrival_time_sim
                        )
                        denominator = (
                            task.latest_start_limit_sim
                            - task.preferred_start_limit_sim
                        )
                        tardiness_value = min(
                            1.0,
                            max(
                                0.0,
                                (
                                    start_delay
                                    - task.preferred_start_limit_sim
                                ) / denominator,
                            ),
                        )
                        tardiness_applicable = True
                    if self.profiler is not None:
                        self.profiler.add(
                            "candidate_feasibility_stream_seconds",
                            time.perf_counter() - feasibility_started,
                        )
                        self.profiler.increment(
                            "stream_feasibility_check_count"
                        )
                    if not feasible:
                        continue
                    if include_candidate_id:
                        identifier_started = time.perf_counter()
                        candidate_id = deterministic_candidate_id_v1_fields(
                            compute_end=compute_end,
                            compute_start=compute_start,
                            forecast_version=context.forecast_version,
                            mode=CandidateMode.COMPLETE.value,
                            node=target_node,
                            path=path.path_id,
                            reservation_version=(
                                context.reservation_snapshot
                                .reservation_version
                            ),
                            schema=CANDIDATE_SCHEMA_VERSION,
                            task=task.task_id,
                            transmission_end=compute_start,
                            transmission_start=transmission_start,
                        )
                        if self.profiler is not None:
                            self.profiler.add(
                                "candidate_id_hash_seconds",
                                time.perf_counter() - identifier_started,
                            )
                    else:
                        candidate_id = ""
                    yield CandidateRecord(
                        candidate_id,
                        target_node,
                        path,
                        transmission_start,
                        compute_start,
                        compute_end,
                        tardiness_value,
                        tardiness_applicable,
                        node_util,
                        path_util,
                        capacity_margin,
                    )
                continue
            for tick in range(first, last + 1):
                compute_start = tick * self.scheduling_cycle_sim
                feasibility_started = time.perf_counter()
                item, _ = self._feasible_item(
                    context,
                    target_node,
                    path,
                    duration,
                    compute_start,
                    resources_unallocated=resources_unallocated,
                    node_capacity=capacity,
                )
                if self.profiler is not None:
                    self.profiler.add(
                        "candidate_feasibility_stream_seconds",
                        time.perf_counter() - feasibility_started,
                    )
                    self.profiler.increment("stream_feasibility_check_count")
                if item is None:
                    continue
                transmission_start = item["transmission_start"]
                if include_candidate_id:
                    identifier_started = time.perf_counter()
                    candidate_id = deterministic_candidate_id_v1_fields(
                        compute_end=item["compute_end"],
                        compute_start=compute_start,
                        forecast_version=context.forecast_version,
                        mode=CandidateMode.COMPLETE.value,
                        node=target_node,
                        path=path.path_id,
                        reservation_version=(
                            context.reservation_snapshot.reservation_version
                        ),
                        schema=CANDIDATE_SCHEMA_VERSION,
                        task=task.task_id,
                        transmission_end=compute_start,
                        transmission_start=transmission_start,
                    )
                    if self.profiler is not None:
                        self.profiler.add(
                            "candidate_id_hash_seconds",
                            time.perf_counter() - identifier_started,
                        )
                else:
                    candidate_id = ""
                yield CandidateRecord(
                    candidate_id,
                    target_node,
                    path,
                    transmission_start,
                    compute_start,
                    item["compute_end"],
                    item["tardiness"],
                    item["tardiness_applicable"],
                    item["node_util"],
                    item["path_util"],
                    item["capacity_margin"],
                )

    def sample_context_candidate_records(
        self,
        context,
        rng,
        *,
        chunk_size=65536,
    ):
        """Uniformly sample the complete feasible stream without audit hashing.

        Training exploration needs the same reservoir-sampling random calls and
        the same selected/earliest candidates as the scalar audit path, but the
        per-candidate SHA-256 evidence is not consumed by learning. Formal
        evaluation keeps using ``iter_context_candidate_records`` and therefore
        retains the frozen candidate-set hash contract.
        """

        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0
        ):
            raise ValueError("chunk_size must be a positive integer")

        task = context.task
        count = 0
        selected_record = None
        earliest_record = None

        def make_record(
            target_node,
            path,
            duration,
            start,
            end,
            transmission_start,
            node_util,
            path_util,
        ):
            if task.sla_type is SlaType.HARD:
                tardiness = 0.0
                tardiness_applicable = False
            else:
                denominator = (
                    task.latest_start_limit_sim
                    - task.preferred_start_limit_sim
                )
                tardiness = min(
                    1.0,
                    max(
                        0.0,
                        (
                            start
                            - task.arrival_time_sim
                            - task.preferred_start_limit_sim
                        )
                        / denominator,
                    ),
                )
                tardiness_applicable = True
            return CandidateRecord(
                "",
                target_node,
                path,
                transmission_start,
                start,
                end,
                tardiness,
                tardiness_applicable,
                node_util,
                path_util,
                min(
                    1.0 - node_util,
                    1.0 - path_util if not path.is_local else 1.0,
                ),
            )

        for target_node, path, duration, first, last in (
            self._declared_path_grids(
                task, context.decision_time_sim
            )
        ):
            if first > last:
                continue
            capacity = self.calendar.node_capacity(target_node)
            resources_unallocated = self.calendar.resources_unallocated(
                context.reservation_snapshot,
                target_node,
                path,
            )
            for chunk_first in range(first, last + 1, chunk_size):
                chunk_last = min(last + 1, chunk_first + chunk_size)
                feasibility_started = time.perf_counter()
                ticks = np.arange(
                    chunk_first, chunk_last, dtype=np.int64
                )
                starts = (
                    ticks.astype(np.float64)
                    * self.scheduling_cycle_sim
                )
                ends = starts + task.execution_duration_sim
                transmission_starts = (
                    starts
                    if path.is_local
                    else starts - duration.total_sim
                )
                feasible = (
                    (
                        transmission_starts
                        >= context.decision_time_sim - 1e-12
                    )
                    & (starts >= task.arrival_time_sim)
                    & (
                        starts
                        <= task.absolute_latest_start_sim
                    )
                )
                if context.forecast_covered_until_sim is not None:
                    feasible &= (
                        ends
                        <= context.forecast_covered_until_sim
                        + self.time_tolerance
                    )

                if (
                    resources_unallocated
                    and capacity is not None
                    and task.cpu_demand
                    <= capacity + self.time_tolerance
                ):
                    cpu_feasible = np.ones(
                        starts.shape, dtype=np.bool_
                    )
                    cpu_projected = np.full(
                        starts.shape,
                        task.cpu_demand,
                        dtype=np.float64,
                    )
                    cpu_capacity = capacity
                else:
                    cpu = self.calendar.cpu_feasible_many(
                        context.reservation_snapshot,
                        target_node,
                        starts,
                        ends,
                        task.cpu_demand,
                    )
                    cpu_feasible = cpu["feasible"]
                    cpu_projected = cpu["projected_peak"]
                    cpu_capacity = cpu["capacity"]
                feasible &= cpu_feasible

                if resources_unallocated:
                    path_feasible = np.ones(
                        starts.shape, dtype=np.bool_
                    )
                    path_projected = np.full(
                        starts.shape,
                        0.0
                        if path.is_local
                        else task.bandwidth_demand_mbps,
                        dtype=np.float64,
                    )
                    path_capacity = np.full(
                        starts.shape,
                        0.0
                        if path.is_local
                        else path.static_bottleneck_mbps,
                        dtype=np.float64,
                    )
                else:
                    path_result = self.calendar.path_feasible_many(
                        context.reservation_snapshot,
                        path,
                        transmission_starts,
                        starts,
                        task.bandwidth_demand_mbps,
                    )
                    path_feasible = path_result["feasible"]
                    path_projected = path_result["projected_peak"]
                    path_capacity = path_result["capacity"]
                feasible &= path_feasible
                positions = np.flatnonzero(feasible)
                if self.profiler is not None:
                    self.profiler.add(
                        "candidate_feasibility_stream_seconds",
                        time.perf_counter() - feasibility_started,
                    )
                    self.profiler.increment(
                        "stream_feasibility_check_count",
                        len(starts),
                    )
                if positions.size == 0:
                    continue

                node_utils = (
                    cpu_projected[positions] / cpu_capacity
                )
                if path.is_local:
                    path_utils = np.zeros(
                        positions.shape, dtype=np.float64
                    )
                else:
                    path_utils = (
                        path_projected[positions]
                        / path_capacity[positions]
                    )

                first_position = int(positions[0])
                contender = make_record(
                    target_node,
                    path,
                    duration,
                    float(starts[first_position]),
                    float(ends[first_position]),
                    float(transmission_starts[first_position]),
                    float(node_utils[0]),
                    float(path_utils[0]),
                )
                if earliest_record is None or (
                    contender.compute_start_sim,
                    contender.target_node,
                    contender.path.path_id,
                ) < (
                    earliest_record.compute_start_sim,
                    earliest_record.target_node,
                    earliest_record.path.path_id,
                ):
                    earliest_record = contender

                for local_index, position in enumerate(positions):
                    count += 1
                    if rng.randrange(count) == 0:
                        integer_position = int(position)
                        selected_record = make_record(
                            target_node,
                            path,
                            duration,
                            float(starts[integer_position]),
                            float(ends[integer_position]),
                            float(
                                transmission_starts[
                                    integer_position
                                ]
                            ),
                            float(node_utils[local_index]),
                            float(path_utils[local_index]),
                        )

        def attach_candidate_id(record):
            identifier_started = time.perf_counter()
            candidate_id = deterministic_candidate_id_v1_fields(
                compute_end=record.compute_end_sim,
                compute_start=record.compute_start_sim,
                forecast_version=context.forecast_version,
                mode=CandidateMode.COMPLETE.value,
                node=record.target_node,
                path=record.path.path_id,
                reservation_version=(
                    context.reservation_snapshot.reservation_version
                ),
                schema=CANDIDATE_SCHEMA_VERSION,
                task=task.task_id,
                transmission_end=record.compute_start_sim,
                transmission_start=record.transmission_start_sim,
            )
            if self.profiler is not None:
                self.profiler.add(
                    "candidate_id_hash_seconds",
                    time.perf_counter() - identifier_started,
                )
            return record._replace(candidate_id=candidate_id)

        selected_record = attach_candidate_id(selected_record)
        if earliest_record == selected_record._replace(candidate_id=""):
            earliest_record = selected_record
        else:
            earliest_record = attach_candidate_id(earliest_record)
        return selected_record, earliest_record, count

    def record_from_feature_chunk(self, chunk, index):
        if not isinstance(chunk, CandidateFeatureRecordChunk):
            raise TypeError("chunk must be a CandidateFeatureRecordChunk")
        position = int(index)
        return CandidateRecord(
            "",
            chunk.target_node,
            chunk.path,
            float(chunk.transmission_starts[position]),
            float(chunk.starts[position]),
            float(chunk.ends[position]),
            float(chunk.tardiness[position]),
            bool(chunk.tardiness_applicable[position]),
            float(chunk.node_utilization[position]),
            float(chunk.path_utilization[position]),
            float(chunk.capacity_margin[position]),
        )

    def attach_context_candidate_id(self, context, record):
        identifier_started = time.perf_counter()
        candidate_id = deterministic_candidate_id_v1_fields(
            compute_end=record.compute_end_sim,
            compute_start=record.compute_start_sim,
            forecast_version=context.forecast_version,
            mode=CandidateMode.COMPLETE.value,
            node=record.target_node,
            path=record.path.path_id,
            reservation_version=(
                context.reservation_snapshot.reservation_version
            ),
            schema=CANDIDATE_SCHEMA_VERSION,
            task=context.task.task_id,
            transmission_end=record.compute_start_sim,
            transmission_start=record.transmission_start_sim,
        )
        if self.profiler is not None:
            self.profiler.add(
                "candidate_id_hash_seconds",
                time.perf_counter() - identifier_started,
            )
        return record._replace(candidate_id=candidate_id)

    def materialize_context_candidate(
        self,
        context,
        record,
        metric_evaluator=None,
    ):
        task = context.task
        metrics = self.evaluate_context_candidate_metrics(
            context,
            record,
            metric_evaluator=metric_evaluator,
        )
        queue_delay = max(
            0.0, context.decision_time_sim - task.arrival_time_sim
        )
        earliest_lead = (
            context.earliest_compute_start_sim - context.decision_time_sim
        )
        object_started = time.perf_counter()
        candidate = Candidate(
            candidate_id=record.candidate_id,
            candidate_schema_version=CANDIDATE_SCHEMA_VERSION,
            candidate_mode=CandidateMode.COMPLETE,
            task_id=task.task_id,
            decision_time_sim=context.decision_time_sim,
            reservation_snapshot_version=context.reservation_snapshot.reservation_version,
            forecast_version=context.forecast_version,
            target_node=record.target_node,
            path=record.path,
            transmission_start_sim=record.transmission_start_sim,
            transmission_end_sim=record.compute_start_sim,
            compute_start_sim=record.compute_start_sim,
            compute_end_sim=record.compute_end_sim,
            cpu_demand=task.cpu_demand,
            bandwidth_demand_mbps=task.bandwidth_demand_mbps,
            scheduler_queue_delay_sim=queue_delay,
            earliest_feasibility_lead_sim=earliest_lead,
            active_wait_sim=(
                record.compute_start_sim - context.earliest_compute_start_sim
            ),
            reservation_lead_sim=(
                record.compute_start_sim - context.decision_time_sim
            ),
            start_delay_sim=record.compute_start_sim - task.arrival_time_sim,
            preferred_start_tardiness_ratio=(
                record.preferred_start_tardiness_ratio
            ),
            preferred_start_tardiness_applicable=(
                record.preferred_start_tardiness_applicable
            ),
            estimated_candidate_marginal_system_cost_yuan=float(
                metrics.get("system_cost_yuan", 0.0)
            ),
            estimated_green_coverage=float(
                metrics.get("green_coverage", 0.0)
            ),
            estimated_candidate_marginal_green_energy_mwh=float(
                metrics.get("marginal_green_energy_mwh", 0.0)
            ),
            estimated_green_absorption_delta=float(
                metrics.get("green_absorption_delta", 0.0)
            ),
            estimated_green_opportunity=bool(
                metrics.get("green_opportunity", False)
            ),
            projected_node_utilization=record.projected_node_utilization,
            projected_path_peak_utilization=(
                record.projected_path_peak_utilization
            ),
            capacity_margin=record.capacity_margin,
        )
        if self.profiler is not None:
            self.profiler.add(
                "candidate_object_construction_seconds",
                time.perf_counter() - object_started,
            )
            self.profiler.increment("candidate_object_count")
        return candidate

    def evaluate_context_candidate_metrics(
        self,
        context,
        record,
        metric_evaluator=None,
    ):
        metric_started = time.perf_counter()
        metrics = (
            metric_evaluator(
                task=context.task,
                path=record.path,
                target_node=record.target_node,
                compute_start_sim=record.compute_start_sim,
                compute_end_sim=record.compute_end_sim,
                reservation_snapshot=context.reservation_snapshot,
            )
            if metric_evaluator
            else {}
        )
        if self.profiler is not None:
            self.profiler.add(
                "candidate_metric_evaluation_seconds",
                time.perf_counter() - metric_started,
            )
            self.profiler.increment("candidate_metric_evaluation_count")
        return metrics

    def iter_context_candidates(self, context, metric_evaluator=None):
        for record in self.iter_context_candidate_records(context):
            yield self.materialize_context_candidate(
                context,
                record,
                metric_evaluator=metric_evaluator,
            )

    def feature_chunks_from_context(
        self,
        context,
        feature_encoder,
        metric_evaluator=None,
        chunk_size=4096,
        with_records=False,
    ):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        batch_evaluator = getattr(
            metric_evaluator, "evaluate_batch", None
        )
        batch_encoder = getattr(
            feature_encoder, "encode_records_batch", None
        )
        array_encoder = getattr(
            feature_encoder, "encode_candidate_arrays", None
        )

        if batch_evaluator is not None and array_encoder is not None:
            task = context.task
            for target_node, path, duration, first, last in (
                self._declared_path_grids(
                    task, context.decision_time_sim
                )
            ):
                if first > last:
                    continue
                capacity = self.calendar.node_capacity(target_node)
                resources_unallocated = self.calendar.resources_unallocated(
                    context.reservation_snapshot,
                    target_node,
                    path,
                )
                for chunk_first in range(first, last + 1, chunk_size):
                    chunk_last = min(last + 1, chunk_first + chunk_size)
                    ticks = np.arange(
                        chunk_first, chunk_last, dtype=np.int64
                    )
                    feasibility_started = time.perf_counter()
                    starts = (
                        ticks.astype(np.float64)
                        * self.scheduling_cycle_sim
                    )
                    ends = starts + task.execution_duration_sim

                    schedule_started = time.perf_counter()
                    transmission_starts = (
                        starts.copy()
                        if path.is_local
                        else starts - duration.total_sim
                    )
                    if self.profiler is not None:
                        self.profiler.add(
                            "candidate_transmission_schedule_seconds",
                            time.perf_counter() - schedule_started,
                        )

                    temporal_feasible = (
                        (transmission_starts
                         >= context.decision_time_sim - 1e-12)
                        & (starts >= task.arrival_time_sim)
                        & (
                            starts
                            <= task.absolute_latest_start_sim
                        )
                    )
                    if context.forecast_covered_until_sim is not None:
                        temporal_feasible &= (
                            ends
                            <= context.forecast_covered_until_sim
                            + self.time_tolerance
                        )

                    cpu_started = time.perf_counter()
                    if (
                        resources_unallocated
                        and capacity is not None
                        and task.cpu_demand
                        <= capacity + self.time_tolerance
                    ):
                        cpu_feasible = np.ones(
                            starts.shape, dtype=np.bool_
                        )
                        cpu_projected = np.full(
                            starts.shape,
                            task.cpu_demand,
                            dtype=np.float64,
                        )
                        cpu_capacity = capacity
                    else:
                        cpu = self.calendar.cpu_feasible_many(
                            context.reservation_snapshot,
                            target_node,
                            starts,
                            ends,
                            task.cpu_demand,
                        )
                        cpu_feasible = cpu["feasible"]
                        cpu_projected = cpu["projected_peak"]
                        cpu_capacity = cpu["capacity"]
                    if self.profiler is not None:
                        self.profiler.add(
                            "candidate_cpu_feasibility_seconds",
                            time.perf_counter() - cpu_started,
                        )

                    path_started = time.perf_counter()
                    if resources_unallocated:
                        path_feasible = np.ones(
                            starts.shape, dtype=np.bool_
                        )
                        if path.is_local:
                            path_projected = np.zeros(
                                starts.shape, dtype=np.float64
                            )
                            path_capacity = np.zeros(
                                starts.shape, dtype=np.float64
                            )
                        else:
                            path_projected = np.full(
                                starts.shape,
                                task.bandwidth_demand_mbps,
                                dtype=np.float64,
                            )
                            path_capacity = np.full(
                                starts.shape,
                                path.static_bottleneck_mbps,
                                dtype=np.float64,
                            )
                    else:
                        path_result = (
                            self.calendar.path_feasible_many(
                                context.reservation_snapshot,
                                path,
                                transmission_starts,
                                starts,
                                task.bandwidth_demand_mbps,
                            )
                        )
                        path_feasible = path_result["feasible"]
                        path_projected = path_result["projected_peak"]
                        path_capacity = path_result["capacity"]
                    if self.profiler is not None:
                        self.profiler.add(
                            "candidate_path_feasibility_seconds",
                            time.perf_counter() - path_started,
                        )

                    item_started = time.perf_counter()
                    feasible = (
                        temporal_feasible
                        & cpu_feasible
                        & path_feasible
                    )
                    positions = np.flatnonzero(feasible)
                    if task.sla_type is SlaType.HARD:
                        tardiness = np.zeros(
                            positions.shape, dtype=np.float64
                        )
                        tardiness_applicable = np.zeros(
                            positions.shape, dtype=np.bool_
                        )
                    else:
                        selected_starts = starts[positions]
                        denominator = (
                            task.latest_start_limit_sim
                            - task.preferred_start_limit_sim
                        )
                        tardiness = np.minimum(
                            1.0,
                            np.maximum(
                                0.0,
                                (
                                    selected_starts
                                    - task.arrival_time_sim
                                    - task.preferred_start_limit_sim
                                )
                                / denominator,
                            ),
                        )
                        tardiness_applicable = np.ones(
                            positions.shape, dtype=np.bool_
                        )
                    if self.profiler is not None:
                        self.profiler.add(
                            "candidate_sla_and_item_seconds",
                            time.perf_counter() - item_started,
                        )
                        self.profiler.add(
                            "candidate_feasibility_stream_seconds",
                            time.perf_counter() - feasibility_started,
                        )
                        self.profiler.increment(
                            "stream_feasibility_check_count",
                            len(starts),
                        )
                    if positions.size == 0:
                        continue

                    selected_starts = starts[positions]
                    selected_ends = ends[positions]
                    selected_transmission_starts = (
                        transmission_starts[positions]
                    )
                    metric_started = time.perf_counter()
                    metrics = batch_evaluator(
                        task=task,
                        path=path,
                        target_node=target_node,
                        compute_start_sim=selected_starts,
                        compute_end_sim=selected_ends,
                        reservation_snapshot=context.reservation_snapshot,
                    )
                    if self.profiler is not None:
                        self.profiler.add(
                            "candidate_metric_evaluation_seconds",
                            time.perf_counter() - metric_started,
                        )
                        self.profiler.increment(
                            "candidate_metric_evaluation_count",
                            len(positions),
                        )

                    node_utilization = (
                        cpu_projected[positions] / cpu_capacity
                    )
                    if path.is_local:
                        path_utilization = np.zeros(
                            positions.shape, dtype=np.float64
                        )
                    else:
                        path_utilization = (
                            path_projected[positions]
                            / path_capacity[positions]
                        )
                    capacity_margin = np.minimum(
                        1.0 - node_utilization,
                        1.0 - path_utilization
                        if not path.is_local
                        else 1.0,
                    )
                    encoded = array_encoder(
                        context,
                        target_node=target_node,
                        compute_start_sim=selected_starts,
                        compute_end_sim=selected_ends,
                        transmission_start_sim=(
                            selected_transmission_starts
                        ),
                        preferred_start_tardiness_ratio=tardiness,
                        preferred_start_tardiness_applicable=(
                            tardiness_applicable
                        ),
                        projected_node_utilization=node_utilization,
                        projected_path_peak_utilization=path_utilization,
                        capacity_margin=capacity_margin,
                        metrics=metrics,
                    )
                    if with_records:
                        yield CandidateFeatureRecordChunk(
                            encoded,
                            target_node,
                            path,
                            selected_transmission_starts,
                            selected_starts,
                            selected_ends,
                            tardiness,
                            tardiness_applicable,
                            node_utilization,
                            path_utilization,
                            capacity_margin,
                        )
                    else:
                        yield encoded
            return

        if with_records:
            raise ValueError(
                "record-backed feature chunks require batch metric and "
                "array feature encoders"
            )

        def encode_chunk(records):
            if not records:
                return ()
            if batch_evaluator is None or batch_encoder is None:
                return tuple(
                    feature_encoder.encode_record(
                        context,
                        record,
                        self.evaluate_context_candidate_metrics(
                            context,
                            record,
                            metric_evaluator=metric_evaluator,
                        ),
                    )
                    for record in records
                )

            numeric_keys = (
                "system_cost_yuan",
                "green_coverage",
                "marginal_green_energy_mwh",
                "green_absorption_delta",
            )
            metrics = {
                key: np.empty(len(records), dtype=np.float64)
                for key in numeric_keys
            }
            metrics["green_opportunity"] = np.empty(
                len(records), dtype=np.bool_
            )
            groups = {}
            for index, record in enumerate(records):
                key = (record.target_node, record.path.path_id)
                group = groups.setdefault(
                    key,
                    {"path": record.path, "indices": []},
                )
                group["indices"].append(index)

            metric_started = time.perf_counter()
            for (target_node, _), group in groups.items():
                indices = np.asarray(group["indices"], dtype=np.intp)
                starts = np.asarray(
                    [
                        records[index].compute_start_sim
                        for index in indices
                    ],
                    dtype=np.float64,
                )
                ends = np.asarray(
                    [
                        records[index].compute_end_sim
                        for index in indices
                    ],
                    dtype=np.float64,
                )
                values = batch_evaluator(
                    task=context.task,
                    path=group["path"],
                    target_node=target_node,
                    compute_start_sim=starts,
                    compute_end_sim=ends,
                    reservation_snapshot=context.reservation_snapshot,
                )
                for key in numeric_keys:
                    metrics[key][indices] = np.asarray(
                        values.get(key, 0.0), dtype=np.float64
                    )
                metrics["green_opportunity"][indices] = np.asarray(
                    values.get("green_opportunity", False),
                    dtype=np.bool_,
                )
            if self.profiler is not None:
                self.profiler.add(
                    "candidate_metric_evaluation_seconds",
                    time.perf_counter() - metric_started,
                )
                self.profiler.increment(
                    "candidate_metric_evaluation_count", len(records)
                )
            return batch_encoder(context, records, metrics)

        chunk = []
        for record in self.iter_context_candidate_records(
            context,
            include_candidate_id=False,
        ):
            chunk.append(record)
            if len(chunk) >= chunk_size:
                yield encode_chunk(chunk)
                chunk = []
        if chunk:
            yield encode_chunk(chunk)

    def generate_complete(
        self,
        task: TaskSpec,
        decision_time_sim: float,
        reservation_snapshot: Optional[ReservationSnapshot] = None,
        forecast_version: str = "perfect-v1",
        forecast_covered_until_sim: Optional[float] = None,
        metric_evaluator: Optional[MetricEvaluator] = None,
    ) -> CandidateSetResult:
        stream = self.prepare_complete_stream(
            task,
            decision_time_sim,
            reservation_snapshot,
            forecast_version,
            forecast_covered_until_sim,
            metric_evaluator,
        )
        if stream.status is not CandidateGenerationStatus.OK:
            return CandidateSetResult(
                stream.status,
                CandidateMode.COMPLETE,
                (),
                stream.theoretical_slot_count,
                0,
                stream.earliest_compute_start_sim,
                stream.reason,
            )
        candidates = list(stream.iter_candidates())
        candidates.sort(
            key=lambda candidate: (
                candidate.compute_start_sim,
                candidate.target_node,
                candidate.path.path_id,
                candidate.candidate_id,
            )
        )
        return CandidateSetResult(
            CandidateGenerationStatus.OK,
            CandidateMode.COMPLETE,
            tuple(candidates),
            stream.theoretical_slot_count,
            stream.feasible_candidate_count,
            stream.earliest_compute_start_sim,
        )

    def generate_approximate(self, *args, **kwargs):
        from .approximate import compress_candidates

        max_candidates = kwargs.pop("max_candidates", None)
        utility_evaluator = kwargs.pop("utility_evaluator", None)
        if max_candidates is None:
            raise ValueError("approximate mode requires explicit max_candidates")
        full = self.generate_complete(*args, **kwargs)
        return compress_candidates(
            full.candidates,
            max_candidates,
            utility_evaluator=utility_evaluator,
        )
