"""Exact half-open interval calendars and atomic CPU/BW reservation commit."""

from dataclasses import dataclass
from bisect import bisect_left, bisect_right
from collections import OrderedDict
import threading
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np

from v1.domain.reservations import (
    CommitStatus,
    Edge,
    PathSpec,
    ReleaseStatus,
    Reservation,
    ReservationRequest,
    TimeInterval,
    canonical_edge,
    deterministic_reservation_hash,
)
from v1.domain.units import positive_finite


ResourceId = Union[str, Edge]


@dataclass(frozen=True)
class CalendarAllocation:
    reservation_id: str
    resource_id: ResourceId
    interval_sim: TimeInterval
    amount: float


@dataclass(frozen=True)
class ReservationSnapshot:
    reservation_version: int
    cpu_calendar_view: Tuple[CalendarAllocation, ...]
    link_calendar_view: Tuple[CalendarAllocation, ...]


@dataclass(frozen=True)
class FeasibilityResult:
    feasible: bool
    resource_id: Optional[ResourceId]
    capacity: float
    existing_peak: float
    projected_peak: float
    reason: Optional[str] = None


@dataclass(frozen=True)
class CommitResult:
    status: CommitStatus
    reservation_version: int
    reservation: Optional[Reservation] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class ReleaseResult:
    status: ReleaseStatus
    reservation_version: int
    reservation_id: str
    reason: Optional[str] = None


@dataclass(frozen=True)
class _PeakUsageIndex:
    """Exact immutable range-maximum index for one snapshot resource."""

    boundaries: Tuple[float, ...]
    maximum_levels: Tuple[Tuple[float, ...], ...]

    @classmethod
    def build(cls, allocations):
        items = tuple(allocations)
        if not items:
            return cls((), ())
        boundaries = tuple(sorted({
            boundary
            for item in items
            for boundary in (
                item.interval_sim.start_sim,
                item.interval_sim.end_sim,
            )
        }))
        values = tuple(
            sum(
                item.amount
                for item in items
                if item.interval_sim.start_sim <= probe
                < item.interval_sim.end_sim
            )
            for probe in (
                left + (right - left) / 2.0
                for left, right in zip(
                    boundaries[:-1], boundaries[1:]
                )
            )
        )
        if not values:
            return cls(boundaries, ())
        levels = [values]
        span = 2
        while span <= len(values):
            previous = levels[-1]
            half = span // 2
            levels.append(tuple(
                max(previous[index], previous[index + half])
                for index in range(len(values) - span + 1)
            ))
            span *= 2
        return cls(boundaries, tuple(levels))

    def peak(self, interval: TimeInterval) -> float:
        if (
            not self.maximum_levels
            or interval.end_sim <= self.boundaries[0]
            or interval.start_sim >= self.boundaries[-1]
        ):
            return 0.0
        start = max(interval.start_sim, self.boundaries[0])
        end = min(interval.end_sim, self.boundaries[-1])
        if end <= start:
            return 0.0
        left = max(0, bisect_right(self.boundaries, start) - 1)
        right = min(
            len(self.maximum_levels[0]) - 1,
            bisect_left(self.boundaries, end) - 1,
        )
        if right < left:
            return 0.0
        count = right - left + 1
        level = count.bit_length() - 1
        span = 1 << level
        values = self.maximum_levels[level]
        return max(values[left], values[right - span + 1])

    def peak_many(self, starts_sim, ends_sim) -> np.ndarray:
        starts = np.asarray(starts_sim, dtype=np.float64)
        ends = np.asarray(ends_sim, dtype=np.float64)
        if starts.ndim != 1 or ends.ndim != 1 or starts.shape != ends.shape:
            raise ValueError(
                "interval arrays must be one-dimensional and aligned"
            )
        peaks = np.zeros(starts.shape, dtype=np.float64)
        if starts.size == 0 or not self.maximum_levels:
            return peaks
        boundaries = np.asarray(self.boundaries, dtype=np.float64)
        overlap = (
            (ends > boundaries[0])
            & (starts < boundaries[-1])
        )
        if not np.any(overlap):
            return peaks
        positions = np.flatnonzero(overlap)
        clipped_starts = np.maximum(starts[positions], boundaries[0])
        clipped_ends = np.minimum(ends[positions], boundaries[-1])
        positive = clipped_ends > clipped_starts
        if not np.any(positive):
            return peaks
        positions = positions[positive]
        clipped_starts = clipped_starts[positive]
        clipped_ends = clipped_ends[positive]
        left = np.maximum(
            0,
            np.searchsorted(
                boundaries, clipped_starts, side="right"
            ) - 1,
        )
        right = np.minimum(
            len(self.maximum_levels[0]) - 1,
            np.searchsorted(
                boundaries, clipped_ends, side="left"
            ) - 1,
        )
        valid = right >= left
        positions = positions[valid]
        left = left[valid]
        right = right[valid]
        counts = right - left + 1
        levels = np.floor(np.log2(counts)).astype(np.intp)
        for level in np.unique(levels):
            selected = levels == level
            span = 1 << int(level)
            values = np.asarray(
                self.maximum_levels[int(level)], dtype=np.float64
            )
            peaks[positions[selected]] = np.maximum(
                values[left[selected]],
                values[right[selected] - span + 1],
            )
        return peaks


def _allocation_sort_key(item: CalendarAllocation):
    return (
        str(item.resource_id),
        item.interval_sim.start_sim,
        item.interval_sim.end_sim,
        item.reservation_id,
    )


def _peak_usage(
    allocations: Iterable[CalendarAllocation],
    interval: TimeInterval,
) -> float:
    overlapping = [
        allocation
        for allocation in allocations
        if allocation.interval_sim.overlaps(interval)
    ]
    if not overlapping:
        return 0.0
    boundaries = {interval.start_sim, interval.end_sim}
    for allocation in overlapping:
        boundaries.add(max(interval.start_sim, allocation.interval_sim.start_sim))
        boundaries.add(min(interval.end_sim, allocation.interval_sim.end_sim))
    ordered = sorted(boundaries)
    peak = 0.0
    for left, right in zip(ordered[:-1], ordered[1:]):
        if right <= left:
            continue
        probe = left + (right - left) / 2.0
        usage = sum(
            allocation.amount
            for allocation in overlapping
            if allocation.interval_sim.start_sim <= probe < allocation.interval_sim.end_sim
        )
        peak = max(peak, usage)
    return peak


class ReservationCalendar:
    """Thread-safe reservation calendar with compare-version transactions."""

    def __init__(
        self,
        node_capacities: Mapping[str, float],
        link_capacities: Mapping[Edge, float],
    ):
        self._node_capacities = {
            str(node): positive_finite(f"node_capacity[{node}]", capacity)
            for node, capacity in node_capacities.items()
        }
        self._link_capacities = {
            canonical_edge(edge): positive_finite(
                f"link_capacity[{edge}]",
                capacity,
            )
            for edge, capacity in link_capacities.items()
        }
        self._cpu_allocations = []
        self._link_allocations = []
        self._reservations: Dict[str, Reservation] = {}
        self._active_reservation_ids = set()
        self._version = 0
        self._lock = threading.RLock()
        self._snapshot_allocation_index_cache = OrderedDict()
        self._snapshot_allocation_index_capacity = 64
        self._snapshot_peak_index_cache = OrderedDict()
        self._snapshot_peak_index_capacity = 64

    def __getstate__(self):
        """Make exact training checkpoints possible without serializing locks."""
        state = dict(self.__dict__)
        state.pop("_lock", None)
        state["_snapshot_allocation_index_cache"] = OrderedDict()
        state["_snapshot_peak_index_cache"] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()
        if "_snapshot_allocation_index_cache" not in self.__dict__:
            self._snapshot_allocation_index_cache = OrderedDict()
        if "_snapshot_allocation_index_capacity" not in self.__dict__:
            self._snapshot_allocation_index_capacity = 64
        if "_snapshot_peak_index_cache" not in self.__dict__:
            self._snapshot_peak_index_cache = OrderedDict()
        if "_snapshot_peak_index_capacity" not in self.__dict__:
            self._snapshot_peak_index_capacity = 64

    def _snapshot_allocations_by_resource(self, snapshot, *, cpu):
        cache_key = (id(snapshot), bool(cpu))
        cached = self._snapshot_allocation_index_cache.get(cache_key)
        if cached is not None and cached[0] is snapshot:
            self._snapshot_allocation_index_cache.move_to_end(cache_key)
            return cached[1]
        source = (
            snapshot.cpu_calendar_view if cpu else snapshot.link_calendar_view
        )
        grouped = {}
        for allocation in source:
            grouped.setdefault(allocation.resource_id, []).append(allocation)
        indexed = {key: tuple(values) for key, values in grouped.items()}
        self._snapshot_allocation_index_cache[cache_key] = (snapshot, indexed)
        self._snapshot_allocation_index_cache.move_to_end(cache_key)
        while (
            len(self._snapshot_allocation_index_cache)
            > self._snapshot_allocation_index_capacity
        ):
            self._snapshot_allocation_index_cache.popitem(last=False)
        return indexed

    def resources_unallocated(
        self,
        snapshot: ReservationSnapshot,
        node: str,
        path: PathSpec,
    ) -> bool:
        """Whether candidate resources have no allocations in this snapshot."""

        cpu_allocations = self._snapshot_allocations_by_resource(
            snapshot, cpu=True
        )
        if cpu_allocations.get(str(node)):
            return False
        if path.is_local:
            return True
        link_allocations = self._snapshot_allocations_by_resource(
            snapshot, cpu=False
        )
        return all(not link_allocations.get(edge) for edge in path.resource_edges)

    def _snapshot_peak_indices(self, snapshot, *, cpu):
        cache_key = (id(snapshot), bool(cpu))
        cached = self._snapshot_peak_index_cache.get(cache_key)
        if cached is not None and cached[0] is snapshot:
            self._snapshot_peak_index_cache.move_to_end(cache_key)
            return cached[1]
        grouped = self._snapshot_allocations_by_resource(
            snapshot,
            cpu=cpu,
        )
        indexed = {
            resource: _PeakUsageIndex.build(allocations)
            for resource, allocations in grouped.items()
        }
        self._snapshot_peak_index_cache[cache_key] = (
            snapshot,
            indexed,
        )
        self._snapshot_peak_index_cache.move_to_end(cache_key)
        while (
            len(self._snapshot_peak_index_cache)
            > self._snapshot_peak_index_capacity
        ):
            self._snapshot_peak_index_cache.popitem(last=False)
        return indexed

    def _snapshot_peak_usage(
        self,
        snapshot,
        resource,
        interval,
        *,
        cpu,
    ):
        index = self._snapshot_peak_indices(
            snapshot,
            cpu=cpu,
        ).get(resource)
        return 0.0 if index is None else index.peak(interval)

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def active_reservation_count(self) -> int:
        with self._lock:
            return len(self._active_reservation_ids)

    def node_capacity(self, node: str) -> Optional[float]:
        """Return immutable configured capacity for static admission checks."""

        return self._node_capacities.get(str(node))

    def link_capacity(self, edge: Edge) -> Optional[float]:
        return self._link_capacities.get(canonical_edge(edge))

    def active_reservations(self) -> Tuple[Reservation, ...]:
        with self._lock:
            return tuple(
                self._reservations[reservation_id]
                for reservation_id in sorted(self._active_reservation_ids)
            )

    def snapshot(self) -> ReservationSnapshot:
        with self._lock:
            return ReservationSnapshot(
                reservation_version=self._version,
                cpu_calendar_view=tuple(sorted(self._cpu_allocations, key=_allocation_sort_key)),
                link_calendar_view=tuple(sorted(self._link_allocations, key=_allocation_sort_key)),
            )

    @staticmethod
    def peak_usage(
        allocations: Iterable[CalendarAllocation],
        interval: TimeInterval,
    ) -> float:
        return _peak_usage(allocations, interval)

    def cpu_feasible(
        self,
        snapshot: ReservationSnapshot,
        node: str,
        interval: TimeInterval,
        amount: float,
    ) -> FeasibilityResult:
        cpu_amount = positive_finite("cpu_amount", amount)
        capacity = self._node_capacities.get(str(node))
        if capacity is None:
            return FeasibilityResult(
                False,
                str(node),
                0.0,
                0.0,
                cpu_amount,
                "unknown compute node",
            )
        existing_peak = self._snapshot_peak_usage(
            snapshot,
            str(node),
            interval,
            cpu=True,
        )
        projected = existing_peak + cpu_amount
        return FeasibilityResult(
            projected <= capacity + 1e-12,
            str(node),
            capacity,
            existing_peak,
            projected,
            None if projected <= capacity + 1e-12 else "CPU capacity exceeded",
        )

    def cpu_feasible_many(
        self,
        snapshot: ReservationSnapshot,
        node: str,
        starts_sim,
        ends_sim,
        amount: float,
    ):
        cpu_amount = positive_finite("cpu_amount", amount)
        starts = np.asarray(starts_sim, dtype=np.float64)
        ends = np.asarray(ends_sim, dtype=np.float64)
        if starts.ndim != 1 or ends.ndim != 1 or starts.shape != ends.shape:
            raise ValueError(
                "interval arrays must be one-dimensional and aligned"
            )
        capacity = self._node_capacities.get(str(node))
        if capacity is None:
            projected = np.full(starts.shape, cpu_amount, dtype=np.float64)
            return {
                "feasible": np.zeros(starts.shape, dtype=np.bool_),
                "capacity": 0.0,
                "existing_peak": np.zeros(starts.shape, dtype=np.float64),
                "projected_peak": projected,
            }
        index = self._snapshot_peak_indices(
            snapshot, cpu=True
        ).get(str(node))
        existing = (
            np.zeros(starts.shape, dtype=np.float64)
            if index is None
            else index.peak_many(starts, ends)
        )
        projected = existing + cpu_amount
        return {
            "feasible": projected <= capacity + 1e-12,
            "capacity": capacity,
            "existing_peak": existing,
            "projected_peak": projected,
        }

    def path_feasible(
        self,
        snapshot: ReservationSnapshot,
        path: PathSpec,
        interval: Optional[TimeInterval],
        amount: float,
    ) -> FeasibilityResult:
        bandwidth = positive_finite("bandwidth_amount_mbps", amount)
        if path.is_local:
            return FeasibilityResult(True, None, 0.0, 0.0, 0.0)
        if interval is None:
            return FeasibilityResult(
                False,
                None,
                0.0,
                0.0,
                bandwidth,
                "remote path requires transmission interval",
            )
        worst = None
        for edge in path.resource_edges:
            capacity = self._link_capacities.get(edge)
            if capacity is None:
                return FeasibilityResult(
                    False,
                    edge,
                    0.0,
                    0.0,
                    bandwidth,
                    "unknown link",
                )
            existing_peak = self._snapshot_peak_usage(
                snapshot,
                edge,
                interval,
                cpu=False,
            )
            projected = existing_peak + bandwidth
            result = FeasibilityResult(
                projected <= capacity + 1e-12,
                edge,
                capacity,
                existing_peak,
                projected,
                None if projected <= capacity + 1e-12 else "link capacity exceeded",
            )
            if not result.feasible:
                return result
            if worst is None or result.projected_peak / result.capacity > (
                worst.projected_peak / worst.capacity
            ):
                worst = result
        return worst or FeasibilityResult(True, None, 0.0, 0.0, 0.0)

    def path_feasible_many(
        self,
        snapshot: ReservationSnapshot,
        path: PathSpec,
        starts_sim,
        ends_sim,
        amount: float,
    ):
        bandwidth = positive_finite("bandwidth_amount_mbps", amount)
        starts = np.asarray(starts_sim, dtype=np.float64)
        ends = np.asarray(ends_sim, dtype=np.float64)
        if starts.ndim != 1 or ends.ndim != 1 or starts.shape != ends.shape:
            raise ValueError(
                "interval arrays must be one-dimensional and aligned"
            )
        if path.is_local:
            return {
                "feasible": np.ones(starts.shape, dtype=np.bool_),
                "capacity": np.zeros(starts.shape, dtype=np.float64),
                "existing_peak": np.zeros(starts.shape, dtype=np.float64),
                "projected_peak": np.zeros(starts.shape, dtype=np.float64),
            }
        feasible = np.ones(starts.shape, dtype=np.bool_)
        worst_ratio = np.full(starts.shape, -np.inf, dtype=np.float64)
        worst_capacity = np.zeros(starts.shape, dtype=np.float64)
        worst_existing = np.zeros(starts.shape, dtype=np.float64)
        worst_projected = np.zeros(starts.shape, dtype=np.float64)
        indices = self._snapshot_peak_indices(snapshot, cpu=False)
        for edge in path.resource_edges:
            capacity = self._link_capacities.get(edge)
            if capacity is None:
                return {
                    "feasible": np.zeros(starts.shape, dtype=np.bool_),
                    "capacity": worst_capacity,
                    "existing_peak": worst_existing,
                    "projected_peak": np.full(
                        starts.shape, bandwidth, dtype=np.float64
                    ),
                }
            index = indices.get(edge)
            existing = (
                np.zeros(starts.shape, dtype=np.float64)
                if index is None
                else index.peak_many(starts, ends)
            )
            projected = existing + bandwidth
            edge_feasible = projected <= capacity + 1e-12
            feasible &= edge_feasible
            ratio = projected / capacity
            replace = ratio > worst_ratio
            worst_ratio[replace] = ratio[replace]
            worst_capacity[replace] = capacity
            worst_existing[replace] = existing[replace]
            worst_projected[replace] = projected[replace]
        return {
            "feasible": feasible,
            "capacity": worst_capacity,
            "existing_peak": worst_existing,
            "projected_peak": worst_projected,
        }

    def try_commit(
        self,
        request: ReservationRequest,
        expected_version: int,
        inject_failure_at: Optional[str] = None,
    ) -> CommitResult:
        with self._lock:
            if expected_version != request.reservation_snapshot_version:
                return CommitResult(
                    CommitStatus.CANDIDATE_STALE,
                    self._version,
                    reason="expected version differs from candidate snapshot",
                )
            if expected_version != self._version:
                return CommitResult(
                    CommitStatus.CONFLICT,
                    self._version,
                    reason="reservation snapshot version changed",
                )

            snapshot = self.snapshot()
            cpu_result = self.cpu_feasible(
                snapshot,
                request.target_node,
                request.compute_interval_sim,
                request.cpu_amount,
            )
            if not cpu_result.feasible:
                return CommitResult(
                    CommitStatus.CPU_INFEASIBLE,
                    self._version,
                    reason=cpu_result.reason,
                )
            path_result = self.path_feasible(
                snapshot,
                request.path,
                request.transmission_interval_sim,
                request.bandwidth_amount_mbps,
            )
            if not path_result.feasible:
                return CommitResult(
                    CommitStatus.BANDWIDTH_INFEASIBLE,
                    self._version,
                    reason=path_result.reason,
                )

            next_version = self._version + 1
            audit_hash = deterministic_reservation_hash(request, next_version)
            reservation_id = "reservation-" + audit_hash[:24]
            reservation = Reservation(
                reservation_id=reservation_id,
                task_id=request.task_id,
                committed_candidate_id=request.committed_candidate_id,
                committed_at_sim=request.committed_at_sim,
                committed_reservation_version=next_version,
                target_node=request.target_node,
                path=request.path,
                transmission_interval_sim=request.transmission_interval_sim,
                compute_interval_sim=request.compute_interval_sim,
                bandwidth_amount_mbps=request.bandwidth_amount_mbps,
                cpu_amount=request.cpu_amount,
                audit_hash=audit_hash,
            )
            cpu_item = CalendarAllocation(
                reservation_id,
                request.target_node,
                request.compute_interval_sim,
                request.cpu_amount,
            )
            link_items = [
                CalendarAllocation(
                    reservation_id,
                    edge,
                    request.transmission_interval_sim,
                    request.bandwidth_amount_mbps,
                )
                for edge in request.path.resource_edges
            ]

            cpu_before = list(self._cpu_allocations)
            link_before = list(self._link_allocations)
            reservations_before = dict(self._reservations)
            active_before = set(self._active_reservation_ids)
            version_before = self._version
            try:
                self._cpu_allocations.append(cpu_item)
                if inject_failure_at == "after_cpu_write":
                    raise RuntimeError("injected failure after CPU write")
                for index, item in enumerate(link_items):
                    self._link_allocations.append(item)
                    if inject_failure_at == f"after_link_write:{index}":
                        raise RuntimeError("injected failure after link write")
                self._reservations[reservation_id] = reservation
                self._active_reservation_ids.add(reservation_id)
                self._version = next_version
            except Exception as exc:
                self._cpu_allocations = cpu_before
                self._link_allocations = link_before
                self._reservations = reservations_before
                self._active_reservation_ids = active_before
                self._version = version_before
                return CommitResult(
                    CommitStatus.INTERNAL_ROLLBACK,
                    self._version,
                    reason=str(exc),
                )
            return CommitResult(
                CommitStatus.COMMITTED,
                self._version,
                reservation=reservation,
            )

    def get_reservation(self, reservation_id: str) -> Optional[Reservation]:
        with self._lock:
            return self._reservations.get(reservation_id)

    def reservations(self) -> Tuple[Reservation, ...]:
        """Return immutable reservation history in deterministic order."""

        with self._lock:
            return tuple(sorted(
                self._reservations.values(),
                key=lambda item: (item.committed_at_sim, item.task_id, item.reservation_id),
            ))

    def verify_reservation(self, reservation_id: str) -> bool:
        with self._lock:
            reservation = self._reservations.get(reservation_id)
            if reservation is None or reservation_id not in self._active_reservation_ids:
                return False
            cpu_matches = [
                item
                for item in self._cpu_allocations
                if item.reservation_id == reservation_id
            ]
            link_matches = [
                item
                for item in self._link_allocations
                if item.reservation_id == reservation_id
            ]
            if len(cpu_matches) != 1:
                return False
            if len(link_matches) != len(reservation.path.resource_edges):
                return False
            return (
                cpu_matches[0].resource_id == reservation.target_node
                and cpu_matches[0].interval_sim == reservation.compute_interval_sim
                and cpu_matches[0].amount == reservation.cpu_amount
                and all(
                    item.interval_sim == reservation.transmission_interval_sim
                    and item.amount == reservation.bandwidth_amount_mbps
                    for item in link_matches
                )
            )

    def release_on_normal_completion(
        self,
        reservation_id: str,
        at_time_sim: Optional[float] = None,
    ) -> ReleaseResult:
        with self._lock:
            reservation = self._reservations.get(reservation_id)
            if reservation is None:
                return ReleaseResult(
                    ReleaseStatus.NOT_FOUND,
                    self._version,
                    reservation_id,
                    "unknown reservation",
                )
            if reservation_id not in self._active_reservation_ids:
                return ReleaseResult(
                    ReleaseStatus.ALREADY_RELEASED,
                    self._version,
                    reservation_id,
                )
            if (
                at_time_sim is not None
                and at_time_sim < reservation.compute_interval_sim.end_sim - 1e-12
            ):
                return ReleaseResult(
                    ReleaseStatus.TOO_EARLY,
                    self._version,
                    reservation_id,
                    "normal completion release cannot precede compute end",
                )
            self._cpu_allocations = [
                item
                for item in self._cpu_allocations
                if item.reservation_id != reservation_id
            ]
            self._link_allocations = [
                item
                for item in self._link_allocations
                if item.reservation_id != reservation_id
            ]
            self._active_reservation_ids.remove(reservation_id)
            self._version += 1
            return ReleaseResult(
                ReleaseStatus.RELEASED,
                self._version,
                reservation_id,
            )

    def release_on_failure(self, reservation_id: str) -> ReleaseResult:
        """Release a broken reservation without treating it as a retry/cancel."""

        with self._lock:
            reservation = self._reservations.get(reservation_id)
            if reservation is None:
                return ReleaseResult(
                    ReleaseStatus.NOT_FOUND,
                    self._version,
                    reservation_id,
                    "unknown reservation",
                )
            if reservation_id not in self._active_reservation_ids:
                return ReleaseResult(
                    ReleaseStatus.ALREADY_RELEASED,
                    self._version,
                    reservation_id,
                )
            self._cpu_allocations = [
                item
                for item in self._cpu_allocations
                if item.reservation_id != reservation_id
            ]
            self._link_allocations = [
                item
                for item in self._link_allocations
                if item.reservation_id != reservation_id
            ]
            self._active_reservation_ids.remove(reservation_id)
            self._version += 1
            return ReleaseResult(
                ReleaseStatus.RELEASED,
                self._version,
                reservation_id,
            )
