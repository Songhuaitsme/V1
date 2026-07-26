"""Fixed-bandwidth pipelined transmission and JIT interval construction."""

from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import json
import math
from typing import Iterable, Optional, Sequence

from v1.domain.models import TaskSpec
from v1.domain.reservations import PathSpec, ReservationValidationError, TimeInterval
from v1.domain.units import (
    DataUnitConverter,
    TimeConverter,
    non_negative_finite,
    positive_finite,
)


@dataclass(frozen=True)
class TransmissionDuration:
    data_seconds: float
    propagation_seconds: float
    total_seconds: float
    total_sim: float
    static_path_feasible: bool


@dataclass(frozen=True)
class JitSchedule:
    transmission_interval_sim: Optional[TimeInterval]
    compute_start_sim: float
    decision_time_sim: float
    feasible_from_decision: bool


def _path_id(nodes: Sequence[str]) -> str:
    encoded = json.dumps(tuple(nodes), separators=(",", ":")).encode("utf-8")
    return "path-" + hashlib.sha256(encoded).hexdigest()[:20]


def build_path_spec(graph, ordered_nodes: Iterable[str], path_id: Optional[str] = None) -> PathSpec:
    """Build and validate a PathSpec from a NetworkX-compatible graph."""

    nodes = tuple(str(node) for node in ordered_nodes)
    if not nodes:
        raise ReservationValidationError("ordered_nodes", "cannot be empty")
    source, target = nodes[0], nodes[-1]
    if len(nodes) == 1:
        return PathSpec(
            path_id=path_id or _path_id(nodes),
            source_node=source,
            target_node=target,
            ordered_nodes=nodes,
            ordered_edges=(),
            total_distance_km=0.0,
            static_bottleneck_mbps=0.0,
            route_cost=0.0,
        )

    edges = tuple(zip(nodes[:-1], nodes[1:]))
    distance = 0.0
    capacities = []
    route_cost = 0.0
    for u, v in edges:
        if not graph.has_edge(u, v):
            raise ReservationValidationError(
                "ordered_edges",
                f"edge {u}-{v} does not exist",
            )
        data = graph[u][v]
        capacity = positive_finite(f"capacity[{u},{v}]", data.get("capacity"))
        edge_distance = non_negative_finite(
            f"distance_km[{u},{v}]",
            data.get("distance_km"),
        )
        cost = data.get("cost", edge_distance)
        route_cost += non_negative_finite(f"route_cost[{u},{v}]", cost)
        distance += edge_distance
        capacities.append(capacity)
    return PathSpec(
        path_id=path_id or _path_id(nodes),
        source_node=source,
        target_node=target,
        ordered_nodes=nodes,
        ordered_edges=edges,
        total_distance_km=distance,
        static_bottleneck_mbps=min(capacities),
        route_cost=route_cost,
    )


class TransmissionModel:
    """End-to-end reserved-rate transmission with one serialization term."""

    def __init__(
        self,
        converter: TimeConverter,
        fiber_speed_km_per_second: float,
    ):
        self.converter = converter
        self.fiber_speed_km_per_second = positive_finite(
            "fiber_speed_km_per_second",
            fiber_speed_km_per_second,
        )
        self._duration_cache = OrderedDict()
        self._duration_cache_capacity = 4096

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_duration_cache"] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_duration_cache" not in self.__dict__:
            self._duration_cache = OrderedDict()
        if "_duration_cache_capacity" not in self.__dict__:
            self._duration_cache_capacity = 4096

    def duration(self, task: TaskSpec, path: PathSpec) -> TransmissionDuration:
        cache_key = (task, path)
        cached = self._duration_cache.get(cache_key)
        if cached is not None:
            self._duration_cache.move_to_end(cache_key)
            return cached
        if task.source_node != path.source_node:
            raise ReservationValidationError(
                "path.source_node",
                "must equal task.source_node",
            )
        if path.is_local:
            result = TransmissionDuration(0.0, 0.0, 0.0, 0.0, True)
            self._duration_cache[cache_key] = result
            self._duration_cache.move_to_end(cache_key)
            while len(self._duration_cache) > self._duration_cache_capacity:
                self._duration_cache.popitem(last=False)
            return result

        data_megabits = DataUnitConverter.decimal_mb_to_megabits(task.data_size_mb)
        data_seconds = data_megabits / task.bandwidth_demand_mbps
        propagation_seconds = (
            path.total_distance_km / self.fiber_speed_km_per_second
        )
        total_seconds = data_seconds + propagation_seconds
        result = TransmissionDuration(
            data_seconds=data_seconds,
            propagation_seconds=propagation_seconds,
            total_seconds=total_seconds,
            total_sim=self.converter.seconds_to_sim(total_seconds),
            static_path_feasible=(
                path.static_bottleneck_mbps >= task.bandwidth_demand_mbps
            ),
        )
        self._duration_cache[cache_key] = result
        self._duration_cache.move_to_end(cache_key)
        while len(self._duration_cache) > self._duration_cache_capacity:
            self._duration_cache.popitem(last=False)
        return result

    def jit_schedule(
        self,
        task: TaskSpec,
        path: PathSpec,
        compute_start_sim: float,
        decision_time_sim: float,
    ) -> JitSchedule:
        compute_start = float(compute_start_sim)
        decision_time = float(decision_time_sim)
        if not math.isfinite(compute_start) or not math.isfinite(decision_time):
            raise ReservationValidationError(
                "jit_time",
                "compute_start_sim and decision_time_sim must be finite",
            )
        duration = self.duration(task, path)
        if not duration.static_path_feasible:
            return JitSchedule(None, compute_start, decision_time, False)
        if path.is_local:
            return JitSchedule(
                transmission_interval_sim=None,
                compute_start_sim=compute_start,
                decision_time_sim=decision_time,
                feasible_from_decision=compute_start >= decision_time,
            )
        transmission_start = compute_start - duration.total_sim
        interval = TimeInterval(transmission_start, compute_start)
        return JitSchedule(
            transmission_interval_sim=interval,
            compute_start_sim=compute_start,
            decision_time_sim=decision_time,
            feasible_from_decision=(transmission_start >= decision_time - 1e-12),
        )
