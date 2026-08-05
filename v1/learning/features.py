"""Frozen v1.0 candidate feature extraction with physical training scales."""

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping, Tuple

import numpy as np

from v1.domain.candidates import Candidate
from v1.domain.units import positive_finite


CANDIDATE_FEATURE_NAMES = (
    "target_node_index_normalized",
    "start_offset_normalized",
    "active_wait_normalized",
    "transmission_duration_normalized",
    "compute_duration_normalized",
    "marginal_cost_normalized",
    "green_coverage",
    "green_absorption_delta_normalized",
    "green_opportunity",
    "projected_node_utilization",
    "projected_path_peak_utilization",
    "capacity_margin",
    "start_delay_normalized",
    "preferred_start_tardiness_ratio",
    "preferred_start_tardiness_applicable",
    "is_earliest_feasible",
    "cpu_demand_normalized",
    "bandwidth_demand_normalized",
)

CANDIDATE_FEATURE_GROUPS = {
    "cost": ("marginal_cost_normalized",),
    "green": (
        "green_coverage",
        "green_absorption_delta_normalized",
        "green_opportunity",
    ),
    "sla": (
        "start_delay_normalized",
        "preferred_start_tardiness_ratio",
        "preferred_start_tardiness_applicable",
    ),
    "load": (
        "projected_node_utilization",
        "projected_path_peak_utilization",
        "capacity_margin",
        "cpu_demand_normalized",
        "bandwidth_demand_normalized",
    ),
}


@dataclass(frozen=True)
class CandidateFeatureConfig:
    time_scale_sim: float
    cost_scale_yuan: float
    absorption_delta_scale: float
    cpu_scale: float
    bandwidth_scale_mbps: float

    def __post_init__(self):
        for field in self.__dataclass_fields__:
            object.__setattr__(self, field, positive_finite(field, getattr(self, field)))


class CandidateFeatureEncoder:
    def __init__(
        self,
        node_index: Mapping[str, int],
        config: CandidateFeatureConfig,
        disabled_feature_groups: Iterable[str] = (),
    ):
        if not node_index:
            raise ValueError("node_index cannot be empty")
        values = tuple(node_index.values())
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("node indices must be non-negative integers")
        if len(set(values)) != len(values):
            raise ValueError("node indices must be unique")
        self.node_index = dict(node_index)
        self.config = config
        groups = tuple(sorted(set(disabled_feature_groups)))
        unknown = set(groups) - set(CANDIDATE_FEATURE_GROUPS)
        if unknown:
            raise ValueError(
                f"unknown candidate feature groups: {sorted(unknown)}"
            )
        self.disabled_feature_groups = groups
        disabled_names = {
            name
            for group in groups
            for name in CANDIDATE_FEATURE_GROUPS[group]
        }
        self._disabled_feature_indices = tuple(
            index
            for index, name in enumerate(CANDIDATE_FEATURE_NAMES)
            if name in disabled_names
        )

    @property
    def feature_dim(self) -> int:
        return len(CANDIDATE_FEATURE_NAMES)

    @property
    def feature_schema_hash(self) -> str:
        schema = {
            "version": "1.0",
            "features": CANDIDATE_FEATURE_NAMES,
            "nodes": sorted(self.node_index.items()),
            "scales": self.config.__dict__,
        }
        if self.disabled_feature_groups:
            schema["disabled_feature_groups"] = self.disabled_feature_groups
        payload = json.dumps(
            schema, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _mask_matrix(self, matrix):
        if self._disabled_feature_indices:
            matrix[:, self._disabled_feature_indices] = 0.0
        return matrix

    def _mask_values(self, values):
        if not self._disabled_feature_indices:
            return values
        mutable = list(values)
        for index in self._disabled_feature_indices:
            mutable[index] = 0.0
        return tuple(mutable)

    def encode(self, candidate: Candidate, earliest_compute_start_sim: float) -> Tuple[float, ...]:
        return self._encode_values(
            target_node=candidate.target_node,
            decision_time_sim=candidate.decision_time_sim,
            compute_start_sim=candidate.compute_start_sim,
            compute_end_sim=candidate.compute_end_sim,
            transmission_start_sim=candidate.transmission_start_sim,
            earliest_compute_start_sim=earliest_compute_start_sim,
            marginal_cost_yuan=(
                candidate.estimated_candidate_marginal_system_cost_yuan
            ),
            green_coverage=candidate.estimated_green_coverage,
            green_absorption_delta=(
                candidate.estimated_green_absorption_delta
            ),
            green_opportunity=candidate.estimated_green_opportunity,
            projected_node_utilization=(
                candidate.projected_node_utilization
            ),
            projected_path_peak_utilization=(
                candidate.projected_path_peak_utilization
            ),
            capacity_margin=candidate.capacity_margin,
            start_delay_sim=candidate.start_delay_sim,
            preferred_start_tardiness_ratio=(
                candidate.preferred_start_tardiness_ratio
            ),
            preferred_start_tardiness_applicable=(
                candidate.preferred_start_tardiness_applicable
            ),
            cpu_demand=candidate.cpu_demand,
            bandwidth_demand_mbps=candidate.bandwidth_demand_mbps,
        )

    def encode_record(self, context, record, metrics) -> Tuple[float, ...]:
        """Encode a complete candidate record without constructing Candidate."""

        task = context.task
        return self._encode_values(
            target_node=record.target_node,
            decision_time_sim=context.decision_time_sim,
            compute_start_sim=record.compute_start_sim,
            compute_end_sim=record.compute_end_sim,
            transmission_start_sim=record.transmission_start_sim,
            earliest_compute_start_sim=context.earliest_compute_start_sim,
            marginal_cost_yuan=float(metrics.get("system_cost_yuan", 0.0)),
            green_coverage=float(metrics.get("green_coverage", 0.0)),
            green_absorption_delta=float(
                metrics.get("green_absorption_delta", 0.0)
            ),
            green_opportunity=bool(
                metrics.get("green_opportunity", False)
            ),
            projected_node_utilization=(
                record.projected_node_utilization
            ),
            projected_path_peak_utilization=(
                record.projected_path_peak_utilization
            ),
            capacity_margin=record.capacity_margin,
            start_delay_sim=record.compute_start_sim - task.arrival_time_sim,
            preferred_start_tardiness_ratio=(
                record.preferred_start_tardiness_ratio
            ),
            preferred_start_tardiness_applicable=(
                record.preferred_start_tardiness_applicable
            ),
            cpu_demand=task.cpu_demand,
            bandwidth_demand_mbps=task.bandwidth_demand_mbps,
        )

    def encode_records_batch(self, context, records, metrics) -> np.ndarray:
        """Encode an aligned record batch with the scalar feature semantics."""

        items = tuple(records)
        count = len(items)
        if count == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        def metric_values(name, default=0.0, dtype=np.float64):
            raw = metrics.get(name)
            if raw is None:
                return np.full(count, default, dtype=dtype)
            values = np.asarray(raw, dtype=dtype)
            if values.shape != (count,):
                raise ValueError(
                    f"batch candidate metric {name} is not aligned"
                )
            return values

        try:
            node_indices = np.asarray(
                [self.node_index[item.target_node] for item in items],
                dtype=np.float64,
            )
        except KeyError as error:
            raise ValueError(
                f"unknown candidate target node {error.args[0]}"
            )
        max_node = max(self.node_index.values())
        if max_node == 0:
            node_normalized = np.zeros(count, dtype=np.float64)
        else:
            node_normalized = node_indices / max_node

        task = context.task
        starts = np.asarray(
            [item.compute_start_sim for item in items],
            dtype=np.float64,
        )
        ends = np.asarray(
            [item.compute_end_sim for item in items],
            dtype=np.float64,
        )
        transmission_starts = np.asarray(
            [item.transmission_start_sim for item in items],
            dtype=np.float64,
        )
        tardiness = np.asarray(
            [item.preferred_start_tardiness_ratio for item in items],
            dtype=np.float64,
        )
        tardiness_applicable = np.asarray(
            [
                1.0 if item.preferred_start_tardiness_applicable else 0.0
                for item in items
            ],
            dtype=np.float64,
        )
        node_utilization = np.asarray(
            [item.projected_node_utilization for item in items],
            dtype=np.float64,
        )
        path_utilization = np.asarray(
            [item.projected_path_peak_utilization for item in items],
            dtype=np.float64,
        )
        capacity_margin = np.asarray(
            [item.capacity_margin for item in items],
            dtype=np.float64,
        )
        green_opportunity = metric_values(
            "green_opportunity", default=False, dtype=np.bool_
        ).astype(np.float64)

        matrix = np.column_stack(
            (
                node_normalized,
                (starts - context.decision_time_sim)
                / self.config.time_scale_sim,
                (starts - context.earliest_compute_start_sim)
                / self.config.time_scale_sim,
                (starts - transmission_starts)
                / self.config.time_scale_sim,
                (ends - starts) / self.config.time_scale_sim,
                metric_values("system_cost_yuan")
                / self.config.cost_scale_yuan,
                metric_values("green_coverage"),
                metric_values("green_absorption_delta")
                / self.config.absorption_delta_scale,
                green_opportunity,
                node_utilization,
                path_utilization,
                capacity_margin,
                (starts - task.arrival_time_sim)
                / self.config.time_scale_sim,
                tardiness,
                tardiness_applicable,
                (
                    np.abs(starts - context.earliest_compute_start_sim)
                    <= 1e-12
                ).astype(np.float64),
                np.full(
                    count,
                    task.cpu_demand / self.config.cpu_scale,
                    dtype=np.float64,
                ),
                np.full(
                    count,
                    task.bandwidth_demand_mbps
                    / self.config.bandwidth_scale_mbps,
                    dtype=np.float64,
                ),
            )
        )
        return self._mask_matrix(matrix).astype(np.float32, copy=False)

    def encode_candidate_arrays(
        self,
        context,
        *,
        target_node,
        compute_start_sim,
        compute_end_sim,
        transmission_start_sim,
        preferred_start_tardiness_ratio,
        preferred_start_tardiness_applicable,
        projected_node_utilization,
        projected_path_peak_utilization,
        capacity_margin,
        metrics,
    ) -> np.ndarray:
        """Encode vectorized candidate columns in canonical feature order."""

        starts = np.asarray(compute_start_sim, dtype=np.float64)
        ends = np.asarray(compute_end_sim, dtype=np.float64)
        transmission_starts = np.asarray(
            transmission_start_sim, dtype=np.float64
        )
        arrays = (
            ends,
            transmission_starts,
            np.asarray(
                preferred_start_tardiness_ratio, dtype=np.float64
            ),
            np.asarray(
                preferred_start_tardiness_applicable, dtype=np.bool_
            ),
            np.asarray(projected_node_utilization, dtype=np.float64),
            np.asarray(
                projected_path_peak_utilization, dtype=np.float64
            ),
            np.asarray(capacity_margin, dtype=np.float64),
        )
        if starts.ndim != 1 or any(
            value.shape != starts.shape for value in arrays
        ):
            raise ValueError(
                "candidate feature arrays must be one-dimensional and aligned"
            )
        count = starts.size
        if isinstance(target_node, str):
            try:
                node_value = self.node_index[target_node]
            except KeyError:
                raise ValueError(
                    f"unknown candidate target node {target_node}"
                )
            node_indices = np.full(count, node_value, dtype=np.float64)
        else:
            try:
                node_indices = np.asarray(
                    [self.node_index[str(node)] for node in target_node],
                    dtype=np.float64,
                )
            except KeyError as error:
                raise ValueError(
                    f"unknown candidate target node {error.args[0]}"
                )
            if node_indices.shape != starts.shape:
                raise ValueError("candidate target nodes are not aligned")
        max_node = max(self.node_index.values())
        node_normalized = (
            np.zeros(count, dtype=np.float64)
            if max_node == 0
            else node_indices / max_node
        )

        def metric_values(name, default=0.0, dtype=np.float64):
            raw = metrics.get(name)
            if raw is None:
                return np.full(count, default, dtype=dtype)
            values = np.asarray(raw, dtype=dtype)
            if values.shape != starts.shape:
                raise ValueError(
                    f"batch candidate metric {name} is not aligned"
                )
            return values

        task = context.task
        matrix = np.column_stack(
            (
                node_normalized,
                (starts - context.decision_time_sim)
                / self.config.time_scale_sim,
                (starts - context.earliest_compute_start_sim)
                / self.config.time_scale_sim,
                (starts - transmission_starts)
                / self.config.time_scale_sim,
                (ends - starts) / self.config.time_scale_sim,
                metric_values("system_cost_yuan")
                / self.config.cost_scale_yuan,
                metric_values("green_coverage"),
                metric_values("green_absorption_delta")
                / self.config.absorption_delta_scale,
                metric_values(
                    "green_opportunity", default=False, dtype=np.bool_
                ).astype(np.float64),
                arrays[4],
                arrays[5],
                arrays[6],
                (starts - task.arrival_time_sim)
                / self.config.time_scale_sim,
                arrays[2],
                arrays[3].astype(np.float64),
                (
                    np.abs(starts - context.earliest_compute_start_sim)
                    <= 1e-12
                ).astype(np.float64),
                np.full(
                    count,
                    task.cpu_demand / self.config.cpu_scale,
                    dtype=np.float64,
                ),
                np.full(
                    count,
                    task.bandwidth_demand_mbps
                    / self.config.bandwidth_scale_mbps,
                    dtype=np.float64,
                ),
            )
        )
        return self._mask_matrix(matrix).astype(np.float32, copy=False)

    def _encode_values(
        self,
        *,
        target_node,
        decision_time_sim,
        compute_start_sim,
        compute_end_sim,
        transmission_start_sim,
        earliest_compute_start_sim,
        marginal_cost_yuan,
        green_coverage,
        green_absorption_delta,
        green_opportunity,
        projected_node_utilization,
        projected_path_peak_utilization,
        capacity_margin,
        start_delay_sim,
        preferred_start_tardiness_ratio,
        preferred_start_tardiness_applicable,
        cpu_demand,
        bandwidth_demand_mbps,
    ) -> Tuple[float, ...]:
        try:
            node = self.node_index[target_node]
        except KeyError:
            raise ValueError(f"unknown candidate target node {target_node}")
        max_node = max(self.node_index.values())
        node_normalized = 0.0 if max_node == 0 else node / max_node
        transmission_duration = compute_start_sim - transmission_start_sim
        return self._mask_values((
            node_normalized,
            (compute_start_sim - decision_time_sim) / self.config.time_scale_sim,
            (compute_start_sim - earliest_compute_start_sim) / self.config.time_scale_sim,
            transmission_duration / self.config.time_scale_sim,
            (compute_end_sim - compute_start_sim) / self.config.time_scale_sim,
            marginal_cost_yuan / self.config.cost_scale_yuan,
            green_coverage,
            green_absorption_delta / self.config.absorption_delta_scale,
            1.0 if green_opportunity else 0.0,
            projected_node_utilization,
            projected_path_peak_utilization,
            capacity_margin,
            start_delay_sim / self.config.time_scale_sim,
            preferred_start_tardiness_ratio,
            1.0 if preferred_start_tardiness_applicable else 0.0,
            1.0 if abs(compute_start_sim - earliest_compute_start_sim) <= 1e-12 else 0.0,
            cpu_demand / self.config.cpu_scale,
            bandwidth_demand_mbps / self.config.bandwidth_scale_mbps,
        ))
