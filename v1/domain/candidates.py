"""Immutable v1.0 node-time-path candidate schema."""

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from functools import lru_cache
from typing import Tuple

from .reservations import PathSpec, ReservationRequest, TimeInterval


CANDIDATE_SCHEMA_VERSION = "1.0"


@lru_cache(maxsize=8192)
def _canonical_json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class CandidateMode(str, Enum):
    COMPLETE = "complete"
    APPROXIMATE = "approximate"


class CandidateGenerationStatus(str, Enum):
    OK = "OK"
    EMPTY_PHYSICAL = "EMPTY_PHYSICAL"
    EXPIRED_BEFORE_DECISION = "EXPIRED_BEFORE_DECISION"
    INVALID_TASK = "INVALID_TASK"
    FORECAST_NOT_COVERED = "FORECAST_NOT_COVERED"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    candidate_schema_version: str
    candidate_mode: CandidateMode
    task_id: str
    decision_time_sim: float
    reservation_snapshot_version: int
    forecast_version: str
    target_node: str
    path: PathSpec
    transmission_start_sim: float
    transmission_end_sim: float
    compute_start_sim: float
    compute_end_sim: float
    cpu_demand: float
    bandwidth_demand_mbps: float
    scheduler_queue_delay_sim: float
    earliest_feasibility_lead_sim: float
    active_wait_sim: float
    reservation_lead_sim: float
    start_delay_sim: float
    preferred_start_tardiness_ratio: float
    preferred_start_tardiness_applicable: bool
    estimated_candidate_marginal_system_cost_yuan: float
    estimated_green_coverage: float
    estimated_candidate_marginal_green_energy_mwh: float
    estimated_green_absorption_delta: float
    estimated_green_opportunity: bool
    projected_node_utilization: float
    projected_path_peak_utilization: float
    capacity_margin: float

    def to_reservation_request(self) -> ReservationRequest:
        tx_interval = None
        if not self.path.is_local:
            tx_interval = TimeInterval(
                self.transmission_start_sim,
                self.transmission_end_sim,
            )
        return ReservationRequest(
            task_id=self.task_id,
            committed_candidate_id=self.candidate_id,
            committed_at_sim=self.decision_time_sim,
            reservation_snapshot_version=self.reservation_snapshot_version,
            target_node=self.target_node,
            path=self.path,
            transmission_interval_sim=tx_interval,
            compute_interval_sim=TimeInterval(
                self.compute_start_sim,
                self.compute_end_sim,
            ),
            bandwidth_amount_mbps=self.bandwidth_demand_mbps,
            cpu_amount=self.cpu_demand,
        )


@dataclass(frozen=True)
class CandidateSetResult:
    status: CandidateGenerationStatus
    candidate_mode: CandidateMode
    candidates: Tuple[Candidate, ...]
    theoretical_slot_count: int
    feasible_candidate_count: int
    earliest_compute_start_sim: float
    reason: str = ""


def deterministic_candidate_id(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "candidate-" + hashlib.sha256(encoded).hexdigest()


def deterministic_candidate_id_v1_fields(
    *,
    compute_end: float,
    compute_start: float,
    forecast_version: str,
    mode: str,
    node: str,
    path: str,
    reservation_version: int,
    schema: str,
    task: str,
    transmission_end: float,
    transmission_start: float,
) -> str:
    """Fast canonical encoder for the frozen v1 candidate-id payload.

    Key order and scalar rendering exactly match ``deterministic_candidate_id``
    with ``sort_keys=True``. Candidate times are already finite validated floats.
    """

    encoded = (
        '{"compute_end":' + repr(float(compute_end))
        + ',"compute_start":' + repr(float(compute_start))
        + ',"forecast_version":' + _canonical_json_string(forecast_version)
        + ',"mode":' + _canonical_json_string(mode)
        + ',"node":' + _canonical_json_string(node)
        + ',"path":' + _canonical_json_string(path)
        + ',"reservation_version":' + str(int(reservation_version))
        + ',"schema":' + _canonical_json_string(schema)
        + ',"task":' + _canonical_json_string(task)
        + ',"transmission_end":' + repr(float(transmission_end))
        + ',"transmission_start":' + repr(float(transmission_start))
        + "}"
    ).encode("utf-8")
    return "candidate-" + hashlib.sha256(encoded).hexdigest()
