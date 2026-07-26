"""Immutable v1.0 task schema and explicit legacy migration adapters."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, Mapping, Optional

from .units import (
    TimeConverter,
    UnitValidationError,
    cpu_work_cpu_hours,
    cpu_work_sim_units,
    finite_number,
    non_negative_finite,
    positive_finite,
)


TASK_SCHEMA_VERSION = "1.0"
SOFT_LATEST_START_MULTIPLIER = 1.2
FLEXIBLE_LATEST_START_MULTIPLIER = 1.5


class TaskValidationError(ValueError):
    """Field-addressable task validation or migration failure."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        self.field_errors = {field: message}
        super().__init__(f"{field}: {message}")


@dataclass(frozen=True)
class TaskValidationResult:
    valid: bool
    task_spec: Optional["TaskSpec"]
    terminal_reason: Optional[str]
    field_errors: Mapping[str, str]


class SlaType(str, Enum):
    HARD = "Hard"
    SOFT = "Soft"
    FLEXIBLE = "Flexible"

    @classmethod
    def parse(cls, value: Any) -> "SlaType":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except (TypeError, ValueError):
            raise TaskValidationError(
                "sla_type",
                "must be one of Hard, Soft, Flexible",
            )


class TaskState(str, Enum):
    ARRIVED = "Arrived"
    QUEUED = "Queued"
    PENDING_UNCOMMITTED = "PendingUncommitted"
    RESERVED = "Reserved"
    TRANSMITTING = "Transmitting"
    RUNNING = "Running"
    COMPLETED = "Completed"
    REJECTED = "Rejected"
    EXPIRED = "Expired"
    FAILED = "Failed"


class MetricStatus(str, Enum):
    VALID = "VALID"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INVALID = "INVALID"


@dataclass(frozen=True)
class MetricValue:
    value: Optional[float]
    status: MetricStatus
    reason: Optional[str] = None
    numerator: Optional[float] = None
    denominator: Optional[float] = None

    @classmethod
    def valid(
        cls,
        value: float,
        numerator: Optional[float] = None,
        denominator: Optional[float] = None,
    ) -> "MetricValue":
        try:
            finite_value = finite_number("metric_value", value)
        except UnitValidationError as exc:
            raise TaskValidationError(exc.field, exc.message)
        return cls(finite_value, MetricStatus.VALID, None, numerator, denominator)

    @classmethod
    def not_applicable(cls, reason: str) -> "MetricValue":
        return cls(None, MetricStatus.NOT_APPLICABLE, reason)

    @classmethod
    def invalid(cls, reason: str) -> "MetricValue":
        return cls(None, MetricStatus.INVALID, reason)


def _field_number(
    field: str,
    value: Any,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    try:
        if positive:
            return positive_finite(field, value)
        if non_negative:
            return non_negative_finite(field, value)
        return finite_number(field, value)
    except UnitValidationError as exc:
        raise TaskValidationError(exc.field, exc.message)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    arrival_time_sim: float
    source_node: str
    cpu_demand: float
    execution_duration_sim: float
    data_size_mb: float
    bandwidth_demand_mbps: float
    sla_type: SlaType
    preferred_start_limit_sim: Optional[float]
    latest_start_limit_sim: float
    schema_version: str = TASK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise TaskValidationError("task_id", "must be a non-empty string")
        if not isinstance(self.source_node, str) or not self.source_node.strip():
            raise TaskValidationError("source_node", "must be a non-empty string")
        if not isinstance(self.sla_type, SlaType):
            raise TaskValidationError("sla_type", "must be a SlaType")
        if self.schema_version != TASK_SCHEMA_VERSION:
            raise TaskValidationError("schema_version", 'must equal "1.0"')

        object.__setattr__(
            self,
            "arrival_time_sim",
            _field_number("arrival_time_sim", self.arrival_time_sim),
        )
        object.__setattr__(
            self,
            "cpu_demand",
            _field_number("cpu_demand", self.cpu_demand, positive=True),
        )
        object.__setattr__(
            self,
            "execution_duration_sim",
            _field_number(
                "execution_duration_sim",
                self.execution_duration_sim,
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "data_size_mb",
            _field_number("data_size_mb", self.data_size_mb, non_negative=True),
        )
        object.__setattr__(
            self,
            "bandwidth_demand_mbps",
            _field_number(
                "bandwidth_demand_mbps",
                self.bandwidth_demand_mbps,
                positive=True,
            ),
        )
        latest = _field_number(
            "latest_start_limit_sim",
            self.latest_start_limit_sim,
            non_negative=True,
        )
        object.__setattr__(self, "latest_start_limit_sim", latest)

        if self.sla_type is SlaType.HARD:
            if self.preferred_start_limit_sim is not None:
                raise TaskValidationError(
                    "preferred_start_limit_sim",
                    "must be omitted for Hard tasks",
                )
            return

        preferred = _field_number(
            "preferred_start_limit_sim",
            self.preferred_start_limit_sim,
            positive=True,
        )
        object.__setattr__(self, "preferred_start_limit_sim", preferred)
        multiplier = (
            SOFT_LATEST_START_MULTIPLIER
            if self.sla_type is SlaType.SOFT
            else FLEXIBLE_LATEST_START_MULTIPLIER
        )
        expected_latest = multiplier * preferred
        if not math.isclose(latest, expected_latest, rel_tol=1e-12, abs_tol=1e-12):
            raise TaskValidationError(
                "latest_start_limit_sim",
                f"must equal {multiplier} * preferred_start_limit_sim",
            )

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        arrival_time_sim: Any,
        source_node: str,
        cpu_demand: Any,
        execution_duration_sim: Any,
        data_size_mb: Any,
        bandwidth_demand_mbps: Any,
        sla_type: Any,
        preferred_start_limit_sim: Optional[Any] = None,
        latest_start_limit_sim: Optional[Any] = None,
    ) -> "TaskSpec":
        parsed_sla = SlaType.parse(sla_type)
        if parsed_sla is SlaType.HARD:
            if latest_start_limit_sim is None:
                raise TaskValidationError(
                    "latest_start_limit_sim",
                    "is required for Hard tasks",
                )
            derived_latest = latest_start_limit_sim
        else:
            if preferred_start_limit_sim is None:
                raise TaskValidationError(
                    "preferred_start_limit_sim",
                    "is required for Soft and Flexible tasks",
                )
            preferred = _field_number(
                "preferred_start_limit_sim",
                preferred_start_limit_sim,
                positive=True,
            )
            multiplier = (
                SOFT_LATEST_START_MULTIPLIER
                if parsed_sla is SlaType.SOFT
                else FLEXIBLE_LATEST_START_MULTIPLIER
            )
            derived_latest = multiplier * preferred
            if latest_start_limit_sim is not None:
                explicit_latest = _field_number(
                    "latest_start_limit_sim",
                    latest_start_limit_sim,
                    non_negative=True,
                )
                if not math.isclose(
                    explicit_latest,
                    derived_latest,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise TaskValidationError(
                        "latest_start_limit_sim",
                        f"must equal {multiplier} * preferred_start_limit_sim",
                    )

        return cls(
            task_id=task_id,
            arrival_time_sim=arrival_time_sim,
            source_node=source_node,
            cpu_demand=cpu_demand,
            execution_duration_sim=execution_duration_sim,
            data_size_mb=data_size_mb,
            bandwidth_demand_mbps=bandwidth_demand_mbps,
            sla_type=parsed_sla,
            preferred_start_limit_sim=preferred_start_limit_sim,
            latest_start_limit_sim=derived_latest,
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TaskSpec":
        legacy_fields = (
            "id",
            "generated_time",
            "cpu",
            "duration",
            "data_size",
            "bw",
            "latency_limit",
        )
        for field in legacy_fields:
            if field in values:
                raise TaskValidationError(
                    field,
                    "legacy field is not allowed in v1.0 TaskSpec; use migrate_legacy_task",
                )
        declared_version = values.get("task_schema_version", TASK_SCHEMA_VERSION)
        if declared_version != TASK_SCHEMA_VERSION:
            raise TaskValidationError(
                "task_schema_version",
                'must equal "1.0"',
            )
        required = (
            "task_id",
            "arrival_time_sim",
            "source_node",
            "cpu_demand",
            "execution_duration_sim",
            "data_size_mb",
            "bandwidth_demand_mbps",
            "sla_type",
        )
        for field in required:
            if field not in values:
                raise TaskValidationError(field, "is required")
        return cls.create(
            task_id=values["task_id"],
            arrival_time_sim=values["arrival_time_sim"],
            source_node=values["source_node"],
            cpu_demand=values["cpu_demand"],
            execution_duration_sim=values["execution_duration_sim"],
            data_size_mb=values["data_size_mb"],
            bandwidth_demand_mbps=values["bandwidth_demand_mbps"],
            sla_type=values["sla_type"],
            preferred_start_limit_sim=values.get("preferred_start_limit_sim"),
            latest_start_limit_sim=values.get("latest_start_limit_sim"),
        )

    @property
    def absolute_preferred_start_sim(self) -> Optional[float]:
        if self.preferred_start_limit_sim is None:
            return None
        return self.arrival_time_sim + self.preferred_start_limit_sim

    @property
    def absolute_latest_start_sim(self) -> float:
        return self.arrival_time_sim + self.latest_start_limit_sim

    @property
    def cpu_work_sim_units(self) -> float:
        return cpu_work_sim_units(self.cpu_demand, self.execution_duration_sim)

    def cpu_work_cpu_hours(self, converter: TimeConverter) -> float:
        return cpu_work_cpu_hours(
            self.cpu_demand,
            self.execution_duration_sim,
            converter,
        )


@dataclass
class TaskRuntime:
    """Mutable execution state kept separate from immutable TaskSpec."""

    task_id: str
    state: TaskState = TaskState.ARRIVED
    state_version: int = 0
    last_state_change_sim: float = 0.0
    pending_attempts: int = 0
    commit_attempts_current_decision: int = 0
    failure_retry_count: int = 0
    terminal_reason: Optional[str] = None
    reservation_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise TaskValidationError("task_id", "must be a non-empty string")
        if not isinstance(self.state, TaskState):
            raise TaskValidationError("state", "must be a TaskState")
        self.last_state_change_sim = _field_number(
            "last_state_change_sim",
            self.last_state_change_sim,
        )
        for field in (
            "state_version",
            "pending_attempts",
            "commit_attempts_current_decision",
            "failure_retry_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TaskValidationError(field, "must be a non-negative integer")


def _legacy_value(values: Mapping[str, Any], canonical: str, legacy: str) -> Any:
    if canonical in values:
        return values[canonical]
    if legacy in values:
        return values[legacy]
    raise TaskValidationError(canonical, f"is required (legacy alias: {legacy})")


def migrate_legacy_task(values: Mapping[str, Any]) -> TaskSpec:
    """Migrate one legacy task dict into a validated immutable TaskSpec.

    Historical ``None`` tasks must carry a valid positive ``latency_limit``;
    that value becomes Flexible's preferred limit and the absolute limit is
    derived as 1.5L.  The adapter never invents a missing limit.
    """

    if not isinstance(values, Mapping):
        raise TaskValidationError("task", "must be a mapping")
    raw_sla = values.get("sla_type")
    if raw_sla is None:
        raise TaskValidationError("sla_type", "is required")

    if raw_sla == "None":
        if "latency_limit" not in values:
            raise TaskValidationError(
                "latency_limit",
                "is required to migrate legacy None to Flexible",
            )
        sla_type = SlaType.FLEXIBLE
        preferred = values["latency_limit"]
        latest = values.get("latest_start_limit_sim")
    else:
        sla_type = SlaType.parse(raw_sla)
        if sla_type is SlaType.HARD:
            preferred = values.get("preferred_start_limit_sim")
            latest = values.get("latest_start_limit_sim", values.get("latency_limit"))
        else:
            preferred = values.get(
                "preferred_start_limit_sim",
                values.get("latency_limit"),
            )
            latest = values.get("latest_start_limit_sim")

    raw_task_id = _legacy_value(values, "task_id", "id")
    if raw_task_id is None or str(raw_task_id).strip() == "":
        raise TaskValidationError("task_id", "must be present and non-empty")

    return TaskSpec.create(
        task_id=str(raw_task_id),
        arrival_time_sim=_legacy_value(values, "arrival_time_sim", "generated_time"),
        source_node=_legacy_value(values, "source_node", "source_node"),
        cpu_demand=_legacy_value(values, "cpu_demand", "cpu"),
        execution_duration_sim=_legacy_value(
            values,
            "execution_duration_sim",
            "duration",
        ),
        data_size_mb=_legacy_value(values, "data_size_mb", "data_size"),
        bandwidth_demand_mbps=_legacy_value(
            values,
            "bandwidth_demand_mbps",
            "bw",
        ),
        sla_type=sla_type,
        preferred_start_limit_sim=preferred,
        latest_start_limit_sim=latest,
    )


def validate_task_mapping(values: Mapping[str, Any], legacy: bool = False) -> TaskValidationResult:
    """Pure validation boundary used before admission/state registration."""

    try:
        task = migrate_legacy_task(values) if legacy else TaskSpec.from_mapping(values)
        return TaskValidationResult(True, task, None, {})
    except TaskValidationError as exc:
        return TaskValidationResult(
            False,
            None,
            "INVALID_TASK",
            dict(exc.field_errors),
        )


def to_legacy_task_dict(
    task_spec: TaskSpec,
    original: Optional[Mapping[str, Any]] = None,
    legacy_latency_limit: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the explicit legacy scheduler view used during WP-1 shadow mode."""

    result: Dict[str, Any] = dict(original or {})
    if legacy_latency_limit is None:
        legacy_latency_limit = (
            task_spec.latest_start_limit_sim
            if task_spec.sla_type is SlaType.HARD
            else task_spec.preferred_start_limit_sim
        )
    result.update({
        "id": task_spec.task_id,
        "generated_time": task_spec.arrival_time_sim,
        "source_node": task_spec.source_node,
        "cpu": task_spec.cpu_demand,
        "duration": task_spec.execution_duration_sim,
        "cpu_time": task_spec.cpu_work_sim_units,
        "data_size": task_spec.data_size_mb,
        "bw": task_spec.bandwidth_demand_mbps,
        "sla_type": task_spec.sla_type.value,
        "latency_limit": legacy_latency_limit,
        "task_schema_version": TASK_SCHEMA_VERSION,
        "task_adapter_mode": "legacy_shadow",
        "task_id": task_spec.task_id,
        "arrival_time_sim": task_spec.arrival_time_sim,
        "cpu_demand": task_spec.cpu_demand,
        "execution_duration_sim": task_spec.execution_duration_sim,
        "data_size_mb": task_spec.data_size_mb,
        "bandwidth_demand_mbps": task_spec.bandwidth_demand_mbps,
        "preferred_start_limit_sim": task_spec.preferred_start_limit_sim,
        "latest_start_limit_sim": task_spec.latest_start_limit_sim,
        "absolute_preferred_start_sim": task_spec.absolute_preferred_start_sim,
        "absolute_latest_start_sim": task_spec.absolute_latest_start_sim,
    })
    return result
