"""Pure v1.0 SLA derivation, feasibility, and tardiness evaluation."""

from dataclasses import dataclass
from typing import Optional, Tuple

from .models import MetricStatus, MetricValue, SlaType, TaskSpec
from .units import UnitValidationError, finite_number


@dataclass(frozen=True)
class SlaLimits:
    preferred_start_limit_sim: Optional[float]
    latest_start_limit_sim: float
    absolute_preferred_start_sim: Optional[float]
    absolute_latest_start_sim: float


class SlaPolicy:
    """SLA rules independent from candidate scores and reward weights."""

    @staticmethod
    def derive_limits(task_spec: TaskSpec) -> SlaLimits:
        return SlaLimits(
            preferred_start_limit_sim=task_spec.preferred_start_limit_sim,
            latest_start_limit_sim=task_spec.latest_start_limit_sim,
            absolute_preferred_start_sim=task_spec.absolute_preferred_start_sim,
            absolute_latest_start_sim=task_spec.absolute_latest_start_sim,
        )

    @staticmethod
    def _compute_start(compute_start_sim: float) -> float:
        try:
            return finite_number("compute_start_sim", compute_start_sim)
        except UnitValidationError as exc:
            raise ValueError(str(exc))

    @classmethod
    def is_start_feasible(cls, task_spec: TaskSpec, compute_start_sim: float) -> bool:
        compute_start = cls._compute_start(compute_start_sim)
        return (
            compute_start >= task_spec.arrival_time_sim
            and compute_start <= task_spec.absolute_latest_start_sim
        )

    @classmethod
    def preferred_start_tardiness(
        cls,
        task_spec: TaskSpec,
        compute_start_sim: float,
    ) -> MetricValue:
        compute_start = cls._compute_start(compute_start_sim)
        if compute_start < task_spec.arrival_time_sim:
            return MetricValue.invalid("compute_start_sim is earlier than arrival_time_sim")
        if task_spec.sla_type is SlaType.HARD:
            return MetricValue.not_applicable(
                "Hard tasks do not have a preferred start limit"
            )

        start_delay = compute_start - task_spec.arrival_time_sim
        preferred = task_spec.preferred_start_limit_sim
        latest = task_spec.latest_start_limit_sim
        denominator = latest - preferred
        ratio = min(1.0, max(0.0, (start_delay - preferred) / denominator))
        return MetricValue.valid(
            ratio,
            numerator=start_delay - preferred,
            denominator=denominator,
        )

    @classmethod
    def tardiness_model_feature(
        cls,
        task_spec: TaskSpec,
        compute_start_sim: float,
    ) -> Tuple[float, bool]:
        metric = cls.preferred_start_tardiness(task_spec, compute_start_sim)
        if metric.status is MetricStatus.NOT_APPLICABLE:
            return 0.0, False
        if metric.value is None:
            raise ValueError(metric.reason or "invalid tardiness metric")
        return metric.value, True
