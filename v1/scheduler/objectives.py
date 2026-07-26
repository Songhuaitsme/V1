"""Fixed-scale v1.0 multi-objective scoring and Pareto diagnostics."""

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Tuple

from v1.domain.candidates import Candidate
from v1.domain.models import SlaType
from v1.domain.units import finite_number, non_negative_finite, positive_finite


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class ObjectiveBreakdown:
    cost_score: float
    green_coverage_score: float
    green_absorption_score: float
    green_score: float
    balance_score: float
    preferred_start_tardiness_penalty: float
    total_score: float


@dataclass(frozen=True)
class ObjectiveConfig:
    reference_marginal_cost_yuan: float
    cost_scale_yuan: float
    absorption_delta_scale: float
    cost_weight: float = 0.5
    green_weight: float = 0.5
    balance_weight: float = 0.1
    soft_tardiness_weight: float = 0.0
    flexible_tardiness_weight: float = 0.0

    def __post_init__(self):
        object.__setattr__(
            self,
            "reference_marginal_cost_yuan",
            finite_number(
                "reference_marginal_cost_yuan",
                self.reference_marginal_cost_yuan,
            ),
        )
        object.__setattr__(
            self,
            "cost_scale_yuan",
            positive_finite("cost_scale_yuan", self.cost_scale_yuan),
        )
        object.__setattr__(
            self,
            "absorption_delta_scale",
            positive_finite(
                "absorption_delta_scale",
                self.absorption_delta_scale,
            ),
        )
        for field in (
            "cost_weight",
            "green_weight",
            "balance_weight",
            "soft_tardiness_weight",
            "flexible_tardiness_weight",
        ):
            object.__setattr__(
                self,
                field,
                non_negative_finite(field, getattr(self, field)),
            )
        if self.cost_weight + self.green_weight <= 0.0:
            raise ValueError("cost_weight + green_weight must be positive")

    @property
    def policy_id(self) -> str:
        payload = json.dumps(
            self.__dict__,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "objective-" + hashlib.sha256(payload).hexdigest()[:16]


class ObjectiveScorer:
    def __init__(self, config: ObjectiveConfig):
        self.config = config

    def score(self, candidate: Candidate, sla_type: SlaType) -> ObjectiveBreakdown:
        if not isinstance(sla_type, SlaType):
            sla_type = SlaType.parse(sla_type)
        cost_score = _clip(
            (
                self.config.reference_marginal_cost_yuan
                - candidate.estimated_candidate_marginal_system_cost_yuan
            ) / self.config.cost_scale_yuan,
            -1.0,
            1.0,
        )
        coverage = _clip(candidate.estimated_green_coverage, 0.0, 1.0)
        absorption = (
            _clip(
                candidate.estimated_green_absorption_delta
                / self.config.absorption_delta_scale,
                0.0,
                1.0,
            )
            if candidate.estimated_green_opportunity
            else 0.0
        )
        green_score = 0.5 * coverage + 0.5 * absorption
        balance = _clip(candidate.capacity_margin, 0.0, 1.0)
        tardiness_weight = {
            SlaType.HARD: 0.0,
            SlaType.SOFT: self.config.soft_tardiness_weight,
            SlaType.FLEXIBLE: self.config.flexible_tardiness_weight,
        }[sla_type]
        tardiness_penalty = (
            tardiness_weight * candidate.preferred_start_tardiness_ratio
            if candidate.preferred_start_tardiness_applicable
            else 0.0
        )
        total = (
            self.config.cost_weight * cost_score
            + self.config.green_weight * green_score
            + self.config.balance_weight * balance
            - tardiness_penalty
        )
        return ObjectiveBreakdown(
            cost_score,
            coverage,
            absorption,
            green_score,
            balance,
            tardiness_penalty,
            total,
        )


def pareto_frontier(candidates: Iterable[Candidate]) -> Tuple[Candidate, ...]:
    """Cost-minimizing, green-score-maximizing non-dominated candidates."""

    items = tuple(candidates)
    frontier = []
    for candidate in items:
        candidate_green = (
            candidate.estimated_green_coverage
            + candidate.estimated_green_absorption_delta
        )
        dominated = False
        for other in items:
            if other is candidate:
                continue
            other_green = (
                other.estimated_green_coverage
                + other.estimated_green_absorption_delta
            )
            no_worse = (
                other.estimated_candidate_marginal_system_cost_yuan
                <= candidate.estimated_candidate_marginal_system_cost_yuan
                and other_green >= candidate_green
            )
            strictly_better = (
                other.estimated_candidate_marginal_system_cost_yuan
                < candidate.estimated_candidate_marginal_system_cost_yuan
                or other_green > candidate_green
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return tuple(sorted(frontier, key=lambda item: item.candidate_id))
