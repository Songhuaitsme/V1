"""Deterministic v1.0 candidate policies; non-empty input always selects."""

from dataclasses import dataclass
import hashlib
from typing import Iterable, Sequence

from v1.domain.candidates import Candidate
from v1.domain.models import SlaType

from .objectives import ObjectiveConfig, ObjectiveScorer


def _require_candidates(candidates: Iterable[Candidate]) -> tuple:
    items = tuple(candidates)
    if not items:
        raise ValueError("candidate policy requires a non-empty candidate set")
    return items


def _stable_tail(candidate: Candidate):
    return (
        candidate.compute_start_sim,
        candidate.target_node,
        candidate.path.path_id,
        candidate.candidate_id,
    )


@dataclass(frozen=True)
class CandidateStreamSelection:
    selected_candidate: Candidate
    earliest_candidate: Candidate
    candidate_count: int
    candidate_set_hash: str
    candidate_context: object = None


def _select_stream(candidates, key, context=None):
    selected = None
    earliest = None
    count = 0
    digest = hashlib.sha256()
    for candidate in candidates:
        count += 1
        digest.update(candidate.candidate_id.encode("utf-8"))
        digest.update(b"\0")
        if earliest is None or _stable_tail(candidate) < _stable_tail(earliest):
            earliest = candidate
        if selected is None or key(candidate) < key(selected):
            selected = candidate
    if selected is None:
        raise ValueError("candidate policy requires a non-empty candidate set")
    return CandidateStreamSelection(
        selected,
        earliest,
        count,
        digest.hexdigest(),
        context,
    )


class EarliestFeasiblePolicy:
    name = "earliest_feasible"

    def select(self, candidates: Sequence[Candidate], task=None) -> Candidate:
        return min(_require_candidates(candidates), key=_stable_tail)

    def select_stream(self, candidates, task=None, context=None):
        return _select_stream(candidates, _stable_tail, context)


class LowestCostPolicy:
    name = "lowest_cost"

    def select(self, candidates: Sequence[Candidate], task=None) -> Candidate:
        return min(
            _require_candidates(candidates),
            key=lambda item: (
                item.estimated_candidate_marginal_system_cost_yuan,
                *_stable_tail(item),
            ),
        )

    def select_stream(self, candidates, task=None, context=None):
        return _select_stream(
            candidates,
            lambda item: (
                item.estimated_candidate_marginal_system_cost_yuan,
                *_stable_tail(item),
            ),
            context,
        )


class HighestGreenPolicy:
    name = "highest_green"

    def select(self, candidates: Sequence[Candidate], task=None) -> Candidate:
        return min(
            _require_candidates(candidates),
            key=lambda item: (
                -item.estimated_green_coverage,
                -item.estimated_green_absorption_delta,
                *_stable_tail(item),
            ),
        )

    def select_stream(self, candidates, task=None, context=None):
        return _select_stream(
            candidates,
            lambda item: (
                -item.estimated_green_coverage,
                -item.estimated_green_absorption_delta,
                *_stable_tail(item),
            ),
            context,
        )


class EqualWeightPolicy:
    """Fixed-scale cost/green objective; never derives scales from test candidates."""

    name = "equal_weight"

    def __init__(self, objective_config=None):
        self.objective_config = objective_config or ObjectiveConfig(
            reference_marginal_cost_yuan=0.0,
            cost_scale_yuan=1.0,
            absorption_delta_scale=1.0,
        )
        self.scorer = ObjectiveScorer(self.objective_config)
        self.policy_id = "equal-weight-" + self.objective_config.policy_id

    def select(self, candidates: Sequence[Candidate], task=None) -> Candidate:
        items = _require_candidates(candidates)
        sla_type = SlaType.HARD if task is None else task.sla_type
        return min(
            items,
            key=lambda item: (
                -self.scorer.score(item, sla_type).total_score,
                *_stable_tail(item),
            ),
        )

    def select_stream(self, candidates, task=None, context=None):
        sla_type = SlaType.HARD if task is None else task.sla_type
        return _select_stream(
            candidates,
            lambda item: (
                -self.scorer.score(item, sla_type).total_score,
                *_stable_tail(item),
            ),
            context,
        )
