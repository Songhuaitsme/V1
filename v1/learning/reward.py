"""Physical-time event reward assembly without retroactive replay mutation."""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from v1.domain.units import TimeConverter, finite_number

from .replay import ReplayTransition, TimestampedReward


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    task_id: str
    candidate_id: str
    decision_time_sim: float
    estimated_local_utility: float


class GammaClock:
    def __init__(self, gamma_per_second: float, time_converter: TimeConverter):
        gamma = finite_number("gamma_per_second", gamma_per_second)
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma_per_second must be in (0,1]")
        self.gamma_per_second = gamma
        self.time_converter = time_converter

    def elapsed_seconds(self, start_sim: float, end_sim: float) -> float:
        start = finite_number("start_sim", start_sim)
        end = finite_number("end_sim", end_sim)
        if end < start:
            raise ValueError("time cannot move backwards")
        return self.time_converter.sim_to_seconds(end - start)

    def discount(self, elapsed_seconds: float) -> float:
        seconds = finite_number("elapsed_seconds", elapsed_seconds)
        if seconds < 0.0:
            raise ValueError("elapsed_seconds cannot be negative")
        return self.gamma_per_second ** seconds


class EventRewardBuffer:
    def __init__(self):
        self._events = []

    def append(self, event: TimestampedReward):
        self._events.append(event)
        self._events.sort(key=lambda item: (item.event_time_sim, item.decision_id, item.event_type))

    def pop_interval(self, start_sim: float, end_sim: float) -> Tuple[TimestampedReward, ...]:
        selected = tuple(
            event for event in self._events
            if start_sim < event.event_time_sim <= end_sim
        )
        selected_set = set(selected)
        self._events = [event for event in self._events if event not in selected_set]
        return selected


class RewardAssembler:
    def __init__(self, clock: GammaClock, event_buffer: Optional[EventRewardBuffer] = None):
        self.clock = clock
        self.event_buffer = event_buffer or EventRewardBuffer()

    @staticmethod
    def commit_reward(decision_record: DecisionRecord) -> float:
        return finite_number("estimated_local_utility", decision_record.estimated_local_utility)

    @staticmethod
    def realization_correction(
        decision_record: DecisionRecord,
        realized_local_utility: float,
        completion_outcome_reward: float,
    ) -> float:
        return (
            finite_number("realized_local_utility", realized_local_utility)
            - finite_number("estimated_local_utility", decision_record.estimated_local_utility)
            + finite_number("completion_outcome_reward", completion_outcome_reward)
        )

    def buffer_event(self, event: TimestampedReward):
        self.event_buffer.append(event)

    def build_transition(
        self,
        *,
        global_state_before: Iterable[float],
        selected_candidate_id: Optional[str],
        selected_candidate_features: Optional[Iterable[float]],
        immediate_reward: float,
        global_state_after: Iterable[float],
        next_candidate_features: Iterable[Iterable[float]],
        decision_time_sim: float,
        next_transition_time_sim: float,
        terminal: bool,
        next_candidate_context=None,
    ) -> ReplayTransition:
        elapsed = self.clock.elapsed_seconds(decision_time_sim, next_transition_time_sim)
        events = self.event_buffer.pop_interval(decision_time_sim, next_transition_time_sim)
        reward = finite_number("immediate_reward", immediate_reward)
        for event in events:
            event_elapsed = self.clock.elapsed_seconds(decision_time_sim, event.event_time_sim)
            reward += self.clock.discount(event_elapsed) * event.reward
        return ReplayTransition(
            tuple(float(value) for value in global_state_before),
            selected_candidate_id,
            None if selected_candidate_features is None else tuple(float(value) for value in selected_candidate_features),
            reward,
            tuple(float(value) for value in global_state_after),
            tuple(tuple(float(value) for value in item) for item in next_candidate_features),
            decision_time_sim,
            next_transition_time_sim,
            elapsed,
            self.clock.discount(elapsed),
            bool(terminal),
            events,
            next_candidate_context,
        )
