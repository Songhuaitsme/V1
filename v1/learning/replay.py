"""Immutable variable-candidate replay contracts and JSON round trips."""

from dataclasses import asdict, dataclass
import json
import random
from typing import Optional, Tuple

from v1.domain.units import finite_number, non_negative_finite


@dataclass(frozen=True)
class TimestampedReward:
    event_time_sim: float
    reward: float
    decision_id: str
    event_type: str

    def __post_init__(self):
        object.__setattr__(self, "event_time_sim", finite_number("event_time_sim", self.event_time_sim))
        object.__setattr__(self, "reward", finite_number("event_reward", self.reward))
        if not self.decision_id or not self.event_type:
            raise ValueError("decision_id and event_type must be non-empty")


@dataclass(frozen=True)
class ReplayTransition:
    global_state_before: Tuple[float, ...]
    selected_candidate_id: Optional[str]
    selected_candidate_features: Optional[Tuple[float, ...]]
    reward: float
    global_state_after: Tuple[float, ...]
    next_candidate_features: Tuple[Tuple[float, ...], ...]
    decision_time_sim: float
    next_transition_time_sim: float
    elapsed_seconds: float
    gamma_elapsed: float
    terminal: bool
    timestamped_event_rewards: Tuple[TimestampedReward, ...] = ()
    next_candidate_context: object = None

    def __post_init__(self):
        for name in ("reward", "decision_time_sim", "next_transition_time_sim", "elapsed_seconds", "gamma_elapsed"):
            object.__setattr__(self, name, finite_number(name, getattr(self, name)))
        if self.next_transition_time_sim < self.decision_time_sim:
            raise ValueError("transition time cannot move backwards")
        # A very long physical-time interval can legitimately underflow
        # gamma**elapsed to exactly zero in IEEE-754 arithmetic.  Zero means
        # that the bootstrap contribution is numerically negligible.
        if self.elapsed_seconds < 0.0 or not 0.0 <= self.gamma_elapsed <= 1.0:
            raise ValueError("invalid elapsed time or discount")
        if (self.selected_candidate_id is None) != (self.selected_candidate_features is None):
            raise ValueError("candidate id and selected features must be both present or absent")

    def to_json(self) -> str:
        if self.next_candidate_context is not None:
            raise ValueError(
                "context-backed replay transitions use checkpoint serialization, not JSON"
            )
        values = asdict(self)
        values["next_candidate_features"] = [
            [float(value) for value in row]
            for row in self.next_candidate_features
        ]
        return json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, payload: str) -> "ReplayTransition":
        values = json.loads(payload)
        events = tuple(TimestampedReward(**event) for event in values.pop("timestamped_event_rewards", ()))
        return cls(
            global_state_before=tuple(values["global_state_before"]),
            selected_candidate_id=values["selected_candidate_id"],
            selected_candidate_features=(
                None if values["selected_candidate_features"] is None
                else tuple(values["selected_candidate_features"])
            ),
            reward=values["reward"],
            global_state_after=tuple(values["global_state_after"]),
            next_candidate_features=tuple(tuple(item) for item in values["next_candidate_features"]),
            decision_time_sim=values["decision_time_sim"],
            next_transition_time_sim=values["next_transition_time_sim"],
            elapsed_seconds=values["elapsed_seconds"],
            gamma_elapsed=values["gamma_elapsed"],
            terminal=values["terminal"],
            timestamped_event_rewards=events,
        )


class CandidateReplayBuffer:
    def __init__(self, capacity: int):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._items = []
        self._position = 0

    def add(self, transition: ReplayTransition):
        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._position] = transition
        self._position = (self._position + 1) % self.capacity

    def __len__(self):
        return len(self._items)

    def items(self):
        return tuple(self._items)

    def sample(self, batch_size: int, rng: Optional[random.Random] = None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if batch_size > len(self._items):
            raise ValueError("batch_size exceeds replay population")
        generator = rng or random
        return tuple(generator.sample(self._items, batch_size))

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "items": tuple(self._items),
            "position": self._position,
        }

    def load_state_dict(self, state):
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("replay capacity mismatch")
        items = list(state.get("items", ()))
        if len(items) > self.capacity or any(
            not isinstance(item, ReplayTransition) for item in items
        ):
            raise ValueError("invalid replay checkpoint items")
        position = int(state.get("position", 0))
        if not 0 <= position < self.capacity:
            raise ValueError("invalid replay checkpoint position")
        self._items = items
        self._position = position
