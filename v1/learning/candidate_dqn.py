"""Variable-cardinality shared candidate Q-network and Double-DQN update."""

from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import json
import random
import time
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import torch
from torch import nn

from v1.domain.candidates import Candidate
from v1.domain.units import finite_number, positive_finite

from .features import CandidateFeatureEncoder
from .replay import ReplayTransition
from v1.scheduler.policies import CandidateStreamSelection


class SharedCandidateQNetwork(nn.Module):
    def __init__(self, global_state_dim: int, candidate_feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        for name, value in (("global_state_dim", global_state_dim), ("candidate_feature_dim", candidate_feature_dim), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.global_state_dim = global_state_dim
        self.candidate_feature_dim = candidate_feature_dim
        self.hidden_dim = hidden_dim
        self.global_encoder = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, global_state, candidate_features):
        if global_state.ndim == 1:
            global_state = global_state.unsqueeze(0)
        if candidate_features.ndim == 2:
            candidate_features = candidate_features.unsqueeze(0)
        if global_state.shape[0] != candidate_features.shape[0]:
            raise ValueError("global state and candidate batch sizes differ")
        global_embedding = self.global_encoder(global_state)
        candidate_embedding = self.candidate_encoder(candidate_features)
        expanded_global = global_embedding.unsqueeze(1).expand(
            -1, candidate_embedding.shape[1], -1
        )
        return self.q_head(torch.cat((expanded_global, candidate_embedding), dim=-1)).squeeze(-1)

    def forward_ragged(
        self, global_state, candidate_features, candidate_batch_index
    ):
        """Score flattened variable-size candidate sets in one network call."""
        if global_state.ndim != 2 or candidate_features.ndim != 2:
            raise ValueError("ragged inputs must be two-dimensional")
        indices = candidate_batch_index.to(
            device=global_state.device, dtype=torch.long
        )
        if indices.ndim != 1 or len(indices) != len(candidate_features):
            raise ValueError("ragged candidate indices are not aligned")
        global_embedding = self.global_encoder(global_state)
        candidate_embedding = self.candidate_encoder(candidate_features)
        expanded_global = global_embedding.index_select(0, indices)
        return self.q_head(
            torch.cat((expanded_global, candidate_embedding), dim=-1)
        ).squeeze(-1)


def mask_q_values(q_values: np.ndarray, valid_mask: Optional[Sequence[bool]]) -> np.ndarray:
    values = np.asarray(q_values, dtype=float).copy()
    if valid_mask is None:
        return values
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError("candidate mask shape does not match q-values")
    values[~mask] = -np.inf
    return values


def select_candidate_id(candidate_ids: Sequence[str], q_values, valid_mask=None) -> str:
    if not candidate_ids:
        raise ValueError("candidate set cannot be empty")
    values = mask_q_values(np.asarray(q_values, dtype=float), valid_mask)
    if values.shape != (len(candidate_ids),):
        raise ValueError("q-value count differs from candidate count")
    maximum = np.max(values)
    if not np.isfinite(maximum):
        raise ValueError("no valid candidate remains after masking")
    tied = [candidate_ids[index] for index, value in enumerate(values) if value == maximum]
    return min(tied)


def double_dqn_target(
    reward: float,
    terminal: bool,
    gamma_elapsed: float,
    online_next_q,
    target_next_q,
    valid_mask=None,
) -> float:
    reward_value = finite_number("reward", reward)
    gamma = finite_number("gamma_elapsed", gamma_elapsed)
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma_elapsed must be in (0,1]")
    if terminal:
        return reward_value
    online = np.asarray(online_next_q, dtype=float)
    target = np.asarray(target_next_q, dtype=float)
    if online.size == 0:
        return reward_value
    if online.shape != target.shape:
        raise ValueError("online and target next q shapes differ")
    masked = mask_q_values(online, valid_mask)
    if not np.isfinite(masked).any():
        return reward_value
    best = int(np.argmax(masked))
    return reward_value + gamma * float(target[best])


@dataclass(frozen=True)
class CandidateDqnMetadata:
    model_schema_version: str
    candidate_schema_version: str
    feature_schema_hash: str
    global_state_dim: int
    candidate_feature_dim: int
    gamma_per_second: float
    architecture: str = "shared_candidate_q_v1"

    @property
    def model_id(self):
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "candidate-dqn-" + hashlib.sha256(payload).hexdigest()[:16]


def validate_checkpoint_metadata(
    metadata,
    expected_feature_schema_hash: str,
    expected_architecture: str = None,
):
    required = {
        "model_schema_version",
        "candidate_schema_version",
        "feature_schema_hash",
        "global_state_dim",
        "candidate_feature_dim",
        "gamma_per_second",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"checkpoint metadata missing fields: {missing}")
    if metadata["model_schema_version"] != "1.0":
        raise ValueError("incompatible model_schema_version")
    if metadata["candidate_schema_version"] != "1.0":
        raise ValueError("incompatible candidate_schema_version")
    if metadata["feature_schema_hash"] != expected_feature_schema_hash:
        raise ValueError("candidate feature schema hash mismatch")
    if (
        expected_architecture is not None
        and metadata.get("architecture", "shared_candidate_q_v1")
        != expected_architecture
    ):
        raise ValueError("candidate DQN architecture mismatch")
    gamma = finite_number("gamma_per_second", metadata["gamma_per_second"])
    if not 0.0 < gamma <= 1.0:
        raise ValueError("checkpoint gamma_per_second must be in (0,1]")
    return True


@dataclass(frozen=True)
class CandidateSelectionTrace:
    task_id: str
    decision_time_sim: float
    global_state: tuple
    candidate_count: int
    selected_candidate_id: str
    selected_candidate_features: tuple
    selected_candidate: Candidate
    candidate_context: object = None
    # Epsilon exploration does not need to evaluate Q for the complete set.
    # None explicitly means that a distribution was not evaluated.
    q_min: Optional[float] = None
    q_max: Optional[float] = None
    q_mean: Optional[float] = None


class CandidateDQNPolicy:
    name = "candidate_dqn"

    def __init__(
        self,
        network: SharedCandidateQNetwork,
        feature_encoder: CandidateFeatureEncoder,
        global_state_provider,
        epsilon: float = 0.0,
        random_seed: int = 0,
        record_selection_traces: bool = False,
        candidate_chunk_size: int = 4096,
        device: str = "cpu",
    ):
        self.network = network
        self.feature_encoder = feature_encoder
        self.global_state_provider = global_state_provider
        self.epsilon = finite_number("epsilon", epsilon)
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0,1]")
        self.random = random.Random(random_seed)
        self.record_selection_traces = bool(record_selection_traces)
        if (
            isinstance(candidate_chunk_size, bool)
            or not isinstance(candidate_chunk_size, int)
            or candidate_chunk_size <= 0
        ):
            raise ValueError("candidate_chunk_size must be a positive integer")
        self.candidate_chunk_size = candidate_chunk_size
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable")
        self.network.to(self.device)
        self._selection_traces = []
        self.profiler = None
        # Formal evaluation keeps the exact per-candidate SHA-256 evidence.
        # Training may disable it because the digest is not an input to
        # selection, replay, rewards, or gradient updates.
        self.audit_candidate_set_hash = True

    def _synchronize_profile_device(self):
        if self.profiler is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def score_candidates(self, global_state, candidate_feature_matrix, candidate_mask=None):
        matrix = np.asarray(candidate_feature_matrix, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("candidate feature matrix must be two-dimensional")
        if matrix.shape[0] == 0:
            return np.asarray([], dtype=float)
        state = torch.as_tensor(
            np.asarray(global_state, dtype=np.float32), device=self.device
        )
        features = torch.as_tensor(matrix, device=self.device)
        self.network.eval()
        with torch.no_grad():
            values = self.network(state, features).cpu().numpy()[0]
        return mask_q_values(values, candidate_mask)

    def select_stream(
        self,
        candidates,
        task=None,
        context=None,
        earliest_compute_start_sim=None,
        *,
        _explore=None,
        _state=None,
    ):
        state = (
            self.global_state_provider(task)
            if _state is None else np.asarray(_state, dtype=np.float32)
        )
        if earliest_compute_start_sim is None and context is not None:
            earliest_compute_start_sim = context.earliest_compute_start_sim
        explore = (
            self.random.random() < self.epsilon
            if _explore is None else bool(_explore)
        )
        selected = None
        selected_features = None
        greedy = None
        greedy_features = None
        greedy_q = None
        earliest_candidate = None
        count = 0
        digest = hashlib.sha256()
        q_min = None
        q_max = None
        q_sum = 0.0
        chunk_candidates = []

        def consume_chunk():
            nonlocal selected, selected_features, greedy, greedy_features
            nonlocal greedy_q, q_min, q_max, q_sum
            if not chunk_candidates:
                return
            encode_started = time.perf_counter()
            features = tuple(
                self.feature_encoder.encode(item, earliest_compute_start_sim)
                for item in chunk_candidates
            )
            if self.profiler is not None:
                self.profiler.add(
                    "candidate_feature_encoding_seconds",
                    time.perf_counter() - encode_started,
                )
            self._synchronize_profile_device()
            inference_started = time.perf_counter()
            values = self.score_candidates(state, features)
            self._synchronize_profile_device()
            if self.profiler is not None:
                self.profiler.add(
                    "selection_inference_seconds",
                    time.perf_counter() - inference_started,
                )
            for candidate, encoded, value in zip(chunk_candidates, features, values):
                numeric = float(value)
                q_min = numeric if q_min is None else min(q_min, numeric)
                q_max = numeric if q_max is None else max(q_max, numeric)
                q_sum += numeric
                if (
                    greedy is None
                    or numeric > greedy_q
                    or (numeric == greedy_q and candidate.candidate_id < greedy.candidate_id)
                ):
                    greedy = candidate
                    greedy_features = tuple(encoded)
                    greedy_q = numeric
                if selected is candidate:
                    selected_features = tuple(encoded)
            chunk_candidates.clear()

        candidate_iterator = iter(candidates)
        exhausted = False
        while not exhausted:
            generated_chunk = []
            generation_started = time.perf_counter()
            for _ in range(self.candidate_chunk_size):
                try:
                    generated_chunk.append(next(candidate_iterator))
                except StopIteration:
                    exhausted = True
                    break
            if self.profiler is not None:
                self.profiler.add(
                    "candidate_stream_seconds",
                    time.perf_counter() - generation_started,
                )
            for candidate in generated_chunk:
                count += 1
                digest.update(candidate.candidate_id.encode("utf-8"))
                digest.update(b"\0")
                if earliest_candidate is None or (
                    candidate.compute_start_sim,
                    candidate.target_node,
                    candidate.path.path_id,
                    candidate.candidate_id,
                ) < (
                    earliest_candidate.compute_start_sim,
                    earliest_candidate.target_node,
                    earliest_candidate.path.path_id,
                    earliest_candidate.candidate_id,
                ):
                    earliest_candidate = candidate
                if explore and self.random.randrange(count) == 0:
                    selected = candidate
                    selected_features = None
                if not explore:
                    chunk_candidates.append(candidate)
            if not explore:
                consume_chunk()
        if count == 0:
            raise ValueError("candidate set cannot be empty")
        if self.profiler is not None:
            self.profiler.increment("selection_candidate_count", count)
        if not explore:
            selected = greedy
            selected_features = greedy_features
        elif selected_features is None:
            encode_started = time.perf_counter()
            selected_features = tuple(
                self.feature_encoder.encode(selected, earliest_compute_start_sim)
            )
            if self.profiler is not None:
                self.profiler.add(
                    "candidate_feature_encoding_seconds",
                    time.perf_counter() - encode_started,
                )
        if self.record_selection_traces:
            self._selection_traces.append(CandidateSelectionTrace(
                task_id="" if task is None else task.task_id,
                decision_time_sim=selected.decision_time_sim,
                global_state=tuple(float(value) for value in state),
                candidate_count=count,
                selected_candidate_id=selected.candidate_id,
                selected_candidate_features=selected_features,
                selected_candidate=selected,
                candidate_context=context,
                q_min=None if explore else float(q_min),
                q_max=None if explore else float(q_max),
                q_mean=None if explore else float(q_sum / count),
            ))
        return CandidateStreamSelection(
            selected,
            earliest_candidate,
            count,
            digest.hexdigest(),
            context,
        )

    def select_complete_stream(self, stream, task=None):
        """Select from a complete stream with a metric-lazy explore path.

        Exploration still visits every feasible candidate record in canonical
        order, hashes every candidate id, and uses the same reservoir-sampling
        random calls as ``select_stream``.  Only the selected and earliest
        records need expensive accounting metrics and Candidate objects.
        """

        state = self.global_state_provider(task)
        explore = self.random.random() < self.epsilon
        if not explore:
            if not self.audit_candidate_set_hash:
                return self._select_complete_stream_training_greedy(
                    stream, task, state
                )
            return self.select_stream(
                stream.iter_candidates(),
                task=task,
                context=stream.context,
                _explore=False,
                _state=state,
            )

        digest = hashlib.sha256()
        if not self.audit_candidate_set_hash:
            generation_started = time.perf_counter()
            (
                selected_record,
                earliest_record,
                count,
            ) = stream.generator.sample_context_candidate_records(
                stream.context,
                self.random,
                chunk_size=self.candidate_chunk_size,
            )
            if self.profiler is not None:
                self.profiler.add(
                    "candidate_stream_seconds",
                    time.perf_counter() - generation_started,
                )
        else:
            selected_record = None
            earliest_record = None
            count = 0
            record_iterator = iter(stream.iter_candidate_records())
            exhausted = False
            while not exhausted:
                generated_chunk = []
                generation_started = time.perf_counter()
                for _ in range(self.candidate_chunk_size):
                    try:
                        generated_chunk.append(
                            next(record_iterator)
                        )
                    except StopIteration:
                        exhausted = True
                        break
                if self.profiler is not None:
                    self.profiler.add(
                        "candidate_stream_seconds",
                        time.perf_counter() - generation_started,
                    )
                for record in generated_chunk:
                    count += 1
                    digest.update(
                        record.candidate_id.encode("utf-8")
                    )
                    digest.update(b"\0")
                    if earliest_record is None or (
                        record.compute_start_sim,
                        record.target_node,
                        record.path.path_id,
                        record.candidate_id,
                    ) < (
                        earliest_record.compute_start_sim,
                        earliest_record.target_node,
                        earliest_record.path.path_id,
                        earliest_record.candidate_id,
                    ):
                        earliest_record = record
                    if self.random.randrange(count) == 0:
                        selected_record = record

        if count == 0:
            raise ValueError("candidate set cannot be empty")
        if count != stream.feasible_candidate_count:
            raise RuntimeError(
                "complete candidate record count changed between prepare and select"
            )
        if self.profiler is not None:
            self.profiler.increment("selection_candidate_count", count)
        selected = stream.materialize_record(selected_record)
        earliest_candidate = (
            selected
            if selected_record is earliest_record
            else stream.materialize_record(earliest_record)
        )
        encode_started = time.perf_counter()
        selected_features = tuple(
            self.feature_encoder.encode(
                selected,
                stream.context.earliest_compute_start_sim,
            )
        )
        if self.profiler is not None:
            self.profiler.add(
                "candidate_feature_encoding_seconds",
                time.perf_counter() - encode_started,
            )
        if self.record_selection_traces:
            self._selection_traces.append(CandidateSelectionTrace(
                task_id="" if task is None else task.task_id,
                decision_time_sim=selected.decision_time_sim,
                global_state=tuple(float(value) for value in state),
                candidate_count=count,
                selected_candidate_id=selected.candidate_id,
                selected_candidate_features=selected_features,
                selected_candidate=selected,
                candidate_context=stream.context,
                q_min=None,
                q_max=None,
                q_mean=None,
            ))
        return CandidateStreamSelection(
            selected,
            earliest_candidate,
            count,
            digest.hexdigest(),
            stream.context,
        )

    def _select_complete_stream_training_greedy(
        self, stream, task, state
    ):
        """Score complete candidates in arrays for non-audit training."""

        chunk_iterator = iter(
            stream.generator.feature_chunks_from_context(
                stream.context,
                self.feature_encoder,
                metric_evaluator=stream.metric_evaluator,
                chunk_size=self.candidate_chunk_size,
                with_records=True,
            )
        )
        count = 0
        greedy_q = None
        greedy_record = None
        greedy_features = None
        earliest_record = None
        q_min = None
        q_max = None
        q_sum = 0.0

        while True:
            generation_started = time.perf_counter()
            try:
                chunk = next(chunk_iterator)
            except StopIteration:
                if self.profiler is not None:
                    self.profiler.add(
                        "candidate_stream_seconds",
                        time.perf_counter() - generation_started,
                    )
                break
            if self.profiler is not None:
                self.profiler.add(
                    "candidate_stream_seconds",
                    time.perf_counter() - generation_started,
                )
            features = np.asarray(
                chunk.features, dtype=np.float32
            )
            if len(features) == 0:
                continue

            self._synchronize_profile_device()
            inference_started = time.perf_counter()
            values = self.score_candidates(state, features)
            self._synchronize_profile_device()
            if self.profiler is not None:
                self.profiler.add(
                    "selection_inference_seconds",
                    time.perf_counter() - inference_started,
                )

            count += len(features)
            local_min = float(np.min(values))
            local_max = float(np.max(values))
            q_min = (
                local_min
                if q_min is None
                else min(q_min, local_min)
            )
            q_max = (
                local_max
                if q_max is None
                else max(q_max, local_max)
            )
            q_sum += float(
                np.asarray(values, dtype=np.float64).sum()
            )

            for index in np.flatnonzero(values == local_max):
                record = (
                    stream.generator.record_from_feature_chunk(
                        chunk, index
                    )
                )
                record = (
                    stream.generator.attach_context_candidate_id(
                        stream.context, record
                    )
                )
                if (
                    greedy_record is None
                    or local_max > greedy_q
                    or (
                        local_max == greedy_q
                        and record.candidate_id
                        < greedy_record.candidate_id
                    )
                ):
                    greedy_q = local_max
                    greedy_record = record

            earliest_start = float(np.min(chunk.starts))
            for index in np.flatnonzero(
                chunk.starts == earliest_start
            ):
                record = (
                    stream.generator.record_from_feature_chunk(
                        chunk, index
                    )
                )
                record = (
                    stream.generator.attach_context_candidate_id(
                        stream.context, record
                    )
                )
                if earliest_record is None or (
                    record.compute_start_sim,
                    record.target_node,
                    record.path.path_id,
                    record.candidate_id,
                ) < (
                    earliest_record.compute_start_sim,
                    earliest_record.target_node,
                    earliest_record.path.path_id,
                    earliest_record.candidate_id,
                ):
                    earliest_record = record

        if count == 0:
            raise ValueError("candidate set cannot be empty")
        if count != stream.feasible_candidate_count:
            raise RuntimeError(
                "complete candidate record count changed between "
                "prepare and select"
            )
        if self.profiler is not None:
            self.profiler.increment(
                "selection_candidate_count", count
            )

        selected = stream.materialize_record(greedy_record)
        earliest_candidate = (
            selected
            if greedy_record == earliest_record
            else stream.materialize_record(earliest_record)
        )
        encode_started = time.perf_counter()
        greedy_features = tuple(
            self.feature_encoder.encode(
                selected,
                stream.context.earliest_compute_start_sim,
            )
        )
        if self.profiler is not None:
            self.profiler.add(
                "candidate_feature_encoding_seconds",
                time.perf_counter() - encode_started,
            )
        if self.record_selection_traces:
            self._selection_traces.append(CandidateSelectionTrace(
                task_id="" if task is None else task.task_id,
                decision_time_sim=selected.decision_time_sim,
                global_state=tuple(float(value) for value in state),
                candidate_count=count,
                selected_candidate_id=selected.candidate_id,
                selected_candidate_features=greedy_features,
                selected_candidate=selected,
                candidate_context=stream.context,
                q_min=float(q_min),
                q_max=float(q_max),
                q_mean=float(q_sum / count),
            ))
        return CandidateStreamSelection(
            selected,
            earliest_candidate,
            count,
            hashlib.sha256().hexdigest(),
            stream.context,
        )

    def select(self, candidates: Sequence[Candidate], task=None) -> Candidate:
        items = tuple(candidates)
        if not items:
            raise ValueError("candidate set cannot be empty")
        earliest = min(item.compute_start_sim for item in items)
        return self.select_stream(
            iter(items),
            task=task,
            earliest_compute_start_sim=earliest,
        ).selected_candidate

    def pop_selection_traces(self):
        traces = tuple(self._selection_traces)
        self._selection_traces.clear()
        return traces


class CandidateDQNTrainer:
    def __init__(
        self,
        online,
        target,
        learning_rate=1e-3,
        *,
        device="cpu",
        candidate_chunk_size=4096,
        bootstrap_candidate_limit=None,
        next_candidate_provider: Optional[Callable[[object], Iterable]] = None,
        double_dqn: bool = True,
    ):
        if (
            isinstance(candidate_chunk_size, bool)
            or not isinstance(candidate_chunk_size, int)
            or candidate_chunk_size <= 0
        ):
            raise ValueError("candidate_chunk_size must be a positive integer")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable")
        self.online = online.to(self.device)
        self.target = target.to(self.device)
        self.candidate_chunk_size = candidate_chunk_size
        if bootstrap_candidate_limit is not None and (
            isinstance(bootstrap_candidate_limit, bool)
            or not isinstance(bootstrap_candidate_limit, int)
            or bootstrap_candidate_limit <= 0
        ):
            raise ValueError(
                "bootstrap_candidate_limit must be None or a positive integer"
            )
        self.bootstrap_candidate_limit = bootstrap_candidate_limit
        self.next_candidate_provider = next_candidate_provider
        if not isinstance(double_dqn, bool):
            raise ValueError("double_dqn must be boolean")
        self.double_dqn = double_dqn
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=positive_finite("learning_rate", learning_rate)
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.profiler = None
        self._next_feature_cache = OrderedDict()
        self._next_feature_cache_capacity = 5000

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_next_feature_cache"] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "double_dqn" not in self.__dict__:
            self.double_dqn = True
        if "_next_feature_cache" not in self.__dict__:
            self._next_feature_cache = OrderedDict()
        if "_next_feature_cache_capacity" not in self.__dict__:
            self._next_feature_cache_capacity = 5000

    def clear_next_feature_cache(self):
        if not hasattr(self, "_next_feature_cache"):
            self._next_feature_cache = OrderedDict()
        if not hasattr(self, "_next_feature_cache_capacity"):
            self._next_feature_cache_capacity = 5000
        self._next_feature_cache.clear()

    def _synchronize_profile_device(self):
        if self.profiler is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def train_transition(self, transition: ReplayTransition) -> float:
        return self.train_batch((transition,))

    def _feature_matrix(self, transition):
        if not hasattr(self, "_next_feature_cache"):
            self.clear_next_feature_cache()
        context = transition.next_candidate_context
        if context is not None:
            if self.next_candidate_provider is None:
                raise ValueError("context-backed replay requires next_candidate_provider")
            key = id(context)
            cached = self._next_feature_cache.get(key)
            if cached is not None and cached[0] is context:
                self._next_feature_cache.move_to_end(key)
                return cached[1]
            chunks = []
            count = 0
            for raw_chunk in self.next_candidate_provider(context):
                chunk = np.asarray(raw_chunk, dtype=np.float32)
                if self.bootstrap_candidate_limit is not None:
                    remaining = self.bootstrap_candidate_limit - count
                    if remaining <= 0:
                        break
                    chunk = chunk[:remaining]
                if len(chunk):
                    chunks.append(chunk)
                    count += len(chunk)
                if (
                    self.bootstrap_candidate_limit is not None
                    and count >= self.bootstrap_candidate_limit
                ):
                    break
            matrix = (
                np.concatenate(chunks, axis=0)
                if chunks
                else np.empty(
                    (0, self.online.candidate_feature_dim), dtype=np.float32
                )
            )
            self._next_feature_cache[key] = (context, matrix)
            self._next_feature_cache.move_to_end(key)
            while len(self._next_feature_cache) > self._next_feature_cache_capacity:
                self._next_feature_cache.popitem(last=False)
            return matrix
        matrix = np.asarray(transition.next_candidate_features, dtype=np.float32)
        if matrix.size == 0:
            return np.empty(
                (0, self.online.candidate_feature_dim), dtype=np.float32
            )
        matrix = matrix.reshape((-1, self.online.candidate_feature_dim))
        if self.bootstrap_candidate_limit is not None:
            matrix = matrix[:self.bootstrap_candidate_limit]
        return matrix

    def _feature_chunks(self, transition):
        features = self._feature_matrix(transition)
        for start in range(0, len(features), self.candidate_chunk_size):
            yield features[start:start + self.candidate_chunk_size]

    def _double_dqn_bootstrap_batch(self, batch):
        matrices = []
        states = []
        positions = []
        counts = []
        for position, transition in enumerate(batch):
            if transition.terminal:
                continue
            generation_started = time.perf_counter()
            matrix = self._feature_matrix(transition)
            if self.profiler is not None:
                self.profiler.add(
                    "replay_candidate_generation_seconds",
                    time.perf_counter() - generation_started,
                )
                self.profiler.increment(
                    "replay_candidate_feature_count", len(matrix)
                )
            if len(matrix) == 0:
                continue
            matrices.append(matrix)
            states.append(transition.global_state_after)
            positions.append(position)
            counts.append(len(matrix))

        target_values = torch.tensor(
            [item.reward for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        if not matrices:
            return target_values
        next_states = torch.as_tensor(
            np.asarray(states, dtype=np.float32), device=self.device
        )
        candidate_features = torch.as_tensor(
            np.concatenate(matrices, axis=0), device=self.device
        )
        batch_indices = torch.repeat_interleave(
            torch.arange(len(counts), device=self.device),
            torch.tensor(counts, device=self.device),
        )
        self._synchronize_profile_device()
        inference_started = time.perf_counter()
        with torch.no_grad():
            target_candidates = self.target.forward_ragged(
                next_states, candidate_features, batch_indices
            )
            selection_values = (
                self.online.forward_ragged(
                    next_states, candidate_features, batch_indices
                )
                if self.double_dqn else target_candidates
            )
            best_indices = []
            offset = 0
            for count in counts:
                best_indices.append(
                    offset
                    + torch.argmax(selection_values[offset:offset + count])
                )
                offset += count
            bootstrap = target_candidates[torch.stack(best_indices)]
            position_tensor = torch.tensor(
                positions, dtype=torch.long, device=self.device
            )
            gamma_tensor = torch.tensor(
                [batch[index].gamma_elapsed for index in positions],
                dtype=torch.float32,
                device=self.device,
            )
            target_values[position_tensor] += gamma_tensor * bootstrap
        self._synchronize_profile_device()
        if self.profiler is not None:
            self.profiler.add(
                "bootstrap_inference_seconds",
                time.perf_counter() - inference_started,
            )
        return target_values

    def _double_dqn_bootstrap(self, transition):
        if transition.terminal:
            return None
        next_state = torch.tensor(
            transition.global_state_after,
            dtype=torch.float32,
            device=self.device,
        )
        best_online = None
        best_target = None
        remaining = self.bootstrap_candidate_limit
        with torch.no_grad():
            chunk_iterator = iter(self._feature_chunks(transition))
            while True:
                if remaining is not None and remaining <= 0:
                    break
                generation_started = time.perf_counter()
                try:
                    raw_chunk = next(chunk_iterator)
                except StopIteration:
                    if self.profiler is not None:
                        self.profiler.add(
                            "replay_candidate_generation_seconds",
                            time.perf_counter() - generation_started,
                        )
                    break
                if self.profiler is not None:
                    self.profiler.add(
                        "replay_candidate_generation_seconds",
                        time.perf_counter() - generation_started,
                    )
                if len(raw_chunk) == 0:
                    continue
                if remaining is not None and len(raw_chunk) > remaining:
                    raw_chunk = raw_chunk[:remaining]
                if self.profiler is not None:
                    self.profiler.increment(
                        "replay_candidate_feature_count", len(raw_chunk)
                    )
                chunk = torch.as_tensor(
                    np.asarray(raw_chunk, dtype=np.float32),
                    dtype=torch.float32,
                    device=self.device,
                )
                self._synchronize_profile_device()
                inference_started = time.perf_counter()
                target_values = self.target(next_state, chunk)[0]
                selection_values = (
                    self.online(next_state, chunk)[0]
                    if self.double_dqn else target_values
                )
                index = int(torch.argmax(selection_values).item())
                value = float(selection_values[index].item())
                self._synchronize_profile_device()
                if self.profiler is not None:
                    self.profiler.add(
                        "bootstrap_inference_seconds",
                        time.perf_counter() - inference_started,
                    )
                if best_online is None or value > best_online:
                    best_online = value
                    best_target = float(target_values[index].item())
                if remaining is not None:
                    remaining -= len(raw_chunk)
        return best_target

    def train_batch(self, transitions: Sequence[ReplayTransition]) -> float:
        batch = tuple(transitions)
        if not batch:
            raise ValueError("training batch cannot be empty")
        if any(item.selected_candidate_features is None for item in batch):
            raise ValueError("time-progression transition has no candidate action to train")
        self._synchronize_profile_device()
        forward_started = time.perf_counter()
        states = torch.tensor(
            [item.global_state_before for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        selected = torch.tensor(
            [[item.selected_candidate_features] for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        predicted = self.online(states, selected)[:, 0]
        self._synchronize_profile_device()
        if self.profiler is not None:
            self.profiler.add(
                "training_forward_seconds",
                time.perf_counter() - forward_started,
            )
        target_tensor = self._double_dqn_bootstrap_batch(batch)
        loss = self.loss_fn(predicted, target_tensor)
        self._synchronize_profile_device()
        backward_started = time.perf_counter()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._synchronize_profile_device()
        if self.profiler is not None:
            self.profiler.add(
                "backpropagation_seconds",
                time.perf_counter() - backward_started,
            )
        return float(loss.item())
