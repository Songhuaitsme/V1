"""Production-oriented v1.0 candidate-DQN training entry point."""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from shared import config
from v1.ablation_settings import apply_ablation_variant, variant_names
from v1.audit_v1 import scan_scheduler_invariants
from v1.domain.models import SlaType, TaskState
from v1.learning import (
    CandidateDQNTrainer,
    CandidateDqnMetadata,
    CandidateReplayBuffer,
    DecisionRecord,
    GammaClock,
    RewardAssembler,
    TimestampedReward,
    validate_checkpoint_metadata,
)
from v1.profiling import TrainingPerformanceProfiler
from v1.scheduler import ObjectiveConfig, ObjectiveScorer
from v1.v1_runtime import (
    create_v1_runtime,
    ensure_v1_runtime_forecasts_for_tasks,
    extend_v1_runtime_forecasts,
    v1_runtime_forecast_end,
)


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_positive_int(name, value):
    if value is None:
        return None
    return _positive_int(name, value)


def _resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _metadata(runtime):
    architecture = "shared_candidate_q_v1"
    if not config.V1_DQN_USE_GLOBAL_STATE or not config.V1_DQN_DOUBLE_DQN:
        architecture += (
            f":global={int(config.V1_DQN_USE_GLOBAL_STATE)}"
            f":double={int(config.V1_DQN_DOUBLE_DQN)}"
        )
    return CandidateDqnMetadata(
        "1.0",
        "1.0",
        runtime.candidate_feature_encoder.feature_schema_hash,
        runtime.candidate_q_network.global_state_dim,
        runtime.candidate_q_network.candidate_feature_dim,
        config.V1_GAMMA_PER_SECOND,
        architecture=architecture,
    )


def _training_config_view(**overrides):
    names = (
        "REQUIREMENTS_VERSION", "ALGORITHM_VERSION", "CANDIDATE_MODE",
        "V1_CANDIDATE_MODE", "SCHEDULING_CYCLE", "V1_CANDIDATE_PATH_K",
        "V1_CANDIDATE_POOL_MAX_BY_SLA",
        "V1_CANDIDATE_POOL_NODE_LIMIT_BY_SLA",
        "V1_CANDIDATE_POOL_TIME_SAMPLES_BY_SLA", "V1_GAMMA_PER_SECOND",
        "V1_CANDIDATE_DQN_HIDDEN_DIM", "LEARNING_RATE", "MEMORY_CAPACITY",
        "EPSILON_START", "EPSILON_MIN", "EPSILON_DECAY",
        "V1_TARGET_UPDATE_INTERVAL", "V1_COST_REFERENCE_YUAN",
        "V1_COST_SCALE_YUAN", "V1_GREEN_ABSORPTION_DELTA_SCALE",
        "V1_OBJECTIVE_COST_WEIGHT", "V1_OBJECTIVE_GREEN_WEIGHT",
        "V1_OBJECTIVE_BALANCE_WEIGHT", "V1_SOFT_TARDINESS_WEIGHT",
        "V1_FLEXIBLE_TARDINESS_WEIGHT",
        "V1_ABLATION_VARIANT", "V1_ACTIVE_WAIT_ENABLED",
        "V1_DISCOUNT_MODE", "V1_DECISION_GAMMA",
        "V1_REWARD_ESTIMATE_ENABLED",
        "V1_REWARD_REALIZATION_CORRECTION_ENABLED",
        "V1_REWARD_TERMINAL_PENALTIES_ENABLED",
        "V1_DISABLED_CANDIDATE_FEATURE_GROUPS",
        "V1_DQN_USE_GLOBAL_STATE", "V1_DQN_DOUBLE_DQN",
        "V1_TARIFF_MODE",
    )
    values = {name: getattr(config, name) for name in names}
    values.update(overrides)
    return values


def _canonical_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resume_config_compatible(saved, requested):
    if not isinstance(saved, dict) or not isinstance(requested, dict):
        return False
    legacy_defaults = {
        "V1_ABLATION_VARIANT": "reference",
        "V1_ACTIVE_WAIT_ENABLED": True,
        "V1_DISCOUNT_MODE": "physical_time",
        "V1_DECISION_GAMMA": 0.95,
        "V1_REWARD_ESTIMATE_ENABLED": True,
        "V1_REWARD_REALIZATION_CORRECTION_ENABLED": True,
        "V1_REWARD_TERMINAL_PENALTIES_ENABLED": True,
        "V1_DISABLED_CANDIDATE_FEATURE_GROUPS": (),
        "V1_DQN_USE_GLOBAL_STATE": True,
        "V1_DQN_DOUBLE_DQN": True,
        "V1_TARIFF_MODE": "tou_uniform",
    }
    if any(
        key not in saved and key in requested and requested.get(key) != value
        for key, value in legacy_defaults.items()
    ):
        return False
    performance_only = {"candidate_chunk_size", "invariant_check_every"}
    # Older checkpoints do not contain semantic keys added in later versions.
    # Validate every setting they did record without rejecting absent metadata.
    semantic_keys = set(saved) | {"bootstrap_candidate_limit"}
    saved_semantics = {
        key: saved.get(key)
        for key in semantic_keys
        if key not in performance_only
    }
    requested_semantics = {
        key: requested.get(key)
        for key in semantic_keys
        if key not in performance_only
    }
    return _canonical_hash(saved_semantics) == _canonical_hash(
        requested_semantics
    )


def _run_profiled_operation(profiler, counter_key, operation):
    if profiler is None:
        return operation()
    nested_before = profiler.nested_stage_seconds()
    started = time.perf_counter()
    result = operation()
    elapsed = time.perf_counter() - started
    nested_delta = profiler.nested_stage_seconds() - nested_before
    profiler.add("environment_update_seconds", max(0.0, elapsed - nested_delta))
    profiler.increment(counter_key)
    return result


def _assert_scheduler_invariants(scheduler, cycle_result=None):
    violations = scan_scheduler_invariants(scheduler, cycle_result)
    if violations:
        detail = "; ".join(
            f"{item.invariant_id}:{item.detail}" for item in violations
        )
        raise RuntimeError(f"v1.0 invariant gate failed: {detail}")


def _should_run_invariant_check(
    completed_cycle,
    *,
    invariant_check_every,
    checkpoint_every,
    final_cycle,
):
    return (
        completed_cycle % invariant_check_every == 0
        or completed_cycle % checkpoint_every == 0
        or completed_cycle == final_cycle
    )


def _write_profile_outputs(profiler, total_wall_seconds, json_path):
    summary = profiler.summary(total_wall_seconds)
    target = Path(json_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    csv_path = target.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("section", "seconds", "percent")
        )
        writer.writeheader()
        for section, seconds in summary["sections_seconds"].items():
            writer.writerow({
                "section": section,
                "seconds": seconds,
                "percent": summary["sections_percent"][section],
            })
    return target, csv_path, summary


class V1TrainingLoop:
    def __init__(
        self,
        runtime,
        *,
        device="cpu",
        candidate_chunk_size=None,
        batch_size=None,
        min_replay_size=None,
        updates_per_transition=None,
        bootstrap_candidate_limit=None,
        random_seed=0,
        profiler=None,
    ):
        self.runtime = runtime
        self.policy = runtime.scheduler.policy
        # Formal evaluation retains the exact complete-set digest. Training
        # only consumes the selected candidate, so hashing every candidate ID
        # is pure audit overhead here.
        self.policy.audit_candidate_set_hash = False
        if runtime.candidate_q_network is None:
            raise ValueError("v1 training requires candidate_dqn policy")
        self.device = _resolve_device(device)
        self.candidate_chunk_size = _positive_int(
            "candidate_chunk_size",
            config.V1_CANDIDATE_CHUNK_SIZE
            if candidate_chunk_size is None else candidate_chunk_size,
        )
        self.batch_size = _positive_int(
            "batch_size", config.BATCH_SIZE if batch_size is None else batch_size
        )
        self.min_replay_size = _positive_int(
            "min_replay_size",
            config.V1_REPLAY_MIN_SIZE
            if min_replay_size is None else min_replay_size,
        )
        self.min_replay_size = max(self.batch_size, self.min_replay_size)
        self.updates_per_transition = _positive_int(
            "updates_per_transition",
            config.V1_TRAIN_UPDATES_PER_TRANSITION
            if updates_per_transition is None else updates_per_transition,
        )
        self.bootstrap_candidate_limit = _optional_positive_int(
            "bootstrap_candidate_limit", bootstrap_candidate_limit
        )
        self.replay_random = random.Random(random_seed + 7919)
        self.target = type(runtime.candidate_q_network)(
            runtime.candidate_q_network.global_state_dim,
            runtime.candidate_q_network.candidate_feature_dim,
            runtime.candidate_q_network.hidden_dim,
        )
        self.trainer = CandidateDQNTrainer(
            runtime.candidate_q_network,
            self.target,
            config.LEARNING_RATE,
            device=self.device,
            candidate_chunk_size=self.candidate_chunk_size,
            bootstrap_candidate_limit=self.bootstrap_candidate_limit,
            next_candidate_provider=self._next_candidate_feature_chunks,
            double_dqn=config.V1_DQN_DOUBLE_DQN,
        )
        self.trainer.update_target()
        self.replay = CandidateReplayBuffer(config.MEMORY_CAPACITY)
        self.reward_assembler = RewardAssembler(
            GammaClock(
                config.V1_GAMMA_PER_SECOND,
                runtime.time_converter,
                mode=config.V1_DISCOUNT_MODE,
                decision_gamma=config.V1_DECISION_GAMMA,
            )
        )
        self.objective = ObjectiveScorer(ObjectiveConfig(
            config.V1_COST_REFERENCE_YUAN,
            config.V1_COST_SCALE_YUAN,
            config.V1_GREEN_ABSORPTION_DELTA_SCALE,
            config.V1_OBJECTIVE_COST_WEIGHT,
            config.V1_OBJECTIVE_GREEN_WEIGHT,
            config.V1_OBJECTIVE_BALANCE_WEIGHT,
            config.V1_SOFT_TARDINESS_WEIGHT,
            config.V1_FLEXIBLE_TARDINESS_WEIGHT,
        ))
        self.pending = None
        self.decision_by_task = {}
        self.transition_count = 0
        self.update_count = 0
        self.losses = []
        self.q_summaries = []
        self.candidate_count = 0
        self.profiler = profiler
        self.runtime.scheduler.profiler = profiler
        self.runtime.scheduler.candidate_generator.profiler = profiler
        self.policy.profiler = profiler
        self.trainer.profiler = profiler

    def _next_candidate_feature_chunks(self, context):
        evaluator = self.runtime.accounting.candidate_metric_evaluator(
            context.reservation_snapshot
        )
        chunk_size = self.candidate_chunk_size
        if self.bootstrap_candidate_limit is not None:
            chunk_size = min(chunk_size, self.bootstrap_candidate_limit)
        return self.runtime.scheduler.candidate_generator.feature_chunks_from_context(
            context,
            self.runtime.candidate_feature_encoder,
            metric_evaluator=evaluator,
            chunk_size=chunk_size,
        )

    def to_device(self, device):
        self.device = _resolve_device(device)
        self.trainer.device = torch.device(self.device)
        self.trainer.online.to(self.trainer.device)
        self.trainer.target.to(self.trainer.device)
        self.policy.device = torch.device(self.device)
        self.policy.network.to(self.policy.device)
        for state in self.trainer.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.trainer.device)

    def set_candidate_chunk_size(self, candidate_chunk_size):
        chunk_size = _positive_int(
            "candidate_chunk_size", candidate_chunk_size
        )
        self.candidate_chunk_size = chunk_size
        self.policy.candidate_chunk_size = chunk_size
        self.trainer.candidate_chunk_size = chunk_size

    def set_bootstrap_candidate_limit(self, bootstrap_candidate_limit):
        limit = _optional_positive_int(
            "bootstrap_candidate_limit", bootstrap_candidate_limit
        )
        self.bootstrap_candidate_limit = limit
        self.trainer.bootstrap_candidate_limit = limit
        self.trainer.clear_next_feature_cache()

    def process_cycle(self, result, *, check_invariants=True):
        if check_invariants:
            _assert_scheduler_invariants(self.runtime.scheduler, result)
        for event in result.domain_events:
            if event.event_type == "TASK_COMPLETED":
                self._buffer_completion(event.task_id, event.event_time_sim)
            elif (
                event.event_type == "TASK_FAILED"
                and config.V1_REWARD_TERMINAL_PENALTIES_ENABLED
            ):
                self.reward_assembler.buffer_event(TimestampedReward(
                    event.event_time_sim,
                    config.V1_FAILURE_PENALTY,
                    self._credit_decision_id(event.task_id),
                    event.event_type,
                ))
        for transition in result.state_transitions:
            if (
                transition.new_state is TaskState.EXPIRED
                and config.V1_REWARD_TERMINAL_PENALTIES_ENABLED
            ):
                self.reward_assembler.buffer_event(TimestampedReward(
                    transition.event_time_sim,
                    config.V1_EXPIRATION_PENALTY,
                    self._credit_decision_id(transition.task_id),
                    "TASK_EXPIRED",
                ))

        for trace in self.policy.pop_selection_traces():
            if self.pending is not None:
                self._close_pending(
                    next_state=trace.global_state,
                    next_context=trace.candidate_context,
                    next_time=trace.decision_time_sim,
                    terminal=False,
                )
            candidate = trace.selected_candidate
            task = self.runtime.scheduler.state_machine.task_spec(trace.task_id)
            estimated_utility = self.objective.score(
                candidate, task.sla_type
            ).total_score
            record = DecisionRecord(
                "decision-" + candidate.candidate_id,
                task.task_id,
                candidate.candidate_id,
                trace.decision_time_sim,
                estimated_utility,
            )
            self.decision_by_task[task.task_id] = (record, candidate)
            self.pending = {
                "record": record,
                "state": trace.global_state,
                "candidate_id": candidate.candidate_id,
                "features": trace.selected_candidate_features,
                "immediate_reward": (
                    estimated_utility
                    if config.V1_REWARD_ESTIMATE_ENABLED else 0.0
                ),
            }
            self.candidate_count += trace.candidate_count
            if trace.q_min is not None:
                self.q_summaries.append(
                    (trace.q_min, trace.q_max, trace.q_mean)
                )

    def _credit_decision_id(self, task_id):
        linked = self.decision_by_task.get(task_id)
        if linked is not None:
            return linked[0].decision_id
        if self.pending is not None:
            return self.pending["record"].decision_id
        return "system-event"

    def _buffer_completion(self, task_id, event_time_sim):
        linked = self.decision_by_task.get(task_id)
        if linked is None:
            return
        record, candidate = linked
        # Only reservations that overlap this task on the same node can affect
        # its physical counterfactual. This avoids repeatedly realizing all
        # historical reservations as the run grows.
        relevant = tuple(
            reservation
            for reservation in self.runtime.calendar.reservations()
            if reservation.target_node == candidate.target_node
            and reservation.compute_interval_sim.overlaps(
                self.runtime.calendar.get_reservation(
                    self.runtime.scheduler.state_machine.runtime(task_id).reservation_id
                ).compute_interval_sim
            )
        )
        report = self.runtime.accounting.realize(relevant)
        realized = next(item for item in report.task_records if item.task_id == task_id)
        task = self.runtime.scheduler.state_machine.task_spec(task_id)
        realized_candidate = replace(
            candidate,
            estimated_candidate_marginal_system_cost_yuan=(
                realized.task_attributed_cost_yuan
            ),
            estimated_green_coverage=realized.green_coverage,
        )
        realized_utility = self.objective.score(
            realized_candidate, task.sla_type
        ).total_score
        if config.V1_REWARD_REALIZATION_CORRECTION_ENABLED:
            correction = (
                realized_utility
                - (
                    record.estimated_local_utility
                    if config.V1_REWARD_ESTIMATE_ENABLED else 0.0
                )
                + config.V1_COMPLETION_OUTCOME_REWARD
            )
        else:
            correction = config.V1_COMPLETION_OUTCOME_REWARD
        self.reward_assembler.buffer_event(TimestampedReward(
            event_time_sim, correction, record.decision_id, "TASK_COMPLETED"
        ))

    def _close_pending(self, *, next_state, next_context, next_time, terminal):
        data = self.pending
        cache_started = time.perf_counter()
        if terminal:
            cached_features = np.empty(
                (0, self.runtime.candidate_feature_encoder.feature_dim),
                dtype=np.float32,
            )
        else:
            chunks = tuple(
                np.asarray(chunk, dtype=np.float32)
                for chunk in self._next_candidate_feature_chunks(next_context)
                if len(chunk)
            )
            cached_features = (
                np.concatenate(chunks, axis=0)
                if chunks
                else np.empty(
                    (0, self.runtime.candidate_feature_encoder.feature_dim),
                    dtype=np.float32,
                )
            )
        if self.profiler is not None:
            self.profiler.add(
                "transition_feature_cache_seconds",
                time.perf_counter() - cache_started,
            )
        transition = self.reward_assembler.build_transition(
            global_state_before=data["state"],
            selected_candidate_id=data["candidate_id"],
            selected_candidate_features=data["features"],
            immediate_reward=data["immediate_reward"],
            global_state_after=next_state,
            next_candidate_features=(),
            next_candidate_context=None,
            decision_time_sim=data["record"].decision_time_sim,
            next_transition_time_sim=next_time,
            terminal=terminal,
        )
        transition = replace(
            transition,
            next_candidate_features=cached_features,
        )
        self.replay.add(transition)
        self.transition_count += 1
        if len(self.replay) >= self.min_replay_size:
            for _ in range(self.updates_per_transition):
                batch = self.replay.sample(self.batch_size, self.replay_random)
                loss = self.trainer.train_batch(batch)
                self.losses.append(loss)
                self.update_count += 1
                self.policy.epsilon = max(
                    config.EPSILON_MIN,
                    self.policy.epsilon * config.EPSILON_DECAY,
                )
                if (
                    self.update_count
                    % max(1, config.V1_TARGET_UPDATE_INTERVAL)
                    == 0
                ):
                    self.trainer.update_target()
        self.pending = None

    def finalize(self, final_time):
        if self.pending is not None:
            self._close_pending(
                next_state=self.runtime.global_state(None),
                next_context=None,
                next_time=final_time,
                terminal=True,
            )


def _settle(runtime, loop, current_time, safety_cap=1000000, profiler=None):
    iterations = 0
    unresolved = {
        TaskState.QUEUED,
        TaskState.PENDING_UNCOMMITTED,
        TaskState.RESERVED,
        TaskState.TRANSMITTING,
        TaskState.RUNNING,
    }
    unsettled_tasks = (
        runtime.scheduler.state_machine.task_spec(task_id)
        for task_id in runtime.scheduler.state_machine.task_ids
        if runtime.scheduler.state_machine.runtime(task_id).state in unresolved
    )
    forecast_end = ensure_v1_runtime_forecasts_for_tasks(runtime, unsettled_tasks)
    while any(
        runtime.scheduler.state_machine.runtime(task_id).state in unresolved
        for task_id in runtime.scheduler.state_machine.task_ids
    ):
        iterations += 1
        if iterations > safety_cap:
            raise RuntimeError("v1 training settlement safety cap exceeded")
        deadlines = [
            runtime.scheduler.state_machine.task_spec(task_id).absolute_latest_start_sim
            for task_id in runtime.scheduler.state_machine.task_ids
            if runtime.scheduler.state_machine.runtime(task_id).state
            in {TaskState.QUEUED, TaskState.PENDING_UNCOMMITTED}
        ]
        next_event = runtime.scheduler.event_engine.next_event_time_sim
        choices = [value for value in deadlines if value >= current_time - 1e-12]
        if next_event is not None and next_event >= current_time - 1e-12:
            choices.append(next_event)
        if not choices:
            raise RuntimeError("unsettled training state has no future event/deadline")
        current_time = min(choices)
        result = _run_profiled_operation(
            profiler,
            "scheduler_cycle_count",
            lambda: runtime.scheduler.run_cycle(
                current_time,
                forecast_covered_until_sim=forecast_end,
            ),
        )
        _run_profiled_operation(
            profiler,
            "training_process_cycle_count",
            lambda: loop.process_cycle(result, check_invariants=False),
        )
    return current_time


def _generate_arrivals(runtime, current, cycle, total_capacity):
    lam, _ = runtime.task_manager.get_dynamic_task_rate(current)
    return runtime.task_manager.generate_task_specs(
        np.random.poisson(lam),
        current,
        cycle,
        cpu_budget=(
            total_capacity
            * config.SCHEDULING_CYCLE
            * config.TASK_PEAK_LOAD_MULTIPLIER
        ),
    )


def _estimate_candidate_work(
    task_count,
    total_slots,
    max_slots,
    *,
    batch_size,
    min_replay_size,
    updates_per_transition,
    bootstrap_candidate_limit=None,
):
    batch_size = _positive_int("batch_size", batch_size)
    min_replay_size = max(
        batch_size,
        _positive_int("min_replay_size", min_replay_size),
    )
    updates_per_transition = _positive_int(
        "updates_per_transition", updates_per_transition
    )
    transition_count = int(task_count)
    update_count = (
        max(0, transition_count - min_replay_size + 1)
        * updates_per_transition
    )
    replay_samples = update_count * batch_size
    selection_visits = 2 * int(total_slots)
    if transition_count:
        if bootstrap_candidate_limit is None:
            # Each replay sample regenerates one complete next-candidate set.
            # The exact sampled contexts are random, so the trace mean is the
            # honest pre-run expectation and max_slots gives a conservative
            # upper bound.
            bootstrap_expected = (
                replay_samples * int(total_slots) + transition_count - 1
            ) // transition_count
            bootstrap_upper = replay_samples * int(max_slots)
        else:
            limit = _positive_int(
                "bootstrap_candidate_limit", bootstrap_candidate_limit
            )
            mean_slots = (int(total_slots) + transition_count - 1) // transition_count
            bootstrap_expected = replay_samples * min(limit, mean_slots)
            bootstrap_upper = replay_samples * min(limit, int(max_slots))
    else:
        bootstrap_expected = 0
        bootstrap_upper = 0
    return {
        "estimated_transition_count": transition_count,
        "estimated_update_count": update_count,
        "estimated_replay_context_samples": replay_samples,
        "selection_candidate_passes": 2,
        "estimated_selection_candidate_visits": selection_visits,
        "estimated_bootstrap_candidate_visits": bootstrap_expected,
        "estimated_bootstrap_candidate_visits_upper_bound": bootstrap_upper,
        "estimated_total_candidate_visits": (
            selection_visits + bootstrap_expected
        ),
        "bootstrap_candidate_limit": bootstrap_candidate_limit,
    }


def preflight_training(
    steps,
    seed,
    *,
    batch_size=None,
    min_replay_size=None,
    updates_per_transition=None,
    bootstrap_candidate_limit=None,
):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    short_horizon = max(config.V1_FORECAST_STEP_SIM, 2.0 * config.V1_FORECAST_STEP_SIM)
    runtime = create_v1_runtime(forecast_end_sim=short_horizon, random_seed=seed)
    total_capacity = sum(
        runtime.calendar.node_capacity(node)
        for node in runtime.infrastructure.compute_nodes
    )
    current = 0.0
    task_count = 0
    total_slots = 0
    max_slots = 0
    declared_slots = 0
    sla_counts = {item.value: 0 for item in SlaType}
    for cycle in range(steps):
        current += config.SCHEDULING_CYCLE
        arrivals = _generate_arrivals(runtime, current, cycle, total_capacity)
        for task in arrivals:
            theoretical = runtime.scheduler.candidate_generator.theoretical_slot_count(
                task, current
            )
            declared_slots += theoretical
            slots = (
                min(
                    theoretical,
                    config.V1_CANDIDATE_POOL_MAX_BY_SLA[task.sla_type.value],
                )
                if config.V1_CANDIDATE_MODE == "layered_pool"
                else theoretical
            )
            task_count += 1
            total_slots += slots
            max_slots = max(max_slots, slots)
            sla_counts[task.sla_type.value] += 1
    batch_size = (
        config.BATCH_SIZE if batch_size is None else batch_size
    )
    min_replay_size = (
        config.V1_REPLAY_MIN_SIZE
        if min_replay_size is None else min_replay_size
    )
    updates_per_transition = (
        config.V1_TRAIN_UPDATES_PER_TRANSITION
        if updates_per_transition is None else updates_per_transition
    )
    bootstrap_candidate_limit = _optional_positive_int(
        "bootstrap_candidate_limit", bootstrap_candidate_limit
    )
    work = _estimate_candidate_work(
        task_count,
        total_slots,
        max_slots,
        batch_size=batch_size,
        min_replay_size=min_replay_size,
        updates_per_transition=updates_per_transition,
        bootstrap_candidate_limit=bootstrap_candidate_limit,
    )
    return {
        "steps": steps,
        "seed": seed,
        "task_count": task_count,
        "sla_counts": sla_counts,
        "theoretical_candidate_slots": total_slots,
        "declared_complete_candidate_slots": declared_slots,
        "max_candidate_slots_per_task": max_slots,
        "training_parameters": {
            "batch_size": batch_size,
            "min_replay_size": max(batch_size, min_replay_size),
            "updates_per_transition": updates_per_transition,
            "bootstrap_candidate_limit": bootstrap_candidate_limit,
        },
        **work,
        "objective_calibration_ready": not (
            config.V1_COST_REFERENCE_YUAN == 0.0
            and config.V1_COST_SCALE_YUAN == 1.0
            and config.V1_SOFT_TARDINESS_WEIGHT == 0.0
            and config.V1_FLEXIBLE_TARDINESS_WEIGHT == 0.0
        ),
        "objective_parameters": {
            "cost_reference_yuan": config.V1_COST_REFERENCE_YUAN,
            "cost_scale_yuan": config.V1_COST_SCALE_YUAN,
            "green_absorption_delta_scale": config.V1_GREEN_ABSORPTION_DELTA_SCALE,
            "soft_tardiness_weight": config.V1_SOFT_TARDINESS_WEIGHT,
            "flexible_tardiness_weight": config.V1_FLEXIBLE_TARDINESS_WEIGHT,
        },
    }


def _checkpoint_path(output):
    return output.with_name(output.stem + ".last.pt")


def _save_checkpoint(path, runtime, loop, *, cycle, current_time, seed, run_config):
    metadata = _metadata(runtime)
    payload = {
        "checkpoint_type": "v1_training_resume",
        "checkpoint_version": 1,
        "runtime": runtime,
        "loop": loop,
        "cycle": cycle,
        "current_time_sim": current_time,
        "seed": seed,
        "run_config": run_config,
        "config_hash": _canonical_hash(run_config),
        "metadata": asdict(metadata),
        "model_id": metadata.model_id,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_checkpoint(
    path,
    *,
    device,
    seed,
    run_config,
    forecast_end_sim=None,
    profiler=None,
):
    # RNG states are CPU ByteTensors even when the resumed model will run on
    # CUDA. Load the checkpoint on CPU first, then move model and optimizer
    # tensors through the explicit, tested device migration below.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_type") != "v1_training_resume":
        raise ValueError("resume file is not a v1 training checkpoint")
    if checkpoint.get("seed") != seed:
        raise ValueError("resume seed does not match --seed")
    saved_run_config = checkpoint.get("run_config")
    if not _resume_config_compatible(saved_run_config, run_config):
        raise ValueError("resume training configuration mismatch")
    runtime = checkpoint["runtime"]
    loop = checkpoint["loop"]
    if forecast_end_sim is not None:
        extend_v1_runtime_forecasts(runtime, forecast_end_sim)
    restored_tasks = (
        runtime.scheduler.state_machine.task_spec(task_id)
        for task_id in runtime.scheduler.state_machine.task_ids
    )
    ensure_v1_runtime_forecasts_for_tasks(runtime, restored_tasks)
    generator = runtime.scheduler.candidate_generator
    if not hasattr(generator, "active_wait_enabled"):
        generator.active_wait_enabled = True
    encoder = runtime.candidate_feature_encoder
    if not hasattr(encoder, "disabled_feature_groups"):
        if run_config.get("V1_DISABLED_CANDIDATE_FEATURE_GROUPS"):
            raise ValueError(
                "legacy checkpoint cannot enable disabled candidate feature groups"
            )
        encoder.disabled_feature_groups = ()
        encoder._disabled_feature_indices = ()
    clock = loop.reward_assembler.clock
    if not hasattr(clock, "mode"):
        clock.mode = "physical_time"
        clock.decision_gamma = 0.95
    validate_checkpoint_metadata(
        checkpoint.get("metadata", {}),
        encoder.feature_schema_hash,
    )
    loop.runtime = runtime
    loop.policy = runtime.scheduler.policy
    loop.trainer.next_candidate_provider = loop._next_candidate_feature_chunks
    loop.to_device(device)
    loop.set_candidate_chunk_size(run_config["candidate_chunk_size"])
    loop.set_bootstrap_candidate_limit(run_config.get("bootstrap_candidate_limit"))
    loop.policy.audit_candidate_set_hash = False
    loop.profiler = profiler
    runtime.scheduler.profiler = profiler
    runtime.scheduler.candidate_generator.profiler = profiler
    loop.policy.profiler = profiler
    loop.trainer.profiler = profiler
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"])
    if device == "cuda" and checkpoint.get("torch_cuda_random_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random_state"])
    _assert_scheduler_invariants(runtime.scheduler)
    return runtime, loop, int(checkpoint["cycle"]), float(checkpoint["current_time_sim"])


def _append_log(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_training(
    steps,
    seed,
    output_path,
    *,
    device="auto",
    candidate_chunk_size=None,
    batch_size=None,
    min_replay_size=None,
    updates_per_transition=None,
    bootstrap_candidate_limit=None,
    checkpoint_every=None,
    log_every=None,
    invariant_check_every=None,
    resume_path=None,
    allow_large_run=False,
    allow_uncalibrated_objective=False,
    run_preflight=True,
    profile=False,
    profile_output_path=None,
):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    device = _resolve_device(device)
    chunk_size = _positive_int(
        "candidate_chunk_size",
        config.V1_CANDIDATE_CHUNK_SIZE
        if candidate_chunk_size is None else candidate_chunk_size,
    )
    batch_size = _positive_int(
        "batch_size", config.BATCH_SIZE if batch_size is None else batch_size
    )
    min_replay_size = _positive_int(
        "min_replay_size",
        config.V1_REPLAY_MIN_SIZE if min_replay_size is None else min_replay_size,
    )
    updates = _positive_int(
        "updates_per_transition",
        config.V1_TRAIN_UPDATES_PER_TRANSITION
        if updates_per_transition is None else updates_per_transition,
    )
    bootstrap_candidate_limit = _optional_positive_int(
        "bootstrap_candidate_limit", bootstrap_candidate_limit
    )
    checkpoint_every = _positive_int(
        "checkpoint_every",
        config.V1_CHECKPOINT_INTERVAL_CYCLES
        if checkpoint_every is None else checkpoint_every,
    )
    log_every = _positive_int(
        "log_every", config.V1_LOG_INTERVAL_CYCLES if log_every is None else log_every,
    )
    invariant_check_every = _positive_int(
        "invariant_check_every",
        config.V1_INVARIANT_CHECK_INTERVAL_CYCLES
        if invariant_check_every is None else invariant_check_every,
    )
    output = Path(output_path)
    log_path = output.with_name(output.stem + ".training.csv")
    profiler = TrainingPerformanceProfiler() if profile else None
    profile_path = (
        Path(profile_output_path)
        if profile_output_path is not None
        else output.with_name(output.stem + ".profile.json")
    )
    run_config = _training_config_view(
        candidate_chunk_size=chunk_size,
        batch_size=batch_size,
        min_replay_size=max(batch_size, min_replay_size),
        updates_per_transition=updates,
        bootstrap_candidate_limit=bootstrap_candidate_limit,
        invariant_check_every=invariant_check_every,
    )
    horizon = steps * config.SCHEDULING_CYCLE + config.V1_MAX_FORECAST_LOOKAHEAD_SIM

    if run_preflight and resume_path is None:
        report = preflight_training(
            steps,
            seed,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            updates_per_transition=updates,
            bootstrap_candidate_limit=bootstrap_candidate_limit,
        )
        unsafe = (
            report["theoretical_candidate_slots"]
            > config.V1_PREFLIGHT_MAX_TOTAL_SLOTS
            or report["max_candidate_slots_per_task"]
            > config.V1_PREFLIGHT_MAX_SLOTS_PER_TASK
            or report["estimated_total_candidate_visits"]
            > config.V1_PREFLIGHT_MAX_ESTIMATED_CANDIDATE_VISITS
        )
        print(json.dumps({
            "preflight": report,
            "requires_large_run_override": unsafe,
            "requires_objective_override": not report["objective_calibration_ready"],
        }, ensure_ascii=False))
        if not report["objective_calibration_ready"] and not allow_uncalibrated_objective:
            raise RuntimeError(
                "objective parameters are still the documented pilot placeholders; "
                "calibrate cost/tardiness scales or rerun with "
                "--allow-uncalibrated-objective for a diagnostic smoke run"
            )
        if unsafe and not allow_large_run:
            raise RuntimeError(
                "preflight candidate volume exceeds the safety gate; inspect the "
                "report and rerun with --allow-large-run if the runtime is intentional"
            )

    if resume_path is not None:
        runtime, loop, start_cycle, current = _load_checkpoint(
            Path(resume_path),
            device=device,
            seed=seed,
            run_config=run_config,
            forecast_end_sim=horizon,
            profiler=profiler,
        )
        if start_cycle > steps:
            raise ValueError("checkpoint cycle exceeds requested total --steps")
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        runtime = create_v1_runtime(
            policy_name="candidate_dqn",
            forecast_end_sim=horizon,
            random_seed=seed,
            device=device,
            candidate_chunk_size=chunk_size,
        )
        runtime.scheduler.policy.epsilon = config.EPSILON_START
        runtime.scheduler.policy.record_selection_traces = True
        loop = V1TrainingLoop(
            runtime,
            device=device,
            candidate_chunk_size=chunk_size,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            updates_per_transition=updates,
            bootstrap_candidate_limit=bootstrap_candidate_limit,
            random_seed=seed,
            profiler=profiler,
        )
        start_cycle = 0
        current = 0.0

    forecast_end = v1_runtime_forecast_end(runtime)

    total_capacity = sum(
        runtime.calendar.node_capacity(node)
        for node in runtime.infrastructure.compute_nodes
    )
    interval_wall_start = time.perf_counter()
    profile_wall_start = interval_wall_start
    interval_candidates = loop.candidate_count
    interval_losses = len(loop.losses)
    for cycle in range(start_cycle, steps):
        current += config.SCHEDULING_CYCLE
        completed_cycle = cycle + 1
        arrivals = _generate_arrivals(runtime, current, cycle, total_capacity)
        if arrivals:
            forecast_end = ensure_v1_runtime_forecasts_for_tasks(runtime, arrivals)
        result = _run_profiled_operation(
            profiler,
            "scheduler_cycle_count",
            lambda: runtime.scheduler.run_cycle(
                current,
                arrivals=arrivals,
                forecast_covered_until_sim=forecast_end,
            ),
        )
        _run_profiled_operation(
            profiler,
            "training_process_cycle_count",
            lambda: loop.process_cycle(
                result,
                check_invariants=_should_run_invariant_check(
                    completed_cycle,
                    invariant_check_every=invariant_check_every,
                    checkpoint_every=checkpoint_every,
                    final_cycle=steps,
                ),
            ),
        )
        if completed_cycle % log_every == 0 or completed_cycle == steps:
            wall = time.perf_counter() - interval_wall_start
            recent_losses = loop.losses[interval_losses:]
            row = {
                "cycle": completed_cycle,
                "time_sim": current,
                "tasks": runtime.scheduler.state_machine.task_count,
                "transitions": loop.transition_count,
                "updates": loop.update_count,
                "replay_size": len(loop.replay),
                "epsilon": loop.policy.epsilon,
                "mean_loss": (
                    float(np.mean(recent_losses)) if recent_losses else ""
                ),
                "candidates": loop.candidate_count,
                "candidates_since_log": loop.candidate_count - interval_candidates,
                "wall_seconds_since_log": wall,
                "device": device,
            }
            log_started = time.perf_counter()
            _append_log(log_path, row)
            progress = {"progress": row}
            if profiler is not None:
                progress["profile"] = profiler.summary(
                    time.perf_counter() - profile_wall_start
                )["sections_percent"]
            print(json.dumps(progress, ensure_ascii=False))
            if profiler is not None:
                profiler.add(
                    "logging_seconds", time.perf_counter() - log_started
                )
            interval_wall_start = time.perf_counter()
            interval_candidates = loop.candidate_count
            interval_losses = len(loop.losses)
        if completed_cycle % checkpoint_every == 0:
            checkpoint_started = time.perf_counter()
            _save_checkpoint(
                _checkpoint_path(output), runtime, loop,
                cycle=completed_cycle, current_time=current, seed=seed,
                run_config=run_config,
            )
            if profiler is not None:
                profiler.add(
                    "checkpoint_seconds",
                    time.perf_counter() - checkpoint_started,
                )

    # A resume checkpoint must represent the online state at the requested
    # cycle boundary. Settlement advances physical time to drain all accepted
    # work and is only for producing the frozen model artifact; saving after
    # settlement would make a later resume silently continue from a different
    # time line.
    _assert_scheduler_invariants(runtime.scheduler)
    checkpoint_started = time.perf_counter()
    _save_checkpoint(
        _checkpoint_path(output), runtime, loop,
        cycle=steps, current_time=current, seed=seed,
        run_config=run_config,
    )
    if profiler is not None:
        profiler.add(
            "checkpoint_seconds",
            time.perf_counter() - checkpoint_started,
        )

    current = _settle(runtime, loop, current, profiler=profiler)
    _assert_scheduler_invariants(runtime.scheduler)
    _run_profiled_operation(
        profiler,
        "training_process_cycle_count",
        lambda: loop.finalize(current),
    )
    metadata = _metadata(runtime)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_started = time.perf_counter()
    torch.save({
        "model_state_dict": runtime.candidate_q_network.state_dict(),
        "target_state_dict": loop.target.state_dict(),
        "optimizer_state_dict": loop.trainer.optimizer.state_dict(),
        "metadata": asdict(metadata),
        "model_id": metadata.model_id,
        "training_steps": steps,
        "transition_count": loop.transition_count,
        "update_count": loop.update_count,
        "mean_loss": float(np.mean(loop.losses)) if loop.losses else None,
        "epsilon": loop.policy.epsilon,
        "seed": seed,
        "device": device,
        "run_config": run_config,
        "config_hash": _canonical_hash(run_config),
    }, output)
    if profiler is not None:
        profiler.add(
            "artifact_save_seconds", time.perf_counter() - artifact_started
        )
        total_profile_wall = time.perf_counter() - profile_wall_start
        _write_profile_outputs(profiler, total_profile_wall, profile_path)
    return output


def main():
    parser = argparse.ArgumentParser(description="Train the frozen v1.0 candidate DQN")
    parser.add_argument("--steps", type=int, default=config.MAX_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/v1/logs/candidate_dqn.pt")
    parser.add_argument("--resume")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--candidate-chunk-size", type=int, default=config.V1_CANDIDATE_CHUNK_SIZE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--min-replay-size", type=int, default=config.V1_REPLAY_MIN_SIZE)
    parser.add_argument("--updates-per-transition", type=int, default=config.V1_TRAIN_UPDATES_PER_TRANSITION)
    parser.add_argument(
        "--bootstrap-candidate-limit",
        type=int,
        help=(
            "training-only cap on candidates scanned for each replay "
            "bootstrap target; omitted keeps exact complete bootstrap"
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=config.V1_CHECKPOINT_INTERVAL_CYCLES)
    parser.add_argument("--log-every", type=int, default=config.V1_LOG_INTERVAL_CYCLES)
    parser.add_argument(
        "--invariant-check-every",
        type=int,
        default=config.V1_INVARIANT_CHECK_INTERVAL_CYCLES,
    )
    parser.add_argument("--allow-large-run", action="store_true")
    parser.add_argument("--allow-uncalibrated-objective", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--ablation-variant", choices=variant_names()
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="record exclusive training-stage wall-clock timings",
    )
    parser.add_argument(
        "--profile-output",
        help="profiling JSON path (default: <model>.profile.json)",
    )
    args = parser.parse_args()
    with apply_ablation_variant(args.ablation_variant):
        if args.preflight_only:
            print(json.dumps(preflight_training(
                args.steps,
                args.seed,
                batch_size=args.batch_size,
                min_replay_size=args.min_replay_size,
                updates_per_transition=args.updates_per_transition,
                bootstrap_candidate_limit=args.bootstrap_candidate_limit,
            ), ensure_ascii=False, indent=2))
            return
        path = run_training(
            args.steps,
            args.seed,
            args.output,
            device=args.device,
            candidate_chunk_size=args.candidate_chunk_size,
            batch_size=args.batch_size,
            min_replay_size=args.min_replay_size,
            updates_per_transition=args.updates_per_transition,
            bootstrap_candidate_limit=args.bootstrap_candidate_limit,
            checkpoint_every=args.checkpoint_every,
            log_every=args.log_every,
            invariant_check_every=args.invariant_check_every,
            resume_path=args.resume,
            allow_large_run=args.allow_large_run,
            allow_uncalibrated_objective=args.allow_uncalibrated_objective,
            run_preflight=not args.skip_preflight,
            profile=args.profile,
            profile_output_path=args.profile_output,
        )
    completed = {"status": "complete", "model": str(path)}
    if args.profile:
        completed["profile"] = str(
            Path(args.profile_output)
            if args.profile_output
            else path.with_name(path.stem + ".profile.json")
        )
    print(json.dumps(completed, ensure_ascii=False))


if __name__ == "__main__":
    main()
