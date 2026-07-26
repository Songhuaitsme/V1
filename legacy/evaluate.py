"""Evaluate a frozen scheduling policy on reproducible task traces."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import tensorflow as tf

from shared import config
from legacy.dqn_agent import DQNAgent
from legacy.gnn_agent import GNNAgent
from legacy.network_env import (
    NetworkEnvironment,
    evaluate_schedule_candidates,
    get_max_retries_for_task,
)
from shared.task_manager import TaskManager
from legacy.train import (
    add_success_allocation,
    build_state,
    compute_valid_actions,
    get_checkpoint_paths,
    model_path_exists,
    update_active_tasks,
    warmup_environment,
)


DEFAULT_SEEDS = (42, 43, 44, 45, 46)
GREEN_RICH_REGIONS = frozenset("GHIJKLM")
CONSTRAINT_KEYS = (
    "sla_violation",
    "drop",
    "cost_over_budget",
    "overload",
)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1e-8)


def existing_model_file(path: str | Path) -> Path:
    path = Path(path)
    candidates = (path, Path(f"{path}.weights.h5"))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Model file not found: {path}")


def default_model_path() -> str:
    _, checkpoint_path, final_model_path = get_checkpoint_paths()
    if model_path_exists(final_model_path):
        return final_model_path
    if model_path_exists(checkpoint_path):
        return checkpoint_path
    raise FileNotFoundError(
        "No trained model found. Pass --model-path explicitly."
    )


def create_agent(env, model_path: str):
    if config.USE_GNN_AGENT:
        agent = GNNAgent(
            graph_state_template=env.get_graph_state(wait_queue=[]),
            action_dim=env.action_space_dim,
        )
    else:
        agent = DQNAgent(
            state_dim=env.state_space_dim,
            action_dim=env.action_space_dim,
        )

    agent.load(model_path)
    agent.epsilon = 0.0

    if not config.USE_GNN_AGENT:
        model_state_dim = int(agent.model.input_shape[-1])
        model_action_dim = int(agent.model.output_shape[-1])
        if model_state_dim != env.state_space_dim:
            raise ValueError(
                f"Model state dimension {model_state_dim} does not match "
                f"environment dimension {env.state_space_dim}."
            )
        if model_action_dim != env.action_space_dim:
            raise ValueError(
                f"Model action dimension {model_action_dim} does not match "
                f"environment dimension {env.action_space_dim}."
            )
    return agent


def build_task_trace(
    seed: int,
    source_nodes: Iterable[str],
    total_compute_capacity: float,
    initial_time: float,
    steps: int,
) -> list[list[dict[str, Any]]]:
    """Generate formal-evaluation tasks independently from policy execution."""
    set_random_seed(seed)
    task_manager = TaskManager(
        source_nodes,
        total_compute_capacity=total_compute_capacity,
    )
    global_time = float(initial_time)
    cycle_supply = total_compute_capacity * config.SCHEDULING_CYCLE
    peak_budget = cycle_supply * getattr(config, "TASK_PEAK_LOAD_MULTIPLIER", 1.3)
    trace: list[list[dict[str, Any]]] = []

    for cycle in range(steps):
        global_time += config.SCHEDULING_CYCLE
        task_rate, _ = task_manager.get_dynamic_task_rate(global_time)
        tasks = task_manager.generate_tasks(
            np.random.poisson(task_rate),
            global_time,
            cycle,
            cpu_budget=peak_budget,
        )
        trace.append(tasks)
    return trace


@dataclass
class EvaluationMetrics:
    generated: int = 0
    processed: int = 0
    succeeded: int = 0
    dropped: int = 0
    deferred: int = 0
    queue_overflow_drops: int = 0
    rewards: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    baseline_costs: list[float] = field(default_factory=list)
    cpu_time_demands: list[float] = field(default_factory=list)
    cost_per_cpu_times: list[float] = field(default_factory=list)
    cost_ratios: list[float] = field(default_factory=list)
    physical_latencies: list[float] = field(default_factory=list)
    e2e_latencies: list[float] = field(default_factory=list)
    green_matches: list[float] = field(default_factory=list)
    green_absorptions: list[float] = field(default_factory=list)
    cost_savings: list[float] = field(default_factory=list)
    target_regions: list[str] = field(default_factory=list)
    constraint_values: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    reward_components: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    system_green_samples: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    cpu_usage_samples: list[float] = field(default_factory=list)

    def record_queue_overflow(self) -> None:
        self.dropped += 1
        self.queue_overflow_drops += 1

    def record_no_action_drop(self) -> None:
        self.dropped += 1
        for key in CONSTRAINT_KEYS:
            self.constraint_values[key].append(1.0 if key == "drop" else 0.0)

    def record_step(self, reward: float, info: dict[str, Any]) -> None:
        self.rewards.append(float(reward))
        constraints = info.get("constraint_costs", {})
        for key in CONSTRAINT_KEYS:
            self.constraint_values[key].append(float(constraints.get(key, 0.0)))
        for key, value in info.get("reward_components", {}).items():
            self.reward_components[key].append(float(value))

    def record_success(self, env, task: dict[str, Any], info: dict[str, Any]) -> None:
        self.succeeded += 1
        coordination = info.get("coordination", {})
        cpu_time = float(
            info.get("cpu_time_demand", env.get_task_cpu_time_demand(task))
        )
        raw_cost = float(info.get("cost", 0.0))

        self.costs.append(raw_cost)
        self.cpu_time_demands.append(cpu_time)
        self.baseline_costs.append(float(coordination.get("baseline_cost", 0.0)))
        self.cost_per_cpu_times.append(
            float(coordination.get("cost_per_cpu_time", safe_ratio(raw_cost, cpu_time)))
        )
        self.cost_ratios.append(float(coordination.get("cost_ratio", 0.0)))
        self.cost_savings.append(float(coordination.get("cost_saving_ratio", 0.0)))
        self.green_matches.append(float(coordination.get("green_match_ratio", 0.0)))
        self.green_absorptions.append(
            float(coordination.get("green_absorption_ratio", 0.0))
        )
        self.physical_latencies.append(float(info["delays"]["physical"]))
        self.e2e_latencies.append(float(info["delays"]["end_to_end"]))

        target = info["target_node"]
        region = env.topo_manager.graph.nodes[target].get("region", "Unknown")
        self.target_regions.append(str(region))

    def sample_system(self, env, global_time: float) -> None:
        green = env.get_system_green_absorption(global_time)
        for key in (
            "system_green_absorption_ratio",
            "green_unused_ratio",
            "green_load_coverage_ratio",
            "green_supply_demand_ratio",
            "total_green_used_mw",
            "total_green_supply_mw",
            "total_green_unused_mw",
            "total_power_demand_mw",
        ):
            self.system_green_samples[key].append(float(green.get(key, 0.0)))

        usages = [
            resource["used"] / resource["total"]
            for resource in env.node_resources.values()
            if resource["total"] > 0
        ]
        self.cpu_usage_samples.append(mean(usages))

    def summary(
        self,
        seed: int,
        steps: int,
        wait_queue_length: int,
        active_task_count: int,
    ) -> dict[str, float | int]:
        total_cost = sum(self.costs)
        total_baseline_cost = sum(self.baseline_costs)
        total_cpu_time = sum(self.cpu_time_demands)
        resolved = self.succeeded + self.dropped
        region_counts = Counter(self.target_regions)
        green_rich = sum(region_counts[region] for region in GREEN_RICH_REGIONS)

        result: dict[str, float | int] = {
            "seed": seed,
            "steps": steps,
            "generated_tasks": self.generated,
            "processed_attempts": self.processed,
            "succeeded_tasks": self.succeeded,
            "dropped_tasks": self.dropped,
            "deferred_actions": self.deferred,
            "queue_overflow_drops": self.queue_overflow_drops,
            "pending_tasks": wait_queue_length,
            "active_task_count": active_task_count,
            "completion_rate": safe_ratio(self.succeeded, resolved),
            "throughput_rate": safe_ratio(self.succeeded, self.generated),
            "drop_rate": safe_ratio(self.dropped, self.generated),
            "weighted_cost_per_cpu_time": safe_ratio(total_cost, total_cpu_time),
            "weighted_cost_ratio": safe_ratio(total_cost, total_baseline_cost),
            "avg_cost_per_cpu_time": mean(self.cost_per_cpu_times),
            "p50_cost_per_cpu_time": percentile(self.cost_per_cpu_times, 50),
            "p95_cost_per_cpu_time": percentile(self.cost_per_cpu_times, 95),
            "avg_cost_ratio": mean(self.cost_ratios),
            "p95_cost_ratio": percentile(self.cost_ratios, 95),
            "avg_cost_saving_ratio": mean(self.cost_savings),
            "avg_physical_latency": mean(self.physical_latencies),
            "avg_end_to_end_latency": mean(self.e2e_latencies),
            "p95_end_to_end_latency": percentile(self.e2e_latencies, 95),
            "action_green_match_ratio": mean(self.green_matches),
            "action_green_absorption_ratio": mean(self.green_absorptions),
            "selected_green_rich_ratio": safe_ratio(green_rich, self.succeeded),
            "avg_cpu_usage": mean(self.cpu_usage_samples),
            "avg_base_reward": mean(self.rewards),
        }

        for key, values in self.system_green_samples.items():
            result[f"avg_{key}"] = mean(values)
        for key in CONSTRAINT_KEYS:
            values = self.constraint_values[key]
            result[f"avg_constraint_{key}"] = mean(values)
            result[f"{key}_event_rate"] = safe_ratio(
                sum(value > 0.0 for value in values),
                len(values),
            )
        for key, values in sorted(self.reward_components.items()):
            result[f"avg_{key}"] = mean(values)
        return result


def run_frozen_policy(
    seed: int,
    env,
    task_manager: TaskManager,
    agent,
    task_trace: list[list[dict[str, Any]]],
    global_time: float,
    active_tasks: list[dict[str, Any]],
    wait_queue: list[dict[str, Any]],
    sample_interval: int,
) -> dict[str, Any]:
    metrics = EvaluationMetrics()

    for cycle, trace_tasks in enumerate(task_trace):
        global_time += config.SCHEDULING_CYCLE
        update_active_tasks(env, active_tasks, global_time)

        new_tasks = copy.deepcopy(trace_tasks)
        metrics.generated += len(new_tasks)
        for task in new_tasks:
            if len(wait_queue) < config.MAX_QUEUE_LENGTH:
                wait_queue.append(task)
            else:
                metrics.record_queue_overflow()

        wait_queue.sort(
            key=lambda task: task_manager.calculate_priority(task, global_time),
            reverse=True,
        )
        deferred_batch: list[dict[str, Any]] = []

        for _ in range(min(len(wait_queue), config.MAX_TASKS_PER_CYCLE)):
            task = wait_queue.pop(0)
            metrics.processed += 1
            task["current_time_context"] = global_time
            candidates = evaluate_schedule_candidates(
                env,
                task,
                wait_queue,
                global_time,
            )
            valid_actions = compute_valid_actions(
                env,
                task,
                wait_queue,
                global_time,
                candidates=candidates,
            )
            if not valid_actions:
                metrics.record_no_action_drop()
                continue

            state = build_state(env, task, wait_queue=wait_queue)
            action = agent.act(state, valid_actions=valid_actions)
            _, reward, _, info = env.step(
                action,
                task,
                wait_queue,
                candidates=candidates,
            )
            metrics.record_step(reward, info)

            status = info.get("status")
            if status == "Success":
                metrics.record_success(env, task, info)
                add_success_allocation(
                    env=env,
                    active_tasks=active_tasks,
                    task=task,
                    info=info,
                    global_time=global_time,
                )
            elif status == "Deferred":
                requeued = info.get("deferred_task", task)
                requeued["retry_count"] = requeued.get("retry_count", 0) + 1
                if requeued["retry_count"] <= get_max_retries_for_task(requeued):
                    deferred_batch.append(requeued)
                    metrics.deferred += 1
                else:
                    metrics.dropped += 1
            else:
                retry_count = task.get("retry_count", 0)
                can_retry = (
                    status != "Failed_Wait"
                    and retry_count < get_max_retries_for_task(task)
                )
                if can_retry:
                    task["retry_count"] = retry_count + 1
                    deferred_batch.append(task)
                else:
                    metrics.dropped += 1

        wait_queue.extend(deferred_batch)
        if cycle % sample_interval == 0 or cycle == len(task_trace) - 1:
            metrics.sample_system(env, global_time)

    return metrics.summary(
        seed=seed,
        steps=len(task_trace),
        wait_queue_length=len(wait_queue),
        active_task_count=len(active_tasks),
    )


def evaluate_seed(
    model_path: str,
    seed: int,
    steps: int,
    initial_time: float,
    sample_interval: int,
    use_warmup: bool,
) -> dict[str, Any]:
    set_random_seed(seed)
    env = NetworkEnvironment()
    total_capacity = sum(resource["total"] for resource in env.node_resources.values())
    task_manager = TaskManager(
        env.base_stations,
        total_compute_capacity=total_capacity,
    )
    agent = create_agent(env, model_path)

    global_time = float(initial_time)
    active_tasks: list[dict[str, Any]] = []
    wait_queue: list[dict[str, Any]] = []
    if use_warmup and getattr(config, "ENABLE_ENV_WARMUP", False):
        global_time, active_tasks, wait_queue = warmup_environment(
            env=env,
            task_manager=task_manager,
            total_compute_capacity=total_capacity,
            global_time=global_time,
        )

    task_trace = build_task_trace(
        seed=seed,
        source_nodes=env.base_stations,
        total_compute_capacity=total_capacity,
        initial_time=global_time,
        steps=steps,
    )
    set_random_seed(seed + 1_000_000)
    agent.epsilon = 0.0

    result = run_frozen_policy(
        seed=seed,
        env=env,
        task_manager=task_manager,
        agent=agent,
        task_trace=task_trace,
        global_time=global_time,
        active_tasks=active_tasks,
        wait_queue=wait_queue,
        sample_interval=sample_interval,
    )
    return result


def serializable_config() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name in sorted(item for item in dir(config) if item.isupper()):
        value = getattr(config, name)
        try:
            json.dumps(value)
        except TypeError:
            value = str(value)
        snapshot[name] = value
    return snapshot


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for metric in results[0]:
        if metric == "seed":
            continue
        values = [float(result[metric]) for result in results]
        summary.append({
            "metric": metric,
            "mean": fmean(values),
            "std": pstdev(values),
            "min": min(values),
            "max": max(values),
        })
    return summary


def print_summary(summary: list[dict[str, Any]]) -> None:
    focus = {
        row["metric"]: row
        for row in summary
        if row["metric"] in {
            "completion_rate",
            "throughput_rate",
            "weighted_cost_per_cpu_time",
            "weighted_cost_ratio",
            "p95_cost_per_cpu_time",
            "avg_system_green_absorption_ratio",
            "avg_green_load_coverage_ratio",
            "avg_green_unused_ratio",
            "sla_violation_event_rate",
            "overload_event_rate",
        }
    }
    print("\n=== Frozen-policy evaluation summary ===")
    for metric, row in focus.items():
        print(f"{metric:42s} {row['mean']:.6f} +/- {row['std']:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen DQN/GNN scheduler on reproducible task traces."
    )
    parser.add_argument("--model-path", default=None, help="trained model path")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="evaluation seeds",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=2.0,
        help="simulated traffic days per seed",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="explicit cycles per seed; overrides --days",
    )
    parser.add_argument("--initial-time", type=float, default=0.0)
    parser.add_argument("--sample-interval", type=int, default=100)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--load-utilization", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        default="artifacts/legacy/evaluation/baseline",
        help="directory for per-seed results and metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.days <= 0:
        raise ValueError("--days must be positive")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.sample_interval <= 0:
        raise ValueError("--sample-interval must be positive")
    if not args.seeds:
        raise ValueError("At least one evaluation seed is required")

    if args.load_utilization is not None:
        config.TASK_LOAD_TARGET_UTILIZATION = float(args.load_utilization)

    model_path = args.model_path or default_model_path()
    model_file = existing_model_file(model_path)
    steps = args.steps or int(round(
        args.days
        * config.TRAFFIC_DAY_DURATION_IN_SIM
        / config.SCHEDULING_CYCLE
    ))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    results = []
    for seed in args.seeds:
        print(f"\n=== Evaluating seed {seed} for {steps} cycles ===")
        result = evaluate_seed(
            model_path=model_path,
            seed=seed,
            steps=steps,
            initial_time=args.initial_time,
            sample_interval=args.sample_interval,
            use_warmup=not args.no_warmup,
        )
        results.append(result)

    summary = summarize_results(results)
    write_csv(output_dir / "per_seed.csv", results)
    write_csv(output_dir / "summary.csv", summary)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_file),
        "model_sha256": file_sha256(model_file),
        "seeds": args.seeds,
        "steps_per_seed": steps,
        "simulated_days_per_seed": (
            steps * config.SCHEDULING_CYCLE / config.TRAFFIC_DAY_DURATION_IN_SIM
        ),
        "initial_time": args.initial_time,
        "warmup_enabled": not args.no_warmup and config.ENABLE_ENV_WARMUP,
        "warmup_cycles": config.ENV_WARMUP_CYCLES,
        "sample_interval": args.sample_interval,
        "config": serializable_config(),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"\nResults saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    if getattr(config, "SCHEDULER_ENGINE", "legacy") == "v1":
        from evaluate_v1 import main as main_v1
        main_v1()
    else:
        main()
