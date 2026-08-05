"""Executable formal v1.0 frozen-policy evaluation entry point."""

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import random

import networkx as nx
import numpy as np
import torch

from shared import config
from v1.ablation_settings import apply_ablation_variant, variant_names
from v1.evaluation_v1 import EvaluationRunner
from v1.learning import validate_checkpoint_metadata
from v1.scheduler import ObjectiveConfig
from v1.v1_runtime import create_v1_runtime


def _canonical_hash(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key.value if isinstance(key, Enum) else key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("formal evaluation output contains NaN or Infinity")
    return value


def _generate_trace(runtime, cutoff, seed):
    random.seed(seed)
    np.random.seed(seed)
    tasks = []
    time_sim = 0.0
    cycle = 0
    total_capacity = sum(
        runtime.calendar.node_capacity(node)
        for node in runtime.infrastructure.compute_nodes
    )
    while time_sim < cutoff - 1e-12:
        time_sim += config.SCHEDULING_CYCLE
        lam, _ = runtime.task_manager.get_dynamic_task_rate(time_sim)
        tasks.extend(runtime.task_manager.generate_task_specs(
            np.random.poisson(lam),
            time_sim,
            cycle,
            cpu_budget=(
                total_capacity
                * config.SCHEDULING_CYCLE
                * config.TASK_PEAK_LOAD_MULTIPLIER
            ),
        ))
        cycle += 1
    return tuple(task for task in tasks if task.arrival_time_sim < cutoff)


def run_evaluation(
    policy,
    cutoff,
    seed,
    safety_cap,
    model_path=None,
    *,
    device="cpu",
    candidate_chunk_size=None,
    soft_tardiness_weight=None,
    flexible_tardiness_weight=None,
    ablation_variant=None,
):
    if ablation_variant is not None:
        with apply_ablation_variant(ablation_variant):
            return run_evaluation(
                policy,
                cutoff,
                seed,
                safety_cap,
                model_path,
                device=device,
                candidate_chunk_size=candidate_chunk_size,
                soft_tardiness_weight=soft_tardiness_weight,
                flexible_tardiness_weight=flexible_tardiness_weight,
            )
    if (
        (soft_tardiness_weight is not None or flexible_tardiness_weight is not None)
        and policy != "equal_weight"
    ):
        raise ValueError("tardiness-weight overrides require equal_weight policy")
    soft_weight = (
        config.V1_SOFT_TARDINESS_WEIGHT
        if soft_tardiness_weight is None else float(soft_tardiness_weight)
    )
    flexible_weight = (
        config.V1_FLEXIBLE_TARDINESS_WEIGHT
        if flexible_tardiness_weight is None else float(flexible_tardiness_weight)
    )
    objective = ObjectiveConfig(
        config.V1_COST_REFERENCE_YUAN,
        config.V1_COST_SCALE_YUAN,
        config.V1_GREEN_ABSORPTION_DELTA_SCALE,
        config.V1_OBJECTIVE_COST_WEIGHT,
        config.V1_OBJECTIVE_GREEN_WEIGHT,
        config.V1_OBJECTIVE_BALANCE_WEIGHT,
        soft_weight,
        flexible_weight,
    )
    horizon = cutoff + config.V1_MAX_FORECAST_LOOKAHEAD_SIM
    runtime = create_v1_runtime(
        policy_name=policy,
        forecast_end_sim=horizon,
        random_seed=seed,
        device=device,
        candidate_chunk_size=candidate_chunk_size,
        objective_config=objective,
    )
    model_hash = _canonical_hash({"policy": policy})
    if policy == "candidate_dqn":
        if not model_path:
            raise ValueError("candidate_dqn evaluation requires --model-path")
        checkpoint_path = Path(model_path)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        metadata = checkpoint.get("metadata", {})
        architecture = "shared_candidate_q_v1"
        if not config.V1_DQN_USE_GLOBAL_STATE or not config.V1_DQN_DOUBLE_DQN:
            architecture += (
                f":global={int(config.V1_DQN_USE_GLOBAL_STATE)}"
                f":double={int(config.V1_DQN_DOUBLE_DQN)}"
            )
        validate_checkpoint_metadata(
            metadata,
            runtime.candidate_feature_encoder.feature_schema_hash,
            architecture,
        )
        runtime.candidate_q_network.load_state_dict(
            checkpoint["model_state_dict"]
        )
        runtime.scheduler.policy.epsilon = 0.0
        model_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    trace = _generate_trace(runtime, cutoff, seed)
    graph_data = nx.node_link_data(runtime.infrastructure.topo_manager.graph)
    config_view = {
        name: getattr(config, name)
        for name in dir(config)
        if name.isupper()
        and isinstance(
            getattr(config, name),
            (str, int, float, bool, tuple, list, dict, type(None)),
        )
    }
    config_view["V1_SOFT_TARDINESS_WEIGHT"] = soft_weight
    config_view["V1_FLEXIBLE_TARDINESS_WEIGHT"] = flexible_weight
    package_root = Path(__file__).resolve().parent
    code_files = (
        package_root / "v1_runtime.py",
        package_root / "evaluate_v1.py",
        package_root / "scheduler" / "v1_scheduler.py",
        package_root / "scheduler" / "candidate_generator.py",
        package_root / "accounting" / "energy.py",
        package_root / "evaluation_v1" / "runner.py",
        package_root / "learning" / "candidate_dqn.py",
    )
    code_hash = hashlib.sha256(
        b"".join(path.read_bytes() for path in code_files)
    ).hexdigest()
    forecast_versions = {
        node: (
            runtime.accounting.tariff_by_node[node].version,
            runtime.accounting.green_by_node[node].version,
        )
        for node in runtime.infrastructure.compute_nodes
    }
    context = {
        "code_hash": code_hash,
        "model_hash": model_hash,
        "config_hash": _canonical_hash(config_view),
        "topology_hash": _canonical_hash(graph_data),
        "task_trace_hash": _canonical_hash([asdict(task) for task in trace]),
        "exogenous_trace_hash": _canonical_hash(forecast_versions),
        "dependency_lock_hash": _canonical_hash({
            "numpy": np.__version__,
            "torch": torch.__version__,
        }),
        "tariff_mode": config.V1_TARIFF_MODE,
        "gamma_per_second": config.V1_GAMMA_PER_SECOND,
    }
    runner = EvaluationRunner(
        runtime.scheduler,
        runtime.time_converter,
        safety_cap,
        metadata_context=context,
    )
    return runner.run_frozen_policy(
        trace,
        arrival_cutoff_sim=cutoff,
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Formal v1.0 frozen-policy evaluation"
    )
    parser.add_argument(
        "--policy",
        choices=(
            "earliest_feasible",
            "lowest_cost",
            "highest_green",
            "equal_weight",
            "candidate_dqn",
        ),
        default="earliest_feasible",
    )
    parser.add_argument(
        "--arrival-cutoff",
        type=float,
        default=config.TRAFFIC_DAY_DURATION_IN_SIM,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--safety-cap", type=int, default=1000000)
    parser.add_argument("--model-path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--candidate-chunk-size",
        type=int,
        default=config.V1_CANDIDATE_CHUNK_SIZE,
    )
    parser.add_argument("--output", default="artifacts/v1/evaluation/report.json")
    parser.add_argument("--soft-tardiness-weight", type=float)
    parser.add_argument("--flexible-tardiness-weight", type=float)
    parser.add_argument(
        "--ablation-variant", choices=variant_names()
    )
    args = parser.parse_args()
    report = run_evaluation(
        args.policy,
        args.arrival_cutoff,
        args.seed,
        args.safety_cap,
        args.model_path,
        device=args.device,
        candidate_chunk_size=args.candidate_chunk_size,
        soft_tardiness_weight=args.soft_tardiness_weight,
        flexible_tardiness_weight=args.flexible_tardiness_weight,
        ablation_variant=args.ablation_variant,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            _jsonable(report),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(
        {"status": report.status.value, "output": str(output)},
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
