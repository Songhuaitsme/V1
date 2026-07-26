import argparse
import os

import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

from shared import config
from legacy.dqn_agent import DQNAgent
from legacy.gnn_agent import GNNAgent
from legacy.network_env import (
    NetworkEnvironment,
    evaluate_schedule_candidates,
    estimate_wait_opportunity,
    estimate_wait_penalty_detail,
    get_max_retries_for_task,
)
from legacy.static_metrics import analyze_task_resource_ratio
from shared.task_manager import TaskManager
from legacy.visualizer import TrainingVisualizer


RESUME_TRAINING = False
CHECKPOINT_DIR = "artifacts/legacy/logs/DQN_CHECKPOINT_GREEN_STATE"
CHECKPOINT_SUFFIX = ""
LOG_INTERVAL = 100


def get_checkpoint_paths():
    suffix = globals().get("CHECKPOINT_SUFFIX", "")
    if config.USE_GNN_AGENT:
        name = "GNN"
        if suffix:
            name = f"{name}_{suffix}"
        checkpoint_dir = os.path.join(CHECKPOINT_DIR, name)
        latest_path = os.path.join(checkpoint_dir, "checkpoint_latest")
        final_path = os.path.join(checkpoint_dir, "final_model")
    else:
        name = "DQN"
        if suffix:
            name = f"{name}_{suffix}"
        checkpoint_dir = os.path.join(CHECKPOINT_DIR, name)
        latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.h5")
        final_path = os.path.join(checkpoint_dir, "final_model.h5")
    return checkpoint_dir, latest_path, final_path


def model_path_exists(path):
    return os.path.exists(path) or os.path.exists(f"{path}.weights.h5")


def build_state(env, task=None, wait_queue=None):
    if config.USE_GNN_AGENT:
        return env.get_graph_state(task, wait_queue=wait_queue)
    return env.get_global_state(task, wait_queue=wait_queue)


class LagrangeManager:
    def __init__(self):
        self.enabled = getattr(config, 'ENABLE_CONSTRAINED_RL', True)
        self.lr = getattr(config, 'LAGRANGE_LR', 0.05)
        self.max_lambda = getattr(config, 'LAGRANGE_MAX', 25.0)
        self.lambdas = {
            "sla_violation": getattr(config, 'INITIAL_LAMBDA_SLA', 1.0),
            "drop": getattr(config, 'INITIAL_LAMBDA_DROP', 1.0),
            "cost_over_budget": getattr(config, 'INITIAL_LAMBDA_COST', 0.5),
            "overload": getattr(config, 'INITIAL_LAMBDA_OVERLOAD', 0.5),
        }
        self.targets = {
            "sla_violation": getattr(config, 'CONSTRAINT_TARGET_SLA_VIOLATION', 0.02),
            "drop": getattr(config, 'CONSTRAINT_TARGET_DROP', 0.03),
            "cost_over_budget": getattr(config, 'CONSTRAINT_TARGET_COST_OVER_BUDGET', 0.05),
            "overload": getattr(config, 'CONSTRAINT_TARGET_OVERLOAD', 0.05),
        }

    def apply(self, reward, constraint_costs):
        if not self.enabled:
            return float(reward), 0.0
        penalty = sum(
            self.lambdas.get(key, 0.0) * float(constraint_costs.get(key, 0.0))
            for key in self.lambdas
        )
        return float(reward) - penalty, penalty

    def update(self, averages):
        if not self.enabled:
            return
        for key, target in self.targets.items():
            current = float(averages.get(key, 0.0))
            self.lambdas[key] = float(np.clip(
                self.lambdas[key] + self.lr * (current - target),
                0.0,
                self.max_lambda,
            ))


def build_wait_decision_detail(env, task, wait_queue, global_time, compute_actions, candidates=None):
    queue_delay = max(0.0, global_time - task['generated_time'])
    next_queue_delay = queue_delay + config.SCHEDULING_CYCLE
    sla_type = task.get("sla_type", "Soft")
    threshold = float(getattr(config, "WAIT_GAIN_THRESHOLD", 0.0))
    penalty_weight = float(getattr(config, "WAIT_GAIN_PENALTY_WEIGHT", 1.0))
    max_retries = get_max_retries_for_task(task)
    detail = {
        "wait_allowed": False,
        "wait_reason": None,
        "wait_blocked_reason": None,
        "wait_gain": None,
        "wait_gain_threshold": threshold,
        "wait_net_gain": None,
        "max_retries": max_retries,
    }

    if task.get('retry_count', 0) >= max_retries:
        detail["wait_blocked_reason"] = "max_retries_exceeded"
        return detail

    if next_queue_delay >= task['latency_limit']:
        detail["wait_blocked_reason"] = "sla_exceeded_next_cycle"
        return detail

    penalty_detail = estimate_wait_penalty_detail(
        env,
        task,
        wait_queue,
        next_queue_delay,
        sla_type,
    )
    detail.update({
        "wait_penalty": penalty_detail.get("wait_penalty"),
        "urgency": penalty_detail.get("urgency"),
        "retry_ratio": penalty_detail.get("retry_ratio"),
        "queue_ratio": penalty_detail.get("queue_ratio"),
    })

    opportunity = estimate_wait_opportunity(env, task, wait_queue, global_time, candidates=candidates)
    wait_gain = opportunity.get("wait_gain")
    wait_net_gain = None if wait_gain is None else (
        wait_gain + penalty_weight * float(penalty_detail.get("wait_penalty", 0.0)) - threshold
    )
    detail.update({
        "wait_gain": wait_gain,
        "immediate_best_score": opportunity.get("immediate_best_score"),
        "future_best_score": opportunity.get("future_best_score"),
        "no_immediate_action": opportunity.get("no_immediate_action", False),
        "wait_net_gain": wait_net_gain,
    })

    if not compute_actions:
        detail["wait_allowed"] = True
        detail["wait_reason"] = "no_feasible_compute_action"
        return detail

    if wait_net_gain is not None and wait_net_gain > 0.0:
        detail["wait_allowed"] = True
        detail["wait_reason"] = "positive_wait_net_gain"
    else:
        detail["wait_blocked_reason"] = "wait_gain_below_penalty_adjusted_threshold"
    return detail


def compute_valid_actions(env, task, wait_queue, global_time, return_wait_detail=False, candidates=None):
    if candidates is None:
        candidates = evaluate_schedule_candidates(env, task, wait_queue, global_time)
    valid_actions = sorted({
        candidate["action_index"]
        for candidate in candidates
        if abs(candidate["schedule_time"] - global_time) <= 1e-9
    })

    wait_detail = build_wait_decision_detail(
        env,
        task,
        wait_queue,
        global_time,
        compute_actions=valid_actions,
        candidates=candidates,
    )
    if wait_detail["wait_allowed"]:
        valid_actions.append(len(env.compute_nodes))

    if return_wait_detail:
        return valid_actions, wait_detail
    return valid_actions


def update_active_tasks(env, active_tasks, global_time):
    """
    Update reserved/running task lifecycle:
    1. tasks whose start_time has arrived begin occupying CPU and bandwidth;
    2. tasks whose finish_time has arrived release CPU and bandwidth;
    3. finished tasks are removed from active_tasks and scheduled_allocations.
    """
    for i in range(len(active_tasks) - 1, -1, -1):
        active = active_tasks[i]

        if not active.get("started", False) and active["start_time"] <= global_time:
            env.node_resources[active["node"]]["used"] += active["cpu"]
            env.allocate_path_bandwidth(active.get("path"), active.get("bw", 0.0))
            active["started"] = True

        if active.get("started", False) and active["finish_time"] <= global_time:
            env.node_resources[active["node"]]["used"] = max(
                0.0,
                env.node_resources[active["node"]]["used"] - active["cpu"],
            )
            env.release_path_bandwidth(active.get("path"), active.get("bw", 0.0))
            env.remove_scheduled_allocation(active)
            active_tasks.pop(i)


def select_warmup_action(env, valid_actions):
    """
    Basic warm-up policy:
    1. randomly choose a feasible compute-node action if any exists;
    2. choose WAIT only when no compute-node action is feasible;
    3. return None when valid_actions is empty.
    """
    if not valid_actions:
        return None

    wait_action = len(env.compute_nodes)
    compute_actions = [
        action for action in valid_actions
        if action != wait_action
    ]

    if compute_actions:
        return int(np.random.choice(compute_actions))

    if wait_action in valid_actions:
        return wait_action

    return None


def add_success_allocation(env, active_tasks, task, info, global_time):
    """
    Write a successfully scheduled task into active_tasks and scheduled_allocations
    so its CPU/bandwidth occupation can start and release naturally.
    """
    target = info["target_node"]
    allocation = {
        "node": target,
        "cpu": info.get("cpu_supply", task["cpu"]),
        "bw": task["bw"],
        "path": info["path"],
        "start_time": info["execute_time"],
        "finish_time": info["finish_time"],
        "started": False,
    }

    if allocation["start_time"] <= global_time:
        env.node_resources[target]["used"] += allocation["cpu"]
        env.allocate_path_bandwidth(allocation.get("path"), allocation.get("bw", 0.0))
        allocation["started"] = True

    active_tasks.append(allocation)
    env.add_scheduled_allocation(allocation)
    return allocation


def warmup_environment(env, task_manager, total_compute_capacity, global_time):
    """
    Basic environment warm-up:
    - generate tasks normally;
    - maintain wait_queue normally;
    - compute valid actions normally;
    - select actions with random_valid_compute;
    - write successful tasks into active_tasks/scheduled_allocations;
    - do not train agent, record formal metrics, or fill replay buffer.
    """
    active_tasks = []
    wait_queue = []
    warmup_cycles = int(getattr(config, "ENV_WARMUP_CYCLES", 0))

    print(f"=== Environment warm-up start: {warmup_cycles} cycles ===")

    for cycle in range(warmup_cycles):
        global_time += config.SCHEDULING_CYCLE
        update_active_tasks(env, active_tasks, global_time)

        lam, _ = task_manager.get_dynamic_task_rate(global_time)
        cycle_cpu_time_supply = total_compute_capacity * config.SCHEDULING_CYCLE
        peak_cpu_budget = cycle_cpu_time_supply * getattr(
            config,
            "TASK_PEAK_LOAD_MULTIPLIER",
            1.3,
        )
        raw_new_tasks = task_manager.generate_tasks(
            np.random.poisson(lam),
            global_time,
            cycle,
            cpu_budget=peak_cpu_budget,
        )

        for task in raw_new_tasks:
            if len(wait_queue) < config.MAX_QUEUE_LENGTH:
                wait_queue.append(task)

        wait_queue.sort(
            key=lambda t: task_manager.calculate_priority(t, global_time),
            reverse=True,
        )

        deferred_batch = []
        for _ in range(min(len(wait_queue), config.MAX_TASKS_PER_CYCLE)):
            task = wait_queue.pop(0)
            task["current_time_context"] = global_time
            candidates = evaluate_schedule_candidates(env, task, wait_queue, global_time)
            valid_actions = compute_valid_actions(env, task, wait_queue, global_time, candidates=candidates)
            if not valid_actions:
                continue

            action = select_warmup_action(env, valid_actions)
            if action is None:
                continue

            _, _, _, info = env.step(action, task, wait_queue, candidates=candidates)
            if info.get("status") == "Success":
                add_success_allocation(
                    env=env,
                    active_tasks=active_tasks,
                    task=task,
                    info=info,
                    global_time=global_time,
                )
            elif info.get("status") == "Deferred":
                requeued_task = info.get("deferred_task", task)
                requeued_task["retry_count"] = requeued_task.get("retry_count", 0) + 1
                if requeued_task["retry_count"] <= get_max_retries_for_task(requeued_task):
                    deferred_batch.append(requeued_task)

        wait_queue.extend(deferred_batch)

    cpu_used = sum(float(res.get("used", 0.0)) for res in env.node_resources.values())
    cpu_total = sum(float(res.get("total", 0.0)) for res in env.node_resources.values())
    cpu_usage = cpu_used / max(cpu_total, 1e-8)

    print(
        "=== Environment warm-up finished | "
        f"global_time={global_time:.3f} | "
        f"active_tasks={len(active_tasks)} | "
        f"wait_queue={len(wait_queue)} | "
        f"scheduled_allocations={len(env.scheduled_allocations)} | "
        f"cpu_usage={cpu_usage:.4f} ==="
    )

    return global_time, active_tasks, wait_queue


def build_transition_next_state(env, remaining_queue, requeued_task, global_time):
    pending_queue = list(remaining_queue)
    if requeued_task is not None:
        pending_queue.append(requeued_task)

    if not pending_queue:
        return build_state(env, wait_queue=[]), True, []

    next_task = remaining_queue[0] if remaining_queue else requeued_task
    next_time = global_time if remaining_queue else global_time + config.SCHEDULING_CYCLE
    next_task_view = dict(next_task)
    next_task_view['current_time_context'] = next_time
    candidates = evaluate_schedule_candidates(env, next_task_view, pending_queue, next_time)
    next_valid_actions = compute_valid_actions(
        env,
        next_task_view,
        pending_queue,
        next_time,
        candidates=candidates,
    )
    transition_done = len(next_valid_actions) == 0
    return build_state(env, next_task_view, wait_queue=pending_queue), transition_done, next_valid_actions


def make_metric_buffers():
    return {
        "rewards": [],
        "base_rewards": [],
        "losses": [],
        "physical_latencies": [],
        "e2e_latencies": [],
        "costs": [],
        "cost_per_cpu_times": [],
        "cost_ratios": [],
        "baseline_costs": [],
        "cpu_time_demands": [],
        "green_matches": [],
        "green_absorptions": [],
        "cost_savings": [],
        "task_green_used_mws": [],
        "task_green_supply_mws": [],
        "task_green_unused_mws": [],
        "task_power_demand_mws": [],
        "target_regions": [],
        "target_tiers": [],
        "target_nodes": [],
        "action_system_absorption_deltas": [],
        "action_system_absorption_delta_clipped": [],
        "green_unused_ratios_after": [],
        "green_load_coverages_after": [],
        "green_reward_match": [],
        "green_reward_absorption_delta": [],
        "green_reward_waste": [],
        "green_reward_load_coverage": [],
        "green_reward_totals": [],
        "wait_penalties": [],
        "wait_urgencies": [],
        "wait_retry_ratios": [],
        "wait_queue_ratios": [],
        "queue_wait_times": [],
        "source_region_queue_ratios": [],
        "same_sla_queue_ratios": [],
        "wait_gains": [],
        "wait_net_gains": [],
        "immediate_best_scores": [],
        "future_best_scores": [],
        "wait_gain_positive_flags": [],
        "wait_allowed_by_gain_count": [],
        "wait_blocked_by_gain_count": [],
        "wait_no_feasible_compute_count": [],
        "constraint_penalties": [],
        "reward_components": {
            "R_latency": [],
            "R_cost": [],
            "R_green": [],
            "R_balance": [],
            "R_success": [],
            "R_cost_spike": [],
        },
        "constraint_costs": {
            "sla_violation": [],
            "drop": [],
            "cost_over_budget": [],
            "overload": [],
        },
        "generated_tasks": [],
    }


def mean_or_zero(values):
    return float(np.mean(values)) if values else 0.0


def percentile_or_zero(values, q):
    return float(np.percentile(values, q)) if values else 0.0


def max_or_zero(values):
    return float(np.max(values)) if values else 0.0


def min_or_zero(values):
    return float(np.min(values)) if values else 0.0


def safe_div(numerator, denominator, default=0.0):
    denominator = float(denominator)
    if abs(denominator) <= 1e-8:
        return float(default)
    return float(numerator) / denominator


def train(lightweight=False, max_steps=None, tensorboard_log_dir='artifacts/legacy/logs/tensorboard'):
    original_batch_size = config.BATCH_SIZE
    if lightweight:
        config.BATCH_SIZE = min(config.BATCH_SIZE, 32)

    env = NetworkEnvironment()
    if config.USE_GNN_AGENT:
        agent = GNNAgent(
            graph_state_template=env.get_graph_state(wait_queue=[]),
            action_dim=env.action_space_dim,
        )
        print("=== Using GNN Dueling Double DQN scheduler ===")
    else:
        agent = DQNAgent(state_dim=env.state_space_dim, action_dim=env.action_space_dim)
        print("=== Using Dueling Double DQN + PER + Action Mask scheduler ===")
    print(f"=== State dim: {env.state_space_dim}, Action dim: {env.action_space_dim} ===")
    if not config.USE_GNN_AGENT:
        print(f"=== DQN model input shape: {agent.model.input_shape} ===")

    total_compute_capacity = sum(res['total'] for res in env.node_resources.values())
    task_manager = TaskManager(env.base_stations, total_compute_capacity=total_compute_capacity)
    viz = TrainingVisualizer(log_dir=tensorboard_log_dir)
    if getattr(viz, 'writer', None) is not None:
        print(f"=== TensorBoard log dir: {viz.train_log_dir} ===")
        print(f"=== View with: tensorboard --logdir {tensorboard_log_dir} ===")
    else:
        print("=== TensorBoard writer unavailable; metrics will only be kept in memory. ===")
    checkpoint_dir, checkpoint_path, final_model_path = get_checkpoint_paths()
    print(f"=== Checkpoint dir: {checkpoint_dir} ===")
    if lightweight:
        checkpoint_dir = os.path.join(checkpoint_dir, "lightweight")
        checkpoint_path = os.path.join(checkpoint_dir, os.path.basename(checkpoint_path))
        final_model_path = os.path.join(checkpoint_dir, os.path.basename(final_model_path))

    resume_path = checkpoint_path if model_path_exists(checkpoint_path) else final_model_path
    if RESUME_TRAINING and model_path_exists(resume_path):
        print(f"=== Resuming training from: {resume_path} ===")
        agent.load(resume_path)
        agent.epsilon = 0.1
    else:
        print("=== Starting training ===")

    lagrange = LagrangeManager()
    total_steps = max_steps if max_steps is not None else (800 if lightweight else config.MAX_STEPS)
    log_interval = 100 if lightweight else LOG_INTERVAL

    active_tasks = []
    wait_queue = []

    if getattr(config, "RANDOMIZE_INITIAL_GLOBAL_TIME", False):
        global_time = float(np.random.uniform(0.0, config.TRAFFIC_DAY_DURATION_IN_SIM))
    else:
        global_time = 0.0

    if getattr(config, "ENABLE_ENV_WARMUP", False):
        global_time, active_tasks, wait_queue = warmup_environment(
            env=env,
            task_manager=task_manager,
            total_compute_capacity=total_compute_capacity,
            global_time=global_time,
        )

    buffers = make_metric_buffers()
    tier_counts = {1: 0, 2: 0, 3: 0}
    processed = succeeded = dropped = deferred = 0
    generated = 0
    generated_cpu = 0.0
    global_training_step = 0
    last_logged_system_green_absorption = None

    try:
        for cycle in range(total_steps):
            global_time += config.SCHEDULING_CYCLE

            update_active_tasks(env, active_tasks, global_time)

            lam, sim_hr = task_manager.get_dynamic_task_rate(global_time)
            cycle_cpu_time_supply = total_compute_capacity * config.SCHEDULING_CYCLE
            peak_cpu_budget = cycle_cpu_time_supply * getattr(config, 'TASK_PEAK_LOAD_MULTIPLIER', 1.3)
            raw_new_tasks = task_manager.generate_tasks(
                np.random.poisson(lam),
                global_time,
                cycle,
                cpu_budget=peak_cpu_budget,
            )
            buffers["generated_tasks"].extend(raw_new_tasks)
            arrival_cpu = sum(
                env.get_task_cpu_time_demand(task)
                for task in raw_new_tasks
            )
            generated += len(raw_new_tasks)
            generated_cpu += arrival_cpu
            cycle_cpu_time_supply = max(1.0, cycle_cpu_time_supply)
            for task in raw_new_tasks:
                if len(wait_queue) < config.MAX_QUEUE_LENGTH:
                    wait_queue.append(task)
                else:
                    dropped += 1
                    buffers["constraint_costs"]["drop"].append(1.0)

            wait_queue.sort(key=lambda t: task_manager.calculate_priority(t, global_time), reverse=True)
            deferred_batch = []

            for _ in range(min(len(wait_queue), config.MAX_TASKS_PER_CYCLE)):
                task = wait_queue.pop(0)
                processed += 1
                task['current_time_context'] = global_time
                candidates = evaluate_schedule_candidates(env, task, wait_queue, global_time)

                valid_actions, wait_decision_detail = compute_valid_actions(
                    env,
                    task,
                    wait_queue,
                    global_time,
                    return_wait_detail=True,
                    candidates=candidates,
                )
                if wait_decision_detail.get("wait_reason") == "positive_wait_net_gain":
                    buffers["wait_allowed_by_gain_count"].append(1.0)
                elif wait_decision_detail.get("wait_reason") == "no_feasible_compute_action":
                    buffers["wait_no_feasible_compute_count"].append(1.0)
                elif wait_decision_detail.get("wait_blocked_reason") == "wait_gain_below_penalty_adjusted_threshold":
                    buffers["wait_blocked_by_gain_count"].append(1.0)

                if not valid_actions:
                    dropped += 1
                    buffers["constraint_costs"]["drop"].append(1.0)
                    continue

                state = build_state(env, task, wait_queue=wait_queue)
                action = agent.act(state, valid_actions=valid_actions)
                _, base_reward, done, info = env.step(action, task, wait_queue, candidates=candidates)
                constraint_costs = info.get('constraint_costs', {})
                adjusted_reward, constraint_penalty = lagrange.apply(base_reward, constraint_costs)

                requeued_task = None
                if info['status'] == 'Success':
                    succeeded += 1
                    buffers["physical_latencies"].append(info['delays']['physical'])
                    buffers["e2e_latencies"].append(info['delays']['end_to_end'])
                    buffers["costs"].append(info.get('cost', 0.0))
                    coordination = info.get('coordination', {})
                    buffers["green_matches"].append(coordination.get('green_match_ratio', 0.0))
                    buffers["green_absorptions"].append(coordination.get('green_absorption_ratio', 0.0))
                    buffers["cost_savings"].append(coordination.get('cost_saving_ratio', 0.0))

                    cpu_time_demand = float(info.get(
                        'cpu_time_demand',
                        env.get_task_cpu_time_demand(task),
                    ))
                    raw_cost = float(info.get('cost', 0.0))
                    buffers["cpu_time_demands"].append(cpu_time_demand)
                    buffers["cost_per_cpu_times"].append(
                        coordination.get(
                            'cost_per_cpu_time',
                            raw_cost / max(cpu_time_demand, 1e-8),
                        )
                    )
                    buffers["cost_ratios"].append(coordination.get('cost_ratio', 0.0))
                    buffers["baseline_costs"].append(coordination.get('baseline_cost', 0.0))
                    buffers["task_green_used_mws"].append(coordination.get('green_used_mw', 0.0))
                    buffers["task_green_supply_mws"].append(coordination.get('green_supply_mw', 0.0))
                    buffers["task_green_unused_mws"].append(coordination.get('green_unused_mw', 0.0))
                    buffers["task_power_demand_mws"].append(coordination.get('power_demand_mw', 0.0))
                    buffers["action_system_absorption_deltas"].append(
                        coordination.get('system_absorption_delta', 0.0)
                    )
                    buffers["action_system_absorption_delta_clipped"].append(
                        coordination.get('system_absorption_delta_clipped', 0.0)
                    )
                    buffers["green_unused_ratios_after"].append(
                        coordination.get('green_unused_ratio_after', 0.0)
                    )
                    buffers["green_load_coverages_after"].append(
                        coordination.get('green_load_coverage_after', 0.0)
                    )
                    buffers["green_reward_match"].append(coordination.get('R_green_match', 0.0))
                    buffers["green_reward_absorption_delta"].append(
                        coordination.get('R_green_absorption_delta', 0.0)
                    )
                    buffers["green_reward_waste"].append(coordination.get('R_green_waste', 0.0))
                    buffers["green_reward_load_coverage"].append(
                        coordination.get('R_green_load_coverage', 0.0)
                    )
                    buffers["green_reward_totals"].append(coordination.get('R_green_total', 0.0))

                    target = info['target_node']
                    target_region = env.topo_manager.graph.nodes[target].get('region', 'Unknown')
                    target_tier = env.topo_manager.graph.nodes[target].get('tier', 2)
                    tier_counts[target_tier] += 1
                    buffers["target_nodes"].append(target)
                    buffers["target_regions"].append(target_region)
                    buffers["target_tiers"].append(target_tier)
                    add_success_allocation(
                        env=env,
                        active_tasks=active_tasks,
                        task=task,
                        info=info,
                        global_time=global_time,
                    )

                elif info['status'] == 'Deferred':
                    #requeued_task = info.get('deferred_task', task)
                    #deferred_batch.append(requeued_task)
                    #deferred += 1
                    wait_detail = info.get("wait_detail", {})
                    buffers["wait_penalties"].append(wait_detail.get("wait_penalty", 0.0))
                    buffers["wait_urgencies"].append(wait_detail.get("urgency", 0.0))
                    buffers["wait_retry_ratios"].append(wait_detail.get("retry_ratio", 0.0))
                    buffers["wait_queue_ratios"].append(wait_detail.get("queue_ratio", 0.0))
                    buffers["queue_wait_times"].append(wait_detail.get("queue_wait_time", 0.0))
                    buffers["source_region_queue_ratios"].append(
                        wait_detail.get("source_region_queue_ratio", 0.0)
                    )
                    buffers["same_sla_queue_ratios"].append(wait_detail.get("same_sla_queue_ratio", 0.0))
                    if wait_detail.get("wait_gain") is not None:
                        buffers["wait_gains"].append(wait_detail["wait_gain"])
                    if wait_detail.get("wait_net_gain") is not None:
                        buffers["wait_net_gains"].append(wait_detail["wait_net_gain"])
                    if wait_detail.get("immediate_best_score") is not None:
                        buffers["immediate_best_scores"].append(wait_detail["immediate_best_score"])
                    if wait_detail.get("future_best_score") is not None:
                        buffers["future_best_scores"].append(wait_detail["future_best_score"])
                    buffers["wait_gain_positive_flags"].append(
                        1.0 if wait_detail.get("wait_gain_positive", False) else 0.0
                    )

                    requeued_task = info.get('deferred_task', task)
                    requeued_task['retry_count'] = requeued_task.get('retry_count', 0) + 1

                    if requeued_task['retry_count'] <= get_max_retries_for_task(requeued_task):
                        deferred_batch.append(requeued_task)
                        deferred += 1
                    else:
                        dropped += 1
                        buffers["constraint_costs"]["drop"].append(1.0)
                else:
                    retry_count = task.get('retry_count', 0)
                    can_retry = (
                        info['status'] != 'Failed_Wait'
                        and retry_count < get_max_retries_for_task(task)
                    )
                    if can_retry:
                        task['retry_count'] = retry_count + 1
                        deferred_batch.append(task)
                        requeued_task = task
                    else:
                        dropped += 1

                next_state, transition_done, next_valid_actions = build_transition_next_state(
                    env,
                    wait_queue,
                    requeued_task,
                    global_time,
                )
                agent.remember(
                    state,
                    action,
                    adjusted_reward,
                    next_state,
                    done or transition_done,
                    valid_actions=valid_actions,
                    next_valid_actions=next_valid_actions,
                    info=info,
                )
                
                loss = None
                if processed % getattr(config, 'TRAIN_EVERY', 1) == 0:
                    loss = agent.replay()

                buffers["rewards"].append(adjusted_reward)
                buffers["base_rewards"].append(base_reward)
                buffers["constraint_penalties"].append(constraint_penalty)
                for key, value in info.get('reward_components', {}).items():
                    buffers["reward_components"].setdefault(key, []).append(float(value))
                for key, value in constraint_costs.items():
                    buffers["constraint_costs"].setdefault(key, []).append(float(value))

                if loss is not None:
                    buffers["losses"].append(loss)
                    global_training_step += 1
                    if global_training_step % 1000 == 0:
                        agent.update_target_model()

            wait_queue.extend(deferred_batch)

            if cycle % log_interval == 0 and cycle > 0:
                avg_constraints = {
                    key: mean_or_zero(values)
                    for key, values in buffers["constraint_costs"].items()
                }
                lagrange.update(avg_constraints)

                cpu_usages = np.array(
                    [res['used'] / res['total'] for res in env.node_resources.values() if res['total'] > 0]
                )
                global_cpu = float(np.mean(cpu_usages)) if len(cpu_usages) > 0 else 0.0
                cpu_mean = max(global_cpu, 1e-8)
                load_cv = float(np.std(cpu_usages) / cpu_mean) if len(cpu_usages) > 0 else 0.0
                load_balance_score = max(0.0, 1.0 - min(1.0, load_cv))

                dispatch_rate = succeeded / processed if processed > 0 else 0.0
                defer_rate = deferred / processed if processed > 0 else 0.0
                drop_rate = dropped / processed if processed > 0 else 0.0
                completion_rate = succeeded / max(1, succeeded + dropped)
                throughput_rate = succeeded / max(1, generated)
                avg_green_match = mean_or_zero(buffers["green_matches"])
                avg_green_absorption = mean_or_zero(buffers["green_absorptions"])
                avg_cost_saving = mean_or_zero(buffers["cost_savings"])
                system_green = env.get_system_green_absorption(global_time)
                system_green_absorption = system_green["system_green_absorption_ratio"]
                system_green_unused = system_green.get("total_green_unused_mw", 0.0)
                green_unused_ratio = system_green.get("green_unused_ratio", 0.0)
                green_load_coverage_ratio = system_green.get("green_load_coverage_ratio", 0.0)
                green_supply_demand_ratio = system_green.get("green_supply_demand_ratio", 0.0)
                if last_logged_system_green_absorption is None:
                    system_green_absorption_delta = 0.0
                else:
                    system_green_absorption_delta = (
                        system_green_absorption - last_logged_system_green_absorption
                    )
                last_logged_system_green_absorption = system_green_absorption
                # Strict alternative: harmonic mean penalizes one-sided green performance.
                # green_eps = 1e-8
                # green_coordination_score = (
                #     2.0 * avg_green_match * system_green_absorption
                #     / (avg_green_match + system_green_absorption + green_eps)
                # )

                green_coordination_score = (
                    config.CECI_GREEN_MATCH_WEIGHT * avg_green_match
                    + config.CECI_GREEN_ABSORPTION_WEIGHT * system_green_absorption
                )
                coordination_score = (
                    config.CECI_GREEN_WEIGHT * green_coordination_score
                    + config.CECI_COST_WEIGHT * avg_cost_saving
                    + config.CECI_BALANCE_WEIGHT * load_balance_score
                )
                ceci_raw = coordination_score
                ceci_effective = coordination_score * completion_rate * throughput_rate
                ceci = ceci_effective

                region_usage = env.get_region_cpu_usage()
                tier_usage = env.get_tier_cpu_usage()

                avg_cost = mean_or_zero(buffers["costs"])
                p50_cost = percentile_or_zero(buffers["costs"], 50)
                p95_cost = percentile_or_zero(buffers["costs"], 95)
                max_cost = max_or_zero(buffers["costs"])
                avg_cost_per_cpu_time = mean_or_zero(buffers["cost_per_cpu_times"])
                p50_cost_per_cpu_time = percentile_or_zero(buffers["cost_per_cpu_times"], 50)
                p95_cost_per_cpu_time = percentile_or_zero(buffers["cost_per_cpu_times"], 95)
                max_cost_per_cpu_time = max_or_zero(buffers["cost_per_cpu_times"])
                avg_cost_ratio = mean_or_zero(buffers["cost_ratios"])
                p95_cost_ratio = percentile_or_zero(buffers["cost_ratios"], 95)
                max_cost_ratio = max_or_zero(buffers["cost_ratios"])
                avg_cpu_time_demand_success = mean_or_zero(buffers["cpu_time_demands"])
                p95_cpu_time_demand_success = percentile_or_zero(buffers["cpu_time_demands"], 95)
                p95_e2e_latency = percentile_or_zero(buffers["e2e_latencies"], 95)
                avg_action_system_absorption_delta = mean_or_zero(
                    buffers["action_system_absorption_deltas"]
                )
                min_action_system_absorption_delta = min_or_zero(
                    buffers["action_system_absorption_deltas"]
                )
                max_action_system_absorption_delta = max_or_zero(
                    buffers["action_system_absorption_deltas"]
                )

                green_rich_regions = {"G", "H", "I", "J", "K", "L", "M"}
                low_green_regions = {"A", "B", "C", "D", "E", "F"}
                target_region_count = {}
                for region in buffers["target_regions"]:
                    target_region_count[region] = target_region_count.get(region, 0) + 1
                total_success_targets = max(1, len(buffers["target_regions"]))
                selected_green_rich_count = sum(
                    count for region, count in target_region_count.items()
                    if region in green_rich_regions
                )
                selected_low_green_count = sum(
                    count for region, count in target_region_count.items()
                    if region in low_green_regions
                )
                selected_green_rich_ratio = safe_div(
                    selected_green_rich_count, total_success_targets
                )
                selected_low_green_ratio = safe_div(
                    selected_low_green_count, total_success_targets
                )

                total_tier = sum(tier_counts.values()) or 1
                t1_pct = tier_counts[1] / total_tier
                t2_pct = tier_counts[2] / total_tier
                t3_pct = tier_counts[3] / total_tier
                static_task_metrics = analyze_task_resource_ratio(
                    buffers["generated_tasks"],
                    env,
                    log_interval * config.SCHEDULING_CYCLE,
                    peak_window=config.SCHEDULING_CYCLE,
                )

                metrics = {
                    'progress_ratio': cycle / max(1, total_steps),
                    'global_time': global_time,
                    'sim_hour': sim_hr,
                    'generated_tasks': generated,
                    'processed_tasks': processed,
                    'succeeded_tasks': succeeded,
                    'dropped_tasks': dropped,
                    'deferred_tasks': deferred,
                    'wait_queue_length': len(wait_queue),
                    'active_task_count': len(active_tasks),
                    'task_lambda': float(lam),
                    'task_load_target_utilization': float(
                        config.TASK_LOAD_TARGET_UTILIZATION
                    ),
                    'arrival_cpu': generated_cpu,
                    'task_count': static_task_metrics['task_count'],
                    'compute_node_count': static_task_metrics['compute_node_count'],
                    'tasks_per_node': static_task_metrics['tasks_per_node'],
                    'total_cpu_capacity': static_task_metrics['total_cpu_capacity'],
                    'total_task_cpu': static_task_metrics['total_task_cpu'],
                    'arrival_rate': static_task_metrics['arrival_rate'],
                    'avg_task_cpu': static_task_metrics['avg_task_cpu'],
                    'avg_task_duration': static_task_metrics['avg_task_duration'],
                    'total_cpu_time_demand': static_task_metrics['total_cpu_time_demand'],
                    'total_cpu_time_capacity': static_task_metrics['total_cpu_time_capacity'],
                    'simple_cpu_ratio': static_task_metrics['simple_cpu_ratio'],
                    'peak_cpu_demand': static_task_metrics['peak_cpu_demand'],
                    'avg_cpu_time_load_ratio': static_task_metrics['avg_cpu_time_load_ratio'],
                    'peak_instant_load_ratio': static_task_metrics['peak_instant_load_ratio'],
                    'peak_cpu_budget_ratio': (
                        peak_cpu_budget / max(1.0, total_compute_capacity * config.SCHEDULING_CYCLE)
                    ),
                    'training_updates': global_training_step,
                    'loss': mean_or_zero(buffers["losses"]),
                    'reward': mean_or_zero(buffers["rewards"]),
                    'base_reward': mean_or_zero(buffers["base_rewards"]),
                    'constraint_penalty': mean_or_zero(buffers["constraint_penalties"]),
                    'avg_wait_penalty': mean_or_zero(buffers["wait_penalties"]),
                    'avg_wait_urgency': mean_or_zero(buffers["wait_urgencies"]),
                    'avg_wait_retry_ratio': mean_or_zero(buffers["wait_retry_ratios"]),
                    'avg_wait_queue_ratio': mean_or_zero(buffers["wait_queue_ratios"]),
                    'avg_queue_wait_time': mean_or_zero(buffers["queue_wait_times"]),
                    'avg_source_region_queue_ratio': mean_or_zero(buffers["source_region_queue_ratios"]),
                    'avg_same_sla_queue_ratio': mean_or_zero(buffers["same_sla_queue_ratios"]),
                    'avg_wait_gain': mean_or_zero(buffers["wait_gains"]),
                    'avg_wait_net_gain': mean_or_zero(buffers["wait_net_gains"]),
                    'avg_immediate_best_score': mean_or_zero(buffers["immediate_best_scores"]),
                    'avg_future_best_score': mean_or_zero(buffers["future_best_scores"]),
                    'wait_gain_positive_ratio': mean_or_zero(buffers["wait_gain_positive_flags"]),
                    'wait_allowed_by_gain_count': float(sum(buffers["wait_allowed_by_gain_count"])),
                    'wait_blocked_by_gain_count': float(sum(buffers["wait_blocked_by_gain_count"])),
                    'wait_no_feasible_compute_count': float(sum(buffers["wait_no_feasible_compute_count"])),
                    'dispatch_rate': dispatch_rate,
                    'defer_rate': defer_rate,
                    'completion_rate': completion_rate,
                    'throughput_rate': throughput_rate,
                    'drop_rate': drop_rate,
                    'avg_physical_latency': mean_or_zero(buffers["physical_latencies"]),
                    'avg_end_to_end_latency': mean_or_zero(buffers["e2e_latencies"]),
                    'p95_end_to_end_latency': p95_e2e_latency,
                    'cpu_usage_mean': global_cpu,
                    'avg_cost': avg_cost,
                    'p50_cost': p50_cost,
                    'p95_cost': p95_cost,
                    'max_cost': max_cost,
                    'avg_cost_per_cpu_time': avg_cost_per_cpu_time,
                    'p50_cost_per_cpu_time': p50_cost_per_cpu_time,
                    'p95_cost_per_cpu_time': p95_cost_per_cpu_time,
                    'max_cost_per_cpu_time': max_cost_per_cpu_time,
                    'avg_cost_ratio': avg_cost_ratio,
                    'p95_cost_ratio': p95_cost_ratio,
                    'max_cost_ratio': max_cost_ratio,
                    'avg_cpu_time_demand_success': avg_cpu_time_demand_success,
                    'p95_cpu_time_demand_success': p95_cpu_time_demand_success,
                    'green_match_ratio': avg_green_match,
                    'green_absorption_ratio': avg_green_absorption,
                    'system_green_absorption_ratio': system_green_absorption,
                    'system_green_absorption_delta': system_green_absorption_delta,
                    'total_green_used_mw': system_green["total_green_used_mw"],
                    'total_green_supply_mw': system_green["total_green_supply_mw"],
                    'total_power_demand_mw': system_green["total_power_demand_mw"],
                    'total_green_unused_mw': system_green_unused,
                    'green_unused_ratio': green_unused_ratio,
                    'green_load_coverage_ratio': green_load_coverage_ratio,
                    'green_supply_demand_ratio': green_supply_demand_ratio,
                    'avg_action_system_absorption_delta': avg_action_system_absorption_delta,
                    'min_action_system_absorption_delta': min_action_system_absorption_delta,
                    'max_action_system_absorption_delta': max_action_system_absorption_delta,
                    'avg_green_unused_ratio_after': mean_or_zero(
                        buffers["green_unused_ratios_after"]
                    ),
                    'avg_green_load_coverage_after': mean_or_zero(
                        buffers["green_load_coverages_after"]
                    ),
                    'R_green_match': mean_or_zero(buffers["green_reward_match"]),
                    'R_green_absorption_delta': mean_or_zero(
                        buffers["green_reward_absorption_delta"]
                    ),
                    'R_green_waste': mean_or_zero(buffers["green_reward_waste"]),
                    'R_green_load_coverage': mean_or_zero(
                        buffers["green_reward_load_coverage"]
                    ),
                    'R_green_total': mean_or_zero(buffers["green_reward_totals"]),
                    'green_coordination_score': green_coordination_score,
                    'cost_saving_ratio': avg_cost_saving,
                    'load_balance_score': load_balance_score,
                    'coordination_score': coordination_score,
                    'ceci_raw': ceci_raw,
                    'ceci_effective': ceci_effective,
                    'ceci': ceci,
                    'selected_green_rich_ratio': selected_green_rich_ratio,
                    'selected_low_green_ratio': selected_low_green_ratio,
                    'epsilon': agent.epsilon,
                }
                for region, usage in sorted(region_usage.items()):
                    metrics[f"region_cpu_{region}"] = usage
                for tier, usage in sorted(tier_usage.items()):
                    metrics[f"tier_cpu_{tier}"] = usage
                for region, count in sorted(target_region_count.items()):
                    metrics[f"selected_region_ratio_{region}"] = safe_div(
                        count, total_success_targets
                    )
                for key, values in buffers["reward_components"].items():
                    metrics[key] = mean_or_zero(values)
                for key, value in avg_constraints.items():
                    metrics[f'constraint_{key}'] = value
                for key, value in lagrange.lambdas.items():
                    metrics[f'lambda_{key}'] = value

                viz.log_step(cycle, metrics, tier_counts=tier_counts, region_usage=region_usage)
                reg_str = " ".join([f"{k}:{v:.0%}" for k, v in list(region_usage.items())[:5]])
                print(
                    f"Cycle {cycle:05d} | T {sim_hr:05.2f}H | Eps {agent.epsilon:.3f} | "
                    f"Rwd {metrics['reward']:>6.2f} | Loss {metrics['loss']:.4f} | "
                    f"Dispatch {dispatch_rate:>6.1%} | Defer {defer_rate:>6.1%} | "
                    f"Done {completion_rate:>6.1%} | Throughput {throughput_rate:>6.1%} | "
                    f"Drop {drop_rate:>5.1%} | "
                    f"Load {metrics['avg_cpu_time_load_ratio']:>6.1%}/{metrics['peak_instant_load_ratio']:>6.1%} | "
                    f"CECI_eff {ceci_effective:.3f} Raw {ceci_raw:.3f} | "
                    f"SysAbs {system_green_absorption:.2%} Waste {green_unused_ratio:.2%} | "
                    f"CostCPU {avg_cost_per_cpu_time:.4f} P95 {p95_cost_per_cpu_time:.4f} | "
                    f"Rgreen {metrics['R_green']:.2f} dAbs {avg_action_system_absorption_delta:+.4f} | "
                    f"Rspike {metrics['R_cost_spike']:.2f} | "
                    f"GreenRich {selected_green_rich_ratio:.1%} | "
                    f"Pen {metrics['constraint_penalty']:.2f} | "
                    f"L_sla {lagrange.lambdas['sla_violation']:.2f} | "
                    f"T1/2/3 {t1_pct:.0%}/{t2_pct:.0%}/{t3_pct:.0%} | "
                    f"CPU {global_cpu:>5.1%} ({reg_str})"
                )

                buffers = make_metric_buffers()
                tier_counts = {1: 0, 2: 0, 3: 0}
                processed = succeeded = dropped = deferred = generated = 0
                generated_cpu = 0.0

                if not lightweight and cycle % getattr(config, 'CHECKPOINT_INTERVAL', 1000) == 0:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_name = f"checkpoint_{cycle}" if config.USE_GNN_AGENT else f"checkpoint_{cycle}.h5"
                    agent.save(os.path.join(checkpoint_dir, checkpoint_name))
                    agent.save(checkpoint_path)

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
    except Exception:
        import traceback
        print("\n[ERROR] Training failed.")
        traceback.print_exc()
    finally:
        report_path = (
            'artifacts/legacy/training_report_lightweight.png'
            if lightweight else 'artifacts/legacy/training_report_new_pricing.png'
        )
        if len(viz.history['step']) > 0:
            viz.generate_final_report(save_path=report_path)
            csv_path = (
                'artifacts/legacy/training_metrics_lightweight.csv'
                if lightweight else 'artifacts/legacy/training_metrics_new_pricing.csv'
            )
            viz.export_history_csv(save_path=csv_path)
            if lightweight:
                print(f"=== Lightweight report saved: {report_path}; metrics: {csv_path} ===")
            else:
                os.makedirs(checkpoint_dir, exist_ok=True)
                agent.save(final_model_path)
                agent.save(checkpoint_path)
                print(f"=== Training finished; report, metrics ({csv_path}), and model saved. ===")
        config.BATCH_SIZE = original_batch_size


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lightweight', action='store_true', help='run a short, non-overwriting training pass')
    parser.add_argument('--max-steps', type=int, default=None, help='override training cycles')
    parser.add_argument(
        '--tensorboard-log-dir',
        default='artifacts/legacy/logs/tensorboard',
        help='directory for TensorBoard event files',
    )
    parser.add_argument(
        '--load-utilization',
        type=float,
        default=None,
        help='override config.TASK_LOAD_TARGET_UTILIZATION for curriculum training',
    )
    parser.add_argument(
        '--checkpoint-suffix',
        type=str,
        default='',
        help='optional suffix for checkpoint directory, used by curriculum stages',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume training from the latest checkpoint',
    )
    return parser.parse_args()


if __name__ == "__main__":
    if getattr(config, "SCHEDULER_ENGINE", "legacy") == "v1":
        from train_v1 import main as main_v1
        main_v1()
        raise SystemExit(0)
    args = parse_args()
    globals()["RESUME_TRAINING"] = bool(args.resume)
    globals()["CHECKPOINT_SUFFIX"] = args.checkpoint_suffix or ""
    if args.load_utilization is not None:
        config.TASK_LOAD_TARGET_UTILIZATION = float(args.load_utilization)
        print(
            "=== Override TASK_LOAD_TARGET_UTILIZATION: "
            f"{config.TASK_LOAD_TARGET_UTILIZATION} ==="
        )
    train(
        lightweight=args.lightweight,
        max_steps=args.max_steps,
        tensorboard_log_dir=args.tensorboard_log_dir,
    )
