import argparse
import json
import random

import numpy as np

from shared import config
from legacy.network_env import NetworkEnvironment
from shared.task_manager import TaskManager


def task_cpu_time(task):
    """CPU-time work demand of one task."""
    if not task:
        return 0.0
    cpu = max(0.0, float(task.get('cpu', 0.0)))
    duration = max(0.0, float(task.get('duration', 0.0)))
    if cpu > 0.0 and duration > 0.0:
        return cpu * duration
    return max(0.0, float(task.get('cpu_time', 0.0)))


def analyze_task_resource_ratio(
    tasks,
    env,
    simulation_duration,
    peak_window=None,
):
    tasks = list(tasks or [])
    task_count = len(tasks)
    compute_node_count = len(getattr(env, 'compute_nodes', []) or [])
    total_cpu_capacity = sum(
        float(res.get('total', 0.0))
        for res in getattr(env, 'node_resources', {}).values()
    )
    simulation_duration = max(float(simulation_duration or 0.0), 0.0)
    peak_window = float(peak_window or getattr(config, 'SCHEDULING_CYCLE', 1.0))
    peak_window = max(peak_window, 1e-8)

    cpu_values = [max(0.0, float(task.get('cpu', 0.0))) for task in tasks]
    duration_values = [max(0.0, float(task.get('duration', 0.0))) for task in tasks]
    cpu_time_values = [task_cpu_time(task) for task in tasks]

    total_task_cpu = float(np.sum(cpu_values)) if cpu_values else 0.0
    total_cpu_time_demand = float(np.sum(cpu_time_values)) if cpu_time_values else 0.0
    total_cpu_time_capacity = total_cpu_capacity * simulation_duration

    peak_cpu_by_window = {}
    for task, cpu in zip(tasks, cpu_values):
        generated_time = float(task.get('generated_time', 0.0))
        bucket = int(generated_time // peak_window)
        peak_cpu_by_window[bucket] = peak_cpu_by_window.get(bucket, 0.0) + cpu
    peak_cpu_demand = max(peak_cpu_by_window.values(), default=0.0)

    average_load_ratio = (
        total_cpu_time_demand / total_cpu_time_capacity
        if total_cpu_time_capacity > 0.0 else 0.0
    )
    offered_peak_ratio = (
        peak_cpu_demand / total_cpu_capacity
        if total_cpu_capacity > 0.0 else 0.0
    )

    return {
        'task_count': task_count,
        'compute_node_count': compute_node_count,
        'tasks_per_node': task_count / compute_node_count if compute_node_count > 0 else 0.0,
        'total_cpu_capacity': total_cpu_capacity,
        'total_compute_capacity': total_cpu_capacity,
        'total_task_cpu': total_task_cpu,
        'avg_task_cpu': float(np.mean(cpu_values)) if cpu_values else 0.0,
        'avg_task_duration': float(np.mean(duration_values)) if duration_values else 0.0,
        'arrival_rate': task_count / simulation_duration if simulation_duration > 0.0 else 0.0,
        'total_cpu_time_demand': total_cpu_time_demand,
        'total_cpu_time_capacity': total_cpu_time_capacity,
        'simple_cpu_ratio': (
            total_task_cpu / total_cpu_capacity
            if total_cpu_capacity > 0.0 else 0.0
        ),
        'average_load_ratio': average_load_ratio,
        'avg_cpu_time_load_ratio': average_load_ratio,
        'peak_cpu_demand': peak_cpu_demand,
        'offered_peak_ratio': offered_peak_ratio,
        'peak_instant_load_ratio': offered_peak_ratio,
    }


def calculate_static_task_metrics(
    tasks,
    simulation_duration,
    total_compute_capacity,
    peak_window=None,
):
    tasks = list(tasks or [])
    task_count = len(tasks)
    simulation_duration = max(float(simulation_duration or 0.0), 0.0)
    total_compute_capacity = max(float(total_compute_capacity or 0.0), 0.0)
    peak_window = float(peak_window or getattr(config, 'SCHEDULING_CYCLE', 1.0))
    peak_window = max(peak_window, 1e-8)

    cpu_values = [float(task.get('cpu', 0.0)) for task in tasks]
    duration_values = [float(task.get('duration', 0.0)) for task in tasks]
    cpu_time_values = [task_cpu_time(task) for task in tasks]
    peak_cpu_by_window = {}
    for task in tasks:
        generated_time = float(task.get('generated_time', 0.0))
        bucket = int(generated_time // peak_window)
        peak_cpu_by_window[bucket] = peak_cpu_by_window.get(bucket, 0.0) + float(task.get('cpu', 0.0))

    peak_cpu_demand = max(peak_cpu_by_window.values(), default=0.0)

    supply_cpu_time = total_compute_capacity * simulation_duration
    avg_cpu_time_load_ratio = (
        float(np.sum(cpu_time_values)) / supply_cpu_time
        if supply_cpu_time > 0.0 else 0.0
    )
    peak_instant_load_ratio = (
        peak_cpu_demand / total_compute_capacity
        if total_compute_capacity > 0.0 else 0.0
    )
    metrics = {
        'task_count': task_count,
        'arrival_rate': task_count / simulation_duration if simulation_duration > 0.0 else 0.0,
        'avg_task_cpu': float(np.mean(cpu_values)) if cpu_values else 0.0,
        'avg_task_duration': float(np.mean(duration_values)) if duration_values else 0.0,
        'avg_cpu_time_load_ratio': avg_cpu_time_load_ratio,
        'peak_instant_load_ratio': peak_instant_load_ratio,
        'total_task_cpu': float(np.sum(cpu_values)) if cpu_values else 0.0,
        'total_cpu_time_demand': float(np.sum(cpu_time_values)) if cpu_time_values else 0.0,
        'total_cpu_time_capacity': supply_cpu_time,
        'simple_cpu_ratio': (
            float(np.sum(cpu_values)) / total_compute_capacity
            if total_compute_capacity > 0.0 else 0.0
        ),
        'average_load_ratio': avg_cpu_time_load_ratio,
        'peak_cpu_demand': peak_cpu_demand,
        'offered_peak_ratio': peak_instant_load_ratio,
    }

    return metrics


def generate_current_static_data(cycles=1000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    env = NetworkEnvironment()
    total_compute_capacity = sum(res['total'] for res in env.node_resources.values())
    task_manager = TaskManager(env.base_stations, total_compute_capacity=total_compute_capacity)
    peak_cpu_time_budget = (
        total_compute_capacity
        * config.SCHEDULING_CYCLE
        * getattr(config, 'TASK_PEAK_LOAD_MULTIPLIER', 1.3)
    )

    tasks = []
    global_time = 0.0
    for cycle in range(int(cycles)):
        global_time += config.SCHEDULING_CYCLE
        lam, _ = task_manager.get_dynamic_task_rate(global_time)
        tasks.extend(task_manager.generate_tasks(
            np.random.poisson(lam),
            global_time,
            cycle,
            cpu_budget=peak_cpu_time_budget,
        ))

    simulation_duration = max(global_time, config.SCHEDULING_CYCLE)
    metrics = calculate_static_task_metrics(
        tasks,
        simulation_duration,
        total_compute_capacity,
        peak_window=config.SCHEDULING_CYCLE,
    )
    metrics.update(analyze_task_resource_ratio(
        tasks,
        env,
        simulation_duration,
        peak_window=config.SCHEDULING_CYCLE,
    ))
    metrics.update({
        'simulation_duration': simulation_duration,
        'total_compute_capacity': total_compute_capacity,
        'compute_node_count': len(env.compute_nodes),
        'base_station_count': len(env.base_stations),
        'scheduling_cycle': config.SCHEDULING_CYCLE,
        'sample_cycles': int(cycles),
        'seed': int(seed),
    })
    return metrics


def format_task_resource_report(stats):
    return (
        "\n========== 任务与资源统计 ==========\n"
        f"任务数量：{stats['task_count']}\n"
        f"计算节点数量：{stats['compute_node_count']}\n"
        f"平均每节点任务数：{stats['tasks_per_node']:.2f}\n"
        f"系统瞬时CPU总容量：{stats['total_cpu_capacity']:.2f}\n"
        f"任务CPU需求总和：{stats['total_task_cpu']:.2f}\n"
        f"任务平均CPU需求：{stats['avg_task_cpu']:.2f}\n"
        f"任务平均持续时间：{stats['avg_task_duration']:.2f}\n"
        f"任务平均到达率：{stats['arrival_rate']:.4f} 个/时间单位\n"
        f"任务CPU时间总需求：{stats['total_cpu_time_demand']:.2f}\n"
        f"系统CPU时间总供给：{stats['total_cpu_time_capacity']:.2f}\n"
        f"简单CPU比例：{stats['simple_cpu_ratio'] * 100:.2f}%\n"
        f"平均CPU时间负载率：{stats['average_load_ratio'] * 100:.2f}%\n"
        f"任务生成峰值CPU需求：{stats['peak_cpu_demand']:.2f}\n"
        f"任务生成峰值负载率：{stats['offered_peak_ratio'] * 100:.2f}%\n"
        "====================================\n"
    )


def main():
    parser = argparse.ArgumentParser(description='Calculate current static workload metrics.')
    parser.add_argument('--cycles', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report', action='store_true', help='Print a human-readable Chinese report.')
    args = parser.parse_args()

    metrics = generate_current_static_data(cycles=args.cycles, seed=args.seed)
    if args.report:
        print(format_task_resource_report(metrics))
    else:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
