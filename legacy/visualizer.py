import datetime
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


TENSORBOARD_SCALAR_GROUPS = {
    "Core": [
        "loss",
        "reward",
        "base_reward",
        "constraint_penalty",
        "epsilon",
        "training_updates",
    ],
    "Outcome": [
        "dispatch_rate",
        "defer_rate",
        "completion_rate",
        "throughput_rate",
        "drop_rate",
    ],
    "Workload": [
        "avg_cpu_time_load_ratio",
        "peak_instant_load_ratio",
        "cpu_usage_mean",
        "task_lambda",
        "task_load_target_utilization",
        "generated_tasks",
        "processed_tasks",
        "avg_task_cpu",
        "avg_task_duration",
        "total_cpu_time_demand",
        "peak_cpu_budget_ratio",
    ],
    "Queue": [
        "wait_queue_length",
        "deferred_tasks",
        "active_task_count",
    ],
    "Wait": [
        "avg_wait_penalty",
        "avg_wait_urgency",
        "avg_wait_retry_ratio",
        "avg_wait_queue_ratio",
        "avg_queue_wait_time",
        "avg_source_region_queue_ratio",
        "avg_same_sla_queue_ratio",
        "avg_wait_gain",
        "avg_wait_net_gain",
        "avg_immediate_best_score",
        "avg_future_best_score",
        "wait_gain_positive_ratio",
        "wait_allowed_by_gain_count",
        "wait_blocked_by_gain_count",
        "wait_no_feasible_compute_count",
    ],
    "Latency": [
        "avg_physical_latency",
        "avg_end_to_end_latency",
        "p95_end_to_end_latency",
    ],
    "Cost": [
        "avg_cost_per_cpu_time",
        "p95_cost_per_cpu_time",
        "avg_cost_ratio",
        "p95_cost_ratio",
        "cost_saving_ratio",
    ],
    "Green": [
        "system_green_absorption_ratio",
        "green_unused_ratio",
        "green_load_coverage_ratio",
        "green_supply_demand_ratio",
        "total_green_used_mw",
        "total_green_supply_mw",
        "total_power_demand_mw",
        "green_coordination_score",
    ],
    "Reward": [
        "R_latency",
        "R_cost",
        "R_green",
        "R_balance",
        "R_success",
        "R_cost_spike",
    ],
    "Constraints": [
        "constraint_sla_violation",
        "constraint_drop",
        "constraint_cost_over_budget",
        "constraint_overload",
        "lambda_sla_violation",
        "lambda_drop",
        "lambda_cost_over_budget",
        "lambda_overload",
    ],
    "Placement": [
        "load_balance_score",
        "selected_green_rich_ratio",
        "selected_low_green_ratio",
        "tier_cpu_1",
        "tier_cpu_2",
        "tier_cpu_3",
        "tier1_ratio",
        "tier2_ratio",
        "tier3_ratio",
    ],
    "CECI": [
        "coordination_score",
        "ceci_raw",
        "ceci_effective",
        "ceci",
    ],
}

# Kept out of TensorBoard by default to keep the Scalars page readable.
# They are still stored in history/CSV and can be restored to the whitelist above.
# Removed duplicate load aliases:
#   "average_load_ratio" == "avg_cpu_time_load_ratio"
#   "offered_peak_ratio" == "peak_instant_load_ratio"
#   "avg_arrival_cpu_per_cycle_ratio" == "avg_cpu_time_load_ratio"
#   "max_arrival_cpu_ratio" == "peak_instant_load_ratio"
# Raw count/capacity fields:
#   "task_count", "compute_node_count", "tasks_per_node", "total_cpu_capacity",
#   "total_task_cpu", "total_cpu_time_capacity", "arrival_rate", "arrival_cpu",
#   "peak_cpu_demand"
# Verbose cost distribution:
#   "avg_cost", "p50_cost", "p95_cost", "max_cost", "p50_cost_per_cpu_time",
#   "max_cost_per_cpu_time", "max_cost_ratio"
# Verbose green/reward internals:
#   "green_match_ratio", "green_absorption_ratio", "system_green_absorption_delta",
#   "total_green_unused_mw", "avg_action_system_absorption_delta",
#   "min_action_system_absorption_delta", "max_action_system_absorption_delta",
#   "avg_green_unused_ratio_after", "avg_green_load_coverage_after",
#   "R_green_match", "R_green_absorption_delta", "R_green_waste",
#   "R_green_load_coverage", "R_green_total"
# Per-region routing fields:
#   "region_cpu_*", "selected_region_ratio_*"


class TrainingVisualizer:
    def __init__(self, log_dir='logs/tensorboard/'):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = os.path.join(log_dir, current_time)
        try:
            self.writer = tf.summary.create_file_writer(self.train_log_dir)
        except Exception:
            self.writer = None

        self.history = {
            'step': [],
            'loss': [],
            'reward': [],
            'base_reward': [],
            'constraint_penalty': [],
            'dispatch_rate': [],
            'defer_rate': [],
            'completion_rate': [],
            'throughput_rate': [],
            'drop_rate': [],
            'generated_tasks': [],
            'processed_tasks': [],
            'succeeded_tasks': [],
            'dropped_tasks': [],
            'deferred_tasks': [],
            'wait_queue_length': [],
            'active_task_count': [],
            'avg_wait_penalty': [],
            'avg_wait_urgency': [],
            'avg_wait_retry_ratio': [],
            'avg_wait_queue_ratio': [],
            'avg_queue_wait_time': [],
            'avg_source_region_queue_ratio': [],
            'avg_same_sla_queue_ratio': [],
            'avg_wait_gain': [],
            'avg_wait_net_gain': [],
            'avg_immediate_best_score': [],
            'avg_future_best_score': [],
            'wait_gain_positive_ratio': [],
            'wait_allowed_by_gain_count': [],
            'wait_blocked_by_gain_count': [],
            'wait_no_feasible_compute_count': [],
            'task_count': [],
            'compute_node_count': [],
            'tasks_per_node': [],
            'total_cpu_capacity': [],
            'total_task_cpu': [],
            'arrival_rate': [],
            'avg_task_cpu': [],
            'avg_task_duration': [],
            'total_cpu_time_demand': [],
            'total_cpu_time_capacity': [],
            'simple_cpu_ratio': [],
            'avg_cpu_time_load_ratio': [],
            'peak_cpu_demand': [],
            'peak_instant_load_ratio': [],
            'avg_physical_latency': [],
            'avg_end_to_end_latency': [],
            'p95_end_to_end_latency': [],
            'cpu_usage_mean': [],
            'avg_cost': [],
            'p50_cost': [],
            'p95_cost': [],
            'max_cost': [],
            'avg_cost_per_cpu_time': [],
            'p50_cost_per_cpu_time': [],
            'p95_cost_per_cpu_time': [],
            'max_cost_per_cpu_time': [],
            'avg_cost_ratio': [],
            'p95_cost_ratio': [],
            'max_cost_ratio': [],
            'avg_cpu_time_demand_success': [],
            'p95_cpu_time_demand_success': [],
            'green_match_ratio': [],
            'green_absorption_ratio': [],
            'system_green_absorption_ratio': [],
            'system_green_absorption_delta': [],
            'total_green_used_mw': [],
            'total_green_supply_mw': [],
            'total_green_unused_mw': [],
            'total_power_demand_mw': [],
            'green_unused_ratio': [],
            'green_load_coverage_ratio': [],
            'green_supply_demand_ratio': [],
            'green_coordination_score': [],
            'cost_saving_ratio': [],
            'load_balance_score': [],
            'coordination_score': [],
            'ceci_raw': [],
            'ceci_effective': [],
            'ceci': [],
            'selected_green_rich_ratio': [],
            'selected_low_green_ratio': [],
            'R_latency': [],
            'R_cost': [],
            'R_green': [],
            'R_balance': [],
            'R_success': [],
            'R_cost_spike': [],
            'constraint_sla_violation': [],
            'constraint_drop': [],
            'constraint_cost_over_budget': [],
            'constraint_overload': [],
            'lambda_sla_violation': [],
            'lambda_drop': [],
            'lambda_cost_over_budget': [],
            'lambda_overload': [],
            'tier1_ratio': [],
            'tier2_ratio': [],
            'tier3_ratio': [],
            'region_cpu': {},
        }

    def log_step(self, step, metrics: dict, tier_counts: dict = None, region_usage: dict = None):
        prior_steps = len(self.history['step'])
        special_keys = {'step', 'region_cpu', 'tier1_ratio', 'tier2_ratio', 'tier3_ratio'}
        for key in metrics:
            if key not in self.history:
                self.history[key] = [np.nan] * prior_steps
        for key, values in self.history.items():
            if key in special_keys or not isinstance(values, list):
                continue
            values.append(metrics.get(key, np.nan))

        if self.writer is not None:
            with self.writer.as_default():
                for group, keys in TENSORBOARD_SCALAR_GROUPS.items():
                    for key in keys:
                        if key in metrics:
                            tf.summary.scalar(f'{group}/{key}', metrics[key], step=step)

                if tier_counts:
                    total = sum(tier_counts.values()) or 1
                    for tier in [1, 2, 3]:
                        ratio = tier_counts.get(tier, 0) / total
                        tf.summary.scalar(f'Placement/tier{tier}_selection_ratio', ratio, step=step)
                        self.history[f'tier{tier}_ratio'].append(ratio)

                if region_usage:
                    for region, usage in region_usage.items():
                        # Per-region CPU is noisy in TensorBoard; keep it in history/CSV
                        # and restore this line when regional debugging is needed.
                        # tf.summary.scalar(f'Placement/region_cpu_{region}', usage, step=step)
                        self.history['region_cpu'].setdefault(region, []).append(usage)

                self.history['step'].append(step)
            self.writer.flush()
            return

        if tier_counts:
            total = sum(tier_counts.values()) or 1
            for tier in [1, 2, 3]:
                self.history[f'tier{tier}_ratio'].append(tier_counts.get(tier, 0) / total)

        if region_usage:
            for region, usage in region_usage.items():
                self.history['region_cpu'].setdefault(region, []).append(usage)

        self.history['step'].append(step)

    def _plot_series(self, ax, keys, title, ylim=None):
        for key in keys:
            values = self.history.get(key, [])
            if values:
                ax.plot(self.history['step'][:len(values)], values, label=key)
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.5)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if len(keys) > 1:
            ax.legend(fontsize=8)

    def _plot_green_ratios(self, ax):
        left_keys = [
            'system_green_absorption_ratio',
            'green_unused_ratio',
            'green_load_coverage_ratio',
        ]
        for key in left_keys:
            values = self.history.get(key, [])
            if values:
                ax.plot(self.history['step'][:len(values)], values, label=key)
        ax.set_title('Green Ratios')
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylabel('Absorption / unused / coverage')

        right_ax = ax.twinx()
        supply_demand = self.history.get('green_supply_demand_ratio', [])
        if supply_demand:
            right_ax.plot(
                self.history['step'][:len(supply_demand)],
                supply_demand,
                color='tab:red',
                linestyle='--',
                label='green_supply_demand_ratio',
            )
        right_ax.set_ylabel('Supply / demand')

        left_lines, left_labels = ax.get_legend_handles_labels()
        right_lines, right_labels = right_ax.get_legend_handles_labels()
        ax.legend(left_lines + right_lines, left_labels + right_labels, fontsize=8)

    def generate_final_report(self, save_path='training_evaluation.png', show=False):
        fig, axes = plt.subplots(4, 3, figsize=(20, 20))
        axes = axes.ravel()
        fig.suptitle("Training Evaluation", fontsize=18)

        self._plot_series(axes[0], ['reward', 'base_reward'], 'Reward')
        self._plot_series(axes[1], ['loss'], 'Loss')
        self._plot_series(
            axes[2],
            ['dispatch_rate', 'defer_rate', 'completion_rate', 'throughput_rate', 'drop_rate'],
            'Dispatch / Defer / Done / Throughput / Drop',
            ylim=(0, 1.05),
        )
        self._plot_series(
            axes[3],
            ['avg_physical_latency', 'avg_end_to_end_latency', 'p95_end_to_end_latency'],
            'Latency',
        )
        self._plot_series(axes[4], ['green_match_ratio', 'system_green_absorption_ratio',
                                    'green_coordination_score', 'cost_saving_ratio',
                                    'load_balance_score', 'ceci_raw', 'ceci_effective'],
                          'CECI Components', ylim=(0, 1.05))
        self._plot_series(axes[5], ['total_green_supply_mw', 'total_green_used_mw',
                                    'total_green_unused_mw', 'total_power_demand_mw'],
                          'Green Power System (MW)')
        self._plot_green_ratios(axes[6])
        self._plot_series(axes[7], ['avg_cost', 'p50_cost', 'p95_cost', 'max_cost'],
                          'Cost Raw')
        self._plot_series(axes[8], ['avg_cost_per_cpu_time', 'p50_cost_per_cpu_time',
                                    'p95_cost_per_cpu_time', 'max_cost_per_cpu_time'],
                          'Cost Per CPU Time')
        self._plot_series(axes[9], ['R_latency', 'R_cost', 'R_green', 'R_balance',
                                    'R_success', 'R_cost_spike'],
                          'Reward Components')
        self._plot_series(axes[10], ['constraint_sla_violation', 'constraint_drop',
                                     'constraint_cost_over_budget', 'constraint_overload'],
                          'Constraint Costs')
        self._plot_series(axes[11], ['selected_green_rich_ratio', 'selected_low_green_ratio',
                                     'tier_cpu_1', 'tier_cpu_2', 'tier_cpu_3'],
                          'Region / Tier Usage', ylim=(0, 1.05))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def export_history_csv(self, save_path='training_metrics.csv'):
        if not self.history or not self.history.get('step'):
            return

        scalar_history = {
            key: values
            for key, values in self.history.items()
            if isinstance(values, list)
        }
        keys = list(scalar_history.keys())
        max_len = max(len(values) for values in scalar_history.values())
        with open(save_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=keys)
            writer.writeheader()
            for index in range(max_len):
                row = {}
                for key, values in scalar_history.items():
                    value = values[index] if index < len(values) else ''
                    row[key] = '' if isinstance(value, float) and np.isnan(value) else value
                writer.writerow(row)
