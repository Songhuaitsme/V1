import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime


class TrainingVisualizer:
    def __init__(self, log_dir='logs/tensorboard/'):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = os.path.join(log_dir, current_time)
        try:
            self.writer = tf.summary.create_file_writer(self.train_log_dir)
        except Exception:
            self.writer = None

        self.history = {
            'step': [], 'reward': [], 'success_rate': [],

            'avg_physical_latency': [], 'avg_end_to_end_latency': [],'cpu_usage_mean': [],
            'tier1_ratio': [], 'tier2_ratio': [], 'tier3_ratio': [] , #分档任务比例
            'region_cpu': {} , # 各分区的 CPU 利用率历史数据
            'loss': []
        }

    def log_step(self, step, metrics: dict, tier_counts: dict = None , region_usage : dict = None):
        """
        记录指标，并支持传入本周期各 tier 的任务分配总数
        :param tier_counts: 例如 {1: 50, 2: 30, 3: 20}
        """
        if self.writer is not None:
            with self.writer.as_default():
                # 1. 记录通用标量
                for key, value in metrics.items():
                    tf.summary.scalar(f'Train/{key}', value, step=step)
                    if key in self.history:
                        self.history[key].append(value)

                # 2. 记录电价区分布
                if tier_counts:
                    total = sum(tier_counts.values()) or 1
                    for t in [1, 2, 3]:
                        ratio = tier_counts.get(t, 0) / total
                        tf.summary.scalar(f'Economics/Tier{t}_Selection_Ratio', ratio, step=step)
                        self.history[f'tier{t}_ratio'].append(ratio)

                # 3. 记录分区 CPU 利用率
                if region_usage:
                    for reg, usage in region_usage.items():
                        tf.summary.scalar(f'Region_CPU/{reg}_Usage', usage, step=step)
                        if reg not in self.history['region_cpu']:
                            self.history['region_cpu'][reg] = []
                        self.history['region_cpu'][reg].append(usage)

                self.history['step'].append(step)
            self.writer.flush()
            return

        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

        if tier_counts:
            total = sum(tier_counts.values()) or 1
            for t in [1, 2, 3]:
                ratio = tier_counts.get(t, 0) / total
                self.history[f'tier{t}_ratio'].append(ratio)

        if region_usage:
            for reg, usage in region_usage.items():
                if reg not in self.history['region_cpu']:
                    self.history['region_cpu'][reg] = []
                self.history['region_cpu'][reg].append(usage)

        self.history['step'].append(step)

    def generate_final_report(self, save_path='training_evaluation.png', show=False):
        plt.figure(figsize=(18, 16))
        plt.suptitle("训练评价报告 ", fontsize=20, y=0.98)

        # 指标 1: 奖励与收敛性 (Reward & Loss)
        plt.subplot(3, 3, 1)
        plt.plot(self.history['reward'], color='blue', label='Reward')
        plt.title("平均奖励趋势")
        plt.grid(True)

        # 指标 2: 任务成功率 (Evaluation: Success Rate)
        plt.subplot(3, 3, 2)
        plt.plot(self.history['success_rate'], color='green')
        plt.title("任务成功率")
        plt.ylim(0, 1.1)

        # 指标 3: 系统负载 (CPU Usage)
        plt.subplot(3, 3, 3)
        plt.plot(self.history['cpu_usage_mean'], color='purple')
        plt.title("平均资源利用率")

        # # 指标 4: 时延评价 (Latency Analysis)
        # plt.subplot(3, 3, 4)
        # plt.plot(self.history['avg_latency'], color='orange')
        # plt.title("时延表现: 平均传输+传播时延")

        # 指标 4: 时延评价 (双线对比展示)
        plt.subplot(3, 3, 4)
        # 【修改】：同时画出两条线
        plt.plot(self.history['avg_physical_latency'], color='orange', label='物理路由时延')
        plt.plot(self.history['avg_end_to_end_latency'], color='red', alpha=0.7, label='端到端总时延')
        plt.title("时延表现对比")
        plt.legend(loc='upper right')

        # 指标 5: 任务区域分布演变 (经济决策倾向)
        plt.subplot(3, 3, 5)
        steps = range(len(self.history['tier1_ratio']))
        plt.stackplot(steps,
                      self.history['tier1_ratio'],
                      self.history['tier2_ratio'],
                      self.history['tier3_ratio'],
                      labels=['一档 (低价)', '二档 (中价)', '三档 (高价)'],
                      colors=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7)
        plt.title(" 各电价区任务分配比例")
        plt.legend(loc='upper left')
        plt.ylabel("比例")

        # 指标 6: 成本 vs 时延 (散点图)
        plt.subplot(3, 3, 6)
        plt.scatter(self.history['avg_end_to_end_latency'], self.history['reward'],
                    c=self.history['success_rate'], cmap='viridis', alpha=0.6)
        plt.colorbar(label='成功率')
        plt.title("e2e时延-奖励分布图")
        plt.xlabel("平均时延 (s)")
        plt.ylabel("奖励值")

        # ==========================================
        # 【新增】指标 7: 各分区 CPU 利用率多线对比图
        # ==========================================
        plt.subplot(3, 3, 7)
        if 'region_cpu' in self.history and self.history['region_cpu']:
            # 按地区字母顺序排列，保证图例颜色稳定
            for reg, usages in sorted(self.history['region_cpu'].items()):
                # 使用 alpha=0.7 增加透明度，防止多条线互相遮挡过于严重
                plt.plot(self.history['step'], usages, label=f'{reg}区', alpha=0.7)
            plt.title("空间负载均衡: 各分区 CPU 平均利用率")
            plt.xlabel("调度周期 (Cycle)")
            plt.ylabel("利用率")
            plt.ylim(0, 1.05)
            # 将图例放在图外下方，水平排列
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)
            plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()
