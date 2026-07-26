import random
import numpy as np
import math
from shared import config
from v1.domain.models import migrate_legacy_task, to_legacy_task_dict
from v1.domain.units import TimeConverter, validate_scheduling_grid


class TaskManager:
    def __init__(self, source_nodes, total_compute_capacity=None):
        self.source_nodes = list(source_nodes)
        self.total_compute_capacity = float(total_compute_capacity or 0.0)
        self.time_converter = TimeConverter.from_traffic_day_duration(
            config.TRAFFIC_DAY_DURATION_IN_SIM,
            getattr(config, "SIM_SECONDS_PER_UNIT", None),
        )
        validate_scheduling_grid(
            config.SCHEDULING_CYCLE,
            getattr(config, "GLOBAL_TIME_STEP_DURATION", None),
        )
        self._task_sequence = 0

        # ==========================================
        # [宏观层] 1. 空间异构性 (三级阶梯架构)
        # ==========================================
        # 初始化必须包含 Middle，否则后续 append 会报错
        self.regions = {'East': [], 'Middle': [], 'West': []}
        for node in self.source_nodes:
            # ILHG为东部核心枢纽，CBEF为西部低成本数据中心，其他归属中部骨干
            if str(node)[0] in ['I', 'L', 'H', 'G']:
                self.regions['East'].append(node)
            elif str(node)[0] in ['C', 'B', 'E', 'F']:
                self.regions['West'].append(node)
            else:
                self.regions['Middle'].append(node)

        # ==========================================
        # [中观层] 2. 级联传染性 MMPP 突发状态机
        # ==========================================
        # 0: 正常 (Normal), 1: 突发拥塞 (Burst)
        # MMPP disabled: no sudden regional high-traffic burst state.
        # self.burst_states = {'East': 0, 'Middle': 0, 'West': 0}

        # 突发倍率呈阶梯递减：东部核心极容易产生巨大峰值
        # self.burst_multipliers = {'East': 4.0, 'Middle': 3.0, 'West': 2.0}

        # 状态转移矩阵 (MMPP)
        # self.p_spontaneous_burst = {'East': 0.0005, 'Middle': 0.0002, 'West': 0.0001}
        # self.p_recover = {'East': 0.005, 'Middle': 0.008, 'West': 0.01}
        # self.p_contagion = 0.002  # 传染概率：上级拥塞导致下级被拖垮的概率

        # ==========================================
        # [微观层] 3. 资源特征画像与 SLA 约束矩阵
        # ==========================================
        self.task_templates = {
            "Realtime_Service": {  # 实时响应型服务
                "sla_type": "Hard", "latency_range": (0.01, 1.0),
                "base_data": 5, "base_cpu": 30, "base_dur":1.0,
                "lognormal_sigma": 0.3
            },
            "Interactive_Query": {  # 交互式分析
                "sla_type": "Soft", "latency_range": (1.0, 10.0),
                "base_data": 80, "base_cpu": 60, "base_dur": 8.0,
                "lognormal_sigma": 0.5
            },
            "Data_Intensive": {  # 数据密集型计算
                "sla_type": "Soft", "latency_range": (10.0, 100.0),
                "base_data": 300, "base_cpu": 100, "base_dur":20.0,
                "lognormal_sigma": 0.8
            },
            "Model_Training": {  # 模型密集型训练
                "sla_type": "Flexible", "latency_range": (100.0,1440),
                "sla_type": "Flexible", "latency_range": (100.0, 1440.0),
                "base_data": 200, "base_cpu": 300, "base_dur": 50.0,
                "lognormal_sigma": 1.2
            }
        }

    # ==========================================
    # 核心计算逻辑模块
    # ==========================================

    def _get_tidal_multiplier(self, task_type, sim_hr):
        """高斯混合模型与 Sigmoid 联合模拟平滑潮汐波动"""

        def gaussian(x, mu, sigma):
            return math.exp(-0.5 * ((x - mu) / sigma) ** 2)

        def sigmoid(x, k=3.3):
            """
            S型平滑函数。
            k: 爬坡系数。k=2.0 意味着大约有两三个小时的过渡期。
            """
            try:
                return 1.0 / (1.0 + math.exp(-k * x))
            except OverflowError:
                return 0.0 if x < 0 else 1.0

        if task_type == "Realtime_Service":
            return 0.3 + 0.8 * gaussian(sim_hr, 9.0, 1.5) + 0.9 * gaussian(sim_hr, 20.0, 2.0)
        elif task_type == "Interactive_Query":
            rise = sigmoid(sim_hr - 9.0, k=3.3)
            fall = sigmoid(sim_hr - 18.0, k=3.3)
            return 0.2 + 1.0 * (rise - fall)
        elif task_type == "Data_Intensive":
            return 0.1 + 1.5 * gaussian(sim_hr, 2.0, 2.0)
        elif task_type == "Model_Training":
            return 0.4 + 0.4 * gaussian(sim_hr, 4.0, 3.0)
        return 1.0

    # def _update_mmpp_network_states(self):
    #     """[中观层] 带有空间级联传染性的马尔可夫链更新 (东 -> 中 -> 西)"""
    #     # MMPP disabled.
    #     # 1. 东部状态流转 (独立)
        # if self.burst_states['East'] == 0:
        #     if random.random() < self.p_spontaneous_burst['East']:
        #         self.burst_states['East'] = 1
        # elif random.random() < self.p_recover['East']:
        #     self.burst_states['East'] = 0

        # 2. 中部状态流转 (自身拥塞 + 东部传染)
        # if self.burst_states['Middle'] == 0:
        #     prob_burst = self.p_spontaneous_burst['Middle']
        #     if self.burst_states['East'] == 1:
        #         prob_burst += self.p_contagion
        #     if random.random() < prob_burst:
        #         self.burst_states['Middle'] = 1
        # elif random.random() < self.p_recover['Middle']:
        #     self.burst_states['Middle'] = 0

        # 3. 西部状态流转 (自身拥塞 + 中部传染)
        # if self.burst_states['West'] == 0:
        #     prob_burst = self.p_spontaneous_burst['West']
        #     if self.burst_states['Middle'] == 1:
        #         prob_burst += self.p_contagion
        #     if random.random() < prob_burst:
        #         self.burst_states['West'] = 1
        # elif random.random() < self.p_recover['West']:
        #     self.burst_states['West'] = 0

    def _get_type_probabilities(self, sim_hr, region_choice=None):
        types = list(self.task_templates.keys())
        base_ratios = getattr(config, 'TASK_TYPE_BASE_RATIO', {})
        type_weights = []

        for task_type in types:
            weight = base_ratios.get(task_type, 1.0) * self._get_tidal_multiplier(task_type, sim_hr)

            if task_type == "Realtime_Service":
                if region_choice == 'West':
                    weight *= 0.3
                elif region_choice == 'Middle':
                    weight *= 0.5

            type_weights.append(max(0.0, weight))

        total_weight = sum(type_weights)
        if total_weight <= 0:
            return types, np.ones(len(types), dtype=float) / len(types)

        return types, np.array(type_weights, dtype=float) / total_weight

    @staticmethod
    def _estimate_template_work(tmpl):
        latent_sigma = tmpl["lognormal_sigma"]
        latent_second_moment = math.exp(2.0 * latent_sigma * latent_sigma)
        cpu_noise_mean = math.exp(0.5 * 0.5 * 0.5)
        dur_noise_mean = math.exp(0.5 * 0.3 * 0.3)
        return tmpl["base_cpu"] * tmpl["base_dur"] * latent_second_moment * cpu_noise_mean * dur_noise_mean

    def _estimate_expected_task_work(self, sim_hr):
        # MMPP burst weighting disabled; use stable regional base weights.
        # region_weights = {
        #     'East': 0.50 * (self.burst_multipliers['East'] if self.burst_states['East'] == 1 else 1.0),
        #     'Middle': 0.30 * (self.burst_multipliers['Middle'] if self.burst_states['Middle'] == 1 else 1.0),
        #     'West': 0.20 * (self.burst_multipliers['West'] if self.burst_states['West'] == 1 else 1.0),
        # }
        region_weights = {
            'East': 0.50,
            'Middle': 0.30,
            'West': 0.20,
        }
        total_region_weight = sum(region_weights.values())
        if total_region_weight <= 0:
            return 1.0

        expected_work = 0.0
        for region, region_weight in region_weights.items():
            types, type_probs = self._get_type_probabilities(sim_hr, region_choice=region)
            region_work = sum(
                prob * self._estimate_template_work(self.task_templates[task_type])
                for task_type, prob in zip(types, type_probs)
            )
            expected_work += (region_weight / total_region_weight) * region_work

        return max(1.0, expected_work)

    def _scheduling_cycle_seconds(self):
        """Return the scheduling cycle in seconds via the v1.0 unit boundary."""

        return self.time_converter.scheduling_cycle_seconds(config.SCHEDULING_CYCLE)

    def _cap_lambda_by_capacity(self, raw_lambda, sim_hr):
        if not getattr(config, 'ENABLE_CAPACITY_AWARE_TASK_GENERATION', True):
            return raw_lambda
        if self.total_compute_capacity <= 0:
            return raw_lambda

        target_utilization = max(0.0, getattr(config, 'TASK_LOAD_TARGET_UTILIZATION', 0.95))
        expected_work = self._estimate_expected_task_work(sim_hr)
        capacity_work_per_cycle = self.total_compute_capacity * target_utilization * config.SCHEDULING_CYCLE
        capacity_lambda = capacity_work_per_cycle / expected_work  *2.5
        return min(raw_lambda, capacity_lambda)

    def get_dynamic_task_rate(self, global_time):
        """计算当前时刻全网的总流量预期"""
        # MMPP burst-state update disabled.
        # self._update_mmpp_network_states()

        day_prog = (global_time % config.TRAFFIC_DAY_DURATION_IN_SIM) / config.TRAFFIC_DAY_DURATION_IN_SIM
        sim_hr = day_prog * 24.0

        # 基础每秒任务数分解到三个区域 (40% : 30% : 30%)
        base_rate_east = config.BASE_TASKS_PER_SECOND * 0.40
        base_rate_mid = config.BASE_TASKS_PER_SECOND * 0.30
        base_rate_west = config.BASE_TASKS_PER_SECOND * 0.30

        # 叠加 MMPP 突发倍率
        # rate_east = base_rate_east * (self.burst_multipliers['East'] if self.burst_states['East'] == 1 else 1.0)
        # rate_mid = base_rate_mid * (self.burst_multipliers['Middle'] if self.burst_states['Middle'] == 1 else 1.0)
        # rate_west = base_rate_west * (self.burst_multipliers['West'] if self.burst_states['West'] == 1 else 1.0)
        rate_east = base_rate_east
        rate_mid = base_rate_mid
        rate_west = base_rate_west

        cycle_seconds = self._scheduling_cycle_seconds()
        raw_lambda = (rate_east + rate_mid + rate_west) * cycle_seconds
        total_lambda = self._cap_lambda_by_capacity(raw_lambda, sim_hr)
        return total_lambda, sim_hr

    def _generate_coupled_attributes(self, tmpl):
        """[微观层] 使用对数正态分布 (Log-Normal) 替换帕累托分布生成高度耦合的资源需求矩阵"""
        # 1. 生成基于对数正态分布的共享潜变量
        # 这里的 mean=0.0，等价于底层正态分布 mu = ln(base_value) 提取出来的公因子
        latent_scale = np.random.lognormal(mean=0.0, sigma=tmpl["lognormal_sigma"])

        # 耦合生成 (基准值 * 变化规模 * 噪声)
        data_size = tmpl["base_data"] * latent_scale * np.random.lognormal(mean=0.0, sigma=0.4)
        cpu_demand = tmpl["base_cpu"] * latent_scale * np.random.lognormal(mean=0.0, sigma=0.5)
        duration = tmpl["base_dur"] * latent_scale * np.random.lognormal(mean=0.0, sigma=0.3)
        # 物理截断
        data_size = int(np.clip(data_size, 1, config.TASK_DATA_SIZE_RANGE[1] * 5))
        cpu_demand = int(np.clip(cpu_demand, 1, config.TASK_CPU_DEMAND[1] * 4))
        duration = max(0.1, duration)
        cpu_time_demand = cpu_demand * duration

        return data_size, cpu_demand, duration, cpu_time_demand

    def generate_tasks(self, num_tasks, global_time, cycle, cpu_budget=None):
        new_tasks = []
        if num_tasks <= 0: return new_tasks
        batch_cpu_time = 0.0
        if getattr(config, 'ENABLE_CAPACITY_AWARE_TASK_GENERATION', True):
            default_peak = (
                self.total_compute_capacity
                * config.SCHEDULING_CYCLE
                * getattr(config, 'TASK_PEAK_LOAD_MULTIPLIER', 1.3)
            )
            cpu_budget = default_peak if cpu_budget is None else cpu_budget
            cpu_budget = float(cpu_budget or 0.0)
        else:
            cpu_budget = 0.0

        day_prog = (global_time % config.TRAFFIC_DAY_DURATION_IN_SIM) / config.TRAFFIC_DAY_DURATION_IN_SIM
        sim_hr = day_prog * 24.0

        for _ in range(num_tasks):
            # 1. 结合实时状态计算各区域生成任务的动态权重
            # MMPP burst weighting disabled.
            # weight_e = 0.50 * (self.burst_multipliers['East'] if self.burst_states['East'] == 1 else 1.0)
            # weight_m = 0.30 * (self.burst_multipliers['Middle'] if self.burst_states['Middle'] == 1 else 1.0)
            # weight_w = 0.20 * (self.burst_multipliers['West'] if self.burst_states['West'] == 1 else 1.0)
            weight_e = 0.50
            weight_m = 0.30
            weight_w = 0.20

            total_weight = weight_e + weight_m + weight_w
            rand_val = random.uniform(0, total_weight)

            # 轮盘赌选择区域
            if rand_val < weight_e:
                region_choice = 'East'
            elif rand_val < weight_e + weight_m:
                region_choice = 'Middle'
            else:
                region_choice = 'West'

            source_node = random.choice(self.regions[region_choice])

            # 2. 根据时间潮汐决定具体业务类型
            type_weights = []
            types = list(self.task_templates.keys())
            base_ratios = getattr(config, 'TASK_TYPE_BASE_RATIO', {})
            for t in types:
                w = base_ratios.get(t, 1.0) * self._get_tidal_multiplier(t, sim_hr)

                # 空间阶梯硬约束：下沉越深，越难以产生实时敏感任务
                if t == "Realtime_Service":
                    if region_choice == 'West':
                        w *= 0.3  # 西部极少产生
                    elif region_choice == 'Middle':
                        w *= 0.5  # 中部适中
                type_weights.append(w)

            type_probs = np.array(type_weights) / sum(type_weights)
            selected_type = np.random.choice(types, p=type_probs)
            tmpl = self.task_templates[selected_type]

            # 3. 生成属性与构建任务字典
            data_size, cpu_demand, duration, cpu_time_demand = self._generate_coupled_attributes(tmpl)
            if cpu_budget > 0 and new_tasks and batch_cpu_time + cpu_time_demand > cpu_budget:
                break
            batch_cpu_time += cpu_time_demand
            latency_limit = random.uniform(*tmpl["latency_range"])
            arrival_time = global_time - random.uniform(0, config.SCHEDULING_CYCLE)
            self._task_sequence += 1
            legacy_task = {
                'id': f"{cycle}_{self._task_sequence:08d}",
                'type': selected_type,
                'sla_type': tmpl["sla_type"],
                'source_node': source_node,
                'data_size': data_size,
                'cpu': cpu_demand,
                'duration': duration,
                'cpu_time': cpu_time_demand,
                'generated_time': arrival_time,
                'latency_limit': latency_limit,
                'bw': max(5, int(data_size * 0.15)),
                'retry_count': 0
            }
            task_spec = migrate_legacy_task(legacy_task)
            new_tasks.append(to_legacy_task_dict(
                task_spec,
                original=legacy_task,
                legacy_latency_limit=latency_limit,
            ))

        return new_tasks

    def generate_task_specs(self, num_tasks, global_time, cycle, cpu_budget=None):
        """Canonical v1.0 generation API; legacy dictionaries never leave adapter."""

        return [
            migrate_legacy_task(task)
            for task in self.generate_tasks(
                num_tasks,
                global_time,
                cycle,
                cpu_budget=cpu_budget,
            )
        ]

    @staticmethod
    def calculate_priority(task, current_time):
        rem = (task['generated_time'] + task['latency_limit']) - current_time
        base = 99999.0 if rem <= 1e-5 else 1.0 / rem
        sla_multiplier = 10.0 if task['sla_type'] == 'Hard' else (2.0 if task['sla_type'] == 'Soft' else 1.0)
        return base * sla_multiplier * (1.0 + task.get('retry_count', 0) * config.RETRY_PRIORITY_WEIGHT)
