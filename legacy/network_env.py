import numpy as np
import math
from shared import config
from shared.topology_manager import TopologyManager
from shared.pricing_manager import PricingManager


def get_max_retries_for_task(task):
    sla_type = task.get("sla_type", "Soft") if task else "Soft"
    return getattr(config, "MAX_RETRIES_BY_SLA", {}).get(
        sla_type,
        getattr(config, "MAX_RETRIES", 5),
    )


def estimate_wait_penalty_detail(env, task, wait_queue, next_queue_delay, sla_type):
    max_retries = get_max_retries_for_task(task)
    wait_params = getattr(config, "WAIT_PENALTY_PARAMS", {})
    sla_wait_params = wait_params.get(
        sla_type,
        wait_params.get("Soft", {"base": 0.8, "urgency": 3.0, "retry": 2.0, "queue": 1.0})
    )
    remaining_sla_time = max(0.0, task['latency_limit'] - next_queue_delay)
    remaining_ratio = remaining_sla_time / max(task['latency_limit'], 1e-8)
    urgency = env._clip(1.0 - env._clip(remaining_ratio, 0.0, 1.0), 0.0, 1.0)
    retry_ratio = env._clip(
        task.get('retry_count', 0) / max(max_retries, 1),
        0.0,
        1.0
    )
    queue_length = len(wait_queue) if wait_queue else 0
    queue_ratio = env._clip(
        queue_length / max(config.MAX_QUEUE_LENGTH, 1),
        0.0,
        1.0
    )
    source_region = env.node_regions.get(task.get("source_node"))
    source_region_count = 0
    same_sla_count = 0
    if wait_queue:
        for queued_task in wait_queue:
            if env.node_regions.get(queued_task.get("source_node")) == source_region:
                source_region_count += 1
            if queued_task.get("sla_type", "Soft") == sla_type:
                same_sla_count += 1
    source_region_queue_ratio = env._clip(
        source_region_count / max(env.region_queue_norm, 1.0),
        0.0,
        1.0
    )
    same_sla_queue_ratio = env._clip(
        same_sla_count / max(1, queue_length),
        0.0,
        1.0
    )
    wait_penalty = -(
        float(sla_wait_params.get("base", 0.8))
        + float(sla_wait_params.get("urgency", 3.0)) * (urgency ** 2)
        + float(sla_wait_params.get("retry", 2.0)) * retry_ratio
        + float(sla_wait_params.get("queue", 1.0)) * queue_ratio
    )
    return {
        "urgency": urgency,
        "retry_ratio": retry_ratio,
        "queue_ratio": queue_ratio,
        "wait_penalty": wait_penalty,
        "queue_wait_time": config.SCHEDULING_CYCLE,
        "retry_count": task.get("retry_count", 0),
        "max_retries": max_retries,
        "remaining_sla_time": remaining_sla_time,
        "wait_queue_length": queue_length,
        "source_region_queue_ratio": source_region_queue_ratio,
        "same_sla_queue_ratio": same_sla_queue_ratio,
    }


def _candidate_times_for_task(task, global_time, include_future=True, include_current=True):
    scheduling_cycle = max(float(getattr(config, "SCHEDULING_CYCLE", 0.0)), 1e-8)
    generated_time = task.get('generated_time', global_time)
    latency_limit = max(float(task.get('latency_limit', 0.0)), 1e-8)
    current_queue_delay = max(0.0, global_time - generated_time)
    remaining_delay_budget = max(0.0, latency_limit - current_queue_delay)
    sla_lookahead_cycles = max(
        0,
        int(math.floor((remaining_delay_budget - 1e-12) / scheduling_cycle)),
    )
    configured_horizon = getattr(config, "WAIT_GAIN_LOOKAHEAD_CYCLES", None)
    lookahead_horizon_cycles = (
        sla_lookahead_cycles
        if configured_horizon is None
        else min(sla_lookahead_cycles, max(0, int(configured_horizon)))
    )

    offsets = [0] if include_current else []
    if include_future and lookahead_horizon_cycles > 0:
        max_samples = getattr(config, "WAIT_GAIN_MAX_LOOKAHEAD_SAMPLES", None)
        if max_samples is None or lookahead_horizon_cycles <= int(max_samples):
            offsets.extend(range(1, lookahead_horizon_cycles + 1))
        else:
            sample_count = max(1, int(max_samples))
            offsets.extend(
                max(1, min(lookahead_horizon_cycles, int(round(offset))))
                for offset in np.linspace(1, lookahead_horizon_cycles, sample_count)
            )

    offsets = sorted(set(offsets))
    times = [global_time + offset * scheduling_cycle for offset in offsets]
    return times, {
        "lookahead_cycles": max(0, len([offset for offset in offsets if offset > 0])),
        "lookahead_horizon_cycles": lookahead_horizon_cycles,
        "sla_lookahead_cycles": sla_lookahead_cycles,
        "lookahead_sampled": (
            include_future
            and lookahead_horizon_cycles > 0
            and len([offset for offset in offsets if offset > 0]) < lookahead_horizon_cycles
        ),
    }


def evaluate_schedule_candidates(
    env,
    task,
    wait_queue,
    global_time,
    candidate_times=None,
    candidate_nodes=None,
    include_future=True,
    include_current=True,
):
    cpu_demand = env.get_task_cpu_demand(task)
    execution_duration = env.estimate_execution_duration(None, task)
    latency_limit = max(float(task.get('latency_limit', 0.0)), 1e-8)
    source_node = task.get('source_node')
    if candidate_times is None:
        candidate_times, _ = _candidate_times_for_task(
            task,
            global_time,
            include_future=include_future,
            include_current=include_current,
        )
    if candidate_nodes is None:
        candidate_nodes = env.compute_nodes

    score_weights = {
        "cost": float(getattr(config, "WAIT_GAIN_COST_WEIGHT", 1.0)),
        "green": float(getattr(config, "WAIT_GAIN_GREEN_WEIGHT", 0.5)),
        "balance": float(getattr(config, "WAIT_GAIN_BALANCE_WEIGHT", 0.1)),
        "latency": float(getattr(config, "WAIT_GAIN_LATENCY_WEIGHT", 0.2)),
    }
    price_norm = max(
        getattr(config, "PRICE_NORMALIZATION_FACTOR", 0.05),
        getattr(config, "BASELINE_ELECTRICITY_PRICE_YUAN_PER_MW", 1.20)
        * getattr(config, "CPU_POWER_UNIT_MW", 0.01),
        1e-8,
    )
    link_usage_cache = {}
    path_cache = {}
    reserved_cpu_cache = {}
    price_cache = {}
    power_profile_cache = {}
    candidates = []

    for schedule_time in sorted(set(float(t) for t in candidate_times)):
        finish_time = schedule_time + execution_duration
        interval_key = (schedule_time, finish_time)
        interval_link_usage = link_usage_cache.get(interval_key)
        if interval_link_usage is None:
            interval_link_usage = env.get_reserved_link_usage(schedule_time, finish_time)
            link_usage_cache[interval_key] = interval_link_usage

        for action_index, node in enumerate(env.compute_nodes):
            if node not in candidate_nodes:
                continue

            reserved_key = (node, schedule_time, finish_time)
            reserved_cpu = reserved_cpu_cache.get(reserved_key)
            if reserved_cpu is None:
                reserved_cpu = env.get_reserved_cpu_usage(node, schedule_time, finish_time)
                reserved_cpu_cache[reserved_key] = reserved_cpu
            resource = env.node_resources[node]
            projected_used = reserved_cpu + cpu_demand
            if projected_used > resource["total"]:
                continue

            path_key = (source_node, node, task.get('bw', 0.0), schedule_time, finish_time)
            path = path_cache.get(path_key)
            if path_key not in path_cache:
                path = [node] if source_node == node else env.topo_manager.find_path(
                    source_node,
                    node,
                    bw_demand=task.get('bw', 0.0),
                    link_usage=interval_link_usage,
                )
                path_cache[path_key] = path
            if path is None:
                continue

            network_delay = env.topo_manager.calculate_transmission_delay(
                path,
                task.get('data_size', 0.0),
                task.get('bw', 0.0),
            )
            queue_delay = max(0.0, schedule_time - task.get('generated_time', schedule_time))
            start_delay = queue_delay + network_delay
            if start_delay > latency_limit:
                continue

            price_key = (node, schedule_time, cpu_demand, finish_time)
            price = price_cache.get(price_key)
            if price is None:
                price = env.get_dynamic_cpu_price(
                    node,
                    schedule_time,
                    cpu_delta=cpu_demand,
                    interval_start=schedule_time,
                    interval_end=finish_time,
                )
                price_cache[price_key] = price
            cost_saving_estimate = max(0.0, min(1.0, 1.0 - price / price_norm))

            projected_usage = {
                "total": resource["total"],
                "used": min(resource["total"], projected_used),
            }
            power_key = (node, schedule_time, projected_usage["used"])
            power_profile = power_profile_cache.get(power_key)
            if power_profile is None:
                power_profile = env.pricing_manager.get_node_power_profile(node, projected_usage, schedule_time)
                power_profile_cache[power_key] = power_profile
            green_match_estimate = float(power_profile.get("green_match_ratio", 0.0))
            capacity_margin = (
                max(0.0, resource["total"] - projected_used) / resource["total"]
                if resource["total"] > 0 else 0.0
            )
            delay_risk = max(0.0, min(1.0, start_delay / latency_limit))
            score = (
                score_weights["cost"] * cost_saving_estimate
                + score_weights["green"] * green_match_estimate
                + score_weights["balance"] * capacity_margin
                - score_weights["latency"] * delay_risk
            )
            candidates.append({
                "node": node,
                "action_index": action_index,
                "schedule_time": schedule_time,
                "finish_time": finish_time,
                "path": path,
                "network_delay": network_delay,
                "queue_delay": queue_delay,
                "start_delay": start_delay,
                "price": price,
                "cost_saving_estimate": float(cost_saving_estimate),
                "green_match_estimate": green_match_estimate,
                "capacity_margin": float(capacity_margin),
                "delay_risk": float(delay_risk),
                "score": float(score),
                "reserved_cpu": reserved_cpu,
                "projected_used": projected_used,
            })
    return candidates


def estimate_wait_opportunity(env, task, wait_queue, global_time, candidates=None):
    if candidates is None:
        candidates = evaluate_schedule_candidates(env, task, wait_queue, global_time)
    _, stats = _candidate_times_for_task(task, global_time)
    immediate = [
        candidate for candidate in candidates
        if abs(candidate["schedule_time"] - global_time) <= 1e-9
    ]
    future = [
        candidate for candidate in candidates
        if candidate["schedule_time"] > global_time + 1e-9
    ]
    immediate_best = max(immediate, key=lambda item: item["score"]) if immediate else None
    future_best = max(future, key=lambda item: item["score"]) if future else None

    immediate_best_score = None if immediate_best is None else immediate_best["score"]
    future_best_score = None if future_best is None else future_best["score"]
    wait_gain = (
        None
        if immediate_best_score is None or future_best_score is None
        else future_best_score - immediate_best_score
    )
    threshold = float(getattr(config, "WAIT_GAIN_THRESHOLD", 0.0))
    return {
        "wait_gain": wait_gain,
        "immediate_best_score": immediate_best_score,
        "future_best_score": future_best_score,
        "wait_gain_positive": bool(wait_gain is not None and wait_gain > 0.0),
        "no_immediate_action": immediate_best is None,
        "wait_gain_threshold": threshold,
        **stats,
    }


class NetworkEnvironment:
    def __init__(self):
        self.topo_manager = TopologyManager()
        self.pricing_manager = PricingManager(self.topo_manager.graph)

        self.all_nodes = list(self.topo_manager.graph.nodes())
        self.base_stations = [n for n in self.all_nodes if str(n).endswith('0')]
        self.compute_nodes = [n for n in self.all_nodes if not str(n).endswith('0')]
        self.node_regions = {
            node: self.topo_manager.graph.nodes[node].get('region', 'Unknown')
            for node in self.all_nodes
        }
        self.region_queue_norm = max(1.0, config.MAX_QUEUE_LENGTH / max(1, len(self.base_stations)))

        # 初始化节点资源
        self.node_resources = {}
        for node in self.all_nodes:
            if node not in self.base_stations:
                cap = self.topo_manager.graph.nodes[node].get('capacity', config.DEFAULT_NODE_CPU)
                self.node_resources[node] = {'total': float(cap or config.DEFAULT_NODE_CPU), 'used': 0.0}

        self.link_usage = {
            tuple(sorted((u, v))): 0.0
            for u, v in self.topo_manager.graph.edges()
        }
        self.scheduled_allocations = []
        self.scheduled_allocations_by_node = {
            node: []
            for node in self.compute_nodes
        }
        self._scheduled_allocation_index_size = 0

        # 动作空间：计算节点数量 + 1 (WAIT 动作)
        self.action_space_dim = len(self.compute_nodes) + 1
        # 状态空间：任务特征(3) + 时间特征(2) + 节点状态(N*8)
        self.node_feature_dim = 8
        self.state_space_dim = 3 + 2 + len(self.compute_nodes) * self.node_feature_dim
        self.node_to_index = {node: idx for idx, node in enumerate(self.all_nodes)}
        self.compute_node_indices = np.array([self.node_to_index[node] for node in self.compute_nodes], dtype=np.int32)
        self.normalized_adjacency = self._build_normalized_adjacency()

    def _build_normalized_adjacency(self) -> np.ndarray:
        """构造带自环的对称归一化邻接矩阵，供 GCN 使用。"""
        node_count = len(self.all_nodes)
        adjacency = np.zeros((node_count, node_count), dtype=np.float32)
        for u, v in self.topo_manager.graph.edges():
            i, j = self.node_to_index[u], self.node_to_index[v]
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0
        adjacency += np.eye(node_count, dtype=np.float32)
        degree = np.sum(adjacency, axis=1)
        inv_sqrt_degree = np.power(np.maximum(degree, 1e-8), -0.5)
        return (adjacency * inv_sqrt_degree[:, None] * inv_sqrt_degree[None, :]).astype(np.float32)

    @staticmethod
    def _edge_key(u, v):
        return tuple(sorted((u, v)))

    def allocate_path_bandwidth(self, path: list, bw_demand: float):
        """按路径锁定链路带宽资源。"""
        if not path or len(path) < 2 or bw_demand <= 0:
            return
        for i in range(len(path) - 1):
            key = self._edge_key(path[i], path[i + 1])
            self.link_usage[key] = self.link_usage.get(key, 0.0) + bw_demand

    def release_path_bandwidth(self, path: list, bw_demand: float):
        """任务结束后释放链路带宽资源。"""
        if not path or len(path) < 2 or bw_demand <= 0:
            return
        for i in range(len(path) - 1):
            key = self._edge_key(path[i], path[i + 1])
            self.link_usage[key] = max(0.0, self.link_usage.get(key, 0.0) - bw_demand)

    @staticmethod
    def _intervals_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
        return start_a < end_b and start_b < end_a

    def rebuild_scheduled_allocation_index(self):
        self.scheduled_allocations_by_node = {
            node: []
            for node in self.compute_nodes
        }
        for item in self.scheduled_allocations:
            node = item.get('node')
            self.scheduled_allocations_by_node.setdefault(node, []).append(item)
        self._scheduled_allocation_index_size = len(self.scheduled_allocations)

    def _ensure_scheduled_allocation_index(self):
        if self._scheduled_allocation_index_size != len(self.scheduled_allocations):
            self.rebuild_scheduled_allocation_index()

    def add_scheduled_allocation(self, allocation: dict):
        path = allocation.get('path')
        if path and len(path) >= 2 and '_edge_keys' not in allocation:
            allocation['_edge_keys'] = [
                self._edge_key(path[i], path[i + 1])
                for i in range(len(path) - 1)
            ]
        self.scheduled_allocations.append(allocation)
        node = allocation.get('node')
        self.scheduled_allocations_by_node.setdefault(node, []).append(allocation)
        self._scheduled_allocation_index_size = len(self.scheduled_allocations)

    def remove_scheduled_allocation(self, allocation: dict):
        if allocation in self.scheduled_allocations:
            self.scheduled_allocations.remove(allocation)
        node = allocation.get('node')
        node_allocations = self.scheduled_allocations_by_node.get(node)
        if node_allocations is not None and allocation in node_allocations:
            node_allocations.remove(allocation)
        self._scheduled_allocation_index_size = len(self.scheduled_allocations)

    @staticmethod
    def get_task_cpu_time_demand(task: dict) -> float:
        """Return task work in CPU-time units."""
        if not task:
            return 0.0
        cpu = max(0.0, float(task.get('cpu', 0.0)))
        duration = max(0.0, float(task.get('duration', 0.0)))
        if cpu > 0.0 and duration > 0.0:
            return cpu * duration
        return max(0.0, float(task.get('cpu_time', 0.0)))

    @staticmethod
    def get_task_cpu_demand(task: dict) -> float:
        """Return instantaneous CPU occupied by a task while it runs."""
        if not task:
            return 0.0
        return max(0.0, float(task.get('cpu', 0.0)))

    def get_task_cpu_supply(self, node_id, task: dict = None) -> float:
        """Compatibility wrapper: task CPU demand, not node total capacity."""
        return self.get_task_cpu_demand(task)

    def estimate_execution_duration(self, node_id, task: dict) -> float:
        if not task:
            return 0.0
        duration = max(0.0, float(task.get('duration', 0.0)))
        if duration > 0.0:
            return duration
        cpu_time_demand = self.get_task_cpu_time_demand(task)
        cpu_demand = self.get_task_cpu_demand(task)
        if cpu_demand <= 0.0:
            return 0.0
        return cpu_time_demand / cpu_demand

    def get_reserved_cpu_usage(self, node_id, start_time: float, finish_time: float) -> float:
        """统计某个时间区间内已经预约/运行的 CPU 占用。"""
        self._ensure_scheduled_allocation_index()
        used = 0.0
        for item in self.scheduled_allocations_by_node.get(node_id, []):
            if self._intervals_overlap(start_time, finish_time, item['start_time'], item['finish_time']):
                used += item.get('cpu', 0.0)
        return used

    def has_node_capacity_for_interval(self, node_id, cpu_demand: float, start_time: float, finish_time: float) -> bool:
        """判断指定时间区间内节点是否还有足够 CPU 容量。"""
        node_res = self.node_resources[node_id]
        reserved_cpu = self.get_reserved_cpu_usage(node_id, start_time, finish_time)
        return reserved_cpu + cpu_demand <= node_res['total']

    def get_reserved_link_usage(self, start_time: float, finish_time: float) -> dict:
        """统计指定时间区间内已经预约/运行的链路带宽占用。"""
        self._ensure_scheduled_allocation_index()
        usage = {}
        for item in self.scheduled_allocations:
            if not self._intervals_overlap(start_time, finish_time, item['start_time'], item['finish_time']):
                continue
            path = item.get('path')
            bw = item.get('bw', 0.0)
            if not path or len(path) < 2 or bw <= 0:
                continue
            edge_keys = item.get('_edge_keys')
            if edge_keys is None:
                edge_keys = [
                    self._edge_key(path[i], path[i + 1])
                    for i in range(len(path) - 1)
                ]
                item['_edge_keys'] = edge_keys
            for key in edge_keys:
                usage[key] = usage.get(key, 0.0) + bw
        return usage

    def get_dynamic_cpu_price(self, node_id, global_time: float = 0.0, cpu_delta: float = 0.0,
                              interval_start: float = None, interval_end: float = None) -> float:
        """获取节点的实时动态价格；cpu_delta 用于估算候选任务加入后的边际价格。"""
        node_data = self.topo_manager.graph.nodes.get(node_id, {})
        resource_usage = self.node_resources.get(node_id, {'total': 0.0, 'used': 0.0})
        if interval_start is not None and interval_end is not None:
            resource_usage = {
                'total': resource_usage['total'],
                'used': self.get_reserved_cpu_usage(node_id, interval_start, interval_end)
            }
        projected_usage = self.pricing_manager.get_projected_resource_usage(resource_usage, cpu_delta)
        return self.pricing_manager.get_dynamic_price(node_id, node_data, projected_usage, global_time)

    def get_global_state(self, task: dict = None, wait_queue: list = None) -> np.ndarray:
        """构建全局状态向量"""
        if task:
            # 归一化任务特征
            gt = task.get('current_time_context', 0.0)
            queue_delay = max(0.0, gt - task.get('generated_time', gt))
            remaining_start_delay_budget = max(0.0, task['latency_limit'] - queue_delay)
            task_feat = [
                task['cpu'] / 300.0,
                task['data_size'] / 200.0,
                remaining_start_delay_budget / max(task['latency_limit'], 1e-6)
            ]
        else:
            task_feat, gt = [0.0, 0.0, 0.0], 0.0

        # 1. 时间特征（正余弦编码）
        day_prog = (gt % config.TRAFFIC_DAY_DURATION_IN_SIM) / config.TRAFFIC_DAY_DURATION_IN_SIM
        hr = day_prog * 24.0
        time_feat = [math.sin(2 * math.pi * hr / 24.0), math.cos(2 * math.pi * hr / 24.0)]

        # 2. 节点特征
        node_usage = []  # 当前 CPU 利用率
        current_prices = []  # 当前实时价格
        future_prices = []  # 预测价格趋势
        queue_pressure = []  # 节点排队压力
        green_match_ratios = [] # 节点绿电匹配率
        green_absorption_ratios = [] # 节点绿电消纳率


        green_unused_ratios = []
        green_supply_demand_ratios = []

        # 预计算当前队列中各节点的任务分布
        queue_counts = {
            self.node_regions[node]: 0
            for node in self.compute_nodes
        }
        if wait_queue:
            for t in wait_queue:
                source_region = self.node_regions.get(t.get('source_node'))
                if source_region in queue_counts:
                    queue_counts[source_region] += 1

        task_cpu_demand = self.get_task_cpu_demand(task) if task else 0.0
        candidate_duration = self.estimate_execution_duration(None, task) if task else 0.0
        task_finish_time = gt + candidate_duration

        for node in self.compute_nodes:
            # (A) 利用率
            res = self.node_resources[node]
            node_usage.append(res['used'] / res['total'] if res['total'] > 0 else 1.0)

            # (B) 当前价格
            p_now = self.get_dynamic_cpu_price(node, gt, cpu_delta=task_cpu_demand)
            current_prices.append(min(1.0, p_now / config.PRICE_NORMALIZATION_FACTOR))

            # (C) 电价趋势 (预测 1 小时/单位模拟时间后的价格)
            future_gt = gt + (config.TRAFFIC_DAY_DURATION_IN_SIM / 24.0)
            p_future = self.get_dynamic_cpu_price(node, future_gt, cpu_delta=task_cpu_demand)
            future_prices.append(min(1.0, p_future / config.PRICE_NORMALIZATION_FACTOR))

            # (D) 节点排队任务数（反映潜在竞争）
            q_norm = min(1.0, queue_counts[self.node_regions[node]] / self.region_queue_norm)
            queue_pressure.append(q_norm)

            reserved_cpu = (
                self.get_reserved_cpu_usage(node, gt, task_finish_time)
                if task else res['used']
            )
            projected_usage = {
                "total": res["total"],
                "used": min(res["total"], reserved_cpu + task_cpu_demand),
            }
            power_profile = self.pricing_manager.get_node_power_profile(
                node,
                projected_usage,
                gt,
            )
            power_demand = power_profile["power_demand_mw"]
            green_supply = power_profile["green_supply_mw"]
            green_unused = power_profile["green_unused_mw"]
            green_unused_ratio = (
                0.0 if green_supply <= 0.0
                else green_unused / green_supply
            )
            green_supply_demand_ratio = (
                0.0 if power_demand <= 0.0
                else min(2.0, green_supply / power_demand) / 2.0
            )
            green_match_ratios.append(power_profile["green_match_ratio"])
            green_absorption_ratios.append(power_profile["green_absorption_ratio"])
            green_unused_ratios.append(green_unused_ratio)
            green_supply_demand_ratios.append(green_supply_demand_ratio)

        # 拼接最终状态向量
        return np.concatenate((
            task_feat,  # 3
            time_feat,  # 2
            node_usage,  # N
            current_prices,  # N
            future_prices,  # N
            queue_pressure,  # N
            green_match_ratios,  # N
            green_absorption_ratios,  # N
            green_unused_ratios,  # N
            green_supply_demand_ratios,  # N
        )).astype(np.float32)

    def _get_sla_one_hot(self, sla_type: str):
        if sla_type == 'Hard':
            return [1.0, 0.0, 0.0]
        if sla_type == 'Flexible':
            return [0.0, 0.0, 1.0]
        return [0.0, 1.0, 0.0]

    def get_graph_state(self, task: dict = None, wait_queue: list = None) -> dict:
        """构建 GNN-RL 使用的图状态。

        返回内容:
        - node_features: 每个节点一行特征，包含节点资源、电价、绿电、任务和源节点信息。
        - adjacency: 带自环的归一化邻接矩阵。
        - compute_node_indices: 计算节点在 node_features 中的行号，用于映射动作。
        """
        gt = task.get('current_time_context', 0.0) if task else 0.0
        queue_delay = max(0.0, gt - task.get('generated_time', gt)) if task else 0.0
        remaining_budget = max(0.0, task['latency_limit'] - queue_delay) if task else 0.0
        remaining_ratio = remaining_budget / max(task.get('latency_limit', 1.0), 1e-6) if task else 0.0
        task_cpu = task.get('cpu', 0.0) if task else 0.0
        task_data = task.get('data_size', 0.0) if task else 0.0
        task_bw = task.get('bw', 0.0) if task else 0.0
        task_duration = task.get('duration', 0.0) if task else 0.0
        task_cpu_demand = self.get_task_cpu_demand(task) if task else 0.0
        task_execution_duration = self.estimate_execution_duration(None, task) if task else 0.0
        task_finish_time = gt + task_execution_duration
        sla_hard, sla_soft, sla_flexible = self._get_sla_one_hot(
            task.get('sla_type', 'Soft') if task else 'Soft'
        )

        day_prog = (gt % config.TRAFFIC_DAY_DURATION_IN_SIM) / config.TRAFFIC_DAY_DURATION_IN_SIM
        hr = day_prog * 24.0
        time_sin = math.sin(2 * math.pi * hr / 24.0)
        time_cos = math.cos(2 * math.pi * hr / 24.0)

        queue_counts = {self.node_regions[node]: 0 for node in self.compute_nodes}
        if wait_queue:
            for queued_task in wait_queue:
                source_region = self.node_regions.get(queued_task.get('source_node'))
                if source_region in queue_counts:
                    queue_counts[source_region] += 1

        source_node = task.get('source_node') if task else None
        max_capacity = max((res['total'] for res in self.node_resources.values()), default=1.0)
        max_tier = 3.0
        feature_rows = []

        for node in self.all_nodes:
            node_data = self.topo_manager.graph.nodes[node]
            is_compute = 1.0 if node in self.node_resources else 0.0
            is_base = 1.0 - is_compute
            tier_norm = node_data.get('tier', 2) / max_tier
            same_region = 1.0 if source_node and self.node_regions.get(node) == self.node_regions.get(source_node) else 0.0
            source_flag = 1.0 if node == source_node else 0.0

            if is_compute:
                res = self.node_resources[node]
                current_usage = res['used'] / res['total'] if res['total'] > 0 else 1.0
                candidate_duration = task_execution_duration
                reserved_cpu = self.get_reserved_cpu_usage(node, gt, task_finish_time) if task else res['used']
                reserved_usage = reserved_cpu / res['total'] if res['total'] > 0 else 1.0
                current_price = self.get_dynamic_cpu_price(node, gt, cpu_delta=task_cpu_demand)
                future_time = gt + (config.TRAFFIC_DAY_DURATION_IN_SIM / 24.0)
                future_price = self.get_dynamic_cpu_price(node, future_time, cpu_delta=task_cpu_demand)
                projected_usage = {
                    'total': res['total'],
                    'used': min(res['total'], reserved_cpu + task_cpu_demand)
                }
                power_profile = self.pricing_manager.get_node_power_profile(node, projected_usage, gt)
                green_match = power_profile["green_match_ratio"]
                queue_pressure = min(1.0, queue_counts[self.node_regions[node]] / self.region_queue_norm)
                capacity_norm = res['total'] / max_capacity
            else:
                candidate_duration = task_duration
                current_usage = reserved_usage = 0.0
                current_price = future_price = 0.0
                green_match = 0.0
                queue_pressure = 0.0
                capacity_norm = 0.0

            if source_node and node != source_node:
                path = self.topo_manager.find_path(source_node, node)
                route_delay = self.topo_manager.calculate_transmission_delay(
                    path,
                    task_data,
                    task_bw,
                ) if path else 1.0
            else:
                route_delay = 0.0

            feature_rows.append([
                is_compute,
                is_base,
                current_usage,
                reserved_usage,
                min(1.0, current_price / config.PRICE_NORMALIZATION_FACTOR),
                min(1.0, future_price / config.PRICE_NORMALIZATION_FACTOR),
                green_match,
                tier_norm,
                queue_pressure,
                source_flag,
                same_region,
                min(1.0, route_delay / max(task.get('latency_limit', 1.0), 1e-6)) if task else 0.0,
                task_cpu / 1200.0,
                task_data / 1000.0,
                task_bw / 200.0,
                candidate_duration / 200.0,
                remaining_ratio,
                sla_hard,
                sla_soft,
                sla_flexible,
                capacity_norm
            ])

        return {
            "node_features": np.asarray(feature_rows, dtype=np.float32),
            "adjacency": self.normalized_adjacency,
            "compute_node_indices": self.compute_node_indices,
            "wait_action_index": len(self.compute_nodes)
        }

    @staticmethod
    def _reward_components(**updates):
        components = {
            "R_latency": 0.0,
            "R_cost": 0.0,
            "R_green": 0.0,
            "R_balance": 0.0,
            "R_success": 0.0,
            "R_cost_spike": 0.0,
        }
        components.update({k: float(v) for k, v in updates.items()})
        return components

    @staticmethod
    def _reward_total(reward_components):
        return float(sum(reward_components.values()))

    @staticmethod
    def _constraint_costs(**updates):
        costs = {
            "sla_violation": 0.0,
            "drop": 0.0,
            "cost_over_budget": 0.0,
            "overload": 0.0
        }
        costs.update({k: float(v) for k, v in updates.items()})
        return costs

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    def _compute_system_green_reward(
        self,
        system_before: dict,
        system_after: dict,
        local_green_match_ratio: float
    ) -> tuple:
        absorption_before = float(system_before.get("system_green_absorption_ratio", 0.0))
        absorption_after = float(system_after.get("system_green_absorption_ratio", 0.0))
        absorption_delta = absorption_after - absorption_before

        delta_clip = getattr(config, "GREEN_ABSORPTION_DELTA_CLIP", 0.05)
        absorption_delta_clipped = self._clip(absorption_delta, -delta_clip, delta_clip)
        green_unused_ratio_after = float(system_after.get("green_unused_ratio", 0.0))
        green_load_coverage_after = float(system_after.get("green_load_coverage_ratio", 0.0))

        r_green_match = (
            getattr(config, "GREEN_MATCH_REWARD_SCALE", 3.0)
            * float(local_green_match_ratio)
        )
        if getattr(config, "ENABLE_SYSTEM_GREEN_REWARD", True):
            r_green_absorption_delta = (
                getattr(config, "GREEN_ABSORPTION_DELTA_REWARD_SCALE", 120.0)
                * absorption_delta_clipped
            )
            r_green_waste = (
                -getattr(config, "GREEN_WASTE_PENALTY_SCALE", 3.0)
                * green_unused_ratio_after
            )
            r_green_load_coverage = (
                getattr(config, "GREEN_LOAD_COVERAGE_REWARD_SCALE", 3.0)
                * green_load_coverage_after
            )
        else:
            r_green_absorption_delta = 0.0
            r_green_waste = 0.0
            r_green_load_coverage = 0.0

        r_green_total = (
            r_green_match
            + r_green_absorption_delta
            + r_green_waste
            + r_green_load_coverage
        )
        detail = {
            "system_absorption_before": absorption_before,
            "system_absorption_after": absorption_after,
            "system_absorption_delta": absorption_delta,
            "system_absorption_delta_clipped": absorption_delta_clipped,
            "green_unused_ratio_after": green_unused_ratio_after,
            "green_load_coverage_after": green_load_coverage_after,
            "R_green_match": r_green_match,
            "R_green_absorption_delta": r_green_absorption_delta,
            "R_green_waste": r_green_waste,
            "R_green_load_coverage": r_green_load_coverage,
            "R_green_total": r_green_total,
        }
        return r_green_total, detail

    def _compute_cost_spike_penalty(
        self,
        raw_cost: float,
        baseline_cost: float,
        cpu_time_demand: float
    ) -> tuple:
        cost_per_cpu_time = raw_cost / max(cpu_time_demand, 1e-8)
        cost_ratio = 0.0 if baseline_cost <= 0.0 else raw_cost / baseline_cost
        if not getattr(config, "ENABLE_COST_SPIKE_PENALTY", True):
            return 0.0, {
                "cost_per_cpu_time": cost_per_cpu_time,
                "cost_ratio": cost_ratio,
                "cost_cpu_excess": 0.0,
                "cost_ratio_excess": 0.0,
                "R_cost_spike": 0.0,
            }

        cpu_threshold = getattr(config, "COST_PER_CPU_TIME_THRESHOLD", 0.0137)
        ratio_threshold = getattr(config, "COST_RATIO_SPIKE_THRESHOLD", 0.85)
        cpu_excess = max(
            0.0,
            (cost_per_cpu_time - cpu_threshold) / max(cpu_threshold, 1e-8),
        )
        ratio_excess = max(0.0, cost_ratio - ratio_threshold)
        raw_penalty = -getattr(config, "COST_SPIKE_PENALTY_SCALE", 8.0) * (
            0.5 * cpu_excess + 0.5 * ratio_excess
        )
        clip_value = getattr(config, "COST_SPIKE_PENALTY_CLIP", 10.0)
        penalty = max(-clip_value, raw_penalty)
        detail = {
            "cost_per_cpu_time": cost_per_cpu_time,
            "cost_ratio": cost_ratio,
            "cost_cpu_excess": cpu_excess,
            "cost_ratio_excess": ratio_excess,
            "R_cost_spike": penalty,
        }
        return penalty, detail

    def _step_info(self, status: str, reward_components=None, constraint_costs=None, **extra):
        info = {
            "status": status,
            "reward_components": reward_components or self._reward_components(),
            "constraint_costs": constraint_costs or self._constraint_costs()
        }
        info.update(extra)
        return info

    def _build_wait_detail(self, task: dict, wait_queue: list, next_queue_delay: float, sla_type: str, candidates=None):
        current_schedule_time = task.get('current_time_context', 0.0)
        wait_detail = {
            "wait_type": "queue_wait",
        }
        wait_detail.update(estimate_wait_penalty_detail(
            self,
            task,
            wait_queue,
            next_queue_delay,
            sla_type,
        ))
        wait_detail.update(estimate_wait_opportunity(
            self,
            task,
            wait_queue,
            current_schedule_time,
            candidates=candidates,
        ))
        threshold = float(getattr(config, "WAIT_GAIN_THRESHOLD", 0.0))
        penalty_weight = float(getattr(config, "WAIT_GAIN_PENALTY_WEIGHT", 1.0))
        wait_gain = wait_detail.get("wait_gain")
        wait_detail.update({
            "wait_gain_threshold": threshold,
            "wait_net_gain": None if wait_gain is None else (
                wait_gain + penalty_weight * wait_detail.get("wait_penalty", 0.0) - threshold
            ),
        })
        return wait_detail

    def step(self, action_index: int, task: dict, wait_queue: list = None, candidates=None):
        """执行调度动作并返回奖励 (深度重构的异构 SLA 奖励机制)"""
        current_schedule_time = task.get('current_time_context', 0.0)
        queue_delay = max(0.0, current_schedule_time - task['generated_time'])

        # 提取 SLA 类型，兼容老版本数据
        sla_type = task.get('sla_type', 'Soft')

        # --- 动作：WAIT (推迟调度) ---
        if action_index == len(self.compute_nodes):
            next_queue_delay = queue_delay + config.SCHEDULING_CYCLE
            wait_detail = self._build_wait_detail(task, wait_queue, next_queue_delay, sla_type, candidates=candidates)
            if next_queue_delay >= task['latency_limit']:
                timeout_penalty = -50.0 if sla_type == 'Hard' else -10.0
                reward_components = self._reward_components(R_latency=timeout_penalty, R_success=-5.0)
                reward = self._reward_total(reward_components)
                return self.get_global_state(task, wait_queue), reward, False, self._step_info(
                    "Timeout",
                    reward_components=reward_components,
                    constraint_costs=self._constraint_costs(sla_violation=1.0, drop=1.0),
                    wait_detail=wait_detail,
                    delays={
                        "network": 0.0,
                        "queue": next_queue_delay,
                        "execution": 0.0,
                        "price_wait": 0.0,
                        "physical": 0.0,
                        "end_to_end": next_queue_delay,
                        "end_to_end_delay": next_queue_delay,
                        "completion_delay": next_queue_delay,
                    },
                )

            deferred_task = dict(task)
            deferred_task['current_time_context'] = current_schedule_time + config.SCHEDULING_CYCLE
            future_queue = list(wait_queue) if wait_queue else []
            future_queue.append(deferred_task)

            wait_penalty = wait_detail["wait_penalty"]
            reward_components = self._reward_components(R_latency=wait_penalty)
            return self.get_global_state(deferred_task, future_queue), wait_penalty, False, self._step_info(
                "Deferred",
                reward_components=reward_components,
                deferred_task=deferred_task,
                wait_detail=wait_detail
            )

        # --- 动作：映射到具体计算节点 ---
        target_node = self.compute_nodes[action_index]
        source_node = task['source_node']
        node_res = self.node_resources[target_node]
        cpu_time_demand = self.get_task_cpu_time_demand(task)
        cpu_demand = self.get_task_cpu_demand(task)
        execution_duration = self.estimate_execution_duration(target_node, task)

        # 1. 物理约束检查：任务必须有正的 CPU-time 需求和瞬时 CPU 占用
        if cpu_time_demand <= 0.0 or cpu_demand <= 0.0:
            reward_components = self._reward_components(R_success=-10.0)
            return self.get_global_state(task, wait_queue), -10.0, False, self._step_info(
                "CPU Full",
                reward_components=reward_components,
                constraint_costs=self._constraint_costs(drop=1.0, overload=1.0)
            )

        # 2. 路由约束检查：是否存在拓扑路径；具体带宽容量在预约执行区间内再检查
        path = [target_node] if source_node == target_node else \
            self.topo_manager.find_path(source_node, target_node)

        if path is None:
            reward_components = self._reward_components(R_success=-10.0)
            return self.get_global_state(task, wait_queue), -10.0, False, self._step_info(
                "No Path",
                reward_components=reward_components,
                constraint_costs=self._constraint_costs(drop=1.0)
            )

        # 3. SLA checks queue/wait/network delay before execution starts; execution duration is separate.
        # SLA delay before execution starts: queue_delay + price_wait + network_delay.
        chosen_candidate = None
        if candidates is not None:
            current_candidates = [
                candidate for candidate in candidates
                if candidate["node"] == target_node
                and abs(candidate["schedule_time"] - current_schedule_time) <= 1e-9
            ]
            if current_candidates:
                chosen_candidate = max(current_candidates, key=lambda candidate: candidate["score"])
                path = chosen_candidate["path"]

        trans_delay = self.topo_manager.calculate_transmission_delay(
            path,
            task['data_size'],
            task.get('bw', 0.0),
        )
        network_delay = trans_delay
        start_delay_without_price_wait = queue_delay + network_delay

        # ====================================================================
        # 【致命惩罚重塑】 超时判定
        # ====================================================================
        if start_delay_without_price_wait > task['latency_limit']:
            if sla_type == 'Hard':
                timeout_penalty = -50.0  # 一票否决：致死级故障
            elif sla_type == 'Soft':
                timeout_penalty = -10.0  # 用户体验劣化
            else:  # Flexible
                timeout_penalty = -15.0  # 极长宽限期还能超时，调度算法极其低劣
            reward_components = self._reward_components(R_latency=timeout_penalty, R_success=-5.0)
            reward = self._reward_total(reward_components)
            return self.get_global_state(task, wait_queue), reward, False, self._step_info(
                "Timeout",
                reward_components=reward_components,
                constraint_costs=self._constraint_costs(sla_violation=1.0, drop=1.0),
                delays={
                    "network": network_delay,
                    "queue": queue_delay,
                    "execution": execution_duration,
                    "price_wait": 0.0,
                    "physical": network_delay,
                    "end_to_end": start_delay_without_price_wait,
                    "end_to_end_delay": start_delay_without_price_wait,
                    "completion_delay": start_delay_without_price_wait + execution_duration,
                },
            )

        # ====================================================================
        # --- 成功调度：异步计费与正向多维激励 ---
        # ====================================================================

        # 1. 异步计费雷达
        remaining_sla_time = max(0.0, task['latency_limit'] - start_delay_without_price_wait)
        max_wait_time = 0.0
        min_price = 999.0
        search_time = current_schedule_time
        step_size = config.PRICE_LOOKAHEAD_STEP
        best_execute_time = None

        if chosen_candidate is not None:
            min_price = chosen_candidate["price"]
            best_execute_time = chosen_candidate["schedule_time"]
            finish_time = chosen_candidate["finish_time"]
            network_delay = chosen_candidate["network_delay"]
            start_delay_without_price_wait = queue_delay + network_delay

        while chosen_candidate is None and search_time <= current_schedule_time + max_wait_time:
            candidate_finish_time = search_time + execution_duration
            if not self.has_node_capacity_for_interval(target_node, cpu_demand, search_time, candidate_finish_time):
                search_time += step_size
                continue
            p = self.get_dynamic_cpu_price(
                target_node,
                search_time,
                cpu_delta=cpu_demand,
                interval_start=search_time,
                interval_end=candidate_finish_time
            )
            if p < min_price:
                min_price = p
                best_execute_time = search_time
            search_time += step_size

        if best_execute_time is None:
            reward_components = self._reward_components(R_success=-10.0)
            return self.get_global_state(task, wait_queue), -10.0, False, self._step_info(
                "Reserved_CPU Full",
                reward_components=reward_components,
                constraint_costs=self._constraint_costs(drop=1.0, overload=1.0)
            )

        finish_time = best_execute_time + execution_duration
        interval_link_usage = self.get_reserved_link_usage(best_execute_time, finish_time)
        scheduled_path = path if chosen_candidate is not None else \
            ([target_node] if source_node == target_node else
            self.topo_manager.find_path(
                source_node,
                target_node,
                bw_demand=task.get('bw', 0.0),
                link_usage=interval_link_usage
            ))
        if scheduled_path is None:
            reward_components = self._reward_components(R_success=-10.0)
            return self.get_global_state(task, wait_queue), -10.0, False, self._step_info(
                "Reserved_BW Full",
                reward_components=reward_components,
                constraint_costs=self._constraint_costs(drop=1.0)
            )

        if scheduled_path != path:
            path = scheduled_path
            trans_delay = self.topo_manager.calculate_transmission_delay(
                path,
                task['data_size'],
                task.get('bw', 0.0),
            )
            network_delay = trans_delay
            actual_wait_for_price_time = max(0.0, best_execute_time - current_schedule_time)
            start_delay_without_price_wait = queue_delay + network_delay
            start_delay = start_delay_without_price_wait + actual_wait_for_price_time
            if start_delay > task['latency_limit']:
                reward_components = self._reward_components(R_latency=-10.0, R_success=-5.0)
                reward = self._reward_total(reward_components)
                return self.get_global_state(task, wait_queue), reward, False, self._step_info(
                    "Timeout",
                    reward_components=reward_components,
                    constraint_costs=self._constraint_costs(sla_violation=1.0, drop=1.0),
                    delays={
                        "network": network_delay,
                        "queue": queue_delay,
                        "execution": execution_duration,
                        "price_wait": actual_wait_for_price_time,
                        "physical": network_delay,
                        "end_to_end": start_delay,
                        "end_to_end_delay": start_delay,
                        "completion_delay": start_delay + execution_duration,
                    },
                )

        raw_cost = min_price * cpu_time_demand

        # --------------------------------------------------------------------
        # 2. 异构效用函数计算 (Heterogeneous Utility)

        # 成本节省率打分：以最贵档位全价(1.2 * 1.5近似最高峰)为基准，计算省了多少钱
        baseline_price = config.BASELINE_ELECTRICITY_PRICE_YUAN_PER_MW * config.CPU_POWER_UNIT_MW
        baseline_cost = baseline_price * cpu_time_demand
        cost_saving_ratio = 0.0 if baseline_cost <= 0 else min(1.0, max(0.0, (baseline_cost - raw_cost) / baseline_cost))
        cost_score = 20.0 * cost_saving_ratio  # 最高 20 分
        r_cost_spike, cost_spike_detail = self._compute_cost_spike_penalty(
            raw_cost=raw_cost,
            baseline_cost=baseline_cost,
            cpu_time_demand=cpu_time_demand,
        )

        projected_resource_usage = {
            'total': node_res['total'],
            'used': self.get_reserved_cpu_usage(target_node, best_execute_time, finish_time) + cpu_demand
        }
        power_profile = self.pricing_manager.get_node_power_profile(
            target_node,
            projected_resource_usage,
            best_execute_time
        )
        system_green_before = self.get_system_green_absorption(best_execute_time)
        system_green_after = self.get_system_green_absorption(
            best_execute_time,
            override_node=target_node,
            override_resource_usage=projected_resource_usage,
        )
        r_green, green_reward_detail = self._compute_system_green_reward(
            system_before=system_green_before,
            system_after=system_green_after,
            local_green_match_ratio=power_profile["green_match_ratio"],
        )

        # 时延余量打分
        actual_wait_for_price_time = max(0.0, best_execute_time - current_schedule_time)
        start_delay = queue_delay + actual_wait_for_price_time + network_delay
        if start_delay > task['latency_limit']:
            reward_components = self._reward_components(R_latency=-10.0, R_success=-5.0)
            reward = self._reward_total(reward_components)
            return self.get_global_state(task, wait_queue), reward, False, self._step_info(
                "Timeout",
                reward_components=reward_components,
                constraint_costs=self._constraint_costs(sla_violation=1.0, drop=1.0),
                delays={
                    "network": network_delay,
                    "queue": queue_delay,
                    "execution": execution_duration,
                    "price_wait": actual_wait_for_price_time,
                    "physical": network_delay,
                    "end_to_end": start_delay,
                    "end_to_end_delay": start_delay,
                    "completion_delay": start_delay + execution_duration,
                },
            )

        delay_margin_ratio = max(0.0, (task['latency_limit'] - start_delay) / task['latency_limit'])
        latency_score = 20.0 * delay_margin_ratio  # 最高 20

        projected_usage_ratio = (
            projected_resource_usage['used'] / projected_resource_usage['total']
            if projected_resource_usage['total'] > 0 else 1.0
        )
        # 核心：按业务 画像完全切分评估标准
        use_rebalanced_success = getattr(config, "ENABLE_REBALANCED_SUCCESS_REWARD", False)
        if sla_type == 'Hard':
            r_success = (
                getattr(config, "REBALANCED_SUCCESS_REWARD_HARD", 10.0)
                if use_rebalanced_success
                else getattr(config, "SUCCESS_REWARD_HARD", 15.0)
            )
            r_latency = 1.0 * latency_score
            r_cost = 0.0 * cost_score
        elif sla_type == 'Flexible':
            r_success = (
                getattr(config, "REBALANCED_SUCCESS_REWARD_FLEXIBLE", 5.0)
                if use_rebalanced_success
                else getattr(config, "SUCCESS_REWARD_FLEXIBLE", 10.0)
            )
            r_latency = 0.0 * latency_score
            r_cost = 1.5 * cost_score
        else:  # Soft SLA
            urgency_weight = math.exp(-task['latency_limit'])
            cost_weight = 1.0 - urgency_weight
            r_success = (
                getattr(config, "REBALANCED_SUCCESS_REWARD_SOFT", 7.0)
                if use_rebalanced_success
                else getattr(config, "SUCCESS_REWARD_SOFT", 10.0)
            )
            r_latency = urgency_weight * latency_score
            r_cost = cost_weight * cost_score

        # --------------------------------------------------------------------
        # 3. 连续指数级防拥塞惩罚 (Anti-Herd Effect)
        usage_ratio = projected_usage_ratio
        r_balance = 0.0
        if usage_ratio > 0.85:
            # 0.85时扣~1分，0.9时扣~4.5分，0.95时扣~12分，逼近1.0时扣~20分
            r_balance = -(math.exp(10 * (usage_ratio - 0.85)) - 1)

        reward_components = self._reward_components(
            R_latency=r_latency,
            R_cost=r_cost,
            R_green=r_green,
            R_balance=r_balance,
            R_success=r_success,
            R_cost_spike=r_cost_spike,
        )
        reward = sum(reward_components.values())

        budget_ratio = getattr(config, 'CONSTRAINT_COST_BUDGET_RATIO', 0.85)
        cost_ratio = cost_spike_detail["cost_ratio"]
        cost_per_cpu_time = cost_spike_detail["cost_per_cpu_time"]
        baseline_cost_per_cpu_time = baseline_price
        cost_over_budget = max(0.0, cost_ratio - budget_ratio)
        overload_threshold = getattr(config, 'CONSTRAINT_OVERLOAD_THRESHOLD', 0.85)
        overload = max(0.0, (projected_usage_ratio - overload_threshold) / max(1e-6, 1.0 - overload_threshold))
        constraint_costs = self._constraint_costs(
            cost_over_budget=cost_over_budget,
            overload=overload
        )

        # ====================================================================

        e2e_latency = start_delay

        return self.get_global_state(task, wait_queue), reward, False, self._step_info(
            "Success",
            reward_components=reward_components,
            constraint_costs=constraint_costs,
            path=path,
            cost=raw_cost,
            coordination={
                "green_match_ratio": power_profile["green_match_ratio"],
                "green_absorption_ratio": power_profile["green_absorption_ratio"],
                "cost_saving_ratio": cost_saving_ratio,
                "power_demand_mw": power_profile["power_demand_mw"],
                "green_supply_mw": power_profile["green_supply_mw"],
                "green_used_mw": power_profile["green_used_mw"],
                "green_unused_mw": power_profile["green_unused_mw"],
                "baseline_cost": baseline_cost,
                "raw_cost": raw_cost,
                "cost_ratio": cost_ratio,
                "cost_per_cpu_time": cost_per_cpu_time,
                "baseline_cost_per_cpu_time": baseline_cost_per_cpu_time,
                "cost_cpu_excess": cost_spike_detail["cost_cpu_excess"],
                "cost_ratio_excess": cost_spike_detail["cost_ratio_excess"],
                "R_cost_spike": cost_spike_detail["R_cost_spike"],
                "system_absorption_before": green_reward_detail["system_absorption_before"],
                "system_absorption_after": green_reward_detail["system_absorption_after"],
                "system_absorption_delta": green_reward_detail["system_absorption_delta"],
                "system_absorption_delta_clipped": green_reward_detail["system_absorption_delta_clipped"],
                "green_unused_ratio_after": green_reward_detail["green_unused_ratio_after"],
                "green_load_coverage_after": green_reward_detail["green_load_coverage_after"],
                "R_green_match": green_reward_detail["R_green_match"],
                "R_green_absorption_delta": green_reward_detail["R_green_absorption_delta"],
                "R_green_waste": green_reward_detail["R_green_waste"],
                "R_green_load_coverage": green_reward_detail["R_green_load_coverage"],
                "R_green_total": green_reward_detail["R_green_total"],
            },
            delays={
                "network": network_delay,
                "queue": queue_delay,
                "execution": execution_duration,
                "price_wait": actual_wait_for_price_time,
                "physical": network_delay,
                "end_to_end": e2e_latency,
                "end_to_end_delay": e2e_latency,
                "completion_delay": e2e_latency + execution_duration,
            },
            target_node=target_node,
            cpu_demand=cpu_demand,
            cpu_supply=cpu_demand,
            cpu_time=cpu_time_demand,
            cpu_time_demand=cpu_time_demand,
            execution_duration=execution_duration,
            execute_time=best_execute_time,
            finish_time=finish_time
        )

    def get_region_cpu_usage(self):
        """统计各地区的 CPU 平均利用率"""
        res_map = {}
        for node in self.compute_nodes:
            reg = self.topo_manager.graph.nodes[node].get('region', 'Unknown')
            if reg not in res_map: res_map[reg] = [0.0, 0.0]
            res_map[reg][0] += self.node_resources[node]['used']
            res_map[reg][1] += self.node_resources[node]['total']
        return {k: v[0] / v[1] if v[1] > 0 else 0 for k, v in res_map.items()}

    def get_system_green_absorption(
        self,
        global_time: float = 0.0,
        override_node=None,
        override_resource_usage: dict = None
    ) -> dict:
        total_green_used = 0.0
        total_green_supply = 0.0
        total_power_demand = 0.0

        for node in self.compute_nodes:
            if (
                override_node is not None
                and node == override_node
                and override_resource_usage is not None
            ):
                resource_usage = override_resource_usage
            else:
                resource_usage = self.node_resources[node]
            power_profile = self.pricing_manager.get_node_power_profile(
                node,
                resource_usage,
                global_time
            )
            power_demand = power_profile["power_demand_mw"]
            green_supply = power_profile["green_supply_mw"]
            green_used = min(power_demand, green_supply)

            total_power_demand += power_demand
            total_green_supply += green_supply
            total_green_used += green_used

        absorption_ratio = (
            0.0 if total_green_supply <= 0.0
            else total_green_used / total_green_supply
        )
        total_green_unused = max(0.0, total_green_supply - total_green_used)
        green_unused_ratio = (
            0.0 if total_green_supply <= 0.0
            else total_green_unused / total_green_supply
        )
        green_load_coverage_ratio = (
            0.0 if total_power_demand <= 0.0
            else min(1.0, total_green_used / total_power_demand)
        )
        green_supply_demand_ratio = (
            0.0 if total_power_demand <= 0.0
            else total_green_supply / total_power_demand
        )

        return {
            "system_green_absorption_ratio": absorption_ratio,
            "total_green_used_mw": total_green_used,
            "total_green_supply_mw": total_green_supply,
            "total_power_demand_mw": total_power_demand,
            "total_green_unused_mw": total_green_unused,
            "green_unused_ratio": green_unused_ratio,
            "green_load_coverage_ratio": green_load_coverage_ratio,
            "green_supply_demand_ratio": green_supply_demand_ratio,
        }

    def get_tier_cpu_usage(self):
        """统计各电价档位的 CPU 平均利用率"""
        res_map = {1: [0.0, 0.0], 2: [0.0, 0.0], 3: [0.0, 0.0]}
        for node in self.compute_nodes:
            t = self.topo_manager.graph.nodes[node].get('tier', 2)
            res_map[t][0] += self.node_resources[node]['used']
            res_map[t][1] += self.node_resources[node]['total']
        return {k: v[0] / v[1] if v[1] > 0 else 0 for k, v in res_map.items()}
