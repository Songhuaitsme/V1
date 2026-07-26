import networkx as nx
from shared import config
from shared.data_loader import DataLoader
from v1.domain.units import DataUnitConverter, TimeConverter, positive_finite
from v1.scheduler.transmission import build_path_spec


class TopologyManager:
    def __init__(self):
        # 修复导入错误：调用类方法加载图
        self.graph = DataLoader.load_network_topology()
        self._initialize_attributes()
        self._build_edge_capacity_index()
        self._shortest_path_cache = {}
        self.time_converter = TimeConverter.from_traffic_day_duration(
            config.TRAFFIC_DAY_DURATION_IN_SIM,
            getattr(config, "SIM_SECONDS_PER_UNIT", None),
        )

    def _initialize_attributes(self):
        """初始化边属性，确保流控和容量字段存在"""
        for u, v, data in self.graph.edges(data=True):
            if 'capacity' not in data: self.graph[u][v]['capacity'] = config.DEFAULT_LINK_BANDWIDTH
            if 'distance_km' not in data:
                default_distance = (
                    config.LOCAL_LINK_DISTANCE_KM
                    if data.get('edge_type') == 'local'
                    else config.BACKBONE_DISTANCE_KM_BY_TIER_GAP.get(1, 900.0)
                )
                self.graph[u][v]['distance_km'] = default_distance
            distance_km = self.graph[u][v]['distance_km']
            self.graph[u][v]['prop_delay'] = distance_km / config.FIBER_PROPAGATION_SPEED_KM_PER_S
            if 'cost' not in data:
                self.graph[u][v]['cost'] = self.graph[u][v]['prop_delay'] * 1000

    @staticmethod
    def _edge_key(u, v):
        return tuple(sorted((u, v)))

    def _build_edge_capacity_index(self):
        self._edge_capacity_by_key = {}
        self._edge_endpoints_by_key = {}
        capacities = []
        for u, v, data in self.graph.edges(data=True):
            edge_key = self._edge_key(u, v)
            capacity = data.get('capacity', config.DEFAULT_LINK_BANDWIDTH)
            self._edge_capacity_by_key[edge_key] = capacity
            self._edge_endpoints_by_key[edge_key] = (u, v)
            capacities.append(capacity)
        self._min_edge_capacity = min(capacities) if capacities else config.DEFAULT_LINK_BANDWIDTH

    def find_path(self, source_node, target_node, bw_demand: float = 0.0, link_usage: dict = None) -> list:
        """寻找满足带宽约束的最短时延路径。"""
        try:
            cache_key = (source_node, target_node)
            if bw_demand <= 0:
                if cache_key not in self._shortest_path_cache:
                    self._shortest_path_cache[cache_key] = nx.shortest_path(
                        self.graph,
                        source=source_node,
                        target=target_node,
                        weight='cost',
                    )
                return list(self._shortest_path_cache[cache_key])

            blocked_edges = []
            current_usage = link_usage or {}
            for edge_key, capacity in self._edge_capacity_by_key.items():
                used_bw = current_usage.get(edge_key, 0.0)
                if used_bw + bw_demand > capacity:
                    endpoints = self._edge_endpoints_by_key.get(edge_key)
                    if endpoints:
                        blocked_edges.append(endpoints)

            if not blocked_edges:
                if cache_key not in self._shortest_path_cache:
                    self._shortest_path_cache[cache_key] = nx.shortest_path(
                        self.graph,
                        source=source_node,
                        target=target_node,
                        weight='cost',
                    )
                return list(self._shortest_path_cache[cache_key])

            feasible_graph = self.graph.copy()
            feasible_graph.remove_edges_from(blocked_edges)
            return nx.shortest_path(feasible_graph, source=source_node, target=target_node, weight='cost')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None


    def calculate_transmission_delay(
        self,
        path: list,
        data_size: float,
        bandwidth_demand: float = None,
    ) -> float:
        """
        返回v1.0仿真时间单位下的固定带宽流水线传输时长。

        数据序列化只计算一次：8*MB/任务预留Mbps；传播时延逐跳累加。
        ``bandwidth_demand``省略时仅为legacy调用回退到路径瓶颈容量。
        """
        if not path or len(path) < 2:
            return 0.0
        path_spec = build_path_spec(self.graph, path)
        reserved_bandwidth = (
            path_spec.static_bottleneck_mbps
            if bandwidth_demand is None
            else positive_finite("bandwidth_demand", bandwidth_demand)
        )
        if reserved_bandwidth > path_spec.static_bottleneck_mbps + 1e-12:
            raise ValueError("bandwidth_demand exceeds static path bottleneck")
        data_seconds = (
            DataUnitConverter.decimal_mb_to_megabits(data_size)
            / reserved_bandwidth
        )
        propagation_seconds = (
            path_spec.total_distance_km
            / config.FIBER_PROPAGATION_SPEED_KM_PER_S
        )
        return self.time_converter.seconds_to_sim(
            data_seconds + propagation_seconds
        )
