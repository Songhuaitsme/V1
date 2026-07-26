import math
import numpy as np
from shared import config
from v1.domain.units import TariffConverter


class PricingManager:
    def __init__(self, G=None):
        self.G = G
        # 预计算电价查找表 (LUT)
        self._price_lut = self._initialize_smooth_price_lut(resolution=144)
        self.region_base_price_map = getattr(config, 'REGION_BASE_ELECTRICITY_PRICE', {})

        # 预定义各节点的绿电装机画像 (单位: MW)
        self.node_green_profiles = self._initialize_green_profiles()

    def _initialize_green_profiles(self):
        """为网络节点注入异构的物理装机容量"""
        profiles = {}
        if not self.G: return profiles

        for node in self.G.nodes():
            node_str = str(node)
            # 东部节点 (A-F): 灰电主导，几乎无绿电装机 (典型的需求中心)
            if node_str[0] in ['A', 'B', 'C', 'D', 'E', 'F']:
                profiles[node] = {'solar': 5.0, 'wind': 5.0}
            # 西部光伏主导节点 (例如 K, I): 沙漠光伏基地
            elif node_str[0] in ['K', 'I']:
                profiles[node] = {'solar': 150.0, 'wind': 20.0}
            # 西部风电主导节点 (例如 G, H, J, L, M): 戈壁风电场
            else:
                profiles[node] = {'solar': 30.0, 'wind': 120.0}
        return profiles

    def _initialize_smooth_price_lut(self, resolution: int = 144) -> np.ndarray:
        hours = np.array(getattr(config, 'TOU_PRICE_HOURS', [0.0, 8.0, 18.0, 22.0, 24.0]))
        prices = np.array(getattr(config, 'TOU_PRICE_MULTIPLIERS', [0.6, 1.0, 1.5, 0.8, 0.6]))
        x_query = np.linspace(0, 24, resolution, endpoint=False)
        return np.interp(x_query, hours, prices)

    @staticmethod
    def _get_static_cpu_price() -> float:
        """Return the fixed CPU price used when dynamic pricing is disabled."""
        baseline_price = getattr(config, 'BASELINE_ELECTRICITY_PRICE_YUAN_PER_MW', 1.0)
        cpu_power_mw = getattr(config, 'CPU_POWER_UNIT_MW', 0.01)
        return max(getattr(config, 'MIN_CPU_PRICE', 1e-6), baseline_price * cpu_power_mw)

    def _calculate_solar_output(self, capacity: float, sim_hr: float) -> float:
        """光伏出力模型：高斯分布，峰值在 13:00，标准差 2.5 小时"""
        if capacity <= 0: return 0.0
        return capacity * math.exp(-0.5 * ((sim_hr - 13.0) / 2.5) ** 2)

    def _calculate_wind_output(self, capacity: float, sim_hr: float) -> float:
        """风电出力模型：反调峰特性，正弦波动，峰值在凌晨 04:00"""
        if capacity <= 0: return 0.0
        base = capacity * 0.3  # 保障 30% 基础风力
        # 调整正弦相位，使其在 x=4 时达到峰值 (sin(pi/2))
        fluctuation = capacity * 0.7 * 0.5 * (math.sin(math.pi * (sim_hr - 22.0) / 12.0) + 1.0)
        return base + fluctuation

    def _fetch_realtime_electricity_price(self, node_data: dict, global_time: float) -> float:
        day_duration = getattr(config, 'TRAFFIC_DAY_DURATION_IN_SIM', 86400)
        progress = (global_time % day_duration) / day_duration
        lut_size = len(self._price_lut)
        idx = int(progress * lut_size) % lut_size
        tou_multiplier = (
            self._price_lut[idx]
            if getattr(config, 'ENABLE_TOU_PRICING', True)
            else 1.0
        )

        use_uniform_base = getattr(config, 'USE_UNIFORM_BASE_ELECTRICITY_PRICE', False)
        use_region_base = getattr(config, 'ENABLE_REGION_BASE_ELECTRICITY_PRICE', True)
        if use_uniform_base or not use_region_base:
            uniform_base_price = getattr(
                config,
                'UNIFORM_BASE_ELECTRICITY_PRICE_YUAN_PER_MW',
                getattr(config, 'BASELINE_ELECTRICITY_PRICE_YUAN_PER_MW', 1.0),
            )
            if not use_uniform_base and not use_region_base:
                uniform_base_price = getattr(
                    config, 'BASELINE_ELECTRICITY_PRICE_YUAN_PER_MW', 1.0
                )
            return uniform_base_price * tou_multiplier

        region = node_data.get('region')
        tier = int(node_data.get('tier', 2))
        fallback_base_price = (
            getattr(config, 'BASELINE_ELECTRICITY_PRICE_YUAN_PER_MW', 1.0)
            * getattr(config, 'PRICE_TIER_MULTIPLIERS', {1: 0.85, 2: 1.0, 3: 1.2}).get(tier, 1.0)
        )
        region_base_price = self.region_base_price_map.get(region, fallback_base_price)
        return region_base_price * tou_multiplier

    def get_external_tariff_yuan_per_mwh(self, node_id: str, global_time: float) -> float:
        """v1.0 external tariff boundary; legacy 0.50~1.30 inputs are yuan/kWh."""

        if self.G is None or node_id not in self.G:
            raise ValueError(f"unknown pricing node: {node_id}")
        yuan_per_kwh = self._fetch_realtime_electricity_price(
            self.G.nodes[node_id],
            global_time,
        )
        return TariffConverter.yuan_per_kwh_to_yuan_per_mwh(yuan_per_kwh)

    def get_green_supply_mw(self, node_id: str, global_time: float) -> float:
        """v1.0 exogenous green-power forecast, independent of task load."""

        return float(self.get_node_power_profile(
            node_id,
            {"total": 0.0, "used": 0.0},
            global_time,
        )["green_supply_mw"])

    @staticmethod
    def get_projected_resource_usage(resource_usage: dict, cpu_delta: float = 0.0) -> dict:
        """返回加入候选任务后的资源占用视图，用于计算边际电价。"""
        total_cpu = resource_usage.get('total', 0.0)
        used_cpu = max(0.0, resource_usage.get('used', 0.0) + cpu_delta)
        return {'total': total_cpu, 'used': used_cpu}

    def get_node_power_profile(self, node_id: str, resource_usage: dict, global_time: float = 0.0) -> dict:
        """返回节点在指定时刻的算力负荷、绿电供给和绿电匹配度。"""
        day_duration = getattr(config, 'TRAFFIC_DAY_DURATION_IN_SIM', 86400)
        sim_hr = ((global_time % day_duration) / day_duration) * 24.0

        used_cpu = resource_usage.get('used', 0.0)
        power_demand_mw = used_cpu * getattr(config, 'CPU_POWER_UNIT_MW', 0.01)

        profile = self.node_green_profiles.get(node_id, {'solar': 0.0, 'wind': 0.0})
        solar_mw = self._calculate_solar_output(profile['solar'], sim_hr)
        wind_mw = self._calculate_wind_output(profile['wind'], sim_hr)
        green_supply_mw = solar_mw + wind_mw

        green_used_mw = min(power_demand_mw, green_supply_mw)
        green_unused_mw = max(0.0, green_supply_mw - green_used_mw)
        green_match_ratio = 1.0 if power_demand_mw <= 0 else min(1.0, green_supply_mw / power_demand_mw)
        green_absorption_ratio = 0.0 if green_supply_mw <= 0 else green_used_mw / green_supply_mw
        return {
            "power_demand_mw": power_demand_mw,
            "green_supply_mw": green_supply_mw,
            "green_used_mw": green_used_mw,
            "green_unused_mw": green_unused_mw,
            "green_match_ratio": green_match_ratio,
            "green_absorption_ratio": green_absorption_ratio
        }

    def get_dynamic_price(self, node_id: str, node_data: dict, resource_usage: dict, global_time: float = 0.0) -> float:
        """计算每 CPU 每单位执行时长的边际价格。"""
        if str(node_id).endswith('0'):
            return 0.0

        if not getattr(config, 'ENABLE_DYNAMIC_PRICING', True):
            return self._get_static_cpu_price()

        # 1. 将算力负载映射为电力负荷，并计算当前绿电供给
        power_profile = self.get_node_power_profile(node_id, resource_usage, global_time)
        power_demand_mw = power_profile["power_demand_mw"]
        total_green_supply_mw = power_profile["green_supply_mw"]
        used_cpu = resource_usage.get('used', 0.0)

        # 2. 获取电网基础指导价
        price_index = self._fetch_realtime_electricity_price(node_data, global_time)
        load_per_yuan = max(getattr(config, 'ELECTRICITY_LOAD_PER_YUAN_MW', 1.0), 1e-8)
        base_electricity_price = price_index / load_per_yuan

        # ==========================================
        # 5. 边际碳定价核心博弈逻辑
        # ==========================================
        if power_demand_mw <= 0:
            final_electricity_price = base_electricity_price * 0.1  # 空载时理论底价，防止除零

        elif total_green_supply_mw >= power_demand_mw:
            # 场景 A: 绿电盈余 (弃风弃光)，算力需求完全被绿电覆盖
            # 盈余比例越大，说明电力越被浪费，降价幅度越狠
            surplus_ratio = (total_green_supply_mw - power_demand_mw) / total_green_supply_mw
            subsidy_rate = (
                getattr(config, 'GREEN_SUBSIDY_RATE', 0.8)
                if getattr(config, 'ENABLE_GREEN_SUBSIDY', True)
                else 0.0
            )
            discount = 1.0 - (subsidy_rate * surplus_ratio)
            final_electricity_price = base_electricity_price * discount

        else:
            # 场景 B: 绿电耗尽，引入电网火电 (灰电)
            green_ratio = total_green_supply_mw / power_demand_mw
            grey_ratio = 1.0 - green_ratio
            tax_rate = (
                getattr(config, 'CARBON_TAX_RATE', 0.5)
                if getattr(config, 'ENABLE_CARBON_TAX', True)
                else 0.0
            )

            # 综合单价 = (绿电占比 * 基础价) + (灰电占比 * 基础价 * 惩罚税率)
            final_electricity_price = base_electricity_price * (green_ratio * 1.0 + grey_ratio * (1.0 + tax_rate))

        # ==========================================
        # 6. 叠加资源紧缺度拥塞费 (硬件磨损与排队成本)
        total_cpu = resource_usage.get('total', 0.0)
        if total_cpu > 0:
            usage_ratio = used_cpu / total_cpu
            # CPU 利用率越高，在此处的计算成本呈指数放大
            if getattr(config, 'ENABLE_CPU_UTILIZATION_MARKUP', True):
                resource_markup = (
                    1
                    + config.PRICE_ALPHA
                    * math.pow(usage_ratio, config.PRICE_BETA)
                )
            else:
                resource_markup = 1.0
        else:
            resource_markup = 1.0

        cpu_power_mw = getattr(config, 'CPU_POWER_UNIT_MW', 0.01)
        cpu_price = final_electricity_price * cpu_power_mw * resource_markup
        return max(getattr(config, 'MIN_CPU_PRICE', 1e-6), cpu_price)
