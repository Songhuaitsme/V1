import networkx as nx
import itertools
from shared import config

class DataLoader:
    @staticmethod
    def load_network_topology():
        """
        基于13个大区的分层星型拓扑加载器。
        区内：所有计算节点直连本区 0 号节点。
        骨干：所有 0 号节点全互联。
        """
        # --- 拓扑原始数据 (保持不变) ---
        topology_data = [
            {'地区': 'A', '编号': 'A1', '容量': 3500.0}, {'地区': 'A', '编号': 'A2', '容量': 3500.0},
            {'地区': 'A', '编号': 'A3', '容量': 4200.0}, {'地区': 'A', '编号': 'A0', '容量': None},
            {'地区': 'B', '编号': 'B1', '容量': 3500.0}, {'地区': 'B', '编号': 'B2', '容量': 6000.0},
            {'地区': 'B', '编号': 'B3', '容量': 5800.0}, {'地区': 'B', '编号': 'B0', '容量': None},
            {'地区': 'C', '编号': 'C1', '容量': 4200.0}, {'地区': 'C', '编号': 'C2', '容量': 3500.0},
            {'地区': 'C', '编号': 'C3', '容量': 7000.0}, {'地区': 'C', '编号': 'C0', '容量': None},
            {'地区': 'D', '编号': 'D1', '容量': 3000.0}, {'地区': 'D', '编号': 'D2', '容量': 6500.0},
            {'地区': 'D', '编号': 'D3', '容量': 4000.0}, {'地区': 'D', '编号': 'D4', '容量': 2500.0},
            {'地区': 'D', '编号': 'D0', '容量': None},
            {'地区': 'E', '编号': 'E1', '容量': 5200.0}, {'地区': 'E', '编号': 'E2', '容量': 6800.0},
            {'地区': 'E', '编号': 'E3', '容量': 4500.0}, {'地区': 'E', '编号': 'E4', '容量': 7200.0},
            {'地区': 'E', '编号': 'E0', '容量': None},
            {'地区': 'F', '编号': 'F1', '容量': 4000.0}, {'地区': 'F', '编号': 'F2', '容量': 6200.0},
            {'地区': 'F', '编号': 'F3', '容量': 7500.0}, {'地区': 'F', '编号': 'F0', '容量': None},
            {'地区': 'G', '编号': 'G1', '容量': 6000.0}, {'地区': 'G', '编号': 'G2', '容量': 5000.0},
            {'地区': 'G', '编号': 'G3', '容量': 5000.0}, {'地区': 'G', '编号': 'G4', '容量': 6000.0},
            {'地区': 'G', '编号': 'G5', '容量': 5300.0}, {'地区': 'G', '编号': 'G0', '容量': None},
            {'地区': 'H', '编号': 'H1', '容量': 4000.0}, {'地区': 'H', '编号': 'H2', '容量': 7300.0},
            {'地区': 'H', '编号': 'H3', '容量': 4500.0}, {'地区': 'H', '编号': 'H4', '容量': 4000.0},
            {'地区': 'H', '编号': 'H5', '容量': 4000.0}, {'地区': 'H', '编号': 'H0', '容量': None},
            {'地区': 'I', '编号': 'I1', '容量': 7200.0}, {'地区': 'I', '编号': 'I2', '容量': 4500.0},
            {'地区': 'I', '编号': 'I3', '容量': 6000.0}, {'地区': 'I', '编号': 'I4', '容量': 4500.0},
            {'地区': 'I', '编号': 'I5', '容量': 4200.0}, {'地区': 'I', '编号': 'I0', '容量': None},
            {'地区': 'J', '编号': 'J1', '容量': 4000.0}, {'地区': 'J', '编号': 'J2', '容量': 5000.0},
            {'地区': 'J', '编号': 'J3', '容量': 5200.0}, {'地区': 'J', '编号': 'J4', '容量': 4500.0},
            {'地区': 'J', '编号': 'J5', '容量': 4000.0}, {'地区': 'J', '编号': 'J6', '容量': 4000.0},
            {'地区': 'J', '编号': 'J0', '容量': None},
            {'地区': 'K', '编号': 'K1', '容量': 7000.0}, {'地区': 'K', '编号': 'K2', '容量': 3000.0},
            {'地区': 'K', '编号': 'K3', '容量': 4500.0}, {'地区': 'K', '编号': 'K4', '容量': 3200.0},
            {'地区': 'K', '编号': 'K0', '容量': None},
            {'地区': 'L', '编号': 'L1', '容量': 4000.0}, {'地区': 'L', '编号': 'L2', '容量': 4000.0},
            {'地区': 'L', '编号': 'L3', '容量': 4000.0}, {'地区': 'L', '编号': 'L4', '容量': 4500.0},
            {'地区': 'L', '编号': 'L0', '容量': None},
            {'地区': 'M', '编号': 'M1', '容量': 3500.0}, {'地区': 'M', '编号': 'M2', '容量': 3800.0},
            {'地区': 'M', '编号': 'M0', '容量': None}
        ]

        regions = set([item['地区'] for item in topology_data])
        zero_nodes = [item['编号'] for item in topology_data if item['编号'].endswith('0')]

        G = nx.Graph()
        # 电价档位划分
        price_tier_mapping = {
            'G': 1, 'H': 1, 'I': 1,
            'J': 2, 'K': 2, 'L': 2, 'M': 2,
            'A': 3, 'B': 3, 'C': 3, 'D': 3, 'E': 3, 'F': 3
        }

        # 节点初始化
        for item in topology_data:
            node_id = item['编号']
            G.add_node(node_id, region=item['地区'],
                       capacity=item['容量'] if item['容量'] else config.DEFAULT_NODE_CPU,
                       tier=price_tier_mapping.get(item['地区'], 2))

        # --- 连接规则与物理时延注入 ---

        # 1. 区域内连接：各计算节点仅与本区0号节点连接 (时延 5)
        for region in regions:
            zero_node = f"{region}0"
            if zero_node not in G.nodes: continue
            # 找到该区域内非0号的计算节点
            compute_nodes = [n for n, d in G.nodes(data=True) if d['region'] == region and n != zero_node]
            for u in compute_nodes:
                distance_km = config.LOCAL_LINK_DISTANCE_KM
                prop_delay = distance_km / config.FIBER_PROPAGATION_SPEED_KM_PER_S
                G.add_edge(
                    u,
                    zero_node,
                    edge_type='local',
                    distance_km=distance_km,
                    prop_delay=prop_delay,
                    cost=prop_delay * 1000
                )

        # 2. 骨干网连接：0号节点全互联 (根据档位计算时延)
        for u, v in itertools.combinations(zero_nodes, 2):
            tier_u = G.nodes[u].get('tier', 2)
            tier_v = G.nodes[v].get('tier', 2)
            tiers = {tier_u, tier_v}

            tier_gap = abs(tier_u - tier_v)
            distance_km = config.BACKBONE_DISTANCE_KM_BY_TIER_GAP.get(tier_gap, 900.0)
            prop_delay = distance_km / config.FIBER_PROPAGATION_SPEED_KM_PER_S

            G.add_edge(
                u,
                v,
                edge_type='backbone',
                distance_km=distance_km,
                prop_delay=prop_delay,
                cost=prop_delay * 1000
            )


        return G
