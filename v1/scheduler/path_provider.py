"""Declared deterministic candidate path set for v1.0 complete enumeration."""

from typing import List

import networkx as nx

from v1.domain.reservations import PathSpec
from v1.domain.units import positive_finite
from .transmission import build_path_spec


class StaticPathProvider:
    def __init__(self, graph, max_paths_per_target: int = 1):
        if (
            isinstance(max_paths_per_target, bool)
            or not isinstance(max_paths_per_target, int)
            or max_paths_per_target <= 0
        ):
            raise ValueError("max_paths_per_target must be a positive integer")
        self.graph = graph
        self.max_paths_per_target = max_paths_per_target

    def candidate_paths(
        self,
        source_node: str,
        target_node: str,
        bandwidth_demand_mbps: float,
    ) -> List[PathSpec]:
        bandwidth = positive_finite(
            "bandwidth_demand_mbps",
            bandwidth_demand_mbps,
        )
        if source_node == target_node:
            if source_node not in self.graph:
                return []
            return [build_path_spec(self.graph, [source_node])]

        feasible_graph = self.graph.copy()
        blocked = [
            (u, v)
            for u, v, data in feasible_graph.edges(data=True)
            if float(data.get("capacity", 0.0)) < bandwidth
        ]
        feasible_graph.remove_edges_from(blocked)
        try:
            iterator = nx.shortest_simple_paths(
                feasible_graph,
                source_node,
                target_node,
                weight="cost",
            )
            paths = []
            for ordered_nodes in iterator:
                paths.append(build_path_spec(self.graph, ordered_nodes))
                if len(paths) >= self.max_paths_per_target:
                    break
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
