"""Minimal physical infrastructure view shared with the v1 runtime.

This deliberately contains no legacy scheduling, reward, WAIT, or action logic.
"""

from dataclasses import dataclass

from . import config
from .pricing_manager import PricingManager
from .topology_manager import TopologyManager


@dataclass
class InfrastructureContext:
    topo_manager: TopologyManager
    pricing_manager: PricingManager
    all_nodes: tuple
    base_stations: tuple
    compute_nodes: tuple
    node_resources: dict

    @classmethod
    def create(cls) -> "InfrastructureContext":
        topology = TopologyManager()
        graph = topology.graph
        all_nodes = tuple(graph.nodes())
        base_stations = tuple(
            node for node in all_nodes if str(node).endswith("0")
        )
        compute_nodes = tuple(
            node for node in all_nodes if not str(node).endswith("0")
        )
        node_resources = {
            node: {
                "total": float(
                    graph.nodes[node].get("capacity", config.DEFAULT_NODE_CPU)
                    or config.DEFAULT_NODE_CPU
                ),
                "used": 0.0,
            }
            for node in compute_nodes
        }
        return cls(
            topology,
            PricingManager(graph),
            all_nodes,
            base_stations,
            compute_nodes,
            node_resources,
        )
