import unittest

import networkx as nx

from shared import config
from shared.pricing_manager import PricingManager
from v1.ablation_settings import ABLATION_VARIANTS, apply_ablation_variant
from v1.domain.models import TaskSpec
from v1.learning.features import CandidateFeatureConfig, CandidateFeatureEncoder
from v1.learning.reward import GammaClock
from v1.scheduler.candidate_generator import CandidateGenerator
from v1.scheduler.path_provider import StaticPathProvider
from v1.scheduler.resource_calendar import ReservationCalendar
from v1.scheduler.transmission import TransmissionModel
from v1.domain.units import TimeConverter


class AblationSettingsV1Test(unittest.TestCase):
    def test_registry_follows_requested_group_order(self):
        groups = []
        for variant in ABLATION_VARIANTS.values():
            if variant.group not in groups:
                groups.append(variant.group)
        self.assertEqual(
            groups,
            ["candidate", "objective", "wait", "reward", "feature", "price"],
        )

    def test_variant_override_is_scoped_and_syncs_candidate_alias(self):
        original = config.V1_CANDIDATE_MODE
        with apply_ablation_variant("candidate.complete"):
            self.assertEqual(config.V1_CANDIDATE_MODE, "complete")
            self.assertEqual(config.CANDIDATE_MODE, "complete")
        self.assertEqual(config.V1_CANDIDATE_MODE, original)

    def test_feature_group_ablation_zeroes_only_declared_group(self):
        encoder = CandidateFeatureEncoder(
            {"N": 0},
            CandidateFeatureConfig(10.0, 10.0, 1.0, 10.0, 10.0),
            disabled_feature_groups=("green",),
        )
        values = encoder._encode_values(
            target_node="N",
            decision_time_sim=0.0,
            compute_start_sim=2.0,
            compute_end_sim=3.0,
            transmission_start_sim=1.0,
            earliest_compute_start_sim=1.0,
            marginal_cost_yuan=5.0,
            green_coverage=0.8,
            green_absorption_delta=0.4,
            green_opportunity=True,
            projected_node_utilization=0.5,
            projected_path_peak_utilization=0.4,
            capacity_margin=0.5,
            start_delay_sim=2.0,
            preferred_start_tardiness_ratio=0.1,
            preferred_start_tardiness_applicable=True,
            cpu_demand=2.0,
            bandwidth_demand_mbps=2.0,
        )
        self.assertEqual(values[6:9], (0.0, 0.0, 0.0))
        self.assertEqual(values[5], 0.5)

    def test_wait_off_keeps_only_immediate_grid_point(self):
        graph = nx.Graph()
        graph.add_nodes_from(("S", "N"))
        graph.add_edge(
            "S", "N", capacity=100.0, distance_km=0.0, cost=1.0
        )
        generator = CandidateGenerator(
            ("N",),
            1.0,
            StaticPathProvider(graph, 1),
            TransmissionModel(TimeConverter(1.0), 200000.0),
            ReservationCalendar({"N": 100.0}, {("S", "N"): 100.0}),
            active_wait_enabled=False,
        )
        task = TaskSpec.create(
            task_id="t",
            arrival_time_sim=0.0,
            source_node="S",
            cpu_demand=1.0,
            execution_duration_sim=1.0,
            data_size_mb=0.0,
            bandwidth_demand_mbps=1.0,
            sla_type="Soft",
            preferred_start_limit_sim=10.0,
        )
        result = generator.generate_complete(task, 0.0)
        self.assertEqual(len(result.candidates), 1)

    def test_discount_modes(self):
        converter = TimeConverter(1.0)
        self.assertEqual(GammaClock(0.9, converter, mode="none").discount(5), 1.0)
        self.assertEqual(
            GammaClock(
                0.9, converter, mode="decision_step", decision_gamma=0.8
            ).discount(5),
            0.8,
        )

    def test_v1_tariff_modes_are_explicit(self):
        graph = nx.Graph()
        graph.add_node("A1", region="A", tier=1)
        manager = PricingManager(graph)
        fixed = manager.get_external_tariff_yuan_per_mwh(
            "A1", 0.0, mode="fixed"
        )
        tou = manager.get_external_tariff_yuan_per_mwh(
            "A1", 0.0, mode="tou_uniform"
        )
        self.assertLess(tou, fixed)


if __name__ == "__main__":
    unittest.main()
