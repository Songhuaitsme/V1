import argparse
import unittest

from v1.ablate_v1_tardiness import _parse_weights, summarize


class TardinessAblationV1Test(unittest.TestCase):
    def test_weight_parser_requires_non_negative_pair(self):
        self.assertEqual(_parse_weights("0.5,0.25"), (0.5, 0.25))
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_weights("0.5")
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_weights("-0.5,0.25")

    def test_recommendation_preserves_baseline_terminal_outcomes(self):
        common = {
            "seed": 2,
            "cost_yuan_per_completed_cpu_hour": 1.0,
            "completed_task_green_coverage": 0.5,
            "system_green_absorption_rate": 0.5,
            "active_wait_count": 0,
            "soft_acceptable_tardy_rate": 1.0,
            "soft_tardiness_p95": 0.0,
            "flexible_acceptable_tardy_rate": 1.0,
            "flexible_tardiness_p95": 0.0,
        }
        rows = [
            {
                **common,
                "soft_weight": 0.0,
                "flexible_weight": 0.0,
                "completion_rate": 1.0,
                "expired_count": 0,
                "failed_count": 0,
                "total_economic_cost_yuan": 10.0,
                "soft_preferred_on_time_rate": 0.5,
                "flexible_preferred_on_time_rate": 0.5,
            },
            {
                **common,
                "soft_weight": 0.5,
                "flexible_weight": 0.25,
                "completion_rate": 1.0,
                "expired_count": 0,
                "failed_count": 0,
                "total_economic_cost_yuan": 11.0,
                "soft_preferred_on_time_rate": 1.0,
                "flexible_preferred_on_time_rate": 1.0,
            },
            {
                **common,
                "soft_weight": 1.0,
                "flexible_weight": 0.5,
                "completion_rate": 0.5,
                "expired_count": 1,
                "failed_count": 0,
                "total_economic_cost_yuan": 9.0,
                "soft_preferred_on_time_rate": 1.0,
                "flexible_preferred_on_time_rate": 1.0,
            },
        ]

        _, recommendation = summarize(rows)

        self.assertEqual(recommendation["soft_weight"], 0.5)
        self.assertEqual(recommendation["flexible_weight"], 0.25)


if __name__ == "__main__":
    unittest.main()
