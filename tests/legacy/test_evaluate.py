import unittest

from legacy.evaluate import EvaluationMetrics, summarize_results


class EvaluationMetricsTest(unittest.TestCase):
    def test_summary_uses_cpu_time_weighted_cost(self):
        metrics = EvaluationMetrics(
            generated=2,
            processed=2,
            succeeded=2,
            costs=[1.0, 9.0],
            baseline_costs=[2.0, 18.0],
            cpu_time_demands=[1.0, 9.0],
            cost_per_cpu_times=[1.0, 1.0],
            cost_ratios=[0.5, 0.5],
        )

        result = metrics.summary(
            seed=42,
            steps=10,
            wait_queue_length=0,
            active_task_count=0,
        )

        self.assertEqual(result["completion_rate"], 1.0)
        self.assertEqual(result["throughput_rate"], 1.0)
        self.assertEqual(result["weighted_cost_per_cpu_time"], 1.0)
        self.assertEqual(result["weighted_cost_ratio"], 0.5)

    def test_multi_seed_summary_reports_population_statistics(self):
        rows = [
            {"seed": 42, "steps": 10, "completion_rate": 0.8},
            {"seed": 43, "steps": 10, "completion_rate": 1.0},
        ]

        summary = {
            row["metric"]: row
            for row in summarize_results(rows)
        }

        self.assertAlmostEqual(summary["completion_rate"]["mean"], 0.9)
        self.assertAlmostEqual(summary["completion_rate"]["std"], 0.1)
        self.assertEqual(summary["steps"]["mean"], 10.0)


if __name__ == "__main__":
    unittest.main()
