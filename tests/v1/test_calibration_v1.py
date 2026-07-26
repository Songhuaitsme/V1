import unittest

from v1.calibrate_v1_objective import _distribution, run_multi_seed_calibration


class CalibrationV1Test(unittest.TestCase):
    def test_distribution_can_filter_tardiness_by_sla(self):
        samples = [
            {
                "preferred_start_tardiness_ratio": 0.0,
                "tardiness_applicable": True,
                "sla_type": "Soft",
            },
            {
                "preferred_start_tardiness_ratio": 0.5,
                "tardiness_applicable": True,
                "sla_type": "Soft",
            },
            {
                "preferred_start_tardiness_ratio": 1.0,
                "tardiness_applicable": True,
                "sla_type": "Flexible",
            },
            {
                "preferred_start_tardiness_ratio": 0.0,
                "tardiness_applicable": False,
                "sla_type": "Hard",
            },
        ]
        soft = _distribution(
            samples,
            "preferred_start_tardiness_ratio",
            applicable_only=True,
            sla_type="Soft",
        )
        flexible = _distribution(
            samples,
            "preferred_start_tardiness_ratio",
            applicable_only=True,
            sla_type="Flexible",
        )
        self.assertEqual(soft["count"], 2)
        self.assertEqual(soft["p50"], 0.25)
        self.assertEqual(flexible["count"], 1)
        self.assertEqual(flexible["p50"], 1.0)

    def test_multi_seed_calibration_rejects_duplicate_seed_weighting(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            run_multi_seed_calibration(1.0, (9, 9), 10, 100)


if __name__ == "__main__":
    unittest.main()
