import unittest

from shared import config
from shared.pricing_manager import PricingManager


class PricingAblationTest(unittest.TestCase):
    def setUp(self):
        self.manager = PricingManager()
        self.node_id = "A1"
        self.node_data = {"region": "A", "tier": 1}
        self.resource_usage = {"total": 1000.0, "used": 800.0}
        self.day_duration = config.TRAFFIC_DAY_DURATION_IN_SIM
        self.original_values = {
            name: getattr(config, name)
            for name in (
                "ENABLE_DYNAMIC_PRICING",
                "ENABLE_CPU_UTILIZATION_MARKUP",
                "ENABLE_TOU_PRICING",
                "ENABLE_REGION_BASE_ELECTRICITY_PRICE",
                "ENABLE_GREEN_SUBSIDY",
                "ENABLE_CARBON_TAX",
                "USE_UNIFORM_BASE_ELECTRICITY_PRICE",
            )
        }

    def tearDown(self):
        for name, value in self.original_values.items():
            setattr(config, name, value)

    def _price(self, global_time=0.0):
        return self.manager.get_dynamic_price(
            self.node_id,
            self.node_data,
            self.resource_usage,
            global_time,
        )

    def _disable_other_effects(self):
        config.ENABLE_CPU_UTILIZATION_MARKUP = False
        config.ENABLE_TOU_PRICING = False
        config.ENABLE_GREEN_SUBSIDY = False
        config.ENABLE_CARBON_TAX = False
        config.USE_UNIFORM_BASE_ELECTRICITY_PRICE = True

    def test_dynamic_pricing_switch_returns_fixed_baseline_price(self):
        config.ENABLE_DYNAMIC_PRICING = False

        expected = (
            config.BASELINE_ELECTRICITY_PRICE_YUAN_PER_MW
            * config.CPU_POWER_UNIT_MW
        )
        self.assertAlmostEqual(self._price(), expected)

    def test_cpu_utilization_markup_switch(self):
        self._disable_other_effects()
        config.ENABLE_CPU_UTILIZATION_MARKUP = True
        enabled_price = self._price()
        config.ENABLE_CPU_UTILIZATION_MARKUP = False
        disabled_price = self._price()

        self.assertGreater(enabled_price, disabled_price)

    def test_tou_pricing_switch(self):
        self._disable_other_effects()
        config.ENABLE_TOU_PRICING = True
        enabled_price = self._price()
        config.ENABLE_TOU_PRICING = False
        disabled_price = self._price()

        self.assertAlmostEqual(enabled_price, disabled_price * 0.55)

    def test_region_base_price_switch(self):
        self._disable_other_effects()
        config.USE_UNIFORM_BASE_ELECTRICITY_PRICE = False
        config.ENABLE_REGION_BASE_ELECTRICITY_PRICE = True
        regional_price = self._price()
        config.ENABLE_REGION_BASE_ELECTRICITY_PRICE = False
        baseline_price = self._price()

        self.assertAlmostEqual(
            regional_price,
            config.REGION_BASE_ELECTRICITY_PRICE["A"] * config.CPU_POWER_UNIT_MW,
        )
        self.assertAlmostEqual(
            baseline_price,
            config.BASELINE_ELECTRICITY_PRICE_YUAN_PER_MW
            * config.CPU_POWER_UNIT_MW,
        )

    def test_green_subsidy_switch(self):
        self._disable_other_effects()
        self.manager.node_green_profiles[self.node_id] = {
            "solar": 1000.0,
            "wind": 0.0,
        }
        midday = self.day_duration * 13.0 / 24.0
        config.ENABLE_GREEN_SUBSIDY = True
        subsidized_price = self._price(midday)
        config.ENABLE_GREEN_SUBSIDY = False
        unsubsidized_price = self._price(midday)

        self.assertLess(subsidized_price, unsubsidized_price)

    def test_carbon_tax_switch(self):
        self._disable_other_effects()
        self.manager.node_green_profiles[self.node_id] = {
            "solar": 0.0,
            "wind": 0.0,
        }
        config.ENABLE_CARBON_TAX = True
        taxed_price = self._price()
        config.ENABLE_CARBON_TAX = False
        untaxed_price = self._price()

        self.assertAlmostEqual(
            taxed_price,
            untaxed_price * (1.0 + config.CARBON_TAX_RATE),
        )


if __name__ == "__main__":
    unittest.main()
