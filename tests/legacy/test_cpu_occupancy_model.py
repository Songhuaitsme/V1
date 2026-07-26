import unittest
import warnings

from shared import config
from legacy.network_env import (
    NetworkEnvironment,
    evaluate_schedule_candidates,
    estimate_wait_opportunity,
    get_max_retries_for_task,
)


class CpuOccupancyModelTest(unittest.TestCase):
    def setUp(self):
        self.env = NetworkEnvironment()
        self.node = self.env.compute_nodes[0]
        self.env.node_resources[self.node] = {"total": 3500.0, "used": 0.0}
        self.env.scheduled_allocations = []

    def _make_wait_task(self, latency_limit=1.0, current_time=0.0, retry_count=0, sla_type="Soft"):
        cpu = 100.0
        duration = 1.0
        return {
            "id": 100,
            "cpu": cpu,
            "duration": duration,
            "cpu_time": cpu * duration,
            "data_size": 1.0,
            "bw": 1.0,
            "source_node": self.env.base_stations[0],
            "latency_limit": latency_limit,
            "generated_time": 0.0,
            "current_time_context": current_time,
            "sla_type": sla_type,
            "retry_count": retry_count,
        }

    def test_task_cpu_demand_and_duration_are_not_node_capacity(self):
        task = {"cpu": 100, "duration": 10, "cpu_time": 1000}

        self.assertEqual(self.env.get_task_cpu_demand(task), 100.0)
        self.assertEqual(self.env.get_task_cpu_supply(self.node, task), 100.0)
        self.assertEqual(self.env.estimate_execution_duration(self.node, task), 10.0)
        self.assertEqual(self.env.get_task_cpu_time_demand(task), 1000.0)

    def test_concurrent_tasks_are_allowed_when_sum_fits_capacity(self):
        self.env.scheduled_allocations = [
            {"node": self.node, "cpu": 100.0, "start_time": 0.0, "finish_time": 10.0},
            {"node": self.node, "cpu": 200.0, "start_time": 0.0, "finish_time": 10.0},
            {"node": self.node, "cpu": 500.0, "start_time": 0.0, "finish_time": 10.0},
        ]

        self.assertEqual(self.env.get_reserved_cpu_usage(self.node, 0.0, 10.0), 800.0)
        self.assertTrue(self.env.has_node_capacity_for_interval(self.node, 2700.0, 0.0, 10.0))

    def test_capacity_rejects_task_when_reserved_plus_demand_exceeds_total(self):
        self.env.scheduled_allocations = [
            {"node": self.node, "cpu": 3400.0, "start_time": 0.0, "finish_time": 10.0},
        ]

        self.assertFalse(self.env.has_node_capacity_for_interval(self.node, 200.0, 0.0, 10.0))

    def test_node_bucket_index_updates_when_allocations_are_added_and_removed(self):
        allocation = {"node": self.node, "cpu": 250.0, "start_time": 0.0, "finish_time": 10.0}

        self.env.add_scheduled_allocation(allocation)
        self.assertEqual(self.env.get_reserved_cpu_usage(self.node, 0.0, 10.0), 250.0)

        self.env.remove_scheduled_allocation(allocation)
        self.assertEqual(self.env.get_reserved_cpu_usage(self.node, 0.0, 10.0), 0.0)

    def test_uniform_base_electricity_price_ignores_region_and_tier(self):
        original_flag = config.USE_UNIFORM_BASE_ELECTRICITY_PRICE
        original_uniform_price = config.UNIFORM_BASE_ELECTRICITY_PRICE_YUAN_PER_MW
        config.USE_UNIFORM_BASE_ELECTRICITY_PRICE = True
        config.UNIFORM_BASE_ELECTRICITY_PRICE_YUAN_PER_MW = 1.23
        try:
            price_a = self.env.pricing_manager._fetch_realtime_electricity_price(
                {"region": "A", "tier": 1},
                0.0,
            )
            price_g = self.env.pricing_manager._fetch_realtime_electricity_price(
                {"region": "G", "tier": 3},
                0.0,
            )
        finally:
            config.USE_UNIFORM_BASE_ELECTRICITY_PRICE = original_flag
            config.UNIFORM_BASE_ELECTRICITY_PRICE_YUAN_PER_MW = original_uniform_price

        self.assertAlmostEqual(price_a, price_g)

    def test_duration_falls_back_to_cpu_time_over_task_cpu_only_when_missing_duration(self):
        task = {"cpu": 100, "cpu_time": 1000}

        self.assertEqual(self.env.estimate_execution_duration(self.node, task), 10.0)

    def test_latency_limit_checks_wait_and_network_but_not_execution_duration(self):
        source_node = self.env.base_stations[0]
        data_size = 1.0
        path = self.env.topo_manager.find_path(source_node, self.node)
        network_delay = self.env.topo_manager.calculate_transmission_delay(
            path,
            data_size,
            bandwidth_demand=1.0,
        )
        latency_limit = max(network_delay * 1.2, 1e-6)
        duration = latency_limit + 5.0
        cpu = 100.0
        task = {
            "id": 1,
            "cpu": cpu,
            "duration": duration,
            "cpu_time": cpu * duration,
            "data_size": data_size,
            "bw": 1.0,
            "source_node": source_node,
            "latency_limit": latency_limit,
            "generated_time": 0.0,
            "current_time_context": 0.0,
            "sla_type": "Soft",
            "retry_count": 0,
        }

        _, _, _, info = self.env.step(0, task, wait_queue=[])

        self.assertEqual(info["status"], "Success")
        self.assertGreater(task["duration"], task["latency_limit"])
        self.assertLessEqual(info["delays"]["end_to_end"], task["latency_limit"])
        self.assertGreater(info["delays"]["completion_delay"], task["latency_limit"])
        self.assertEqual(info["delays"]["execution"], duration)

    def test_latency_limit_rejects_queue_plus_network_delay(self):
        source_node = self.env.base_stations[0]
        data_size = 1.0
        path = self.env.topo_manager.find_path(source_node, self.node)
        network_delay = self.env.topo_manager.calculate_transmission_delay(
            path,
            data_size,
            bandwidth_demand=1.0,
        )
        latency_limit = max(network_delay * 1.2, 1e-6)
        queue_delay = latency_limit - network_delay * 0.5
        cpu = 100.0
        duration = 1.0
        task = {
            "id": 2,
            "cpu": cpu,
            "duration": duration,
            "cpu_time": cpu * duration,
            "data_size": data_size,
            "bw": 1.0,
            "source_node": source_node,
            "latency_limit": latency_limit,
            "generated_time": 0.0,
            "current_time_context": queue_delay,
            "sla_type": "Soft",
            "retry_count": 0,
        }

        _, _, _, info = self.env.step(0, task, wait_queue=[])

        self.assertEqual(info["status"], "Timeout")
        self.assertGreater(info["delays"]["queue"] + info["delays"]["network"], task["latency_limit"])

    def test_wait_timeout_reward_matches_reward_components_sum(self):
        wait_action = len(self.env.compute_nodes)
        task = self._make_wait_task(latency_limit=config.SCHEDULING_CYCLE)

        _, reward, _, info = self.env.step(wait_action, task, wait_queue=[])

        self.assertEqual(info["status"], "Timeout")
        self.assertAlmostEqual(reward, sum(info["reward_components"].values()))
        self.assertIn("wait_detail", info)
        for key in [
            "wait_type",
            "queue_wait_time",
            "retry_count",
            "max_retries",
            "remaining_sla_time",
            "wait_queue_length",
            "wait_gain",
            "wait_net_gain",
            "immediate_best_score",
            "future_best_score",
            "wait_gain_positive",
            "no_immediate_action",
            "queue_ratio",
            "source_region_queue_ratio",
            "same_sla_queue_ratio",
        ]:
            self.assertIn(key, info["wait_detail"])

    def test_wait_deferred_includes_wait_detail(self):
        wait_action = len(self.env.compute_nodes)
        task = self._make_wait_task(latency_limit=1.0)

        _, _, _, info = self.env.step(wait_action, task, wait_queue=[])

        self.assertEqual(info["status"], "Deferred")
        self.assertIn("wait_detail", info)
        self.assertEqual(info["wait_detail"]["wait_type"], "queue_wait")
        for key in [
            "urgency",
            "retry_ratio",
            "queue_ratio",
            "wait_penalty",
            "queue_wait_time",
            "retry_count",
            "max_retries",
            "remaining_sla_time",
            "wait_queue_length",
            "wait_gain",
            "wait_net_gain",
            "immediate_best_score",
            "future_best_score",
            "wait_gain_positive",
            "no_immediate_action",
            "source_region_queue_ratio",
            "same_sla_queue_ratio",
        ]:
            self.assertIn(key, info["wait_detail"])

    def test_wait_penalty_becomes_more_negative_as_retry_ratio_increases(self):
        wait_action = len(self.env.compute_nodes)
        low_retry = self._make_wait_task(latency_limit=1.0, retry_count=0)
        high_retry = self._make_wait_task(latency_limit=1.0, retry_count=config.MAX_RETRIES)

        _, _, _, low_info = self.env.step(wait_action, low_retry, wait_queue=[])
        _, _, _, high_info = self.env.step(wait_action, high_retry, wait_queue=[])

        self.assertLess(
            high_info["wait_detail"]["wait_penalty"],
            low_info["wait_detail"]["wait_penalty"],
        )

    def test_wait_penalty_becomes_more_negative_as_queue_ratio_increases(self):
        wait_action = len(self.env.compute_nodes)
        task = self._make_wait_task(latency_limit=1.0)
        queued_task = self._make_wait_task(latency_limit=1.0)
        long_queue = [queued_task] * config.MAX_QUEUE_LENGTH

        _, _, _, low_info = self.env.step(wait_action, task, wait_queue=[])
        _, _, _, high_info = self.env.step(wait_action, task, wait_queue=long_queue)

        self.assertLess(
            high_info["wait_detail"]["wait_penalty"],
            low_info["wait_detail"]["wait_penalty"],
        )

    def test_wait_penalty_becomes_more_negative_as_urgency_increases(self):
        wait_action = len(self.env.compute_nodes)
        low_urgency = self._make_wait_task(latency_limit=1.0, current_time=0.0)
        high_urgency = self._make_wait_task(latency_limit=1.0, current_time=0.9)

        _, _, _, low_info = self.env.step(wait_action, low_urgency, wait_queue=[])
        _, _, _, high_info = self.env.step(wait_action, high_urgency, wait_queue=[])

        self.assertLess(
            high_info["wait_detail"]["wait_penalty"],
            low_info["wait_detail"]["wait_penalty"],
        )

    def test_estimate_wait_opportunity_returns_wait_gain_fields(self):
        task = self._make_wait_task(latency_limit=1.0)

        detail = estimate_wait_opportunity(self.env, task, [], task["current_time_context"])

        self.assertIsInstance(detail, dict)
        self.assertIn("wait_gain", detail)
        self.assertIn("immediate_best_score", detail)
        self.assertIn("future_best_score", detail)

    def test_wait_gain_increases_when_future_price_is_lower(self):
        task = self._make_wait_task(latency_limit=10.0)
        original_price_fn = self.env.get_dynamic_cpu_price

        self.env.get_dynamic_cpu_price = lambda *args, **kwargs: 0.02
        flat_detail = estimate_wait_opportunity(self.env, task, [], task["current_time_context"])

        def lower_future_price(node_id, global_time=0.0, cpu_delta=0.0,
                               interval_start=None, interval_end=None):
            return 0.04 if global_time <= task["current_time_context"] else 0.001

        self.env.get_dynamic_cpu_price = lower_future_price
        lower_future_detail = estimate_wait_opportunity(self.env, task, [], task["current_time_context"])
        self.env.get_dynamic_cpu_price = original_price_fn

        self.assertIsNotNone(flat_detail["wait_gain"])
        self.assertIsNotNone(lower_future_detail["wait_gain"])
        self.assertGreater(lower_future_detail["wait_gain"], flat_detail["wait_gain"])

    def test_wait_gain_looks_beyond_legacy_five_cycle_window(self):
        task = self._make_wait_task(latency_limit=1.0)
        original_price_fn = self.env.get_dynamic_cpu_price
        original_horizon = config.WAIT_GAIN_LOOKAHEAD_CYCLES
        original_max_samples = config.WAIT_GAIN_MAX_LOOKAHEAD_SAMPLES
        trigger_time = task["current_time_context"] + 10 * config.SCHEDULING_CYCLE

        def lower_price_after_ten_cycles(node_id, global_time=0.0, cpu_delta=0.0,
                                         interval_start=None, interval_end=None):
            return 0.04 if global_time < trigger_time else 0.001

        try:
            config.WAIT_GAIN_LOOKAHEAD_CYCLES = None
            config.WAIT_GAIN_MAX_LOOKAHEAD_SAMPLES = 200
            self.env.get_dynamic_cpu_price = lower_price_after_ten_cycles

            detail = estimate_wait_opportunity(self.env, task, [], task["current_time_context"])
        finally:
            self.env.get_dynamic_cpu_price = original_price_fn
            config.WAIT_GAIN_LOOKAHEAD_CYCLES = original_horizon
            config.WAIT_GAIN_MAX_LOOKAHEAD_SAMPLES = original_max_samples

        self.assertGreater(detail["sla_lookahead_cycles"], 5)
        self.assertGreaterEqual(detail["lookahead_horizon_cycles"], 10)
        self.assertIsNotNone(detail["wait_gain"])
        self.assertGreater(detail["wait_gain"], 0.0)

    def test_wait_gain_marks_no_immediate_action_when_compute_is_infeasible(self):
        task = self._make_wait_task(latency_limit=1.0)
        for node in self.env.compute_nodes:
            self.env.node_resources[node] = {"total": 0.0, "used": 0.0}

        detail = estimate_wait_opportunity(self.env, task, [], task["current_time_context"])

        self.assertTrue(detail["no_immediate_action"])
        self.assertIsNone(detail["immediate_best_score"])

    def test_schedule_candidates_include_current_and_future_slots(self):
        task = self._make_wait_task(latency_limit=1.0)

        candidates = evaluate_schedule_candidates(self.env, task, [], task["current_time_context"])
        candidate_times = {round(candidate["schedule_time"], 8) for candidate in candidates}

        self.assertIn(round(task["current_time_context"], 8), candidate_times)
        self.assertTrue(any(candidate["schedule_time"] > task["current_time_context"] for candidate in candidates))
        self.assertTrue(all("score" in candidate for candidate in candidates))

    def test_wait_opportunity_can_reuse_precomputed_candidates(self):
        task = self._make_wait_task(latency_limit=1.0)
        now = task["current_time_context"]
        candidates = [
            {"schedule_time": now, "score": 1.0},
            {"schedule_time": now + config.SCHEDULING_CYCLE, "score": 1.4},
        ]

        detail = estimate_wait_opportunity(self.env, task, [], now, candidates=candidates)

        self.assertAlmostEqual(detail["wait_gain"], 0.4)
        self.assertEqual(detail["immediate_best_score"], 1.0)
        self.assertEqual(detail["future_best_score"], 1.4)

    def test_compute_valid_actions_uses_only_current_candidates_for_compute_actions(self):
        train = self._import_train_module()
        task = self._make_wait_task(latency_limit=1.0)
        now = task["current_time_context"]
        candidates = [{
            "node": self.node,
            "action_index": 0,
            "schedule_time": now + config.SCHEDULING_CYCLE,
            "score": 1.0,
        }]

        actions, detail = train.compute_valid_actions(
            self.env,
            task,
            [],
            now,
            return_wait_detail=True,
            candidates=candidates,
        )

        self.assertNotIn(0, actions)
        self.assertIn(len(self.env.compute_nodes), actions)
        self.assertEqual(detail["wait_reason"], "no_feasible_compute_action")

    def test_step_accepts_precomputed_current_candidates(self):
        task = self._make_wait_task(latency_limit=1.0)
        now = task["current_time_context"]
        candidates = evaluate_schedule_candidates(self.env, task, [], now)
        current = [
            candidate for candidate in candidates
            if abs(candidate["schedule_time"] - now) <= 1e-9
        ]
        self.assertTrue(current)

        _, _, _, info = self.env.step(
            current[0]["action_index"],
            task,
            wait_queue=[],
            candidates=candidates,
        )

        self.assertEqual(info["status"], "Success")
        self.assertEqual(info["execute_time"], now)

    def _import_train_module(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
        from legacy import train
        return train

    def test_wait_blocked_when_gain_below_threshold_and_compute_action_exists(self):
        train = self._import_train_module()
        task = self._make_wait_task(latency_limit=1.0)
        original_threshold = config.WAIT_GAIN_THRESHOLD
        original_estimator = train.estimate_wait_opportunity
        config.WAIT_GAIN_THRESHOLD = 1.0
        train.estimate_wait_opportunity = lambda *args, **kwargs: {
            "wait_gain": 0.0,
            "immediate_best_score": 1.0,
            "future_best_score": 1.0,
            "no_immediate_action": False,
        }
        try:
            actions, detail = train.compute_valid_actions(
                self.env,
                task,
                [],
                task["current_time_context"],
                return_wait_detail=True,
            )
        finally:
            train.estimate_wait_opportunity = original_estimator
            config.WAIT_GAIN_THRESHOLD = original_threshold

        self.assertGreater(len(actions), 0)
        self.assertNotIn(len(self.env.compute_nodes), actions)
        self.assertEqual(detail["wait_blocked_reason"], "wait_gain_below_penalty_adjusted_threshold")

    def test_wait_blocked_when_positive_gain_does_not_cover_wait_penalty(self):
        train = self._import_train_module()
        task = self._make_wait_task(latency_limit=1.0)
        original_threshold = config.WAIT_GAIN_THRESHOLD
        original_penalty_weight = config.WAIT_GAIN_PENALTY_WEIGHT
        original_estimator = train.estimate_wait_opportunity
        config.WAIT_GAIN_THRESHOLD = 0.0
        config.WAIT_GAIN_PENALTY_WEIGHT = 1.0
        train.estimate_wait_opportunity = lambda *args, **kwargs: {
            "wait_gain": 0.05,
            "immediate_best_score": 1.0,
            "future_best_score": 1.05,
            "no_immediate_action": False,
        }
        try:
            actions, detail = train.compute_valid_actions(
                self.env,
                task,
                [],
                task["current_time_context"],
                return_wait_detail=True,
            )
        finally:
            train.estimate_wait_opportunity = original_estimator
            config.WAIT_GAIN_THRESHOLD = original_threshold
            config.WAIT_GAIN_PENALTY_WEIGHT = original_penalty_weight

        self.assertGreater(len(actions), 0)
        self.assertNotIn(len(self.env.compute_nodes), actions)
        self.assertLess(detail["wait_net_gain"], 0.0)
        self.assertEqual(
            detail["wait_blocked_reason"],
            "wait_gain_below_penalty_adjusted_threshold",
        )

    def test_wait_allowed_when_gain_exceeds_penalty_adjusted_threshold(self):
        train = self._import_train_module()
        task = self._make_wait_task(latency_limit=1.0)
        wait_action = len(self.env.compute_nodes)
        original_threshold = config.WAIT_GAIN_THRESHOLD
        original_penalty_weight = config.WAIT_GAIN_PENALTY_WEIGHT
        original_estimator = train.estimate_wait_opportunity
        config.WAIT_GAIN_THRESHOLD = 0.5
        config.WAIT_GAIN_PENALTY_WEIGHT = 1.0
        train.estimate_wait_opportunity = lambda *args, **kwargs: {
            "wait_gain": 2.0,
            "immediate_best_score": 1.0,
            "future_best_score": 3.0,
            "no_immediate_action": False,
        }
        try:
            actions, detail = train.compute_valid_actions(
                self.env,
                task,
                [],
                task["current_time_context"],
                return_wait_detail=True,
            )
        finally:
            train.estimate_wait_opportunity = original_estimator
            config.WAIT_GAIN_THRESHOLD = original_threshold
            config.WAIT_GAIN_PENALTY_WEIGHT = original_penalty_weight

        self.assertIn(wait_action, actions)
        self.assertGreater(detail["wait_net_gain"], 0.0)
        self.assertEqual(detail["wait_reason"], "positive_wait_net_gain")

    def test_wait_allowed_when_no_compute_action_is_feasible(self):
        train = self._import_train_module()
        task = self._make_wait_task(latency_limit=1.0)
        wait_action = len(self.env.compute_nodes)
        for node in self.env.compute_nodes:
            self.env.node_resources[node] = {"total": 0.0, "used": 0.0}

        actions, detail = train.compute_valid_actions(
            self.env,
            task,
            [],
            task["current_time_context"],
            return_wait_detail=True,
        )

        self.assertEqual(actions, [wait_action])
        self.assertEqual(detail["wait_reason"], "no_feasible_compute_action")

    def test_wait_not_allowed_after_max_retries(self):
        train = self._import_train_module()
        task = self._make_wait_task(latency_limit=1.0, retry_count=config.MAX_RETRIES)

        actions, detail = train.compute_valid_actions(
            self.env,
            task,
            [],
            task["current_time_context"],
            return_wait_detail=True,
        )

        self.assertNotIn(len(self.env.compute_nodes), actions)
        self.assertEqual(detail["wait_blocked_reason"], "max_retries_exceeded")

    def test_wait_not_allowed_when_next_cycle_exceeds_sla(self):
        train = self._import_train_module()
        task = self._make_wait_task(latency_limit=config.SCHEDULING_CYCLE)

        actions, detail = train.compute_valid_actions(
            self.env,
            task,
            [],
            task["current_time_context"],
            return_wait_detail=True,
        )

        self.assertNotIn(len(self.env.compute_nodes), actions)
        self.assertEqual(detail["wait_blocked_reason"], "sla_exceeded_next_cycle")

    def test_hard_uses_smaller_max_retries(self):
        hard_task = self._make_wait_task(sla_type="Hard")
        soft_task = self._make_wait_task(sla_type="Soft")

        self.assertLess(
            get_max_retries_for_task(hard_task),
            get_max_retries_for_task(soft_task),
        )
        self.assertEqual(get_max_retries_for_task(hard_task), config.MAX_RETRIES_BY_SLA["Hard"])

    def test_flexible_uses_larger_max_retries(self):
        flexible_task = self._make_wait_task(sla_type="Flexible")
        soft_task = self._make_wait_task(sla_type="Soft")

        self.assertGreater(
            get_max_retries_for_task(flexible_task),
            get_max_retries_for_task(soft_task),
        )
        self.assertEqual(
            get_max_retries_for_task(flexible_task),
            config.MAX_RETRIES_BY_SLA["Flexible"],
        )

    def test_wait_not_allowed_after_sla_specific_max_retries(self):
        train = self._import_train_module()
        task = self._make_wait_task(
            latency_limit=1.0,
            retry_count=config.MAX_RETRIES_BY_SLA["Hard"],
            sla_type="Hard",
        )

        actions, detail = train.compute_valid_actions(
            self.env,
            task,
            [],
            task["current_time_context"],
            return_wait_detail=True,
        )

        self.assertNotIn(len(self.env.compute_nodes), actions)
        self.assertEqual(detail["max_retries"], config.MAX_RETRIES_BY_SLA["Hard"])
        self.assertEqual(detail["wait_blocked_reason"], "max_retries_exceeded")

    def test_wait_detail_max_retries_uses_sla_type(self):
        wait_action = len(self.env.compute_nodes)
        task = self._make_wait_task(latency_limit=1.0, sla_type="Hard")

        _, _, _, info = self.env.step(wait_action, task, wait_queue=[])

        self.assertEqual(info["wait_detail"]["max_retries"], config.MAX_RETRIES_BY_SLA["Hard"])

    def test_wait_detail_includes_new_queue_pressure_fields(self):
        wait_action = len(self.env.compute_nodes)
        task = self._make_wait_task(latency_limit=1.0)

        _, _, _, info = self.env.step(wait_action, task, wait_queue=[])

        for key in [
            "queue_ratio",
            "source_region_queue_ratio",
            "same_sla_queue_ratio",
        ]:
            self.assertIn(key, info["wait_detail"])

    def test_source_region_queue_ratio_increases_with_same_region_tasks(self):
        wait_action = len(self.env.compute_nodes)
        task = self._make_wait_task(latency_limit=1.0)
        other_source = "__other_region_source__"
        self.env.node_regions[other_source] = "__Other__"
        other_region_task = self._make_wait_task(latency_limit=1.0)
        other_region_task["source_node"] = other_source
        same_region_task = self._make_wait_task(latency_limit=1.0)

        _, _, _, low_info = self.env.step(wait_action, task, wait_queue=[other_region_task])
        _, _, _, high_info = self.env.step(wait_action, task, wait_queue=[other_region_task, same_region_task])

        self.assertGreater(
            high_info["wait_detail"]["source_region_queue_ratio"],
            low_info["wait_detail"]["source_region_queue_ratio"],
        )

    def test_same_sla_queue_ratio_increases_with_same_sla_tasks(self):
        wait_action = len(self.env.compute_nodes)
        task = self._make_wait_task(latency_limit=1.0, sla_type="Soft")
        other_sla_task = self._make_wait_task(latency_limit=1.0, sla_type="Hard")
        same_sla_task = self._make_wait_task(latency_limit=1.0, sla_type="Soft")

        _, _, _, low_info = self.env.step(wait_action, task, wait_queue=[other_sla_task])
        _, _, _, high_info = self.env.step(wait_action, task, wait_queue=[other_sla_task, same_sla_task])

        self.assertGreater(
            high_info["wait_detail"]["same_sla_queue_ratio"],
            low_info["wait_detail"]["same_sla_queue_ratio"],
        )


if __name__ == "__main__":
    unittest.main()
