import json
from pathlib import Path
from unittest.mock import patch
import unittest
import uuid

import torch

from v1.domain.models import SlaType, TaskSpec
from v1.train_v1 import (
    TrainingPerformanceProfiler,
    _estimate_candidate_work,
    _resume_config_compatible,
    _should_run_invariant_check,
    run_training,
)
from v1.v1_runtime import (
    create_v1_runtime,
    ensure_v1_runtime_forecasts_for_tasks,
    extend_v1_runtime_forecasts,
    v1_runtime_forecast_end,
)


class TrainingPerformanceProfilerTest(unittest.TestCase):
    def test_task_driven_forecast_extension_covers_unbounded_duration(self):
        runtime = create_v1_runtime(forecast_end_sim=4.0, random_seed=7)
        task = TaskSpec.create(
            task_id="long-flexible",
            arrival_time_sim=1.0,
            source_node=runtime.infrastructure.base_stations[0],
            cpu_demand=1.0,
            execution_duration_sim=100.0,
            data_size_mb=1.0,
            bandwidth_demand_mbps=1.0,
            sla_type=SlaType.FLEXIBLE,
            preferred_start_limit_sim=10.0,
        )

        covered_until = ensure_v1_runtime_forecasts_for_tasks(runtime, (task,))

        required = (
            task.absolute_latest_start_sim
            + task.execution_duration_sim
            + 2.0
        )
        self.assertGreaterEqual(covered_until, required)
        self.assertEqual(covered_until, v1_runtime_forecast_end(runtime))

    def test_resume_forecasts_are_extended_without_replacing_old_values(self):
        runtime = create_v1_runtime(forecast_end_sim=4.0, random_seed=7)
        node = runtime.infrastructure.compute_nodes[0]
        old_tariff_segments = runtime.accounting.tariff_by_node[node].segments
        old_green_segments = runtime.accounting.green_by_node[node].segments
        runtime.accounting._candidate_index_cache["stale"] = object()

        extended = extend_v1_runtime_forecasts(runtime, 8.0)

        self.assertEqual(extended, len(runtime.infrastructure.compute_nodes))
        self.assertEqual(
            runtime.accounting.tariff_by_node[node].segments[:len(old_tariff_segments)],
            old_tariff_segments,
        )
        self.assertEqual(
            runtime.accounting.green_by_node[node].segments[:len(old_green_segments)],
            old_green_segments,
        )
        self.assertEqual(
            runtime.accounting.tariff_by_node[node].segments[-1].interval_sim.end_sim,
            8.0,
        )
        self.assertEqual(
            runtime.accounting.green_by_node[node].segments[-1].interval_sim.end_sim,
            8.0,
        )
        runtime.accounting.tariff_by_node[node].value_at(7.5)
        runtime.accounting.green_by_node[node].value_at(7.5)
        self.assertFalse(runtime.accounting._candidate_index_cache)
        self.assertIs(runtime.metrics_ledger.accounting, runtime.accounting)

    def test_resume_allows_only_candidate_chunk_size_to_change(self):
        saved = {
            "candidate_chunk_size": 4096,
            "batch_size": 1,
            "epsilon_decay": 0.99,
        }
        requested = dict(saved, candidate_chunk_size=65536)
        self.assertTrue(_resume_config_compatible(saved, requested))
        self.assertTrue(
            _resume_config_compatible(
                saved, dict(requested, bootstrap_candidate_limit=None)
            )
        )
        self.assertFalse(
            _resume_config_compatible(
                saved, dict(requested, batch_size=2)
            )
        )
        self.assertFalse(
            _resume_config_compatible(
                saved, dict(requested, epsilon_decay=0.9)
            )
        )
        self.assertFalse(
            _resume_config_compatible(
                saved, dict(requested, bootstrap_candidate_limit=8192)
            )
        )

    def test_resume_allows_invariant_interval_to_change(self):
        saved = {
            "candidate_chunk_size": 4096,
            "invariant_check_every": 1,
            "batch_size": 128,
            "bootstrap_candidate_limit": None,
        }
        requested = dict(saved, invariant_check_every=500)
        self.assertTrue(_resume_config_compatible(saved, requested))

    def test_invariant_checks_run_periodically_and_before_checkpoints(self):
        values = {
            "invariant_check_every": 500,
            "checkpoint_every": 5000,
            "final_cycle": 600000,
        }
        self.assertFalse(_should_run_invariant_check(499, **values))
        self.assertTrue(_should_run_invariant_check(500, **values))
        self.assertTrue(_should_run_invariant_check(5000, **values))
        self.assertTrue(_should_run_invariant_check(600000, **values))

    def test_preflight_work_estimate_includes_replay_regeneration(self):
        estimate = _estimate_candidate_work(
            10,
            1000,
            200,
            batch_size=4,
            min_replay_size=4,
            updates_per_transition=1,
        )

        self.assertEqual(estimate["estimated_update_count"], 7)
        self.assertEqual(estimate["estimated_replay_context_samples"], 28)
        self.assertEqual(estimate["estimated_selection_candidate_visits"], 2000)
        self.assertEqual(estimate["estimated_bootstrap_candidate_visits"], 2800)
        self.assertEqual(estimate["estimated_total_candidate_visits"], 4800)
        self.assertEqual(
            estimate["estimated_bootstrap_candidate_visits_upper_bound"],
            5600,
        )

    def test_preflight_work_estimate_applies_bootstrap_candidate_limit(self):
        estimate = _estimate_candidate_work(
            10,
            1000,
            200,
            batch_size=4,
            min_replay_size=4,
            updates_per_transition=1,
            bootstrap_candidate_limit=30,
        )
        self.assertEqual(estimate["estimated_replay_context_samples"], 28)
        self.assertEqual(estimate["estimated_bootstrap_candidate_visits"], 840)
        self.assertEqual(
            estimate["estimated_bootstrap_candidate_visits_upper_bound"],
            840,
        )
        self.assertEqual(estimate["estimated_total_candidate_visits"], 2840)

    def test_summary_aggregates_exclusive_sections(self):
        profiler = TrainingPerformanceProfiler()
        profiler.add("candidate_prepare_seconds", 2.0)
        profiler.add("candidate_stream_seconds", 3.0)
        profiler.add("candidate_feature_encoding_seconds", 1.0)
        profiler.add("selection_inference_seconds", 1.0)
        profiler.add("backpropagation_seconds", 1.0)
        profiler.add("environment_update_seconds", 1.0)
        profiler.add("logging_seconds", 0.5)
        profiler.increment("selection_candidate_count", 700)

        summary = profiler.summary(10.0)

        self.assertEqual(summary["sections_seconds"]["candidate_slot_generation"], 5.0)
        self.assertEqual(summary["sections_percent"]["candidate_slot_generation"], 50.0)
        self.assertEqual(summary["sections_seconds"]["other_or_profiler_overhead"], 0.5)
        self.assertEqual(summary["selection_candidates_per_second"], 100.0)

    def test_zero_step_training_writes_profile_json_and_csv(self):
        token = uuid.uuid4().hex
        model_path = Path("artifacts/v1") / f"profile-smoke-{token}.pt"
        profile_path = Path("artifacts/v1") / f"profile-smoke-{token}.profile.json"
        created = (
            model_path,
            model_path.with_name(model_path.stem + ".last.pt"),
            profile_path,
            profile_path.with_suffix(".csv"),
        )
        try:
            with patch("v1.train_v1._settle", return_value=123.0):
                run_training(
                    0,
                    17,
                    model_path,
                    device="cpu",
                    allow_uncalibrated_objective=True,
                    profile=True,
                    profile_output_path=profile_path,
                )

            self.assertTrue(model_path.exists())
            self.assertTrue(profile_path.exists())
            self.assertTrue(profile_path.with_suffix(".csv").exists())
            report = json.loads(profile_path.read_text(encoding="utf-8"))
            self.assertIn("sections_percent", report)
            self.assertIn("detail_seconds", report)
            self.assertEqual(report["counters"]["selection_candidate_count"], 0)
            checkpoint = torch.load(
                model_path.with_name(model_path.stem + ".last.pt"),
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint["cycle"], 0)
            self.assertEqual(checkpoint["current_time_sim"], 0.0)
        finally:
            for path in created:
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
