import random
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from shared import config
from v1.domain.candidates import (
    deterministic_candidate_id,
    deterministic_candidate_id_v1_fields,
)
from v1.domain.models import (
    MetricStatus,
    SlaType,
    TaskRuntime,
    TaskSpec,
    TaskState,
    TaskValidationError,
    migrate_legacy_task,
    to_legacy_task_dict,
)
from v1.domain.sla import SlaPolicy
from v1.domain.units import (
    DataUnitConverter,
    TimeConverter,
    UnitValidationError,
    cpu_work_cpu_hours,
    cpu_work_sim_units,
    validate_scheduling_grid,
)
from shared.task_manager import TaskManager


class DomainFoundationTest(unittest.TestCase):
    def test_specialized_candidate_id_matches_generic_canonical_hash(self):
        values = {
            "schema": "1.0",
            "mode": "complete",
            "task": '任务-"quoted"',
            "node": "K\\2",
            "path": "path-α",
            "transmission_start": -0.0,
            "transmission_end": 1.0,
            "compute_start": 1.0,
            "compute_end": 3.141592653589793,
            "reservation_version": 17,
            "forecast_version": "预测-v1",
        }
        self.assertEqual(
            deterministic_candidate_id(values),
            deterministic_candidate_id_v1_fields(
                compute_end=values["compute_end"],
                compute_start=values["compute_start"],
                forecast_version=values["forecast_version"],
                mode=values["mode"],
                node=values["node"],
                path=values["path"],
                reservation_version=values["reservation_version"],
                schema=values["schema"],
                task=values["task"],
                transmission_end=values["transmission_end"],
                transmission_start=values["transmission_start"],
            ),
        )

    def _task(
        self,
        sla_type="Soft",
        preferred=10.0,
        latest=None,
        arrival=0.0,
        duration=3.0,
    ):
        if sla_type == "Hard":
            preferred = None
            latest = 10.0 if latest is None else latest
        return TaskSpec.create(
            task_id=f"task-{sla_type}-{duration}",
            arrival_time_sim=arrival,
            source_node="S",
            cpu_demand=4.0,
            execution_duration_sim=duration,
            data_size_mb=100.0,
            bandwidth_demand_mbps=100.0,
            sla_type=sla_type,
            preferred_start_limit_sim=preferred,
            latest_start_limit_sim=latest,
        )

    def _legacy_task(self, **overrides):
        task = {
            "id": "legacy-1",
            "generated_time": 2.0,
            "source_node": "S",
            "cpu": 4.0,
            "duration": 3.0,
            "data_size": 100.0,
            "bw": 20.0,
            "sla_type": "Soft",
            "latency_limit": 10.0,
            "retry_count": 0,
        }
        task.update(overrides)
        return task

    # UNIT-001
    def test_unit_001_seconds_per_sim_unit(self):
        converter = TimeConverter.from_traffic_day_duration(288.0)
        self.assertEqual(converter.seconds_per_sim_unit, 300.0)

    # UNIT-002
    def test_unit_002_scheduling_cycle_seconds(self):
        converter = TimeConverter.from_traffic_day_duration(288.0)
        self.assertEqual(converter.scheduling_cycle_seconds(0.005), 1.5)

    # UNIT-003
    def test_unit_003_time_round_trips(self):
        converter = TimeConverter(300.0)
        for seconds in (0.0, 1.5, 300.0, 86400.0):
            with self.subTest(seconds=seconds):
                self.assertAlmostEqual(
                    converter.sim_to_seconds(converter.seconds_to_sim(seconds)),
                    seconds,
                )
        for sim_time in (0.0, 0.005, 12.0):
            with self.subTest(sim_time=sim_time):
                self.assertAlmostEqual(
                    converter.hours_to_sim(converter.sim_to_hours(sim_time)),
                    sim_time,
                )

    # UNIT-004
    def test_unit_004_decimal_mb_and_transmission_seconds(self):
        megabits = DataUnitConverter.decimal_mb_to_megabits(100.0)
        self.assertEqual(megabits, 800.0)
        self.assertEqual(megabits / 100.0, 8.0)

    # UNIT-005
    def test_unit_005_cpu_work_units(self):
        converter = TimeConverter(300.0)
        self.assertEqual(cpu_work_sim_units(100.0, 12.0), 1200.0)
        self.assertEqual(cpu_work_cpu_hours(100.0, 12.0, converter), 100.0)

    # UNIT-006
    def test_unit_006_invalid_time_configuration_is_rejected(self):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(seconds_per_sim_unit=value):
                with self.assertRaises(UnitValidationError):
                    TimeConverter(value)
            with self.subTest(scheduling_cycle=value):
                with self.assertRaises(UnitValidationError):
                    validate_scheduling_grid(value)
        with self.assertRaises(UnitValidationError):
            validate_scheduling_grid(0.005, 0.01)

    # TASK-001
    def test_task_001_cpu_and_duration_do_not_depend_on_target_node(self):
        task = self._task()
        observations = {
            node: (task.cpu_demand, task.execution_duration_sim, task.cpu_work_sim_units)
            for node in ("A", "B", "C")
        }
        self.assertEqual(len(set(observations.values())), 1)

    # TASK-002
    def test_task_002_cpu_work_sim_units(self):
        self.assertEqual(self._task().cpu_work_sim_units, 12.0)

    # TASK-003
    def test_task_003_missing_or_invalid_fields_are_rejected(self):
        canonical = {
            "task_id": "t",
            "arrival_time_sim": 0.0,
            "source_node": "S",
            "cpu_demand": 1.0,
            "execution_duration_sim": 1.0,
            "data_size_mb": 0.0,
            "bandwidth_demand_mbps": 1.0,
            "sla_type": "Hard",
            "latest_start_limit_sim": 1.0,
        }
        cases = [
            ("task_id", None),
            ("cpu_demand", 0.0),
            ("data_size_mb", -1.0),
            ("sla_type", "Unknown"),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                invalid = dict(canonical)
                if value is None:
                    invalid.pop(field)
                else:
                    invalid[field] = value
                with self.assertRaises(TaskValidationError) as caught:
                    TaskSpec.from_mapping(invalid)
                self.assertIn(caught.exception.field, (field, "task_id"))

    # TASK-004
    def test_task_004_sla_limits_are_derived(self):
        hard = self._task("Hard", latest=7.0)
        soft = self._task("Soft", preferred=10.0)
        flexible = self._task("Flexible", preferred=10.0)
        self.assertIsNone(hard.preferred_start_limit_sim)
        self.assertEqual(hard.latest_start_limit_sim, 7.0)
        self.assertEqual(soft.latest_start_limit_sim, 12.0)
        self.assertEqual(flexible.latest_start_limit_sim, 15.0)

    # TASK-005
    def test_task_005_zero_or_non_finite_resource_fields_are_rejected(self):
        for field, value in (
            ("execution_duration_sim", 0.0),
            ("bandwidth_demand_mbps", 0.0),
            ("cpu_demand", float("nan")),
            ("data_size_mb", float("inf")),
        ):
            with self.subTest(field=field, value=value):
                values = {
                    "task_id": "t",
                    "arrival_time_sim": 0.0,
                    "source_node": "S",
                    "cpu_demand": 1.0,
                    "execution_duration_sim": 1.0,
                    "data_size_mb": 0.0,
                    "bandwidth_demand_mbps": 1.0,
                    "sla_type": "Hard",
                    "latest_start_limit_sim": 1.0,
                }
                values[field] = value
                with self.assertRaises(TaskValidationError) as caught:
                    TaskSpec.from_mapping(values)
                self.assertEqual(caught.exception.field, field)

    def test_task_spec_is_frozen_and_runtime_is_separate(self):
        task = self._task()
        runtime = TaskRuntime(task.task_id)
        with self.assertRaises(FrozenInstanceError):
            task.cpu_demand = 99.0
        runtime.state = TaskState.QUEUED
        self.assertEqual(task.cpu_demand, 4.0)
        self.assertEqual(runtime.state, TaskState.QUEUED)

    def test_legacy_adapter_exposes_v1_shadow_fields(self):
        spec = migrate_legacy_task(self._legacy_task())
        view = to_legacy_task_dict(spec, self._legacy_task())
        self.assertEqual(view["task_schema_version"], "1.0")
        self.assertEqual(view["task_adapter_mode"], "legacy_shadow")
        self.assertEqual(view["latency_limit"], 10.0)
        self.assertEqual(view["preferred_start_limit_sim"], 10.0)
        self.assertEqual(view["latest_start_limit_sim"], 12.0)

    def test_v1_mapping_rejects_legacy_fields_and_wrong_version(self):
        values = {
            "task_schema_version": "1.0",
            "task_id": "t",
            "arrival_time_sim": 0.0,
            "source_node": "S",
            "cpu_demand": 1.0,
            "execution_duration_sim": 1.0,
            "data_size_mb": 0.0,
            "bandwidth_demand_mbps": 1.0,
            "sla_type": "Hard",
            "latest_start_limit_sim": 1.0,
        }
        with self.assertRaises(TaskValidationError) as caught:
            TaskSpec.from_mapping(dict(values, latency_limit=1.0))
        self.assertEqual(caught.exception.field, "latency_limit")
        with self.assertRaises(TaskValidationError) as caught:
            TaskSpec.from_mapping(dict(values, task_schema_version="0.3"))
        self.assertEqual(caught.exception.field, "task_schema_version")

    # SLA-001
    def test_sla_001_hard_boundary_is_feasible(self):
        self.assertTrue(SlaPolicy.is_start_feasible(self._task("Hard"), 10.0))

    # SLA-002
    def test_sla_002_hard_after_boundary_is_infeasible(self):
        self.assertFalse(
            SlaPolicy.is_start_feasible(self._task("Hard"), 10.0 + 1e-9)
        )

    # SLA-003
    def test_sla_003_soft_preferred_boundary_has_zero_tardiness(self):
        metric = SlaPolicy.preferred_start_tardiness(self._task("Soft"), 10.0)
        self.assertEqual(metric.status, MetricStatus.VALID)
        self.assertEqual(metric.value, 0.0)

    # SLA-004
    def test_sla_004_soft_midpoint_has_half_tardiness(self):
        metric = SlaPolicy.preferred_start_tardiness(self._task("Soft"), 11.0)
        self.assertAlmostEqual(metric.value, 0.5)

    # SLA-005
    def test_sla_005_soft_latest_boundary_has_full_tardiness(self):
        task = self._task("Soft")
        self.assertTrue(SlaPolicy.is_start_feasible(task, 12.0))
        self.assertEqual(SlaPolicy.preferred_start_tardiness(task, 12.0).value, 1.0)

    # SLA-006
    def test_sla_006_soft_after_latest_is_infeasible(self):
        self.assertFalse(
            SlaPolicy.is_start_feasible(self._task("Soft"), 12.0 + 1e-9)
        )

    # SLA-007
    def test_sla_007_soft_tardiness_is_linear_and_continuous(self):
        task = self._task("Soft")
        starts = [10.0, 10.5, 11.0, 11.5, 12.0]
        ratios = [
            SlaPolicy.preferred_start_tardiness(task, start).value
            for start in starts
        ]
        self.assertEqual(ratios, [0.0, 0.25, 0.5, 0.75, 1.0])

    # SLA-008
    def test_sla_008_flexible_preferred_boundary_has_zero_tardiness(self):
        metric = SlaPolicy.preferred_start_tardiness(
            self._task("Flexible"),
            10.0,
        )
        self.assertEqual(metric.value, 0.0)

    # SLA-009
    def test_sla_009_flexible_midpoint_has_half_tardiness(self):
        metric = SlaPolicy.preferred_start_tardiness(
            self._task("Flexible"),
            12.5,
        )
        self.assertAlmostEqual(metric.value, 0.5)

    # SLA-010
    def test_sla_010_execution_duration_does_not_change_start_sla(self):
        short = self._task("Soft", duration=1.0)
        long = self._task("Soft", duration=100.0)
        self.assertEqual(
            SlaPolicy.is_start_feasible(short, 11.0),
            SlaPolicy.is_start_feasible(long, 11.0),
        )
        self.assertEqual(
            SlaPolicy.preferred_start_tardiness(short, 11.0).value,
            SlaPolicy.preferred_start_tardiness(long, 11.0).value,
        )

    # SLA-011
    def test_sla_011_invalid_soft_limits_are_rejected(self):
        with self.assertRaises(TaskValidationError):
            self._task("Soft", preferred=0.0)
        with self.assertRaises(TaskValidationError):
            self._task("Soft", preferred=10.0, latest=11.0)

    # SLA-012
    def test_sla_012_flexible_latest_boundary_has_full_tardiness(self):
        task = self._task("Flexible")
        self.assertTrue(SlaPolicy.is_start_feasible(task, 15.0))
        self.assertEqual(SlaPolicy.preferred_start_tardiness(task, 15.0).value, 1.0)

    # SLA-013
    def test_sla_013_flexible_after_latest_is_infeasible(self):
        self.assertFalse(
            SlaPolicy.is_start_feasible(self._task("Flexible"), 15.0 + 1e-9)
        )

    # SLA-014
    def test_sla_014_flexible_tardiness_is_linear_and_continuous(self):
        task = self._task("Flexible")
        starts = [10.0, 11.25, 12.5, 13.75, 15.0]
        ratios = [
            SlaPolicy.preferred_start_tardiness(task, start).value
            for start in starts
        ]
        self.assertEqual(ratios, [0.0, 0.25, 0.5, 0.75, 1.0])

    # SLA-015
    def test_sla_015_invalid_flexible_and_legacy_none_inputs_fail(self):
        with self.assertRaises(TaskValidationError):
            self._task("Flexible", preferred=0.0)
        with self.assertRaises(TaskValidationError):
            self._task("Flexible", preferred=10.0, latest=14.0)
        missing_limit = self._legacy_task(sla_type="None")
        missing_limit.pop("latency_limit")
        with self.assertRaises(TaskValidationError) as caught:
            migrate_legacy_task(missing_limit)
        self.assertEqual(caught.exception.field, "latency_limit")
        for invalid_limit in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(legacy_none_limit=invalid_limit):
                with self.assertRaises(TaskValidationError):
                    migrate_legacy_task(
                        self._legacy_task(
                            sla_type="None",
                            latency_limit=invalid_limit,
                        )
                    )

    def test_legacy_none_migrates_to_flexible_with_one_point_five_limit(self):
        spec = migrate_legacy_task(
            self._legacy_task(sla_type="None", latency_limit=20.0)
        )
        self.assertEqual(spec.sla_type, SlaType.FLEXIBLE)
        self.assertEqual(spec.preferred_start_limit_sim, 20.0)
        self.assertEqual(spec.latest_start_limit_sim, 30.0)
        self.assertAlmostEqual(
            SlaPolicy.preferred_start_tardiness(spec, 2.0 + 25.0).value,
            0.5,
        )

    def test_hard_tardiness_is_not_applicable_and_model_feature_is_flagged(self):
        task = self._task("Hard")
        metric = SlaPolicy.preferred_start_tardiness(task, 5.0)
        feature, applicable = SlaPolicy.tardiness_model_feature(task, 5.0)
        self.assertEqual(metric.status, MetricStatus.NOT_APPLICABLE)
        self.assertIsNone(metric.value)
        self.assertEqual(feature, 0.0)
        self.assertFalse(applicable)

    def test_task_manager_generates_valid_v1_shadow_tasks(self):
        random.seed(7)
        np.random.seed(7)
        manager = TaskManager(["I0", "A0", "C0"], total_compute_capacity=10000.0)
        tasks = manager.generate_tasks(
            12,
            global_time=1.0,
            cycle=1,
            cpu_budget=1e12,
        )
        self.assertEqual(len(tasks), 12)
        self.assertEqual(len({task["task_id"] for task in tasks}), len(tasks))
        for task in tasks:
            with self.subTest(task_id=task["task_id"]):
                self.assertNotEqual(task["sla_type"], "None")
                self.assertEqual(task["task_schema_version"], "1.0")
                spec = migrate_legacy_task(task)
                self.assertEqual(spec.task_id, task["task_id"])

    def test_all_v1_version_fields_are_frozen(self):
        fields = (
            "REQUIREMENTS_VERSION",
            "ALGORITHM_VERSION",
            "TASK_SCHEMA_VERSION",
            "CANDIDATE_SCHEMA_VERSION",
            "MODEL_SCHEMA_VERSION",
            "METRIC_SCHEMA_VERSION",
            "AGGREGATION_SCHEMA_VERSION",
        )
        self.assertTrue(all(getattr(config, field) == "1.0" for field in fields))


if __name__ == "__main__":
    unittest.main()
