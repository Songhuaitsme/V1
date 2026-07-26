import unittest

import networkx as nx

from v1.accounting import (
    ExogenousEnergyAccounting,
    ForecastSegment,
    LinearPowerModel,
    MetricsLedger,
    PiecewiseConstantForecast,
)
from v1.domain.models import MetricStatus, SlaType, TaskSpec, TaskState
from v1.domain.reservations import CommitStatus, ReservationRequest, TimeInterval
from v1.domain.units import TimeConverter
from v1.evaluation_v1 import (
    EvaluationRunner,
    EvaluationStatus,
    TaskOutcome,
    UtilizationInterval,
    linear_percentile,
    paired_bootstrap,
    paired_t_summary,
    relative_change,
    summarize_load,
    summarize_sla,
)
from v1.domain.models import MetricValue
from v1.scheduler.candidate_generator import CandidateGenerator
from v1.scheduler.path_provider import StaticPathProvider
from v1.scheduler.resource_calendar import ReservationCalendar
from v1.scheduler.transmission import TransmissionModel, build_path_spec
from v1.scheduler.v1_scheduler import V1Scheduler


class EvaluationV1Test(unittest.TestCase):
    def _forecast(self, value, *, green=False):
        segment = ForecastSegment(TimeInterval(0.0, 1000.0), value)
        if green:
            return PiecewiseConstantForecast.green_power_mw((segment,))
        return PiecewiseConstantForecast.tariff_yuan_per_mwh((segment,))

    def _system(self, capacity=1.0, safety_cap=100):
        graph = nx.Graph()
        graph.add_node("N")
        calendar = ReservationCalendar({"N": capacity}, {})
        converter = TimeConverter(3600.0)
        generator = CandidateGenerator(
            ("N",),
            1.0,
            StaticPathProvider(graph, 1),
            TransmissionModel(converter, 200000.0),
            calendar,
        )
        accounting = ExogenousEnergyAccounting(
            converter,
            LinearPowerModel(1.0),
            {"N": self._forecast(100.0)},
            {"N": self._forecast(1.0, green=True)},
        )
        scheduler = V1Scheduler(
            calendar,
            generator,
            100,
            100,
            3,
            metrics_ledger=MetricsLedger(accounting),
        )
        return graph, calendar, scheduler, EvaluationRunner(
            scheduler,
            converter,
            safety_cap,
        )

    def _task(
        self,
        task_id,
        arrival=0.0,
        duration=1.0,
        latest=20.0,
        sla="Hard",
        preferred=None,
    ):
        if sla != "Hard":
            latest = None
            preferred = 10.0 if preferred is None else preferred
        return TaskSpec.create(
            task_id=task_id,
            arrival_time_sim=arrival,
            source_node="N",
            cpu_demand=1.0,
            execution_duration_sim=duration,
            data_size_mb=0.0,
            bandwidth_demand_mbps=1.0,
            sla_type=sla,
            preferred_start_limit_sim=preferred,
            latest_start_limit_sim=latest,
        )

    # EVAL-001 / EVAL-002 / EVAL-013 / EVAL-014
    def test_three_phase_runner_stops_arrivals_and_fully_drains(self):
        _, _, scheduler, runner = self._system(capacity=1.0)
        report = runner.run_frozen_policy(
            (
                self._task("a", arrival=0.0, duration=5.0),
                self._task("b", arrival=0.0, duration=1.0),
                self._task("after-cutoff", arrival=2.0),
            ),
            arrival_cutoff_sim=1.0,
            seed=7,
        )
        self.assertEqual(report.status, EvaluationStatus.VALID)
        self.assertEqual(report.metrics.arrival_count, 2)
        self.assertEqual(report.metrics.reserved_ever_count, 2)
        self.assertEqual(report.metrics.completed_count, 2)
        self.assertEqual(report.unsettled_task_ids, ())
        self.assertNotIn("after-cutoff", scheduler.state_machine.task_ids)
        for state in (
            TaskState.QUEUED,
            TaskState.PENDING_UNCOMMITTED,
            TaskState.RESERVED,
            TaskState.TRANSMITTING,
            TaskState.RUNNING,
        ):
            self.assertEqual(report.metrics.final_state_counts[state], 0)
        self.assertGreater(report.phase_batch_counts["execution_drain"], 0)
        self.assertGreaterEqual(report.metadata.final_settlement_time_sim, 6.0)

    # EVAL-003 / EVAL-004 / EVAL-005 / EVAL-007 / EVAL-008
    def test_service_cost_and_green_metrics_use_declared_denominators(self):
        _, _, _, runner = self._system(capacity=2.0)
        report = runner.run_frozen_policy(
            (self._task("a"), self._task("b")),
            arrival_cutoff_sim=1.0,
        )
        metrics = report.metrics
        self.assertEqual(metrics.acceptance_rate.value, 1.0)
        self.assertEqual(metrics.completion_rate.value, 1.0)
        self.assertEqual(metrics.reservation_reliability.value, 1.0)
        self.assertEqual(metrics.total_economic_cost_yuan, 200.0)
        self.assertEqual(metrics.completed_cpu_hours, 2.0)
        self.assertEqual(metrics.cost_yuan_per_completed_cpu_hour.value, 100.0)
        self.assertEqual(metrics.completed_task_green_coverage.value, 0.5)
        self.assertEqual(metrics.system_green_absorption_rate.value, 1.0)

    # EVAL-015 / STAT-011
    def test_safety_cap_invalidates_seed_and_suppresses_formal_metrics(self):
        _, _, _, runner = self._system(safety_cap=1)
        report = runner.run_frozen_policy(
            (self._task("long", duration=10.0),),
            arrival_cutoff_sim=1.0,
        )
        self.assertEqual(
            report.status,
            EvaluationStatus.INVALID_INCOMPLETE_SETTLEMENT,
        )
        self.assertIsNone(report.metrics)
        self.assertEqual(report.unsettled_task_ids, ("long",))

    # METRIC-001 / METRIC-004
    def test_zero_arrivals_and_empty_sla_subgroups_are_not_applicable(self):
        _, _, _, runner = self._system()
        report = runner.run_frozen_policy((), arrival_cutoff_sim=1.0)
        self.assertEqual(report.status, EvaluationStatus.VALID)
        self.assertEqual(report.metrics.acceptance_rate.status, MetricStatus.NOT_APPLICABLE)
        self.assertEqual(report.metrics.completion_rate.status, MetricStatus.NOT_APPLICABLE)
        for sla_metrics in report.metrics.sla_metrics.values():
            self.assertEqual(sla_metrics.count, 0)
            self.assertEqual(sla_metrics.expired_rate.status, MetricStatus.NOT_APPLICABLE)

    def _reservation(self, task, start):
        graph = nx.Graph()
        graph.add_node("N")
        calendar = ReservationCalendar({"N": 10.0}, {})
        request = ReservationRequest(
            task_id=task.task_id,
            committed_candidate_id="candidate-" + task.task_id,
            committed_at_sim=0.0,
            reservation_snapshot_version=0,
            target_node="N",
            path=build_path_spec(graph, ["N"]),
            transmission_interval_sim=None,
            compute_interval_sim=TimeInterval(start, start + task.execution_duration_sim),
            bandwidth_amount_mbps=1.0,
            cpu_amount=task.cpu_demand,
        )
        result = calendar.try_commit(request, 0)
        self.assertEqual(result.status, CommitStatus.COMMITTED)
        return result.reservation

    # EVAL-006 / EVAL-016 / SLA subgroup percentiles
    def test_flexible_on_time_tardy_expired_and_hard_na_semantics(self):
        on_time = self._task("on", sla="Flexible", preferred=10.0)
        tardy = self._task("late", sla="Flexible", preferred=10.0)
        expired = self._task("expired", sla="Flexible", preferred=10.0)
        hard = self._task("hard")
        outcomes = (
            TaskOutcome(on_time, TaskState.COMPLETED, self._reservation(on_time, 10.0)),
            TaskOutcome(tardy, TaskState.COMPLETED, self._reservation(tardy, 12.5)),
            TaskOutcome(expired, TaskState.EXPIRED, None),
            TaskOutcome(hard, TaskState.COMPLETED, self._reservation(hard, 0.0)),
        )
        summary = summarize_sla(outcomes)
        flexible = summary[SlaType.FLEXIBLE]
        self.assertEqual(flexible.count, 3)
        self.assertAlmostEqual(flexible.preferred_on_time_rate.value, 1.0 / 3.0)
        self.assertAlmostEqual(flexible.acceptable_tardy_rate.value, 1.0 / 3.0)
        self.assertAlmostEqual(flexible.expired_rate.value, 1.0 / 3.0)
        self.assertEqual(flexible.preferred_start_tardiness_p50.value, 0.25)
        self.assertAlmostEqual(flexible.preferred_start_tardiness_p95.value, 0.475)
        self.assertEqual(
            summary[SlaType.HARD].preferred_start_tardiness_p50.status,
            MetricStatus.NOT_APPLICABLE,
        )

    # AGG-003 / METRIC-005
    def test_system_green_absorption_includes_idle_formal_interval(self):
        graph = nx.Graph()
        graph.add_node("N")
        calendar = ReservationCalendar({"N": 10.0}, {})
        task = self._task("green", arrival=1.0, duration=1.0)
        reservation = self._reservation(task, 1.0)
        converter = TimeConverter(3600.0)
        accounting = ExogenousEnergyAccounting(
            converter,
            LinearPowerModel(5.0),
            {"N": self._forecast(100.0)},
            {"N": self._forecast(10.0, green=True)},
        )
        report = accounting.realize(
            (reservation,),
            accounting_interval=TimeInterval(0.0, 2.0),
        )
        self.assertEqual(report.system_green_supply_mwh, 20.0)
        self.assertEqual(report.total_task_attributed_green_energy_mwh, 5.0)
        self.assertEqual(report.system_green_idle_mwh, 15.0)
        self.assertEqual(report.system_green_absorption_rate.value, 0.25)

    # AGG-004
    def test_linear_percentile_fixture(self):
        self.assertEqual(linear_percentile((0, 10, 20, 30), 50).value, 15.0)
        self.assertEqual(linear_percentile((0, 10, 20, 30), 95).value, 28.5)

    # SCHEMA-001 / SCHEMA-002 / SCHEMA-003
    def test_formal_task_decision_and_metadata_records_are_traceable(self):
        _, _, _, runner = self._system()
        report = runner.run_frozen_policy(
            (self._task("traceable"),),
            arrival_cutoff_sim=1.0,
            seed=9,
        )
        task_record = report.task_records[0]
        self.assertEqual(task_record.final_state, "Completed")
        self.assertIsNone(task_record.transmission_start_sim)
        self.assertEqual(
            task_record.start_delay_sim,
            task_record.scheduler_queue_delay_sim
            + task_record.earliest_feasibility_lead_sim
            + task_record.active_wait_sim,
        )
        self.assertIsNotNone(task_record.task_attributed_cost_yuan)
        decision = report.decision_records[0]
        self.assertEqual(len(decision.candidate_set_hash), 64)
        self.assertTrue(decision.decision_id.startswith("decision-"))
        self.assertEqual(decision.commit_status, "COMMITTED")
        self.assertIsNotNone(decision.earliest_counterfactual_candidate_id)
        metadata = report.metadata
        self.assertEqual(
            {
                metadata.requirements_version,
                metadata.algorithm_version,
                metadata.task_schema_version,
                metadata.candidate_schema_version,
                metadata.model_schema_version,
                metadata.metric_schema_version,
                metadata.aggregation_schema_version,
            },
            {"1.0"},
        )

    # STAT-001 / STAT-002 / AGG-007
    def test_seed_pairing_uses_seed_ids_and_sample_ddof_one(self):
        baseline = {
            2: MetricValue.valid(5.0),
            1: MetricValue.valid(1.0),
            3: MetricValue.valid(9.0),
        }
        treatment = {
            3: MetricValue.valid(12.0),
            1: MetricValue.valid(2.0),
            2: MetricValue.valid(7.0),
        }
        summary = paired_t_summary(baseline, treatment)
        self.assertEqual(summary.seeds, (1, 2, 3))
        self.assertEqual(summary.differences, (1.0, 2.0, 3.0))
        self.assertEqual(summary.mean_difference.value, 2.0)
        self.assertEqual(summary.sample_standard_deviation.value, 1.0)

    # STAT-003 / METRIC-010 / STAT-011 / STAT-012
    def test_pair_invalidity_is_symmetric_and_bootstrap_is_reproducible(self):
        invalid = paired_t_summary(
            {1: MetricValue.valid(1.0), 2: MetricValue.not_applicable("missing")},
            {1: MetricValue.valid(2.0), 2: MetricValue.valid(3.0)},
        )
        self.assertEqual(invalid.status.value, "INVALID_PAIR")
        self.assertEqual(invalid.differences, ())
        first = paired_bootstrap((1.0, -1.0, 2.0), resample_count=500, random_seed=17)
        second = paired_bootstrap((1.0, -1.0, 2.0), resample_count=500, random_seed=17)
        self.assertEqual(first, second)
        self.assertEqual(relative_change(1.0, 0.0).status, MetricStatus.NOT_APPLICABLE)
        self.assertEqual(relative_change(-2.0, -1.0).status, MetricStatus.NOT_APPLICABLE)

    # LOAD-001 / LOAD-002 / AGG-005 / AGG-006 / METRIC-006
    def test_load_metrics_are_duration_weighted_and_separate_hotspot_from_overload(self):
        summary = summarize_load((
            UtilizationInterval(9.0, (0.2,)),
            UtilizationInterval(1.0, (1.0,)),
        ))
        self.assertAlmostEqual(summary.time_node_mean_utilization, 0.28)
        self.assertEqual(summary.weighted_p95_utilization, 1.0)
        self.assertEqual(summary.maximum_utilization, 1.0)
        self.assertEqual(summary.hotspot_time_ratio, 0.1)
        self.assertEqual(summary.physical_overcapacity_time_ratio, 0.0)
        overloaded = summarize_load((UtilizationInterval(1.0, (1.01, 0.0)),))
        self.assertEqual(overloaded.physical_overcapacity_time_ratio, 1.0)
        zero = summarize_load((UtilizationInterval(2.0, (0.0, 0.0)),))
        self.assertEqual(zero.time_weighted_node_cv, 0.0)


if __name__ == "__main__":
    unittest.main()
