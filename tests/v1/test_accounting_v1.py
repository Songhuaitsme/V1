import math
import unittest

import networkx as nx
import numpy as np

from v1.accounting import (
    ExogenousEnergyAccounting,
    ForecastSegment,
    LinearPowerModel,
    MetricsLedger,
    PiecewiseConstantForecast,
)
from v1.domain.models import MetricStatus, TaskSpec
from v1.domain.reservations import CommitStatus, ReservationRequest, TimeInterval
from v1.domain.units import TariffConverter, TimeConverter, UnitValidationError
from v1.scheduler.candidate_generator import CandidateGenerator
from v1.scheduler.path_provider import StaticPathProvider
from v1.scheduler.resource_calendar import ReservationCalendar
from v1.scheduler.transmission import TransmissionModel, build_path_spec
from v1.scheduler.v1_scheduler import V1Scheduler


class AccountingV1Test(unittest.TestCase):
    def _forecast(self, values, *, green=False):
        segments = [
            ForecastSegment(TimeInterval(start, end), value)
            for start, end, value in values
        ]
        if green:
            return PiecewiseConstantForecast.green_power_mw(segments)
        return PiecewiseConstantForecast.tariff_yuan_per_mwh(segments)

    def _accounting(
        self,
        *,
        seconds_per_sim=3600.0,
        tariff=((0.0, 20.0, 100.0),),
        green=((0.0, 20.0, 0.0),),
        power_per_cpu=1.0,
        billing=None,
    ):
        return ExogenousEnergyAccounting(
            TimeConverter(seconds_per_sim),
            LinearPowerModel(power_per_cpu),
            {"N": self._forecast(tariff)},
            {"N": self._forecast(green, green=True)},
            node_bill_rate_model=billing,
        )

    def _task(self, cpu=1.0, duration=1.0):
        return TaskSpec.create(
            task_id="candidate-task",
            arrival_time_sim=0.0,
            source_node="N",
            cpu_demand=cpu,
            execution_duration_sim=duration,
            data_size_mb=0.0,
            bandwidth_demand_mbps=1.0,
            sla_type="Hard",
            latest_start_limit_sim=20.0,
        )

    def _commit(self, calendar, task_id, cpu, start=0.0, end=1.0):
        graph = nx.Graph()
        graph.add_node("N")
        request = ReservationRequest(
            task_id=task_id,
            committed_candidate_id=f"candidate-{task_id}",
            committed_at_sim=0.0,
            reservation_snapshot_version=calendar.version,
            target_node="N",
            path=build_path_spec(graph, ["N"]),
            transmission_interval_sim=None,
            compute_interval_sim=TimeInterval(start, end),
            bandwidth_amount_mbps=1.0,
            cpu_amount=cpu,
        )
        result = calendar.try_commit(request, calendar.version)
        self.assertEqual(result.status, CommitStatus.COMMITTED)
        return result.reservation

    # COST-001 / COST-003 / COST-007 / COST-008 / COST-009
    def test_physical_power_energy_and_cost_hand_fixtures(self):
        accounting = self._accounting()
        empty = ReservationCalendar({"N": 100.0}, {}).snapshot()
        metrics = accounting.evaluate_candidate(
            task=self._task(cpu=1.0, duration=1.5),
            target_node="N",
            compute_start_sim=0.0,
            compute_end_sim=1.5,
            reservation_snapshot=empty,
        )
        self.assertEqual(metrics.task_energy_mwh, 1.5)
        self.assertEqual(metrics.task_direct_energy_cost_yuan, 150.0)
        self.assertEqual(metrics.candidate_marginal_system_cost_yuan, 150.0)
        self.assertEqual(LinearPowerModel(0.01).task_power_mw(100.0), 1.0)

        two_hours = accounting.evaluate_candidate(
            task=self._task(cpu=1.0, duration=2.0),
            target_node="N",
            compute_start_sim=0.0,
            compute_end_sim=2.0,
            reservation_snapshot=empty,
        )
        self.assertEqual(two_hours.task_energy_mwh, 2.0)
        self.assertEqual(two_hours.task_direct_energy_cost_yuan, 200.0)

    # COST-002
    def test_cost_integrates_every_tariff_slot(self):
        accounting = self._accounting(
            tariff=((0.0, 1.0, 100.0), (1.0, 3.0, 200.0)),
        )
        metrics = accounting.evaluate_candidate(
            task=self._task(duration=2.0),
            target_node="N",
            compute_start_sim=0.5,
            compute_end_sim=2.5,
            reservation_snapshot=ReservationCalendar({"N": 2.0}, {}).snapshot(),
        )
        self.assertEqual(metrics.task_direct_energy_cost_yuan, 350.0)

    def test_candidate_metric_batch_matches_scalar_integral_index(self):
        accounting = self._accounting(
            tariff=(
                (0.0, 1.0, 100.0),
                (1.0, 3.0, 200.0),
                (3.0, 20.0, 50.0),
            ),
            green=(
                (0.0, 2.0, 0.5),
                (2.0, 20.0, 2.0),
            ),
        )
        calendar = ReservationCalendar({"N": 4.0}, {})
        self._commit(calendar, "existing", 1.0, start=1.0, end=4.0)
        snapshot = calendar.snapshot()
        task = self._task(cpu=1.0, duration=1.5)
        graph = nx.Graph()
        graph.add_node("N")
        path = build_path_spec(graph, ["N"])
        starts = np.asarray((0.0, 0.5, 1.0, 2.5, 4.0))
        ends = starts + task.execution_duration_sim
        evaluator = accounting.candidate_metric_evaluator(snapshot)
        scalar = [
            evaluator(
                task=task,
                path=path,
                target_node="N",
                compute_start_sim=float(start),
                compute_end_sim=float(end),
                reservation_snapshot=snapshot,
            )
            for start, end in zip(starts, ends)
        ]
        batch = evaluator.evaluate_batch(
            task=task,
            path=path,
            target_node="N",
            compute_start_sim=starts,
            compute_end_sim=ends,
            reservation_snapshot=snapshot,
        )
        for key in (
            "system_cost_yuan",
            "green_coverage",
            "marginal_green_energy_mwh",
            "green_absorption_delta",
        ):
            np.testing.assert_allclose(
                batch[key],
                [row[key] for row in scalar],
                rtol=0.0,
                atol=1e-12,
            )
        np.testing.assert_array_equal(
            batch["green_opportunity"],
            [row["green_opportunity"] for row in scalar],
        )

    # COST-010 / COST-012
    def test_time_scaling_and_kwh_conversion_are_explicit(self):
        first = self._accounting(
            seconds_per_sim=1.0,
            tariff=((0.0, 4000.0, 100.0),),
            green=((0.0, 4000.0, 0.0),),
        )
        second = self._accounting(seconds_per_sim=300.0)
        snapshot = ReservationCalendar({"N": 2.0}, {}).snapshot()
        a = first.evaluate_candidate(
            task=self._task(duration=3600.0),
            target_node="N",
            compute_start_sim=0.0,
            compute_end_sim=3600.0,
            reservation_snapshot=snapshot,
        )
        b = second.evaluate_candidate(
            task=self._task(duration=12.0),
            target_node="N",
            compute_start_sim=0.0,
            compute_end_sim=12.0,
            reservation_snapshot=snapshot,
        )
        self.assertEqual(a.task_energy_mwh, b.task_energy_mwh)
        self.assertEqual(a.task_direct_energy_cost_yuan, b.task_direct_energy_cost_yuan)
        self.assertEqual(TariffConverter.yuan_per_kwh_to_yuan_per_mwh(1.0), 1000.0)

    # COST-006 / COST-018 / GREEN-009
    def test_strict_physical_validation_and_negative_tariff(self):
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(UnitValidationError):
                LinearPowerModel(invalid)
        for invalid in (-1.0, float("nan"), float("inf")):
            with self.subTest(green=invalid), self.assertRaises(UnitValidationError):
                self._forecast(((0.0, 1.0, invalid),), green=True)
        negative = self._accounting(tariff=((0.0, 20.0, -50.0),))
        metrics = negative.evaluate_candidate(
            task=self._task(),
            target_node="N",
            compute_start_sim=0.0,
            compute_end_sim=1.0,
            reservation_snapshot=ReservationCalendar({"N": 2.0}, {}).snapshot(),
        )
        self.assertEqual(metrics.task_direct_energy_cost_yuan, -50.0)

    # COST-015
    def test_exogenous_single_candidate_marginal_direct_and_attributed_match(self):
        calendar = ReservationCalendar({"N": 2.0}, {})
        reservation = self._commit(calendar, "one", 1.0)
        accounting = self._accounting(green=((0.0, 20.0, 1.0),))
        estimate = accounting.evaluate_candidate(
            task=self._task(),
            target_node="N",
            compute_start_sim=0.0,
            compute_end_sim=1.0,
            reservation_snapshot=ReservationCalendar({"N": 2.0}, {}).snapshot(),
        )
        realized = accounting.realize((reservation,)).task_records[0]
        self.assertEqual(estimate.candidate_marginal_system_cost_yuan, 100.0)
        self.assertEqual(realized.task_direct_energy_cost_yuan, 100.0)
        self.assertEqual(realized.task_attributed_cost_yuan, 100.0)

    # COST-016 / COST-017
    def test_state_dependent_bill_uses_counterfactual_and_order_free_attribution(self):
        def nonlinear_bill(**values):
            power = values["total_task_power_mw"]
            return values["tariff_yuan_per_mwh"] * power * (1.0 + power)

        accounting = self._accounting(billing=nonlinear_bill)
        calendar = ReservationCalendar({"N": 4.0}, {})
        first = self._commit(calendar, "a", 1.0)
        snapshot_with_one = calendar.snapshot()
        marginal = accounting.evaluate_candidate(
            task=self._task(),
            target_node="N",
            compute_start_sim=0.0,
            compute_end_sim=1.0,
            reservation_snapshot=snapshot_with_one,
        )
        self.assertEqual(marginal.candidate_marginal_system_cost_yuan, 400.0)
        self.assertNotEqual(marginal.candidate_marginal_system_cost_yuan, 300.0)
        second = self._commit(calendar, "b", 1.0)
        ab = accounting.realize((first, second))
        ba = accounting.realize((second, first))
        self.assertEqual(ab.task_records, ba.task_records)
        self.assertEqual(ab.node_bill_yuan["N"], 600.0)
        self.assertEqual(ab.total_task_attributed_cost_yuan, 600.0)
        self.assertEqual(
            [record.task_attributed_cost_yuan for record in ab.task_records],
            [300.0, 300.0],
        )

    # GREEN-001 through GREEN-005 / LEDGER-001
    def test_green_power_proportional_attribution_is_conservative_and_order_free(self):
        accounting = self._accounting(green=((0.0, 20.0, 6.0),))
        calendar = ReservationCalendar({"N": 20.0}, {})
        a = self._commit(calendar, "a", 4.0)
        b = self._commit(calendar, "b", 6.0)
        report = accounting.realize((a, b))
        reversed_report = accounting.realize((b, a))
        self.assertEqual(report.task_records, reversed_report.task_records)
        for actual, expected in zip(
            [record.task_attributed_green_energy_mwh for record in report.task_records],
            [2.4, 3.6],
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(report.total_task_attributed_green_energy_mwh, 6.0)
        self.assertEqual(report.node_green_used_mwh["N"], 6.0)
        self.assertEqual(report.total_task_attributed_cost_yuan, 1000.0)

        abundant = self._accounting(green=((0.0, 20.0, 20.0),)).realize((a, b))
        self.assertTrue(all(record.green_coverage == 1.0 for record in abundant.task_records))
        zero = self._accounting().realize((a, b))
        self.assertTrue(all(record.green_coverage == 0.0 for record in zero.task_records))

    # GREEN-006 / GREEN-008
    def test_marginal_green_and_final_attribution_remain_distinct(self):
        calendar = ReservationCalendar({"N": 20.0}, {})
        existing = self._commit(calendar, "existing", 6.0)
        accounting = self._accounting(green=((0.0, 20.0, 6.0),))
        candidate = accounting.evaluate_candidate(
            task=self._task(cpu=4.0),
            target_node="N",
            compute_start_sim=0.0,
            compute_end_sim=1.0,
            reservation_snapshot=calendar.snapshot(),
        )
        self.assertEqual(candidate.candidate_marginal_green_energy_mwh, 0.0)
        self.assertEqual(candidate.green_absorption_delta.value, 0.0)
        added = self._commit(calendar, "added", 4.0)
        realized = accounting.realize((existing, added))
        added_record = next(item for item in realized.task_records if item.task_id == "added")
        self.assertAlmostEqual(added_record.task_attributed_green_energy_mwh, 2.4)

    # Green zero opportunity semantics
    def test_zero_green_supply_is_not_applicable_for_absorption(self):
        metrics = self._accounting().evaluate_candidate(
            task=self._task(),
            target_node="N",
            compute_start_sim=0.0,
            compute_end_sim=1.0,
            reservation_snapshot=ReservationCalendar({"N": 2.0}, {}).snapshot(),
        )
        self.assertEqual(metrics.green_absorption_delta.status, MetricStatus.NOT_APPLICABLE)
        self.assertIsNone(metrics.green_absorption_delta.value)
        self.assertFalse(metrics.green_opportunity)

    # Candidate generator integration
    def test_candidate_schema_receives_physical_accounting_metrics(self):
        graph = nx.Graph()
        graph.add_node("N")
        calendar = ReservationCalendar({"N": 2.0}, {})
        generator = CandidateGenerator(
            ("N",),
            1.0,
            StaticPathProvider(graph, 1),
            TransmissionModel(TimeConverter(3600.0), 200000.0),
            calendar,
        )
        accounting = self._accounting(
            tariff=((0.0, 100.0, 100.0),),
            green=((0.0, 100.0, 1.0),),
        )
        candidate = generator.generate_complete(
            self._task(),
            0.0,
            metric_evaluator=accounting.candidate_metric_evaluator(calendar.snapshot()),
        ).candidates[0]
        self.assertEqual(candidate.estimated_candidate_marginal_system_cost_yuan, 100.0)
        self.assertEqual(candidate.estimated_green_coverage, 1.0)
        self.assertEqual(candidate.estimated_candidate_marginal_green_energy_mwh, 1.0)
        self.assertTrue(candidate.estimated_green_opportunity)

    def test_indexed_candidate_metrics_match_reference_interval_integrator(self):
        accounting = self._accounting(
            tariff=(
                (0.0, 1.0, 80.0),
                (1.0, 2.5, 120.0),
                (2.5, 6.0, 60.0),
                (6.0, 20.0, 100.0),
            ),
            green=(
                (0.0, 1.5, 2.0),
                (1.5, 4.0, 8.0),
                (4.0, 20.0, 1.0),
            ),
        )
        calendar = ReservationCalendar({"N": 20.0}, {})
        self._commit(calendar, "existing-a", 3.0, start=0.5, end=2.25)
        self._commit(calendar, "existing-b", 2.0, start=3.0, end=5.0)
        snapshot = calendar.snapshot()
        task = self._task(cpu=4.0, duration=1.75)
        graph = nx.Graph()
        graph.add_node("N")
        path = build_path_spec(graph, ["N"])
        indexed = accounting.candidate_metric_evaluator(snapshot)

        for start in (0.25, 0.75, 1.5, 2.5, 3.75, 5.0):
            end = start + task.execution_duration_sim
            expected = accounting.evaluate_candidate(
                task=task,
                target_node="N",
                compute_start_sim=start,
                compute_end_sim=end,
                reservation_snapshot=snapshot,
            ).as_candidate_metrics()
            actual = indexed(
                task=task,
                path=path,
                target_node="N",
                compute_start_sim=start,
                compute_end_sim=end,
                reservation_snapshot=snapshot,
            )
            for key in expected:
                if isinstance(expected[key], bool):
                    self.assertEqual(actual[key], expected[key])
                else:
                    self.assertAlmostEqual(actual[key], expected[key], places=10)

    def test_indexed_evaluator_falls_back_for_state_dependent_billing(self):
        def nonlinear_bill(**values):
            power = values["total_task_power_mw"]
            return values["tariff_yuan_per_mwh"] * power * (1.0 + power)

        accounting = self._accounting(billing=nonlinear_bill)
        calendar = ReservationCalendar({"N": 4.0}, {})
        self._commit(calendar, "existing", 1.0)
        snapshot = calendar.snapshot()
        task = self._task()
        graph = nx.Graph()
        graph.add_node("N")
        path = build_path_spec(graph, ["N"])
        expected = accounting.evaluate_candidate(
            task=task,
            target_node="N",
            compute_start_sim=0.0,
            compute_end_sim=1.0,
            reservation_snapshot=snapshot,
        ).as_candidate_metrics()
        actual = accounting.candidate_metric_evaluator(snapshot)(
            task=task,
            path=path,
            target_node="N",
            compute_start_sim=0.0,
            compute_end_sim=1.0,
            reservation_snapshot=snapshot,
        )
        self.assertEqual(actual, expected)

    # CONTRACT-016
    def test_metrics_ledger_is_idempotent_and_freezes_after_finalize(self):
        calendar = ReservationCalendar({"N": 2.0}, {})
        reservation = self._commit(calendar, "one", 1.0)
        ledger = MetricsLedger(self._accounting())
        self.assertTrue(ledger.record_completed_reservation(reservation))
        self.assertFalse(ledger.record_completed_reservation(reservation))
        first = ledger.finalize_after_full_settlement()
        self.assertIs(first, ledger.finalize_after_full_settlement())
        with self.assertRaises(RuntimeError):
            ledger.record_completed_reservation(reservation)

    def test_scheduler_writes_realized_ledger_only_on_completion(self):
        graph = nx.Graph()
        graph.add_node("N")
        calendar = ReservationCalendar({"N": 2.0}, {})
        generator = CandidateGenerator(
            ("N",),
            1.0,
            StaticPathProvider(graph, 1),
            TransmissionModel(TimeConverter(3600.0), 200000.0),
            calendar,
        )
        ledger = MetricsLedger(
            self._accounting(
                tariff=((0.0, 100.0, 100.0),),
                green=((0.0, 100.0, 1.0),),
            )
        )
        scheduler = V1Scheduler(calendar, generator, 10, 10, 3, metrics_ledger=ledger)
        scheduler.run_cycle(0.0, arrivals=(self._task(),))
        with self.assertRaises(RuntimeError):
            scheduler.finalize_metrics_after_full_settlement()
        scheduler.run_cycle(1.0)
        report = scheduler.finalize_metrics_after_full_settlement()
        self.assertEqual(len(report.task_records), 1)
        self.assertEqual(report.task_records[0].task_id, "candidate-task")


if __name__ == "__main__":
    unittest.main()
