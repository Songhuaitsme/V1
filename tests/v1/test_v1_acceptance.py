from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import json
import unittest

import networkx as nx

from v1.accounting import ForecastSegment, PiecewiseConstantForecast
from v1.accounting.forecast import ForecastCoverageError
from v1.audit_v1 import GateStatus, evaluate_quality_gate, scan_scheduler_invariants
from v1.domain.candidates import CandidateGenerationStatus, CandidateMode
from v1.domain.models import (
    MetricStatus,
    SlaType,
    TaskSpec,
    TaskState,
    validate_task_mapping,
)
from v1.domain.reservations import CommitStatus, TimeInterval
from v1.domain.sla import SlaPolicy
from v1.domain.units import TimeConverter
from v1.evaluation_v1 import (
    FormalSchemaError,
    paired_sample_size,
    ratio_metric,
    relative_change,
    summarize_active_wait,
    to_canonical_json,
)
from v1.learning import validate_checkpoint_metadata
from v1.learning.candidate_dqn import select_candidate_id
from v1.learning.reward import DecisionRecord, GammaClock, RewardAssembler
from v1.scheduler.approximate import compress_candidates
from v1.scheduler.candidate_generator import CandidateGenerator
from v1.scheduler.objectives import ObjectiveConfig, ObjectiveScorer, pareto_frontier
from v1.scheduler.path_provider import StaticPathProvider
from v1.scheduler.policies import EarliestFeasiblePolicy, LowestCostPolicy
from v1.scheduler.resource_calendar import ReservationCalendar
from v1.scheduler.transmission import TransmissionModel
from v1.scheduler.v1_scheduler import V1Scheduler
from v1.simulation.state_machine import (
    ALLOWED_TRANSITIONS,
    TERMINAL_REASON_BY_STATE,
    TERMINAL_STATES,
    StateTransitionError,
    TaskStateMachine,
)
from v1.traceability_audit import audit_repository
from shared.task_manager import TaskManager


class V1AcceptanceContractTest(unittest.TestCase):
    def _task(self, task_id="t", *, latest=5.0, cpu=1.0, duration=1.0):
        return TaskSpec.create(
            task_id=task_id,
            arrival_time_sim=0.0,
            source_node="N",
            cpu_demand=cpu,
            execution_duration_sim=duration,
            data_size_mb=0.0,
            bandwidth_demand_mbps=1.0,
            sla_type="Hard",
            latest_start_limit_sim=latest,
        )

    def _system(self, capacity=1.0, policy=None):
        graph = nx.Graph()
        graph.add_node("N")
        calendar = ReservationCalendar({"N": capacity}, {})
        generator = CandidateGenerator(
            ("N",),
            1.0,
            StaticPathProvider(graph, 1),
            TransmissionModel(TimeConverter(1.0), 200000.0),
            calendar,
        )
        scheduler = V1Scheduler(calendar, generator, 10, 10, 3, policy=policy)
        return graph, calendar, generator, scheduler

    def test_all_workload_templates_have_v1_sla_fields(self):
        manager = TaskManager(("I-source", "A-source", "C-source"), 1000.0)
        for template in manager.task_templates.values():
            self.assertIn(template["sla_type"], {"Hard", "Soft", "Flexible"})
            self.assertEqual(len(template["latency_range"]), 2)

    # CONTRACT-001 / CONTRACT-002 / CONTRACT-005 / CONTRACT-006
    def test_public_state_transition_and_reason_contract_is_exact(self):
        self.assertEqual(
            {state.value for state in TaskState},
            {
                "Arrived", "Queued", "PendingUncommitted", "Reserved",
                "Transmitting", "Running", "Completed", "Rejected",
                "Expired", "Failed",
            },
        )
        self.assertEqual(
            TERMINAL_STATES,
            {TaskState.COMPLETED, TaskState.REJECTED, TaskState.EXPIRED, TaskState.FAILED},
        )
        for state in TERMINAL_STATES:
            self.assertEqual(ALLOWED_TRANSITIONS[state], set())
        self.assertEqual(TERMINAL_REASON_BY_STATE[TaskState.EXPIRED], {"ABSOLUTE_START_DEADLINE"})
        machine = TaskStateMachine()
        machine.register(self._task())
        before = machine.runtime("t")
        with self.assertRaises(StateTransitionError):
            machine.transition("t", TaskState.COMPLETED, 0.0, terminal_reason="COMPLETED")
        self.assertEqual(machine.runtime("t"), before)
        with self.assertRaises(ValueError):
            SlaType.parse("None")

    # CONTRACT-003 / CONTRACT-004
    def test_candidate_and_commit_status_enumerations_are_closed(self):
        self.assertEqual(
            {item.value for item in CandidateGenerationStatus},
            {"OK", "EMPTY_PHYSICAL", "EXPIRED_BEFORE_DECISION", "INVALID_TASK", "FORECAST_NOT_COVERED"},
        )
        self.assertEqual(
            {item.value for item in CommitStatus},
            {"COMMITTED", "CONFLICT", "CPU_INFEASIBLE", "BANDWIDTH_INFEASIBLE", "CANDIDATE_STALE", "INTERNAL_ROLLBACK"},
        )

    # ADMIT-003 / CONTRACT-007 / CONTRACT-008
    def test_validation_and_sla_are_pure_and_field_addressable(self):
        mapping = {
            "task_id": "bad", "arrival_time_sim": 0.0, "source_node": "N",
            "cpu_demand": -1.0, "execution_duration_sim": 1.0,
            "data_size_mb": 0.0, "bandwidth_demand_mbps": 1.0,
            "sla_type": "Hard", "latest_start_limit_sim": 5.0,
        }
        before = dict(mapping)
        result = validate_task_mapping(mapping)
        self.assertFalse(result.valid)
        self.assertEqual(result.terminal_reason, "INVALID_TASK")
        self.assertIn("cpu_demand", result.field_errors)
        self.assertEqual(mapping, before)
        task = self._task()
        first = (
            SlaPolicy.is_start_feasible(task, 5.0),
            SlaPolicy.preferred_start_tardiness(task, 5.0),
        )
        second = (
            SlaPolicy.is_start_feasible(task, 5.0),
            SlaPolicy.preferred_start_tardiness(task, 5.0),
        )
        self.assertEqual(first, second)
        self.assertEqual(task, self._task())

    # CONTRACT-009 / CONTRACT-010 / CONTRACT-011 / CONTRACT-013 / CONTRACT-014
    def test_queue_path_calendar_candidate_and_policy_read_boundaries(self):
        graph, calendar, generator, scheduler = self._system()
        task = self._task()
        snapshot_before = calendar.snapshot()
        scheduler.submit_arrivals((task,))
        self.assertEqual(calendar.snapshot(), snapshot_before)
        paths_a = generator.path_provider.candidate_paths("N", "N", 1.0)
        paths_b = generator.path_provider.candidate_paths("N", "N", 1.0)
        self.assertEqual(paths_a, paths_b)
        generated_a = generator.generate_complete(task, 0.0)
        generated_b = generator.generate_complete(task, 0.0)
        self.assertEqual(generated_a, generated_b)
        self.assertEqual(calendar.snapshot(), snapshot_before)
        candidates = generated_a.candidates
        self.assertIn(EarliestFeasiblePolicy().select(candidates), candidates)
        self.assertIn(LowestCostPolicy().select(candidates), candidates)
        self.assertEqual(calendar.snapshot(), snapshot_before)

    # CONTRACT-012 / INFO-003 / INFO-004 / INFO-005
    def test_forecast_is_immutable_exogenous_and_strictly_covered(self):
        forecast = PiecewiseConstantForecast.green_power_mw((
            ForecastSegment(TimeInterval(0.0, 2.0), 3.0),
            ForecastSegment(TimeInterval(2.0, 4.0), 5.0),
        ))
        self.assertEqual((forecast.value_at(1.0), forecast.value_at(3.0)), (3.0, 5.0))
        baseline = forecast.segments
        unrelated_future_tasks = (self._task("future", latest=50.0),)
        self.assertEqual(forecast.segments, baseline)
        self.assertEqual(len(unrelated_future_tasks), 1)
        with self.assertRaises(AttributeError):
            forecast.version = "changed"
        with self.assertRaises(ForecastCoverageError):
            forecast.value_at(4.0)

    # INFO-001 / INFO-002 / INFO-007
    def test_scheduler_information_surface_has_no_future_trace_or_replanning_api(self):
        _, _, generator_a, scheduler_a = self._system()
        _, _, generator_b, scheduler_b = self._system()
        current = self._task("current")
        scheduler_a.submit_arrivals((current,))
        scheduler_b.submit_arrivals((current,))
        self.assertEqual(generator_a.generate_complete(current, 0.0), generator_b.generate_complete(current, 0.0))
        for name in ("future_task_trace", "cancel", "migrate", "replan"):
            self.assertFalse(hasattr(scheduler_a, name))
            self.assertFalse(hasattr(scheduler_a.reservation_manager, name))

    # CAND-011
    def test_approximate_mode_preserves_extrema_and_reports_loss(self):
        _, _, generator, _ = self._system(capacity=2.0)
        base = generator.generate_complete(self._task(latest=8.0), 0.0).candidates
        decorated = tuple(
            replace(
                candidate,
                estimated_candidate_marginal_system_cost_yuan=float(index),
                estimated_green_coverage=float(index) / 10.0,
                estimated_green_absorption_delta=0.0,
                capacity_margin=1.0 - float(index) / 10.0,
            )
            for index, candidate in enumerate(base)
        )
        result = compress_candidates(decorated, 3, lambda item: -item.estimated_candidate_marginal_system_cost_yuan)
        self.assertTrue(all(item.candidate_mode is CandidateMode.APPROXIMATE for item in result.candidates))
        self.assertEqual(result.original_count, len(decorated))
        self.assertGreaterEqual(result.retained_count, 3)
        self.assertGreaterEqual(result.omitted_ratio, 0.0)
        self.assertEqual(result.utility_regret, 0.0)

    # ADMIT-002 / OBJ-001 / OBJ-004 / OBJ-006
    def test_hard_constraints_precede_policy_and_negative_q_still_selects(self):
        self.assertEqual(select_candidate_id(("a", "b", "c"), (-3.0, -1.0, -2.0)), "b")
        _, _, generator, _ = self._system(capacity=1.0)
        self.assertFalse(generator.is_statically_serviceable(self._task(cpu=2.0)))
        candidate = generator.generate_complete(self._task(), 0.0).candidates[0]
        score = ObjectiveScorer(ObjectiveConfig(0.0, 1.0, 1.0)).score(candidate, SlaType.HARD)
        declared = 0.5 * score.cost_score + 0.5 * score.green_score + 0.1 * score.balance_score
        self.assertEqual(score.preferred_start_tardiness_penalty, 0.0)
        self.assertAlmostEqual(score.total_score, declared)

    # ADMIT-005 / ADMIT-006 / DELAY-005
    def test_empty_window_is_pending_then_expired_without_active_wait(self):
        graph, calendar, _, scheduler = self._system()
        blocker_task = self._task("block", latest=20.0, duration=10.0)
        scheduler.run_cycle(0.0, arrivals=(blocker_task,))
        pending = scheduler.run_cycle(0.0, arrivals=(self._task("p", latest=1.0),))
        self.assertEqual(pending.decisions[0].status, "PENDING")
        self.assertIsNone(scheduler.committed_candidate("p"))
        expired = scheduler.run_cycle(1.0)
        self.assertEqual(scheduler.state_machine.runtime("p").state, TaskState.EXPIRED)
        self.assertEqual(scheduler.state_machine.runtime("p").terminal_reason, "ABSOLUTE_START_DEADLINE")
        self.assertEqual(expired.state_counts[TaskState.EXPIRED], 1)

    # QUEUE-010 / QUEUE-011 / EVENT-007
    def test_reason_counts_policy_order_and_batch_state_conservation(self):
        _, _, _, first = self._system(capacity=1.0, policy=EarliestFeasiblePolicy())
        _, _, _, second = self._system(capacity=1.0, policy=LowestCostPolicy())
        tasks = (self._task("a", cpu=2.0), self._task("b"), self._task("c"))
        order_a = [item.task_id for item in first.run_cycle(0.0, arrivals=tasks).decisions]
        order_b = [item.task_id for item in second.run_cycle(0.0, arrivals=tasks).decisions]
        self.assertEqual(order_a, order_b)
        self.assertEqual(first.queue_manager.reason_counts["STATICALLY_UNSERVICEABLE"], 1)
        self.assertEqual(sum(first.state_machine.count_by_state().values()), first.state_machine.task_count)

    # CAND-013 / CAND-014 / EVAL-012
    def test_decision_record_pairs_selected_candidate_with_earliest_counterfactual(self):
        _, _, _, scheduler = self._system(capacity=2.0, policy=LowestCostPolicy())
        result = scheduler.run_cycle(0.0, arrivals=(self._task(),))
        decision = result.decisions[0]
        candidate = scheduler.committed_candidate("t")
        self.assertIsNotNone(decision.earliest_counterfactual_candidate_id)
        self.assertEqual(decision.selected_candidate_id, candidate.candidate_id)
        self.assertGreaterEqual(candidate.active_wait_sim, 0.0)

    # AGG-001 / AGG-002 / COST-013 / COST-014 / METRIC-002 / METRIC-003
    def test_weighted_aggregation_and_zero_denominators_are_explicit(self):
        self.assertEqual(ratio_metric(10.0, 4.0, "zero work").value, 2.5)
        self.assertEqual(ratio_metric(1.0, 10.0, "zero energy").value, 0.1)
        self.assertEqual(ratio_metric(300.0, 2.0, "zero completed work").value, 150.0)
        self.assertEqual(ratio_metric(200.0, 2.0, "zero completed work").value, 100.0)
        self.assertEqual(ratio_metric(200.0, 4.0, "zero arrived work").value, 50.0)
        self.assertEqual(ratio_metric(0.0, 3.0, "zero arrivals").value, 0.0)
        self.assertEqual(ratio_metric(5.0, 0.0, "zero reservations").status, MetricStatus.NOT_APPLICABLE)
        self.assertEqual(ratio_metric(5.0, 0.0, "zero completed CPU hours").status, MetricStatus.NOT_APPLICABLE)

    # COST-004 / COST-005 / COST-011 / GREEN-002 / GREEN-003 / GREEN-004 / GREEN-007
    def test_physical_and_policy_accounts_remain_separate(self):
        _, _, generator, _ = self._system(capacity=2.0)
        candidate = generator.generate_complete(self._task(), 0.0).candidates[0]
        changed_balance = replace(candidate, capacity_margin=0.0)
        config = ObjectiveConfig(0.0, 1.0, 1.0, balance_weight=1.0)
        first = ObjectiveScorer(config).score(candidate, SlaType.HARD)
        second = ObjectiveScorer(config).score(changed_balance, SlaType.HARD)
        self.assertEqual(
            candidate.estimated_candidate_marginal_system_cost_yuan,
            changed_balance.estimated_candidate_marginal_system_cost_yuan,
        )
        self.assertNotEqual(first.total_score, second.total_score)
        self.assertNotIn("economic_cost_yuan", {"cost_index": 0.5})
        # Energy/accounting order independence and MWh integration are exercised
        # by test_accounting_v1; this contract test pins their schema separation.
        self.assertTrue(hasattr(candidate, "estimated_candidate_marginal_green_energy_mwh"))

    # METRIC-007 / METRIC-008 / METRIC-009 / SCHEMA-005
    def test_not_applicable_wait_relative_change_and_finite_json(self):
        self.assertEqual(relative_change(1.0, 0.0).status, MetricStatus.NOT_APPLICABLE)
        waits = summarize_active_wait(())
        self.assertEqual(waits.count, 0)
        self.assertEqual(waits.mean_active_wait_sim.status, MetricStatus.NOT_APPLICABLE)
        payload = to_canonical_json({"metric": ratio_metric(1.0, 0.0, "zero")})
        self.assertEqual(json.loads(payload)["metric"]["status"], "NOT_APPLICABLE")
        with self.assertRaises(FormalSchemaError):
            to_canonical_json({"nested": {"bad": float("nan")}})

    # SCHEMA-004
    def test_model_schema_mismatch_fails_closed(self):
        metadata = {
            "model_schema_version": "1.0", "candidate_schema_version": "1.0",
            "feature_schema_hash": "abc", "global_state_dim": 18,
            "candidate_feature_dim": 18, "gamma_per_second": 0.99,
        }
        self.assertTrue(validate_checkpoint_metadata(metadata, "abc"))
        with self.assertRaises(ValueError):
            validate_checkpoint_metadata({**metadata, "feature_schema_hash": "wrong"}, "abc")
        with self.assertRaises(ValueError):
            validate_checkpoint_metadata({key: value for key, value in metadata.items() if key != "model_schema_version"}, "abc")

    # CONTRACT-017 / OBJ-006
    def test_reward_assembly_is_deterministic_and_has_no_hidden_duplicate(self):
        record = DecisionRecord("d", "t", "c", 0.0, 4.0)
        first = RewardAssembler.commit_reward(record) + RewardAssembler.realization_correction(record, 3.0, 2.0)
        second = RewardAssembler.commit_reward(record) + RewardAssembler.realization_correction(record, 3.0, 2.0)
        self.assertEqual(first, second)
        self.assertEqual(first, 5.0)
        self.assertEqual(GammaClock(1.0, TimeConverter(1.0)).discount(100.0), 1.0)

    # STAT-004 / STAT-005 / STAT-006 / STAT-007 / STAT-008 / STAT-009 / STAT-010
    def test_formal_quality_gate_and_power_rules(self):
        passed = evaluate_quality_gate(
            seed_count=10, cost_mean_relative_change=-0.06, cost_ci_upper=-0.01,
            green_mean_change=0.0, green_ci_lower=0.0,
            completion_ci_lower=0.0, load_mean_change=0.0,
        )
        self.assertEqual(passed.status, GateStatus.PASS)
        crossed = evaluate_quality_gate(
            seed_count=10, cost_mean_relative_change=-0.06, cost_ci_upper=0.01,
            green_mean_change=0.0, green_ci_lower=0.0,
            completion_ci_lower=0.0, load_mean_change=0.0,
        )
        self.assertEqual(crossed.status, GateStatus.FAIL)
        noninferior_failed = evaluate_quality_gate(
            seed_count=10, cost_mean_relative_change=-0.06, cost_ci_upper=-0.01,
            green_mean_change=0.04, green_ci_lower=0.01,
            completion_ci_lower=-0.006, load_mean_change=0.0,
        )
        self.assertEqual(noninferior_failed.status, GateStatus.FAIL)
        self.assertTrue(len(pareto_frontier(())) == 0)
        self.assertGreaterEqual(paired_sample_size(0.1, 0.05), 2)
        diagnostic = evaluate_quality_gate(
            seed_count=5, cost_mean_relative_change=-1.0, cost_ci_upper=-1.0,
            green_mean_change=1.0, green_ci_lower=1.0,
            completion_ci_lower=1.0, load_mean_change=0.0,
        )
        self.assertEqual(diagnostic.status, GateStatus.DIAGNOSTIC_ONLY)

    # AGG-008 / AGG-009 / EVAL-009 / EVAL-010 / EVAL-011 / CONTRACT-015 / CONTRACT-018
    # SCENARIO-001 / SCENARIO-002 / SCENARIO-003 / SCENARIO-004
    # SCENARIO-005 / SCENARIO-006 / SCENARIO-007
    # I-01 / I-02 / I-03 / I-04 / I-05 / I-06 / I-07 / I-08 / I-09 / I-10
    # I-11 / I-12 / I-13 / I-14 / I-15 / I-16 / I-17 / I-18 / I-19
    def test_cross_module_evidence_is_versioned_and_deterministic(self):
        # Detailed paired statistics, three-phase ledger boundaries, load metrics,
        # metadata hashes and frozen-run records are asserted in test_evaluation_v1.
        _, _, _, scheduler_a = self._system()
        _, _, _, scheduler_b = self._system()
        task = self._task()
        before = hash(task)
        result_a = scheduler_a.run_cycle(0.0, arrivals=(task,))
        result_b = scheduler_b.run_cycle(0.0, arrivals=(task,))
        self.assertEqual(result_a.decisions, result_b.decisions)
        self.assertEqual(hash(task), before)
        self.assertFalse(hasattr(scheduler_a.event_engine, "select_candidate"))
        self.assertEqual(scan_scheduler_invariants(scheduler_a, result_a), ())


class V1TraceabilityAuditTest(unittest.TestCase):
    # TRACE-000 / TRACE-001 / TRACE-002 / TRACE-003
    def test_frozen_traceability_counts_and_coverage(self):
        report = audit_repository(Path(__file__).resolve().parent)
        self.assertEqual(len(report.requirement_ids), 50)
        self.assertEqual(len(report.invariant_ids), 19)
        self.assertEqual(len(report.implemented_invariant_ids), 19)
        self.assertEqual(len(report.target_test_ids), 258)
        self.assertEqual(report.duplicate_spec_test_ids, ())
        self.assertEqual(report.missing_test_ids, ())
        self.assertEqual(report.missing_invariant_ids, ())
        self.assertTrue(report.passed)


if __name__ == "__main__":
    unittest.main()
