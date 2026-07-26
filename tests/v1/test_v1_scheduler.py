from dataclasses import replace
import unittest

import networkx as nx

from v1.domain.models import SlaType, TaskSpec, TaskState
from v1.domain.reservations import CommitStatus, ReservationRequest, TimeInterval
from v1.domain.units import TimeConverter
from v1.scheduler.candidate_generator import CandidateGenerator
from v1.scheduler.path_provider import StaticPathProvider
from v1.scheduler.policies import (
    EarliestFeasiblePolicy,
    EqualWeightPolicy,
    HighestGreenPolicy,
    LowestCostPolicy,
)
from v1.scheduler.objectives import (
    ObjectiveConfig,
    ObjectiveScorer,
    pareto_frontier,
)
from v1.scheduler.resource_calendar import ReservationCalendar
from v1.scheduler.transmission import TransmissionModel, build_path_spec
from v1.scheduler.v1_scheduler import V1Scheduler


class V1SchedulerEndToEndTest(unittest.TestCase):
    def _build(self, capacity=1.0, max_tasks=10, max_queue=20):
        graph = nx.Graph()
        graph.add_node("L")
        calendar = ReservationCalendar({"L": capacity}, {})
        generator = CandidateGenerator(
            ("L",),
            1.0,
            StaticPathProvider(graph, max_paths_per_target=1),
            TransmissionModel(TimeConverter(1.0), 200000.0),
            calendar,
        )
        scheduler = V1Scheduler(
            calendar,
            generator,
            max_queue_length=max_queue,
            max_tasks_per_cycle=max_tasks,
            max_commit_attempts_per_decision=3,
        )
        return graph, calendar, generator, scheduler

    def _task(
        self,
        task_id,
        arrival=0.0,
        latest=10.0,
        duration=1.0,
        cpu=1.0,
    ):
        return TaskSpec.create(
            task_id=task_id,
            arrival_time_sim=arrival,
            source_node="L",
            cpu_demand=cpu,
            execution_duration_sim=duration,
            data_size_mb=0.0,
            bandwidth_demand_mbps=1.0,
            sla_type="Hard",
            latest_start_limit_sim=latest,
        )

    def _block(self, graph, calendar, start=0.0, end=10.0):
        request = ReservationRequest(
            task_id="external-blocker",
            committed_candidate_id="external-candidate",
            committed_at_sim=0.0,
            reservation_snapshot_version=calendar.version,
            target_node="L",
            path=build_path_spec(graph, ["L"]),
            transmission_interval_sim=None,
            compute_interval_sim=TimeInterval(start, end),
            bandwidth_amount_mbps=1.0,
            cpu_amount=1.0,
        )
        result = calendar.try_commit(request, calendar.version)
        self.assertEqual(result.status, CommitStatus.COMMITTED)
        return result.reservation

    # EVENT-005 / LIFE-003 / ADMIT-001
    def test_local_commit_starts_in_same_timestamp_and_completes_at_end(self):
        _, calendar, _, scheduler = self._build()
        task = self._task("t", duration=2.0)
        result = scheduler.run_cycle(0.0, arrivals=(task,))
        self.assertEqual(result.decisions[0].status, "RESERVED")
        self.assertEqual(
            [event.event_type for event in result.domain_events],
            ["RESERVATION_COMMITTED", "COMPUTE_STARTED"],
        )
        self.assertEqual(scheduler.state_machine.runtime("t").state, TaskState.RUNNING)
        self.assertEqual(calendar.active_reservation_count, 1)

        completed = scheduler.run_cycle(2.0)
        self.assertEqual(
            [event.event_type for event in completed.domain_events],
            ["TASK_COMPLETED"],
        )
        self.assertEqual(
            scheduler.state_machine.runtime("t").state,
            TaskState.COMPLETED,
        )
        self.assertEqual(calendar.active_reservation_count, 0)

    # EVENT-006 / R-11
    def test_each_same_cycle_task_sees_previous_atomic_commit(self):
        _, calendar, _, scheduler = self._build(capacity=2.0)
        tasks = (self._task("a"), self._task("b"))
        result = scheduler.run_cycle(0.0, arrivals=tasks)
        self.assertEqual([item.task_id for item in result.decisions], ["a", "b"])
        self.assertEqual([item.status for item in result.decisions], ["RESERVED", "RESERVED"])
        self.assertEqual(calendar.version, 2)
        self.assertEqual(calendar.active_reservation_count, 2)

    # QUEUE-009 / R-41
    def test_processing_cap_leaves_remaining_task_queued(self):
        _, _, _, scheduler = self._build(capacity=2.0, max_tasks=1)
        result = scheduler.run_cycle(
            0.0,
            arrivals=(self._task("a"), self._task("b")),
        )
        self.assertEqual(len(result.decisions), 1)
        self.assertEqual(result.state_counts[TaskState.QUEUED], 1)

    # Arrival boundary
    def test_future_arrival_is_not_scheduled_early(self):
        _, _, _, scheduler = self._build()
        task = self._task("future", arrival=2.0, latest=5.0)
        scheduler.submit_arrivals((task,))
        self.assertEqual(scheduler.run_cycle(0.0).decisions, ())
        self.assertEqual(scheduler.run_cycle(2.0).decisions[0].task_id, "future")

    # ADMIT-004
    def test_static_cpu_impossibility_is_deterministically_rejected(self):
        _, _, _, scheduler = self._build(capacity=1.0)
        transition = scheduler.submit_arrivals((self._task("huge", cpu=2.0),))[0]
        self.assertEqual(transition.new_state, TaskState.REJECTED)
        self.assertEqual(transition.terminal_reason, "STATICALLY_UNSERVICEABLE")
        self.assertEqual(scheduler.run_cycle(0.0).decisions, ())

    # LIFE-008 / LIFE-009 / CAND-015
    def test_pending_is_not_researched_without_physical_change(self):
        graph, _, _, scheduler = self._build()
        self._block(graph, scheduler.calendar)
        first = scheduler.run_cycle(0.0, arrivals=(self._task("p", latest=5.0),))
        self.assertEqual(first.decisions[0].status, "PENDING")
        self.assertEqual(scheduler.state_machine.runtime("p").pending_attempts, 1)
        second = scheduler.run_cycle(1.0)
        self.assertEqual(second.decisions, ())
        self.assertEqual(scheduler.state_machine.runtime("p").pending_attempts, 1)

    # EVENT-002 / LIFE-010
    def test_capacity_release_reactivates_pending_at_closed_deadline(self):
        graph, calendar, _, scheduler = self._build()
        blocker = self._block(graph, calendar)
        scheduler.run_cycle(0.0, arrivals=(self._task("p", latest=5.0),))
        calendar.release_on_failure(blocker.reservation_id)
        boundary = scheduler.run_cycle(
            5.0,
            physical_event_types=("RESERVATION_RELEASED",),
        )
        self.assertEqual(boundary.decisions[0].status, "RESERVED")
        self.assertEqual(
            scheduler.state_machine.runtime("p").state,
            TaskState.RUNNING,
        )

    # EVENT-003 / LIFE-011
    def test_pending_expires_once_after_failed_boundary_opportunity(self):
        graph, _, _, scheduler = self._build()
        self._block(graph, scheduler.calendar)
        scheduler.run_cycle(0.0, arrivals=(self._task("p", latest=5.0),))
        boundary = scheduler.run_cycle(5.0)
        self.assertEqual(boundary.decisions, ())
        self.assertEqual(boundary.state_transitions[-1].new_state, TaskState.EXPIRED)
        self.assertEqual(scheduler.run_cycle(5.0).state_transitions, ())

    # EVENT-001 / EVENT-004 / EVENT-008
    def test_completion_precedes_same_timestamp_arrival_and_repeat_is_idempotent(self):
        _, _, _, scheduler = self._build()
        scheduler.run_cycle(0.0, arrivals=(self._task("old", duration=1.0),))
        batch = scheduler.run_cycle(1.0, arrivals=(self._task("new", arrival=1.0),))
        self.assertEqual(batch.domain_events[0].event_type, "TASK_COMPLETED")
        self.assertEqual(batch.decisions[0].task_id, "new")
        repeated = scheduler.run_cycle(1.0)
        self.assertEqual(repeated.domain_events, ())
        self.assertEqual(repeated.decisions, ())

    def test_scheduler_rejects_mismatched_candidate_calendar(self):
        graph, calendar, generator, _ = self._build()
        other = ReservationCalendar({"L": 1.0}, {})
        with self.assertRaises(ValueError):
            V1Scheduler(other, generator, 10, 10, 3)


class CandidatePolicyTest(unittest.TestCase):
    def _candidates(self):
        _, _, generator, _ = V1SchedulerEndToEndTest()._build(capacity=2.0)
        task = V1SchedulerEndToEndTest()._task("policy", cpu=1.0)
        base = generator.generate_complete(task, 0.0).candidates[0]
        early_expensive = replace(
            base,
            candidate_id="early-expensive",
            estimated_candidate_marginal_system_cost_yuan=10.0,
            estimated_green_coverage=0.9,
            estimated_green_absorption_delta=0.9,
            capacity_margin=0.2,
        )
        late_cheap = replace(
            base,
            candidate_id="late-cheap",
            compute_start_sim=1.0,
            compute_end_sim=2.0,
            estimated_candidate_marginal_system_cost_yuan=1.0,
            estimated_green_coverage=0.1,
            estimated_green_absorption_delta=0.1,
            capacity_margin=0.8,
        )
        return early_expensive, late_cheap

    def test_deterministic_policies_always_select_from_nonempty_candidates(self):
        early, late = self._candidates()
        candidates = (late, early)
        self.assertIs(EarliestFeasiblePolicy().select(candidates), early)
        self.assertIs(LowestCostPolicy().select(candidates), late)
        self.assertIs(HighestGreenPolicy().select(candidates), early)
        self.assertIn(EqualWeightPolicy().select(candidates), candidates)
        for policy in (
            EarliestFeasiblePolicy(),
            LowestCostPolicy(),
            HighestGreenPolicy(),
            EqualWeightPolicy(),
        ):
            with self.assertRaises(ValueError):
                policy.select(())

    # OBJ-002 / OBJ-005 / OBJ-007
    def test_objective_uses_frozen_scales_and_declared_components_once(self):
        base, _ = self._candidates()
        candidate = replace(
            base,
            estimated_candidate_marginal_system_cost_yuan=2.0,
            estimated_green_coverage=0.4,
            estimated_green_absorption_delta=0.25,
            estimated_green_opportunity=True,
            capacity_margin=0.0,
            preferred_start_tardiness_ratio=0.0,
        )
        config = ObjectiveConfig(
            reference_marginal_cost_yuan=10.0,
            cost_scale_yuan=10.0,
            absorption_delta_scale=0.5,
            cost_weight=0.5,
            green_weight=0.5,
            balance_weight=0.0,
        )
        score = ObjectiveScorer(config).score(candidate, SlaType.HARD)
        self.assertEqual(score.cost_score, 0.8)
        self.assertEqual(score.green_absorption_score, 0.5)
        self.assertEqual(score.green_score, 0.45)
        self.assertAlmostEqual(score.total_score, 0.625)
        unrelated = replace(candidate, candidate_id="extreme", estimated_candidate_marginal_system_cost_yuan=9999.0)
        self.assertEqual(
            ObjectiveScorer(config).score(candidate, SlaType.HARD),
            score,
        )
        self.assertNotEqual(candidate.candidate_id, unrelated.candidate_id)

    # OBJ-003 / OBJ-010
    def test_objective_preferences_have_stable_ids_and_validate_scales(self):
        ids = {
            ObjectiveConfig(0.0, 1.0, 1.0, cost_weight=cost, green_weight=green).policy_id
            for cost, green in ((0.8, 0.2), (0.5, 0.5), (0.2, 0.8))
        }
        self.assertEqual(len(ids), 3)
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(Exception):
                ObjectiveConfig(0.0, invalid, 1.0)
            with self.subTest(absorption=invalid), self.assertRaises(Exception):
                ObjectiveConfig(0.0, 1.0, invalid)

    # OBJ-008
    def test_flexible_linear_tardiness_weight_is_independent_from_soft(self):
        base, _ = self._candidates()
        config = ObjectiveConfig(
            0.0,
            1.0,
            1.0,
            cost_weight=0.0,
            green_weight=1.0,
            balance_weight=0.0,
            soft_tardiness_weight=9.0,
            flexible_tardiness_weight=2.0,
        )
        scorer = ObjectiveScorer(config)
        penalties = []
        for ratio in (0.0, 0.5, 1.0):
            candidate = replace(
                base,
                preferred_start_tardiness_ratio=ratio,
                preferred_start_tardiness_applicable=True,
            )
            penalties.append(
                scorer.score(candidate, SlaType.FLEXIBLE).preferred_start_tardiness_penalty
            )
        self.assertEqual(penalties, [0.0, 1.0, 2.0])
        candidate = replace(
            base,
            preferred_start_tardiness_ratio=0.5,
            preferred_start_tardiness_applicable=True,
        )
        self.assertEqual(
            scorer.score(candidate, SlaType.SOFT).preferred_start_tardiness_penalty,
            4.5,
        )
        self.assertEqual(
            scorer.score(candidate, SlaType.HARD).preferred_start_tardiness_penalty,
            0.0,
        )

    # OBJ-009
    def test_pareto_frontier_matches_cost_green_dominance(self):
        base, _ = self._candidates()
        cheap = replace(
            base,
            candidate_id="cheap",
            estimated_candidate_marginal_system_cost_yuan=1.0,
            estimated_green_coverage=0.2,
            estimated_green_absorption_delta=0.0,
        )
        green = replace(
            base,
            candidate_id="green",
            estimated_candidate_marginal_system_cost_yuan=3.0,
            estimated_green_coverage=0.9,
            estimated_green_absorption_delta=0.0,
        )
        dominated = replace(
            base,
            candidate_id="dominated",
            estimated_candidate_marginal_system_cost_yuan=4.0,
            estimated_green_coverage=0.1,
            estimated_green_absorption_delta=0.0,
        )
        self.assertEqual(
            [item.candidate_id for item in pareto_frontier((dominated, green, cheap))],
            ["cheap", "green"],
        )


if __name__ == "__main__":
    unittest.main()
