import unittest

import networkx as nx

from v1.domain.candidates import CandidateGenerationStatus, CandidateMode
from v1.domain.models import SlaType, TaskSpec, TaskState
from v1.domain.reservations import CommitStatus, ReservationRequest, TimeInterval
from v1.domain.units import TimeConverter
from v1.scheduler.candidate_generator import CandidateGenerator, complete_time_grid
from v1.scheduler.path_provider import StaticPathProvider
from v1.scheduler.queue_manager import TaskQueueManager, queue_order_key
from v1.scheduler.reservation_manager import CommitDecisionStatus, ReservationManager
from v1.scheduler.resource_calendar import ReservationCalendar
from v1.scheduler.transmission import TransmissionModel, build_path_spec
from v1.simulation.event_engine import EventEngine
from v1.simulation.state_machine import TaskStateMachine


class CandidateAndQueueTest(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_edge("S", "N", capacity=100.0, distance_km=0.0, cost=1.0)
        self.graph.add_node("L")

    def _task(
        self,
        task_id="t",
        sla="Hard",
        arrival=0.0,
        preferred=None,
        latest=10.0,
        source="S",
        data=100.0,
        bw=100.0,
        duration=1.0,
        cpu=1.0,
    ):
        if sla != "Hard":
            latest = None
            preferred = 10.0 if preferred is None else preferred
        return TaskSpec.create(
            task_id=task_id,
            arrival_time_sim=arrival,
            source_node=source,
            cpu_demand=cpu,
            execution_duration_sim=duration,
            data_size_mb=data,
            bandwidth_demand_mbps=bw,
            sla_type=sla,
            preferred_start_limit_sim=preferred,
            latest_start_limit_sim=latest,
        )

    def _generator(
        self,
        compute_nodes=("N",),
        cycle=1.0,
        cpu_capacity=100.0,
        link_capacity=100.0,
        calendar=None,
    ):
        calendar = calendar or ReservationCalendar(
            {node: cpu_capacity for node in compute_nodes},
            {("S", "N"): link_capacity},
        )
        provider = StaticPathProvider(self.graph, max_paths_per_target=1)
        model = TransmissionModel(TimeConverter(1.0), 200000.0)
        return CandidateGenerator(
            compute_nodes,
            cycle,
            provider,
            model,
            calendar,
        ), calendar

    # CAND-001
    def test_cand_001_hard_never_exceeds_absolute_boundary(self):
        generator, _ = self._generator()
        task = self._task(latest=10.0)
        result = generator.generate_complete(task, 0.0)
        self.assertEqual(result.status, CandidateGenerationStatus.OK)
        self.assertTrue(all(c.compute_start_sim <= 10.0 for c in result.candidates))

    # CAND-002
    def test_cand_002_soft_includes_one_point_two_l_boundary(self):
        generator, _ = self._generator()
        task = self._task(sla="Soft", preferred=10.0)
        starts = [c.compute_start_sim for c in generator.generate_complete(task, 0.0).candidates]
        self.assertIn(12.0, starts)
        self.assertEqual(max(starts), 12.0)

    # CAND-003
    def test_cand_003_flexible_complete_window_ends_at_one_point_five_l(self):
        generator, _ = self._generator()
        task = self._task(sla="Flexible", preferred=10.0)
        result = generator.generate_complete(task, 0.0)
        starts = [candidate.compute_start_sim for candidate in result.candidates]
        self.assertEqual(max(starts), 15.0)
        self.assertEqual(result.candidate_mode, CandidateMode.COMPLETE)

    def test_complete_stream_matches_materialized_compatibility_result(self):
        generator, calendar = self._generator()
        task = self._task(sla="Soft", preferred=10.0)
        snapshot = calendar.snapshot()
        stream = generator.prepare_complete_stream(
            task, 0.0, reservation_snapshot=snapshot
        )
        streamed = tuple(stream.iter_candidates())
        materialized = generator.generate_complete(
            task, 0.0, reservation_snapshot=snapshot
        )
        self.assertEqual(
            tuple(item.candidate_id for item in streamed),
            tuple(item.candidate_id for item in materialized.candidates),
        )
        self.assertEqual(stream.feasible_candidate_count, len(streamed))
        self.assertEqual(stream.theoretical_slot_count, materialized.theoretical_slot_count)

    def test_unallocated_fast_path_matches_general_feasibility_path(self):
        for sla in ("Hard", "Soft", "Flexible"):
            with self.subTest(sla=sla):
                generator, calendar = self._generator()
                task = self._task(
                    task_id="fast-" + sla,
                    sla=sla,
                    preferred=None if sla == "Hard" else 10.0,
                    latest=10.0,
                )
                snapshot = calendar.snapshot()
                fast_stream = generator.prepare_complete_stream(
                    task,
                    0.0,
                    reservation_snapshot=snapshot,
                    forecast_covered_until_sim=11.5,
                )
                fast_records = tuple(
                    fast_stream.iter_candidate_records()
                )

                original = calendar.resources_unallocated
                calendar.resources_unallocated = (
                    lambda snapshot, node, path: False
                )
                try:
                    general_stream = generator.prepare_complete_stream(
                        task,
                        0.0,
                        reservation_snapshot=snapshot,
                        forecast_covered_until_sim=11.5,
                    )
                    general_records = tuple(
                        general_stream.iter_candidate_records()
                    )
                finally:
                    calendar.resources_unallocated = original

                self.assertEqual(fast_stream.status, general_stream.status)
                self.assertEqual(
                    fast_stream.theoretical_slot_count,
                    general_stream.theoretical_slot_count,
                )
                self.assertEqual(
                    fast_stream.feasible_candidate_count,
                    general_stream.feasible_candidate_count,
                )
                self.assertEqual(
                    fast_stream.earliest_compute_start_sim,
                    general_stream.earliest_compute_start_sim,
                )
                self.assertEqual(fast_records, general_records)

    # CAND-004
    def test_cand_004_research_uses_original_arrival_deadline(self):
        generator, _ = self._generator(compute_nodes=("L",))
        task = self._task(
            sla="Flexible",
            preferred=10.0,
            arrival=0.0,
            source="L",
            data=0.0,
            bw=1.0,
        )
        result = generator.generate_complete(task, 12.0)
        self.assertEqual(max(c.compute_start_sim for c in result.candidates), 15.0)

    # CAND-005
    def test_cand_005_remote_earliest_start_equals_transmission_duration(self):
        generator, _ = self._generator()
        result = generator.generate_complete(self._task(), 0.0)
        self.assertEqual(result.earliest_compute_start_sim, 8.0)

    # CAND-006
    def test_cand_006_remote_earliest_start_ceil_to_global_grid(self):
        generator, _ = self._generator()
        result = generator.generate_complete(self._task(), 0.1)
        self.assertEqual(result.earliest_compute_start_sim, 9.0)

    # CAND-007
    def test_cand_007_local_can_start_at_current_aligned_time(self):
        generator, _ = self._generator(compute_nodes=("L",))
        task = self._task(source="L", data=0.0, bw=1.0)
        result = generator.generate_complete(task, 2.0)
        self.assertEqual(result.earliest_compute_start_sim, 2.0)

    # CAND-008 / CAND-009
    def test_cand_008_009_every_output_is_directly_committable_and_sla_legal(self):
        generator, calendar = self._generator()
        task = self._task(sla="Flexible", preferred=10.0)
        result = generator.generate_complete(task, 0.0)
        snapshot = calendar.snapshot()
        for candidate in result.candidates:
            with self.subTest(candidate=candidate.candidate_id):
                request = candidate.to_reservation_request()
                self.assertTrue(
                    calendar.cpu_feasible(
                        snapshot,
                        request.target_node,
                        request.compute_interval_sim,
                        request.cpu_amount,
                    ).feasible
                )
                self.assertTrue(
                    calendar.path_feasible(
                        snapshot,
                        request.path,
                        request.transmission_interval_sim,
                        request.bandwidth_amount_mbps,
                    ).feasible
                )
                self.assertLessEqual(candidate.start_delay_sim, task.latest_start_limit_sim)

    # CAND-010 / INFO-006
    def test_cand_010_forecast_must_cover_compute_end(self):
        generator, _ = self._generator()
        task = self._task(latest=10.0, duration=5.0)
        result = generator.generate_complete(
            task,
            0.0,
            forecast_covered_until_sim=10.0,
        )
        self.assertEqual(result.status, CandidateGenerationStatus.FORECAST_NOT_COVERED)
        self.assertEqual(result.candidates, ())

    # CAND-012 / CAND-016
    def test_cand_012_016_complete_grid_matches_exhaustive_reference(self):
        self.assertEqual(complete_time_grid(2.0, 10.0, 2.0), (2.0, 4.0, 6.0, 8.0, 10.0))
        generator, _ = self._generator(cycle=2.0)
        task = self._task(data=25.0, latest=10.0)
        result = generator.generate_complete(task, 0.0)
        self.assertEqual(
            tuple(c.compute_start_sim for c in result.candidates),
            (2.0, 4.0, 6.0, 8.0, 10.0),
        )
        self.assertEqual(result.theoretical_slot_count, 5)

    # CAND-017
    def test_cand_017_complete_mode_has_no_sampling_limit(self):
        generator, _ = self._generator(compute_nodes=("L",), cycle=1.0)
        task = self._task(source="L", data=0.0, bw=1.0, latest=10.0)
        result = generator.generate_complete(task, 0.0)
        self.assertEqual(len(result.candidates), 11)
        self.assertEqual(result.candidate_mode, CandidateMode.COMPLETE)

    # CAND-018
    def test_cand_018_non_aligned_bounds_only_emit_global_grid(self):
        self.assertEqual(complete_time_grid(2.1, 9.3, 2.0), (4.0, 6.0, 8.0))

    def _block_local_cpu(self, calendar, node, start, end, amount):
        path = build_path_spec(self.graph, [node])
        request = ReservationRequest(
            task_id="blocker",
            committed_candidate_id="blocker-candidate",
            committed_at_sim=0.0,
            reservation_snapshot_version=calendar.version,
            target_node=node,
            path=path,
            transmission_interval_sim=None,
            compute_interval_sim=TimeInterval(start, end),
            bandwidth_amount_mbps=1.0,
            cpu_amount=amount,
        )
        result = calendar.try_commit(request, calendar.version)
        self.assertEqual(result.status, CommitStatus.COMMITTED)

    # DELAY-001 / DELAY-006
    def test_delay_001_006_decomposition_identity(self):
        calendar = ReservationCalendar({"L": 1.0}, {})
        self._block_local_cpu(calendar, "L", 0.0, 5.0, 1.0)
        generator, _ = self._generator(
            compute_nodes=("L",), cycle=1.0, calendar=calendar
        )
        task = self._task(source="L", data=0.0, bw=1.0, latest=10.0)
        result = generator.generate_complete(task, 2.0)
        selected = next(c for c in result.candidates if c.compute_start_sim == 8.0)
        self.assertEqual(selected.scheduler_queue_delay_sim, 2.0)
        self.assertEqual(selected.earliest_feasibility_lead_sim, 3.0)
        self.assertEqual(selected.active_wait_sim, 3.0)
        self.assertEqual(selected.reservation_lead_sim, 6.0)
        self.assertEqual(selected.start_delay_sim, 8.0)
        self.assertEqual(
            selected.start_delay_sim,
            selected.scheduler_queue_delay_sim
            + selected.earliest_feasibility_lead_sim
            + selected.active_wait_sim,
        )

    # DELAY-002
    def test_delay_002_network_is_not_added_twice(self):
        generator, _ = self._generator()
        task = self._task(data=25.0, latest=10.0)
        candidate = generator.generate_complete(task, 0.0).candidates[0]
        self.assertEqual(candidate.earliest_feasibility_lead_sim, 2.0)
        self.assertEqual(candidate.active_wait_sim, 0.0)
        self.assertEqual(candidate.start_delay_sim, 2.0)

    # DELAY-003
    def test_delay_003_resource_and_transmission_form_one_earliest_lead(self):
        calendar = ReservationCalendar({"N": 1.0}, {("S", "N"): 100.0})
        self._block_local_cpu(calendar, "N", 0.0, 7.0, 1.0)
        generator, _ = self._generator(calendar=calendar, cpu_capacity=1.0)
        task = self._task(data=25.0, latest=10.0)
        candidate = generator.generate_complete(task, 0.0).candidates[0]
        self.assertEqual(candidate.compute_start_sim, 7.0)
        self.assertEqual(candidate.earliest_feasibility_lead_sim, 7.0)

    # DELAY-004
    def test_delay_004_pending_research_decomposition(self):
        calendar = ReservationCalendar({"L": 1.0}, {})
        self._block_local_cpu(calendar, "L", 0.0, 7.0, 1.0)
        generator, _ = self._generator(compute_nodes=("L",), calendar=calendar)
        task = self._task(source="L", data=0.0, bw=1.0, latest=12.0)
        result = generator.generate_complete(task, 5.0)
        selected = next(c for c in result.candidates if c.compute_start_sim == 9.0)
        self.assertEqual(selected.scheduler_queue_delay_sim, 5.0)
        self.assertEqual(selected.earliest_feasibility_lead_sim, 2.0)
        self.assertEqual(selected.active_wait_sim, 2.0)

    def _queue_task(self, task_id, sla, arrival, preferred=None, latest=None):
        return self._task(
            task_id=task_id,
            sla=sla,
            arrival=arrival,
            preferred=preferred,
            latest=latest,
            source="L",
            data=0.0,
            bw=1.0,
        )

    # QUEUE-001 / QUEUE-002
    def test_queue_001_002_absolute_deadline_is_primary(self):
        hard = self._queue_task("hard", "Hard", 100.0, latest=6.0)
        soft = self._queue_task("soft", "Soft", 98.0, preferred=10.0)
        flexible = self._queue_task("flex", "Flexible", 90.0, preferred=40.0)
        self.assertEqual(
            [task.task_id for task in sorted((flexible, soft, hard), key=queue_order_key)],
            ["hard", "soft", "flex"],
        )
        later_hard = self._queue_task("later-hard", "Hard", 0.0, latest=20.0)
        early_soft = self._queue_task("early-soft", "Soft", 0.0, preferred=10.0)
        self.assertEqual(
            [task.task_id for task in sorted((later_hard, early_soft), key=queue_order_key)],
            ["early-soft", "later-hard"],
        )

    # QUEUE-003
    def test_queue_003_sla_type_breaks_equal_deadline_tie(self):
        hard = self._queue_task("h", "Hard", 0.0, latest=12.0)
        soft = self._queue_task("s", "Soft", 0.0, preferred=10.0)
        flexible = self._queue_task("f", "Flexible", 0.0, preferred=8.0)
        self.assertEqual(
            [t.sla_type for t in sorted((flexible, soft, hard), key=queue_order_key)],
            [SlaType.HARD, SlaType.SOFT, SlaType.FLEXIBLE],
        )

    # QUEUE-004 / QUEUE-012
    def test_queue_004_012_preferred_deadline_breaks_soft_and_flexible_ties(self):
        soft_early_pref = self._queue_task("s1", "Soft", 0.0, preferred=10.0)
        soft_late_pref = self._queue_task("s2", "Soft", 6.0, preferred=5.0)
        self.assertEqual(
            sorted((soft_late_pref, soft_early_pref), key=queue_order_key)[0].task_id,
            "s1",
        )
        flex_early_pref = self._queue_task("f1", "Flexible", 0.0, preferred=10.0)
        flex_late_pref = self._queue_task("f2", "Flexible", 7.5, preferred=5.0)
        self.assertEqual(
            sorted((flex_late_pref, flex_early_pref), key=queue_order_key)[0].task_id,
            "f1",
        )

    # QUEUE-005 / QUEUE-006
    def test_queue_005_006_arrival_then_id_are_stable_ties(self):
        first = self._queue_task("z", "Hard", 0.0, latest=10.0)
        later = self._queue_task("a", "Hard", 1.0, latest=9.0)
        self.assertEqual(sorted((later, first), key=queue_order_key)[0].task_id, "z")
        a = self._queue_task("a", "Hard", 0.0, latest=10.0)
        b = self._queue_task("b", "Hard", 0.0, latest=10.0)
        self.assertEqual([t.task_id for t in sorted((b, a), key=queue_order_key)], ["a", "b"])

    # QUEUE-007
    def test_queue_007_runtime_attempt_counts_do_not_change_order(self):
        machine = TaskStateMachine()
        manager = TaskQueueManager(machine, 10)
        a = self._queue_task("a", "Hard", 0.0, latest=10.0)
        b = self._queue_task("b", "Hard", 0.0, latest=10.0)
        manager.enqueue_new(a)
        manager.enqueue_new(b)
        machine.increment_pending_attempts("a")
        machine.increment_commit_attempts("b")
        self.assertEqual([t.task_id for t in manager.ordered_queued_tasks()], ["a", "b"])

    # QUEUE-008 / QUEUE-009
    def test_queue_008_009_processing_limit_and_capacity(self):
        machine = TaskStateMachine()
        manager = TaskQueueManager(machine, 10)
        for index in range(10):
            manager.enqueue_new(self._queue_task(f"t{index:02d}", "Hard", 0.0, latest=10.0))
        self.assertEqual(len(manager.eligible_tasks(3)), 3)
        self.assertEqual(machine.count_by_state()[TaskState.QUEUED], 10)
        overflow = manager.enqueue_new(self._queue_task("overflow", "Hard", 0.0, latest=10.0))
        self.assertEqual(overflow.new_state, TaskState.REJECTED)
        self.assertEqual(overflow.terminal_reason, "SCHEDULER_QUEUE_CAPACITY")

    # CAND-015 / LIFE-008 / LIFE-009 / LIFE-010 / LIFE-011
    def test_pending_only_reactivates_for_physical_event_and_expires_once(self):
        machine = TaskStateMachine()
        manager = TaskQueueManager(machine, 10)
        task = self._queue_task("p", "Hard", 0.0, latest=10.0)
        manager.enqueue_new(task)
        manager.mark_pending("p", 1.0)
        self.assertEqual(manager.reactivate_pending(2.0, ["PRICE_CHANGED"]), [])
        self.assertEqual(machine.runtime("p").state, TaskState.PENDING_UNCOMMITTED)
        transitions = manager.reactivate_pending(5.0, ["CPU_INTERVAL_ENDED"])
        self.assertEqual(transitions[0].new_state, TaskState.QUEUED)
        manager.mark_pending("p", 5.0)
        expired = manager.expire_due_tasks_after_boundary_opportunity(10.0)
        self.assertEqual(expired[0].new_state, TaskState.EXPIRED)
        self.assertEqual(manager.expire_due_tasks_after_boundary_opportunity(10.0), [])

    # ATOM-006 / ATOM-009
    def test_atom_006_009_conflicts_are_bounded_and_not_failure_retries(self):
        generator, calendar = self._generator(compute_nodes=("L",))
        task = self._task(task_id="q", source="L", data=0.0, bw=1.0)
        candidate = generator.generate_complete(task, 0.0).candidates[0]
        self._block_local_cpu(calendar, "L", 5.0, 6.0, 1.0)
        machine = TaskStateMachine()
        engine = EventEngine(calendar, machine)
        engine.register_task(task)
        manager = ReservationManager(calendar, machine, engine, 3)
        first = manager.commit_selected(candidate)
        second = manager.commit_selected(candidate)
        third = manager.commit_selected(candidate)
        self.assertEqual(first.status, CommitDecisionStatus.RETRY_WITH_NEW_SNAPSHOT)
        self.assertEqual(second.status, CommitDecisionStatus.RETRY_WITH_NEW_SNAPSHOT)
        self.assertEqual(third.status, CommitDecisionStatus.ATTEMPT_LIMIT_REACHED)
        runtime = machine.runtime(task.task_id)
        self.assertEqual(runtime.state, TaskState.QUEUED)
        self.assertEqual(runtime.failure_retry_count, 0)
        self.assertEqual(runtime.commit_attempts_current_decision, 3)


if __name__ == "__main__":
    unittest.main()
