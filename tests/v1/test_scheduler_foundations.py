import unittest
from dataclasses import FrozenInstanceError

import networkx as nx
import numpy as np

from v1.domain.models import TaskSpec, TaskState
from v1.domain.reservations import (
    CommitStatus,
    PathSpec,
    ReservationRequest,
    TimeInterval,
)
from v1.domain.units import TimeConverter
from v1.scheduler.resource_calendar import ReservationCalendar
from v1.scheduler.transmission import TransmissionModel, build_path_spec
from v1.simulation.event_engine import EventEngine


class SchedulerFoundationTest(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_edge("S", "B", capacity=100.0, distance_km=100.0, cost=1.0)
        self.graph.add_edge("B", "C", capacity=100.0, distance_km=300.0, cost=1.0)
        self.graph.add_edge("S", "C", capacity=50.0, distance_km=50.0, cost=1.0)
        self.graph.add_node("N")
        self.local_path = build_path_spec(self.graph, ["N"], "local-N")
        self.single_path = build_path_spec(self.graph, ["S", "B"], "S-B")
        self.multi_path = build_path_spec(self.graph, ["S", "B", "C"], "S-B-C")
        self._candidate_counter = 0

    def _task(
        self,
        task_id="t",
        source="S",
        data=100.0,
        bandwidth=100.0,
        cpu=2.0,
        duration=5.0,
    ):
        return TaskSpec.create(
            task_id=task_id,
            arrival_time_sim=0.0,
            source_node=source,
            cpu_demand=cpu,
            execution_duration_sim=duration,
            data_size_mb=data,
            bandwidth_demand_mbps=bandwidth,
            sla_type="Hard",
            latest_start_limit_sim=100.0,
        )

    def _calendar(self, cpu=10.0, links=None):
        return ReservationCalendar(
            {"N": cpu, "B": cpu, "C": cpu},
            links or {("S", "B"): 100.0, ("B", "C"): 100.0},
        )

    def _request(
        self,
        calendar,
        task_id,
        path,
        compute_start,
        compute_end,
        cpu=2.0,
        bandwidth=10.0,
        tx_start=None,
        committed_at=0.0,
    ):
        self._candidate_counter += 1
        tx_interval = None
        if not path.is_local:
            if tx_start is None:
                tx_start = compute_start - 1.0
            tx_interval = TimeInterval(tx_start, compute_start)
        return ReservationRequest(
            task_id=task_id,
            committed_candidate_id=f"candidate-{self._candidate_counter}",
            committed_at_sim=committed_at,
            reservation_snapshot_version=calendar.version,
            target_node=path.target_node,
            path=path,
            transmission_interval_sim=tx_interval,
            compute_interval_sim=TimeInterval(compute_start, compute_end),
            bandwidth_amount_mbps=bandwidth,
            cpu_amount=cpu,
        )

    def _commit(self, calendar, **kwargs):
        request = self._request(calendar, **kwargs)
        return calendar.try_commit(request, request.reservation_snapshot_version)

    def test_batch_calendar_feasibility_matches_scalar_queries(self):
        calendar = self._calendar(cpu=10.0)
        committed = self._commit(
            calendar,
            task_id="existing",
            path=self.single_path,
            compute_start=2.0,
            compute_end=6.0,
            cpu=6.0,
            bandwidth=60.0,
            tx_start=1.0,
        )
        self.assertEqual(committed.status, CommitStatus.COMMITTED)
        snapshot = calendar.snapshot()

        starts = np.asarray((0.0, 1.0, 2.0, 5.0, 6.0, 7.0))
        ends = starts + 2.0
        scalar_cpu = [
            calendar.cpu_feasible(
                snapshot,
                "B",
                TimeInterval(float(start), float(end)),
                5.0,
            )
            for start, end in zip(starts, ends)
        ]
        batch_cpu = calendar.cpu_feasible_many(
            snapshot, "B", starts, ends, 5.0
        )
        np.testing.assert_array_equal(
            batch_cpu["feasible"],
            [item.feasible for item in scalar_cpu],
        )
        np.testing.assert_array_equal(
            batch_cpu["existing_peak"],
            [item.existing_peak for item in scalar_cpu],
        )
        np.testing.assert_array_equal(
            batch_cpu["projected_peak"],
            [item.projected_peak for item in scalar_cpu],
        )

        transmission_starts = starts - 1.0
        scalar_path = [
            calendar.path_feasible(
                snapshot,
                self.single_path,
                TimeInterval(float(start), float(end)),
                50.0,
            )
            for start, end in zip(transmission_starts, starts)
        ]
        batch_path = calendar.path_feasible_many(
            snapshot,
            self.single_path,
            transmission_starts,
            starts,
            50.0,
        )
        np.testing.assert_array_equal(
            batch_path["feasible"],
            [item.feasible for item in scalar_path],
        )
        np.testing.assert_array_equal(
            batch_path["existing_peak"],
            [item.existing_peak for item in scalar_path],
        )
        np.testing.assert_array_equal(
            batch_path["projected_peak"],
            [item.projected_peak for item in scalar_path],
        )

    # TX-001
    def test_tx_001_local_has_zero_duration_and_no_bandwidth_interval(self):
        model = TransmissionModel(TimeConverter(1.0), 200000.0)
        task = self._task(source="N")
        duration = model.duration(task, self.local_path)
        schedule = model.jit_schedule(task, self.local_path, 10.0, 10.0)
        self.assertEqual(duration.total_sim, 0.0)
        self.assertIsNone(schedule.transmission_interval_sim)
        self.assertTrue(schedule.feasible_from_decision)

    # TX-002
    def test_tx_002_single_hop_hand_calculation(self):
        model = TransmissionModel(TimeConverter(1.0), 200000.0)
        duration = model.duration(self._task(), self.single_path)
        self.assertAlmostEqual(duration.total_seconds, 8.0005)

    # TX-003
    def test_tx_003_multihop_serializes_data_once(self):
        model = TransmissionModel(TimeConverter(1.0), 200000.0)
        duration = model.duration(self._task(), self.multi_path)
        self.assertAlmostEqual(duration.total_seconds, 8.002)
        self.assertNotAlmostEqual(duration.total_seconds, 16.002)

    # TX-004
    def test_tx_004_data_size_doubles_serialization_time(self):
        model = TransmissionModel(TimeConverter(1.0), 200000.0)
        one = model.duration(self._task(data=100.0), self.single_path)
        two = model.duration(self._task(data=200.0), self.single_path)
        self.assertAlmostEqual(two.data_seconds, 2.0 * one.data_seconds)

    # TX-005
    def test_tx_005_bandwidth_doubles_halves_serialization_time(self):
        model = TransmissionModel(TimeConverter(1.0), 200000.0)
        low = model.duration(self._task(bandwidth=50.0), self.single_path)
        high = model.duration(self._task(bandwidth=100.0), self.single_path)
        self.assertAlmostEqual(high.data_seconds, 0.5 * low.data_seconds)

    # TX-006
    def test_tx_006_seconds_are_converted_to_sim_time(self):
        model = TransmissionModel(TimeConverter(300.0), 200000.0)
        duration = model.duration(self._task(), self.multi_path)
        self.assertAlmostEqual(duration.total_sim, 8.002 / 300.0)

    # TX-007
    def test_tx_007_static_capacity_is_checked_without_existing_usage(self):
        model = TransmissionModel(TimeConverter(1.0), 200000.0)
        direct = build_path_spec(self.graph, ["S", "C"], "S-C")
        task = self._task(bandwidth=100.0)
        self.assertFalse(model.duration(task, direct).static_path_feasible)
        calendar = self._calendar(links={("S", "C"): 50.0})
        result = calendar.path_feasible(
            calendar.snapshot(), direct, TimeInterval(0.0, 1.0), 100.0
        )
        self.assertFalse(result.feasible)

    # TX-008
    def test_tx_008_equal_static_capacity_is_feasible(self):
        calendar = self._calendar(links={("S", "B"): 100.0})
        result = calendar.path_feasible(
            calendar.snapshot(),
            self.single_path,
            TimeInterval(0.0, 1.0),
            100.0,
        )
        self.assertTrue(result.feasible)

    def _two_second_path(self):
        return PathSpec(
            path_id="two-second-path",
            source_node="S",
            target_node="B",
            ordered_nodes=("S", "B"),
            ordered_edges=(("S", "B"),),
            total_distance_km=0.0,
            static_bottleneck_mbps=100.0,
            route_cost=0.0,
        )

    # JIT-001 / JIT-002
    def test_jit_001_002_exact_interval_and_decision_boundary(self):
        model = TransmissionModel(TimeConverter(1.0), 200000.0)
        task = self._task(data=25.0, bandwidth=100.0)
        schedule = model.jit_schedule(task, self._two_second_path(), 10.0, 8.0)
        self.assertEqual(schedule.transmission_interval_sim, TimeInterval(8.0, 10.0))
        self.assertTrue(schedule.feasible_from_decision)

    # JIT-003
    def test_jit_003_cannot_transmit_before_decision(self):
        model = TransmissionModel(TimeConverter(1.0), 200000.0)
        task = self._task(data=25.0, bandwidth=100.0)
        schedule = model.jit_schedule(
            task, self._two_second_path(), 10.0, 8.0 + 1e-9
        )
        self.assertFalse(schedule.feasible_from_decision)

    # JIT-005
    def test_jit_005_interval_moves_with_compute_start(self):
        model = TransmissionModel(TimeConverter(1.0), 200000.0)
        task = self._task(data=25.0, bandwidth=100.0)
        first = model.jit_schedule(task, self._two_second_path(), 10.0, 8.0)
        second = model.jit_schedule(task, self._two_second_path(), 12.0, 8.0)
        self.assertEqual(first.transmission_interval_sim.duration_sim, 2.0)
        self.assertEqual(second.transmission_interval_sim, TimeInterval(10.0, 12.0))

    def _commit_local(self, calendar, task_id, start, end, cpu):
        return self._commit(
            calendar,
            task_id=task_id,
            path=self.local_path,
            compute_start=start,
            compute_end=end,
            cpu=cpu,
            bandwidth=1.0,
        )

    # CAL-001 / CAL-004
    def test_cal_001_004_true_peak_and_half_open_boundary(self):
        calendar = self._calendar(cpu=10.0)
        self.assertEqual(
            self._commit_local(calendar, "a", 0.0, 5.0, 6.0).status,
            CommitStatus.COMMITTED,
        )
        self.assertEqual(
            self._commit_local(calendar, "b", 5.0, 10.0, 6.0).status,
            CommitStatus.COMMITTED,
        )
        snapshot = calendar.snapshot()
        allocations = [
            item for item in snapshot.cpu_calendar_view if item.resource_id == "N"
        ]
        self.assertEqual(
            calendar.peak_usage(allocations, TimeInterval(0.0, 10.0)),
            6.0,
        )

    # CAL-002 / CAL-003
    def test_cal_002_003_candidate_amount_boundary(self):
        calendar = self._calendar(cpu=10.0)
        self._commit_local(calendar, "a", 0.0, 5.0, 6.0)
        self._commit_local(calendar, "b", 5.0, 10.0, 6.0)
        snapshot = calendar.snapshot()
        self.assertTrue(
            calendar.cpu_feasible(snapshot, "N", TimeInterval(0.0, 10.0), 4.0).feasible
        )
        result = calendar.cpu_feasible(snapshot, "N", TimeInterval(0.0, 10.0), 5.0)
        self.assertFalse(result.feasible)
        self.assertEqual(result.projected_peak, 11.0)

    def test_cached_peak_queries_match_reference_for_all_boundaries(self):
        calendar = self._calendar(cpu=100.0)
        self._commit_local(calendar, "a", 0.0, 6.0, 5.0)
        self._commit_local(calendar, "b", 4.0, 10.0, 7.0)
        self._commit_local(calendar, "c", 9.0, 12.0, 3.0)
        snapshot = calendar.snapshot()
        allocations = snapshot.cpu_calendar_view
        intervals = (
            (-2.0, -1.0),
            (-1.0, 0.0),
            (-1.0, 0.1),
            (0.0, 4.0),
            (3.0, 4.0),
            (4.0, 6.0),
            (5.999999999, 6.000000001),
            (6.0, 9.0),
            (9.0, 10.0),
            (10.0, 12.0),
            (12.0, 13.0),
            (0.0, 12.0),
            (2.5, 11.5),
        )
        for start, end in intervals:
            with self.subTest(start=start, end=end):
                interval = TimeInterval(start, end)
                expected = calendar.peak_usage(
                    allocations,
                    interval,
                )
                actual = calendar.cpu_feasible(
                    snapshot,
                    "N",
                    interval,
                    1.0,
                ).existing_peak
                self.assertEqual(actual, expected)

    # CAL-005
    def test_cal_005_epsilon_overlap_is_detected(self):
        calendar = self._calendar(cpu=10.0)
        self._commit_local(calendar, "a", 0.0, 5.0 + 1e-9, 6.0)
        result = calendar.cpu_feasible(
            calendar.snapshot(), "N", TimeInterval(5.0, 10.0), 6.0
        )
        self.assertFalse(result.feasible)
        self.assertEqual(result.projected_peak, 12.0)

    # CAL-006
    def test_cal_006_link_usage_uses_true_peak(self):
        calendar = self._calendar(cpu=100.0, links={("S", "B"): 10.0})
        first = self._commit(
            calendar,
            task_id="a",
            path=self.single_path,
            tx_start=0.0,
            compute_start=5.0,
            compute_end=6.0,
            cpu=1.0,
            bandwidth=6.0,
        )
        second = self._commit(
            calendar,
            task_id="b",
            path=self.single_path,
            tx_start=5.0,
            compute_start=10.0,
            compute_end=11.0,
            cpu=1.0,
            bandwidth=6.0,
        )
        self.assertEqual(first.status, CommitStatus.COMMITTED)
        self.assertEqual(second.status, CommitStatus.COMMITTED)
        self.assertTrue(
            calendar.path_feasible(
                calendar.snapshot(),
                self.single_path,
                TimeInterval(0.0, 10.0),
                4.0,
            ).feasible
        )

    # CAL-007 / ATOM-004
    def test_cal_007_atom_004_second_edge_failure_leaves_no_partial_write(self):
        calendar = self._calendar(
            cpu=100.0,
            links={("S", "B"): 100.0, ("B", "C"): 5.0},
        )
        before = calendar.snapshot()
        result = self._commit(
            calendar,
            task_id="a",
            path=self.multi_path,
            tx_start=0.0,
            compute_start=1.0,
            compute_end=2.0,
            bandwidth=10.0,
        )
        after = calendar.snapshot()
        self.assertEqual(result.status, CommitStatus.BANDWIDTH_INFEASIBLE)
        self.assertEqual(before, after)

    # CAL-008
    def test_cal_008_insertion_order_does_not_change_peak(self):
        def build(order):
            calendar = self._calendar(cpu=20.0)
            intervals = {"a": (0.0, 6.0, 5.0), "b": (4.0, 10.0, 7.0)}
            for task_id in order:
                start, end, cpu = intervals[task_id]
                self._commit_local(calendar, task_id, start, end, cpu)
            allocations = calendar.snapshot().cpu_calendar_view
            return calendar.peak_usage(allocations, TimeInterval(0.0, 10.0))
        self.assertEqual(build(("a", "b")), build(("b", "a")))

    # CAL-009
    def test_cal_009_reservation_is_immutable(self):
        calendar = self._calendar()
        result = self._commit_local(calendar, "a", 0.0, 5.0, 2.0)
        with self.assertRaises(FrozenInstanceError):
            result.reservation.cpu_amount = 3.0

    # CAL-010 / ATOM-008
    def test_cal_010_atom_008_committed_reservation_is_complete(self):
        calendar = self._calendar(cpu=100.0)
        result = self._commit(
            calendar,
            task_id="a",
            path=self.multi_path,
            tx_start=0.0,
            compute_start=1.0,
            compute_end=5.0,
            cpu=3.0,
            bandwidth=10.0,
        )
        reservation = result.reservation
        self.assertTrue(calendar.verify_reservation(reservation.reservation_id))
        self.assertEqual(reservation.path, self.multi_path)
        self.assertEqual(reservation.compute_interval_sim, TimeInterval(1.0, 5.0))
        self.assertEqual(reservation.transmission_interval_sim, TimeInterval(0.0, 1.0))

    # ATOM-001
    def test_atom_001_success_writes_all_resources_and_increments_once(self):
        calendar = self._calendar(cpu=100.0)
        result = self._commit(
            calendar,
            task_id="a",
            path=self.multi_path,
            tx_start=0.0,
            compute_start=1.0,
            compute_end=2.0,
            bandwidth=10.0,
        )
        snapshot = calendar.snapshot()
        self.assertEqual(result.status, CommitStatus.COMMITTED)
        self.assertEqual(snapshot.reservation_version, 1)
        self.assertEqual(len(snapshot.cpu_calendar_view), 1)
        self.assertEqual(len(snapshot.link_calendar_view), 2)

    # ATOM-002
    def test_atom_002_cpu_failure_writes_nothing(self):
        calendar = self._calendar(cpu=1.0)
        before = calendar.snapshot()
        result = self._commit_local(calendar, "a", 0.0, 1.0, 2.0)
        self.assertEqual(result.status, CommitStatus.CPU_INFEASIBLE)
        self.assertEqual(calendar.snapshot(), before)

    # ATOM-003
    def test_atom_003_first_link_failure_writes_no_cpu(self):
        calendar = self._calendar(cpu=100.0, links={("S", "B"): 5.0})
        before = calendar.snapshot()
        result = self._commit(
            calendar,
            task_id="a",
            path=self.single_path,
            tx_start=0.0,
            compute_start=1.0,
            compute_end=2.0,
            bandwidth=10.0,
        )
        self.assertEqual(result.status, CommitStatus.BANDWIDTH_INFEASIBLE)
        self.assertEqual(calendar.snapshot(), before)

    # ATOM-005
    def test_atom_005_stale_snapshot_returns_conflict(self):
        calendar = self._calendar()
        stale_request = self._request(
            calendar,
            task_id="stale",
            path=self.local_path,
            compute_start=5.0,
            compute_end=6.0,
        )
        self._commit_local(calendar, "other", 0.0, 1.0, 1.0)
        result = calendar.try_commit(stale_request, stale_request.reservation_snapshot_version)
        self.assertEqual(result.status, CommitStatus.CONFLICT)
        self.assertEqual(calendar.active_reservation_count, 1)

    # ATOM-007
    def test_atom_007_injected_write_failure_rolls_back(self):
        calendar = self._calendar(cpu=100.0)
        request = self._request(
            calendar,
            task_id="a",
            path=self.multi_path,
            tx_start=0.0,
            compute_start=1.0,
            compute_end=2.0,
        )
        before = calendar.snapshot()
        result = calendar.try_commit(
            request,
            request.reservation_snapshot_version,
            inject_failure_at="after_link_write:0",
        )
        self.assertEqual(result.status, CommitStatus.INTERNAL_ROLLBACK)
        self.assertEqual(calendar.snapshot(), before)
        self.assertEqual(calendar.active_reservation_count, 0)

    def _engine_with_local_reservation(self, compute_start=1.0, compute_end=5.0):
        calendar = self._calendar(cpu=10.0)
        task = self._task(task_id="local-task", source="N", duration=compute_end-compute_start)
        result = self._commit_local(
            calendar, task.task_id, compute_start, compute_end, task.cpu_demand
        )
        engine = EventEngine(calendar, initial_time_sim=0.0)
        queued_events = engine.register_task(task)
        reserve_events = engine.register_reservation(result.reservation)
        return calendar, engine, task, queued_events, reserve_events

    # LIFE-001 / LIFE-002
    def test_life_001_002_arrival_queue_and_reservation(self):
        _, engine, task, queued, reserved = self._engine_with_local_reservation()
        self.assertEqual(queued[0].previous_state, TaskState.ARRIVED)
        self.assertEqual(queued[0].new_state, TaskState.QUEUED)
        self.assertEqual(reserved[0].new_state, TaskState.RESERVED)
        self.assertEqual(engine.state_machine.runtime(task.task_id).state, TaskState.RESERVED)

    # LIFE-003 / LIFE-004 / JIT-004
    def test_life_003_004_remote_transmission_then_running(self):
        calendar = self._calendar(cpu=100.0)
        task = self._task(task_id="remote")
        commit = self._commit(
            calendar,
            task_id=task.task_id,
            path=self.single_path,
            tx_start=1.0,
            compute_start=3.0,
            compute_end=5.0,
            cpu=task.cpu_demand,
            bandwidth=10.0,
        )
        engine = EventEngine(calendar)
        engine.register_task(task)
        engine.register_reservation(commit.reservation)
        events = engine.advance_to(1.0)
        self.assertEqual(events[-1].new_state, TaskState.TRANSMITTING)
        events = engine.advance_to(3.0)
        self.assertEqual(events[-1].new_state, TaskState.RUNNING)
        snapshot = calendar.snapshot()
        edge_allocations = [
            item for item in snapshot.link_calendar_view if item.resource_id == ("B", "S")
        ]
        self.assertEqual(
            calendar.peak_usage(edge_allocations, TimeInterval(3.0, 4.0)),
            0.0,
        )

    # LIFE-005 / LIFE-006
    def test_life_005_006_completion_occurs_exactly_once_at_end(self):
        calendar, engine, task, _, _ = self._engine_with_local_reservation()
        engine.advance_to(1.0)
        engine.advance_to(5.0 - 1e-9)
        self.assertEqual(engine.state_machine.runtime(task.task_id).state, TaskState.RUNNING)
        self.assertEqual(engine.completed_count, 0)
        events = engine.advance_to(5.0)
        self.assertEqual(events[-1].new_state, TaskState.COMPLETED)
        self.assertEqual(engine.completed_count, 1)
        self.assertEqual(engine.advance_to(5.0), [])
        self.assertEqual(calendar.active_reservation_count, 0)

    # LIFE-007
    def test_life_007_local_skips_transmitting(self):
        _, engine, task, _, _ = self._engine_with_local_reservation()
        events = engine.advance_to(1.0)
        self.assertEqual([event.new_state for event in events], [TaskState.RUNNING])
        self.assertEqual(engine.state_machine.runtime(task.task_id).state, TaskState.RUNNING)

    # LIFE-012 / LIFE-014
    def test_life_012_014_reserved_tasks_are_not_counted_until_completed(self):
        _, engine, task, _, _ = self._engine_with_local_reservation(
            compute_start=2.0, compute_end=10.0
        )
        engine.advance_to(5.0)
        self.assertEqual(engine.state_machine.runtime(task.task_id).state, TaskState.RUNNING)
        self.assertEqual(engine.completed_count, 0)
        engine.advance_to(10.0)
        self.assertEqual(engine.completed_count, 1)

    # LIFE-013 / LIFE-015
    def test_life_013_015_failure_is_terminal_without_retry(self):
        calendar, engine, task, _, _ = self._engine_with_local_reservation(
            compute_start=1.0, compute_end=10.0
        )
        engine.advance_to(1.0)
        events = engine.fail_task(task.task_id, 2.0, "EXECUTION_FAILURE")
        runtime = engine.state_machine.runtime(task.task_id)
        self.assertEqual(events[-1].new_state, TaskState.FAILED)
        self.assertEqual(runtime.terminal_reason, "EXECUTION_FAILURE")
        self.assertEqual(runtime.failure_retry_count, 0)
        self.assertEqual(calendar.active_reservation_count, 0)
        self.assertEqual(engine.advance_to(10.0), [])

    # EVENT-001 / EVENT-008
    def test_event_001_008_end_before_start_and_repeated_advance_is_idempotent(self):
        calendar = self._calendar(cpu=10.0)
        first_task = self._task(task_id="first", source="N", cpu=10.0, duration=5.0)
        second_task = self._task(task_id="second", source="N", cpu=10.0, duration=5.0)
        first = self._commit_local(calendar, "first", 0.0, 5.0, 10.0)
        second = self._commit_local(calendar, "second", 5.0, 10.0, 10.0)
        self.assertEqual(first.status, CommitStatus.COMMITTED)
        self.assertEqual(second.status, CommitStatus.COMMITTED)
        engine = EventEngine(calendar)
        engine.register_task(first_task)
        engine.register_task(second_task)
        engine.register_reservation(first.reservation)
        engine.register_reservation(second.reservation)
        engine.advance_to(0.0)
        events = engine.advance_to(5.0)
        self.assertEqual(events[0].new_state, TaskState.COMPLETED)
        self.assertEqual(events[1].new_state, TaskState.RUNNING)
        self.assertEqual(engine.advance_to(5.0), [])


if __name__ == "__main__":
    unittest.main()
