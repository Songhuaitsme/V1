"""Deterministic EDF queue and event-triggered Pending management."""

from collections import Counter
from typing import Iterable, List, Tuple

from v1.domain.models import SlaType, TaskSpec, TaskState
from v1.simulation.state_machine import StateTransition, TaskStateMachine


SLA_TIE_RANK = {
    SlaType.HARD: 0,
    SlaType.SOFT: 1,
    SlaType.FLEXIBLE: 2,
}

PHYSICAL_REACTIVATION_EVENTS = {
    "CPU_INTERVAL_ENDED",
    "BANDWIDTH_INTERVAL_ENDED",
    "RESERVATION_RELEASED",
    "TOPOLOGY_CAPACITY_CHANGED",
    "FORECAST_COVERAGE_EXTENDED",
}


def queue_order_key(task: TaskSpec) -> Tuple:
    preferred = task.absolute_preferred_start_sim
    return (
        task.absolute_latest_start_sim,
        SLA_TIE_RANK[task.sla_type],
        float("inf") if preferred is None else preferred,
        task.arrival_time_sim,
        task.task_id,
    )


class TaskQueueManager:
    def __init__(self, state_machine: TaskStateMachine, max_queue_length: int):
        if (
            isinstance(max_queue_length, bool)
            or not isinstance(max_queue_length, int)
            or max_queue_length <= 0
        ):
            raise ValueError("max_queue_length must be a positive integer")
        self.state_machine = state_machine
        self.max_queue_length = max_queue_length
        self.reason_counts = Counter()

    def _uncommitted_count(self) -> int:
        counts = self.state_machine.count_by_state()
        return counts[TaskState.QUEUED] + counts[TaskState.PENDING_UNCOMMITTED]

    def enqueue_new(
        self,
        task: TaskSpec,
        statically_serviceable: bool = True,
    ) -> StateTransition:
        self.state_machine.register(task)
        if not statically_serviceable:
            transition = self.state_machine.transition(
                task.task_id,
                TaskState.REJECTED,
                task.arrival_time_sim,
                terminal_reason="STATICALLY_UNSERVICEABLE",
            )
            self.reason_counts["STATICALLY_UNSERVICEABLE"] += 1
            return transition
        if self._uncommitted_count() >= self.max_queue_length:
            transition = self.state_machine.transition(
                task.task_id,
                TaskState.REJECTED,
                task.arrival_time_sim,
                terminal_reason="SCHEDULER_QUEUE_CAPACITY",
            )
            self.reason_counts["SCHEDULER_QUEUE_CAPACITY"] += 1
            return transition
        return self.state_machine.transition(
            task.task_id,
            TaskState.QUEUED,
            task.arrival_time_sim,
        )

    def ordered_queued_tasks(self) -> List[TaskSpec]:
        tasks = []
        for task_id in self._task_ids():
            if self.state_machine.runtime(task_id).state is TaskState.QUEUED:
                tasks.append(self.state_machine.task_spec(task_id))
        return sorted(tasks, key=queue_order_key)

    def eligible_tasks(
        self,
        max_tasks: int,
        now_sim: float = None,
    ) -> List[TaskSpec]:
        if isinstance(max_tasks, bool) or not isinstance(max_tasks, int) or max_tasks < 0:
            raise ValueError("max_tasks must be a non-negative integer")
        tasks = self.ordered_queued_tasks()
        if now_sim is not None:
            tasks = [
                task
                for task in tasks
                if task.arrival_time_sim <= now_sim + 1e-12
            ]
        return tasks[:max_tasks]

    def mark_pending(self, task_id: str, now_sim: float) -> StateTransition:
        transition = self.state_machine.transition(
            task_id,
            TaskState.PENDING_UNCOMMITTED,
            now_sim,
        )
        self.state_machine.increment_pending_attempts(task_id)
        return transition

    def reactivate_pending(
        self,
        now_sim: float,
        event_types: Iterable[str],
    ) -> List[StateTransition]:
        if not (set(event_types) & PHYSICAL_REACTIVATION_EVENTS):
            return []
        transitions = []
        for task_id in self._task_ids():
            if self.state_machine.runtime(task_id).state is TaskState.PENDING_UNCOMMITTED:
                transitions.append(self.state_machine.transition(
                    task_id,
                    TaskState.QUEUED,
                    now_sim,
                ))
        return transitions

    def expire_due_tasks_after_boundary_opportunity(
        self,
        now_sim: float,
    ) -> List[StateTransition]:
        transitions = []
        for task_id in self._task_ids():
            runtime = self.state_machine.runtime(task_id)
            if runtime.state not in {
                TaskState.QUEUED,
                TaskState.PENDING_UNCOMMITTED,
            }:
                continue
            task = self.state_machine.task_spec(task_id)
            if now_sim >= task.absolute_latest_start_sim - 1e-12:
                transitions.append(self.state_machine.transition(
                    task_id,
                    TaskState.EXPIRED,
                    now_sim,
                    terminal_reason="ABSOLUTE_START_DEADLINE",
                ))
                self.reason_counts["ABSOLUTE_START_DEADLINE"] += 1
        return transitions

    def _task_ids(self):
        return self.state_machine.task_ids
