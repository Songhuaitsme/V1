"""Linear incremental CPU power and dimensionally correct energy billing."""

import math
from dataclasses import dataclass
from bisect import bisect_right
from collections import OrderedDict
from typing import Callable, Iterable, Mapping, Optional, Tuple

import numpy as np

from v1.domain.models import MetricValue, TaskSpec
from v1.domain.reservations import Reservation, TimeInterval
from v1.domain.units import TimeConverter, finite_number, positive_finite
from v1.scheduler.resource_calendar import CalendarAllocation, ReservationSnapshot

from .forecast import PiecewiseConstantForecast


@dataclass(frozen=True)
class LinearPowerModel:
    incremental_cpu_power_mw_per_cpu: float

    def __post_init__(self):
        object.__setattr__(
            self,
            "incremental_cpu_power_mw_per_cpu",
            positive_finite(
                "incremental_cpu_power_mw_per_cpu",
                self.incremental_cpu_power_mw_per_cpu,
            ),
        )

    def task_power_mw(self, cpu_demand: float) -> float:
        return positive_finite("cpu_demand", cpu_demand) * (
            self.incremental_cpu_power_mw_per_cpu
        )


@dataclass(frozen=True)
class CandidateAccountingMetrics:
    task_energy_mwh: float
    task_direct_energy_cost_yuan: float
    candidate_marginal_system_cost_yuan: float
    estimated_task_attributed_green_energy_mwh: float
    estimated_green_coverage: float
    candidate_marginal_green_energy_mwh: float
    green_absorption_delta: MetricValue
    green_opportunity: bool

    def as_candidate_metrics(self) -> dict:
        return {
            "system_cost_yuan": self.candidate_marginal_system_cost_yuan,
            "green_coverage": self.estimated_green_coverage,
            "marginal_green_energy_mwh": (
                self.candidate_marginal_green_energy_mwh
            ),
            "green_absorption_delta": (
                0.0
                if self.green_absorption_delta.value is None
                else self.green_absorption_delta.value
            ),
            "green_opportunity": self.green_opportunity,
        }


@dataclass(frozen=True)
class TaskAccountingRecord:
    reservation_id: str
    task_id: str
    target_node: str
    task_power_mw: float
    task_energy_mwh: float
    task_direct_energy_cost_yuan: float
    task_attributed_cost_yuan: float
    task_attributed_green_energy_mwh: float
    green_coverage: float


@dataclass(frozen=True)
class AccountingReport:
    task_records: Tuple[TaskAccountingRecord, ...]
    node_bill_yuan: Mapping[str, float]
    node_green_used_mwh: Mapping[str, float]
    total_task_attributed_cost_yuan: float
    total_task_attributed_green_energy_mwh: float
    system_green_supply_mwh: float
    system_green_idle_mwh: float
    system_green_absorption_rate: MetricValue


@dataclass(frozen=True)
class _CandidateIntegralIndex:
    boundaries: Tuple[float, ...]
    rates: Tuple[Tuple[float, ...], ...]
    prefixes: Tuple[Tuple[float, ...], ...]
    hours_per_sim: float

    def covers(self, start_sim: float, end_sim: float) -> bool:
        return (
            start_sim >= self.boundaries[0] - 1e-12
            and end_sim <= self.boundaries[-1] + 1e-12
        )

    def _cumulative(self, time_sim: float) -> Tuple[float, ...]:
        if time_sim <= self.boundaries[0] + 1e-12:
            return tuple(0.0 for _ in self.prefixes)
        if time_sim >= self.boundaries[-1] - 1e-12:
            return tuple(prefix[-1] for prefix in self.prefixes)
        index = bisect_right(self.boundaries, time_sim) - 1
        elapsed_hours = (
            time_sim - self.boundaries[index]
        ) * self.hours_per_sim
        return tuple(
            prefix[index] + rate[index] * elapsed_hours
            for prefix, rate in zip(self.prefixes, self.rates)
        )

    def integrate(self, start_sim: float, end_sim: float) -> Tuple[float, ...]:
        if not self.covers(start_sim, end_sim):
            raise ValueError("candidate interval is outside integral-index coverage")
        before = self._cumulative(start_sim)
        after = self._cumulative(end_sim)
        return tuple(right - left for left, right in zip(before, after))

    def _cumulative_many(self, times_sim) -> np.ndarray:
        times = np.asarray(times_sim, dtype=np.float64)
        if times.ndim != 1:
            raise ValueError("candidate times must be one-dimensional")
        result = np.empty((times.size, len(self.prefixes)), dtype=np.float64)
        if times.size == 0:
            return result
        boundaries = np.asarray(self.boundaries, dtype=np.float64)
        rates = np.asarray(self.rates, dtype=np.float64)
        prefixes = np.asarray(self.prefixes, dtype=np.float64)
        low = times <= boundaries[0] + 1e-12
        high = times >= boundaries[-1] - 1e-12
        middle = ~(low | high)
        result[low] = 0.0
        result[high] = prefixes[:, -1]
        if np.any(middle):
            middle_times = times[middle]
            indices = np.searchsorted(
                boundaries, middle_times, side="right"
            ) - 1
            elapsed_hours = (
                middle_times - boundaries[indices]
            ) * self.hours_per_sim
            result[middle] = (
                prefixes[:, indices].T
                + rates[:, indices].T * elapsed_hours[:, None]
            )
        return result

    def integrate_many(self, starts_sim, ends_sim) -> np.ndarray:
        starts = np.asarray(starts_sim, dtype=np.float64)
        ends = np.asarray(ends_sim, dtype=np.float64)
        if starts.ndim != 1 or ends.ndim != 1 or starts.shape != ends.shape:
            raise ValueError(
                "candidate interval arrays must be one-dimensional and aligned"
            )
        if starts.size == 0:
            return np.empty((0, len(self.prefixes)), dtype=np.float64)
        if (
            np.any(starts < self.boundaries[0] - 1e-12)
            or np.any(ends > self.boundaries[-1] + 1e-12)
        ):
            raise ValueError("candidate interval is outside integral-index coverage")
        return self._cumulative_many(ends) - self._cumulative_many(starts)


class ExogenousEnergyAccounting:
    """Interval integrator for candidate counterfactuals and realized ledgers."""

    def __init__(
        self,
        time_converter: TimeConverter,
        power_model: LinearPowerModel,
        tariff_by_node: Mapping[str, PiecewiseConstantForecast],
        green_by_node: Mapping[str, PiecewiseConstantForecast],
        node_bill_rate_model: Optional[Callable[..., float]] = None,
    ):
        self.time_converter = time_converter
        self.power_model = power_model
        self.tariff_by_node = dict(tariff_by_node)
        self.green_by_node = dict(green_by_node)
        if not self.tariff_by_node or not self.green_by_node:
            raise ValueError("tariff and green forecasts cannot be empty")
        self.node_bill_rate_model = node_bill_rate_model
        self._candidate_index_cache = OrderedDict()
        self._candidate_index_cache_capacity = 256

    def __getstate__(self):
        state = dict(self.__dict__)
        # Performance caches are reproducible from immutable forecasts and the
        # reservation snapshot; they must not inflate exact resume checkpoints.
        state["_candidate_index_cache"] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_candidate_index_cache" not in self.__dict__:
            self._candidate_index_cache = OrderedDict()
        if "_candidate_index_cache_capacity" not in self.__dict__:
            self._candidate_index_cache_capacity = 256

    def _forecasts(self, node: str):
        try:
            return self.tariff_by_node[node], self.green_by_node[node]
        except KeyError:
            raise ValueError(f"missing physical forecast for node {node}")

    def _bill_rate(
        self,
        node: str,
        time_sim: float,
        total_power_mw: float,
        tariff_yuan_per_mwh: float,
        green_power_mw: float,
    ) -> float:
        if self.node_bill_rate_model is None:
            return tariff_yuan_per_mwh * total_power_mw
        value = self.node_bill_rate_model(
            node=node,
            time_sim=time_sim,
            total_task_power_mw=total_power_mw,
            tariff_yuan_per_mwh=tariff_yuan_per_mwh,
            green_power_mw=green_power_mw,
        )
        return finite_number("node_bill_rate_yuan_per_hour", value)

    @staticmethod
    def _allocation_power_at(
        allocations: Iterable[CalendarAllocation],
        node: str,
        time_sim: float,
        power_model: LinearPowerModel,
    ) -> float:
        return sum(
            power_model.task_power_mw(item.amount)
            for item in allocations
            if item.resource_id == node and item.interval_sim.contains(time_sim)
        )

    def _candidate_boundaries(
        self,
        interval: TimeInterval,
        node: str,
        snapshot: ReservationSnapshot,
    ) -> Tuple[float, ...]:
        tariff, green = self._forecasts(node)
        boundaries = set(tariff.boundaries(interval)) | set(green.boundaries(interval))
        for allocation in snapshot.cpu_calendar_view:
            if allocation.resource_id == node and allocation.interval_sim.overlaps(interval):
                boundaries.add(max(interval.start_sim, allocation.interval_sim.start_sim))
                boundaries.add(min(interval.end_sim, allocation.interval_sim.end_sim))
        return tuple(sorted(boundaries))

    def evaluate_candidate(
        self,
        *,
        task: TaskSpec,
        target_node: str,
        compute_start_sim: float,
        compute_end_sim: float,
        reservation_snapshot: ReservationSnapshot,
    ) -> CandidateAccountingMetrics:
        interval = TimeInterval(compute_start_sim, compute_end_sim)
        tariff, green = self._forecasts(target_node)
        task_power = self.power_model.task_power_mw(task.cpu_demand)
        task_energy = 0.0
        direct_cost = 0.0
        marginal_cost = 0.0
        attributed_green = 0.0
        marginal_green = 0.0
        green_supply_energy = 0.0
        boundaries = self._candidate_boundaries(
            interval,
            target_node,
            reservation_snapshot,
        )
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if right <= left:
                continue
            probe = left + (right - left) / 2.0
            hours = self.time_converter.sim_to_hours(right - left)
            tariff_value = tariff.value_at(probe)
            green_value = green.value_at(probe)
            existing = self._allocation_power_at(
                reservation_snapshot.cpu_calendar_view,
                target_node,
                probe,
                self.power_model,
            )
            with_task = existing + task_power
            task_energy += task_power * hours
            direct_cost += tariff_value * task_power * hours
            marginal_cost += (
                self._bill_rate(
                    target_node, probe, with_task, tariff_value, green_value
                )
                - self._bill_rate(
                    target_node, probe, existing, tariff_value, green_value
                )
            ) * hours
            coverage = min(1.0, green_value / with_task)
            attributed_green += task_power * coverage * hours
            marginal_green += (
                min(green_value, with_task) - min(green_value, existing)
            ) * hours
            green_supply_energy += green_value * hours
        if green_supply_energy > 0.0:
            absorption = MetricValue.valid(
                marginal_green / green_supply_energy,
                numerator=marginal_green,
                denominator=green_supply_energy,
            )
            opportunity = True
        else:
            absorption = MetricValue.not_applicable("zero green supply over interval")
            opportunity = False
        return CandidateAccountingMetrics(
            task_energy,
            direct_cost,
            marginal_cost,
            attributed_green,
            attributed_green / task_energy,
            marginal_green,
            absorption,
            opportunity,
        )

    def _build_candidate_integral_index(
        self,
        task: TaskSpec,
        target_node: str,
        reservation_snapshot: ReservationSnapshot,
        domain_start_sim: float,
        domain_end_sim: float,
    ) -> _CandidateIntegralIndex:
        interval = TimeInterval(domain_start_sim, domain_end_sim)
        tariff, green = self._forecasts(target_node)
        boundaries = set(tariff.boundaries(interval)) | set(
            green.boundaries(interval)
        )
        for allocation in reservation_snapshot.cpu_calendar_view:
            if (
                allocation.resource_id == target_node
                and allocation.interval_sim.overlaps(interval)
            ):
                boundaries.add(max(domain_start_sim, allocation.interval_sim.start_sim))
                boundaries.add(min(domain_end_sim, allocation.interval_sim.end_sim))
        ordered = tuple(sorted(boundaries))
        if len(ordered) < 2:
            raise ValueError("candidate integral index has no positive interval")

        task_power = self.power_model.task_power_mw(task.cpu_demand)
        rate_rows = [[] for _ in range(6)]
        for left, right in zip(ordered[:-1], ordered[1:]):
            probe = left + (right - left) / 2.0
            tariff_value = tariff.value_at(probe)
            green_value = green.value_at(probe)
            existing = self._allocation_power_at(
                reservation_snapshot.cpu_calendar_view,
                target_node,
                probe,
                self.power_model,
            )
            with_task = existing + task_power
            rate_rows[0].append(task_power)
            rate_rows[1].append(tariff_value * task_power)
            # This index is used only for the exogenous linear billing case.
            rate_rows[2].append(tariff_value * task_power)
            rate_rows[3].append(
                task_power * min(1.0, green_value / with_task)
            )
            rate_rows[4].append(
                min(green_value, with_task) - min(green_value, existing)
            )
            rate_rows[5].append(green_value)

        hours_per_sim = self.time_converter.sim_to_hours(1.0)
        rates = tuple(tuple(row) for row in rate_rows)
        prefixes = []
        for row in rates:
            values = [0.0]
            for index, rate in enumerate(row):
                duration_hours = (
                    ordered[index + 1] - ordered[index]
                ) * hours_per_sim
                values.append(values[-1] + rate * duration_hours)
            prefixes.append(tuple(values))
        return _CandidateIntegralIndex(
            ordered,
            rates,
            tuple(prefixes),
            hours_per_sim,
        )

    def _candidate_integral_index(
        self,
        task: TaskSpec,
        target_node: str,
        reservation_snapshot: ReservationSnapshot,
        first_start_sim: float,
    ) -> _CandidateIntegralIndex:
        tariff, green = self._forecasts(target_node)
        forecast_start = max(
            tariff.segments[0].interval_sim.start_sim,
            green.segments[0].interval_sim.start_sim,
        )
        forecast_end = min(
            tariff.segments[-1].interval_sim.end_sim,
            green.segments[-1].interval_sim.end_sim,
        )
        domain_start = max(forecast_start, first_start_sim)
        domain_end = min(
            forecast_end,
            task.absolute_latest_start_sim + task.execution_duration_sim,
        )
        key = (
            task,
            target_node,
            reservation_snapshot,
            domain_start,
            domain_end,
        )
        cached = self._candidate_index_cache.get(key)
        if cached is not None:
            self._candidate_index_cache.move_to_end(key)
            return cached
        index = self._build_candidate_integral_index(
            task,
            target_node,
            reservation_snapshot,
            domain_start,
            domain_end,
        )
        self._candidate_index_cache[key] = index
        self._candidate_index_cache.move_to_end(key)
        while len(self._candidate_index_cache) > self._candidate_index_cache_capacity:
            self._candidate_index_cache.popitem(last=False)
        return index

    def candidate_metric_evaluator(self, reservation_snapshot: ReservationSnapshot):
        indices = {}

        def evaluate(**kwargs):
            task = kwargs["task"]
            target_node = kwargs.get("target_node", kwargs["path"].target_node)
            snapshot = kwargs.get("reservation_snapshot", reservation_snapshot)
            start = kwargs["compute_start_sim"]
            end = kwargs["compute_end_sim"]
            if self.node_bill_rate_model is None:
                local_key = (task, target_node, snapshot)
                index = indices.get(local_key)
                if index is None or not index.covers(start, end):
                    index = self._candidate_integral_index(
                        task,
                        target_node,
                        snapshot,
                        start,
                    )
                    indices[local_key] = index
                values = index.integrate(start, end)
                (
                    task_energy,
                    direct_cost,
                    marginal_cost,
                    attributed_green,
                    marginal_green,
                    green_supply_energy,
                ) = values
                if green_supply_energy > 0.0:
                    absorption = MetricValue.valid(
                        marginal_green / green_supply_energy,
                        numerator=marginal_green,
                        denominator=green_supply_energy,
                    )
                    opportunity = True
                else:
                    absorption = MetricValue.not_applicable(
                        "zero green supply over interval"
                    )
                    opportunity = False
                return CandidateAccountingMetrics(
                    task_energy,
                    direct_cost,
                    marginal_cost,
                    attributed_green,
                    attributed_green / task_energy,
                    marginal_green,
                    absorption,
                    opportunity,
                ).as_candidate_metrics()
            return self.evaluate_candidate(
                task=task,
                target_node=target_node,
                compute_start_sim=start,
                compute_end_sim=end,
                reservation_snapshot=snapshot,
            ).as_candidate_metrics()

        def evaluate_batch(**kwargs):
            starts = np.asarray(
                kwargs["compute_start_sim"], dtype=np.float64
            )
            ends = np.asarray(
                kwargs["compute_end_sim"], dtype=np.float64
            )
            if (
                starts.ndim != 1
                or ends.ndim != 1
                or starts.shape != ends.shape
            ):
                raise ValueError(
                    "candidate interval arrays must be one-dimensional and aligned"
                )
            task = kwargs["task"]
            target_node = kwargs.get(
                "target_node", kwargs["path"].target_node
            )
            snapshot = kwargs.get(
                "reservation_snapshot", reservation_snapshot
            )
            if starts.size == 0:
                return {
                    "system_cost_yuan": np.empty(0, dtype=np.float64),
                    "green_coverage": np.empty(0, dtype=np.float64),
                    "marginal_green_energy_mwh": np.empty(
                        0, dtype=np.float64
                    ),
                    "green_absorption_delta": np.empty(
                        0, dtype=np.float64
                    ),
                    "green_opportunity": np.empty(0, dtype=np.bool_),
                }
            if self.node_bill_rate_model is not None:
                rows = [
                    evaluate(
                        task=task,
                        path=kwargs["path"],
                        target_node=target_node,
                        compute_start_sim=float(start),
                        compute_end_sim=float(end),
                        reservation_snapshot=snapshot,
                    )
                    for start, end in zip(starts, ends)
                ]
                return {
                    key: np.asarray([row[key] for row in rows])
                    for key in rows[0]
                }
            local_key = (task, target_node, snapshot)
            index = indices.get(local_key)
            if (
                index is None
                or not index.covers(
                    float(np.min(starts)), float(np.max(ends))
                )
            ):
                index = self._candidate_integral_index(
                    task,
                    target_node,
                    snapshot,
                    float(np.min(starts)),
                )
                indices[local_key] = index
            values = index.integrate_many(starts, ends)
            task_energy = values[:, 0]
            marginal_cost = values[:, 2]
            attributed_green = values[:, 3]
            marginal_green = values[:, 4]
            green_supply_energy = values[:, 5]
            green_opportunity = green_supply_energy > 0.0
            green_coverage = attributed_green / task_energy
            absorption = np.divide(
                marginal_green,
                green_supply_energy,
                out=np.zeros_like(marginal_green),
                where=green_opportunity,
            )
            return {
                "system_cost_yuan": marginal_cost,
                "green_coverage": green_coverage,
                "marginal_green_energy_mwh": marginal_green,
                "green_absorption_delta": absorption,
                "green_opportunity": green_opportunity,
            }

        evaluate.evaluate_batch = evaluate_batch
        return evaluate

    def realize(
        self,
        reservations: Iterable[Reservation],
        accounting_interval: Optional[TimeInterval] = None,
    ) -> AccountingReport:
        items = tuple(reservations)
        by_node = {}
        for reservation in items:
            by_node.setdefault(reservation.target_node, []).append(reservation)
        records = []
        node_bills = {}
        node_greens = {}
        attributed_cost = {item.reservation_id: 0.0 for item in items}
        attributed_green = {item.reservation_id: 0.0 for item in items}
        direct_cost = {item.reservation_id: 0.0 for item in items}

        for node, node_items in by_node.items():
            tariff, green = self._forecasts(node)
            start = min(item.compute_interval_sim.start_sim for item in node_items)
            end = max(item.compute_interval_sim.end_sim for item in node_items)
            envelope = TimeInterval(start, end)
            boundaries = set(tariff.boundaries(envelope)) | set(green.boundaries(envelope))
            for item in node_items:
                boundaries.add(item.compute_interval_sim.start_sim)
                boundaries.add(item.compute_interval_sim.end_sim)
            ordered = sorted(boundaries)
            node_bill = 0.0
            node_green = 0.0
            for left, right in zip(ordered[:-1], ordered[1:]):
                if right <= left:
                    continue
                probe = left + (right - left) / 2.0
                active = [
                    item for item in node_items
                    if item.compute_interval_sim.contains(probe)
                ]
                if not active:
                    continue
                hours = self.time_converter.sim_to_hours(right - left)
                powers = {
                    item.reservation_id: self.power_model.task_power_mw(item.cpu_amount)
                    for item in active
                }
                total_power = sum(powers.values())
                tariff_value = tariff.value_at(probe)
                green_value = green.value_at(probe)
                bill_rate = self._bill_rate(
                    node, probe, total_power, tariff_value, green_value
                )
                used_green = min(green_value, total_power)
                node_bill += bill_rate * hours
                node_green += used_green * hours
                for item in active:
                    share = powers[item.reservation_id] / total_power
                    attributed_cost[item.reservation_id] += bill_rate * share * hours
                    attributed_green[item.reservation_id] += used_green * share * hours
                    direct_cost[item.reservation_id] += (
                        tariff_value * powers[item.reservation_id] * hours
                    )
            node_bills[node] = node_bill
            node_greens[node] = node_green

        for item in sorted(items, key=lambda value: value.task_id):
            power = self.power_model.task_power_mw(item.cpu_amount)
            energy = power * self.time_converter.sim_to_hours(
                item.compute_interval_sim.duration_sim
            )
            records.append(TaskAccountingRecord(
                item.reservation_id,
                item.task_id,
                item.target_node,
                power,
                energy,
                direct_cost[item.reservation_id],
                attributed_cost[item.reservation_id],
                attributed_green[item.reservation_id],
                attributed_green[item.reservation_id] / energy,
            ))
        total_cost = math.fsum(
            record.task_attributed_cost_yuan for record in records
        )
        total_green = math.fsum(
            record.task_attributed_green_energy_mwh for record in records
        )
        if not math.isclose(
            total_cost,
            math.fsum(node_bills.values()),
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise RuntimeError("task attributed cost does not conserve node bill")
        if not math.isclose(
            total_green,
            math.fsum(node_greens.values()),
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise RuntimeError("task attributed green energy is not conserved")
        if accounting_interval is None and items:
            accounting_interval = TimeInterval(
                min(item.compute_interval_sim.start_sim for item in items),
                max(item.compute_interval_sim.end_sim for item in items),
            )
        system_supply = 0.0
        if accounting_interval is not None:
            for green in self.green_by_node.values():
                boundaries = green.boundaries(accounting_interval)
                for left, right in zip(boundaries[:-1], boundaries[1:]):
                    if right > left:
                        probe = left + (right - left) / 2.0
                        system_supply += green.value_at(probe) * (
                            self.time_converter.sim_to_hours(right - left)
                        )
        system_used = sum(node_greens.values())
        idle = system_supply - system_used
        if idle < -1e-9:
            raise RuntimeError("green energy use exceeds accounting-interval supply")
        idle = max(0.0, idle)
        absorption = (
            MetricValue.valid(
                system_used / system_supply,
                numerator=system_used,
                denominator=system_supply,
            )
            if system_supply > 0.0
            else MetricValue.not_applicable("zero system green supply")
        )
        return AccountingReport(
            tuple(records),
            node_bills,
            node_greens,
            total_cost,
            total_green,
            system_supply,
            idle,
            absorption,
        )
