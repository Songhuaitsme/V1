"""Composition root that builds the frozen v1.0 scheduler from project data."""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from shared import config
from shared.infrastructure import InfrastructureContext
from shared.task_manager import TaskManager
from v1.accounting import (
    ExogenousEnergyAccounting,
    ForecastSegment,
    LinearPowerModel,
    MetricsLedger,
    PiecewiseConstantForecast,
)
from v1.domain.models import SlaType, TaskSpec, TaskState
from v1.domain.reservations import TimeInterval
from v1.domain.units import TimeConverter, positive_finite, validate_scheduling_grid
from v1.learning import (
    CandidateDQNPolicy,
    CandidateFeatureConfig,
    CandidateFeatureEncoder,
    SharedCandidateQNetwork,
)
from v1.scheduler import (
    CandidateGenerator,
    EarliestFeasiblePolicy,
    EqualWeightPolicy,
    HighestGreenPolicy,
    LowestCostPolicy,
    ObjectiveConfig,
    ReservationCalendar,
    StaticPathProvider,
    TransmissionModel,
    V1Scheduler,
)


@dataclass
class V1Runtime:
    infrastructure: InfrastructureContext
    task_manager: TaskManager
    calendar: ReservationCalendar
    accounting: ExogenousEnergyAccounting
    metrics_ledger: MetricsLedger
    scheduler: V1Scheduler
    time_converter: TimeConverter
    candidate_feature_encoder: CandidateFeatureEncoder
    candidate_q_network: Optional[SharedCandidateQNetwork] = None

    def global_state(self, task: Optional[TaskSpec] = None) -> np.ndarray:
        counts = self.scheduler.state_machine.count_by_state()
        now = self.scheduler.event_engine.current_time_sim
        day = config.TRAFFIC_DAY_DURATION_IN_SIM
        phase = 2.0 * math.pi * ((now % day) / day)
        snapshot = self.calendar.snapshot()
        utilizations = []
        for node in self.infrastructure.compute_nodes:
            capacity = self.calendar.node_capacity(node)
            used = sum(
                item.amount
                for item in snapshot.cpu_calendar_view
                if item.resource_id == node and item.interval_sim.contains(now)
            )
            utilizations.append(used / capacity)
        sla_one_hot = [0.0, 0.0, 0.0]
        if task is not None:
            sla_one_hot[list(SlaType).index(task.sla_type)] = 1.0
            max_capacity = max(
                self.calendar.node_capacity(node)
                for node in self.infrastructure.compute_nodes
            )
            task_values = [
                task.cpu_demand / max_capacity,
                task.execution_duration_sim / day,
                task.data_size_mb / 1000.0,
                task.bandwidth_demand_mbps / config.DEFAULT_LINK_BANDWIDTH,
                task.latest_start_limit_sim / day,
            ]
        else:
            task_values = [0.0] * 5
        queue_norm = max(1, config.MAX_QUEUE_LENGTH)
        state = np.asarray([
            math.sin(phase),
            math.cos(phase),
            *sla_one_hot,
            *task_values,
            counts[TaskState.QUEUED] / queue_norm,
            counts[TaskState.PENDING_UNCOMMITTED] / queue_norm,
            counts[TaskState.RESERVED] / queue_norm,
            counts[TaskState.TRANSMITTING] / queue_norm,
            counts[TaskState.RUNNING] / queue_norm,
            float(np.mean(utilizations)) if utilizations else 0.0,
            float(np.max(utilizations)) if utilizations else 0.0,
            float(np.std(utilizations, ddof=0)) if utilizations else 0.0,
        ], dtype=np.float32)
        if not config.V1_DQN_USE_GLOBAL_STATE:
            state.fill(0.0)
        return state


def _build_forecast_segments(start_sim, end_sim, step_sim, value_provider):
    segments = []
    cursor = start_sim
    while cursor < end_sim - 1e-12:
        right = min(end_sim, cursor + step_sim)
        segments.append(ForecastSegment(
            TimeInterval(cursor, right),
            value_provider(cursor + (right - cursor) / 2.0),
        ))
        cursor = right
    return tuple(segments)


def _v1_node_bill_rate_model(
    *, node, time_sim, total_task_power_mw,
    tariff_yuan_per_mwh, green_power_mw,
):
    """Optional V1 subsidy/tax layer kept separate from exogenous tariff."""

    mode = config.V1_TARIFF_MODE
    power = max(0.0, float(total_task_power_mw))
    if power <= 0.0:
        return 0.0
    multiplier = 1.0
    if green_power_mw >= power and mode in {"green_subsidy", "full"}:
        surplus = (green_power_mw - power) / max(green_power_mw, 1e-12)
        multiplier = 1.0 - config.GREEN_SUBSIDY_RATE * surplus
    elif green_power_mw < power and mode in {"carbon_tax", "full"}:
        grey_ratio = 1.0 - max(0.0, green_power_mw) / power
        multiplier = 1.0 + config.CARBON_TAX_RATE * grey_ratio
    return tariff_yuan_per_mwh * power * multiplier


def create_v1_runtime(
    *,
    policy_name: str = "earliest_feasible",
    forecast_start_sim: float = 0.0,
    forecast_end_sim: Optional[float] = None,
    random_seed: int = 0,
    device: str = "cpu",
    candidate_chunk_size: Optional[int] = None,
    objective_config: Optional[ObjectiveConfig] = None,
) -> V1Runtime:
    validate_scheduling_grid(config.SCHEDULING_CYCLE, config.GLOBAL_TIME_STEP_DURATION)
    converter = TimeConverter.from_traffic_day_duration(
        config.TRAFFIC_DAY_DURATION_IN_SIM,
        config.SIM_SECONDS_PER_UNIT,
    )
    infrastructure = InfrastructureContext.create()
    graph = infrastructure.topo_manager.graph
    node_capacities = {
        node: infrastructure.node_resources[node]["total"]
        for node in infrastructure.compute_nodes
    }
    link_capacities = {
        (u, v): data.get("capacity", config.DEFAULT_LINK_BANDWIDTH)
        for u, v, data in graph.edges(data=True)
    }
    calendar = ReservationCalendar(node_capacities, link_capacities)
    path_provider = StaticPathProvider(
        graph,
        max_paths_per_target=int(config.V1_CANDIDATE_PATH_K),
    )
    transmission = TransmissionModel(
        converter,
        config.FIBER_PROPAGATION_SPEED_KM_PER_S,
    )
    generator = CandidateGenerator(
        infrastructure.compute_nodes,
        config.SCHEDULING_CYCLE,
        path_provider,
        transmission,
        calendar,
        config.V1_TIME_TOLERANCE,
        candidate_mode=config.V1_CANDIDATE_MODE,
        active_wait_enabled=config.V1_ACTIVE_WAIT_ENABLED,
        pool_max_by_sla=config.V1_CANDIDATE_POOL_MAX_BY_SLA,
        pool_node_limit_by_sla=config.V1_CANDIDATE_POOL_NODE_LIMIT_BY_SLA,
        pool_time_samples_by_sla=(
            config.V1_CANDIDATE_POOL_TIME_SAMPLES_BY_SLA
        ),
    )
    forecast_end = (
        positive_finite("forecast_end_sim", forecast_end_sim)
        if forecast_end_sim is not None
        else config.V1_FORECAST_HORIZON_SIM
    )
    step = positive_finite("V1_FORECAST_STEP_SIM", config.V1_FORECAST_STEP_SIM)
    tariff = {}
    green = {}
    for node in infrastructure.compute_nodes:
        tariff[node] = PiecewiseConstantForecast.tariff_yuan_per_mwh(
            _build_forecast_segments(
                forecast_start_sim,
                forecast_end,
                step,
                lambda time_sim, node=node: (
                    infrastructure.pricing_manager.get_external_tariff_yuan_per_mwh(
                        node, time_sim, mode=config.V1_TARIFF_MODE
                    )
                ),
            ),
            version="perfect-exogenous-tariff-v1",
        )
        green[node] = PiecewiseConstantForecast.green_power_mw(
            _build_forecast_segments(
                forecast_start_sim,
                forecast_end,
                step,
                lambda time_sim, node=node: (
                    infrastructure.pricing_manager.get_green_supply_mw(node, time_sim)
                ),
            ),
            version="perfect-exogenous-green-v1",
        )
    accounting = ExogenousEnergyAccounting(
        converter,
        LinearPowerModel(config.INCREMENTAL_CPU_POWER_MW_PER_CPU),
        tariff,
        green,
        node_bill_rate_model=(
            _v1_node_bill_rate_model
            if config.V1_TARIFF_MODE
            in {"green_subsidy", "carbon_tax", "full"}
            else None
        ),
    )
    ledger = MetricsLedger(accounting)
    task_manager = TaskManager(
        infrastructure.base_stations,
        total_compute_capacity=sum(node_capacities.values()),
    )
    objective = objective_config or ObjectiveConfig(
        config.V1_COST_REFERENCE_YUAN,
        config.V1_COST_SCALE_YUAN,
        config.V1_GREEN_ABSORPTION_DELTA_SCALE,
        config.V1_OBJECTIVE_COST_WEIGHT,
        config.V1_OBJECTIVE_GREEN_WEIGHT,
        config.V1_OBJECTIVE_BALANCE_WEIGHT,
        config.V1_SOFT_TARDINESS_WEIGHT,
        config.V1_FLEXIBLE_TARDINESS_WEIGHT,
    )
    policies = {
        "earliest_feasible": EarliestFeasiblePolicy(),
        "lowest_cost": LowestCostPolicy(),
        "highest_green": HighestGreenPolicy(),
        "equal_weight": EqualWeightPolicy(objective),
    }
    initial_policy = policies.get(policy_name, EarliestFeasiblePolicy())
    scheduler = V1Scheduler(
        calendar,
        generator,
        config.MAX_QUEUE_LENGTH,
        config.MAX_TASKS_PER_CYCLE,
        config.MAX_COMMIT_ATTEMPTS_PER_DECISION,
        policy=initial_policy,
        metrics_ledger=ledger,
    )
    feature_encoder = CandidateFeatureEncoder(
        {node: index for index, node in enumerate(infrastructure.compute_nodes)},
        CandidateFeatureConfig(
            time_scale_sim=config.TRAFFIC_DAY_DURATION_IN_SIM,
            cost_scale_yuan=config.V1_COST_SCALE_YUAN,
            absorption_delta_scale=config.V1_GREEN_ABSORPTION_DELTA_SCALE,
            cpu_scale=max(node_capacities.values()),
            bandwidth_scale_mbps=config.DEFAULT_LINK_BANDWIDTH,
        ),
        disabled_feature_groups=(
            config.V1_DISABLED_CANDIDATE_FEATURE_GROUPS
        ),
    )
    runtime = V1Runtime(
        infrastructure,
        task_manager,
        calendar,
        accounting,
        ledger,
        scheduler,
        converter,
        feature_encoder,
    )
    if policy_name == "candidate_dqn":
        sample_state = runtime.global_state(None)
        network = SharedCandidateQNetwork(
            len(sample_state),
            feature_encoder.feature_dim,
            config.V1_CANDIDATE_DQN_HIDDEN_DIM,
        )
        scheduler.policy = CandidateDQNPolicy(
            network,
            feature_encoder,
            runtime.global_state,
            epsilon=0.0,
            random_seed=random_seed,
            candidate_chunk_size=(
                config.V1_CANDIDATE_CHUNK_SIZE
                if candidate_chunk_size is None
                else candidate_chunk_size
            ),
            device=device,
        )
        runtime.candidate_q_network = network
    elif policy_name not in policies:
        raise ValueError(f"unknown v1 policy: {policy_name}")
    return runtime
