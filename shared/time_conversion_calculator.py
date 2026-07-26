import argparse
import json
from dataclasses import asdict, dataclass
from typing import Optional

from shared import config
from v1.domain.units import (
    SECONDS_PER_DAY,
    TimeConverter,
    non_negative_finite,
    positive_finite,
)



@dataclass
class TimeConversion:
    scheduling_cycle: float
    traffic_day_duration_in_sim: float
    max_steps: int
    base_tasks_per_second: float
    seconds_per_sim_unit: float
    cycle_seconds: float
    cycles_per_sim_day: float
    cycles_per_sim_hour: float
    total_sim_units: float
    total_business_seconds: float
    total_business_hours: float
    total_business_days: float
    raw_lambda_per_cycle: float


def calculate_time_conversion(
    scheduling_cycle: float,
    traffic_day_duration_in_sim: float,
    max_steps: int,
    base_tasks_per_second: float,
) -> TimeConversion:
    scheduling_cycle = positive_finite("scheduling_cycle", scheduling_cycle)
    traffic_day_duration_in_sim = positive_finite(
        "traffic_day_duration_in_sim",
        traffic_day_duration_in_sim,
    )
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer")
    base_tasks_per_second = non_negative_finite(
        "base_tasks_per_second",
        base_tasks_per_second,
    )

    converter = TimeConverter.from_traffic_day_duration(
        traffic_day_duration_in_sim
    )
    seconds_per_sim_unit = converter.seconds_per_sim_unit
    cycle_seconds = converter.scheduling_cycle_seconds(scheduling_cycle)
    cycles_per_sim_day = traffic_day_duration_in_sim / scheduling_cycle
    cycles_per_sim_hour = cycles_per_sim_day / 24.0
    total_sim_units = max_steps * scheduling_cycle
    total_business_seconds = max_steps * cycle_seconds
    total_business_hours = total_business_seconds / 3600.0
    total_business_days = total_business_seconds / SECONDS_PER_DAY
    raw_lambda_per_cycle = base_tasks_per_second * cycle_seconds

    return TimeConversion(
        scheduling_cycle=scheduling_cycle,
        traffic_day_duration_in_sim=traffic_day_duration_in_sim,
        max_steps=max_steps,
        base_tasks_per_second=base_tasks_per_second,
        seconds_per_sim_unit=seconds_per_sim_unit,
        cycle_seconds=cycle_seconds,
        cycles_per_sim_day=cycles_per_sim_day,
        cycles_per_sim_hour=cycles_per_sim_hour,
        total_sim_units=total_sim_units,
        total_business_seconds=total_business_seconds,
        total_business_hours=total_business_hours,
        total_business_days=total_business_days,
        raw_lambda_per_cycle=raw_lambda_per_cycle,
    )


def sim_hour_at_cycle(cycle: int, scheduling_cycle: float, traffic_day_duration_in_sim: float) -> float:
    if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 0:
        raise ValueError("cycle must be a non-negative integer")
    scheduling_cycle = positive_finite("scheduling_cycle", scheduling_cycle)
    traffic_day_duration_in_sim = positive_finite(
        "traffic_day_duration_in_sim",
        traffic_day_duration_in_sim,
    )
    global_time = cycle * scheduling_cycle
    day_progress = (global_time % traffic_day_duration_in_sim) / traffic_day_duration_in_sim
    return day_progress * 24.0


def print_report(result: TimeConversion, cycle: Optional[int] = None) -> None:
    print("=== Time Conversion ===")
    print(f"SCHEDULING_CYCLE              = {result.scheduling_cycle:g}")
    print(f"TRAFFIC_DAY_DURATION_IN_SIM   = {result.traffic_day_duration_in_sim:g}")
    print(f"MAX_STEPS                     = {result.max_steps}")
    print(f"BASE_TASKS_PER_SECOND         = {result.base_tasks_per_second:g}")
    print()

    print("Formulas:")
    print("seconds_per_sim_unit = 86400 / TRAFFIC_DAY_DURATION_IN_SIM")
    print("cycle_seconds        = SCHEDULING_CYCLE * seconds_per_sim_unit")
    print("cycles_per_sim_day   = TRAFFIC_DAY_DURATION_IN_SIM / SCHEDULING_CYCLE")
    print("cycles_per_sim_hour  = cycles_per_sim_day / 24")
    print("total_sim_units      = MAX_STEPS * SCHEDULING_CYCLE")
    print("total_seconds        = MAX_STEPS * cycle_seconds")
    print("raw_lambda_per_cycle = BASE_TASKS_PER_SECOND * cycle_seconds")
    print("sim_hr(cycle)        = ((cycle * SCHEDULING_CYCLE) % TRAFFIC_DAY_DURATION_IN_SIM) / TRAFFIC_DAY_DURATION_IN_SIM * 24")
    print()

    print("Results:")
    print(f"1 sim unit                   = {result.seconds_per_sim_unit:.6f} seconds")
    print(f"1 scheduling cycle           = {result.cycle_seconds:.6f} seconds")
    print(f"cycles per simulated hour    = {result.cycles_per_sim_hour:.6f}")
    print(f"cycles per simulated day     = {result.cycles_per_sim_day:.6f}")
    print(f"total simulated units        = {result.total_sim_units:.6f}")
    print(f"total business seconds       = {result.total_business_seconds:.6f}")
    print(f"total business hours         = {result.total_business_hours:.6f}")
    print(f"total business days          = {result.total_business_days:.6f}")
    print(f"raw task lambda per cycle    = {result.raw_lambda_per_cycle:.6f}")

    if cycle is not None:
        hour = sim_hour_at_cycle(
            cycle,
            result.scheduling_cycle,
            result.traffic_day_duration_in_sim,
        )
        global_time = cycle * result.scheduling_cycle
        business_seconds = cycle * result.cycle_seconds
        print()
        print(f"Cycle query: {cycle}")
        print(f"global_time                  = {global_time:.6f}")
        print(f"business_seconds             = {business_seconds:.6f}")
        print(f"sim_hr                       = {hour:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate time conversion values for the DQN simulation.",
    )
    parser.add_argument(
        "--scheduling-cycle",
        type=float,
        default=config.SCHEDULING_CYCLE,
        help="Override config.SCHEDULING_CYCLE.",
    )
    parser.add_argument(
        "--traffic-day-duration",
        type=float,
        default=config.TRAFFIC_DAY_DURATION_IN_SIM,
        help="Override config.TRAFFIC_DAY_DURATION_IN_SIM.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=config.MAX_STEPS,
        help="Override config.MAX_STEPS.",
    )
    parser.add_argument(
        "--base-tasks-per-second",
        type=float,
        default=config.BASE_TASKS_PER_SECOND,
        help="Override config.BASE_TASKS_PER_SECOND.",
    )
    parser.add_argument(
        "--cycle",
        type=int,
        default=None,
        help="Show global_time, business seconds, and sim_hr at this cycle.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = calculate_time_conversion(
        scheduling_cycle=args.scheduling_cycle,
        traffic_day_duration_in_sim=args.traffic_day_duration,
        max_steps=args.max_steps,
        base_tasks_per_second=args.base_tasks_per_second,
    )

    if args.json:
        payload = asdict(result)
        if args.cycle is not None:
            payload["cycle_query"] = {
                "cycle": args.cycle,
                "global_time": args.cycle * result.scheduling_cycle,
                "business_seconds": args.cycle * result.cycle_seconds,
                "sim_hr": sim_hour_at_cycle(
                    args.cycle,
                    result.scheduling_cycle,
                    result.traffic_day_duration_in_sim,
                ),
            }
        print(json.dumps(payload, indent=2))
        return

    print_report(result, cycle=args.cycle)


if __name__ == "__main__":
    main()
