"""Explicit unit conversion primitives for the v1.0 domain model.

The legacy scheduler stores most physical values as bare floats.  This module
is the single conversion boundary used by new v1.0 code so invalid values are
rejected instead of being silently replaced with epsilon defaults.
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional


SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 24.0 * SECONDS_PER_HOUR
MEGABITS_PER_DECIMAL_MEGABYTE = 8.0
YUAN_PER_MWH_PER_YUAN_PER_KWH = 1000.0


class UnitValidationError(ValueError):
    """Raised when a conversion receives an invalid physical value."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


def finite_number(field: str, value: Real) -> float:
    """Return *value* as float after strict finite-number validation."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise UnitValidationError(field, "must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise UnitValidationError(field, "must be finite")
    return converted


def positive_finite(field: str, value: Real) -> float:
    converted = finite_number(field, value)
    if converted <= 0.0:
        raise UnitValidationError(field, "must be greater than 0")
    return converted


def non_negative_finite(field: str, value: Real) -> float:
    converted = finite_number(field, value)
    if converted < 0.0:
        raise UnitValidationError(field, "must be greater than or equal to 0")
    return converted


@dataclass(frozen=True)
class TimeConverter:
    """Convert simulation time to physical seconds and hours."""

    seconds_per_sim_unit: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "seconds_per_sim_unit",
            positive_finite("seconds_per_sim_unit", self.seconds_per_sim_unit),
        )

    @classmethod
    def from_traffic_day_duration(
        cls,
        traffic_day_duration_in_sim: Real,
        declared_seconds_per_sim_unit: Optional[Real] = None,
    ) -> "TimeConverter":
        day_duration = positive_finite(
            "traffic_day_duration_in_sim",
            traffic_day_duration_in_sim,
        )
        derived = SECONDS_PER_DAY / day_duration
        if declared_seconds_per_sim_unit is not None:
            declared = positive_finite(
                "declared_seconds_per_sim_unit",
                declared_seconds_per_sim_unit,
            )
            if not math.isclose(declared, derived, rel_tol=1e-12, abs_tol=1e-12):
                raise UnitValidationError(
                    "declared_seconds_per_sim_unit",
                    f"must equal 86400 / traffic_day_duration_in_sim ({derived})",
                )
        return cls(derived)

    def seconds_to_sim(self, seconds: Real) -> float:
        return finite_number("seconds", seconds) / self.seconds_per_sim_unit

    def sim_to_seconds(self, sim_time: Real) -> float:
        return finite_number("sim_time", sim_time) * self.seconds_per_sim_unit

    def sim_to_hours(self, sim_time: Real) -> float:
        return self.sim_to_seconds(sim_time) / SECONDS_PER_HOUR

    def hours_to_sim(self, hours: Real) -> float:
        return self.seconds_to_sim(finite_number("hours", hours) * SECONDS_PER_HOUR)

    def scheduling_cycle_seconds(self, scheduling_cycle_sim: Real) -> float:
        cycle = positive_finite("scheduling_cycle", scheduling_cycle_sim)
        return self.sim_to_seconds(cycle)


def validate_scheduling_grid(
    scheduling_cycle: Real,
    global_time_step_duration: Optional[Real] = None,
) -> float:
    """Validate the single scheduling-grid source required by v1.0."""

    cycle = positive_finite("scheduling_cycle", scheduling_cycle)
    if global_time_step_duration is not None:
        global_step = positive_finite(
            "global_time_step_duration",
            global_time_step_duration,
        )
        if not math.isclose(cycle, global_step, rel_tol=1e-12, abs_tol=1e-12):
            raise UnitValidationError(
                "global_time_step_duration",
                "must equal scheduling_cycle",
            )
    return cycle


class DataUnitConverter:
    """Decimal data-unit conversions used by the transmission contract."""

    @staticmethod
    def decimal_mb_to_megabits(data_mb: Real) -> float:
        return (
            non_negative_finite("data_size_mb", data_mb)
            * MEGABITS_PER_DECIMAL_MEGABYTE
        )


class TariffConverter:
    """Electricity tariff conversions; negative external prices remain valid."""

    @staticmethod
    def yuan_per_kwh_to_yuan_per_mwh(value: Real) -> float:
        return finite_number("yuan_per_kwh", value) * YUAN_PER_MWH_PER_YUAN_PER_KWH


def cpu_work_sim_units(cpu_demand: Real, execution_duration_sim: Real) -> float:
    cpu = positive_finite("cpu_demand", cpu_demand)
    duration = positive_finite("execution_duration_sim", execution_duration_sim)
    return cpu * duration


def cpu_work_cpu_hours(
    cpu_demand: Real,
    execution_duration_sim: Real,
    converter: TimeConverter,
) -> float:
    return positive_finite("cpu_demand", cpu_demand) * converter.sim_to_hours(
        positive_finite("execution_duration_sim", execution_duration_sim)
    )
