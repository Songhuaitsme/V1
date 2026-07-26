"""Strict piecewise-constant physical forecasts for v1.0 accounting."""

from dataclasses import dataclass
from bisect import bisect_right
from typing import Iterable, Tuple

from v1.domain.reservations import TimeInterval
from v1.domain.units import finite_number, non_negative_finite


class ForecastCoverageError(ValueError):
    pass


@dataclass(frozen=True)
class ForecastSegment:
    interval_sim: TimeInterval
    value: float


class PiecewiseConstantForecast:
    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("physical forecast is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        segments: Iterable[ForecastSegment],
        *,
        value_name: str,
        non_negative: bool,
        version: str = "forecast-1.0",
    ):
        if not isinstance(version, str) or not version:
            raise ValueError("forecast version must be non-empty")
        self.value_name = str(value_name)
        self.version = version
        normalized = []
        for segment in segments:
            value = (
                non_negative_finite(self.value_name, segment.value)
                if non_negative
                else finite_number(self.value_name, segment.value)
            )
            normalized.append(ForecastSegment(segment.interval_sim, value))
        normalized.sort(key=lambda item: item.interval_sim.start_sim)
        if not normalized:
            raise ValueError("forecast must contain at least one segment")
        for previous, current in zip(normalized[:-1], normalized[1:]):
            if current.interval_sim.start_sim < previous.interval_sim.end_sim:
                raise ValueError("forecast segments must not overlap")
        self._segments: Tuple[ForecastSegment, ...] = tuple(normalized)
        self._starts: Tuple[float, ...] = tuple(
            segment.interval_sim.start_sim for segment in normalized
        )
        self._frozen = True

    @classmethod
    def tariff_yuan_per_mwh(cls, segments, version="tariff-1.0"):
        return cls(
            segments,
            value_name="tariff_yuan_per_mwh",
            non_negative=False,
            version=version,
        )

    @classmethod
    def green_power_mw(cls, segments, version="green-1.0"):
        return cls(
            segments,
            value_name="green_power_mw",
            non_negative=True,
            version=version,
        )

    @property
    def segments(self) -> Tuple[ForecastSegment, ...]:
        return self._segments

    def value_at(self, time_sim: float) -> float:
        time_value = finite_number("time_sim", time_sim)
        index = bisect_right(self._starts, time_value) - 1
        if index >= 0:
            segment = self._segments[index]
            if segment.interval_sim.contains(time_value):
                return segment.value
        raise ForecastCoverageError(
            f"{self.value_name} does not cover simulation time {time_value}"
        )

    def boundaries(self, interval: TimeInterval) -> Tuple[float, ...]:
        boundaries = {interval.start_sim, interval.end_sim}
        index = max(0, bisect_right(self._starts, interval.start_sim) - 1)
        for segment in self._segments[index:]:
            if segment.interval_sim.start_sim >= interval.end_sim:
                break
            if segment.interval_sim.overlaps(interval):
                boundaries.add(max(interval.start_sim, segment.interval_sim.start_sim))
                boundaries.add(min(interval.end_sim, segment.interval_sim.end_sim))
        ordered = tuple(sorted(boundaries))
        for left, right in zip(ordered[:-1], ordered[1:]):
            if right > left:
                self.value_at(left + (right - left) / 2.0)
        return ordered
