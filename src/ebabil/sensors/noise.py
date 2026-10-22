"""Sample-and-hold, bias, noise, delay ve dropout sensör modeli."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ebabil.sensors.config import SensorImperfectionConfig

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SensorOutput:
    value: FloatArray
    sampled: bool
    dropped_out: bool
    measurement_time_s: float


class SampledSensor:
    """Sabit periyotlu sensörü deterministik episode seed'iyle taklit eder."""

    def __init__(
        self,
        config: SensorImperfectionConfig,
        *,
        lower_bound: float = -math.inf,
        upper_bound: float = math.inf,
    ) -> None:
        config.validate()
        if lower_bound > upper_bound:
            raise ValueError("lower_bound upper_bound'dan büyük olamaz")
        self.config = config
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._rng = np.random.default_rng()
        self._bias = np.empty(0, dtype=np.float64)
        self._history: deque[tuple[float, FloatArray]] = deque()
        self._time_s = 0.0
        self._next_sample_time_s = config.sample_period_s

    def reset(
        self, initial_value: npt.ArrayLike, *, seed: int | None = None
    ) -> SensorOutput:
        value = self._validated_value(initial_value)
        self._rng = np.random.default_rng(seed)
        self._bias = self._rng.normal(0.0, self.config.bias_std, size=value.shape)
        initial_measurement = self._corrupt(value)
        self._history.clear()
        self._history.append((0.0, initial_measurement))
        self._time_s = 0.0
        self._next_sample_time_s = self.config.sample_period_s
        return SensorOutput(initial_measurement.copy(), True, False, 0.0)

    def update(self, true_value: npt.ArrayLike, dt_s: float) -> SensorOutput:
        value = self._validated_value(true_value)
        if self._bias.shape != value.shape or not self._history:
            raise RuntimeError("SampledSensor.update öncesinde reset çağrılmalı")
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s pozitif ve sonlu olmalı")

        self._time_s += dt_s
        sampled = False
        dropped_out = False
        epsilon = 1e-12
        while self._time_s + epsilon >= self._next_sample_time_s:
            sampled = True
            if self._rng.random() < self.config.dropout_probability:
                dropped_out = True
            else:
                self._history.append((self._next_sample_time_s, self._corrupt(value)))
            self._next_sample_time_s += self.config.sample_period_s

        cutoff_time = self._time_s - self.config.delay_s
        selected_time, selected_value = self._history[0]
        for measurement_time, measurement in self._history:
            if measurement_time <= cutoff_time + epsilon:
                selected_time, selected_value = measurement_time, measurement
            else:
                break
        while len(self._history) > 1 and self._history[1][0] <= cutoff_time + epsilon:
            self._history.popleft()
        return SensorOutput(
            value=selected_value.copy(),
            sampled=sampled,
            dropped_out=dropped_out,
            measurement_time_s=selected_time,
        )

    def _corrupt(self, value: FloatArray) -> FloatArray:
        noise = self._rng.normal(0.0, self.config.noise_std, size=value.shape)
        return np.clip(value + self._bias + noise, self.lower_bound, self.upper_bound)

    @staticmethod
    def _validated_value(value: npt.ArrayLike) -> FloatArray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            raise ValueError("Sensör girdisi sonlu, tek boyutlu bir dizi olmalı")
        return array
