"""MuJoCo sensörleri ile rangefinder'ı tek observation sözleşmesinde birleştir."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt

from ebabil.sensors.config import SensorSuiteConfig
from ebabil.sensors.noise import SampledSensor, SensorOutput
from ebabil.sensors.rangefinder import PlanarRangefinder, RangefinderReading

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class IdealSensorReading:
    position_world_m: FloatArray
    orientation_wxyz: FloatArray
    linear_velocity_body_m_s: FloatArray
    angular_velocity_body_rad_s: FloatArray
    specific_force_body_m_s2: FloatArray
    rangefinder: RangefinderReading


@dataclass(frozen=True)
class NavigationSensorReading:
    ideal: IdealSensorReading
    rangefinder_m: FloatArray
    rangefinder_normalized: FloatArray
    rangefinder_sampled: bool
    rangefinder_dropped_out: bool
    rangefinder_measurement_time_s: float


def read_ideal_sensors(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rangefinder: PlanarRangefinder,
) -> IdealSensorReading:
    """MJCF isim sözleşmesine göre ideal sensör paketini oku."""

    return IdealSensorReading(
        position_world_m=data.sensor("position").data.copy(),
        orientation_wxyz=data.sensor("orientation").data.copy(),
        linear_velocity_body_m_s=data.sensor("linear_velocity_body").data.copy(),
        angular_velocity_body_rad_s=data.sensor("angular_velocity_body").data.copy(),
        specific_force_body_m_s2=data.sensor("specific_force_body").data.copy(),
        rangefinder=rangefinder.read(model, data),
    )


class NavigationSensorSuite:
    """İdeal state/IMU ile bozulmuş rangefinder'ı Gym ortamına hazırlar."""

    def __init__(self, config: SensorSuiteConfig) -> None:
        config.validate()
        self.config = config
        self.rangefinder = PlanarRangefinder(config.rangefinder)
        self.range_imperfections = SampledSensor(
            config.imperfections,
            lower_bound=config.rangefinder.min_range_m,
            upper_bound=config.rangefinder.max_range_m,
        )

    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        seed: int | None = None,
    ) -> NavigationSensorReading:
        ideal = read_ideal_sensors(model, data, self.rangefinder)
        range_output = self.range_imperfections.reset(
            ideal.rangefinder.distances_m, seed=seed
        )
        return self._combine(ideal, range_output)

    def update(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dt_s: float,
    ) -> NavigationSensorReading:
        ideal = read_ideal_sensors(model, data, self.rangefinder)
        range_output = self.range_imperfections.update(
            ideal.rangefinder.distances_m, dt_s
        )
        return self._combine(ideal, range_output)

    def _combine(
        self, ideal: IdealSensorReading, range_output: SensorOutput
    ) -> NavigationSensorReading:
        normalized = np.asarray(range_output.value, dtype=np.float64) / (
            self.config.rangefinder.max_range_m
        )
        return NavigationSensorReading(
            ideal=ideal,
            rangefinder_m=range_output.value,
            rangefinder_normalized=normalized,
            rangefinder_sampled=range_output.sampled,
            rangefinder_dropped_out=range_output.dropped_out,
            rangefinder_measurement_time_s=range_output.measurement_time_s,
        )
