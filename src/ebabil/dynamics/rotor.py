"""Rotor hızını ve üretilen thrust/reaction torque'u modelle."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ebabil.config import RotorConfig

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RotorOutput:
    """Bir rotor-bank zaman adımının fiziksel çıktıları."""

    requested_thrusts_n: FloatArray
    clipped_thrusts_n: FloatArray
    rotor_speeds_rad_s: FloatArray
    actual_thrusts_n: FloatArray
    reaction_torques_nm: FloatArray


class RotorDynamics:
    """Dört rotor için exact-discretized birinci dereceden hız dinamiği."""

    def __init__(self, config: RotorConfig) -> None:
        config.validate()
        self.config = config
        self._rotor_speeds_rad_s = np.zeros(config.count, dtype=np.float64)
        self._reaction_signs = np.asarray(
            config.reaction_torque_signs, dtype=np.float64
        )

    @property
    def rotor_speeds_rad_s(self) -> FloatArray:
        return self._rotor_speeds_rad_s.copy()

    def reset(self, initial_thrusts_n: npt.ArrayLike | None = None) -> None:
        """Rotor state'ini sıfırla veya belirtilen steady-state thrust'a kur."""

        if initial_thrusts_n is None:
            self._rotor_speeds_rad_s.fill(0.0)
            return
        thrusts = self._validate_thrust_vector(initial_thrusts_n)
        thrusts = np.clip(thrusts, 0.0, self.config.max_thrust_n)
        self._rotor_speeds_rad_s[:] = np.sqrt(thrusts / self.config.thrust_coefficient)

    def step(self, desired_thrusts_n: npt.ArrayLike, dt_s: float) -> RotorOutput:
        """Thrust hedeflerini uygula ve bir zaman adımı rotor dinamiğini ilerlet."""

        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError(f"dt_s pozitif ve sonlu olmalı, alınan: {dt_s}")
        requested = self._validate_thrust_vector(desired_thrusts_n)
        clipped = np.clip(requested, 0.0, self.config.max_thrust_n)
        target_speeds = np.sqrt(clipped / self.config.thrust_coefficient)

        accelerating = target_speeds >= self._rotor_speeds_rad_s
        time_constants = np.where(
            accelerating,
            self.config.time_constant_up_s,
            self.config.time_constant_down_s,
        )
        alpha = 1.0 - np.exp(-dt_s / time_constants)
        self._rotor_speeds_rad_s += alpha * (target_speeds - self._rotor_speeds_rad_s)
        self._rotor_speeds_rad_s[:] = np.clip(
            self._rotor_speeds_rad_s, 0.0, self.config.max_speed_rad_s
        )

        actual_thrusts = self.config.thrust_coefficient * self._rotor_speeds_rad_s**2
        reaction_torques = (
            self._reaction_signs * self.config.yaw_moment_ratio_m * actual_thrusts
        )
        return RotorOutput(
            requested_thrusts_n=requested.copy(),
            clipped_thrusts_n=clipped.copy(),
            rotor_speeds_rad_s=self._rotor_speeds_rad_s.copy(),
            actual_thrusts_n=actual_thrusts,
            reaction_torques_nm=reaction_torques,
        )

    def _validate_thrust_vector(self, thrusts_n: npt.ArrayLike) -> FloatArray:
        thrusts = np.asarray(thrusts_n, dtype=np.float64)
        expected_shape = (self.config.count,)
        if thrusts.shape != expected_shape:
            raise ValueError(
                f"Rotor thrust shape {expected_shape} olmalı, alınan: {thrusts.shape}"
            )
        if not np.all(np.isfinite(thrusts)):
            raise ValueError("Rotor thrust değerleri sonlu olmalı")
        return thrusts
