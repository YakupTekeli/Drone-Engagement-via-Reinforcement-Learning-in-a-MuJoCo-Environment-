"""Mixer, rotor dinamiği ve MuJoCo actuator girdisini birleştir."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy.typing as npt

from ebabil.config import VehicleConfig
from ebabil.dynamics.mixer import MixerResult, XQuadrotorMixer
from ebabil.dynamics.rotor import RotorDynamics, RotorOutput


@dataclass(frozen=True)
class PropulsionOutput:
    """Bir propulsion adımının mixer ve rotor sonuçları."""

    mixer: MixerResult
    rotors: RotorOutput


class QuadrotorPropulsion:
    """İstenen gövde wrench'ini gecikmeli MuJoCo rotor thrust'larına dönüştür."""

    def __init__(self, config: VehicleConfig) -> None:
        config.validate()
        self.config = config
        self.mixer = XQuadrotorMixer.from_config(config)
        self.rotors = RotorDynamics(config.rotor)

    def reset(self, initial_thrusts_n: npt.ArrayLike | None = None) -> None:
        self.rotors.reset(initial_thrusts_n)

    def reset_to_hover(self) -> None:
        hover = self.config.hover_thrust_per_rotor_n
        self.rotors.reset([hover] * self.config.rotor.count)

    def step(
        self,
        data: mujoco.MjData,
        collective_thrust_n: float,
        body_torque_nm: npt.ArrayLike,
        dt_s: float,
    ) -> PropulsionOutput:
        if data.ctrl.shape != (self.config.rotor.count,):
            raise ValueError(
                "MuJoCo actuator sayısı rotor konfigürasyonuyla eşleşmiyor: "
                f"{data.ctrl.shape} != {(self.config.rotor.count,)}"
            )
        mixer_result = self.mixer.allocate(collective_thrust_n, body_torque_nm)
        rotor_output = self.rotors.step(mixer_result.applied_thrusts, dt_s)
        data.ctrl[:] = rotor_output.actual_thrusts_n
        return PropulsionOutput(mixer=mixer_result, rotors=rotor_output)
