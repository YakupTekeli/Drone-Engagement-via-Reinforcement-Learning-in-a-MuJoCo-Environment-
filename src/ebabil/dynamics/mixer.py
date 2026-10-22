"""X-konfigürasyonlu quadrotor için kuvvet/moment dağıtımı."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ebabil.config import VehicleConfig

FloatArray = npt.NDArray[np.float64]
ROTOR_ORDER = (
    "front_left",
    "front_right",
    "rear_right",
    "rear_left",
)


@dataclass(frozen=True)
class MixerResult:
    """Mixer isteği, uygulanabilir rotor thrust'ları ve ulaşılan wrench."""

    requested_wrench: FloatArray
    unconstrained_thrusts: FloatArray
    applied_thrusts: FloatArray
    achieved_wrench: FloatArray
    saturated: bool


class XQuadrotorMixer:
    """`[Fz, tau_x, tau_y, tau_z]` wrench'ini rotor thrust'larına dağıt."""

    def __init__(
        self,
        arm_length_m: float,
        yaw_moment_ratio_m: float,
        reaction_torque_signs: tuple[int, int, int, int],
        max_thrust_n: float,
    ) -> None:
        if arm_length_m <= 0.0 or yaw_moment_ratio_m <= 0.0 or max_thrust_n <= 0.0:
            raise ValueError("Mixer fizik parametreleri pozitif olmalı")
        if len(reaction_torque_signs) != 4:
            raise ValueError("Mixer dört reaction torque işareti bekler")

        diagonal = arm_length_m / np.sqrt(2.0)
        self.rotor_positions_body_m = np.array(
            [
                [diagonal, diagonal, 0.0],
                [diagonal, -diagonal, 0.0],
                [-diagonal, -diagonal, 0.0],
                [-diagonal, diagonal, 0.0],
            ],
            dtype=np.float64,
        )
        self.max_thrust_n = float(max_thrust_n)
        self.reaction_torque_signs = np.asarray(reaction_torque_signs, dtype=np.float64)

        x = self.rotor_positions_body_m[:, 0]
        y = self.rotor_positions_body_m[:, 1]
        self.allocation_matrix = np.vstack(
            (
                np.ones(4, dtype=np.float64),
                y,
                -x,
                yaw_moment_ratio_m * self.reaction_torque_signs,
            )
        )
        if np.linalg.matrix_rank(self.allocation_matrix) != 4:
            raise ValueError("Mixer allocation matrix tekil olmamalı")

    @classmethod
    def from_config(cls, config: VehicleConfig) -> XQuadrotorMixer:
        return cls(
            arm_length_m=config.arm_length_m,
            yaw_moment_ratio_m=config.rotor.yaw_moment_ratio_m,
            reaction_torque_signs=config.rotor.reaction_torque_signs,
            max_thrust_n=config.rotor.max_thrust_n,
        )

    def allocate(
        self,
        collective_thrust_n: float,
        body_torque_nm: npt.ArrayLike,
    ) -> MixerResult:
        """İstenen wrench'i rotor thrust'larına çöz ve fiziksel limitlere kırp."""

        torque = np.asarray(body_torque_nm, dtype=np.float64)
        if torque.shape != (3,):
            raise ValueError(
                f"body_torque_nm shape (3,) olmalı, alınan: {torque.shape}"
            )
        wrench = np.concatenate(
            (np.array([collective_thrust_n], dtype=np.float64), torque)
        )
        if not np.all(np.isfinite(wrench)):
            raise ValueError("Mixer girdileri sonlu olmalı")

        unconstrained = np.linalg.solve(self.allocation_matrix, wrench)
        applied = np.clip(unconstrained, 0.0, self.max_thrust_n)
        achieved = self.allocation_matrix @ applied
        saturated = not np.allclose(unconstrained, applied, rtol=0.0, atol=1e-12)
        return MixerResult(
            requested_wrench=wrench,
            unconstrained_thrusts=unconstrained,
            applied_thrusts=applied,
            achieved_wrench=achieved,
            saturated=saturated,
        )
