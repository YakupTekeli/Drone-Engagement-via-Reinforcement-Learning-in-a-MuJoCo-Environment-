"""Sadeleştirilmiş gövde sürüklemesi ve rüzgâr etkileşimi."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt

from ebabil.config import VehicleConfig

FloatArray = npt.NDArray[np.float64]


def _vector3(value: npt.ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} shape (3,) olmalı, alınan: {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} sonlu değerler içermeli")
    return vector


@dataclass(frozen=True)
class DragOutput:
    """Bir fizik adımında hesaplanan hız ve sürükleme kuvvetleri."""

    velocity_body_m_s: FloatArray
    wind_body_m_s: FloatArray
    relative_air_velocity_body_m_s: FloatArray
    force_body_n: FloatArray
    force_world_n: FloatArray


class QuadraticBodyDrag:
    """Her gövde ekseninde `-c * |v_air| * v_air` sürüklemesi uygula."""

    def __init__(self, coefficients_n_s2_m2: npt.ArrayLike) -> None:
        coefficients = _vector3(coefficients_n_s2_m2, "coefficients_n_s2_m2")
        if np.any(coefficients < 0.0):
            raise ValueError("Sürükleme katsayıları negatif olamaz")
        self.coefficients_n_s2_m2 = coefficients.copy()

    @classmethod
    def from_config(cls, config: VehicleConfig) -> QuadraticBodyDrag:
        return cls(config.aerodynamics.quadratic_drag_coefficients_n_s2_m2)

    def force_body(
        self,
        velocity_body_m_s: npt.ArrayLike,
        wind_body_m_s: npt.ArrayLike,
    ) -> FloatArray:
        """Gövdeye göre hava hızından gövde eksenli sürükleme kuvveti hesapla."""

        velocity = _vector3(velocity_body_m_s, "velocity_body_m_s")
        wind = _vector3(wind_body_m_s, "wind_body_m_s")
        relative_air_velocity = velocity - wind
        return (
            -self.coefficients_n_s2_m2
            * np.abs(relative_air_velocity)
            * relative_air_velocity
        )

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        wind_world_m_s: npt.ArrayLike,
        body_name: str = "quadrotor",
    ) -> DragOutput:
        """Sürüklemeyi `xfrc_applied` üzerine ekle; mevcut wrench'i silmez."""

        wind_world = _vector3(wind_world_m_s, "wind_world_m_s")
        body_id = model.body(body_name).id
        velocity_six = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity_six,
            1,
        )
        velocity_body = velocity_six[3:].copy()
        rotation_body_to_world = data.xmat[body_id].reshape(3, 3)
        wind_body = rotation_body_to_world.T @ wind_world
        relative_air_velocity = velocity_body - wind_body
        force_body = (
            -self.coefficients_n_s2_m2
            * np.abs(relative_air_velocity)
            * relative_air_velocity
        )
        force_world = rotation_body_to_world @ force_body
        data.xfrc_applied[body_id, :3] += force_world
        return DragOutput(
            velocity_body_m_s=velocity_body,
            wind_body_m_s=wind_body,
            relative_air_velocity_body_m_s=relative_air_velocity,
            force_body_n=force_body,
            force_world_n=force_world,
        )
