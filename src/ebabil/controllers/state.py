"""Kontrolcü için simulator-bağımsız state ve target tipleri."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _default_zero3() -> FloatArray:
    return np.zeros(3, dtype=np.float64)


@dataclass(frozen=True)
class FlightState:
    position_world_m: FloatArray
    velocity_world_m_s: FloatArray
    rotation_body_to_world: FloatArray
    angular_velocity_body_rad_s: FloatArray


@dataclass(frozen=True)
class FlightTarget:
    position_world_m: FloatArray
    velocity_world_m_s: FloatArray = field(default_factory=_default_zero3)
    acceleration_world_m_s2: FloatArray = field(default_factory=_default_zero3)
    yaw_world_rad: float = 0.0
    yaw_rate_rad_s: float = 0.0


def read_flight_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str = "quadrotor",
) -> FlightState:
    """MuJoCo state'ini kontrolcü sözleşmesine dönüştür."""

    body_id = model.body(body_name).id
    velocity_world = np.zeros(6, dtype=np.float64)
    velocity_body = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        velocity_world,
        0,
    )
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        velocity_body,
        1,
    )
    return FlightState(
        position_world_m=data.xpos[body_id].copy(),
        velocity_world_m_s=velocity_world[3:].copy(),
        rotation_body_to_world=data.xmat[body_id].reshape(3, 3).copy(),
        angular_velocity_body_rad_s=velocity_body[:3].copy(),
    )
