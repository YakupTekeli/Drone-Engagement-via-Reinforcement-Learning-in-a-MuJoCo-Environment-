"""Headless kapalı çevrim uçuş senaryoları ve ölçüm yardımcıları."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt

from ebabil.config import load_vehicle_config
from ebabil.controllers import (
    CascadedFlightController,
    FlightTarget,
    load_controller_config,
    read_flight_state,
)
from ebabil.dynamics.aerodynamics import QuadraticBodyDrag
from ebabil.dynamics.propulsion import QuadrotorPropulsion
from ebabil.models import load_quadrotor

FloatArray = npt.NDArray[np.float64]


def _vector3(value: npt.ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} shape (3,) olmalı, alınan: {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} sonlu değerler içermeli")
    return vector


def quaternion_wxyz_from_rpy(
    roll_rad: float, pitch_rad: float, yaw_rad: float
) -> FloatArray:
    """İçsel XYZ (roll-pitch-yaw) açılarını MuJoCo wxyz quaternion'una çevir."""

    angles = np.asarray([roll_rad, pitch_rad, yaw_rad], dtype=np.float64)
    if not np.all(np.isfinite(angles)):
        raise ValueError("Başlangıç Euler açıları sonlu olmalı")
    cr, cp, cy = np.cos(angles / 2.0)
    sr, sp, sy = np.sin(angles / 2.0)
    quaternion = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    return quaternion / np.linalg.norm(quaternion)


def rotation_angle_rad(rotation_error: npt.ArrayLike) -> float:
    """Bir relative rotation matrisinin jeodezik açısını döndür."""

    rotation = np.asarray(rotation_error, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation_body_to_world shape (3, 3) olmalı")
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.acos(float(cosine)))


def _yaw_rotation(yaw_rad: float) -> FloatArray:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class ClosedLoopResult:
    """Kapalı çevrim koşusunun temel izleri ve özet ölçümleri."""

    time_s: FloatArray
    qpos: FloatArray
    qvel: FloatArray
    positions_world_m: FloatArray
    velocities_world_m_s: FloatArray
    attitude_angles_rad: FloatArray
    target_position_world_m: FloatArray
    target_yaw_rad: float
    mixer_saturated: npt.NDArray[np.bool_]
    torque_saturated: npt.NDArray[np.bool_]
    collective_thrust_n: FloatArray
    body_torque_nm: FloatArray
    rotor_thrusts_n: FloatArray

    @property
    def final_position_error_m(self) -> float:
        return float(
            np.linalg.norm(self.target_position_world_m - self.positions_world_m[-1])
        )

    @property
    def final_speed_m_s(self) -> float:
        return float(np.linalg.norm(self.velocities_world_m_s[-1]))

    @property
    def final_attitude_error_rad(self) -> float:
        return float(self.attitude_angles_rad[-1])

    @property
    def mixer_saturation_fraction(self) -> float:
        return float(np.mean(self.mixer_saturated))

    @property
    def torque_saturation_fraction(self) -> float:
        return float(np.mean(self.torque_saturated))


def run_hover_simulation(
    *,
    duration_s: float = 5.0,
    initial_position_world_m: npt.ArrayLike = (0.0, 0.0, 1.0),
    initial_rpy_rad: npt.ArrayLike = (0.0, 0.0, 0.0),
    target_position_world_m: npt.ArrayLike = (0.0, 0.0, 1.0),
    target_yaw_rad: float = 0.0,
    wind_world_m_s: npt.ArrayLike = (0.0, 0.0, 0.0),
) -> ClosedLoopResult:
    """Kademeli kontrolcüyle deterministik, headless bir hover senaryosu çalıştır."""

    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s pozitif ve sonlu olmalı")
    initial_position = _vector3(initial_position_world_m, "initial_position_world_m")
    initial_rpy = _vector3(initial_rpy_rad, "initial_rpy_rad")
    target_position = _vector3(target_position_world_m, "target_position_world_m")
    wind_world = _vector3(wind_world_m_s, "wind_world_m_s")
    if not math.isfinite(target_yaw_rad):
        raise ValueError("target_yaw_rad sonlu olmalı")

    vehicle_config = load_vehicle_config()
    controller_config = load_controller_config()
    model = load_quadrotor()
    data = mujoco.MjData(model)
    data.qpos[:3] = initial_position
    data.qpos[3:7] = quaternion_wxyz_from_rpy(*initial_rpy)
    mujoco.mj_forward(model, data)

    controller = CascadedFlightController(controller_config, vehicle_config)
    propulsion = QuadrotorPropulsion(vehicle_config)
    drag = QuadraticBodyDrag.from_config(vehicle_config)
    controller.reset()
    propulsion.reset_to_hover()
    target = FlightTarget(
        position_world_m=target_position.copy(),
        yaw_world_rad=target_yaw_rad,
    )

    dt_s = float(model.opt.timestep)
    step_count = int(math.ceil(duration_s / dt_s))
    time_s = np.arange(step_count + 1, dtype=np.float64) * dt_s
    positions = np.empty((step_count + 1, 3), dtype=np.float64)
    velocities = np.empty((step_count + 1, 3), dtype=np.float64)
    qpos = np.empty((step_count + 1, model.nq), dtype=np.float64)
    qvel = np.empty((step_count + 1, model.nv), dtype=np.float64)
    attitude_angles = np.empty(step_count + 1, dtype=np.float64)
    mixer_saturated = np.empty(step_count, dtype=np.bool_)
    torque_saturated = np.empty(step_count, dtype=np.bool_)
    collective_thrust = np.empty(step_count, dtype=np.float64)
    body_torque = np.empty((step_count, 3), dtype=np.float64)
    rotor_thrusts = np.empty((step_count, vehicle_config.rotor.count), dtype=np.float64)

    initial_state = read_flight_state(model, data)
    target_rotation = _yaw_rotation(target_yaw_rad)
    qpos[0] = data.qpos
    qvel[0] = data.qvel
    positions[0] = initial_state.position_world_m
    velocities[0] = initial_state.velocity_world_m_s
    attitude_angles[0] = rotation_angle_rad(
        target_rotation.T @ initial_state.rotation_body_to_world
    )

    for index in range(step_count):
        data.xfrc_applied[:] = 0.0
        state = read_flight_state(model, data)
        control = controller.compute(state, target, dt_s)
        propulsion_output = propulsion.step(
            data,
            control.collective_thrust_n,
            control.body_torque_nm,
            dt_s,
        )
        drag.apply(model, data, wind_world)
        mujoco.mj_step(model, data)

        next_state = read_flight_state(model, data)
        qpos[index + 1] = data.qpos
        qvel[index + 1] = data.qvel
        positions[index + 1] = next_state.position_world_m
        velocities[index + 1] = next_state.velocity_world_m_s
        attitude_angles[index + 1] = rotation_angle_rad(
            target_rotation.T @ next_state.rotation_body_to_world
        )
        mixer_saturated[index] = propulsion_output.mixer.saturated
        torque_saturated[index] = control.body_rate.torque_saturated
        collective_thrust[index] = control.collective_thrust_n
        body_torque[index] = control.body_torque_nm
        rotor_thrusts[index] = propulsion_output.rotors.actual_thrusts_n

    return ClosedLoopResult(
        time_s=time_s,
        qpos=qpos,
        qvel=qvel,
        positions_world_m=positions,
        velocities_world_m_s=velocities,
        attitude_angles_rad=attitude_angles,
        target_position_world_m=target_position.copy(),
        target_yaw_rad=target_yaw_rad,
        mixer_saturated=mixer_saturated,
        torque_saturated=torque_saturated,
        collective_thrust_n=collective_thrust,
        body_torque_nm=body_torque,
        rotor_thrusts_n=rotor_thrusts,
    )
