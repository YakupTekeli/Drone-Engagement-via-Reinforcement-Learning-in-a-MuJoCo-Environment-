"""Position, attitude ve body-rate katmanlarından oluşan uçuş kontrolcüsü."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ebabil.config import VehicleConfig
from ebabil.controllers.config import ControllerConfig
from ebabil.controllers.state import FlightState, FlightTarget

FloatArray = npt.NDArray[np.float64]


def _vector3(value: npt.ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} shape (3,) olmalı, alınan: {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} sonlu değerler içermeli")
    return vector


def _vee(skew_symmetric: FloatArray) -> FloatArray:
    return np.array(
        [
            skew_symmetric[2, 1],
            skew_symmetric[0, 2],
            skew_symmetric[1, 0],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class PositionOutput:
    position_error_world_m: FloatArray
    velocity_error_world_m_s: FloatArray
    acceleration_command_world_m_s2: FloatArray
    force_command_world_n: FloatArray
    desired_rotation_body_to_world: FloatArray
    collective_thrust_n: float
    collective_saturated: bool


@dataclass(frozen=True)
class AttitudeOutput:
    rotation_error: FloatArray
    desired_body_rate_rad_s: FloatArray
    rate_saturated: bool


@dataclass(frozen=True)
class BodyRateOutput:
    rate_error_rad_s: FloatArray
    integral_error_rad: FloatArray
    unconstrained_torque_nm: FloatArray
    applied_torque_nm: FloatArray
    torque_saturated: bool


@dataclass(frozen=True)
class CascadedControlOutput:
    collective_thrust_n: float
    body_torque_nm: FloatArray
    position: PositionOutput
    attitude: AttitudeOutput
    body_rate: BodyRateOutput


class PositionVelocityController:
    """Dünya-frame position/velocity hatasını force ve desired attitude'a çevir."""

    def __init__(
        self,
        controller_config: ControllerConfig,
        vehicle_config: VehicleConfig,
    ) -> None:
        self.config = controller_config.position
        self.mass_kg = vehicle_config.mass_kg
        self.gravity_m_s2 = vehicle_config.gravity_m_s2
        self.max_collective_thrust_n = (
            vehicle_config.rotor.count * vehicle_config.rotor.max_thrust_n
        )
        self.kp = np.asarray(self.config.kp, dtype=np.float64)
        self.kd = np.asarray(self.config.kd, dtype=np.float64)

    def compute(self, state: FlightState, target: FlightTarget) -> PositionOutput:
        position_error = _vector3(
            target.position_world_m - state.position_world_m,
            "position_error_world_m",
        )
        velocity_error = _vector3(
            target.velocity_world_m_s - state.velocity_world_m_s,
            "velocity_error_world_m_s",
        )
        acceleration = (
            _vector3(target.acceleration_world_m_s2, "target.acceleration_world_m_s2")
            + self.kp * position_error
            + self.kd * velocity_error
        )

        horizontal_norm = float(np.linalg.norm(acceleration[:2]))
        max_horizontal = self.config.max_horizontal_acceleration_m_s2
        if horizontal_norm > max_horizontal:
            acceleration[:2] *= max_horizontal / horizontal_norm
        acceleration[2] = np.clip(
            acceleration[2],
            -self.config.max_vertical_acceleration_m_s2,
            self.config.max_vertical_acceleration_m_s2,
        )

        specific_force = acceleration + np.array(
            [0.0, 0.0, self.gravity_m_s2], dtype=np.float64
        )
        force_world = self.mass_kg * specific_force
        force_world[2] = max(force_world[2], 1e-6)

        max_horizontal_force = force_world[2] * math.tan(self.config.max_tilt_rad)
        horizontal_force_norm = float(np.linalg.norm(force_world[:2]))
        if horizontal_force_norm > max_horizontal_force:
            force_world[:2] *= max_horizontal_force / horizontal_force_norm

        desired_rotation = self._desired_rotation(force_world, target.yaw_world_rad)
        current_body_z_world = state.rotation_body_to_world[:, 2]
        unconstrained_collective = float(force_world @ current_body_z_world)
        collective = float(
            np.clip(unconstrained_collective, 0.0, self.max_collective_thrust_n)
        )
        return PositionOutput(
            position_error_world_m=position_error,
            velocity_error_world_m_s=velocity_error,
            acceleration_command_world_m_s2=acceleration,
            force_command_world_n=force_world,
            desired_rotation_body_to_world=desired_rotation,
            collective_thrust_n=collective,
            collective_saturated=not math.isclose(
                unconstrained_collective, collective, rel_tol=0.0, abs_tol=1e-12
            ),
        )

    @staticmethod
    def _desired_rotation(force_world: FloatArray, yaw_world_rad: float) -> FloatArray:
        if not math.isfinite(yaw_world_rad):
            raise ValueError("yaw_world_rad sonlu olmalı")
        force_norm = float(np.linalg.norm(force_world))
        if force_norm <= 0.0:
            raise ValueError("Desired force sıfır olamaz")
        body_z_desired = force_world / force_norm
        heading = np.array(
            [math.cos(yaw_world_rad), math.sin(yaw_world_rad), 0.0],
            dtype=np.float64,
        )
        body_y_desired = np.cross(body_z_desired, heading)
        body_y_norm = float(np.linalg.norm(body_y_desired))
        if body_y_norm < 1e-9:
            heading = np.array(
                [-math.sin(yaw_world_rad), math.cos(yaw_world_rad), 0.0],
                dtype=np.float64,
            )
            body_y_desired = np.cross(body_z_desired, heading)
            body_y_norm = float(np.linalg.norm(body_y_desired))
        body_y_desired /= body_y_norm
        body_x_desired = np.cross(body_y_desired, body_z_desired)
        return np.column_stack((body_x_desired, body_y_desired, body_z_desired))


class AttitudeController:
    """SO(3) rotation hatasını body-rate hedeflerine dönüştür."""

    def __init__(self, controller_config: ControllerConfig) -> None:
        self.kp = np.asarray(controller_config.attitude.kp, dtype=np.float64)
        self.max_body_rate = np.asarray(
            controller_config.attitude.max_body_rate_rad_s, dtype=np.float64
        )

    def compute(
        self,
        current_rotation_body_to_world: npt.ArrayLike,
        desired_rotation_body_to_world: npt.ArrayLike,
        desired_yaw_rate_rad_s: float = 0.0,
    ) -> AttitudeOutput:
        current = np.asarray(current_rotation_body_to_world, dtype=np.float64)
        desired = np.asarray(desired_rotation_body_to_world, dtype=np.float64)
        if current.shape != (3, 3) or desired.shape != (3, 3):
            raise ValueError("Rotation matrisleri shape (3, 3) olmalı")
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(desired)):
            raise ValueError("Rotation matrisleri sonlu olmalı")
        if not math.isfinite(desired_yaw_rate_rad_s):
            raise ValueError("desired_yaw_rate_rad_s sonlu olmalı")

        error_matrix = 0.5 * (desired.T @ current - current.T @ desired)
        rotation_error = _vee(error_matrix)
        unconstrained_rate = -self.kp * rotation_error
        unconstrained_rate[2] += desired_yaw_rate_rad_s
        desired_rate = np.clip(
            unconstrained_rate, -self.max_body_rate, self.max_body_rate
        )
        return AttitudeOutput(
            rotation_error=rotation_error,
            desired_body_rate_rad_s=desired_rate,
            rate_saturated=not np.allclose(
                unconstrained_rate, desired_rate, rtol=0.0, atol=1e-12
            ),
        )


class BodyRateController:
    """Body-rate hatasını torque'a dönüştüren PI kontrolcü ve anti-windup."""

    def __init__(
        self,
        controller_config: ControllerConfig,
        vehicle_config: VehicleConfig,
    ) -> None:
        self.kp = np.asarray(controller_config.body_rate.kp, dtype=np.float64)
        self.ki = np.asarray(controller_config.body_rate.ki, dtype=np.float64)
        self.integral_limit = np.asarray(
            controller_config.body_rate.integral_limit_rad, dtype=np.float64
        )
        self.max_torque = np.asarray(
            controller_config.body_rate.max_torque_nm, dtype=np.float64
        )
        self.inertia = np.asarray(
            vehicle_config.inertia_diagonal_kg_m2, dtype=np.float64
        )
        self.integral_error = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        self.integral_error.fill(0.0)

    def compute(
        self,
        desired_body_rate_rad_s: npt.ArrayLike,
        current_body_rate_rad_s: npt.ArrayLike,
        dt_s: float,
    ) -> BodyRateOutput:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s pozitif ve sonlu olmalı")
        desired = _vector3(desired_body_rate_rad_s, "desired_body_rate_rad_s")
        current = _vector3(current_body_rate_rad_s, "current_body_rate_rad_s")
        rate_error = desired - current
        candidate_integral = np.clip(
            self.integral_error + rate_error * dt_s,
            -self.integral_limit,
            self.integral_limit,
        )
        angular_acceleration = self.kp * rate_error + self.ki * candidate_integral
        gyroscopic_compensation = np.cross(current, self.inertia * current)
        unconstrained_torque = (
            self.inertia * angular_acceleration + gyroscopic_compensation
        )
        applied_torque = np.clip(
            unconstrained_torque, -self.max_torque, self.max_torque
        )
        saturated_axes = ~np.isclose(
            unconstrained_torque, applied_torque, rtol=0.0, atol=1e-12
        )
        self.integral_error[:] = np.where(
            saturated_axes, self.integral_error, candidate_integral
        )
        return BodyRateOutput(
            rate_error_rad_s=rate_error,
            integral_error_rad=self.integral_error.copy(),
            unconstrained_torque_nm=unconstrained_torque,
            applied_torque_nm=applied_torque,
            torque_saturated=bool(np.any(saturated_axes)),
        )


class CascadedFlightController:
    """Position → attitude → body-rate kontrol zinciri."""

    def __init__(
        self,
        controller_config: ControllerConfig,
        vehicle_config: VehicleConfig,
    ) -> None:
        controller_config.validate()
        vehicle_config.validate()
        self.position = PositionVelocityController(controller_config, vehicle_config)
        self.attitude = AttitudeController(controller_config)
        self.body_rate = BodyRateController(controller_config, vehicle_config)

    def reset(self) -> None:
        self.body_rate.reset()

    def compute(
        self,
        state: FlightState,
        target: FlightTarget,
        dt_s: float,
    ) -> CascadedControlOutput:
        position_output = self.position.compute(state, target)
        attitude_output = self.attitude.compute(
            state.rotation_body_to_world,
            position_output.desired_rotation_body_to_world,
            target.yaw_rate_rad_s,
        )
        body_rate_output = self.body_rate.compute(
            attitude_output.desired_body_rate_rad_s,
            state.angular_velocity_body_rad_s,
            dt_s,
        )
        return CascadedControlOutput(
            collective_thrust_n=position_output.collective_thrust_n,
            body_torque_nm=body_rate_output.applied_torque_nm,
            position=position_output,
            attitude=attitude_output,
            body_rate=body_rate_output,
        )
