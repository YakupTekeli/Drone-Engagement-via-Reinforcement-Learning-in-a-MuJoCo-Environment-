"""MuJoCo tabanlı Gymnasium quadrotor navigasyon ortamı."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import gymnasium as gym
import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from ebabil.config import VehicleConfig, load_vehicle_config
from ebabil.controllers import (
    CascadedFlightController,
    FlightState,
    FlightTarget,
    load_controller_config,
    read_flight_state,
)
from ebabil.dynamics.aerodynamics import QuadraticBodyDrag
from ebabil.dynamics.propulsion import QuadrotorPropulsion
from ebabil.envs.config import NavigationEnvConfig, load_navigation_config
from ebabil.models import load_quadrotor_timestep_s
from ebabil.rewards import compute_navigation_reward
from ebabil.scenes import (
    EndpointMode,
    SceneGeneratorConfig,
    SceneMode,
    StaticScene,
    TargetAreaConfig,
    build_scene_model,
    generate_static_scene,
    load_scene_config,
)
from ebabil.sensors import (
    NavigationSensorReading,
    NavigationSensorSuite,
    load_sensor_config,
)

FloatArray = npt.NDArray[np.float64]
SensorMode = Literal["ideal", "imperfect"]
OBSERVATION_VERSION = "v1"
OBSERVATION_COMPONENTS = (
    "goal_position_body_xyz",
    "goal_distance",
    "linear_velocity_body_xyz",
    "angular_velocity_body_xyz",
    "gravity_direction_body_xyz",
    "altitude",
    "rangefinder_16",
)
OBSERVATION_SIZE = 30


@dataclass(frozen=True)
class VelocityCommand:
    velocity_world_m_s: FloatArray
    yaw_rate_rad_s: float


class StepReward(Protocol):
    """Bileşenleri ayrı loglanabilen tek adım reward sözleşmesi."""

    @property
    def total(self) -> float: ...

    def as_dict(self) -> dict[str, float]: ...


@dataclass(frozen=True)
class TerminalDecision:
    """Bir policy adımının başarı/terminal sebebi ve ek info alanları."""

    success: bool
    termination_reason: str | None
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NavigationPhysicsTrace:
    time_s: FloatArray
    qpos: FloatArray
    qvel: FloatArray
    rotor_thrusts_n: FloatArray
    velocity_command_world_m_s: FloatArray
    yaw_rate_command_rad_s: FloatArray
    mixer_saturated: npt.NDArray[np.bool_]
    torque_saturated: npt.NDArray[np.bool_]


class EbabilNavigationEnv(gym.Env[Any, npt.NDArray[np.float32]]):
    """Yüksek seviye hız komutlarıyla 3B hedefe ulaşma ortamı."""

    metadata = {"render_modes": [], "render_fps": 20}
    ACTION_SIZE = 4
    observation_version = OBSERVATION_VERSION

    def __init__(
        self,
        *,
        scene_mode: SceneMode = "clear",
        endpoint_mode: EndpointMode = "fixed",
        sensor_mode: SensorMode = "ideal",
        config: NavigationEnvConfig | None = None,
        scene_config: SceneGeneratorConfig | None = None,
        max_episode_steps: int | None = None,
        record_physics_trace: bool = False,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode is not None:
            raise ValueError(
                "EbabilNavigationEnv headless çalışır; kayıtları ayrı viewer scriptiyle açın"
            )
        if scene_mode not in ("clear", "single_blocker", "random_obstacles"):
            raise ValueError(f"Bilinmeyen scene_mode: {scene_mode}")
        if endpoint_mode not in (
            "fixed",
            "random_start",
            "random_goal",
            "random_both",
        ):
            raise ValueError(f"Bilinmeyen endpoint_mode: {endpoint_mode}")
        if sensor_mode not in ("ideal", "imperfect"):
            raise ValueError(f"Bilinmeyen sensor_mode: {sensor_mode}")
        self.render_mode = render_mode
        self.record_physics_trace = record_physics_trace
        self.config = config or load_navigation_config()
        self.config.validate()
        self._physics_timestep_s = load_quadrotor_timestep_s()
        self._physics_steps_per_action = self._policy_step_ratio(
            self.config.simulation.policy_frequency_hz,
            self._physics_timestep_s,
        )
        self._goal_hold_required_steps = math.ceil(
            self.config.termination.goal_hold_seconds
            * self.config.simulation.policy_frequency_hz
        )
        self.scene_config = scene_config or load_scene_config()
        self.scene_config.validate()
        self.default_scene_mode = scene_mode
        self.default_endpoint_mode = endpoint_mode
        self.default_sensor_mode = sensor_mode
        configured_steps = math.ceil(
            self.config.simulation.max_episode_seconds
            * self.config.simulation.policy_frequency_hz
        )
        self.max_episode_steps = (
            configured_steps
            if max_episode_steps is None
            else self._validated_episode_step_limit(max_episode_steps)
        )

        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.ACTION_SIZE,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )
        self.vehicle_config: VehicleConfig = load_vehicle_config()
        self.controller_config = load_controller_config()
        self.sensor_config = load_sensor_config()
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.scene: StaticScene | None = None
        self.controller: CascadedFlightController | None = None
        self.propulsion: QuadrotorPropulsion | None = None
        self.drag: QuadraticBodyDrag | None = None
        self.sensors: NavigationSensorSuite | None = None
        self._sensor_reading: NavigationSensorReading | None = None
        self._position_reference = np.zeros(3, dtype=np.float64)
        self._yaw_reference_rad = 0.0
        self._previous_action = np.zeros(self.ACTION_SIZE, dtype=np.float64)
        self._previous_distance_m = 0.0
        self._episode_steps = 0
        self._goal_hold_steps = 0
        self._scene_seed = 0
        self._sensor_seed = 0
        self._current_scene_mode: SceneMode = scene_mode
        self._current_endpoint_mode: EndpointMode = endpoint_mode
        self._current_sensor_mode: SensorMode = sensor_mode
        self._trace_time_s: list[float] = []
        self._trace_qpos: list[FloatArray] = []
        self._trace_qvel: list[FloatArray] = []
        self._trace_rotor_thrusts_n: list[FloatArray] = []
        self._trace_velocity_command_world_m_s: list[FloatArray] = []
        self._trace_yaw_rate_command_rad_s: list[float] = []
        self._trace_mixer_saturated: list[bool] = []
        self._trace_torque_saturated: list[bool] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        scene_mode = options.get("scene_mode", self.default_scene_mode)
        endpoint_mode = options.get("endpoint_mode", self.default_endpoint_mode)
        sensor_mode = options.get("sensor_mode", self.default_sensor_mode)
        if scene_mode not in ("clear", "single_blocker", "random_obstacles"):
            raise ValueError(f"Bilinmeyen scene_mode: {scene_mode}")
        if sensor_mode not in ("ideal", "imperfect"):
            raise ValueError(f"Bilinmeyen sensor_mode: {sensor_mode}")
        if endpoint_mode not in (
            "fixed",
            "random_start",
            "random_goal",
            "random_both",
        ):
            raise ValueError(f"Bilinmeyen endpoint_mode: {endpoint_mode}")
        self._current_scene_mode = scene_mode
        self._current_endpoint_mode = endpoint_mode
        self._current_sensor_mode = sensor_mode
        scene_seed = options.get("scene_seed")
        sensor_seed = options.get("sensor_seed")
        self._scene_seed = (
            self._next_episode_seed()
            if scene_seed is None
            else self._validated_component_seed(scene_seed, "scene_seed")
        )
        self._sensor_seed = (
            self._next_episode_seed()
            if sensor_seed is None
            else self._validated_component_seed(sensor_seed, "sensor_seed")
        )
        self.scene = generate_static_scene(
            self._scene_seed,
            self.scene_config,
            mode=scene_mode,
            endpoint_mode=endpoint_mode,
            target_config=self._scene_target_config(),
        )
        self.model = self._build_scene_model(self.scene)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        physics_timestep_s = float(self.model.opt.timestep)
        if not math.isclose(
            physics_timestep_s,
            self._physics_timestep_s,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError("Sahne modeli ile temel model timestep değerleri farklı")

        self.controller = CascadedFlightController(
            self.controller_config, self.vehicle_config
        )
        self.propulsion = QuadrotorPropulsion(self.vehicle_config)
        self.drag = QuadraticBodyDrag.from_config(self.vehicle_config)
        self.sensors = NavigationSensorSuite(self.sensor_config)
        self.controller.reset()
        self.propulsion.reset_to_hover()
        self._sensor_reading = self.sensors.reset(
            self.model, self.data, seed=self._sensor_seed
        )
        self._after_simulation_reset()

        state = read_flight_state(self.model, self.data)
        self._position_reference[:] = state.position_world_m
        self._yaw_reference_rad = self._yaw_from_rotation(state.rotation_body_to_world)
        self._previous_action.fill(0.0)
        self._previous_distance_m = self._goal_distance(state)
        self._episode_steps = 0
        self._goal_hold_steps = 0
        self._reset_physics_trace()
        observation = self._observation(self._sensor_reading)
        info = self._base_info(state)
        info.update(
            {
                "is_success": False,
                "termination_reason": None,
                "reward_components": {},
                "observation_version": self.observation_version,
            }
        )
        return observation, info

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self._require_reset()
        assert self.model is not None
        assert self.data is not None
        assert self.scene is not None
        assert self.controller is not None
        assert self.propulsion is not None
        assert self.drag is not None
        assert self.sensors is not None
        assert self._sensor_reading is not None

        raw_action = np.asarray(action, dtype=np.float64)
        expected_shape = self.action_space.shape
        if raw_action.shape != expected_shape or not np.all(np.isfinite(raw_action)):
            raise ValueError(f"action sonlu ve shape {expected_shape} olmalı")
        clipped_action = np.clip(raw_action, -1.0, 1.0)
        action_clipped = not np.allclose(raw_action, clipped_action)
        dt_s = float(self.model.opt.timestep)
        termination_reason: str | None = None
        last_control = None
        last_propulsion = None

        for _ in range(self._physics_steps_per_action):
            self.data.xfrc_applied[:] = 0.0
            state = read_flight_state(self.model, self.data)
            command = self._velocity_command(clipped_action, state)
            self._position_reference += command.velocity_world_m_s * dt_s
            margin = self.scene_config.vehicle_clearance_m
            self._position_reference = np.clip(
                self._position_reference,
                self.scene_config.workspace_min_m + margin,
                self.scene_config.workspace_max_m - margin,
            )
            self._yaw_reference_rad = self._wrap_angle(
                self._yaw_reference_rad + command.yaw_rate_rad_s * dt_s
            )
            target = FlightTarget(
                position_world_m=self._position_reference.copy(),
                velocity_world_m_s=command.velocity_world_m_s,
                yaw_world_rad=self._yaw_reference_rad,
                yaw_rate_rad_s=command.yaw_rate_rad_s,
            )
            last_control = self.controller.compute(state, target, dt_s)
            last_propulsion = self.propulsion.step(
                self.data,
                last_control.collective_thrust_n,
                last_control.body_torque_nm,
                dt_s,
            )
            self.drag.apply(self.model, self.data, (0.0, 0.0, 0.0))
            mujoco.mj_step(self.model, self.data)
            self._sensor_reading = self.sensors.update(self.model, self.data, dt_s)
            self._after_physics_step()
            self._append_physics_trace(
                command,
                last_control.body_rate.torque_saturated,
                last_propulsion.rotors.actual_thrusts_n,
                last_propulsion.mixer.saturated,
            )
            termination_reason = self._failure_reason()
            if termination_reason is not None:
                break

        state = read_flight_state(self.model, self.data)
        current_distance_m = self._goal_distance(state)
        decision = self._terminal_decision(
            state, current_distance_m, termination_reason
        )
        termination_reason = decision.termination_reason
        success = decision.success
        terminated = termination_reason is not None
        self._episode_steps += 1
        truncated = self._episode_steps >= self.max_episode_steps and not terminated
        if truncated:
            termination_reason = "time_limit"

        minimum_range_m = float(
            np.min(self._sensor_reading.ideal.rangefinder.distances_m)
        )
        collision = termination_reason in ("obstacle_collision", "ground_collision")
        reward_breakdown = self._compute_step_reward(
            state=state,
            action=clipped_action,
            current_distance_m=current_distance_m,
            minimum_range_m=minimum_range_m,
            success=success,
            collision=collision,
            terminated=terminated,
            truncated=truncated,
            termination_reason=termination_reason,
        )
        self._previous_distance_m = current_distance_m
        self._previous_action[:] = clipped_action
        observation = self._observation(self._sensor_reading)
        info = self._base_info(state)
        info.update(
            {
                "is_success": success,
                "termination_reason": termination_reason,
                "reward_components": reward_breakdown.as_dict(),
                "minimum_range_m": minimum_range_m,
                "action_clipped": action_clipped,
                "velocity_command_world_m_s": command.velocity_world_m_s.copy(),
                "yaw_rate_command_rad_s": command.yaw_rate_rad_s,
                "mixer_saturated": bool(
                    last_propulsion is not None and last_propulsion.mixer.saturated
                ),
                "torque_saturated": bool(
                    last_control is not None and last_control.body_rate.torque_saturated
                ),
                "observation_version": self.observation_version,
            }
        )
        info.update(decision.info)
        return observation, reward_breakdown.total, terminated, truncated, info

    def close(self) -> None:
        self._close_additional_resources()
        self.model = None
        self.data = None
        self.scene = None
        self.controller = None
        self.propulsion = None
        self.drag = None
        self.sensors = None
        self._sensor_reading = None

    def _build_scene_model(self, scene: StaticScene) -> mujoco.MjModel:
        return build_scene_model(scene)

    def _terminal_decision(
        self,
        state: FlightState,
        current_distance_m: float,
        failure_reason: str | None,
    ) -> TerminalDecision:
        """Başarı ve terminal sebebi karar noktası.

        Varsayılan uygulama ADR-008 legacy nokta/hold semantiğidir: hedef
        yarıçapı içinde, hız sınırı altında ve kesintisiz `goal_hold_seconds`.
        Hedef üstü hover görevi olan alt sınıflar bunu geçersiz kılar.
        """

        current_speed_m_s = float(np.linalg.norm(state.velocity_world_m_s))
        goal_conditions_met = (
            failure_reason is None
            and current_distance_m <= self.config.termination.goal_radius_m
            and current_speed_m_s <= self.config.termination.goal_max_speed_m_s
        )
        self._goal_hold_steps = self._goal_hold_steps + 1 if goal_conditions_met else 0
        success = self._goal_hold_steps >= self._goal_hold_required_steps
        return TerminalDecision(
            success=success,
            termination_reason="goal" if success else failure_reason,
        )

    def _compute_step_reward(
        self,
        *,
        state: FlightState,
        action: FloatArray,
        current_distance_m: float,
        minimum_range_m: float,
        success: bool,
        collision: bool,
        terminated: bool,
        truncated: bool,
        termination_reason: str | None,
    ) -> StepReward:
        """Tek policy adımının bileşen bazlı reward'ı.

        Varsayılan uygulama ADR-008 legacy reward'ıdır. Terminal hover görevi
        olan alt sınıflar bunu kendi bileşenleriyle geçersiz kılar.
        """

        del truncated, termination_reason
        other_failure = terminated and not success and not collision
        return compute_navigation_reward(
            self.config.reward,
            previous_distance_m=self._previous_distance_m,
            current_distance_m=current_distance_m,
            minimum_range_m=minimum_range_m,
            current_action=action,
            previous_action=self._previous_action,
            body_z_world_z=float(state.rotation_body_to_world[2, 2]),
            success=success,
            collision=collision,
            other_failure=other_failure,
        )

    def _scene_target_config(self) -> TargetAreaConfig | None:
        """Sahne üretiminde uygulanacak hedef hover açıklığı kısıtı.

        Legacy `EbabilNavigation-v0` sahne dizisi ADR-008 ile dondurulduğu için
        varsayılan `None` döner ve hedef-farkında kısıt uygulanmaz. Hedef üstü
        hover görevi olan alt sınıflar bunu geçersiz kılar.
        """

        return None

    def _after_simulation_reset(self) -> None:
        """Alt sınıfların model-bağımlı kaynakları kurduğu reset hook'u."""

    def _after_physics_step(self) -> None:
        """Alt sınıfların fizik saatiyle sensör örneklediği tick hook'u."""

    def _close_additional_resources(self) -> None:
        """Alt sınıfların renderer gibi kaynakları kapattığı hook."""

    def physics_trace(self) -> NavigationPhysicsTrace:
        """Evaluation için son episode'un fizik çözünürlüklü izini kopyala."""

        if not self.record_physics_trace:
            raise RuntimeError("Fizik izi için record_physics_trace=True gerekli")
        self._require_reset()
        return NavigationPhysicsTrace(
            time_s=np.asarray(self._trace_time_s, dtype=np.float64),
            qpos=np.asarray(self._trace_qpos, dtype=np.float64),
            qvel=np.asarray(self._trace_qvel, dtype=np.float64),
            rotor_thrusts_n=np.asarray(
                self._trace_rotor_thrusts_n, dtype=np.float64
            ).reshape(-1, 4),
            velocity_command_world_m_s=np.asarray(
                self._trace_velocity_command_world_m_s, dtype=np.float64
            ).reshape(-1, 3),
            yaw_rate_command_rad_s=np.asarray(
                self._trace_yaw_rate_command_rad_s, dtype=np.float64
            ),
            mixer_saturated=np.asarray(self._trace_mixer_saturated, dtype=np.bool_),
            torque_saturated=np.asarray(self._trace_torque_saturated, dtype=np.bool_),
        )

    def _reset_physics_trace(self) -> None:
        self._trace_time_s.clear()
        self._trace_qpos.clear()
        self._trace_qvel.clear()
        self._trace_rotor_thrusts_n.clear()
        self._trace_velocity_command_world_m_s.clear()
        self._trace_yaw_rate_command_rad_s.clear()
        self._trace_mixer_saturated.clear()
        self._trace_torque_saturated.clear()
        if self.record_physics_trace:
            assert self.data is not None
            self._trace_time_s.append(float(self.data.time))
            self._trace_qpos.append(self.data.qpos.copy())
            self._trace_qvel.append(self.data.qvel.copy())

    def _append_physics_trace(
        self,
        command: VelocityCommand,
        torque_saturated: bool,
        rotor_thrusts_n: FloatArray,
        mixer_saturated: bool,
    ) -> None:
        if not self.record_physics_trace:
            return
        assert self.data is not None
        self._trace_time_s.append(float(self.data.time))
        self._trace_qpos.append(self.data.qpos.copy())
        self._trace_qvel.append(self.data.qvel.copy())
        self._trace_rotor_thrusts_n.append(rotor_thrusts_n.copy())
        self._trace_velocity_command_world_m_s.append(command.velocity_world_m_s.copy())
        self._trace_yaw_rate_command_rad_s.append(command.yaw_rate_rad_s)
        self._trace_mixer_saturated.append(mixer_saturated)
        self._trace_torque_saturated.append(torque_saturated)

    def _observation(self, reading: NavigationSensorReading) -> Any:
        assert self.scene is not None
        rotation_flat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation_flat, reading.ideal.orientation_wxyz)
        rotation = rotation_flat.reshape(3, 3)
        goal_delta_world = self.scene.goal_world_m - reading.ideal.position_world_m
        goal_delta_body = rotation.T @ goal_delta_world
        distance_m = float(np.linalg.norm(goal_delta_world))
        action_config = self.config.action
        velocity_scale = np.array(
            [
                action_config.max_horizontal_speed_m_s,
                action_config.max_horizontal_speed_m_s,
                action_config.max_vertical_speed_m_s,
            ],
            dtype=np.float64,
        )
        angular_scale = np.asarray(
            self.controller_config.attitude.max_body_rate_rad_s, dtype=np.float64
        )
        gravity_direction_body = rotation.T @ np.array([0.0, 0.0, -1.0])
        altitude_fraction = (
            reading.ideal.position_world_m[2] - self.scene_config.workspace_min_m[2]
        ) / (
            self.scene_config.workspace_max_m[2] - self.scene_config.workspace_min_m[2]
        )
        if self._current_sensor_mode == "imperfect":
            ranges_normalized = reading.rangefinder_normalized
        else:
            ranges_normalized = reading.ideal.rangefinder.normalized_distances
        observation = np.concatenate(
            (
                goal_delta_body / self.config.observation.goal_distance_scale_m,
                [distance_m / self.config.observation.goal_distance_scale_m],
                reading.ideal.linear_velocity_body_m_s / velocity_scale,
                reading.ideal.angular_velocity_body_rad_s / angular_scale,
                gravity_direction_body,
                [2.0 * altitude_fraction - 1.0],
                2.0 * ranges_normalized - 1.0,
            )
        )
        return np.clip(observation, -1.0, 1.0).astype(np.float32)

    def _velocity_command(
        self, action: FloatArray, state: FlightState
    ) -> VelocityCommand:
        yaw = self._yaw_from_rotation(state.rotation_body_to_world)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        horizontal_body = action[:2] * self.config.action.max_horizontal_speed_m_s
        velocity_world = np.array(
            [
                cosine * horizontal_body[0] - sine * horizontal_body[1],
                sine * horizontal_body[0] + cosine * horizontal_body[1],
                action[2] * self.config.action.max_vertical_speed_m_s,
            ],
            dtype=np.float64,
        )
        return VelocityCommand(
            velocity_world_m_s=velocity_world,
            yaw_rate_rad_s=float(action[3] * self.config.action.max_yaw_rate_rad_s),
        )

    def _failure_reason(self) -> str | None:
        assert self.model is not None
        assert self.data is not None
        state = read_flight_state(self.model, self.data)
        arrays = (
            self.data.qpos,
            self.data.qvel,
            state.rotation_body_to_world,
        )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            return "invalid_state"
        collision_reason = self._collision_reason()
        if collision_reason is not None:
            return collision_reason
        margin = 1e-6
        if (
            np.any(
                state.position_world_m[:2]
                < self.scene_config.workspace_min_m[:2] - margin
            )
            or np.any(
                state.position_world_m[:2]
                > self.scene_config.workspace_max_m[:2] + margin
            )
            or state.position_world_m[2] > self.scene_config.workspace_max_m[2] + margin
        ):
            return "out_of_bounds"
        if state.position_world_m[2] < self.config.termination.minimum_altitude_m:
            return "minimum_altitude"
        tilt = math.acos(float(np.clip(state.rotation_body_to_world[2, 2], -1.0, 1.0)))
        if tilt > self.config.termination.maximum_tilt_rad:
            return "unsafe_attitude"
        return None

    def _collision_reason(self) -> str | None:
        assert self.model is not None
        assert self.data is not None
        drone_body_id = self.model.body("quadrotor").id
        ground_geom_id = self.model.geom("ground").id
        for contact in self.data.contact[: self.data.ncon]:
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            if body1 == drone_body_id:
                other_geom = geom2
            elif body2 == drone_body_id:
                other_geom = geom1
            else:
                continue
            if other_geom == ground_geom_id:
                return "ground_collision"
            if int(self.model.geom_group[other_geom]) == 1:
                return "obstacle_collision"
        return None

    def _base_info(self, state: FlightState) -> dict[str, Any]:
        assert self.scene is not None
        return {
            "scene_seed": self._scene_seed,
            "sensor_seed": self._sensor_seed,
            "scene_mode": self._current_scene_mode,
            "endpoint_mode": self._current_endpoint_mode,
            "sensor_mode": self._current_sensor_mode,
            "episode_step": self._episode_steps,
            "goal_hold_steps": self._goal_hold_steps,
            "goal_hold_required_steps": self._goal_hold_required_steps,
            "goal_hold_time_s": (
                self._goal_hold_steps / self.config.simulation.policy_frequency_hz
            ),
            "physics_time_s": float(self.data.time) if self.data is not None else 0.0,
            "distance_to_goal_m": self._goal_distance(state),
        }

    def _goal_distance(self, state: FlightState) -> float:
        assert self.scene is not None
        return float(np.linalg.norm(self.scene.goal_world_m - state.position_world_m))

    def _require_reset(self) -> None:
        if self.model is None or self.data is None or self.scene is None:
            raise RuntimeError("step öncesinde reset çağrılmalı")

    def _next_episode_seed(self) -> int:
        return int(self.np_random.integers(0, 2**31 - 1))

    @staticmethod
    def _validated_episode_step_limit(value: object) -> int:
        if isinstance(value, bool):
            raise ValueError("max_episode_steps pozitif bir tamsayı olmalı")
        try:
            step_limit = operator.index(value)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError("max_episode_steps pozitif bir tamsayı olmalı") from error
        if step_limit <= 0:
            raise ValueError("max_episode_steps pozitif bir tamsayı olmalı")
        return int(step_limit)

    @staticmethod
    def _policy_step_ratio(
        policy_frequency_hz: float, physics_timestep_s: float
    ) -> int:
        policy_period_s = 1.0 / policy_frequency_hz
        steps_per_action = round(policy_period_s / physics_timestep_s)
        if steps_per_action < 1 or not math.isclose(
            steps_per_action * physics_timestep_s,
            policy_period_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Policy periyodu fizik timestep'inin tam katı olmalı")
        return steps_per_action

    @staticmethod
    def _validated_component_seed(value: object, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} negatif olmayan bir tamsayı olmalı")
        try:
            seed = operator.index(value)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError(f"{name} negatif olmayan bir tamsayı olmalı") from error
        if seed < 0:
            raise ValueError(f"{name} negatif olmayan bir tamsayı olmalı")
        return int(seed)

    @staticmethod
    def _yaw_from_rotation(rotation: FloatArray) -> float:
        return float(math.atan2(rotation[1, 0], rotation[0, 0]))

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)
