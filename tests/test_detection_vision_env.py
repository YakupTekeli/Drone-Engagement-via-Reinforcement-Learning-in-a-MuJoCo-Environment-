"""ADR-023 `vision-v4`: kutu yuvaları, zemin düzlemi menzili, dedektör kapısı."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env

from ebabil.controllers import read_flight_state
from ebabil.envs import (
    DEFAULT_DETECTION_VISION_CONFIG,
    DEFAULT_LATENT_VISION_CONFIG,
    DETECTION_VISION_ENVIRONMENT_ID,
    DETECTION_VISION_OBSERVATION_KEYS,
    DETECTION_VISION_OBSERVATION_VERSION,
    EbabilDetectionVisionNavigationEnv,
    load_vision_navigation_config,
)
from ebabil.envs.latent_vision import RandomizedActorCamera
from ebabil.envs.motion_constraint import (
    SpeedGovernorConfig,
    apply_speed_limit,
    governor_speed_limit,
    turn_speed_limit,
)
from ebabil.missions.hover_state import HoverState
from ebabil.perception import (
    ActorCameraFrame,
    TargetVisibility,
    load_camera_config,
)
from ebabil.perception.camera import project_world_points
from ebabil.perception.detection_slots import (
    EMPTY_SLOT,
    SLOT_FIELD_COUNT,
    CameraGeometry,
    DetectionSlotConfig,
    encode_detection_slots,
    ground_range_m,
)
from ebabil.perception.detections import OBSTACLE_CLASS_ID, TARGET_CLASS_ID, Detection
from ebabil.perception.detector import load_detector_config
from ebabil.perception.detector_quality import ApprovedDetector
from ebabil.scenes import (
    generate_static_scene,
    load_scene_config,
    load_target_area_config,
)
from ebabil.scenes.generator import build_scene_model
from ebabil.training.config import load_training_config
from ebabil.training.env_factory import monitor_info_keys

D1A_SCENE = Path("src/ebabil/configs/environment/static_scene_d1a.toml")
LEVEL = (0.0, 0.0, -1.0)


def _rotation(roll: float, pitch: float) -> np.ndarray:
    """Gövdeden dünyaya: önce pitch (+y ekseni, burun aşağı), sonra roll (+x)."""

    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    return rx @ ry


class GroundRangeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = load_camera_config()
        self.geometry = CameraGeometry.from_camera_config(self.camera)
        cx, cy = self.camera.principal_point_px
        self.center = (cx + 0.5, cy + 0.5)

    def test_level_center_pixel_matches_mount_pitch(self) -> None:
        distance = ground_range_m(
            self.geometry, *self.center, gravity_direction_body=LEVEL, altitude_m=1.5
        )
        expected = (1.5 - 0.01) / math.tan(self.camera.pitch_down_rad)
        self.assertAlmostEqual(distance or 0.0, expected, places=9)

    def test_nose_down_pitch_adds_to_depression(self) -> None:
        pitch = math.radians(8.0)
        gravity = _rotation(0.0, pitch).T @ np.array(LEVEL)
        distance = ground_range_m(
            self.geometry, *self.center, gravity_direction_body=gravity, altitude_m=2.0
        )
        camera_height = 2.0 - float(np.asarray(self.geometry.position_body_m) @ gravity)
        expected = camera_height / math.tan(self.camera.pitch_down_rad + pitch)
        self.assertAlmostEqual(distance or 0.0, expected, places=9)

    def test_ray_above_horizon_or_camera_below_ground_is_invalid(self) -> None:
        # Burun 60 derece yukarı: merkez ışını ufkun üstüne çıkar.
        gravity = _rotation(0.0, math.radians(-60.0)).T @ np.array(LEVEL)
        self.assertIsNone(
            ground_range_m(
                self.geometry,
                *self.center,
                gravity_direction_body=gravity,
                altitude_m=1.5,
            )
        )
        self.assertIsNone(
            ground_range_m(
                self.geometry,
                *self.center,
                gravity_direction_body=LEVEL,
                altitude_m=-0.1,
            )
        )

    def test_matches_mujoco_projection_under_roll_and_pitch(self) -> None:
        scene = generate_static_scene(20260929, load_scene_config(), mode="clear")
        model = build_scene_model(
            scene,
            camera_config=self.camera,
            target_config=load_target_area_config(),
            include_legacy_goal_marker=False,
        )
        data = mujoco.MjData(model)
        free = next(
            j
            for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        )
        address = int(model.jnt_qposadr[free])
        rotation = _rotation(math.radians(6.0), math.radians(9.0))
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
        data.qpos[address : address + 3] = (0.3, -0.2, 1.8)
        data.qpos[address + 3 : address + 7] = quaternion
        mujoco.mj_forward(model, data)
        state = read_flight_state(model, data)
        gravity = np.asarray(state.rotation_body_to_world).T @ np.array(LEVEL)
        altitude = float(state.position_world_m[2])
        camera_id = model.camera(self.camera.name).id
        camera_world = data.cam_xpos[camera_id].copy()
        grid = [
            (camera_world[0] + dx, camera_world[1] + dy, 0.0)
            for dx in np.linspace(0.8, 7.0, 7)
            for dy in np.linspace(-2.5, 2.5, 5)
        ]
        projection = project_world_points(model, data, grid, self.camera)
        checked = 0
        for point, pixel, inside in zip(
            grid, projection.pixels_xy, projection.inside_image, strict=True
        ):
            if not inside:
                continue
            distance = ground_range_m(
                self.geometry,
                float(pixel[0]) + 0.5,
                float(pixel[1]) + 0.5,
                gravity_direction_body=gravity,
                altitude_m=altitude,
            )
            truth = math.hypot(point[0] - camera_world[0], point[1] - camera_world[1])
            self.assertIsNotNone(distance)
            self.assertAlmostEqual(distance or 0.0, truth, delta=1e-6 * max(truth, 1.0))
            checked += 1
        self.assertGreaterEqual(checked, 10)


class SlotEncodingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DetectionSlotConfig()
        self.geometry = CameraGeometry.from_camera_config(load_camera_config())

    def encode(self, detections: list[Detection]) -> np.ndarray:
        return encode_detection_slots(
            detections,
            config=self.config,
            geometry=self.geometry,
            gravity_direction_body=LEVEL,
            altitude_m=1.5,
        ).vector

    def test_empty_input_gives_empty_slots(self) -> None:
        vector = self.encode([])
        self.assertEqual(vector.shape, (72,))
        np.testing.assert_array_equal(vector, np.tile(np.float32(EMPTY_SLOT), 9))

    def test_obstacles_sorted_by_range_invalid_last_and_thresholded(self) -> None:
        far = Detection(OBSTACLE_CLASS_ID, 100, 10, 120, 60, 0.9)
        near = Detection(OBSTACLE_CLASS_ID, 200, 20, 240, 200, 0.8)
        cut = Detection(OBSTACLE_CLASS_ID, 0, 50, 90, 240, 0.7)  # taban kadraj dışı
        weak = Detection(OBSTACLE_CLASS_ID, 150, 20, 160, 100, 0.1)  # güven eşiği altı
        encoding = encode_detection_slots(
            [far, cut, weak, near],
            config=self.config,
            geometry=self.geometry,
            gravity_direction_body=LEVEL,
            altitude_m=1.5,
        )
        slots = encoding.vector.reshape(9, SLOT_FIELD_COUNT)
        self.assertEqual(encoding.obstacle_detections, 3)
        self.assertEqual(encoding.obstacles_encoded, 3)
        self.assertEqual(encoding.range_valid_count, 2)
        self.assertAlmostEqual(float(slots[0, 1]), 220 / 320, places=6)  # yakın önce
        self.assertAlmostEqual(float(slots[1, 1]), 110 / 320, places=6)
        self.assertLess(slots[0, 6], slots[1, 6])
        self.assertEqual(tuple(slots[2, 6:]), (1.0, 0.0))  # geçersiz: range 1, valid 0
        self.assertAlmostEqual(float(slots[2, 5]), 0.7, places=6)
        np.testing.assert_array_equal(slots[3], np.float32(EMPTY_SLOT))
        self.assertTrue(np.all((encoding.vector >= 0.0) & (encoding.vector <= 1.0)))

    def test_truncation_keeps_nearest_obstacles(self) -> None:
        boxes = [
            Detection(OBSTACLE_CLASS_ID, 10 + 25 * i, 10, 30 + 25 * i, 40 + 15 * i, 0.9)
            for i in range(10)
        ]
        encoding = encode_detection_slots(
            boxes,
            config=self.config,
            geometry=self.geometry,
            gravity_direction_body=LEVEL,
            altitude_m=1.5,
        )
        self.assertEqual(encoding.obstacle_detections, 10)
        self.assertEqual(encoding.obstacles_encoded, 8)
        kept_x = encoding.vector.reshape(9, SLOT_FIELD_COUNT)[:8, 1] * 320
        self.assertNotIn(20.0, np.round(kept_x, 3))  # en uzak iki kutu düştü
        self.assertNotIn(45.0, np.round(kept_x, 3))

    def test_target_slot_uses_most_confident_valid_target(self) -> None:
        detections = [
            Detection(TARGET_CLASS_ID, 40, 150, 120, 200, 0.6),
            Detection(TARGET_CLASS_ID, 200, 150, 280, 200, 0.9),
            Detection(TARGET_CLASS_ID, 0, 0, 3, 3, 0.99),  # alan 9 < 16
            Detection(TARGET_CLASS_ID, 100, 100, 180, 140, 0.4),  # güven < 0.5
        ]
        encoding = encode_detection_slots(
            detections,
            config=self.config,
            geometry=self.geometry,
            gravity_direction_body=LEVEL,
            altitude_m=1.5,
        )
        target = encoding.vector.reshape(9, SLOT_FIELD_COUNT)[8]
        self.assertTrue(encoding.target_present)
        self.assertAlmostEqual(float(target[5]), 0.9, places=6)
        self.assertAlmostEqual(encoding.target_center_x, 2.0 * 240 / 320 - 1.0)
        self.assertEqual(float(target[7]), 1.0)


class FakeActorCamera:
    def __init__(self) -> None:
        config = load_camera_config()
        self.shape = (config.output_height_px, config.output_width_px, 3)

    def capture_actor(self, data: mujoco.MjData) -> ActorCameraFrame:
        timestamp_s = float(data.time)
        rgb = np.full(self.shape, 60 + int(timestamp_s * 1000.0) % 120, dtype=np.uint8)
        return ActorCameraFrame(rgb=rgb, timestamp_s=timestamp_s)

    def close(self) -> None:
        pass


class FakeDetector:
    def __init__(self) -> None:
        self.target = False
        self.calls: list[tuple[tuple[int, ...], np.dtype, float]] = []

    def detect(self, rgb: np.ndarray, *, conf: float) -> tuple[Detection, ...]:
        self.calls.append((rgb.shape, rgb.dtype, conf))
        boxes = [
            Detection(OBSTACLE_CLASS_ID, 100, 40, 140, 150, 0.9),
            Detection(OBSTACLE_CLASS_ID, 250, 10, 290, 90, 0.6),
        ]
        if self.target:
            boxes.append(Detection(TARGET_CLASS_ID, 150, 170, 230, 215, 0.8))
        return tuple(boxes)


class FakeTargetObserver:
    def __init__(self) -> None:
        self.visible = False

    def observe(self, data: mujoco.MjData) -> TargetVisibility:
        return TargetVisibility(
            self.visible, 64 if self.visible else 0, 0.0, 0.0, float(data.time)
        )

    def close(self) -> None:
        pass


class DetectionVisionEnvironmentTests(unittest.TestCase):
    def make_env(
        self,
        *,
        randomize: bool = True,
        diagnostics: bool = True,
        assist: bool = False,
        **kwargs: object,
    ) -> tuple[EbabilDetectionVisionNavigationEnv, FakeDetector, FakeTargetObserver]:
        config = load_vision_navigation_config(DEFAULT_DETECTION_VISION_CONFIG)
        config = replace(
            config,
            visual_randomization=replace(
                config.visual_randomization, enabled=randomize
            ),
            detections=replace(config.detections, privileged_diagnostics=diagnostics),
            terminal_assist=replace(config.terminal_assist, enabled=assist),
        )
        detector, observer = FakeDetector(), FakeTargetObserver()
        env = EbabilDetectionVisionNavigationEnv(
            approval_manifest="synthetic-detector.json",
            config=config,
            camera_factory=lambda _: FakeActorCamera(),
            detector_factory=lambda: detector,
            target_observer_factory=lambda _: observer,
            **kwargs,  # type: ignore[arg-type]
        )
        self.addCleanup(env.close)
        return env, detector, observer

    def test_contract_keys_shapes_bounds_and_version(self) -> None:
        env, _, _ = self.make_env(scene_mode="random_obstacles")
        observation, info = env.reset(seed=10, options={"scene_seed": 11})

        self.assertEqual(tuple(sorted(observation)), DETECTION_VISION_OBSERVATION_KEYS)
        self.assertEqual(
            tuple(sorted(env.observation_space.spaces)),
            DETECTION_VISION_OBSERVATION_KEYS,
        )
        self.assertEqual(observation["detections"].shape, (72,))
        self.assertEqual(sum(v.shape[0] for v in observation.values()), 88)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(
            info["observation_version"], DETECTION_VISION_OBSERVATION_VERSION
        )
        prohibited = ("depth", "segmentation", "qpos", "qvel", "world", "latent", "map")
        self.assertFalse(
            any(token in key for key in observation for token in prohibited)
        )
        self.assertEqual(ActorCameraFrame.field_names(), ("rgb", "timestamp_s"))

    def test_detector_sees_only_actor_rgb_and_observation_is_its_encoding(self) -> None:
        env, detector, observer = self.make_env(randomize=True)
        observation, _ = env.reset(seed=12)

        self.assertTrue(detector.calls)
        shape, dtype, conf = detector.calls[-1]
        self.assertEqual(shape, (240, 320, 3))
        self.assertEqual(dtype, np.uint8)
        self.assertAlmostEqual(conf, 0.25)
        camera = env._camera
        self.assertIsInstance(camera, RandomizedActorCamera)
        self.assertFalse(hasattr(camera, "capture_privileged"))
        self.assertIs(env._target_observer, observer)
        gravity, altitude = env._attitude_and_altitude()
        expected = encode_detection_slots(
            detector.detect(np.zeros((240, 320, 3), np.uint8), conf=0.25),
            config=env.vision_config.detections.slot_config(),
            geometry=CameraGeometry.from_camera_config(env.camera_config),
            gravity_direction_body=gravity,
            altitude_m=altitude,
        )
        np.testing.assert_allclose(
            observation["detections"], expected.vector, atol=1e-6
        )
        slots = observation["detections"].reshape(9, SLOT_FIELD_COUNT)
        self.assertEqual(float(slots[0, 0]), 1.0)
        self.assertEqual(float(slots[1, 0]), 1.0)
        self.assertEqual(float(slots[8, 0]), 0.0)

    def test_hover_gate_follows_detector_not_privileged_observer(self) -> None:
        env, detector, observer = self.make_env(randomize=False)
        _, info = env.reset(seed=20)
        action = np.zeros(4, dtype=np.float32)
        self.assertFalse(info["target_visible_detector"])

        observer.visible = True  # yalnız tanı: kapıyı açmamalı
        for _ in range(8):
            _, _, _, _, info = env.step(action)
        self.assertTrue(info["target_visible_privileged"])
        self.assertFalse(info["hover_target_visible"])
        self.assertIn(info["hover_state"], ("transit", "search"))

        detector.target = True
        for _ in range(8):
            _, _, terminated, _, info = env.step(action)
        self.assertTrue(info["target_visible_detector"])
        self.assertTrue(info["hover_target_visible"])
        self.assertIn(info["hover_state"], ("acquired", "approach", "settle", "hold"))
        self.assertFalse(terminated)

        detector.target = False
        for _ in range(80):
            _, _, terminated, _, info = env.step(action)
            if terminated:
                break
        self.assertTrue(terminated)
        self.assertEqual(info["termination_reason"], "hover_abort_target_lost")

    def test_diagnostics_can_be_disabled(self) -> None:
        env, _, _ = self.make_env(randomize=False, diagnostics=False)
        _, info = env.reset(seed=21)
        self.assertIsNone(env._target_observer)
        self.assertIsNone(info["target_visible_privileged"])

    def test_same_seed_and_actions_repeat_observations_and_rewards(self) -> None:
        first, _, _ = self.make_env(scene_mode="random_obstacles")
        second, _, _ = self.make_env(scene_mode="random_obstacles")
        first_obs, _ = first.reset(seed=22)
        second_obs, _ = second.reset(seed=22)
        for action in (
            np.array([0.4, 0.1, -0.2, 0.15], dtype=np.float32),
            np.array([0.1, -0.1, 0.3, -0.2], dtype=np.float32),
        ):
            first_obs, first_reward, *_ = first.step(action)
            second_obs, second_reward, *_ = second.step(action)
            for key in DETECTION_VISION_OBSERVATION_KEYS:
                np.testing.assert_array_equal(first_obs[key], second_obs[key])
            self.assertEqual(first_reward, second_reward)

    def test_gymnasium_checker_and_registered_environment(self) -> None:
        env, _, _ = self.make_env()
        check_env(env, warn=True, skip_render_check=True)
        detector = FakeDetector()
        registered = gym.make(
            DETECTION_VISION_ENVIRONMENT_ID,
            approval_manifest="synthetic-detector.json",
            camera_factory=lambda _: FakeActorCamera(),
            detector_factory=lambda: detector,
            target_observer_factory=lambda _: FakeTargetObserver(),
        )
        self.addCleanup(registered.close)
        observation, info = registered.reset(seed=23)
        self.assertTrue(registered.observation_space.contains(observation))
        self.assertIn("range_valid_fraction", info)

    def _approved(self, geometries: tuple[str, ...] | None) -> ApprovedDetector:
        return ApprovedDetector(
            weights=Path("weights.pt"),
            config=load_detector_config(),
            dataset={},
            basis="user_decision" if geometries else "audit_passed",
            obstacle_geometries=geometries,
        )

    def test_approved_detector_is_loaded_from_manifest_without_factory(self) -> None:
        detector = FakeDetector()
        with (
            patch(
                "ebabil.perception.detector_quality.verify_detector_approval",
                return_value=self._approved(None),
            ) as verify,
            patch(
                "ebabil.perception.detector.YoloDetector", return_value=detector
            ) as constructor,
        ):
            env = EbabilDetectionVisionNavigationEnv(
                approval_manifest="runs/some/approval_manifest.json",
                camera_factory=lambda _: FakeActorCamera(),
                target_observer_factory=lambda _: FakeTargetObserver(),
            )
            self.addCleanup(env.close)
        verify.assert_called_once()
        self.assertEqual(constructor.call_args.kwargs["device"], "mps")
        self.assertIs(env.detector, detector)

    def test_scoped_approval_rejects_unapproved_obstacle_geometry(self) -> None:
        scoped = self._approved(("box", "cylinder", "capsule", "sphere"))
        d1a = load_scene_config(D1A_SCENE)
        with (
            patch(
                "ebabil.perception.detector_quality.verify_detector_approval",
                return_value=scoped,
            ),
            patch(
                "ebabil.perception.detector.YoloDetector", return_value=FakeDetector()
            ),
        ):
            env = EbabilDetectionVisionNavigationEnv(
                approval_manifest="runs/some/approval_manifest.json",
                scene_mode="random_obstacles",
                scene_config=d1a,
                camera_factory=lambda _: FakeActorCamera(),
                target_observer_factory=lambda _: FakeTargetObserver(),
            )
            self.addCleanup(env.close)
            env.reset(seed=31)
            with self.assertRaisesRegex(ValueError, "REV-050"):
                EbabilDetectionVisionNavigationEnv(
                    approval_manifest="runs/some/approval_manifest.json",
                    scene_mode="random_obstacles",
                    scene_config=replace(d1a, obstacle_geometries=("box", "ellipsoid")),
                    camera_factory=lambda _: FakeActorCamera(),
                    target_observer_factory=lambda _: FakeTargetObserver(),
                )


class CameraConeMotionEnvironmentTests(unittest.TestCase):
    def make_env(self, *, governor: bool = False) -> EbabilDetectionVisionNavigationEnv:
        config = load_vision_navigation_config(DEFAULT_DETECTION_VISION_CONFIG)
        config = replace(
            config,
            visual_randomization=replace(config.visual_randomization, enabled=False),
            terminal_assist=replace(config.terminal_assist, enabled=False),
            speed_governor=replace(config.speed_governor, enabled=governor),
        )
        env = EbabilDetectionVisionNavigationEnv(
            approval_manifest="synthetic-detector.json",
            config=config,
            camera_factory=lambda _: FakeActorCamera(),
            detector_factory=FakeDetector,
            target_observer_factory=lambda _: FakeTargetObserver(),
        )
        self.addCleanup(env.close)
        return env

    def test_constraint_depends_on_hover_state(self) -> None:
        env = self.make_env()
        env.reset(seed=40)
        self.assertEqual(env.hover_state_machine.state, HoverState.TRANSIT)
        np.testing.assert_allclose(
            env._constrain_horizontal_body(np.array([-1.5, 0.0])), (0.0, 0.0)
        )
        np.testing.assert_allclose(
            env._constrain_horizontal_body(np.array([1.5, 0.0])), (1.5, 0.0)
        )
        env.hover_state_machine._state = HoverState.APPROACH
        np.testing.assert_allclose(
            env._constrain_horizontal_body(np.array([-1.5, 0.0])), (0.0, 0.0)
        )
        env.hover_state_machine._state = HoverState.SETTLE
        np.testing.assert_allclose(
            env._constrain_horizontal_body(np.array([-1.5, 0.0])),
            (-0.5, 0.0),
            atol=1e-12,
        )

    def test_backward_policy_cannot_fly_backward_in_transit(self) -> None:
        env = self.make_env()
        _, info = env.reset(seed=41)
        self.assertEqual(info["action_version"], "vision-action-v3")
        backward = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(20):
            _, _, terminated, truncated, info = env.step(backward)
            self.assertFalse(terminated or truncated)
            state = read_flight_state(env.model, env.data)
            velocity_body = np.asarray(state.rotation_body_to_world).T @ np.asarray(
                state.velocity_world_m_s
            )
            self.assertGreater(float(velocity_body[0]), -0.05)
        self.assertEqual(info["hover_state"], "transit")
        self.assertAlmostEqual(info["motion_constrained_fraction"], 1.0)

    def test_v0_and_v3_keep_holonomic_action(self) -> None:
        self.assertFalse(load_vision_navigation_config().motion_constraint.enabled)
        self.assertFalse(
            load_vision_navigation_config(
                DEFAULT_LATENT_VISION_CONFIG
            ).motion_constraint.enabled
        )
        v4 = load_vision_navigation_config(
            DEFAULT_DETECTION_VISION_CONFIG
        ).motion_constraint
        self.assertTrue(v4.enabled)
        self.assertEqual((v4.half_angle_deg, v4.terminal_blind_speed_m_s), (30.0, 0.5))
        self.assertEqual(v4.relaxed_states, ("settle", "hold"))


class SpeedGovernorTests(unittest.TestCase):
    def test_limit_from_nearest_corridor_obstacle(self) -> None:
        config = SpeedGovernorConfig(enabled=True)
        slots = np.tile(np.float32(EMPTY_SLOT), 9)
        self.assertIsNone(governor_speed_limit(slots, config, obstacle_slots=8))
        # Koridor dışı engel (cx_norm 0.8) sınır üretmez.
        slots[0:8] = [1.0, 0.9, 0.5, 0.1, 0.2, 0.9, 2.0 / 12.0, 1.0]
        self.assertIsNone(governor_speed_limit(slots, config, obstacle_slots=8))
        # Koridor içi 2.4 m -> 1.0 x (2.4 - 0.6) = 1.8 m/s.
        slots[8:16] = [1.0, 0.45, 0.5, 0.1, 0.2, 0.9, 2.4 / 12.0, 1.0]
        self.assertAlmostEqual(
            governor_speed_limit(slots, config, obstacle_slots=8) or 0.0, 1.8
        )
        # Tabanı kadraj dışı yuva 0.5 m sayılır -> taban hız 0.15.
        slots[16:24] = [1.0, 0.5, 0.8, 0.3, 0.5, 0.9, 1.0, 0.0]
        self.assertAlmostEqual(
            governor_speed_limit(slots, config, obstacle_slots=8) or 0.0, 0.15
        )
        # Hedef yuvası (9.) sayılmaz.
        slots[:] = np.tile(np.float32(EMPTY_SLOT), 9)
        slots[64:72] = [1.0, 0.5, 0.8, 0.3, 0.2, 0.9, 0.05, 1.0]
        self.assertIsNone(governor_speed_limit(slots, config, obstacle_slots=8))
        np.testing.assert_allclose(apply_speed_limit((1.5, 0.0), 0.6), (0.6, 0.0))
        np.testing.assert_allclose(apply_speed_limit((0.3, 0.1), 0.6), (0.3, 0.1))

    def test_close_edge_obstacle_and_turn_coupling(self) -> None:
        # Dönüş bağlaması varsayılan kapalı (0); test için açıkça 0.8.
        config = SpeedGovernorConfig(enabled=True, turn_penalty=0.8)
        slots = np.tile(np.float32(EMPTY_SLOT), 9)
        # Kenardaki (koridor dışı) yakın engel sınır üretmez (koridor kuralı).
        slots[0:8] = [1.0, 0.05, 0.5, 0.1, 0.6, 0.9, 1.0, 0.0]
        self.assertIsNone(governor_speed_limit(slots, config, obstacle_slots=8))
        # Dönüş-hız bağlaması: tam yaw'da 1.5 x (1 - 0.8) = 0.3; yaw yokken sınır yok etkisi (1.5).
        self.assertAlmostEqual(
            turn_speed_limit(1.5, config, max_yaw_rate_rad_s=1.5, max_speed_m_s=1.5)
            or 0.0,
            0.3,
        )
        self.assertAlmostEqual(
            turn_speed_limit(0.0, config, max_yaw_rate_rad_s=1.5, max_speed_m_s=1.5)
            or 0.0,
            1.5,
        )
        self.assertIsNone(
            turn_speed_limit(
                1.0,
                replace(config, turn_penalty=0.0),
                max_yaw_rate_rad_s=1.5,
                max_speed_m_s=1.5,
            )
        )

    def test_environment_caps_forward_speed_when_obstacle_ahead(self) -> None:
        helper = CameraConeMotionEnvironmentTests()
        env = CameraConeMotionEnvironmentTests.make_env(helper, governor=True)
        self.addCleanup(env.close)
        env.reset(seed=60)
        # Sınır komut üretilirken hesaplanır: bir adım ileri komut ver.
        _, _, _, _, info = env.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        # Sahte engel kutusu (100-140 px, alt kenar 150) koridorda, ~2 m menzilde.
        limit = info["speed_governor_limit_m_s"]
        self.assertIsNotNone(limit)
        self.assertLess(limit, 1.5)
        capped = env._constrain_horizontal_body(np.array([1.5, 0.0]))
        self.assertAlmostEqual(float(np.linalg.norm(capped)), limit, places=6)
        self.assertTrue(
            load_vision_navigation_config(
                DEFAULT_DETECTION_VISION_CONFIG
            ).speed_governor.enabled
        )
        self.assertFalse(load_vision_navigation_config().speed_governor.enabled)


class TerminalAssistEnvironmentTests(unittest.TestCase):
    def test_assist_replaces_policy_action_only_in_terminal_states(self) -> None:
        helper = DetectionVisionEnvironmentTests()
        env, detector, _ = DetectionVisionEnvironmentTests.make_env(
            helper, randomize=False, assist=True
        )
        self.addCleanup(env.close)
        _, info = env.reset(seed=50)
        self.assertFalse(info["terminal_assist_active"])
        detector.target = True
        action = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        self.assertFalse(info["terminal_assist_active"])  # transit: politika sürer
        env.hover_state_machine._state = HoverState.APPROACH
        _, _, _, _, info = env.step(action)
        self.assertTrue(info["terminal_assist_active"])
        self.assertGreater(info["terminal_assist_fraction"], 0.0)
        assisted = env._terminal_assist_action()
        self.assertEqual(assisted.shape, (4,))
        self.assertTrue(np.all(np.abs(assisted) <= 1.0))
        # Sahte hedef kutusu kadrajın sağ-altında: ileri komut ve sağa dönüş.
        self.assertGreater(float(assisted[0]), 0.0)
        self.assertLess(float(assisted[3]), 0.0)

    def test_v4_config_enables_assist_in_terminal_states(self) -> None:
        assist = load_vision_navigation_config(
            DEFAULT_DETECTION_VISION_CONFIG
        ).terminal_assist
        self.assertTrue(assist.enabled)
        self.assertEqual(assist.states, ("approach", "settle", "hold"))
        self.assertFalse(load_vision_navigation_config().terminal_assist.enabled)


class DetectionVisionConfigTests(unittest.TestCase):
    def test_v4_config_keeps_action_and_hover_semantics(self) -> None:
        v4 = load_vision_navigation_config(DEFAULT_DETECTION_VISION_CONFIG)
        v3 = load_vision_navigation_config(DEFAULT_LATENT_VISION_CONFIG)
        v0 = load_vision_navigation_config()
        for reference in (v0, v3):
            self.assertEqual(v4.navigation.action, reference.navigation.action)
            # ADR-017 Revizyon 1 bayrağı dışında hover parametreleri v0/v3 ile aynı.
            self.assertEqual(
                replace(v4.terminal_hover, far_loss_returns_to_transit=False),
                reference.terminal_hover,
            )
            self.assertFalse(reference.terminal_hover.far_loss_returns_to_transit)
            self.assertEqual(v4.terminal_hover_reward, reference.terminal_hover_reward)
        self.assertTrue(v4.terminal_hover.far_loss_returns_to_transit)
        self.assertEqual(v4.detections.slot_config().vector_size, 72)
        # Eğitimde tanı render'ı kapalı (throughput kapısı, ADR-023 §8).
        self.assertFalse(v4.detections.privileged_diagnostics)
        self.assertTrue(v4.visual_randomization.enabled)

    def test_training_config_kind_and_parity(self) -> None:
        config = load_training_config(
            "src/ebabil/configs/training/ppo_detector_d1a.toml"
        )
        reference = load_training_config(
            "src/ebabil/configs/training/ppo_vision_d1a.toml"
        )
        self.assertEqual(config.environment.kind, "detection_vision")
        self.assertEqual(config.environment.visual_domain_split, "train")
        self.assertIn("detector_audit", str(config.environment.approval_manifest))
        self.assertEqual(config.ppo, reference.ppo)
        self.assertEqual(config.run, reference.run)
        self.assertIn("hover_horizontal_distance_m", monitor_info_keys(config))


if __name__ == "__main__":
    unittest.main()
