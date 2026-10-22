"""Vision-v1 observation, scheduler ve dinamik horizon testleri."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import mujoco
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from torch.nn import functional

from ebabil.envs import (
    VISION_ENVIRONMENT_ID,
    VISION_OBSERVATION_KEYS,
    VISION_OBSERVATION_VERSION,
    EbabilVisionNavigationEnv,
    SimulationDeadlineScheduler,
    compute_segment_horizon,
    load_vision_navigation_config,
)
from ebabil.envs.vision import OBSTACLE_MAP_SHAPE, OBSTACLE_MAP_SIZE
from ebabil.perception import (
    ActorCameraFrame,
    TinyVisionEncoder,
    load_camera_config,
    load_vision_encoder_config,
)
from ebabil.perception.model import preprocess_actor_rgb


class FakeActorCamera:
    def __init__(self) -> None:
        config = load_camera_config()
        self.shape = (config.output_height_px, config.output_width_px, 3)
        self.closed = False

    def capture_actor(self, data: mujoco.MjData) -> ActorCameraFrame:
        timestamp_s = float(data.time)
        value = int(round(timestamp_s * 1000.0)) % 256
        return ActorCameraFrame(
            rgb=np.full(self.shape, value, dtype=np.uint8),
            timestamp_s=timestamp_s,
        )

    def close(self) -> None:
        self.closed = True


def _approved_encoder(_: Path) -> TinyVisionEncoder:
    torch.manual_seed(20260911)
    config, _ = load_vision_encoder_config()
    return TinyVisionEncoder(config)


class SegmentTimingTests(unittest.TestCase):
    def test_deadline_scheduler_uses_simulation_time(self) -> None:
        scheduler = SimulationDeadlineScheduler(30.0)
        scheduler.reset(0.0)

        self.assertFalse(scheduler.sample_due(0.032))
        self.assertTrue(scheduler.sample_due(0.034))
        self.assertFalse(scheduler.sample_due(0.050))
        self.assertTrue(scheduler.sample_due(0.068))
        self.assertTrue(scheduler.sample_due(0.100))

    def test_segment_horizon_is_monotonic_clamped_and_adds_terminal_reserve(
        self,
    ) -> None:
        config = load_vision_navigation_config()
        budget = config.segment_budget

        short = compute_segment_horizon(4.0, 20.0, budget, final_segment=False)
        long = compute_segment_horizon(30.0, 20.0, budget, final_segment=False)
        final = compute_segment_horizon(30.0, 20.0, budget, final_segment=True)

        self.assertEqual(short.budget_seconds, 20.0)
        self.assertEqual(short.maximum_steps, 400)
        self.assertEqual(long.budget_seconds, 40.0)
        self.assertEqual(long.maximum_steps, 800)
        self.assertEqual(final.budget_seconds, 53.0)
        self.assertEqual(final.maximum_steps, 1060)
        with self.assertRaisesRegex(ValueError, "üst sınır"):
            compute_segment_horizon(31.0, 20.0, budget, final_segment=False)


class VisionNavigationEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        loader = patch(
            "ebabil.envs.vision.load_approved_vision_encoder",
            side_effect=_approved_encoder,
        )
        loader.start()
        self.addCleanup(loader.stop)

    def make_env(self, **kwargs: object) -> EbabilVisionNavigationEnv:
        env = EbabilVisionNavigationEnv(
            approval_manifest="synthetic-approval.json",
            camera_factory=lambda _: FakeActorCamera(),
            **kwargs,
        )
        self.addCleanup(env.close)
        return env

    def test_config_and_encoder_are_versioned_and_frozen(self) -> None:
        env = self.make_env()

        self.assertEqual(env.observation_version, VISION_OBSERVATION_VERSION)
        self.assertEqual(env.action_space.shape, (4,))
        self.assertEqual(
            tuple(sorted(env.observation_space.spaces)), VISION_OBSERVATION_KEYS
        )
        self.assertFalse(env.encoder.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in env.encoder.parameters())
        )

    def test_observation_contains_only_declared_deployable_channels(self) -> None:
        env = self.make_env(scene_mode="single_blocker")

        observation, info = env.reset(seed=10, options={"scene_seed": 11})

        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(tuple(sorted(observation)), VISION_OBSERVATION_KEYS)
        self.assertEqual(info["observation_version"], VISION_OBSERVATION_VERSION)
        # `obstacle_map` (REV-037) ayrıcalıklı DEĞİLDİR: encoder'ın RGB'den
        # ürettiği tahmindir, `free_space` ile aynı statüde. Bu yüzden isim
        # tabanlı yasak listesinden "obstacle" kalktı ve yerine aşağıdaki
        # `test_obstacle_map_is_an_estimate_not_privileged_truth` konuldu.
        prohibited = (
            "depth",
            "ray",
            "range",
            "obstacle_mask",
            "segmentation",
            "qpos",
            "qvel",
            "world",
        )
        self.assertFalse(
            any(token in key for key in observation for token in prohibited)
        )
        self.assertEqual(ActorCameraFrame.field_names(), ("rgb", "timestamp_s"))

    def test_obstacle_map_is_an_estimate_not_privileged_truth(self) -> None:
        """REV-037: kanal encoder çıktısı olmalı, ayrıcalıklı maske olmamalı.

        İsim tabanlı yasak listesi gevşetildiği için buradaki iki iddia asıl
        güvenceyi sağlar: (1) kanal encoder'ın ısı haritasından birebir gelir,
        (2) ayrıcalıklı gerçek maske actor yoluna hiç girmez.
        """

        env = self.make_env(scene_mode="random_obstacles")
        observation, _ = env.reset(seed=31, options={"scene_seed": 32})
        channel = np.asarray(observation["obstacle_map"], dtype=np.float64)

        self.assertEqual(channel.shape, (OBSTACLE_MAP_SIZE,))
        self.assertTrue(np.all((channel >= 0.0) & (channel <= 1.0)))

        # (1) Encoder'ın ham çıktısıyla birebir aynı.
        camera = env._camera
        assert camera is not None
        with torch.no_grad():
            frame = camera.capture_actor(env.data)
            output = env.encoder(preprocess_actor_rgb(frame.rgb).unsqueeze(0))
            expected = (
                torch.sigmoid(output.obstacle_heatmap_logits)[0]
                .reshape(-1)
                .cpu()
                .numpy()
            )
        np.testing.assert_allclose(channel, expected, atol=1e-6)

        # (2) Actor kamerası ayrıcalıklı çıktı üretmiyor; kanalın kaynağı
        # yalnız RGB olabilir.
        self.assertEqual(ActorCameraFrame.field_names(), ("rgb", "timestamp_s"))
        self.assertFalse(hasattr(camera, "capture_privileged"))
        self.assertEqual(
            functional.adaptive_max_pool2d(
                torch.zeros(1, 1, 72, 96), OBSTACLE_MAP_SHAPE
            ).numel(),
            OBSTACLE_MAP_SIZE,
        )

    def test_camera_runs_at_30_hz_and_features_are_sample_and_hold(self) -> None:
        env = self.make_env()
        first, first_info = env.reset(seed=1)

        second, _, _, _, second_info = env.step(np.zeros(4, dtype=np.float32))

        self.assertEqual(first_info["camera_frame_count"], 1)
        self.assertEqual(second_info["camera_frame_count"], 2)
        self.assertAlmostEqual(
            second_info["camera_frame_timestamp_s"], 0.034, places=12
        )
        self.assertAlmostEqual(second_info["camera_frame_age_s"], 0.016, places=12)
        self.assertGreater(second["timing"][1], first["timing"][1])

        third, _, _, _, third_info = env.step(np.zeros(4, dtype=np.float32))

        self.assertEqual(third_info["camera_frame_count"], 4)
        self.assertAlmostEqual(third_info["camera_frame_timestamp_s"], 0.1, places=12)
        self.assertAlmostEqual(third_info["camera_frame_age_s"], 0.0, places=12)
        self.assertEqual(third["timing"][1], 0.0)

    def test_holonomic_action_and_invalid_action_semantics(self) -> None:
        """ADR-020: araç yatay düzlemde her yöne uçar, hız sınırı izotropiktir.

        `vision-action-v1` üç boyutluydu ve ileri hızı `max(0, a[0])` ile
        kırpıyordu; araç yana ve geriye hiç uçamıyordu. Bu fiziksel olarak
        gerçek dışıydı ve terminal hover'ı çözülemez kılıyordu: araç hedefin
        `0.65 m` içine giriyor fakat `0.243 m`'lik halkada duramıyordu, çünkü
        yanal düzeltme için yaw çevirmesi gerekiyor ve `38` dereceden fazla
        çevirince hedefi kadraj dışına atıyordu.
        """

        env = self.make_env()
        maximum = 1.5

        for label, action, expected_world in (
            ("ileri", [1.0, 0.0, 0.0, 0.0], [maximum, 0.0, 0.0]),
            ("geri", [-1.0, 0.0, 0.0, 0.0], [-maximum, 0.0, 0.0]),
            ("sağa", [0.0, 1.0, 0.0, 0.0], [0.0, maximum, 0.0]),
            ("sola", [0.0, -1.0, 0.0, 0.0], [0.0, -maximum, 0.0]),
        ):
            with self.subTest(label=label):
                env.reset(seed=2)
                _, _, _, _, info = env.step(np.array(action, dtype=np.float32))

                np.testing.assert_allclose(
                    info["velocity_command_world_m_s"], expected_world, atol=1e-9
                )

        # Çapraz komut hız sınırını aşmamalı; bileşen bazlı kırpma `sqrt(2)`
        # kat hızlı uçmaya izin verirdi.
        env.reset(seed=2)
        _, _, _, _, info = env.step(np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))
        horizontal = np.asarray(info["velocity_command_world_m_s"])[:2]

        self.assertAlmostEqual(float(np.linalg.norm(horizontal)), maximum, places=9)

        env.reset(seed=2)
        _, _, _, _, info = env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

        self.assertAlmostEqual(info["yaw_rate_command_rad_s"], 1.5)
        with self.assertRaisesRegex(ValueError, "shape"):
            env.step(np.zeros(3, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "sonlu"):
            env.step(np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32))

    def test_dynamic_horizon_and_time_limit_semantics(self) -> None:
        base = load_vision_navigation_config()
        tiny_budget = replace(
            base.segment_budget,
            minimum_seconds=0.05,
            maximum_seconds=0.05,
        )
        env = self.make_env(config=replace(base, segment_budget=tiny_budget))
        _, reset_info = env.reset(seed=3, options={"planned_route_length_m": 5.0})

        _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))

        self.assertEqual(reset_info["segment_maximum_steps"], 1)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["termination_reason"], "time_limit")
        self.assertEqual(info["remaining_budget_fraction"], 0.0)

    def test_route_length_cannot_be_shorter_than_direct_distance(self) -> None:
        env = self.make_env()

        with self.assertRaisesRegex(ValueError, "düz çizgi"):
            env.reset(seed=4, options={"planned_route_length_m": 3.0})

    def test_same_seed_and_actions_repeat_observation_and_timing(self) -> None:
        first_env = self.make_env(scene_mode="random_obstacles")
        second_env = self.make_env(scene_mode="random_obstacles")
        first, first_info = first_env.reset(seed=5)
        second, second_info = second_env.reset(seed=5)
        actions = (
            np.array([0.4, 0.1, -0.2, 0.15], dtype=np.float32),
            np.array([0.1, -0.1, 0.3, -0.2], dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        )

        for key in VISION_OBSERVATION_KEYS:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(first_info["scene_seed"], second_info["scene_seed"])
        for action in actions:
            first, first_reward, first_terminated, first_truncated, first_info = (
                first_env.step(action)
            )
            second, second_reward, second_terminated, second_truncated, second_info = (
                second_env.step(action)
            )
            for key in VISION_OBSERVATION_KEYS:
                np.testing.assert_array_equal(first[key], second[key])
            self.assertEqual(first_reward, second_reward)
            self.assertEqual(first_terminated, second_terminated)
            self.assertEqual(first_truncated, second_truncated)
            self.assertEqual(
                first_info["camera_frame_timestamp_s"],
                second_info["camera_frame_timestamp_s"],
            )

    def test_gymnasium_checker_and_registered_environment(self) -> None:
        env = self.make_env()
        check_env(env, warn=True, skip_render_check=True)

        registered = gym.make(
            VISION_ENVIRONMENT_ID,
            approval_manifest="synthetic-approval.json",
            camera_factory=lambda _: FakeActorCamera(),
        )
        self.addCleanup(registered.close)
        observation, info = registered.reset(seed=6)

        self.assertTrue(registered.observation_space.contains(observation))
        self.assertEqual(info["observation_version"], VISION_OBSERVATION_VERSION)


if __name__ == "__main__":
    unittest.main()
