"""ADR-022 `vision-v3` ortamı: etiketsiz latent yığını, DR ve privileged kapı."""

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

from ebabil.envs import (
    DEFAULT_LATENT_VISION_CONFIG,
    LATENT_VISION_ENVIRONMENT_ID,
    LATENT_VISION_OBSERVATION_KEYS,
    LATENT_VISION_OBSERVATION_VERSION,
    EbabilLatentVisionNavigationEnv,
    load_vision_navigation_config,
)
from ebabil.envs.latent_vision import RandomizedActorCamera
from ebabil.perception import (
    ActorCameraFrame,
    FrameVisualParameters,
    TargetVisibility,
    VisualVAE,
    apply_frame_visual_parameters,
    apply_frame_visual_parameters_torch,
    load_camera_config,
    load_visual_vae_config,
    make_noise_bank,
    preprocess_actor_rgb,
)
from ebabil.scenes import load_scene_config
from ebabil.training.config import load_training_config
from ebabil.training.env_factory import monitor_info_keys

D1A_SCENE = Path("src/ebabil/configs/environment/static_scene_d1a.toml")


class FakeActorCamera:
    def __init__(self) -> None:
        config = load_camera_config()
        self.shape = (config.output_height_px, config.output_width_px, 3)
        self.closed = False

    def capture_actor(self, data: mujoco.MjData) -> ActorCameraFrame:
        timestamp_s = float(data.time)
        value = 40 + int(round(timestamp_s * 1000.0)) % 160
        rgb = np.full(self.shape, value, dtype=np.uint8)
        rgb[: self.shape[0] // 2, :, 1] = 200
        return ActorCameraFrame(rgb=rgb, timestamp_s=timestamp_s)

    def close(self) -> None:
        self.closed = True


class FakeTargetObserver:
    def __init__(self) -> None:
        self.visible = False
        self.pixels = 64
        self.closed = False

    def observe(self, data: mujoco.MjData) -> TargetVisibility:
        return TargetVisibility(
            visible=self.visible,
            pixel_count=self.pixels if self.visible else 0,
            center_x=0.25 if self.visible else 0.0,
            center_y=0.0,
            timestamp_s=float(data.time),
        )

    def close(self) -> None:
        self.closed = True


def _approved_vae(_: Path) -> VisualVAE:
    torch.manual_seed(20260923)
    config, _ = load_visual_vae_config()
    return VisualVAE(config).eval()


class LatentVisionEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        loader = patch(
            "ebabil.envs.latent_vision.load_approved_visual_vae",
            side_effect=_approved_vae,
        )
        loader.start()
        self.addCleanup(loader.stop)

    def make_env(
        self, *, randomize: bool = True, **kwargs: object
    ) -> tuple[EbabilLatentVisionNavigationEnv, FakeTargetObserver]:
        config = load_vision_navigation_config(DEFAULT_LATENT_VISION_CONFIG)
        if not randomize:
            config = replace(
                config,
                visual_randomization=replace(
                    config.visual_randomization, enabled=False
                ),
            )
        observer = FakeTargetObserver()
        env = EbabilLatentVisionNavigationEnv(
            approval_manifest="synthetic-vae.json",
            config=config,
            camera_factory=lambda _: FakeActorCamera(),
            target_observer_factory=lambda _: observer,
            **kwargs,  # type: ignore[arg-type]
        )
        self.addCleanup(env.close)
        return env, observer

    def test_v3_config_and_v0_defaults(self) -> None:
        v3 = load_vision_navigation_config(DEFAULT_LATENT_VISION_CONFIG)
        v0 = load_vision_navigation_config()

        self.assertEqual(v3.observation.latent_size, 128)
        self.assertEqual(v3.observation.latent_stack_size, 4)
        self.assertTrue(v3.visual_randomization.enabled)
        self.assertEqual(v3.visual_randomization.split, "train")
        self.assertEqual(v3.privileged_visibility.minimum_target_pixels, 8)
        # Reward ve hover sözleşmesi v0 ile birebir aynı kalmalı.
        self.assertEqual(v3.terminal_hover_reward, v0.terminal_hover_reward)
        self.assertEqual(v3.terminal_hover, v0.terminal_hover)
        self.assertEqual(v3.navigation.action, v0.navigation.action)
        self.assertEqual(v0.observation.latent_stack_size, 1)
        self.assertFalse(v0.visual_randomization.enabled)
        with self.assertRaisesRegex(ValueError, "latent_stack_size"):
            replace(v3.observation, latent_stack_size=0).validate()
        with self.assertRaisesRegex(ValueError, "split"):
            replace(v3.visual_randomization, split="test").validate()

    def test_contract_keys_shapes_bounds_and_version(self) -> None:
        env, _ = self.make_env(scene_mode="random_obstacles")
        observation, info = env.reset(seed=10, options={"scene_seed": 11})

        self.assertEqual(tuple(sorted(observation)), LATENT_VISION_OBSERVATION_KEYS)
        self.assertEqual(
            tuple(sorted(env.observation_space.spaces)), LATENT_VISION_OBSERVATION_KEYS
        )
        self.assertEqual(observation["vision_latent"].shape, (4 * 128,))
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["observation_version"], LATENT_VISION_OBSERVATION_VERSION)
        self.assertEqual(env.observation_version, LATENT_VISION_OBSERVATION_VERSION)
        self.assertEqual(env.action_space.shape, (4,))
        self.assertNotIn("vision_aux", observation)
        self.assertNotIn("obstacle_map", observation)
        prohibited = (
            "depth",
            "ray",
            "range",
            "obstacle",
            "segmentation",
            "qpos",
            "qvel",
            "world",
            "target",
            "score",
        )
        self.assertFalse(
            any(token in key for key in observation for token in prohibited)
        )
        self.assertFalse(env.vae.training)
        self.assertTrue(all(not p.requires_grad for p in env.vae.parameters()))
        self.assertEqual(ActorCameraFrame.field_names(), ("rgb", "timestamp_s"))

    def test_latent_is_frozen_vae_mean_and_stack_replicates_on_reset(self) -> None:
        env, _ = self.make_env(randomize=False)
        observation, _ = env.reset(seed=12)
        assert env.data is not None
        frame = FakeActorCamera().capture_actor(env.data)
        with torch.no_grad():
            mean, _ = env.vae.encode(preprocess_actor_rgb(frame.rgb).unsqueeze(0))
        expected = np.clip(mean[0].numpy() / 5.0, -1.0, 1.0)

        stack = observation["vision_latent"].reshape(4, 128)
        for row in range(4):
            np.testing.assert_allclose(stack[row], expected, atol=1e-6)

    def test_stack_shifts_newest_first_each_step(self) -> None:
        env, _ = self.make_env(randomize=False)
        first, _ = env.reset(seed=13)
        first_stack = first["vision_latent"].reshape(4, 128).copy()

        second, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        second_stack = second["vision_latent"].reshape(4, 128)
        current = np.clip(env._latent_current / 5.0, -1.0, 1.0)

        np.testing.assert_allclose(second_stack[0], current, atol=1e-6)
        np.testing.assert_array_equal(second_stack[1], first_stack[0])
        np.testing.assert_array_equal(second_stack[2], first_stack[1])
        self.assertFalse(np.allclose(second_stack[0], second_stack[1]))

    def test_actor_path_has_no_privileged_capture(self) -> None:
        env, observer = self.make_env(randomize=True)
        env.reset(seed=14)

        camera = env._camera
        assert camera is not None
        self.assertIsInstance(camera, RandomizedActorCamera)
        self.assertFalse(hasattr(camera, "capture_privileged"))
        self.assertFalse(hasattr(camera, "observe"))
        self.assertIs(env._target_observer, observer)
        self.assertIsNot(env._target_observer, camera)

    def test_visual_randomization_is_deterministic_per_seed(self) -> None:
        scene_config = load_scene_config(D1A_SCENE)
        first, _ = self.make_env(
            scene_mode="random_obstacles", scene_config=scene_config
        )
        second, _ = self.make_env(
            scene_mode="random_obstacles", scene_config=scene_config
        )
        other, _ = self.make_env(
            scene_mode="random_obstacles", scene_config=scene_config
        )
        plain, _ = self.make_env(
            randomize=False, scene_mode="random_obstacles", scene_config=scene_config
        )
        first_obs, first_info = first.reset(seed=15, options={"scene_seed": 16})
        second_obs, _ = second.reset(seed=15, options={"scene_seed": 16})
        other.reset(seed=15, options={"scene_seed": 17})
        plain.reset(seed=15, options={"scene_seed": 16})
        assert first.model is not None and second.model is not None
        assert other.model is not None and plain.model is not None

        np.testing.assert_array_equal(
            first_obs["vision_latent"], second_obs["vision_latent"]
        )
        np.testing.assert_array_equal(first.model.geom_rgba, second.model.geom_rgba)
        assert first.data is not None and second.data is not None
        assert first._camera is not None and second._camera is not None
        np.testing.assert_array_equal(
            first._camera.capture_actor(first.data).rgb,
            second._camera.capture_actor(second.data).rgb,
        )
        obstacle_ids = [
            index
            for index in range(first.model.ngeom)
            if first.model.geom(index).name.startswith("obstacle_")
        ]
        self.assertGreater(len(obstacle_ids), 0)
        self.assertFalse(
            np.allclose(
                first.model.geom_rgba[obstacle_ids][:, :3],
                other.model.geom_rgba[obstacle_ids][:, :3],
            )
        )
        np.testing.assert_allclose(
            plain.model.geom_rgba[obstacle_ids][:, :3],
            np.tile([0.55, 0.22, 0.12], (len(obstacle_ids), 1)),
        )
        self.assertEqual(first_info["visual_domain"], "train_broad_randomized")
        self.assertIsNotNone(first_info["scene_visual_parameters"])
        # Aynı seed aynı bayt dizisi: fotometrik etkiler kare indeksine bağlı,
        # farklı kareler farklı gürültü alır.
        assert first.data is not None
        frame_a = first._camera.capture_actor(first.data).rgb
        frame_b = first._camera.capture_actor(first.data).rgb
        self.assertFalse(np.array_equal(frame_a, frame_b))

    def test_obstacle_geometries_follow_scene_config(self) -> None:
        d1a, _ = self.make_env(
            randomize=False,
            scene_mode="random_obstacles",
            scene_config=load_scene_config(D1A_SCENE),
        )
        legacy, _ = self.make_env(randomize=False, scene_mode="random_obstacles")
        _, d1a_info = d1a.reset(seed=18, options={"scene_seed": 19})
        _, legacy_info = legacy.reset(seed=18, options={"scene_seed": 19})
        assert d1a.scene is not None and legacy.scene is not None

        self.assertTrue(
            {o.geometry for o in d1a.scene.obstacles} <= {"box", "cylinder", "capsule"}
        )
        self.assertEqual(
            d1a_info["obstacle_geometries"],
            sorted({o.geometry for o in d1a.scene.obstacles}),
        )
        self.assertEqual({o.geometry for o in legacy.scene.obstacles}, {"box"})
        # Model geometri türleri sahneyle tutarlı olmalı.
        assert d1a.model is not None
        expected_types = {
            "box": mujoco.mjtGeom.mjGEOM_BOX,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
            "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
        }
        for obstacle in d1a.scene.obstacles:
            geom = d1a.model.geom(obstacle.name)
            self.assertEqual(int(geom.type[0]), int(expected_types[obstacle.geometry]))

    def test_privileged_visibility_gates_the_hover_state_machine(self) -> None:
        env, observer = self.make_env(randomize=False)
        _, info = env.reset(seed=20)
        self.assertFalse(info["target_visible_privileged"])
        self.assertEqual(info["hover_state"], "transit")
        action = np.zeros(4, dtype=np.float32)

        for _ in range(3):
            _, _, _, _, info = env.step(action)
        self.assertIn(info["hover_state"], ("transit", "search"))
        self.assertFalse(info["hover_target_visible"])

        observer.visible = True
        for _ in range(7):
            _, _, terminated, _, info = env.step(action)
        self.assertTrue(info["hover_target_visible"])
        self.assertTrue(info["target_visible_privileged"])
        self.assertEqual(info["target_pixel_count"], 64)
        self.assertIn(info["hover_state"], ("acquired", "approach", "settle", "hold"))
        self.assertIn("hover_horizontal_distance_m", info)
        self.assertFalse(terminated)

        observer.visible = False
        for _ in range(60):
            _, _, terminated, _, info = env.step(action)
            if terminated:
                break
        self.assertTrue(terminated)
        self.assertEqual(info["termination_reason"], "hover_abort_target_lost")

    def test_same_seed_and_actions_repeat_observations_and_rewards(self) -> None:
        first, _ = self.make_env(scene_mode="random_obstacles")
        second, _ = self.make_env(scene_mode="random_obstacles")
        first_obs, _ = first.reset(seed=21)
        second_obs, _ = second.reset(seed=21)
        np.testing.assert_array_equal(
            first_obs["vision_latent"], second_obs["vision_latent"]
        )
        for action in (
            np.array([0.4, 0.1, -0.2, 0.15], dtype=np.float32),
            np.array([0.1, -0.1, 0.3, -0.2], dtype=np.float32),
        ):
            first_obs, first_reward, *_ = first.step(action)
            second_obs, second_reward, *_ = second.step(action)
            for key in LATENT_VISION_OBSERVATION_KEYS:
                np.testing.assert_array_equal(first_obs[key], second_obs[key])
            self.assertEqual(first_reward, second_reward)

    def test_visual_domain_split_override_and_rejection(self) -> None:
        env, _ = self.make_env(randomize=True, visual_domain_split="validation")
        _, info = env.reset(seed=22)

        self.assertEqual(info["visual_domain"], "heldout_dim_cool")
        with self.assertRaisesRegex(ValueError, "randomization kapalı"):
            self.make_env(randomize=False, visual_domain_split="validation")

    def test_gymnasium_checker_and_registered_environment(self) -> None:
        env, _ = self.make_env()
        check_env(env, warn=True, skip_render_check=True)

        observer = FakeTargetObserver()
        registered = gym.make(
            LATENT_VISION_ENVIRONMENT_ID,
            approval_manifest="synthetic-vae.json",
            camera_factory=lambda _: FakeActorCamera(),
            target_observer_factory=lambda _: observer,
        )
        self.addCleanup(registered.close)
        observation, info = registered.reset(seed=23)

        self.assertTrue(registered.observation_space.contains(observation))
        self.assertEqual(info["observation_version"], LATENT_VISION_OBSERVATION_VERSION)
        self.assertIn("latent_clip_fraction", info)

    def test_latent_size_mismatch_is_rejected(self) -> None:
        config = load_vision_navigation_config(DEFAULT_LATENT_VISION_CONFIG)
        mismatched = replace(
            config, observation=replace(config.observation, latent_size=64)
        )
        with self.assertRaisesRegex(ValueError, "latent boyutu"):
            EbabilLatentVisionNavigationEnv(
                approval_manifest="synthetic-vae.json",
                config=mismatched,
                camera_factory=lambda _: FakeActorCamera(),
                target_observer_factory=lambda _: FakeTargetObserver(),
            )


class PhotometricTorchParityTests(unittest.TestCase):
    def test_torch_matches_numpy_without_noise(self) -> None:
        rng = np.random.default_rng(3)
        rgb = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
        for blur, tolerance in ((False, 1), (True, 2)):
            with self.subTest(blur=blur):
                parameters = FrameVisualParameters(
                    exposure_gain=1.2,
                    gamma=0.9,
                    noise_std=0.0,
                    vignette_strength=0.2,
                    apply_blur=blur,
                )
                reference = apply_frame_visual_parameters(rgb, parameters, noise_seed=0)
                fast = apply_frame_visual_parameters_torch(
                    rgb, parameters, noise_seed=0
                )
                self.assertEqual(fast.shape, rgb.shape)
                self.assertEqual(fast.dtype, np.uint8)
                self.assertLessEqual(
                    int(np.max(np.abs(fast.astype(int) - reference.astype(int)))),
                    tolerance,
                )

    def test_noise_is_seeded_and_bounded(self) -> None:
        rgb = np.full((24, 32, 3), 128, dtype=np.uint8)
        parameters = FrameVisualParameters(1.0, 1.0, 0.03, 0.0, False)
        first = apply_frame_visual_parameters_torch(rgb, parameters, noise_seed=7)
        second = apply_frame_visual_parameters_torch(rgb, parameters, noise_seed=7)
        third = apply_frame_visual_parameters_torch(rgb, parameters, noise_seed=8)

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, third))
        self.assertLess(abs(float(first.mean()) - 128.0), 2.0)
        with self.assertRaisesRegex(ValueError, "uint8"):
            apply_frame_visual_parameters_torch(
                rgb.astype(np.float32), parameters, noise_seed=1
            )

    def test_noise_bank_path_is_seeded_and_validated(self) -> None:
        rgb = np.full((24, 32, 3), 128, dtype=np.uint8)
        parameters = FrameVisualParameters(1.0, 1.0, 0.03, 0.0, False)
        bank = make_noise_bank(24, 32, 4, seed=11)
        same_bank = make_noise_bank(24, 32, 4, seed=11)
        other_bank = make_noise_bank(24, 32, 4, seed=12)

        self.assertEqual(tuple(bank.shape), (4, 24, 32, 3))
        torch.testing.assert_close(bank, same_bank)
        self.assertFalse(torch.equal(bank, other_bank))
        self.assertLess(abs(float(bank.mean())), 0.05)
        self.assertLess(abs(float(bank.std()) - 1.0), 0.05)

        first = apply_frame_visual_parameters_torch(
            rgb, parameters, noise_seed=7, noise_bank=bank
        )
        second = apply_frame_visual_parameters_torch(
            rgb, parameters, noise_seed=7, noise_bank=same_bank
        )
        third = apply_frame_visual_parameters_torch(
            rgb, parameters, noise_seed=8, noise_bank=bank
        )
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, third))
        self.assertLess(abs(float(first.mean()) - 128.0), 2.0)
        with self.assertRaisesRegex(ValueError, "noise_bank"):
            apply_frame_visual_parameters_torch(
                rgb,
                parameters,
                noise_seed=1,
                noise_bank=make_noise_bank(8, 8, 2, seed=1),
            )
        with self.assertRaisesRegex(ValueError, "pozitif"):
            make_noise_bank(24, 32, 0, seed=1)


class LatentVisionTrainingConfigTests(unittest.TestCase):
    def test_latent_vision_kind_and_split(self) -> None:
        config = load_training_config("src/ebabil/configs/training/ppo_latent_d1a.toml")
        reference = load_training_config(
            "src/ebabil/configs/training/ppo_vision_d1a.toml"
        )

        self.assertEqual(config.environment.kind, "latent_vision")
        self.assertEqual(config.environment.visual_domain_split, "train")
        self.assertEqual(config.ppo, reference.ppo)
        self.assertEqual(config.run, reference.run)
        self.assertFalse(config.normalization.observations)
        self.assertIn("hover_horizontal_distance_m", monitor_info_keys(config))
        with self.assertRaisesRegex(ValueError, "visual_domain_split"):
            replace(reference.environment, visual_domain_split="train").validate()
        legacy = load_training_config().environment
        with self.assertRaisesRegex(ValueError, "visual_domain_split"):
            replace(legacy, visual_domain_split="train").validate()


if __name__ == "__main__":
    unittest.main()
