"""Gymnasium navigasyon ortamı sözleşme ve entegrasyon testleri."""

from __future__ import annotations

import unittest
from dataclasses import replace

import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

from ebabil.envs import ENVIRONMENT_ID, EbabilNavigationEnv
from ebabil.envs.config import load_navigation_config
from ebabil.scenes import load_scene_config


class NavigationEnvironmentTests(unittest.TestCase):
    def test_stable_baselines_environment_checker(self) -> None:
        env = EbabilNavigationEnv(scene_mode="clear", max_episode_steps=10)
        self.addCleanup(env.close)

        check_env(env, warn=True)

    def test_registered_environment_can_be_created(self) -> None:
        env = gym.make(ENVIRONMENT_ID, scene_mode="clear", max_episode_steps=2)
        self.addCleanup(env.close)

        observation, info = env.reset(seed=5)

        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["observation_version"], "v1")

    def test_same_seed_and_actions_reproduce_rollout(self) -> None:
        first = EbabilNavigationEnv(
            scene_mode="single_blocker", sensor_mode="imperfect"
        )
        second = EbabilNavigationEnv(
            scene_mode="single_blocker", sensor_mode="imperfect"
        )
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        first_observation, first_info = first.reset(seed=42)
        second_observation, second_info = second.reset(seed=42)
        np.testing.assert_array_equal(first_observation, second_observation)
        self.assertEqual(first_info["scene_seed"], second_info["scene_seed"])
        self.assertEqual(first_info["sensor_seed"], second_info["sensor_seed"])
        actions = (
            np.array([0.3, -0.1, 0.1, 0.2], dtype=np.float32),
            np.array([-0.2, 0.4, 0.0, -0.3], dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        )

        for action in actions:
            first_step = first.step(action)
            second_step = second.step(action)
            np.testing.assert_array_equal(first_step[0], second_step[0])
            self.assertEqual(first_step[1], second_step[1])
            self.assertEqual(first_step[2], second_step[2])
            self.assertEqual(first_step[3], second_step[3])
            first_step_info = first_step[4]
            second_step_info = second_step[4]
            self.assertEqual(
                first_step_info["reward_components"],
                second_step_info["reward_components"],
            )
            np.testing.assert_array_equal(
                first_step_info["velocity_command_world_m_s"],
                second_step_info["velocity_command_world_m_s"],
            )
            for key in (
                "distance_to_goal_m",
                "termination_reason",
                "minimum_range_m",
                "yaw_rate_command_rad_s",
            ):
                self.assertEqual(first_step_info[key], second_step_info[key])

    def test_explicit_scene_seed_keeps_scene_but_episode_seed_changes_sensor_noise(
        self,
    ) -> None:
        first = EbabilNavigationEnv(
            scene_mode="single_blocker", sensor_mode="imperfect"
        )
        second = EbabilNavigationEnv(
            scene_mode="single_blocker", sensor_mode="imperfect"
        )
        self.addCleanup(first.close)
        self.addCleanup(second.close)

        first_observation, first_info = first.reset(seed=1, options={"scene_seed": 42})
        second_observation, second_info = second.reset(
            seed=999, options={"scene_seed": 42}
        )

        self.assertEqual(first_info["scene_seed"], second_info["scene_seed"])
        self.assertNotEqual(first_info["sensor_seed"], second_info["sensor_seed"])
        self.assertFalse(np.array_equal(first_observation, second_observation))

    def test_explicit_component_seeds_reproduce_imperfect_observation(self) -> None:
        first = EbabilNavigationEnv(
            scene_mode="single_blocker", sensor_mode="imperfect"
        )
        second = EbabilNavigationEnv(
            scene_mode="single_blocker", sensor_mode="imperfect"
        )
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        options = {"scene_seed": 42, "sensor_seed": 84}

        first_observation, first_info = first.reset(seed=1, options=options)
        second_observation, second_info = second.reset(seed=999, options=options)

        np.testing.assert_array_equal(first_observation, second_observation)
        self.assertEqual(first_info["sensor_seed"], 84)
        self.assertEqual(second_info["sensor_seed"], 84)

    def test_endpoint_mode_can_be_selected_per_reset(self) -> None:
        env = EbabilNavigationEnv(scene_mode="clear", endpoint_mode="fixed")
        self.addCleanup(env.close)

        _, info = env.reset(
            seed=5,
            options={"scene_seed": 77, "endpoint_mode": "random_both"},
        )

        assert env.scene is not None
        self.assertEqual(info["endpoint_mode"], "random_both")
        self.assertFalse(
            np.array_equal(env.scene.start_world_m, env.scene_config.start_world_m)
        )
        self.assertFalse(
            np.array_equal(env.scene.goal_world_m, env.scene_config.goal_world_m)
        )

    def test_component_seeds_must_be_non_negative_integers(self) -> None:
        env = EbabilNavigationEnv(scene_mode="clear")
        self.addCleanup(env.close)

        for options in (
            {"scene_seed": -1},
            {"scene_seed": 1.5},
            {"sensor_seed": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                env.reset(seed=1, options=options)

    def test_reward_components_sum_to_reward(self) -> None:
        env = EbabilNavigationEnv(scene_mode="clear")
        self.addCleanup(env.close)
        env.reset(seed=7)

        _, reward, _, _, info = env.step(np.zeros(4, dtype=np.float32))

        self.assertAlmostEqual(reward, sum(info["reward_components"].values()))
        self.assertEqual(
            set(info["reward_components"]),
            {
                "progress",
                "goal",
                "collision",
                "failure",
                "proximity",
                "smoothness",
                "attitude",
                "time",
            },
        )

    def test_action_is_clipped_and_invalid_values_are_rejected(self) -> None:
        env = EbabilNavigationEnv(scene_mode="clear")
        self.addCleanup(env.close)
        env.reset(seed=3)

        _, _, _, _, info = env.step(np.array([2.0, -2.0, 0.0, 0.0], dtype=np.float32))

        self.assertTrue(info["action_clipped"])
        self.assertAlmostEqual(
            float(np.linalg.norm(info["velocity_command_world_m_s"][:2])),
            np.sqrt(2.0) * 1.5,
            places=10,
        )
        with self.assertRaises(ValueError):
            env.step(np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32))
        with self.assertRaises(ValueError):
            env.step(np.zeros(3, dtype=np.float32))

    def test_time_limit_uses_truncated_not_terminated(self) -> None:
        env = EbabilNavigationEnv(scene_mode="clear", max_episode_steps=1)
        self.addCleanup(env.close)
        env.reset(seed=8)

        _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["termination_reason"], "time_limit")

    def test_invalid_episode_step_limits_are_rejected(self) -> None:
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                EbabilNavigationEnv(max_episode_steps=value)  # type: ignore[arg-type]

    def test_incompatible_policy_frequency_is_rejected_during_construction(
        self,
    ) -> None:
        config = load_navigation_config()
        incompatible = replace(
            config,
            simulation=replace(config.simulation, policy_frequency_hz=30.0),
        )

        with self.assertRaisesRegex(ValueError, "tam katı"):
            EbabilNavigationEnv(config=incompatible)

    def test_simple_goal_controller_reaches_goal_in_clear_scene(self) -> None:
        env = EbabilNavigationEnv(scene_mode="clear")
        self.addCleanup(env.close)
        env.reset(seed=1, options={"scene_seed": 1})

        terminated = truncated = False
        info: dict[str, object] = {}
        for _ in range(150):
            assert env.scene is not None
            assert env.data is not None
            delta = env.scene.goal_world_m - env.data.qpos[:3]
            action = np.array(
                [
                    np.clip(0.4 * delta[0], -1.0, 1.0),
                    np.clip(0.4 * delta[1], -1.0, 1.0),
                    np.clip(0.6 * delta[2], -1.0, 1.0),
                    0.0,
                ],
                dtype=np.float32,
            )
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["is_success"])
        self.assertEqual(info["termination_reason"], "goal")
        self.assertEqual(info["goal_hold_steps"], info["goal_hold_required_steps"])

    def test_goal_requires_continuous_low_speed_hold(self) -> None:
        scene_config = replace(
            load_scene_config(),
            start_world_m=np.array([1.9, 0.0, 1.0]),
            goal_world_m=np.array([2.0, 0.0, 1.0]),
        )
        env = EbabilNavigationEnv(scene_mode="clear", scene_config=scene_config)
        self.addCleanup(env.close)
        env.reset(seed=10)
        required_steps = round(
            env.config.termination.goal_hold_seconds
            * env.config.simulation.policy_frequency_hz
        )

        for expected_step in range(1, required_steps):
            _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["goal_hold_steps"], expected_step)
            self.assertEqual(info["reward_components"]["goal"], 0.0)

        _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["is_success"])
        self.assertEqual(info["termination_reason"], "goal")
        self.assertGreater(info["reward_components"]["goal"], 0.0)

    def test_direct_goal_controller_collides_with_single_blocker(self) -> None:
        env = EbabilNavigationEnv(scene_mode="single_blocker")
        self.addCleanup(env.close)
        env.reset(seed=1, options={"scene_seed": 20260910})

        terminated = truncated = False
        info: dict[str, object] = {}
        for _ in range(100):
            assert env.scene is not None
            assert env.data is not None
            delta_x = env.scene.goal_world_m[0] - env.data.qpos[0]
            action = np.array(
                [np.clip(0.4 * delta_x, -1.0, 1.0), 0.0, 0.0, 0.0],
                dtype=np.float32,
            )
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["is_success"])
        self.assertEqual(info["termination_reason"], "obstacle_collision")
        self.assertLess(info["reward_components"]["collision"], 0.0)

    def test_random_action_rollout_remains_finite(self) -> None:
        env = EbabilNavigationEnv(scene_mode="random_obstacles", max_episode_steps=30)
        self.addCleanup(env.close)
        observation, _ = env.reset(seed=20260910)
        rng = np.random.default_rng(20260910)

        for episode_step in range(100):
            action = rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
            observation, reward, terminated, truncated, _ = env.step(action)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertTrue(np.isfinite(reward))
            if terminated or truncated:
                observation, _ = env.reset(seed=20260910 + episode_step + 1)


if __name__ == "__main__":
    unittest.main()
