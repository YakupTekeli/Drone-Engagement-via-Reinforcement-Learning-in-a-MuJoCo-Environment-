"""PPO config'i ve vektör environment fabrikası testleri."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

from ebabil.evaluation import evaluate_policy, load_replay
from ebabil.scenes import load_target_area_config
from ebabil.training import load_training_config, make_vector_env, train_ppo
from ebabil.training.config import EnvironmentConfig, require_legacy_environment
from ebabil.training.env_factory import MONITOR_INFO_KEYS, monitor_info_keys

PACKAGE_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "src" / "ebabil" / "configs"


class TrainingConfigurationTests(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        config = load_training_config()

        self.assertEqual(config.environment.scene_mode, "clear")
        self.assertEqual(config.environment.endpoint_mode, "fixed")
        self.assertEqual(config.environment.sensor_mode, "ideal")
        self.assertEqual(config.run.n_envs, 4)
        self.assertFalse(config.normalization.observations)
        self.assertTrue(config.normalization.rewards)

    def test_rollout_size_must_be_divisible_by_batch_size(self) -> None:
        config = load_training_config()
        invalid = replace(config, ppo=replace(config.ppo, batch_size=257))

        with self.assertRaisesRegex(ValueError, "kalansız"):
            invalid.validate()


class VectorEnvironmentTests(unittest.TestCase):
    def test_dummy_vector_environment_resets_with_independent_seeds(self) -> None:
        config = load_training_config()
        with tempfile.TemporaryDirectory() as directory:
            env = make_vector_env(
                config,
                seed=41,
                n_envs=2,
                vector_backend="dummy",
                monitor_dir=Path(directory),
            )
            self.addCleanup(env.close)

            observations = env.reset()
            base_env = env.venv if isinstance(env, VecNormalize) else env
            scene_seeds = base_env.get_attr("_scene_seed")
            sensor_seeds = base_env.get_attr("_sensor_seed")

            self.assertEqual(observations.shape, (2, 30))
            self.assertTrue(np.all(np.isfinite(observations)))
            self.assertNotEqual(scene_seeds[0], scene_seeds[1])
            self.assertNotEqual(sensor_seeds[0], sensor_seeds[1])

    def test_invalid_factory_arguments_are_rejected(self) -> None:
        config = load_training_config()

        with self.assertRaises(ValueError):
            make_vector_env(config, n_envs=0)
        with self.assertRaises(ValueError):
            make_vector_env(config, seed=-1)

    def test_subprocess_vector_environment_runs_multiple_instances(self) -> None:
        config = load_training_config()
        env = make_vector_env(
            config,
            seed=51,
            n_envs=2,
            vector_backend="subproc",
            training=False,
        )
        self.addCleanup(env.close)

        observations = env.reset()

        self.assertEqual(observations.shape, (2, 30))
        self.assertTrue(np.all(np.isfinite(observations)))


class TrainingRunnerTests(unittest.TestCase):
    def test_runtime_failure_is_recorded_in_metadata(self) -> None:
        config = load_training_config()
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory) / "failed"
            with (
                patch(
                    "ebabil.training.runner.make_vector_env",
                    side_effect=RuntimeError("simulated failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "simulated failure"),
            ):
                train_ppo(config, run_path, device="cpu")

            metadata = json.loads((run_path / "metadata.json").read_text())

        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["error"]["type"], "RuntimeError")

    def test_short_training_writes_reloadable_artifacts(self) -> None:
        base = load_training_config()
        config = replace(
            base,
            run=replace(
                base.run,
                total_timesteps=8,
                n_envs=1,
                vector_backend="dummy",
            ),
            environment=replace(base.environment, max_episode_steps=4),
            ppo=replace(
                base.ppo,
                n_steps=8,
                batch_size=8,
                n_epochs=1,
                network_arch=(16,),
            ),
            checkpoint=replace(
                base.checkpoint,
                save_frequency_timesteps=8,
                evaluation_frequency_timesteps=8,
                evaluation_episodes=1,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            artifacts = train_ppo(config, Path(directory) / "run", device="cpu")

            self.assertTrue(artifacts.final_model.is_file())
            self.assertIsNotNone(artifacts.best_model)
            assert artifacts.best_model is not None
            self.assertTrue(artifacts.best_model.is_file())
            self.assertIsNotNone(artifacts.best_normalization)
            assert artifacts.best_normalization is not None
            self.assertTrue(artifacts.best_normalization.is_file())
            self.assertIsNotNone(artifacts.normalization)
            assert artifacts.normalization is not None
            self.assertTrue(artifacts.normalization.is_file())
            self.assertTrue(artifacts.metadata.is_file())
            self.assertTrue(artifacts.progress_csv.is_file())
            self.assertTrue(any(artifacts.progress_csv.parent.glob("events.out.*")))
            self.assertEqual(artifacts.actual_total_timesteps, 8)

            first_evaluation = evaluate_policy(
                artifacts.final_model,
                config,
                Path(directory) / "evaluation_first",
                episodes=2,
                seed=123,
                record_replays=1,
                device="cpu",
            )
            second_evaluation = evaluate_policy(
                artifacts.final_model,
                config,
                Path(directory) / "evaluation_second",
                episodes=2,
                seed=123,
                record_replays=1,
                device="cpu",
            )

            self.assertEqual(first_evaluation.metrics, second_evaluation.metrics)
            first_replay, first_metadata = load_replay(first_evaluation.replay_files[0])
            second_replay, second_metadata = load_replay(
                second_evaluation.replay_files[0]
            )
            np.testing.assert_array_equal(first_replay.qpos, second_replay.qpos)
            np.testing.assert_array_equal(first_replay.actions, second_replay.actions)
            self.assertEqual(first_metadata, second_metadata)
            self.assertEqual(first_replay.time_s.size, 101)
            self.assertEqual(first_replay.actions.shape[0], 4)
            self.assertEqual(first_replay.rotor_thrusts_n.shape, (100, 4))


if __name__ == "__main__":
    unittest.main()


class VisionEnvironmentConfigTests(unittest.TestCase):
    """D0-V: eğitim altyapısının vision ortamına bağlanması."""

    def setUp(self) -> None:
        self.config = load_training_config(
            PACKAGE_CONFIG_ROOT / "training" / "ppo_vision_d0v.toml"
        )

    def test_d0v_config_selects_vision_environment(self) -> None:
        self.config.validate()

        self.assertEqual(self.config.environment.kind, "vision")
        self.assertEqual(self.config.environment.scene_mode, "clear")
        self.assertIsNotNone(self.config.environment.approval_manifest)

    def test_d0v_uses_fixed_endpoints_by_adr_010(self) -> None:
        """ADR-010: D0-V legacy geometridir; değişken endpoint D2'dedir."""

        self.assertEqual(self.config.environment.endpoint_mode, "fixed")

    def test_vision_observations_are_not_renormalized(self) -> None:
        """vision-v1 zaten [-1, 1]; tekrar normalize etmek latent ölçeğini bozar."""

        self.assertFalse(self.config.normalization.observations)

    def test_vision_rejects_legacy_only_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "sensor_mode verilemez"):
            replace(self.config.environment, sensor_mode="ideal").validate()
        with self.assertRaisesRegex(ValueError, "max_episode_steps verilemez"):
            replace(self.config.environment, max_episode_steps=400).validate()

    def test_vision_requires_approval_manifest(self) -> None:
        with self.assertRaisesRegex(ValueError, "approval_manifest gerektirir"):
            replace(self.config.environment, approval_manifest=None).validate()

    def test_legacy_rejects_approval_manifest(self) -> None:
        legacy = load_training_config().environment

        with self.assertRaisesRegex(ValueError, "yalnız vision ortamında"):
            replace(legacy, approval_manifest="x.json").validate()

    def test_legacy_config_still_requires_its_own_fields(self) -> None:
        legacy = load_training_config().environment
        legacy.validate()

        self.assertEqual(legacy.kind, "legacy")
        with self.assertRaisesRegex(ValueError, "sensor_mode"):
            replace(legacy, sensor_mode=None).validate()
        with self.assertRaisesRegex(ValueError, "max_episode_steps"):
            replace(legacy, max_episode_steps=None).validate()

    def test_monitor_keys_include_terminal_hover_metrics(self) -> None:
        vision_keys = monitor_info_keys(self.config)
        legacy_keys = monitor_info_keys(load_training_config())

        self.assertIn("hover_state", vision_keys)
        self.assertIn("hover_abort_reason", vision_keys)
        self.assertIn("hover_hold_seconds", vision_keys)
        self.assertNotIn("hover_state", legacy_keys)
        self.assertEqual(legacy_keys, MONITOR_INFO_KEYS)

    def test_legacy_evaluation_refuses_vision_config(self) -> None:
        """Vision ortamı legacy evaluation akışına sessizce düşmemeli."""

        with self.assertRaisesRegex(ValueError, "yalnız legacy"):
            require_legacy_environment(self.config)

        sensor_mode, max_steps = require_legacy_environment(load_training_config())
        self.assertEqual(sensor_mode, "ideal")
        self.assertEqual(max_steps, 400)


class HoldLadderConfigTests(unittest.TestCase):
    """REV-045: hold süresi kademelendirmesi yalnız eğitim protokolüdür."""

    LADDER = (("hold05", 0.5), ("hold10", 1.0), ("hold20", 2.0))

    def test_gate_config_keeps_three_seconds(self) -> None:
        """D1 kapısı `hold_seconds = 3.0` ile ölçülür; varsayılan değişmez."""

        self.assertEqual(load_target_area_config().hold_seconds, 3.0)

    def test_ladder_changes_only_hold_seconds(self) -> None:
        """Kademeler arası fark tek değişkenle sınırlı kalmalı."""

        default = load_target_area_config()
        for label, seconds in self.LADDER:
            with self.subTest(label=label):
                ladder = load_target_area_config(
                    f"src/ebabil/configs/environment/target_area_{label}.toml"
                )
                ladder.validate()

                self.assertEqual(ladder.hold_seconds, seconds)
                self.assertLess(ladder.hold_seconds, default.hold_seconds)
                self.assertEqual(
                    replace(ladder, hold_seconds=default.hold_seconds), default
                )

    def test_target_config_path_requires_vision_kind(self) -> None:
        """Legacy ortamda hedef alan config'i anlamsızdır; reddedilir."""

        with self.assertRaisesRegex(ValueError, "target_config_path"):
            EnvironmentConfig(
                kind="legacy",
                scene_mode="clear",
                endpoint_mode="fixed",
                sensor_mode="ideal",
                max_episode_steps=100,
                target_config_path="src/ebabil/configs/environment/target_area.toml",
            ).validate()

    def test_empty_target_config_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_config_path"):
            EnvironmentConfig(
                kind="vision",
                scene_mode="clear",
                endpoint_mode="fixed",
                approval_manifest="x.json",
                target_config_path="",
            ).validate()
