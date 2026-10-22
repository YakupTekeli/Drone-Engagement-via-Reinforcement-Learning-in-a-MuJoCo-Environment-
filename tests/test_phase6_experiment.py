"""Faz 6 deney orkestrasyonu smoke testi."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ebabil.evaluation import load_phase6_protocol
from ebabil.experiments import run_phase6_experiment
from ebabil.training import load_training_config


class Phase6ExperimentTests(unittest.TestCase):
    def test_short_experiment_writes_aggregate_summary(self) -> None:
        base = load_training_config()
        protocol = load_phase6_protocol()
        config = replace(
            base,
            run=replace(
                base.run,
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
        protocol = replace(protocol, pilot_timesteps=8)
        with tempfile.TemporaryDirectory() as directory:
            artifacts = run_phase6_experiment(
                config,
                protocol,
                Path(directory) / "experiment",
                stage="pilot",
                training_seeds=protocol.training_seeds[:1],
                held_out_episode_seeds=protocol.held_out_episode_seeds[:2],
                record_replays_per_seed=1,
                device="cpu",
            )

            self.assertTrue(artifacts.summary.is_file())
            self.assertEqual(len(artifacts.metrics["ppo_seeds"]), 1)
            self.assertEqual(artifacts.metrics["status"], "completed")
            self.assertIn("random", artifacts.metrics["baselines"])
            self.assertIn("direct_goal", artifacts.metrics["baselines"])
            self.assertFalse(artifacts.metrics["quality_gate_passed"])
            self.assertFalse(artifacts.metrics["protocol_coverage"]["full_stage"])


if __name__ == "__main__":
    unittest.main()
