"""Faz 6 karşılaştırma baseline testleri."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ebabil.evaluation import evaluate_baseline, load_phase6_protocol
from ebabil.training import load_training_config


class BaselineEvaluationTests(unittest.TestCase):
    def test_direct_goal_reaches_fixed_clear_target(self) -> None:
        config = load_training_config()
        protocol = load_phase6_protocol()
        config = replace(
            config,
            environment=replace(config.environment, max_episode_steps=180),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_baseline(
                "direct_goal",
                config,
                Path(directory) / "direct",
                protocol.held_out_episode_seeds[:2],
            )

        self.assertEqual(result.metrics["success_rate"], 1.0)
        self.assertEqual(result.metrics["termination_counts"], {"goal": 2})

    def test_random_baseline_is_repeatable_for_same_episode_seeds(self) -> None:
        config = load_training_config()
        protocol = load_phase6_protocol()
        config = replace(
            config,
            environment=replace(config.environment, max_episode_steps=4),
        )
        seeds = protocol.held_out_episode_seeds[:2]
        with tempfile.TemporaryDirectory() as directory:
            first = evaluate_baseline(
                "random", config, Path(directory) / "first", seeds
            )
            second = evaluate_baseline(
                "random", config, Path(directory) / "second", seeds
            )

        self.assertEqual(first.metrics, second.metrics)


if __name__ == "__main__":
    unittest.main()
