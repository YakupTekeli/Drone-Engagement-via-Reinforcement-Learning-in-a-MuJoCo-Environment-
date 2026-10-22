"""Faz 6 evaluation protokolü testleri."""

from __future__ import annotations

import unittest
from dataclasses import replace

from ebabil.evaluation import load_phase6_protocol


class Phase6ProtocolTests(unittest.TestCase):
    def test_default_protocol_has_disjoint_seed_sets_and_scenarios(self) -> None:
        protocol = load_phase6_protocol()

        self.assertEqual(len(protocol.training_seeds), 3)
        self.assertEqual(len(protocol.held_out_episode_seeds), 20)
        self.assertFalse(
            set(protocol.training_seeds) & set(protocol.held_out_episode_seeds)
        )
        self.assertEqual(protocol.scenario().endpoint_mode, "fixed")
        self.assertEqual(
            protocol.scenario("random_start_random_goal").endpoint_mode,
            "random_both",
        )

    def test_overlapping_seed_sets_are_rejected(self) -> None:
        protocol = load_phase6_protocol()
        invalid = replace(
            protocol,
            held_out_episode_seeds=(
                protocol.training_seeds[0],
                *protocol.held_out_episode_seeds[1:],
            ),
        )

        with self.assertRaisesRegex(ValueError, "ayrık"):
            invalid.validate()


if __name__ == "__main__":
    unittest.main()
