#!/usr/bin/env python3
"""Faz 6 random ve doğrudan-hedef baseline'larını held-out seed'lerde ölç."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ebabil.evaluation import (
    DEFAULT_PHASE6_PROTOCOL,
    evaluate_baseline_suite,
    load_phase6_protocol,
)
from ebabil.training import DEFAULT_PPO_CONFIG, load_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_PPO_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PHASE6_PROTOCOL)
    parser.add_argument("--scenario")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        help="Hızlı kontrol için manifestteki ilk N held-out seed'i kullan",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.training_config)
    protocol = load_phase6_protocol(args.protocol)
    scenario = protocol.scenario(args.scenario)
    config = replace(
        config,
        environment=replace(
            config.environment,
            scene_mode=scenario.scene_mode,
            endpoint_mode=scenario.endpoint_mode,
            sensor_mode=scenario.sensor_mode,
        ),
    )
    episode_seeds = protocol.held_out_episode_seeds
    if args.episodes is not None:
        if args.episodes <= 0 or args.episodes > len(episode_seeds):
            raise ValueError("--episodes manifest uzunluğu içinde pozitif olmalı")
        episode_seeds = episode_seeds[: args.episodes]
    comparison = evaluate_baseline_suite(config, args.output_dir, episode_seeds)
    print(f"comparison={comparison}")


if __name__ == "__main__":
    main()
