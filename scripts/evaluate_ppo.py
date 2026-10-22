#!/usr/bin/env python3
"""Bir PPO checkpoint'ini deterministic episode'larda değerlendir."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ebabil.evaluation import (
    DEFAULT_PHASE6_PROTOCOL,
    evaluate_policy,
    load_phase6_protocol,
)
from ebabil.training import DEFAULT_PPO_CONFIG, load_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_PPO_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PHASE6_PROTOCOL)
    parser.add_argument("--scenario")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument(
        "--seed",
        type=int,
        help="Manifest yerine bu seed'den başlayan ad-hoc ardışık episode'lar",
    )
    parser.add_argument("--record-replays", type=int, default=3)
    parser.add_argument(
        "--scene-mode",
        choices=("clear", "single_blocker", "random_obstacles"),
    )
    parser.add_argument("--sensor-mode", choices=("ideal", "imperfect"))
    parser.add_argument(
        "--endpoint-mode",
        choices=("fixed", "random_start", "random_goal", "random_both"),
    )
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    protocol = load_phase6_protocol(args.protocol)
    scenario = protocol.scenario(args.scenario)
    environment = replace(
        config.environment,
        scene_mode=scenario.scene_mode,
        endpoint_mode=scenario.endpoint_mode,
        sensor_mode=scenario.sensor_mode,
    )
    if args.scene_mode is not None:
        environment = replace(environment, scene_mode=args.scene_mode)
    if args.sensor_mode is not None:
        environment = replace(environment, sensor_mode=args.sensor_mode)
    if args.endpoint_mode is not None:
        environment = replace(environment, endpoint_mode=args.endpoint_mode)
    if args.max_episode_steps is not None:
        environment = replace(environment, max_episode_steps=args.max_episode_steps)
    config = replace(config, environment=environment)
    episodes = (
        len(protocol.held_out_episode_seeds) if args.episodes is None else args.episodes
    )
    if episodes <= 0:
        raise ValueError("--episodes pozitif olmalı")
    episode_seeds = None
    base_seed = args.seed if args.seed is not None else 0
    if args.seed is None:
        if episodes > len(protocol.held_out_episode_seeds):
            raise ValueError("--episodes held-out manifest uzunluğunu aşamaz")
        episode_seeds = protocol.held_out_episode_seeds[:episodes]
    artifacts = evaluate_policy(
        args.checkpoint,
        config,
        args.output_dir,
        episodes=episodes,
        seed=base_seed,
        episode_seeds=episode_seeds,
        record_replays=args.record_replays,
        normalization_path=args.normalization,
        device=args.device,
    )
    metrics = artifacts.metrics
    print(f"summary={artifacts.summary}")
    print(f"episodes={metrics['episode_count']}")
    print(f"success_rate={metrics['success_rate']:.3f}")
    print(f"collision_rate={metrics['collision_rate']:.3f}")
    print(f"mean_return={metrics['mean_return']:.6f}")
    print(f"replays={len(artifacts.replay_files)}")


if __name__ == "__main__":
    main()
