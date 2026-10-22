#!/usr/bin/env python3
"""Faz 6 PPO deney matrisini baseline ve held-out evaluation ile çalıştır."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ebabil.evaluation import DEFAULT_PHASE6_PROTOCOL, load_phase6_protocol
from ebabil.experiments import run_phase6_experiment
from ebabil.training import DEFAULT_PPO_CONFIG, load_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_PPO_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PHASE6_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--record-replays", type=int, default=1)
    parser.add_argument(
        "--training-seed-count",
        type=int,
        help="Manifestteki ilk N training seed'i çalıştır",
    )
    parser.add_argument(
        "--held-out-episodes",
        type=int,
        help="Manifestteki ilk N held-out episode'u kullan",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Bir train seed ve iki held-out episode ile altyapı kontrolü",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.training_config)
    protocol = load_phase6_protocol(args.protocol)
    training_seeds = None
    held_out_seeds = None
    if args.smoke:
        config = replace(
            config,
            run=replace(config.run, n_envs=1, vector_backend="dummy"),
            environment=replace(config.environment, max_episode_steps=4),
            ppo=replace(
                config.ppo,
                n_steps=8,
                batch_size=8,
                n_epochs=1,
                network_arch=(16,),
            ),
            checkpoint=replace(
                config.checkpoint,
                save_frequency_timesteps=8,
                evaluation_frequency_timesteps=8,
                evaluation_episodes=1,
            ),
        )
        protocol = replace(protocol, pilot_timesteps=8)
        training_seeds = protocol.training_seeds[:1]
        held_out_seeds = protocol.held_out_episode_seeds[:2]
    else:
        if args.training_seed_count is not None:
            if not 1 <= args.training_seed_count <= len(protocol.training_seeds):
                raise ValueError("--training-seed-count manifest sınırları dışında")
            training_seeds = protocol.training_seeds[: args.training_seed_count]
        if args.held_out_episodes is not None:
            if not 1 <= args.held_out_episodes <= len(protocol.held_out_episode_seeds):
                raise ValueError("--held-out-episodes manifest sınırları dışında")
            held_out_seeds = protocol.held_out_episode_seeds[: args.held_out_episodes]
    artifacts = run_phase6_experiment(
        config,
        protocol,
        args.output_dir,
        stage="pilot" if args.smoke else args.stage,
        training_seeds=training_seeds,
        held_out_episode_seeds=held_out_seeds,
        record_replays_per_seed=args.record_replays,
        device=args.device,
    )
    print(f"summary={artifacts.summary}")
    print(f"performance_checks_passed={artifacts.metrics['performance_checks_passed']}")
    print(f"quality_gate_passed={artifacts.metrics['quality_gate_passed']}")


if __name__ == "__main__":
    main()
