#!/usr/bin/env python3
"""Ebabil PPO politikasını headless eğit ve koşu artifact'lerini kaydet."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ebabil.training import (
    DEFAULT_PPO_CONFIG,
    default_run_name,
    load_training_config,
    train_ppo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PPO_CONFIG)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--vector-backend", choices=("dummy", "subproc"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-normalization", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Altyapıyı doğrulayan kısa, kalite iddiası taşımayan eğitim",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    if args.smoke:
        config = replace(
            config,
            run=replace(
                config.run,
                total_timesteps=64,
                n_envs=1,
                vector_backend="dummy",
            ),
            # Vision ortamında episode sınırı ADR-013 dinamik horizon'ından
            # gelir; smoke bu alanı override edemez.
            environment=(
                config.environment
                if config.environment.kind == "vision"
                else replace(config.environment, max_episode_steps=40)
            ),
            ppo=replace(
                config.ppo,
                n_steps=32,
                batch_size=32,
                n_epochs=1,
                network_arch=(32, 32),
            ),
            checkpoint=replace(
                config.checkpoint,
                save_frequency_timesteps=32,
                evaluation_frequency_timesteps=32,
                evaluation_episodes=1,
            ),
        )
    run = config.run
    if args.total_timesteps is not None:
        run = replace(run, total_timesteps=args.total_timesteps)
    if args.seed is not None:
        run = replace(run, seed=args.seed)
    if args.n_envs is not None:
        run = replace(run, n_envs=args.n_envs)
    if args.vector_backend is not None:
        run = replace(run, vector_backend=args.vector_backend)
    config = replace(config, run=run)
    config.validate()

    run_name = args.run_name or default_run_name(config.run.seed)
    artifacts = train_ppo(
        config,
        args.output_root / run_name,
        device=args.device,
        resume_checkpoint=args.resume,
        resume_normalization=args.resume_normalization,
    )
    print(f"run_directory={artifacts.run_directory}")
    print(f"final_model={artifacts.final_model}")
    print(f"best_model={artifacts.best_model}")
    print(f"best_normalization={artifacts.best_normalization}")
    print(f"normalization={artifacts.normalization}")


if __name__ == "__main__":
    main()
