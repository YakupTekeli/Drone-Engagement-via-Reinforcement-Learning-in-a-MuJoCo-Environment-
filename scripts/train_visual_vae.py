#!/usr/bin/env python3
"""Vision dataset'in yalnız actor RGB arşivleriyle etiketsiz VAE eğit (ADR-022)."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ebabil.perception import load_visual_vae_config, train_visual_vae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config, training_config = (
        load_visual_vae_config(args.config) if args.config else load_visual_vae_config()
    )
    if args.epochs is not None:
        training_config = replace(training_config, epochs=args.epochs)
    if args.batch_size is not None:
        training_config = replace(training_config, batch_size=args.batch_size)
    if args.seed is not None:
        training_config = replace(training_config, seed=args.seed)
    training_config.validate()

    def show_progress(epoch: int, total: int, loss: float) -> None:
        print(f"epoch={epoch}/{total} loss={loss:.3f}", flush=True)

    artifacts = train_visual_vae(
        args.dataset,
        args.output,
        model_config,
        training_config,
        progress_callback=show_progress,
    )
    print(f"checkpoint={artifacts.checkpoint}")
    print(f"metrics={artifacts.metrics}")


if __name__ == "__main__":
    main()
