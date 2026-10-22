#!/usr/bin/env python3
"""Etiketsiz VAE'yi çoklu seed ile eğit, probe'la ve kalite kapısını raporla."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ebabil.perception import load_default_vae_quality_configs, run_visual_vae_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config, training_config, quality_config = (
        load_default_vae_quality_configs(args.config)
        if args.config
        else load_default_vae_quality_configs()
    )
    if args.epochs is not None:
        training_config = replace(training_config, epochs=args.epochs)
    if args.seeds is not None:
        quality_config = replace(quality_config, model_seeds=tuple(args.seeds))

    def show_progress(completed: int, total: int, seed: int) -> None:
        print(f"training={completed}/{total}, seed={seed}", flush=True)

    artifacts = run_visual_vae_quality(
        args.dataset,
        args.output,
        model_config,
        training_config,
        quality_config,
        progress_callback=show_progress,
    )
    print(f"quality_passed={artifacts.passed}")
    print(f"report={artifacts.report}")
    print(f"candidate_checkpoint={artifacts.candidate_checkpoint}")
    if not artifacts.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
