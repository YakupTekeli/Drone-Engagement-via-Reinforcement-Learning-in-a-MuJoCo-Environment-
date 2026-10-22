#!/usr/bin/env python3
"""Vision dataset üzerinde küçük hedef/serbest-alan encoder baseline'ı eğit."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ebabil.perception import (
    load_vision_encoder_config,
    train_vision_encoder,
    validate_dataset_leakage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config, training_config = (
        load_vision_encoder_config(args.config)
        if args.config
        else load_vision_encoder_config()
    )
    if args.epochs is not None:
        training_config = replace(training_config, epochs=args.epochs)
    if args.batch_size is not None:
        training_config = replace(training_config, batch_size=args.batch_size)
    training_config.validate()
    validate_dataset_leakage(args.dataset)
    artifacts = train_vision_encoder(
        args.dataset, args.output, model_config, training_config
    )
    print(f"checkpoint={artifacts.checkpoint}")
    print(f"metrics={artifacts.metrics}")


if __name__ == "__main__":
    main()
