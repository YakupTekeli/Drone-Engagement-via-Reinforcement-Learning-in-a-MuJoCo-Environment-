#!/usr/bin/env python3
"""Seed ayrımlı MuJoCo RGB/privileged vision dataset üret."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ebabil.perception import (
    generate_vision_dataset,
    load_vision_dataset_config,
    validate_dataset_leakage,
)
from ebabil.perception.dataset import Split
from ebabil.scenes import load_scene_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--scenes-per-split", type=int)
    parser.add_argument("--samples-per-scene", type=int)
    parser.add_argument(
        "--scene-config",
        type=Path,
        help=(
            "Sahne config yolu. Verilmezse legacy workspace kullanılır. "
            "Encoder, politikanın uçacağı ortamda eğitilmeli (REV-037)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = (
        load_vision_dataset_config(args.config)
        if args.config
        else load_vision_dataset_config()
    )
    if args.scenes_per_split is not None:
        if args.scenes_per_split <= 0:
            raise ValueError("--scenes-per-split pozitif olmalı")
        config = replace(
            config,
            train_scene_count=args.scenes_per_split,
            validation_scene_count=args.scenes_per_split,
            development_ood_scene_count=args.scenes_per_split,
            audit_scene_count=args.scenes_per_split,
        )
    if args.samples_per_scene is not None:
        if args.samples_per_scene <= 0:
            raise ValueError("--samples-per-scene pozitif olmalı")
        config = replace(config, samples_per_scene=args.samples_per_scene)

    def show_progress(completed: int, total: int, split: Split, seed: int) -> None:
        print(f"scene={completed}/{total}, split={split}, seed={seed}", flush=True)

    scene_config = (
        load_scene_config(args.scene_config)
        if args.scene_config
        else load_scene_config()
    )
    records = generate_vision_dataset(
        args.output,
        config,
        scene_config=scene_config,
        progress_callback=show_progress,
    )
    validate_dataset_leakage(args.output)
    split_counts = {
        split: sum(record.split == split for record in records)
        for split in ("train", "validation", "development_ood", "audit")
    }
    print(f"dataset={args.output.expanduser().resolve()}")
    print(f"scene_config={args.scene_config or '(legacy varsayilan)'}")
    print(f"records={len(records)}, splits={split_counts}, leakage_check=passed")


if __name__ == "__main__":
    main()
