"""Tek seed YOLO11n eğitimi (tanı/smoke). Onay zinciri için run_detector_quality.py kullanılır.

Kullanım:
    .venv/bin/python scripts/train_detector.py runs/yolo_dataset_d1nr_20260926 runs/detector_smoke --seed 20260911 --epochs 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ebabil.perception.detector import (
    DEFAULT_DETECTOR_CONFIG,
    load_detector_config,
    train_detector,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yolo_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_DETECTOR_CONFIG)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    artifacts = train_detector(
        args.yolo_dir,
        args.output_dir,
        load_detector_config(args.config),
        args.seed,
        epochs_override=args.epochs,
    )
    print(f"weights={artifacts.weights}")
    print(f"weights_sha256={artifacts.weights_sha256}")
    print(f"metadata={artifacts.metadata}")


if __name__ == "__main__":
    main()
