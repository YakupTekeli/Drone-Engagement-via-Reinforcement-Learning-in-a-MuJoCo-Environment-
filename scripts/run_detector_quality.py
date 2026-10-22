"""YOLO11n dedektörünü üç seed eğit ve ADR-023 §3 kapılarını ölç.

Kullanım:
    .venv/bin/python scripts/run_detector_quality.py runs/yolo_dataset_d1nr_20260926 runs/detector_quality_v1_20260929
    # uçtan uca smoke (audit'e alınamaz):
    .venv/bin/python scripts/run_detector_quality.py <yolo_dir> <out> --epochs-override 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ebabil.perception.detector import DEFAULT_DETECTOR_CONFIG
from ebabil.perception.detector_quality import (
    load_detector_quality_configs,
    run_detector_quality,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yolo_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_DETECTOR_CONFIG)
    parser.add_argument("--epochs-override", type=int)
    args = parser.parse_args()
    detector_config, quality_config = load_detector_quality_configs(args.config)
    artifacts = run_detector_quality(
        args.yolo_dir,
        args.output_dir,
        detector_config,
        quality_config,
        progress_callback=lambda done, total, seed: print(
            f"training={done}/{total}, seed={seed}", flush=True
        ),
        epochs_override=args.epochs_override,
    )
    report = json.loads(artifacts.report.read_text(encoding="utf-8"))
    for seed, splits in report["checks"].items():
        for split, checks in splits.items():
            values = {k: round(v, 4) for k, v in checks.items() if k.endswith("_value")}
            gates = {k: v for k, v in checks.items() if isinstance(v, bool)}
            print(
                f"seed={seed} split={split} gates={gates} values={values}", flush=True
            )
    print(f"quality_passed={artifacts.passed}")
    print(f"report={artifacts.report}")
    print(f"candidate_weights={artifacts.candidate_weights}")


if __name__ == "__main__":
    main()
