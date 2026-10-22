"""Yarıda kesilmiş dedektör kalite koşusunu tamamlanmış seed'lerle raporla.

Yalnız kullanıcı kararıyla kullanılır; karar metni rapora yazılır.

Kullanım:
    .venv/bin/python scripts/finalize_detector_quality.py runs/yolo_dataset_d1nr_20260926 runs/detector_quality_v1_20260929 --seeds 20260911 --decision "..."
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ebabil.perception.detector import DEFAULT_DETECTOR_CONFIG
from ebabil.perception.detector_quality import (
    finalize_detector_quality,
    load_detector_quality_configs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yolo_dir", type=Path)
    parser.add_argument("quality_dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_DETECTOR_CONFIG)
    args = parser.parse_args()
    detector_config, quality_config = load_detector_quality_configs(args.config)
    artifacts = finalize_detector_quality(
        args.yolo_dir,
        args.quality_dir,
        detector_config,
        quality_config,
        completed_seeds=args.seeds,
        decision=args.decision,
    )
    print(f"quality_passed={artifacts.passed}")
    print(f"report={artifacts.report}")
    print(f"candidate_weights={artifacts.candidate_weights}")


if __name__ == "__main__":
    main()
