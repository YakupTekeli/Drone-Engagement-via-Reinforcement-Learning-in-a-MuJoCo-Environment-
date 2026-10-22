"""NPZ görsel dataset'inden YOLO kutu etiketli dataset üret (ADR-023 §2, plan §24 Dilim B).

Kullanım:
    .venv/bin/python scripts/export_yolo_dataset.py runs/vision_dataset_d1_p40_20260914 runs/yolo_dataset_d1_20260926
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ebabil.perception.detections import export_yolo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--min-side-px", type=int, default=3)
    parser.add_argument("--min-area-px", type=int, default=16)
    args = parser.parse_args()
    summary = export_yolo_dataset(
        args.dataset_dir,
        args.output_dir,
        min_side_px=args.min_side_px,
        min_area_px=args.min_area_px,
    )
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    print(f"output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
