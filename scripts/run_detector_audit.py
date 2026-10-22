"""Seçilen dedektör adayını audit bölümünde bir kez ölç ve onay manifestini yaz.

Kullanım:
    .venv/bin/python scripts/run_detector_audit.py runs/yolo_dataset_d1nr_20260926 runs/detector_quality_v1_20260929/quality_report.json runs/detector_audit_v1_20260929
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ebabil.perception.detector_quality import run_detector_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yolo_dir", type=Path)
    parser.add_argument("quality_report", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    artifacts = run_detector_audit(args.yolo_dir, args.quality_report, args.output_dir)
    report = json.loads(artifacts.report.read_text(encoding="utf-8"))
    print(json.dumps(report["checks"], indent=2, sort_keys=True))
    print(f"audit_passed={artifacts.passed}")
    print(f"report={artifacts.report}")
    print(f"approval_manifest={artifacts.approval_manifest}")
    print(f"approved_weights={artifacts.approved_weights}")


if __name__ == "__main__":
    main()
