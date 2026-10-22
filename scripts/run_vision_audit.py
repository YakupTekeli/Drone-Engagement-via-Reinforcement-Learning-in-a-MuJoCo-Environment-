#!/usr/bin/env python3
"""Dondurulmuş vision adayını audit split'inde değerlendir ve onayla."""

from __future__ import annotations

import argparse
from pathlib import Path

from ebabil.perception import run_vision_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("quality_report", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_vision_audit(args.dataset, args.quality_report, args.output)
    print(f"audit_passed={artifacts.passed}")
    print(f"report={artifacts.report}")
    if artifacts.approval_manifest is None or artifacts.approved_checkpoint is None:
        raise SystemExit(1)
    print(f"approval_manifest={artifacts.approval_manifest}")
    print(f"approved_checkpoint={artifacts.approved_checkpoint}")


if __name__ == "__main__":
    main()
