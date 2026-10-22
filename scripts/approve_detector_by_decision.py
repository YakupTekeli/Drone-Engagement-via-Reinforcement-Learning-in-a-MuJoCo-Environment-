"""Audit'i geçmeyen dedektör adayını kullanıcı kararıyla, engel şekli kısıtlı onayla (REV-050).

Başarısız audit raporu hash zincirinde kalır; manifest kararı, bulguyu, geçmeyen
audit kapılarını ve onaylı engel şekillerini taşır. EbabilVisionNavigation-v2
bu şekil kümesi dışındaki sahne config'lerini reddeder.

Kullanım:
    .venv/bin/python scripts/approve_detector_by_decision.py runs/detector_quality_v1_20260929/quality_report.json runs/detector_audit_v1_20260929/audit_report.json --finding REV-050 --geometries box cylinder capsule sphere --decision "..."
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ebabil.perception.detector_quality import (
    approve_detector_by_decision,
    verify_detector_approval,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quality_report", type=Path)
    parser.add_argument("audit_report", type=Path)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--finding", required=True)
    parser.add_argument("--geometries", nargs="+", required=True)
    args = parser.parse_args()
    manifest = approve_detector_by_decision(
        args.quality_report,
        args.audit_report,
        decision=args.decision,
        finding=args.finding,
        obstacle_geometries=args.geometries,
    )
    approved = verify_detector_approval(manifest)
    print(f"approval_manifest={manifest}")
    print(f"basis={approved.basis}")
    print(f"approved_obstacle_geometries={list(approved.obstacle_geometries or ())}")
    print(f"approved_weights={approved.weights}")


if __name__ == "__main__":
    main()
