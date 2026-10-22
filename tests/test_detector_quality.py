"""ADR-023 §3 dedektör metrik tanımları, kapılar ve fail-closed onay testleri."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ebabil.perception.detections import OBSTACLE_CLASS_ID, TARGET_CLASS_ID, Detection
from ebabil.perception.detector import (
    DEFAULT_DETECTOR_CONFIG,
    PROJECT_ROOT,
    DetectorConfig,
    load_detector_config,
)
from ebabil.perception.detector_quality import (
    APPROVAL_KIND,
    APPROVAL_MANIFEST_SCHEMA,
    AUDIT_REPORT_SCHEMA,
    QUALITY_REPORT_SCHEMA,
    REQUIRED_GATES,
    DetectorQualityConfig,
    FrameResult,
    all_gates_passed,
    approve_detector_by_decision,
    box_iou,
    compute_detection_metrics,
    finalize_detector_quality,
    gate_checks,
    greedy_match,
    load_detector_quality_configs,
    verify_detector_approval,
)

OB, TG = OBSTACLE_CLASS_ID, TARGET_CLASS_ID


def _det(
    cls: int, x0: float, y0: float, x1: float, y1: float, conf: float = 1.0
) -> Detection:
    return Detection(cls, x0, y0, x1, y1, conf)


class IouAndMatchingTests(unittest.TestCase):
    def test_box_iou_known_values(self) -> None:
        first = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float64)
        second = np.array(
            [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=np.float64
        )
        iou = box_iou(first, second)
        np.testing.assert_allclose(iou[0], [1.0, 50.0 / 150.0, 0.0])
        self.assertEqual(box_iou(np.zeros((0, 4)), second).shape, (0, 3))

    def test_greedy_match_prefers_confident_prediction_and_counts_duplicates(
        self,
    ) -> None:
        truth = [_det(OB, 0, 0, 10, 10)]
        predictions = [_det(OB, 0, 0, 10, 10, 0.4), _det(OB, 0, 0, 10, 9, 0.9)]
        predicted, matched = greedy_match(predictions, truth, 0.5)
        self.assertEqual(predicted.tolist(), [False, True])
        self.assertEqual(matched.tolist(), [True])


class MetricDefinitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DetectorQualityConfig()

    def test_metrics_follow_documented_definitions(self) -> None:
        frames = [
            # Büyük engel bulundu, küçük engel kaçtı; hedef 3 m'de bulundu.
            FrameResult(
                ground_truth=(
                    _det(OB, 0, 0, 40, 40),
                    _det(OB, 100, 100, 110, 110),
                    _det(TG, 200, 100, 260, 140),
                ),
                predictions=(
                    _det(OB, 1, 1, 40, 40, 0.8),
                    _det(TG, 200, 100, 258, 140, 0.9),
                    _det(OB, 150, 0, 155, 5, 0.6),  # küçük, eşleşmeyen: paydadan çıkar
                ),
                target_range_m=3.0,
                target_visible_privileged=True,
            ),
            # Büyük engel kaçtı; büyük sahte engel var; hedef 8 m'de kaçtı.
            FrameResult(
                ground_truth=(_det(OB, 0, 0, 30, 30), _det(TG, 10, 0, 50, 10)),
                predictions=(
                    _det(OB, 200, 100, 260, 160, 0.7),
                    _det(OB, 0, 0, 30, 30, 0.1),  # güven eşiğinin altında
                ),
                target_range_m=8.0,
                target_visible_privileged=True,
            ),
            # Hedefsiz kare, sahte hedef.
            FrameResult(
                ground_truth=(),
                predictions=(_det(TG, 0, 0, 20, 20, 0.7),),
                target_range_m=None,
                target_visible_privileged=False,
            ),
            # Hedefsiz ama privileged görünür (filtreye takılmış küçük hedef): sayılmaz.
            FrameResult(
                ground_truth=(),
                predictions=(_det(TG, 0, 0, 20, 20, 0.7),),
                target_range_m=None,
                target_visible_privileged=True,
            ),
        ]
        metrics = compute_detection_metrics(frames, self.config)
        self.assertAlmostEqual(metrics["obstacle_recall"], 0.5)  # 2 büyük, 1 bulundu
        self.assertAlmostEqual(metrics["obstacle_recall_all_sizes"], 1.0 / 3.0)
        self.assertAlmostEqual(
            metrics["obstacle_precision"], 0.5
        )  # 1 doğru, 1 büyük sahte
        self.assertAlmostEqual(metrics["target_recall"], 1.0)  # yalnız <= 5 m kare
        self.assertAlmostEqual(metrics["target_recall_all_ranges"], 0.5)
        self.assertAlmostEqual(metrics["target_precision"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["false_target_rate"], 1.0)
        self.assertEqual(metrics["negative_frames"], 1.0)

    def test_gates_reject_nan_and_respect_thresholds(self) -> None:
        metrics = {
            "obstacle_recall": 0.90,
            "obstacle_precision": 0.95,
            "target_recall": 0.85,
            "target_precision": 0.93,
            "false_target_rate": 0.05,
            "latency_median_ms": 4.0,
        }
        checks = gate_checks(metrics, self.config)
        self.assertTrue(all_gates_passed(checks))
        self.assertEqual(set(REQUIRED_GATES) <= set(checks), True)
        self.assertFalse(
            all_gates_passed(
                gate_checks({**metrics, "target_recall": 0.849}, self.config)
            )
        )
        self.assertFalse(
            all_gates_passed(
                gate_checks({**metrics, "false_target_rate": math.nan}, self.config)
            )
        )
        self.assertFalse(
            all_gates_passed(
                gate_checks({**metrics, "latency_median_ms": 15.1}, self.config)
            )
        )


class ConfigTests(unittest.TestCase):
    def test_repository_config_matches_adr_023(self) -> None:
        detector, quality = load_detector_quality_configs(DEFAULT_DETECTOR_CONFIG)
        self.assertEqual(detector.model, "yolo11n")
        self.assertEqual(detector.imgsz, 320)
        self.assertEqual(quality.model_seeds, (20260911, 20260912, 20260913))
        self.assertEqual(
            (
                quality.minimum_obstacle_recall,
                quality.minimum_obstacle_precision,
                quality.minimum_target_recall,
                quality.minimum_target_precision,
                quality.maximum_false_target_rate,
                quality.maximum_latency_ms,
            ),
            (0.90, 0.90, 0.85, 0.90, 0.05, 15.0),
        )
        self.assertEqual(load_detector_config(DEFAULT_DETECTOR_CONFIG), detector)

    def test_config_validation(self) -> None:
        detector = load_detector_config(DEFAULT_DETECTOR_CONFIG)
        with self.assertRaisesRegex(ValueError, "yolo11n"):
            replace(detector, model="yolo11s").validate()
        with self.assertRaisesRegex(ValueError, "32"):
            replace(detector, imgsz=300).validate()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            replace(detector, pretrained_weights_sha256="abc").validate()
        with self.assertRaisesRegex(ValueError, "tekrarsız"):
            DetectorQualityConfig(model_seeds=(1, 1)).validate()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ApprovalChainTests(unittest.TestCase):
    def _chain(
        self, root: Path, *, gate_value: bool = True, drop_gate: str | None = None
    ) -> Path:
        quality_dir = root / "quality"
        weights = quality_dir / "seed_1" / "weights" / "last.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"weights")
        dataset = {"schema_version": "yolo-dataset-v1", "frames_sha256": "f" * 64}
        gates = {name: True for name in REQUIRED_GATES}
        gates["target_recall"] = gate_value
        if drop_gate is not None:
            gates.pop(drop_gate)
        config = asdict(DetectorConfig(pretrained_weights_sha256="0" * 64))
        quality = {
            "schema_version": QUALITY_REPORT_SCHEMA,
            "kind": APPROVAL_KIND,
            "passed": True,
            "dataset": dataset,
            "checks": {"1": {"validation": gates, "development_ood": gates}},
        }
        (quality_dir / "quality_report.json").write_text(json.dumps(quality))
        audit_dir = root / "audit"
        audit_dir.mkdir()
        audit = {
            "schema_version": AUDIT_REPORT_SCHEMA,
            "kind": APPROVAL_KIND,
            "passed": True,
            "dataset": dataset,
            "checks": gates,
        }
        (audit_dir / "audit_report.json").write_text(json.dumps(audit))
        manifest = {
            "schema_version": APPROVAL_MANIFEST_SCHEMA,
            "kind": APPROVAL_KIND,
            "audit_report": "audit_report.json",
            "audit_report_sha256": _sha(audit_dir / "audit_report.json"),
            "quality_report": "../quality/quality_report.json",
            "quality_report_sha256": _sha(quality_dir / "quality_report.json"),
            "approved_weights": "../quality/seed_1/weights/last.pt",
            "approved_weights_sha256": _sha(weights),
            "detector_config": config,
            "dataset": dataset,
        }
        path = audit_dir / "approval_manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_valid_chain_resolves_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._chain(Path(tmp))
            approved = verify_detector_approval(manifest)
            self.assertEqual(approved.weights.name, "last.pt")
            self.assertEqual(approved.config.model, "yolo11n")

    def test_tampered_weights_and_wrong_kind_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._chain(Path(tmp))
            (Path(tmp) / "quality" / "seed_1" / "weights" / "last.pt").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_detector_approval(manifest)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._chain(Path(tmp))
            payload = json.loads(manifest.read_text())
            payload["kind"] = "visual-vae"
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "yolo-detector"):
                verify_detector_approval(manifest)

    def test_failed_or_missing_gate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "geçilmemiş: target_recall"):
                verify_detector_approval(self._chain(Path(tmp), gate_value=False))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "eksik"):
                verify_detector_approval(self._chain(Path(tmp), drop_gate="latency"))


class UserDecisionApprovalTests(unittest.TestCase):
    """REV-050: audit'i geçmeyen adayın kullanıcı kararıyla şekil kısıtlı onayı."""

    def _chain(self, root: Path) -> tuple[Path, Path]:
        dataset = {"schema_version": "yolo-dataset-v1", "frames_sha256": "f" * 64}
        quality_dir = root / "quality"
        weights = quality_dir / "seed_1" / "weights" / "last.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"weights")
        passed = {name: True for name in REQUIRED_GATES}
        quality = {
            "schema_version": QUALITY_REPORT_SCHEMA,
            "kind": APPROVAL_KIND,
            "passed": True,
            "dataset": dataset,
            "detector_config": asdict(
                DetectorConfig(pretrained_weights_sha256="0" * 64)
            ),
            "candidate_weights": "seed_1/weights/last.pt",
            "candidate_weights_sha256": _sha(weights),
            "checks": {"1": {"validation": passed, "development_ood": passed}},
        }
        quality_path = quality_dir / "quality_report.json"
        quality_path.write_text(json.dumps(quality))
        audit_dir = root / "audit"
        audit_dir.mkdir()
        failed = dict(passed, obstacle_recall=False, obstacle_precision=False)
        audit = {
            "schema_version": AUDIT_REPORT_SCHEMA,
            "kind": APPROVAL_KIND,
            "passed": False,
            "dataset": dataset,
            "quality_report_sha256": _sha(quality_path),
            "checks": failed,
        }
        audit_path = audit_dir / "audit_report.json"
        audit_path.write_text(json.dumps(audit))
        (quality_dir / "audit_consumed.json").write_text(
            json.dumps({"status": "completed", "audit_report_sha256": _sha(audit_path)})
        )
        return quality_path, audit_path

    def test_scoped_approval_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            quality_path, audit_path = self._chain(Path(tmp))
            manifest = approve_detector_by_decision(
                quality_path,
                audit_path,
                decision="kullanıcı kararı",
                finding="REV-050",
                obstacle_geometries=("box", "cylinder", "capsule", "sphere"),
            )
            approved = verify_detector_approval(manifest)
            self.assertEqual(approved.basis, "user_decision")
            self.assertEqual(
                approved.obstacle_geometries, ("box", "cylinder", "capsule", "sphere")
            )
            payload = json.loads(manifest.read_text())
            self.assertEqual(
                payload["failed_audit_gates"], ["obstacle_precision", "obstacle_recall"]
            )
            self.assertFalse(payload["audit_passed"])
            with self.assertRaises(FileExistsError):
                approve_detector_by_decision(
                    quality_path,
                    audit_path,
                    decision="x",
                    finding="REV-050",
                    obstacle_geometries=("box",),
                )

    def test_scoped_approval_rejects_tampering_and_missing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            quality_path, audit_path = self._chain(Path(tmp))
            with self.assertRaisesRegex(ValueError, "zorunlu"):
                approve_detector_by_decision(
                    quality_path,
                    audit_path,
                    decision=" ",
                    finding="REV-050",
                    obstacle_geometries=("box",),
                )
            with self.assertRaisesRegex(ValueError, "şekli"):
                approve_detector_by_decision(
                    quality_path,
                    audit_path,
                    decision="x",
                    finding="REV-050",
                    obstacle_geometries=("cone",),
                )
            manifest = approve_detector_by_decision(
                quality_path,
                audit_path,
                decision="x",
                finding="REV-050",
                obstacle_geometries=("box",),
            )
            payload = json.loads(manifest.read_text())
            payload["failed_audit_gates"] = ["obstacle_recall"]
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "geçmeyen audit kapıları"):
                verify_detector_approval(manifest)
            payload["failed_audit_gates"] = ["obstacle_precision", "obstacle_recall"]
            payload.pop("approval_basis")
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "audit raporu başarısız"):
                verify_detector_approval(manifest)


class FinalizeFromCompletedSeedsTests(unittest.TestCase):
    def _setup(
        self, root: Path
    ) -> tuple[Path, Path, DetectorConfig, DetectorQualityConfig]:
        yolo = root / "yolo"
        yolo.mkdir()
        (yolo / "frames.jsonl").write_text('{"sample_id": "a"}\n')
        summary = {
            "schema_version": "yolo-dataset-v1",
            "class_names": ["obstacle", "target"],
            "frames_sha256": _sha(yolo / "frames.jsonl"),
            "source_provenance": {"schema_version": "vision-dataset-v4"},
            "min_side_px": 3,
            "min_area_px": 16,
        }
        (yolo / "export_summary.json").write_text(json.dumps(summary))
        from ebabil.perception.detector import yolo_dataset_provenance

        provenance = yolo_dataset_provenance(yolo)
        detector = DetectorConfig(pretrained_weights_sha256="0" * 64)
        quality = DetectorQualityConfig()
        run = root / "quality" / "seed_20260911"
        (run / "weights").mkdir(parents=True)
        (run / "weights" / "last.pt").write_bytes(b"w")
        (root / "quality" / "seed_20260912").mkdir()  # yarıda kesilen seed
        metadata = {
            "seed": 20260911,
            "epochs": detector.epochs,
            "dataset": provenance,
            "weights": "weights/last.pt",
            "weights_sha256": _sha(run / "weights" / "last.pt"),
        }
        (run / "training_metadata.json").write_text(json.dumps(metadata))
        metrics = {
            "obstacle_recall": 0.95,
            "obstacle_precision": 0.94,
            "target_recall": 0.97,
            "target_precision": 0.92,
            "false_target_rate": 0.01,
            "latency_median_ms": 5.8,
        }
        stored = {
            "checks": {
                split: gate_checks(metrics, quality)
                for split in ("validation", "development_ood")
            },
            "metrics": {"validation": metrics, "development_ood": metrics},
            "grayscale_validation": metrics,
            "map_info": {},
        }
        (run / "quality_metrics.json").write_text(json.dumps(stored))
        return yolo, root / "quality", detector, quality

    def test_single_seed_report_records_decision_and_is_audit_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yolo, quality_dir, detector, quality = self._setup(Path(tmp))
            artifacts = finalize_detector_quality(
                yolo,
                quality_dir,
                detector,
                quality,
                completed_seeds=(20260911,),
                decision="test kararı",
            )
            report = json.loads(artifacts.report.read_text())
            self.assertTrue(report["passed"])
            self.assertIsNone(report["epochs_override"])
            self.assertEqual(report["quality_config"]["model_seeds"], [20260911])
            self.assertEqual(report["planned_model_seeds"], list(quality.model_seeds))
            self.assertEqual(report["aborted_seed_directories"], [20260912])
            self.assertEqual(report["decision"], "test kararı")
            self.assertEqual(
                report["candidate_weights"], "seed_20260911/weights/last.pt"
            )
            with self.assertRaises(FileExistsError):
                finalize_detector_quality(
                    yolo,
                    quality_dir,
                    detector,
                    quality,
                    completed_seeds=(20260911,),
                    decision="tekrar",
                )

    def test_tampered_weights_unplanned_seed_and_empty_decision_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yolo, quality_dir, detector, quality = self._setup(Path(tmp))
            with self.assertRaisesRegex(ValueError, "gerekçe"):
                finalize_detector_quality(
                    yolo,
                    quality_dir,
                    detector,
                    quality,
                    completed_seeds=(20260911,),
                    decision=" ",
                )
            with self.assertRaisesRegex(ValueError, "Planlanmamış"):
                finalize_detector_quality(
                    yolo,
                    quality_dir,
                    detector,
                    quality,
                    completed_seeds=(1,),
                    decision="x",
                )
            (quality_dir / "seed_20260911" / "weights" / "last.pt").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                finalize_detector_quality(
                    yolo,
                    quality_dir,
                    detector,
                    quality,
                    completed_seeds=(20260911,),
                    decision="x",
                )


PRETRAINED = PROJECT_ROOT / "runs" / "weights" / "yolo11n.pt"


@unittest.skipUnless(
    importlib.util.find_spec("ultralytics") is not None and PRETRAINED.is_file(),
    "ultralytics veya önceden eğitilmiş ağırlık yok (detector bağımlılık grubu kurulu değil)",
)
class YoloDetectorContractTests(unittest.TestCase):
    def test_coco_weights_are_rejected_for_wrong_classes(self) -> None:
        from ebabil.perception.detector import YoloDetector

        with self.assertRaisesRegex(ValueError, "sınıfları"):
            YoloDetector(PRETRAINED, device="cpu")


if __name__ == "__main__":
    unittest.main()
