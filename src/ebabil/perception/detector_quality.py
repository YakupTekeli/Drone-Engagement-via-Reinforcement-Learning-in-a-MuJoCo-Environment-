"""YOLO dedektör kalite kapıları, audit ve onay zinciri (ADR-023 §3).

Desen ADR-014/ADR-022 ile aynıdır: çoklu seed kalite koşusu -> tek
kullanımlık audit -> SHA-256 hash zinciri taşıyan onay manifesti ->
fail-closed yükleyici. Metrik tanımları bu modülde ve testlerde sabittir:

- Eşleşme: aynı sınıf, IoU `>= iou_threshold`, tahminler güvene göre azalan
  sırada açgözlü eşlenir; her gerçek kutu en fazla bir tahminle eşleşir.
- Engel recall: alanı `>= obstacle_min_area_px` olan gerçek engel kutuları
  içinde, güveni `>= obstacle_conf` tahminlerle eşleşen pay.
- Engel precision: güveni `>= obstacle_conf` engel tahminleri içinde
  eşleşen pay; eşleşmeyen ve alanı `< obstacle_min_area_px` olan tahminler
  paydadan çıkarılır (etiket filtresine takılan küçük dilimler ve kapının
  kapsamadığı uzak kutular cezalandırılmaz).
- Hedef recall: gerçek hedef kutusu olan ve hedef menzili
  `<= target_near_range_m` olan karelerde, güveni `>= target_conf` bir hedef
  tahmininin eşleştiği kare payı.
- Hedef precision: güveni `>= target_conf` hedef tahminleri içinde eşleşen pay.
- Sahte hedef oranı: gerçek hedef kutusu olmayan ve privileged olarak da hedef
  görünmeyen karelerde, güveni `>= target_conf` en az bir hedef tahmini olan
  kare payı.
"""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ebabil.perception.detections import (
    OBSTACLE_CLASS_ID,
    TARGET_CLASS_ID,
    Detection,
    load_frame_records,
    parse_yolo_lines,
)
from ebabil.perception.detector import (
    DEFAULT_DETECTOR_CONFIG,
    DetectorConfig,
    YoloDetector,
    detector_config_from_raw,
    file_sha256,
    measure_latency,
    train_detector,
    yolo_dataset_provenance,
)

QUALITY_REPORT_SCHEMA = "yolo-detector-quality-report-v1"
AUDIT_REPORT_SCHEMA = "yolo-detector-audit-report-v1"
APPROVAL_MANIFEST_SCHEMA = "yolo-detector-approval-manifest-v1"
AUDIT_CONSUMPTION_SCHEMA = "yolo-detector-audit-consumption-v1"
APPROVAL_KIND = "yolo-detector"
REQUIRED_GATES = (
    "obstacle_recall",
    "obstacle_precision",
    "target_recall",
    "target_precision",
    "false_target_rate",
    "latency",
)
DEVELOPMENT_SPLITS = ("validation", "development_ood")

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DetectorQualityConfig:
    model_seeds: tuple[int, ...] = (20260911, 20260912, 20260913)
    iou_threshold: float = 0.5
    obstacle_conf: float = 0.25
    target_conf: float = 0.50
    obstacle_min_area_px: float = 400.0
    target_near_range_m: float = 5.0
    minimum_obstacle_recall: float = 0.90
    minimum_obstacle_precision: float = 0.90
    minimum_target_recall: float = 0.85
    minimum_target_precision: float = 0.90
    maximum_false_target_rate: float = 0.05
    maximum_latency_ms: float = 15.0
    latency_frames: int = 100

    def validate(self) -> None:
        if not self.model_seeds or len(set(self.model_seeds)) != len(self.model_seeds):
            raise ValueError("model_seeds boş olmayan ve tekrarsız olmalı")
        for name in (
            "iou_threshold",
            "obstacle_conf",
            "target_conf",
            "minimum_obstacle_recall",
            "minimum_obstacle_precision",
            "minimum_target_recall",
            "minimum_target_precision",
            "maximum_false_target_rate",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} [0, 1] aralığında olmalı")
        if self.obstacle_min_area_px <= 0.0 or self.target_near_range_m <= 0.0:
            raise ValueError(
                "obstacle_min_area_px ve target_near_range_m pozitif olmalı"
            )
        if self.maximum_latency_ms <= 0.0 or self.latency_frames <= 0:
            raise ValueError("Gecikme eşiği ve kare sayısı pozitif olmalı")


def quality_config_from_raw(raw: Mapping[str, Any]) -> DetectorQualityConfig:
    defaults = DetectorQualityConfig()
    values: dict[str, Any] = {}
    for key, default in asdict(defaults).items():
        if key not in raw:
            continue
        if key == "model_seeds":
            values[key] = tuple(int(seed) for seed in raw[key])
        elif isinstance(default, int) and not isinstance(default, bool):
            values[key] = int(raw[key])
        else:
            values[key] = float(raw[key])
    config = DetectorQualityConfig(**values)
    config.validate()
    return config


def load_detector_quality_configs(
    path: str | Path = DEFAULT_DETECTOR_CONFIG,
) -> tuple[DetectorConfig, DetectorQualityConfig]:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    detector_raw = raw.get("detector")
    quality_raw = raw.get("quality")
    if not isinstance(detector_raw, dict) or not isinstance(quality_raw, dict):
        raise ValueError("Config [detector] ve [quality] bölümlerini içermeli")
    return detector_config_from_raw(detector_raw), quality_config_from_raw(quality_raw)


# --------------------------------------------------------------------------
# Metrikler (saf NumPy)
# --------------------------------------------------------------------------


def _as_array(detections: Sequence[Detection]) -> FloatArray:
    if not detections:
        return np.zeros((0, 4), dtype=np.float64)
    return np.array(
        [[d.x_min, d.y_min, d.x_max, d.y_max] for d in detections], dtype=np.float64
    )


def box_iou(first: FloatArray, second: FloatArray) -> FloatArray:
    """`(N, 4)` ve `(M, 4)` xyxy kutular için `(N, M)` IoU."""

    if first.size == 0 or second.size == 0:
        return np.zeros((first.shape[0], second.shape[0]), dtype=np.float64)
    x0 = np.maximum(first[:, None, 0], second[None, :, 0])
    y0 = np.maximum(first[:, None, 1], second[None, :, 1])
    x1 = np.minimum(first[:, None, 2], second[None, :, 2])
    y1 = np.minimum(first[:, None, 3], second[None, :, 3])
    intersection = np.clip(x1 - x0, 0.0, None) * np.clip(y1 - y0, 0.0, None)
    area_first = (first[:, 2] - first[:, 0]) * (first[:, 3] - first[:, 1])
    area_second = (second[:, 2] - second[:, 0]) * (second[:, 3] - second[:, 1])
    union = area_first[:, None] + area_second[None, :] - intersection
    return np.where(union > 0.0, intersection / np.where(union > 0.0, union, 1.0), 0.0)


def greedy_match(
    predictions: Sequence[Detection],
    ground_truth: Sequence[Detection],
    iou_threshold: float,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Güvene göre azalan açgözlü eşleşme; `(tahmin eşleşti, gerçek eşleşti)`."""

    order = sorted(range(len(predictions)), key=lambda i: -predictions[i].confidence)
    ious = box_iou(_as_array(predictions), _as_array(ground_truth))
    predicted = np.zeros(len(predictions), dtype=bool)
    matched = np.zeros(len(ground_truth), dtype=bool)
    for index in order:
        if ious.shape[1] == 0:
            break
        candidates = np.where(~matched, ious[index], -1.0)
        best = int(np.argmax(candidates))
        if candidates[best] >= iou_threshold:
            predicted[index] = True
            matched[best] = True
    return predicted, matched


@dataclass(frozen=True)
class FrameResult:
    ground_truth: tuple[Detection, ...]
    predictions: tuple[Detection, ...]
    target_range_m: float | None
    target_visible_privileged: bool


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else float("nan")


def compute_detection_metrics(
    frames: Iterable[FrameResult], config: DetectorQualityConfig
) -> dict[str, float]:
    obstacle_gt_large = obstacle_gt_large_hit = 0
    obstacle_gt_all = obstacle_gt_all_hit = 0
    obstacle_pred_counted = obstacle_pred_hit = 0
    target_near = target_near_hit = 0
    target_all = target_all_hit = 0
    target_pred = target_pred_hit = 0
    negative_frames = negative_false = 0
    frame_count = 0
    for frame in frames:
        frame_count += 1
        gt_obstacles = [
            d for d in frame.ground_truth if d.class_id == OBSTACLE_CLASS_ID
        ]
        gt_targets = [d for d in frame.ground_truth if d.class_id == TARGET_CLASS_ID]
        obstacle_preds = [
            d
            for d in frame.predictions
            if d.class_id == OBSTACLE_CLASS_ID and d.confidence >= config.obstacle_conf
        ]
        target_preds = [
            d
            for d in frame.predictions
            if d.class_id == TARGET_CLASS_ID and d.confidence >= config.target_conf
        ]
        predicted, matched = greedy_match(
            obstacle_preds, gt_obstacles, config.iou_threshold
        )
        for detection, hit in zip(gt_obstacles, matched, strict=True):
            obstacle_gt_all += 1
            obstacle_gt_all_hit += int(hit)
            if detection.area >= config.obstacle_min_area_px:
                obstacle_gt_large += 1
                obstacle_gt_large_hit += int(hit)
        for detection, hit in zip(obstacle_preds, predicted, strict=True):
            if hit or detection.area >= config.obstacle_min_area_px:
                obstacle_pred_counted += 1
                obstacle_pred_hit += int(hit)
        target_predicted, target_matched = greedy_match(
            target_preds, gt_targets, config.iou_threshold
        )
        target_pred += len(target_preds)
        target_pred_hit += int(target_predicted.sum())
        if gt_targets:
            hit = bool(target_matched.any())
            target_all += 1
            target_all_hit += int(hit)
            if (
                frame.target_range_m is not None
                and frame.target_range_m <= config.target_near_range_m
            ):
                target_near += 1
                target_near_hit += int(hit)
        elif not frame.target_visible_privileged:
            negative_frames += 1
            negative_false += int(bool(target_preds))
    return {
        "frames": float(frame_count),
        "obstacle_recall": _ratio(obstacle_gt_large_hit, obstacle_gt_large),
        "obstacle_precision": _ratio(obstacle_pred_hit, obstacle_pred_counted),
        "obstacle_recall_all_sizes": _ratio(obstacle_gt_all_hit, obstacle_gt_all),
        "obstacle_ground_truth_large": float(obstacle_gt_large),
        "obstacle_predictions_counted": float(obstacle_pred_counted),
        "target_recall": _ratio(target_near_hit, target_near),
        "target_recall_all_ranges": _ratio(target_all_hit, target_all),
        "target_frames_near": float(target_near),
        "target_precision": _ratio(target_pred_hit, target_pred),
        "target_predictions": float(target_pred),
        "false_target_rate": _ratio(negative_false, negative_frames),
        "negative_frames": float(negative_frames),
    }


def gate_checks(
    metrics: Mapping[str, float], config: DetectorQualityConfig
) -> dict[str, bool | float]:
    """ADR-023 §3 kapıları; NaN (ölçülemeyen) kapı geçmiş sayılmaz."""

    def at_least(value: float, threshold: float) -> bool:
        return bool(np.isfinite(value) and value >= threshold)

    def at_most(value: float, threshold: float) -> bool:
        return bool(np.isfinite(value) and value <= threshold)

    checks: dict[str, bool | float] = {
        "obstacle_recall": at_least(
            metrics["obstacle_recall"], config.minimum_obstacle_recall
        ),
        "obstacle_precision": at_least(
            metrics["obstacle_precision"], config.minimum_obstacle_precision
        ),
        "target_recall": at_least(
            metrics["target_recall"], config.minimum_target_recall
        ),
        "target_precision": at_least(
            metrics["target_precision"], config.minimum_target_precision
        ),
        "false_target_rate": at_most(
            metrics["false_target_rate"], config.maximum_false_target_rate
        ),
        "latency": at_most(metrics["latency_median_ms"], config.maximum_latency_ms),
    }
    for key in (
        "obstacle_recall",
        "obstacle_precision",
        "target_recall",
        "target_precision",
        "false_target_rate",
        "latency_median_ms",
    ):
        checks[f"{key}_value"] = float(metrics[key])
    return checks


def all_gates_passed(checks: Mapping[str, bool | float]) -> bool:
    return all(bool(checks[name]) for name in REQUIRED_GATES)


# --------------------------------------------------------------------------
# Bölüm değerlendirmesi (dedektör gerekir)
# --------------------------------------------------------------------------


def _load_rgb(path: Path) -> npt.NDArray[np.uint8]:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _grayscale(rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    luminance = rgb.astype(np.float32) @ np.array(
        [0.299, 0.587, 0.114], dtype=np.float32
    )
    gray = np.clip(np.round(luminance), 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def evaluate_detector_split(
    detector: YoloDetector,
    yolo_dir: str | Path,
    split: str,
    config: DetectorQualityConfig,
    *,
    grayscale: bool = False,
    measure_latency_frames: bool = True,
) -> dict[str, float]:
    root = Path(yolo_dir).expanduser().resolve()
    records = [record for record in load_frame_records(root) if record.split == split]
    if not records:
        raise ValueError(f"Bölümde kare yok: {split}")
    conf = min(config.obstacle_conf, config.target_conf)
    frames: list[FrameResult] = []
    latency_frames: list[npt.NDArray[np.uint8]] = []
    for record in records:
        rgb = _load_rgb(root / record.image_file)
        if grayscale:
            rgb = _grayscale(rgb)
        lines = (root / record.label_file).read_text(encoding="utf-8").splitlines()
        ground_truth = parse_yolo_lines(lines, rgb.shape[1], rgb.shape[0])
        predictions = detector.detect(rgb, conf=conf)
        frames.append(
            FrameResult(
                ground_truth=ground_truth,
                predictions=predictions,
                target_range_m=record.target_range_m,
                target_visible_privileged=record.target_visible_privileged,
            )
        )
        if len(latency_frames) < config.latency_frames:
            latency_frames.append(rgb)
    metrics = compute_detection_metrics(frames, config)
    if measure_latency_frames:
        latency = measure_latency(detector, latency_frames, conf=conf)
        metrics.update({f"latency_{key}": value for key, value in latency.items()})
    return metrics


def _ultralytics_map(
    weights: Path, yolo_dir: Path, *, audit: bool, detector_config: DetectorConfig
) -> dict[str, dict[str, float]]:
    """Bilgi amaçlı mAP50 / mAP50-95 (kapı değil)."""

    from ultralytics import YOLO

    model = YOLO(str(weights))
    output: dict[str, dict[str, float]] = {}
    targets = (
        (("audit", "dataset_audit.yaml", "val"),)
        if audit
        else (
            ("validation", "dataset.yaml", "val"),
            ("development_ood", "dataset.yaml", "test"),
        )
    )
    for split, yaml_name, ultralytics_split in targets:
        result = model.val(
            data=str(yolo_dir / yaml_name),
            split=ultralytics_split,
            imgsz=detector_config.imgsz,
            batch=detector_config.batch,
            device=detector_config.device,
            plots=False,
            verbose=False,
            save_json=False,
        )
        output[split] = {
            "map50": float(result.box.map50),
            "map50_95": float(result.box.map),
            "map50_obstacle": float(result.box.ap50[OBSTACLE_CLASS_ID]),
            "map50_target": float(result.box.ap50[TARGET_CLASS_ID]),
        }
    return output


def _detector_for(weights: Path, detector_config: DetectorConfig) -> YoloDetector:
    return YoloDetector(
        weights,
        imgsz=detector_config.imgsz,
        device=detector_config.device,
        iou=detector_config.inference_iou,
        max_det=detector_config.max_det,
    )


def _candidate_score(checks: Mapping[str, Any]) -> float:
    validation = checks["validation"]
    return (
        (1.0 - float(validation["obstacle_recall_value"]))
        + (1.0 - float(validation["obstacle_precision_value"]))
        + (1.0 - float(validation["target_recall_value"]))
        + (1.0 - float(validation["target_precision_value"]))
        + float(validation["false_target_rate_value"])
    )


def _portable_relative_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


@dataclass(frozen=True)
class DetectorQualityArtifacts:
    report: Path
    candidate_weights: Path
    passed: bool


def run_detector_quality(
    yolo_dir: str | Path,
    output_dir: str | Path,
    detector_config: DetectorConfig,
    quality_config: DetectorQualityConfig,
    *,
    progress_callback: Callable[[int, int, int], None] | None = None,
    epochs_override: int | None = None,
) -> DetectorQualityArtifacts:
    """Üç seed eğit, iki geliştirme bölümünde kapıları ölç, adayı seç."""

    detector_config.validate()
    quality_config.validate()
    yolo_root = Path(yolo_dir).expanduser().resolve()
    provenance = yolo_dataset_provenance(yolo_root)
    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Quality çıktı dizini boş değil: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    seed_results: dict[str, dict[str, Any]] = {}
    checks: dict[str, dict[str, dict[str, bool | float]]] = {}
    weights_by_seed: dict[int, Path] = {}
    total = len(quality_config.model_seeds)
    for completed, seed in enumerate(quality_config.model_seeds, start=1):
        artifacts = train_detector(
            yolo_root,
            output_path,
            detector_config,
            seed,
            epochs_override=epochs_override,
        )
        detector = _detector_for(artifacts.weights, detector_config)
        metrics = {
            split: evaluate_detector_split(detector, yolo_root, split, quality_config)
            for split in DEVELOPMENT_SPLITS
        }
        grayscale = evaluate_detector_split(
            detector,
            yolo_root,
            "validation",
            quality_config,
            grayscale=True,
            measure_latency_frames=False,
        )
        map_info = _ultralytics_map(
            artifacts.weights, yolo_root, audit=False, detector_config=detector_config
        )
        seed_checks = {
            split: gate_checks(metrics[split], quality_config)
            for split in DEVELOPMENT_SPLITS
        }
        result = {
            "metrics": metrics,
            "grayscale_validation": grayscale,
            "map_info": map_info,
            "training_metadata": artifacts.metadata.relative_to(output_path).as_posix(),
        }
        (artifacts.run_dir / "quality_metrics.json").write_text(
            json.dumps({"checks": seed_checks, **result}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        seed_results[str(seed)] = result
        checks[str(seed)] = seed_checks
        weights_by_seed[seed] = artifacts.weights
        if progress_callback is not None:
            progress_callback(completed, total, seed)
    return _write_quality_report(
        output_path,
        provenance=provenance,
        detector_config=detector_config,
        quality_config=quality_config,
        checks=checks,
        seed_results=seed_results,
        weights_by_seed=weights_by_seed,
        epochs_override=epochs_override,
        extra={},
    )


def _write_quality_report(
    output_path: Path,
    *,
    provenance: dict[str, Any],
    detector_config: DetectorConfig,
    quality_config: DetectorQualityConfig,
    checks: dict[str, dict[str, dict[str, bool | float]]],
    seed_results: dict[str, dict[str, Any]],
    weights_by_seed: dict[int, Path],
    epochs_override: int | None,
    extra: Mapping[str, Any],
) -> DetectorQualityArtifacts:
    passed = all(
        all_gates_passed(split_checks)
        for seed_checks in checks.values()
        for split_checks in seed_checks.values()
    )
    best_seed = min(
        quality_config.model_seeds, key=lambda seed: _candidate_score(checks[str(seed)])
    )
    candidate = weights_by_seed[best_seed]
    report = {
        "schema_version": QUALITY_REPORT_SCHEMA,
        "kind": APPROVAL_KIND,
        "passed": passed,
        "dataset": provenance,
        "detector_config": asdict(detector_config),
        "quality_config": asdict(quality_config),
        "epochs_override": epochs_override,
        "candidate_seed": best_seed,
        "candidate_weights": candidate.relative_to(output_path).as_posix(),
        "candidate_weights_sha256": file_sha256(candidate),
        "weights_artifacts": {
            str(seed): {
                "path": path.relative_to(output_path).as_posix(),
                "sha256": file_sha256(path),
            }
            for seed, path in weights_by_seed.items()
        },
        "checks": checks,
        "seed_results": seed_results,
        **dict(extra),
    }
    report_path = output_path / "quality_report.json"
    with report_path.open("x", encoding="utf-8") as report_file:
        report_file.write(json.dumps(report, indent=2, sort_keys=True))
    return DetectorQualityArtifacts(
        report=report_path, candidate_weights=candidate, passed=passed
    )


def finalize_detector_quality(
    yolo_dir: str | Path,
    output_dir: str | Path,
    detector_config: DetectorConfig,
    quality_config: DetectorQualityConfig,
    *,
    completed_seeds: Sequence[int],
    decision: str,
) -> DetectorQualityArtifacts:
    """Yarıda kesilmiş bir kalite koşusunu tamamlanmış seed'lerle raporla.

    Yalnız kullanıcı kararıyla kullanılır (ör. 29 Eylül 2026: tek seed).
    Her tamamlanmış seed için eğitim meta verisi, ağırlık SHA-256'sı ve
    dataset kimliği doğrulanır; kapılar kayıtlı metriklerden yeniden hesaplanır
    ve kayıtlı kapılarla karşılaştırılır. Rapor uygulanan seed listesini
    (`quality_config.model_seeds = completed_seeds`), planlanan listeyi ve
    kararı açıkça taşır; audit bu raporu normal raporla aynı kurallarla okur.
    """

    detector_config.validate()
    if not decision.strip():
        raise ValueError("Tamamlanmış seed'lerle raporlama bir karar gerekçesi ister")
    seeds = tuple(int(seed) for seed in completed_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("completed_seeds boş olmayan ve tekrarsız olmalı")
    unknown = sorted(set(seeds) - set(quality_config.model_seeds))
    if unknown:
        raise ValueError(f"Planlanmamış seed raporlanamaz: {unknown}")
    applied = replace(quality_config, model_seeds=seeds)
    applied.validate()
    output_path = Path(output_dir).expanduser().resolve()
    if (output_path / "quality_report.json").exists():
        raise FileExistsError("Bu koşunun quality raporu zaten var")
    provenance = yolo_dataset_provenance(yolo_dir)
    seed_results: dict[str, dict[str, Any]] = {}
    checks: dict[str, dict[str, dict[str, bool | float]]] = {}
    weights_by_seed: dict[int, Path] = {}
    for seed in seeds:
        run_dir = output_path / f"seed_{seed}"
        metadata = json.loads(
            (run_dir / "training_metadata.json").read_text(encoding="utf-8")
        )
        if int(metadata.get("seed", -1)) != seed:
            raise ValueError(f"Eğitim meta verisi seed'i eşleşmiyor: {seed}")
        if metadata.get("epochs") != detector_config.epochs:
            raise ValueError(f"Seed {seed} config epoch sayısıyla eğitilmemiş")
        if metadata.get("dataset") != provenance:
            raise ValueError(f"Seed {seed} dataset kimliği eşleşmiyor")
        weights = (run_dir / str(metadata["weights"])).resolve()
        if file_sha256(weights) != metadata.get("weights_sha256"):
            raise ValueError(f"Seed {seed} ağırlık SHA-256'sı eşleşmiyor")
        stored = json.loads(
            (run_dir / "quality_metrics.json").read_text(encoding="utf-8")
        )
        recomputed = {
            split: gate_checks(stored["metrics"][split], applied)
            for split in DEVELOPMENT_SPLITS
        }
        for split in DEVELOPMENT_SPLITS:
            for gate in REQUIRED_GATES:
                if bool(recomputed[split][gate]) != bool(stored["checks"][split][gate]):
                    raise ValueError(
                        f"Seed {seed} {split} {gate}: kayıtlı kapı yeniden "
                        "hesapla uyuşmuyor"
                    )
        checks[str(seed)] = recomputed
        seed_results[str(seed)] = {
            "metrics": stored["metrics"],
            "grayscale_validation": stored["grayscale_validation"],
            "map_info": stored["map_info"],
            "training_metadata": (run_dir / "training_metadata.json")
            .relative_to(output_path)
            .as_posix(),
        }
        weights_by_seed[seed] = weights
    aborted = sorted(
        int(path.name.removeprefix("seed_"))
        for path in output_path.glob("seed_*")
        if path.is_dir() and int(path.name.removeprefix("seed_")) not in seeds
    )
    return _write_quality_report(
        output_path,
        provenance=provenance,
        detector_config=detector_config,
        quality_config=applied,
        checks=checks,
        seed_results=seed_results,
        weights_by_seed=weights_by_seed,
        epochs_override=None,
        extra={
            "finalized_from_completed_seeds": True,
            "planned_model_seeds": list(quality_config.model_seeds),
            "aborted_seed_directories": aborted,
            "decision": decision,
        },
    )


@dataclass(frozen=True)
class DetectorAuditArtifacts:
    report: Path
    approval_manifest: Path | None
    approved_weights: Path | None
    passed: bool


def run_detector_audit(
    yolo_dir: str | Path,
    quality_report: str | Path,
    output_dir: str | Path,
) -> DetectorAuditArtifacts:
    """Adayı audit bölümünde bir kez ölç; geçerse onay manifestini yaz."""

    yolo_root = Path(yolo_dir).expanduser().resolve()
    provenance = yolo_dataset_provenance(yolo_root)
    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Audit çıktı dizini boş değil: {output_path}")
    quality_report_path = Path(quality_report).expanduser().resolve()
    quality_root = quality_report_path.parent
    quality = json.loads(quality_report_path.read_text(encoding="utf-8"))
    if quality.get("schema_version") != QUALITY_REPORT_SCHEMA:
        raise ValueError(f"Audit için {QUALITY_REPORT_SCHEMA} gerekli")
    if quality.get("kind") != APPROVAL_KIND:
        raise ValueError("Quality raporu yolo-detector türünde değil")
    if not bool(quality.get("passed")):
        raise ValueError("Başarısız quality raporu audit'e alınamaz")
    if quality.get("epochs_override") is not None:
        raise ValueError("Epoch'u kısaltılmış (smoke) quality raporu audit'e alınamaz")
    if quality.get("dataset") != provenance:
        raise ValueError("Audit dataset provenance quality raporuyla eşleşmiyor")
    candidate_relative = Path(str(quality["candidate_weights"]))
    if candidate_relative.is_absolute():
        raise ValueError("Aday ağırlık yolu göreli olmalı")
    candidate = (quality_root / candidate_relative).resolve()
    if not candidate.is_relative_to(quality_root):
        raise ValueError("Aday ağırlık quality dizini dışına çıkamaz")
    expected_sha256 = str(quality["candidate_weights_sha256"])
    if file_sha256(candidate) != expected_sha256:
        raise ValueError("Aday ağırlık SHA-256 eşleşmiyor")
    detector_config = detector_config_from_raw(dict(quality["detector_config"]))
    quality_config = quality_config_from_raw(quality["quality_config"])
    output_path.mkdir(parents=True, exist_ok=True)

    consumption_path = quality_root / "audit_consumed.json"
    consumption: dict[str, object] = {
        "schema_version": AUDIT_CONSUMPTION_SCHEMA,
        "status": "started",
        "quality_report": quality_report_path.name,
        "quality_report_sha256": file_sha256(quality_report_path),
        "dataset": provenance,
        "audit_output": _portable_relative_path(output_path, quality_root),
    }
    try:
        with consumption_path.open("x", encoding="utf-8") as consumption_file:
            consumption_file.write(json.dumps(consumption, indent=2, sort_keys=True))
    except FileExistsError as error:
        raise RuntimeError(
            "Bu quality raporu audit bölümünü daha önce tüketti; yeni audit için "
            "yeni dataset ve yeni quality koşusu gerekli"
        ) from error

    detector = _detector_for(candidate, detector_config)
    metrics = evaluate_detector_split(detector, yolo_root, "audit", quality_config)
    grayscale = evaluate_detector_split(
        detector,
        yolo_root,
        "audit",
        quality_config,
        grayscale=True,
        measure_latency_frames=False,
    )
    map_info = _ultralytics_map(
        candidate, yolo_root, audit=True, detector_config=detector_config
    )
    checks = gate_checks(metrics, quality_config)
    passed = all_gates_passed(checks)
    report = {
        "schema_version": AUDIT_REPORT_SCHEMA,
        "kind": APPROVAL_KIND,
        "passed": passed,
        "dataset": provenance,
        "quality_report": _portable_relative_path(quality_report_path, output_path),
        "quality_report_sha256": file_sha256(quality_report_path),
        "candidate_weights": _portable_relative_path(candidate, output_path),
        "candidate_weights_sha256": expected_sha256,
        "detector_config": asdict(detector_config),
        "quality_config": asdict(quality_config),
        "metrics": {
            "audit": metrics,
            "grayscale_audit": grayscale,
            "map_info": map_info,
        },
        "checks": checks,
    }
    report_path = output_path / "audit_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    consumption.update(
        {
            "status": "completed",
            "audit_passed": passed,
            "audit_report": _portable_relative_path(report_path, quality_root),
            "audit_report_sha256": file_sha256(report_path),
        }
    )
    consumption_path.write_text(
        json.dumps(consumption, indent=2, sort_keys=True), encoding="utf-8"
    )
    approval_path: Path | None = None
    approved: Path | None = None
    if passed:
        approval_path = output_path / "approval_manifest.json"
        approval = {
            "schema_version": APPROVAL_MANIFEST_SCHEMA,
            "kind": APPROVAL_KIND,
            "audit_report": report_path.name,
            "audit_report_sha256": file_sha256(report_path),
            "quality_report": _portable_relative_path(quality_report_path, output_path),
            "quality_report_sha256": file_sha256(quality_report_path),
            "approved_weights": _portable_relative_path(candidate, output_path),
            "approved_weights_sha256": expected_sha256,
            "detector_config": asdict(detector_config),
            "dataset": provenance,
        }
        approval_path.write_text(
            json.dumps(approval, indent=2, sort_keys=True), encoding="utf-8"
        )
        approved = candidate
    return DetectorAuditArtifacts(
        report=report_path,
        approval_manifest=approval_path,
        approved_weights=approved,
        passed=passed,
    )


def _require_gates(checks: object, context: str) -> None:
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key in REQUIRED_GATES and isinstance(value, bool):
                    if not value:
                        raise ValueError(
                            f"{context}: dedektör kapısı geçilmemiş: {key}"
                        )
                    found.add(key)
                walk(value)

    walk(checks)
    missing = sorted(set(REQUIRED_GATES) - found)
    if missing:
        raise ValueError(f"{context}: ADR-023 kapıları eksik ({', '.join(missing)})")


OBSTACLE_GEOMETRY_NAMES = ("box", "cylinder", "capsule", "sphere", "ellipsoid")
APPROVAL_BASES = ("audit_passed", "user_decision")


@dataclass(frozen=True)
class ApprovedDetector:
    weights: Path
    config: DetectorConfig
    dataset: dict[str, Any]
    #: `audit_passed` veya `user_decision` (REV-050).
    basis: str
    #: Kullanıcı kararıyla onayda dedektörün kullanılabileceği engel şekilleri;
    #: `None` = audit geçmiş, kısıt yok.
    obstacle_geometries: tuple[str, ...] | None


def _failed_gates(checks: Mapping[str, Any]) -> list[str]:
    return sorted(name for name in REQUIRED_GATES if not bool(checks.get(name)))


def verify_detector_approval(approval_manifest: str | Path) -> ApprovedDetector:
    """Hash zincirini ve kapıları doğrula; ultralytics gerektirmez (fail-closed).

    İki onay temeli vardır. `audit_passed`: kalite ve audit kapılarının hepsi
    geçmiştir. `user_decision` (REV-050): kalite kapıları geçmiş, audit
    koşulmuş fakat geçmemiştir; manifest kararı, bulguyu, audit'te geçmeyen
    kapıları (audit raporuyla birebir) ve dedektörün kullanılabileceği engel
    şekillerini açıkça taşır. İkinci temelde ortam bu şekil kümesini zorlar.
    """

    manifest_path = Path(approval_manifest).expanduser().resolve()
    approval = json.loads(manifest_path.read_text(encoding="utf-8"))
    if approval.get("schema_version") != APPROVAL_MANIFEST_SCHEMA:
        raise ValueError("Desteklenmeyen dedektör onay manifesti sürümü")
    if approval.get("kind") != APPROVAL_KIND:
        raise ValueError("Onay manifesti yolo-detector türünde değil")
    basis = str(approval.get("approval_basis", "audit_passed"))
    if basis not in APPROVAL_BASES:
        raise ValueError(f"Bilinmeyen onay temeli: {basis}")
    artifact_root = manifest_path.parent.parent

    def resolve(key: str) -> Path:
        relative = Path(str(approval[key]))
        if relative.is_absolute():
            raise ValueError(f"Onay artefact yolu göreli olmalı: {key}")
        resolved = (manifest_path.parent / relative).resolve()
        if not resolved.is_relative_to(artifact_root):
            raise ValueError(f"Onay artefact kök dışına çıkıyor: {key}")
        return resolved

    audit_report = resolve("audit_report")
    quality_report = resolve("quality_report")
    weights = resolve("approved_weights")
    for path, hash_key in (
        (audit_report, "audit_report_sha256"),
        (quality_report, "quality_report_sha256"),
        (weights, "approved_weights_sha256"),
    ):
        if file_sha256(path) != str(approval[hash_key]):
            raise ValueError(f"Onay SHA-256 eşleşmiyor: {path.name}")
    audit = json.loads(audit_report.read_text(encoding="utf-8"))
    quality = json.loads(quality_report.read_text(encoding="utf-8"))
    if audit.get("schema_version") != AUDIT_REPORT_SCHEMA:
        raise ValueError("Desteklenmeyen dedektör audit raporu sürümü")
    if quality.get("schema_version") != QUALITY_REPORT_SCHEMA:
        raise ValueError("Desteklenmeyen dedektör quality raporu sürümü")
    if audit.get("kind") != APPROVAL_KIND or quality.get("kind") != APPROVAL_KIND:
        raise ValueError("Onay zinciri yolo-detector türünde değil")
    if not bool(quality.get("passed")):
        raise ValueError("Onay zincirindeki quality raporu başarısız")
    if audit.get("dataset") != approval.get("dataset") or quality.get(
        "dataset"
    ) != approval.get("dataset"):
        raise ValueError("Onay dataset provenance eşleşmiyor")
    _require_gates(quality.get("checks"), "quality")
    geometries: tuple[str, ...] | None = None
    if basis == "audit_passed":
        if not bool(audit.get("passed")):
            raise ValueError("Onay zincirindeki audit raporu başarısız")
        _require_gates({"audit": audit.get("checks")}, "audit")
    else:
        if not str(approval.get("decision", "")).strip():
            raise ValueError("Kullanıcı kararıyla onay karar metni gerektirir")
        if not str(approval.get("finding", "")).strip():
            raise ValueError("Kullanıcı kararıyla onay bulgu kimliği gerektirir")
        audit_checks = audit.get("checks")
        if not isinstance(audit_checks, Mapping):
            raise ValueError("Audit raporunda checks yok")
        if list(approval.get("failed_audit_gates", [])) != _failed_gates(audit_checks):
            raise ValueError(
                "Manifestteki geçmeyen audit kapıları audit raporuyla eşleşmiyor"
            )
        raw_geometries = approval.get("approved_obstacle_geometries")
        if not isinstance(raw_geometries, list) or not raw_geometries:
            raise ValueError("Kullanıcı kararıyla onay engel şekli kümesi gerektirir")
        geometries = tuple(str(name) for name in raw_geometries)
        unknown = sorted(set(geometries) - set(OBSTACLE_GEOMETRY_NAMES))
        if unknown or len(set(geometries)) != len(geometries):
            raise ValueError(
                f"Geçersiz veya tekrarlı engel şekli: {unknown or geometries}"
            )
    config = detector_config_from_raw(dict(approval["detector_config"]))
    return ApprovedDetector(
        weights=weights,
        config=config,
        dataset=dict(approval["dataset"]),
        basis=basis,
        obstacle_geometries=geometries,
    )


def approve_detector_by_decision(
    quality_report: str | Path,
    audit_report: str | Path,
    *,
    decision: str,
    finding: str,
    obstacle_geometries: Sequence[str],
) -> Path:
    """REV-050: audit'i geçmeyen adayı kullanıcı kararıyla, şekil kısıtlı onayla.

    Manifest audit raporunun yanına yazılır; başarısız audit raporu hash
    zincirinde kalır. Audit tüketim kaydı tamamlanmış ve bu audit raporuna
    işaret etmiş olmalıdır.
    """

    if not decision.strip() or not finding.strip():
        raise ValueError("Karar metni ve bulgu kimliği zorunlu")
    geometries = tuple(str(name) for name in obstacle_geometries)
    if not geometries or set(geometries) - set(OBSTACLE_GEOMETRY_NAMES):
        raise ValueError(f"Geçersiz engel şekli kümesi: {geometries}")
    quality_path = Path(quality_report).expanduser().resolve()
    audit_path = Path(audit_report).expanduser().resolve()
    output_path = audit_path.parent
    manifest_path = output_path / "approval_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Onay manifesti zaten var: {manifest_path}")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if quality.get("schema_version") != QUALITY_REPORT_SCHEMA or not bool(
        quality.get("passed")
    ):
        raise ValueError("Geçmiş bir yolo-detector quality raporu gerekli")
    if audit.get("schema_version") != AUDIT_REPORT_SCHEMA:
        raise ValueError("yolo-detector audit raporu gerekli")
    if bool(audit.get("passed")):
        raise ValueError("Audit geçmiş; normal onay manifesti kullanılmalı")
    if audit.get("quality_report_sha256") != file_sha256(quality_path):
        raise ValueError("Audit raporu bu quality raporuna ait değil")
    if audit.get("dataset") != quality.get("dataset"):
        raise ValueError("Audit ve quality dataset kimliği eşleşmiyor")
    consumption = json.loads(
        (quality_path.parent / "audit_consumed.json").read_text(encoding="utf-8")
    )
    if consumption.get("status") != "completed" or consumption.get(
        "audit_report_sha256"
    ) != file_sha256(audit_path):
        raise ValueError("Audit tüketim kaydı bu audit raporuyla tamamlanmamış")
    candidate = (quality_path.parent / str(quality["candidate_weights"])).resolve()
    if file_sha256(candidate) != quality.get("candidate_weights_sha256"):
        raise ValueError("Aday ağırlık SHA-256 eşleşmiyor")
    checks = audit.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("Audit raporunda checks yok")
    approval = {
        "schema_version": APPROVAL_MANIFEST_SCHEMA,
        "kind": APPROVAL_KIND,
        "approval_basis": "user_decision",
        "decision": decision,
        "finding": finding,
        "audit_passed": False,
        "failed_audit_gates": _failed_gates(checks),
        "approved_obstacle_geometries": list(geometries),
        "audit_report": audit_path.name,
        "audit_report_sha256": file_sha256(audit_path),
        "quality_report": _portable_relative_path(quality_path, output_path),
        "quality_report_sha256": file_sha256(quality_path),
        "approved_weights": _portable_relative_path(candidate, output_path),
        "approved_weights_sha256": str(quality["candidate_weights_sha256"]),
        "detector_config": quality["detector_config"],
        "dataset": quality["dataset"],
    }
    with manifest_path.open("x", encoding="utf-8") as manifest_file:
        manifest_file.write(json.dumps(approval, indent=2, sort_keys=True))
    return manifest_path


def load_approved_detector(
    approval_manifest: str | Path, *, device: str | None = None
) -> YoloDetector:
    approved = verify_detector_approval(approval_manifest)
    config = approved.config
    return YoloDetector(
        approved.weights,
        imgsz=config.imgsz,
        device=device or config.device,
        iou=config.inference_iou,
        max_det=config.max_det,
    )


__all__ = [
    "APPROVAL_KIND",
    "APPROVAL_MANIFEST_SCHEMA",
    "AUDIT_REPORT_SCHEMA",
    "QUALITY_REPORT_SCHEMA",
    "REQUIRED_GATES",
    "APPROVAL_BASES",
    "OBSTACLE_GEOMETRY_NAMES",
    "ApprovedDetector",
    "approve_detector_by_decision",
    "DetectorAuditArtifacts",
    "DetectorQualityArtifacts",
    "DetectorQualityConfig",
    "FrameResult",
    "all_gates_passed",
    "box_iou",
    "compute_detection_metrics",
    "evaluate_detector_split",
    "finalize_detector_quality",
    "gate_checks",
    "greedy_match",
    "load_approved_detector",
    "load_detector_quality_configs",
    "quality_config_from_raw",
    "run_detector_audit",
    "run_detector_quality",
    "verify_detector_approval",
]
