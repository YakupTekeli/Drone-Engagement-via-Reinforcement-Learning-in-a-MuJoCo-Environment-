"""Çoklu-seed perception kalite koşusu ve açık kabul kapısı."""

from __future__ import annotations

import json
import math
import os
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ebabil.perception.dataset import (
    file_sha256,
    validate_dataset_leakage,
    vision_dataset_provenance,
)
from ebabil.perception.model import (
    DEFAULT_VISION_ENCODER_CONFIG,
    TinyVisionEncoder,
    VisionEncoderConfig,
    VisionTrainingConfig,
    evaluate_vision_encoder,
    evaluate_vision_encoder_ablation,
    load_vision_encoder_checkpoint,
    load_vision_encoder_config,
    train_vision_encoder,
)


@dataclass(frozen=True)
class VisionQualityConfig:
    model_seeds: tuple[int, ...]
    minimum_target_precision: float
    minimum_target_recall: float
    maximum_target_center_error: float
    maximum_free_space_mae: float
    minimum_free_space_improvement: float
    minimum_risk_precision: float
    minimum_risk_recall: float
    minimum_risk_specificity: float
    minimum_obstacle_map_iou: float
    minimum_obstacle_map_precision: float
    minimum_obstacle_map_recall: float
    #: REV-038: `collision_risk` kanalı actor observation'ından çıkarıldı.
    #: Tanım gereği `free_space < collision_risk_threshold`'dı, yani
    #: politikanın zaten sahip olduğu kanalın eşiklenmiş kopyası;
    #: `obstacle_map` (108 hücre) aynı bilgiyi kat kat yüksek çözünürlükte
    #: taşıyor. Metrikler raporlanmaya devam eder fakat `False` iken approval'ı
    #: bloke etmez. `True`, onaylı `v9`/`v10` zincirinin davranışıdır ve
    #: varsayılandır; yeni sürümler açıkça `false` verir.
    gate_collision_risk: bool = True

    def validate(self) -> None:
        if not self.model_seeds or any(seed < 0 for seed in self.model_seeds):
            raise ValueError("quality.model_seeds boş olamaz ve negatif seed içeremez")
        if len(set(self.model_seeds)) != len(self.model_seeds):
            raise ValueError("quality.model_seeds yinelenen seed içeremez")
        probabilities = (
            self.minimum_target_precision,
            self.minimum_target_recall,
            self.minimum_risk_precision,
            self.minimum_risk_recall,
            self.minimum_risk_specificity,
            self.minimum_obstacle_map_iou,
            self.minimum_obstacle_map_precision,
            self.minimum_obstacle_map_recall,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("Target quality eşikleri [0, 1] aralığında olmalı")
        upper_bounds = (
            self.maximum_target_center_error,
            self.maximum_free_space_mae,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in upper_bounds):
            raise ValueError("Maksimum hata eşikleri negatif olmayan sonlu olmalı")
        if (
            not math.isfinite(self.minimum_free_space_improvement)
            or self.minimum_free_space_improvement < 0.0
        ):
            raise ValueError("Minimum free-space iyileşmesi negatif olamaz")


@dataclass(frozen=True)
class VisionQualityArtifacts:
    report: Path
    candidate_checkpoint: Path
    passed: bool


@dataclass(frozen=True)
class VisionAuditArtifacts:
    report: Path
    approval_manifest: Path | None
    approved_checkpoint: Path | None
    passed: bool


def load_vision_quality_config(
    path: str | Path = DEFAULT_VISION_ENCODER_CONFIG,
) -> VisionQualityConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Vision quality config bulunamadı: {config_path}")
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)["quality"]
    config = VisionQualityConfig(
        model_seeds=tuple(int(seed) for seed in raw["model_seeds"]),
        minimum_target_precision=float(raw["minimum_target_precision"]),
        minimum_target_recall=float(raw["minimum_target_recall"]),
        maximum_target_center_error=float(raw["maximum_target_center_error"]),
        maximum_free_space_mae=float(raw["maximum_free_space_mae"]),
        minimum_free_space_improvement=float(raw["minimum_free_space_improvement"]),
        minimum_risk_precision=float(raw["minimum_risk_precision"]),
        minimum_risk_recall=float(raw["minimum_risk_recall"]),
        minimum_risk_specificity=float(raw["minimum_risk_specificity"]),
        minimum_obstacle_map_iou=float(raw["minimum_obstacle_map_iou"]),
        minimum_obstacle_map_precision=float(raw["minimum_obstacle_map_precision"]),
        minimum_obstacle_map_recall=float(raw["minimum_obstacle_map_recall"]),
        gate_collision_risk=bool(raw.get("gate_collision_risk", True)),
    )
    config.validate()
    return config


def _metric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _aggregate_seed_metrics(
    seed_results: dict[str, dict[str, object]], split: str
) -> dict[str, dict[str, float]]:
    metric_names = (
        "target_precision",
        "target_recall",
        "target_center_mean_error",
        "free_space_mae",
        "free_space_mae_left",
        "free_space_mae_center",
        "free_space_mae_right",
        "risk_precision",
        "risk_recall",
        "risk_specificity",
        "risk_balanced_accuracy",
    )
    return {
        metric_name: _metric_summary(
            [
                float(result[split][metric_name])  # type: ignore[index]
                for result in seed_results.values()
            ]
        )
        for metric_name in metric_names
    }


def _quality_checks(
    metrics: Mapping[str, object], config: VisionQualityConfig, split: str
) -> dict[str, bool | float]:
    model_metrics = metrics[split]
    baseline_metrics = metrics[f"color_baseline_{split}"]
    assert isinstance(model_metrics, dict)
    assert isinstance(baseline_metrics, dict)
    free_space_improvement = float(baseline_metrics["free_space_mae"]) - float(
        model_metrics["free_space_mae"]
    )
    checks: dict[str, bool | float] = {
        "target_precision": float(model_metrics["target_precision"])
        >= config.minimum_target_precision,
        "target_recall": float(model_metrics["target_recall"])
        >= config.minimum_target_recall,
        "target_center": float(model_metrics["target_center_mean_error"])
        <= config.maximum_target_center_error,
        "free_space_mae": float(model_metrics["free_space_mae"])
        <= config.maximum_free_space_mae,
        "free_space_improvement": free_space_improvement,
        "free_space_improved_enough": free_space_improvement
        >= config.minimum_free_space_improvement,
        # REV-037: engel haritası politikaya `obstacle_map` olarak verilir
        # (vision-v2); ölçülmemiş bir kanalı actor observation'ına koymak
        # approval zincirinin amacına aykırı olurdu.
        "obstacle_map_iou": float(model_metrics["obstacle_map_iou"])
        >= config.minimum_obstacle_map_iou,
        "obstacle_map_precision": float(model_metrics["obstacle_map_precision"])
        >= config.minimum_obstacle_map_precision,
        "obstacle_map_recall": float(model_metrics["obstacle_map_recall"])
        >= config.minimum_obstacle_map_recall,
    }
    if config.gate_collision_risk:
        checks.update(
            {
                "risk_precision": float(model_metrics["risk_precision"])
                >= config.minimum_risk_precision,
                "risk_recall": float(model_metrics["risk_recall"])
                >= config.minimum_risk_recall,
                "risk_specificity": float(model_metrics["risk_specificity"])
                >= config.minimum_risk_specificity,
            }
        )
    return checks


def _grayscale_obstacle_checks(
    metrics: Mapping[str, object], config: VisionQualityConfig, split: str
) -> dict[str, bool]:
    grayscale_metrics = metrics[f"grayscale_{split}"]
    assert isinstance(grayscale_metrics, dict)
    checks: dict[str, bool] = {
        "grayscale_free_space_mae": float(grayscale_metrics["free_space_mae"])
        <= config.maximum_free_space_mae,
    }
    if config.gate_collision_risk:
        checks.update(
            {
                "grayscale_risk_precision": float(grayscale_metrics["risk_precision"])
                >= config.minimum_risk_precision,
                "grayscale_risk_recall": float(grayscale_metrics["risk_recall"])
                >= config.minimum_risk_recall,
                "grayscale_risk_specificity": float(
                    grayscale_metrics["risk_specificity"]
                )
                >= config.minimum_risk_specificity,
            }
        )
    return checks


def run_vision_quality(
    dataset_dir: str | Path,
    output_dir: str | Path,
    model_config: VisionEncoderConfig,
    training_config: VisionTrainingConfig,
    quality_config: VisionQualityConfig,
    *,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> VisionQualityArtifacts:
    """Aynı dataset üzerinde çoklu seed eğit, aggregate et ve kapıyı değerlendir."""

    model_config.validate()
    training_config.validate()
    quality_config.validate()
    validate_dataset_leakage(dataset_dir)
    dataset_provenance = vision_dataset_provenance(dataset_dir)
    if dataset_provenance.schema_version != "vision-dataset-v4":
        raise ValueError("Quality dataset'i vision-dataset-v4 olmalı")
    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Quality çıktı dizini boş değil: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    seed_results: dict[str, dict[str, object]] = {}
    checkpoints: dict[int, Path] = {}
    total = len(quality_config.model_seeds)
    for completed, seed in enumerate(quality_config.model_seeds, start=1):
        seed_training_config = replace(training_config, seed=seed)
        artifacts = train_vision_encoder(
            dataset_dir,
            output_path / f"seed_{seed}",
            model_config,
            seed_training_config,
        )
        metrics = json.loads(artifacts.metrics.read_text(encoding="utf-8"))
        seed_results[str(seed)] = metrics
        checkpoints[seed] = artifacts.checkpoint
        if progress_callback is not None:
            progress_callback(completed, total, seed)
    checks: dict[str, dict[str, dict[str, bool | float]]] = {}
    passed = True
    for seed_name, metrics in seed_results.items():
        checks[seed_name] = {}
        for split in ("validation", "development_ood"):
            split_checks = _quality_checks(metrics, quality_config, split)
            split_checks.update(
                _grayscale_obstacle_checks(metrics, quality_config, split)
            )
            checks[seed_name][split] = split_checks
            passed = passed and all(
                bool(value)
                for name, value in split_checks.items()
                if name != "free_space_improvement"
            )
    best_seed = min(
        quality_config.model_seeds,
        key=lambda seed: (
            1.0
            - float(seed_results[str(seed)]["validation"]["target_precision"])  # type: ignore[index]
            + 1.0
            - float(seed_results[str(seed)]["validation"]["target_recall"])  # type: ignore[index]
            + float(
                seed_results[str(seed)]["validation"][  # type: ignore[index]
                    "target_center_mean_error"
                ]
            )
            + float(seed_results[str(seed)]["validation"]["free_space_mae"])  # type: ignore[index]
        ),
    )
    checkpoint_artifacts = {
        str(seed): {
            "path": checkpoints[seed].relative_to(output_path).as_posix(),
            "sha256": file_sha256(checkpoints[seed]),
        }
        for seed in quality_config.model_seeds
    }
    candidate_checkpoint = checkpoints[best_seed]
    report = {
        "schema_version": "vision-quality-report-v2",
        "passed": passed,
        "dataset": asdict(dataset_provenance),
        "quality_config": asdict(quality_config),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "candidate_seed": best_seed,
        "candidate_checkpoint": candidate_checkpoint.relative_to(
            output_path
        ).as_posix(),
        "candidate_checkpoint_sha256": file_sha256(candidate_checkpoint),
        "checkpoint_artifacts": checkpoint_artifacts,
        "aggregate": {
            split: _aggregate_seed_metrics(seed_results, split)
            for split in ("validation", "development_ood")
        },
        "checks": checks,
        "seed_results": seed_results,
    }
    report_path = output_path / "quality_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return VisionQualityArtifacts(
        report=report_path,
        candidate_checkpoint=candidate_checkpoint,
        passed=passed,
    )


def load_default_quality_configs(
    path: str | Path = DEFAULT_VISION_ENCODER_CONFIG,
) -> tuple[VisionEncoderConfig, VisionTrainingConfig, VisionQualityConfig]:
    model_config, training_config = load_vision_encoder_config(path)
    return model_config, training_config, load_vision_quality_config(path)


def _quality_config_from_report(raw: Mapping[str, Any]) -> VisionQualityConfig:
    return VisionQualityConfig(
        model_seeds=tuple(int(seed) for seed in raw["model_seeds"]),
        minimum_target_precision=float(raw["minimum_target_precision"]),
        minimum_target_recall=float(raw["minimum_target_recall"]),
        maximum_target_center_error=float(raw["maximum_target_center_error"]),
        maximum_free_space_mae=float(raw["maximum_free_space_mae"]),
        minimum_free_space_improvement=float(raw["minimum_free_space_improvement"]),
        minimum_risk_precision=float(raw["minimum_risk_precision"]),
        minimum_risk_recall=float(raw["minimum_risk_recall"]),
        minimum_risk_specificity=float(raw["minimum_risk_specificity"]),
        minimum_obstacle_map_iou=float(raw["minimum_obstacle_map_iou"]),
        minimum_obstacle_map_precision=float(raw["minimum_obstacle_map_precision"]),
        minimum_obstacle_map_recall=float(raw["minimum_obstacle_map_recall"]),
        gate_collision_risk=bool(raw.get("gate_collision_risk", True)),
    )


def _portable_relative_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def run_vision_audit(
    dataset_dir: str | Path,
    quality_report: str | Path,
    output_dir: str | Path,
) -> VisionAuditArtifacts:
    """Dondurulmuş development adayını audit split'inde bir kez değerlendir."""

    validate_dataset_leakage(dataset_dir)
    dataset_provenance = vision_dataset_provenance(dataset_dir)
    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Audit çıktı dizini boş değil: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    quality_report_path = Path(quality_report).expanduser().resolve()
    quality_root = quality_report_path.parent
    quality = json.loads(quality_report_path.read_text(encoding="utf-8"))
    if quality.get("schema_version") != "vision-quality-report-v2":
        raise ValueError("Audit için vision-quality-report-v2 gerekli")
    if not bool(quality.get("passed")):
        raise ValueError("Başarısız development quality raporu audit'e alınamaz")
    if quality.get("dataset") != asdict(dataset_provenance):
        raise ValueError("Audit dataset provenance quality raporuyla eşleşmiyor")
    candidate_relative = Path(str(quality["candidate_checkpoint"]))
    if candidate_relative.is_absolute():
        raise ValueError("Candidate checkpoint yolu göreli olmalı")
    candidate = (quality_root / candidate_relative).resolve()
    if not candidate.is_relative_to(quality_root):
        raise ValueError("Candidate checkpoint quality dizini dışına çıkamaz")
    expected_checkpoint_sha256 = str(quality["candidate_checkpoint_sha256"])
    if file_sha256(candidate) != expected_checkpoint_sha256:
        raise ValueError("Candidate checkpoint SHA-256 eşleşmiyor")
    checkpoint_payload = torch.load(candidate, map_location="cpu", weights_only=True)
    if checkpoint_payload.get("dataset_provenance") != asdict(dataset_provenance):
        raise ValueError("Checkpoint dataset provenance eşleşmiyor")
    quality_config_raw = quality.get("quality_config")
    if not isinstance(quality_config_raw, dict):
        raise ValueError("Quality raporunda quality_config nesnesi eksik")
    quality_config = _quality_config_from_report(quality_config_raw)
    quality_config.validate()

    consumption_path = quality_root / "audit_consumed.json"
    consumption: dict[str, object] = {
        "schema_version": "vision-audit-consumption-v1",
        "status": "started",
        "quality_report": quality_report_path.name,
        "quality_report_sha256": file_sha256(quality_report_path),
        "dataset": asdict(dataset_provenance),
        "audit_output": _portable_relative_path(output_path, quality_root),
    }
    try:
        with consumption_path.open("x", encoding="utf-8") as consumption_file:
            consumption_file.write(json.dumps(consumption, indent=2, sort_keys=True))
    except FileExistsError as error:
        raise RuntimeError(
            "Bu quality raporu audit split'ini daha önce tüketti; "
            "yeni audit için yeni dataset salt ve quality koşusu gerekli"
        ) from error

    model = load_vision_encoder_checkpoint(candidate)
    model_metrics, baseline_metrics = evaluate_vision_encoder(
        model, dataset_dir, "audit"
    )
    grayscale_metrics = evaluate_vision_encoder_ablation(model, dataset_dir, "audit")
    metrics = {
        "audit": asdict(model_metrics),
        "color_baseline_audit": asdict(baseline_metrics),
        "grayscale_audit": asdict(grayscale_metrics),
    }
    checks = _quality_checks(metrics, quality_config, "audit")
    checks.update(_grayscale_obstacle_checks(metrics, quality_config, "audit"))
    passed = all(
        bool(value)
        for name, value in checks.items()
        if name != "free_space_improvement"
    )

    report = {
        "schema_version": "vision-audit-report-v1",
        "passed": passed,
        "dataset": asdict(dataset_provenance),
        "quality_report": _portable_relative_path(quality_report_path, output_path),
        "quality_report_sha256": file_sha256(quality_report_path),
        "candidate_checkpoint": _portable_relative_path(candidate, output_path),
        "candidate_checkpoint_sha256": expected_checkpoint_sha256,
        "quality_config": asdict(quality_config),
        "metrics": metrics,
        "checks": checks,
    }
    report_path = output_path / "audit_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    consumption["status"] = "completed"
    consumption["audit_passed"] = passed
    consumption["audit_report"] = _portable_relative_path(report_path, quality_root)
    consumption["audit_report_sha256"] = file_sha256(report_path)
    consumption_path.write_text(
        json.dumps(consumption, indent=2, sort_keys=True), encoding="utf-8"
    )
    approval_path: Path | None = None
    approved_checkpoint: Path | None = None
    if passed:
        approval_path = output_path / "approval_manifest.json"
        approval = {
            "schema_version": "vision-approval-manifest-v1",
            "audit_report": report_path.name,
            "audit_report_sha256": file_sha256(report_path),
            "quality_report": _portable_relative_path(quality_report_path, output_path),
            "quality_report_sha256": file_sha256(quality_report_path),
            "approved_checkpoint": _portable_relative_path(candidate, output_path),
            "approved_checkpoint_sha256": expected_checkpoint_sha256,
            "dataset": asdict(dataset_provenance),
        }
        approval_path.write_text(
            json.dumps(approval, indent=2, sort_keys=True), encoding="utf-8"
        )
        approved_checkpoint = candidate
    return VisionAuditArtifacts(
        report=report_path,
        approval_manifest=approval_path,
        approved_checkpoint=approved_checkpoint,
        passed=passed,
    )


def load_approved_vision_encoder(
    approval_manifest: str | Path,
) -> TinyVisionEncoder:
    """Audit onay manifestindeki hash zincirini doğrulayıp encoder'ı yükle."""

    manifest_path = Path(approval_manifest).expanduser().resolve()
    approval = json.loads(manifest_path.read_text(encoding="utf-8"))
    if approval.get("schema_version") != "vision-approval-manifest-v1":
        raise ValueError("Desteklenmeyen vision approval manifest sürümü")
    artifact_root = manifest_path.parent.parent

    def resolve_artifact(key: str) -> Path:
        relative = Path(str(approval[key]))
        if relative.is_absolute():
            raise ValueError(f"Approval artefact yolu göreli olmalı: {key}")
        resolved = (manifest_path.parent / relative).resolve()
        if not resolved.is_relative_to(artifact_root):
            raise ValueError(f"Approval artefact kök dışına çıkıyor: {key}")
        return resolved

    audit_report = resolve_artifact("audit_report")
    quality_report = resolve_artifact("quality_report")
    checkpoint = resolve_artifact("approved_checkpoint")
    for path, hash_key in (
        (audit_report, "audit_report_sha256"),
        (quality_report, "quality_report_sha256"),
        (checkpoint, "approved_checkpoint_sha256"),
    ):
        if file_sha256(path) != str(approval[hash_key]):
            raise ValueError(f"Approval SHA-256 eşleşmiyor: {path.name}")
    audit = json.loads(audit_report.read_text(encoding="utf-8"))
    quality = json.loads(quality_report.read_text(encoding="utf-8"))
    if audit.get("schema_version") != "vision-audit-report-v1":
        raise ValueError("Desteklenmeyen vision audit raporu sürümü")
    if quality.get("schema_version") != "vision-quality-report-v2":
        raise ValueError("Desteklenmeyen vision quality raporu sürümü")
    if not bool(audit.get("passed")) or not bool(quality.get("passed")):
        raise ValueError("Approval zincirindeki kalite raporu başarısız")
    if audit.get("dataset") != approval.get("dataset") or quality.get(
        "dataset"
    ) != approval.get("dataset"):
        raise ValueError("Approval dataset provenance eşleşmiyor")
    _require_gated_obstacle_map(quality)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint_payload.get("dataset_provenance") != approval.get("dataset"):
        raise ValueError("Approved checkpoint dataset provenance eşleşmiyor")
    return load_vision_encoder_checkpoint(checkpoint)


def _require_gated_obstacle_map(quality: Mapping[str, object]) -> None:
    """ADR-021: `obstacle_map` kanalı ölçülmüş bir encoder'dan gelmek zorunda.

    Observation `vision-v2`'de bu kanal encoder'ın engel ısı haritasından
    türetilir ve actor'a verilir. Zincirde harita kapısı yoksa, politikaya
    doğruluğu hiç ölçülmemiş bir kanal verilmiş olur; approval zincirinin
    amacı tam olarak bunu engellemektir. Fail-closed: eksikse yüklemez.
    """

    required = (
        "obstacle_map_iou",
        "obstacle_map_precision",
        "obstacle_map_recall",
    )
    checks = quality.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("Quality raporunda `checks` bulunamadı")
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key in required and isinstance(value, bool):
                    if not value:
                        raise ValueError(
                            f"Approval zincirinde engel haritası kapısı geçilmemiş: {key}"
                        )
                    found.add(key)
                walk(value)

    walk(checks)
    missing = sorted(set(required) - found)
    if missing:
        raise ValueError(
            "Approval zinciri engel haritası kapısını içermiyor, fakat "
            "vision-v2 observation'ı `obstacle_map` kanalını actor'a veriyor "
            f"(eksik kapılar: {', '.join(missing)}). Encoder'ı ADR-021 "
            "kapılarıyla yeniden onaylayın"
        )
