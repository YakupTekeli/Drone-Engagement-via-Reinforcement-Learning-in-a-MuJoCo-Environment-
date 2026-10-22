"""Etiketsiz VAE için çoklu-seed kalite, tek seferlik audit ve fail-closed onay.

ADR-014 deseni aynen korunur (dataset -> quality -> audit -> approval, SHA-256
zinciri, tek seferlik audit tüketimi); yalnız kapılar değişir (ADR-022 §3):
etkin latent boyutu, yeniden üretim hatası ve donmuş latent üzerinde eğitilen
doğrusal probe'ların etiketli zincir eşiklerini sağlaması.
"""

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
from ebabil.perception.model import grayscale_actor_batch
from ebabil.perception.probes import ProbeConfig, probe_visual_vae
from ebabil.perception.vae import (
    DEFAULT_VISUAL_VAE_CONFIG,
    VAE_CHECKPOINT_FORMAT_VERSION,
    VisualVAE,
    VisualVAEConfig,
    VisualVAETrainingConfig,
    evaluate_visual_vae,
    load_visual_vae_checkpoint,
    load_visual_vae_config,
    train_visual_vae,
)

QUALITY_REPORT_SCHEMA = "visual-vae-quality-report-v1"
AUDIT_REPORT_SCHEMA = "visual-vae-audit-report-v1"
APPROVAL_MANIFEST_SCHEMA = "visual-vae-approval-manifest-v1"
AUDIT_CONSUMPTION_SCHEMA = "visual-vae-audit-consumption-v1"
APPROVAL_KIND = "visual-vae"
#: ADR-022 §3: bu kapılar approval zincirinde bulunmak VE geçilmiş olmak
#: zorundadır; eksikse yükleyici fail-closed reddeder.
REQUIRED_GATES = (
    "active_latent_units",
    "free_space_mae",
    "target_auroc",
    "target_average_precision",
    "obstacle_map_best_iou",
)
#: Bilgi amaçlı alanlar; geçme kararına girmez.
INFORMATIONAL_SUFFIX = "_value"
LABEL_FREE_METRICS = ("reconstruction_mse", "psnr_db", "kl_nats", "active_latent_units")
PROBE_METRICS = (
    "free_space_mae",
    "target_precision",
    "target_recall",
    "target_auroc",
    "target_average_precision",
    "target_best_f1",
    "target_center_mean_error",
    "obstacle_map_iou",
    "obstacle_map_precision",
    "obstacle_map_recall",
    "obstacle_map_best_iou",
)


@dataclass(frozen=True)
class VisualVAEQualityConfig:
    model_seeds: tuple[int, ...]
    minimum_active_latent_units: int
    maximum_reconstruction_mse: float
    maximum_free_space_mae: float
    #: ADR-022 Revizyon 1 (REV-049): hedef kapısı eşikten bağımsızdır. Sabit
    #: `0.5` eşiğindeki precision/recall raporlanır fakat kapı değildir.
    minimum_target_auroc: float
    minimum_target_average_precision: float
    #: Engel haritası için en iyi eşikte IoU; kalibrasyondan bağımsız. Sabit
    #: `0.5` eşiğindeki IoU (ADR-021 mirası) Revizyon 1b ile kapı olmaktan
    #: çıktı; değeri raporlanır.
    minimum_obstacle_map_best_iou: float
    probe: ProbeConfig

    def validate(self) -> None:
        if not self.model_seeds or any(seed < 0 for seed in self.model_seeds):
            raise ValueError("quality.model_seeds boş olamaz ve negatif seed içeremez")
        if len(set(self.model_seeds)) != len(self.model_seeds):
            raise ValueError("quality.model_seeds yinelenen seed içeremez")
        if self.minimum_active_latent_units < 0:
            raise ValueError("quality.minimum_active_latent_units negatif olamaz")
        for name, value in (
            ("maximum_reconstruction_mse", self.maximum_reconstruction_mse),
            ("maximum_free_space_mae", self.maximum_free_space_mae),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"quality.{name} negatif olmayan sonlu olmalı")
        probabilities = (
            self.minimum_target_auroc,
            self.minimum_target_average_precision,
            self.minimum_obstacle_map_best_iou,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("Probe olasılık eşikleri [0, 1] aralığında olmalı")
        self.probe.validate()


@dataclass(frozen=True)
class VisualVAEQualityArtifacts:
    report: Path
    candidate_checkpoint: Path
    passed: bool


@dataclass(frozen=True)
class VisualVAEAuditArtifacts:
    report: Path
    approval_manifest: Path | None
    approved_checkpoint: Path | None
    passed: bool


def _quality_config_from_raw(raw: Mapping[str, Any]) -> VisualVAEQualityConfig:
    probe_raw = raw.get("probe")
    if isinstance(probe_raw, Mapping):
        probe = ProbeConfig(
            ridge_lambda=float(probe_raw["ridge_lambda"]),
            logistic_l2=float(probe_raw["logistic_l2"]),
            logistic_iterations=int(probe_raw["logistic_iterations"]),
            batch_size=int(probe_raw["batch_size"]),
        )
    else:
        probe = ProbeConfig(
            ridge_lambda=float(raw["probe_ridge_lambda"]),
            logistic_l2=float(raw["probe_logistic_l2"]),
            logistic_iterations=int(raw["probe_logistic_iterations"]),
            batch_size=int(raw["probe_batch_size"]),
        )
    config = VisualVAEQualityConfig(
        model_seeds=tuple(int(seed) for seed in raw["model_seeds"]),
        minimum_active_latent_units=int(raw["minimum_active_latent_units"]),
        maximum_reconstruction_mse=float(raw["maximum_reconstruction_mse"]),
        maximum_free_space_mae=float(raw["maximum_free_space_mae"]),
        minimum_target_auroc=float(raw["minimum_target_auroc"]),
        minimum_target_average_precision=float(raw["minimum_target_average_precision"]),
        minimum_obstacle_map_best_iou=float(raw["minimum_obstacle_map_best_iou"]),
        probe=probe,
    )
    config.validate()
    return config


def load_visual_vae_quality_config(
    path: str | Path = DEFAULT_VISUAL_VAE_CONFIG,
) -> VisualVAEQualityConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Visual VAE quality config bulunamadı: {config_path}")
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)["quality"]
    return _quality_config_from_raw(raw)


def load_default_vae_quality_configs(
    path: str | Path = DEFAULT_VISUAL_VAE_CONFIG,
) -> tuple[VisualVAEConfig, VisualVAETrainingConfig, VisualVAEQualityConfig]:
    model_config, training_config = load_visual_vae_config(path)
    return model_config, training_config, load_visual_vae_quality_config(path)


def _checks(
    label_free: Mapping[str, Any],
    probes: Mapping[str, Any],
    config: VisualVAEQualityConfig,
    split: str,
) -> dict[str, bool | float | int]:
    representation = label_free[split]
    probe = probes[split]
    grayscale = probes[f"grayscale_{split}"]
    active_units = int(representation["active_latent_units"])
    return {
        "active_latent_units": active_units >= config.minimum_active_latent_units,
        "reconstruction_mse": float(representation["reconstruction_mse"])
        <= config.maximum_reconstruction_mse,
        "free_space_mae": float(probe["free_space_mae"])
        <= config.maximum_free_space_mae,
        # ADR-022 Revizyon 1: hedef kapısı eşikten bağımsız çözülebilirlik.
        "target_auroc": float(probe["target_auroc"]) >= config.minimum_target_auroc,
        "target_average_precision": float(probe["target_average_precision"])
        >= config.minimum_target_average_precision,
        "obstacle_map_best_iou": float(probe["obstacle_map_best_iou"])
        >= config.minimum_obstacle_map_best_iou,
        # ADR-014 renk kısayolu kapısıyla aynı ruh: grayscale'de de serbest alan
        # bilgisi çözülebilir kalmalı.
        "grayscale_free_space_mae": float(grayscale["free_space_mae"])
        <= config.maximum_free_space_mae,
        # Bilgi amaçlı sabit eşik değerleri; geçme kararına girmez.
        f"active_latent_units{INFORMATIONAL_SUFFIX}": active_units,
        f"free_space_mae{INFORMATIONAL_SUFFIX}": float(probe["free_space_mae"]),
        f"target_auroc{INFORMATIONAL_SUFFIX}": float(probe["target_auroc"]),
        f"target_average_precision{INFORMATIONAL_SUFFIX}": float(
            probe["target_average_precision"]
        ),
        f"target_precision{INFORMATIONAL_SUFFIX}": float(probe["target_precision"]),
        f"target_recall{INFORMATIONAL_SUFFIX}": float(probe["target_recall"]),
        f"obstacle_map_iou{INFORMATIONAL_SUFFIX}": float(probe["obstacle_map_iou"]),
        f"obstacle_map_best_iou{INFORMATIONAL_SUFFIX}": float(
            probe["obstacle_map_best_iou"]
        ),
    }


def _all_passed(checks: Mapping[str, bool | float | int]) -> bool:
    return all(
        bool(value)
        for name, value in checks.items()
        if not name.endswith(INFORMATIONAL_SUFFIX)
    )


def _metric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _aggregate(
    seed_results: Mapping[str, Mapping[str, Any]], split: str
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in LABEL_FREE_METRICS:
        summary[name] = _metric_summary(
            [
                float(result["label_free"][split][name])
                for result in seed_results.values()
            ]
        )
    for name in PROBE_METRICS:
        summary[f"probe_{name}"] = _metric_summary(
            [float(result["probes"][split][name]) for result in seed_results.values()]
        )
    return summary


def _candidate_score(result: Mapping[str, Any]) -> float:
    probe = result["probes"]["validation"]
    return (
        float(probe["free_space_mae"])
        + (1.0 - float(probe["target_auroc"]))
        + (1.0 - float(probe["obstacle_map_best_iou"]))
    )


def run_visual_vae_quality(
    dataset_dir: str | Path,
    output_dir: str | Path,
    model_config: VisualVAEConfig,
    training_config: VisualVAETrainingConfig,
    quality_config: VisualVAEQualityConfig,
    *,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> VisualVAEQualityArtifacts:
    """Çoklu seed VAE eğit, probe ölç, aggregate et ve kapıyı değerlendir.

    Eğitim adımı privileged dosyalara dokunmaz (`train_visual_vae`); probe
    adımı okur fakat yalnız ölçüm üretir. Bu ayrım seed dizinindeki iki dosyayla
    görünür kılınır: `metrics.json` (etiketsiz) ve `probe_metrics.json`.
    """

    model_config.validate()
    training_config.validate()
    quality_config.validate()
    validate_dataset_leakage(dataset_dir)
    dataset_provenance = vision_dataset_provenance(dataset_dir)
    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Quality çıktı dizini boş değil: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    seed_results: dict[str, dict[str, Any]] = {}
    checkpoints: dict[int, Path] = {}
    total = len(quality_config.model_seeds)
    for completed, seed in enumerate(quality_config.model_seeds, start=1):
        seed_dir = output_path / f"seed_{seed}"
        artifacts = train_visual_vae(
            dataset_dir, seed_dir, model_config, replace(training_config, seed=seed)
        )
        label_free = json.loads(artifacts.metrics.read_text(encoding="utf-8"))
        model = load_visual_vae_checkpoint(artifacts.checkpoint)
        probes = probe_visual_vae(
            model, dataset_dir, ("validation", "development_ood"), quality_config.probe
        )
        (seed_dir / "probe_metrics.json").write_text(
            json.dumps(probes, indent=2, sort_keys=True), encoding="utf-8"
        )
        seed_results[str(seed)] = {"label_free": label_free, "probes": probes}
        checkpoints[seed] = artifacts.checkpoint
        if progress_callback is not None:
            progress_callback(completed, total, seed)
    checks: dict[str, dict[str, dict[str, bool | float | int]]] = {}
    passed = True
    for seed_name, result in seed_results.items():
        checks[seed_name] = {}
        for split in ("validation", "development_ood"):
            split_checks = _checks(
                result["label_free"], result["probes"], quality_config, split
            )
            checks[seed_name][split] = split_checks
            passed = passed and _all_passed(split_checks)
    best_seed = min(
        quality_config.model_seeds,
        key=lambda seed: _candidate_score(seed_results[str(seed)]),
    )
    candidate = checkpoints[best_seed]
    report = {
        "schema_version": QUALITY_REPORT_SCHEMA,
        "kind": APPROVAL_KIND,
        "passed": passed,
        "dataset": asdict(dataset_provenance),
        "quality_config": asdict(quality_config),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "candidate_seed": best_seed,
        "candidate_checkpoint": candidate.relative_to(output_path).as_posix(),
        "candidate_checkpoint_sha256": file_sha256(candidate),
        "checkpoint_artifacts": {
            str(seed): {
                "path": path.relative_to(output_path).as_posix(),
                "sha256": file_sha256(path),
            }
            for seed, path in checkpoints.items()
        },
        "aggregate": {
            split: _aggregate(seed_results, split)
            for split in ("validation", "development_ood")
        },
        "checks": checks,
        "seed_results": seed_results,
    }
    report_path = output_path / "quality_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return VisualVAEQualityArtifacts(
        report=report_path, candidate_checkpoint=candidate, passed=passed
    )


def _portable_relative_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def run_visual_vae_audit(
    dataset_dir: str | Path,
    quality_report: str | Path,
    output_dir: str | Path,
) -> VisualVAEAuditArtifacts:
    """Dondurulmuş adayı audit split'inde bir kez probe'la ve onayla."""

    validate_dataset_leakage(dataset_dir)
    dataset_provenance = vision_dataset_provenance(dataset_dir)
    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Audit çıktı dizini boş değil: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    quality_report_path = Path(quality_report).expanduser().resolve()
    quality_root = quality_report_path.parent
    quality = json.loads(quality_report_path.read_text(encoding="utf-8"))
    if quality.get("schema_version") != QUALITY_REPORT_SCHEMA:
        raise ValueError(f"Audit için {QUALITY_REPORT_SCHEMA} gerekli")
    if quality.get("kind") != APPROVAL_KIND:
        raise ValueError("Quality raporu visual-vae türünde değil")
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
    expected_sha256 = str(quality["candidate_checkpoint_sha256"])
    if file_sha256(candidate) != expected_sha256:
        raise ValueError("Candidate checkpoint SHA-256 eşleşmiyor")
    payload = torch.load(candidate, map_location="cpu", weights_only=True)
    if payload.get("checkpoint_format_version") != VAE_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Candidate checkpoint visual VAE formatında değil")
    if payload.get("dataset_provenance") != asdict(dataset_provenance):
        raise ValueError("Checkpoint dataset provenance eşleşmiyor")
    quality_config_raw = quality.get("quality_config")
    if not isinstance(quality_config_raw, Mapping):
        raise ValueError("Quality raporunda quality_config nesnesi eksik")
    quality_config = _quality_config_from_raw(quality_config_raw)

    consumption_path = quality_root / "audit_consumed.json"
    consumption: dict[str, object] = {
        "schema_version": AUDIT_CONSUMPTION_SCHEMA,
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

    model = load_visual_vae_checkpoint(candidate)
    label_free = json.loads(
        (candidate.parent / "metrics.json").read_text(encoding="utf-8")
    )
    probes = probe_visual_vae(model, dataset_dir, ("audit",), quality_config.probe)
    # Etiketsiz metrikler audit split'inde de ölçülür; trainer yalnız
    # development split'lerini raporladığı için burada üretilir.
    training_config_raw = payload.get("training_config")
    if not isinstance(training_config_raw, Mapping):
        raise ValueError("Checkpoint training_config nesnesi eksik")
    threshold = float(training_config_raw["active_unit_kl_threshold_nats"])
    batch_size = int(training_config_raw["batch_size"])
    audit_label_free = {
        "audit": asdict(
            evaluate_visual_vae(
                model,
                dataset_dir,
                "audit",
                batch_size=batch_size,
                active_unit_kl_threshold_nats=threshold,
            )
        ),
        "grayscale_audit": asdict(
            evaluate_visual_vae(
                model,
                dataset_dir,
                "audit",
                batch_size=batch_size,
                active_unit_kl_threshold_nats=threshold,
                image_transform=grayscale_actor_batch,
            )
        ),
    }
    checks = _checks(audit_label_free, probes, quality_config, "audit")
    passed = _all_passed(checks)
    report = {
        "schema_version": AUDIT_REPORT_SCHEMA,
        "kind": APPROVAL_KIND,
        "passed": passed,
        "dataset": asdict(dataset_provenance),
        "quality_report": _portable_relative_path(quality_report_path, output_path),
        "quality_report_sha256": file_sha256(quality_report_path),
        "candidate_checkpoint": _portable_relative_path(candidate, output_path),
        "candidate_checkpoint_sha256": expected_sha256,
        "quality_config": asdict(quality_config),
        "development_label_free": {
            key: label_free[key]
            for key in ("validation", "development_ood")
            if key in label_free
        },
        "metrics": {"label_free": audit_label_free, "probes": probes},
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
            "schema_version": APPROVAL_MANIFEST_SCHEMA,
            "kind": APPROVAL_KIND,
            "audit_report": report_path.name,
            "audit_report_sha256": file_sha256(report_path),
            "quality_report": _portable_relative_path(quality_report_path, output_path),
            "quality_report_sha256": file_sha256(quality_report_path),
            "approved_checkpoint": _portable_relative_path(candidate, output_path),
            "approved_checkpoint_sha256": expected_sha256,
            "dataset": asdict(dataset_provenance),
        }
        approval_path.write_text(
            json.dumps(approval, indent=2, sort_keys=True), encoding="utf-8"
        )
        approved_checkpoint = candidate
    return VisualVAEAuditArtifacts(
        report=report_path,
        approval_manifest=approval_path,
        approved_checkpoint=approved_checkpoint,
        passed=passed,
    )


def _require_gated_probes(quality: Mapping[str, object]) -> None:
    """ADR-022 §3: probe kapıları zincirde bulunmalı ve geçilmiş olmalı."""

    checks = quality.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("Quality raporunda `checks` bulunamadı")
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key in REQUIRED_GATES and isinstance(value, bool):
                    if not value:
                        raise ValueError(
                            f"Approval zincirinde probe kapısı geçilmemiş: {key}"
                        )
                    found.add(key)
                walk(value)

    walk(checks)
    missing = sorted(set(REQUIRED_GATES) - found)
    if missing:
        raise ValueError(
            "Approval zinciri ADR-022 probe kapılarını içermiyor "
            f"(eksik: {', '.join(missing)})"
        )


def load_approved_visual_vae(approval_manifest: str | Path) -> VisualVAE:
    """Onay manifestindeki hash zincirini doğrulayıp etiketsiz kodlayıcıyı yükle.

    Etiketli encoder zincirinin (`vision-approval-manifest-v1`) manifest'i
    burada reddedilir; iki zincir birbirinin yerine geçemez.
    """

    manifest_path = Path(approval_manifest).expanduser().resolve()
    approval = json.loads(manifest_path.read_text(encoding="utf-8"))
    if approval.get("schema_version") != APPROVAL_MANIFEST_SCHEMA:
        raise ValueError("Desteklenmeyen visual VAE approval manifest sürümü")
    if approval.get("kind") != APPROVAL_KIND:
        raise ValueError("Approval manifest'i visual-vae türünde değil")
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
    if audit.get("schema_version") != AUDIT_REPORT_SCHEMA:
        raise ValueError("Desteklenmeyen visual VAE audit raporu sürümü")
    if quality.get("schema_version") != QUALITY_REPORT_SCHEMA:
        raise ValueError("Desteklenmeyen visual VAE quality raporu sürümü")
    if audit.get("kind") != APPROVAL_KIND or quality.get("kind") != APPROVAL_KIND:
        raise ValueError("Approval zinciri visual-vae türünde değil")
    if not bool(audit.get("passed")) or not bool(quality.get("passed")):
        raise ValueError("Approval zincirindeki kalite raporu başarısız")
    if audit.get("dataset") != approval.get("dataset") or quality.get(
        "dataset"
    ) != approval.get("dataset"):
        raise ValueError("Approval dataset provenance eşleşmiyor")
    _require_gated_probes(quality)
    _require_gated_probes({"checks": {"audit": audit.get("checks")}})
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("dataset_provenance") != approval.get("dataset"):
        raise ValueError("Approved checkpoint dataset provenance eşleşmiyor")
    return load_visual_vae_checkpoint(checkpoint)


__all__ = [
    "APPROVAL_KIND",
    "APPROVAL_MANIFEST_SCHEMA",
    "AUDIT_REPORT_SCHEMA",
    "QUALITY_REPORT_SCHEMA",
    "REQUIRED_GATES",
    "VisualVAEAuditArtifacts",
    "VisualVAEQualityArtifacts",
    "VisualVAEQualityConfig",
    "load_approved_visual_vae",
    "load_default_vae_quality_configs",
    "load_visual_vae_quality_config",
    "run_visual_vae_audit",
    "run_visual_vae_quality",
]
