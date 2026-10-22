"""Donmuş VAE latent'i üzerinde yalnız ölçüm amaçlı doğrusal probe'lar (ADR-022).

Bu modül privileged etiketleri **okur** ve bilinçli olarak `vae.py`'den
ayrıdır. Ürettiği hiçbir ağırlık actor'a, checkpoint'e veya observation'a
gitmez; çıktısı yalnız kalite raporudur. Sorduğu soru şudur: etiketsiz
kodlayıcının latent'i, etiketli zincirin kapılarının (ADR-014/021) istediği
engel ve hedef bilgisini doğrusal olarak çözülebilir biçimde taşıyor mu?

ADR-022 Revizyon 1 (REV-049): sabit `0.5` eşiğindeki recall/precision probe'un
**kalibrasyonunu** ölçer, latent'in bilgisini değil. Bu yüzden eşikten
bağımsız ölçümler (AUROC, average precision, en iyi F1, en iyi eşikte harita
IoU) de üretilir ve kapı bunlara bağlanır; sabit eşik değerleri raporlanmaya
devam eder.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import DataLoader

from ebabil.perception.dataset import Split
from ebabil.perception.model import (
    HEATMAP_HEIGHT,
    HEATMAP_WIDTH,
    PerceptionMetrics,
    VisionNpzDataset,
    compute_metrics,
    grayscale_actor_batch,
)
from ebabil.perception.vae import VisualVAE

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
#: En iyi F1 ve en iyi harita eşiği taramalarında kullanılan ızgaralar.
F1_THRESHOLDS: FloatArray = np.linspace(0.05, 0.95, 91)
MAP_THRESHOLDS: FloatArray = np.linspace(0.10, 0.90, 17)


@dataclass(frozen=True)
class ProbeConfig:
    """Doğrusal probe hiperparametreleri; sonuç kalite raporuna yazılır."""

    ridge_lambda: float
    logistic_l2: float
    logistic_iterations: int
    batch_size: int

    def validate(self) -> None:
        for name, value in (
            ("ridge_lambda", self.ridge_lambda),
            ("logistic_l2", self.logistic_l2),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"probe.{name} negatif olmayan ve sonlu olmalı")
        if self.logistic_iterations <= 0 or self.batch_size <= 0:
            raise ValueError("probe iterasyon ve batch sayıları pozitif olmalı")


@dataclass(frozen=True)
class ProbeLabels:
    """Privileged etiketler; yalnız ölçüm için bellekte tutulur."""

    target_visible: BoolArray
    target_center_xy: FloatArray
    free_space: FloatArray
    obstacle_map: FloatArray


@dataclass(frozen=True)
class LinearProbes:
    """Standardize latent üzerinde bias'lı doğrusal ağırlıklar, shape `(D + 1, K)`."""

    latent_mean: FloatArray
    latent_std: FloatArray
    free_space_weight: FloatArray
    center_weight: FloatArray
    target_weight: FloatArray
    obstacle_weight: FloatArray


@dataclass(frozen=True)
class ProbePredictions:
    target_scores: FloatArray
    target_center_xy: FloatArray
    free_space: FloatArray
    obstacle_map: FloatArray


@dataclass(frozen=True)
class ThresholdFreeMetrics:
    """Eşikten bağımsız çözülebilirlik ölçümleri (ADR-022 Revizyon 1)."""

    target_auroc: float
    target_average_precision: float
    target_best_f1: float
    target_best_f1_recall: float
    target_best_f1_precision: float
    target_best_f1_threshold: float
    obstacle_map_best_iou: float
    obstacle_map_best_iou_threshold: float


def collect_probe_data(
    model: VisualVAE,
    dataset_dir: str | Path,
    split: Split,
    *,
    batch_size: int,
    image_transform: Callable[[Tensor], Tensor] | None = None,
) -> tuple[FloatArray, ProbeLabels]:
    """Bir split'in latent ortalamalarını ve privileged etiketlerini topla."""

    if batch_size <= 0:
        raise ValueError("Probe batch_size pozitif olmalı")
    dataset = VisionNpzDataset(dataset_dir, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    latents: list[FloatArray] = []
    visible: list[BoolArray] = []
    centers: list[FloatArray] = []
    free_space: list[FloatArray] = []
    maps: list[FloatArray] = []
    with torch.no_grad():
        for images, target_visible, target_center, targets_free, _, mask in loader:
            batch = image_transform(images) if image_transform is not None else images
            mean, _ = model.encode(batch)
            latents.append(mean.double().cpu().numpy())
            visible.append(target_visible.numpy() > 0.5)
            centers.append(target_center.double().numpy())
            free_space.append(targets_free.double().numpy())
            pooled = functional.adaptive_max_pool2d(
                mask.unsqueeze(1), (HEATMAP_HEIGHT, HEATMAP_WIDTH)
            ).squeeze(1)
            maps.append(pooled.double().numpy())
    return (
        np.concatenate(latents),
        ProbeLabels(
            target_visible=np.concatenate(visible),
            target_center_xy=np.concatenate(centers),
            free_space=np.concatenate(free_space),
            obstacle_map=np.concatenate(maps),
        ),
    )


def _augment(features: FloatArray) -> FloatArray:
    return np.concatenate(
        (features, np.ones((features.shape[0], 1), dtype=np.float64)), axis=1
    )


def _fit_ridge(
    features: FloatArray, targets: FloatArray, ridge_lambda: float
) -> FloatArray:
    """Kapalı form ridge regresyon; bias sütunu düzenlileştirilmez."""

    gram = features.T @ features
    regularizer = ridge_lambda * np.eye(features.shape[1], dtype=np.float64)
    regularizer[-1, -1] = 0.0
    return np.linalg.solve(gram + regularizer, features.T @ targets)


def _fit_logistic(
    features: FloatArray, targets: FloatArray, l2: float, iterations: int
) -> FloatArray:
    """Sıfırdan başlayan deterministik L-BFGS lojistik regresyon."""

    inputs = torch.tensor(features, dtype=torch.float64)
    labels = torch.tensor(targets, dtype=torch.float64)
    weights = torch.zeros(
        (features.shape[1], targets.shape[1]), dtype=torch.float64, requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [weights],
        max_iter=iterations,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=20,
    )

    def closure() -> float:
        optimizer.zero_grad()
        loss = (
            functional.binary_cross_entropy_with_logits(inputs @ weights, labels)
            + l2 * weights[:-1].square().sum()
        )
        loss.backward()
        return float(loss.detach())

    optimizer.step(closure)
    return weights.detach().numpy()


def fit_linear_probes(
    latents: FloatArray, labels: ProbeLabels, config: ProbeConfig
) -> LinearProbes:
    config.validate()
    if latents.ndim != 2 or latents.shape[0] != labels.target_visible.shape[0]:
        raise ValueError("Latent ve etiket sayıları eşleşmeli")
    mean = latents.mean(axis=0)
    std = latents.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    features = _augment((latents - mean) / std)
    visible = labels.target_visible
    center_weight = (
        _fit_ridge(
            features[visible], labels.target_center_xy[visible], config.ridge_lambda
        )
        if bool(visible.any())
        else np.zeros((features.shape[1], 2), dtype=np.float64)
    )
    return LinearProbes(
        latent_mean=mean,
        latent_std=std,
        free_space_weight=_fit_ridge(features, labels.free_space, config.ridge_lambda),
        center_weight=center_weight,
        target_weight=_fit_logistic(
            features,
            visible.astype(np.float64).reshape(-1, 1),
            config.logistic_l2,
            config.logistic_iterations,
        ),
        obstacle_weight=_fit_logistic(
            features,
            labels.obstacle_map.reshape(latents.shape[0], -1),
            config.logistic_l2,
            config.logistic_iterations,
        ),
    )


def _sigmoid(logits: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-logits))


def predict_linear_probes(
    probes: LinearProbes, latents: FloatArray
) -> ProbePredictions:
    features = _augment((latents - probes.latent_mean) / probes.latent_std)
    return ProbePredictions(
        target_scores=_sigmoid(features @ probes.target_weight)[:, 0],
        target_center_xy=np.clip(features @ probes.center_weight, -1.0, 1.0),
        free_space=np.clip(features @ probes.free_space_weight, 0.0, 1.0),
        obstacle_map=_sigmoid(features @ probes.obstacle_weight).reshape(
            latents.shape[0], HEATMAP_HEIGHT, HEATMAP_WIDTH
        ),
    )


def evaluate_linear_probes(
    probes: LinearProbes, latents: FloatArray, labels: ProbeLabels
) -> PerceptionMetrics:
    """Etiketli zincirle aynı metrik tanımları (`compute_metrics`, eşik `0.5`)."""

    predictions = predict_linear_probes(probes, latents)
    return compute_metrics(
        target_scores=predictions.target_scores,
        predicted_centers=predictions.target_center_xy,
        predicted_free_space=predictions.free_space,
        target_visible=labels.target_visible,
        target_centers=labels.target_center_xy,
        target_free_space=labels.free_space,
        predicted_obstacle_map=predictions.obstacle_map,
        target_obstacle_map=labels.obstacle_map,
    )


def binary_auroc(scores: FloatArray, labels: BoolArray) -> float:
    """Bağlarda ortalama sıra kullanan Mann-Whitney AUROC; tek sınıfta `0.5`."""

    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    n_positive = int(positive.sum())
    n_negative = int(positive.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start
        while end + 1 < scores.size and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        ranks[order[start : end + 1]] = 0.5 * (start + end) + 1.0
        start = end + 1
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def average_precision(scores: FloatArray, labels: BoolArray) -> float:
    """Sıralama tabanlı average precision; pozitif yoksa `0.0`."""

    positive = np.asarray(labels, dtype=bool)
    if not positive.any():
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    hits = positive[order].astype(np.float64)
    precision_at_k = np.cumsum(hits) / np.arange(1, hits.size + 1)
    return float((precision_at_k * hits).sum() / hits.sum())


def threshold_free_metrics(
    predictions: ProbePredictions, labels: ProbeLabels
) -> ThresholdFreeMetrics:
    """AUROC, AP, en iyi F1 ve en iyi eşikte harita IoU."""

    visible = np.asarray(labels.target_visible, dtype=bool)
    scores = predictions.target_scores
    best_f1 = (0.0, 0.0, 0.0, 0.5)
    for threshold in F1_THRESHOLDS:
        predicted = scores >= threshold
        true_positive = float(np.count_nonzero(predicted & visible))
        precision = true_positive / max(float(predicted.sum()), 1.0)
        recall = true_positive / max(float(visible.sum()), 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1[0]:
            best_f1 = (f1, recall, precision, float(threshold))
    actual_cells = labels.obstacle_map.reshape(visible.size, -1) >= 0.5
    predicted_map = predictions.obstacle_map.reshape(visible.size, -1)
    best_iou = (0.0, 0.5)
    for threshold in MAP_THRESHOLDS:
        predicted_cells = predicted_map >= threshold
        union = float(np.count_nonzero(predicted_cells | actual_cells))
        iou = (
            float(np.count_nonzero(predicted_cells & actual_cells)) / union
            if union > 0.0
            else 1.0
        )
        if iou > best_iou[0]:
            best_iou = (iou, float(threshold))
    return ThresholdFreeMetrics(
        target_auroc=binary_auroc(scores, visible),
        target_average_precision=average_precision(scores, visible),
        target_best_f1=best_f1[0],
        target_best_f1_recall=best_f1[1],
        target_best_f1_precision=best_f1[2],
        target_best_f1_threshold=best_f1[3],
        obstacle_map_best_iou=best_iou[0],
        obstacle_map_best_iou_threshold=best_iou[1],
    )


def evaluate_probe_split(
    probes: LinearProbes, latents: FloatArray, labels: ProbeLabels
) -> dict[str, float | int]:
    """Sabit eşik (`compute_metrics`) ve eşikten bağımsız ölçümleri tek sözlükte ver."""

    predictions = predict_linear_probes(probes, latents)
    fixed = compute_metrics(
        target_scores=predictions.target_scores,
        predicted_centers=predictions.target_center_xy,
        predicted_free_space=predictions.free_space,
        target_visible=labels.target_visible,
        target_centers=labels.target_center_xy,
        target_free_space=labels.free_space,
        predicted_obstacle_map=predictions.obstacle_map,
        target_obstacle_map=labels.obstacle_map,
    )
    return asdict(fixed) | asdict(threshold_free_metrics(predictions, labels))


def probe_visual_vae(
    model: VisualVAE,
    dataset_dir: str | Path,
    splits: tuple[Split, ...],
    config: ProbeConfig,
) -> dict[str, Any]:
    """Train latent'inde probe'ları eğit, istenen split'lerde nominal + grayscale ölçüm yap."""

    config.validate()
    if not splits or "train" in splits:
        raise ValueError(
            "Probe değerlendirme split'leri train dışı ve boş olmayan olmalı"
        )
    train_latents, train_labels = collect_probe_data(
        model, dataset_dir, "train", batch_size=config.batch_size
    )
    probes = fit_linear_probes(train_latents, train_labels, config)
    report: dict[str, Any] = {
        "probe_config": asdict(config),
        "train_sample_count": int(train_latents.shape[0]),
    }
    for split in splits:
        latents, labels = collect_probe_data(
            model, dataset_dir, split, batch_size=config.batch_size
        )
        report[split] = evaluate_probe_split(probes, latents, labels)
        grayscale_latents, grayscale_labels = collect_probe_data(
            model,
            dataset_dir,
            split,
            batch_size=config.batch_size,
            image_transform=grayscale_actor_batch,
        )
        report[f"grayscale_{split}"] = evaluate_probe_split(
            probes, grayscale_latents, grayscale_labels
        )
    return report


__all__ = [
    "LinearProbes",
    "ProbeConfig",
    "ProbeLabels",
    "ProbePredictions",
    "ThresholdFreeMetrics",
    "average_precision",
    "binary_auroc",
    "collect_probe_data",
    "evaluate_linear_probes",
    "evaluate_probe_split",
    "fit_linear_probes",
    "predict_linear_probes",
    "probe_visual_vae",
    "threshold_free_metrics",
]
