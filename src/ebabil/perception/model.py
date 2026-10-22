"""Hedef ve serbest alan için küçük RGB encoder eğitim/değerlendirme hattı."""

from __future__ import annotations

import json
import math
import time
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ebabil.perception.dataset import (
    Split,
    load_manifest,
    validate_dataset_leakage,
    vision_dataset_provenance,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VISION_ENCODER_CONFIG = (
    PACKAGE_ROOT / "configs" / "training" / "vision_encoder.toml"
)
CHECKPOINT_FORMAT_VERSION = "ebabil-vision-checkpoint-v9"
HEATMAP_HEIGHT = 9
HEATMAP_WIDTH = 12


@dataclass(frozen=True)
class VisionEncoderConfig:
    schema_version: str
    input_width_px: int
    input_height_px: int
    latent_size: int
    adaptive_exposure_target_mean: float
    adaptive_exposure_min_gain: float
    adaptive_exposure_max_gain: float
    collision_risk_score_threshold: float
    #: REV-021: risk head kapasitesi. `0` tek doğrusal katmandır ve onaylı
    #: `vision-encoder-v9` bu değerle birebir yüklenir. Pozitif değer gizli
    #: katmanlı bir MLP kurar ve `vision-encoder-v10` şemasını gerektirir.
    risk_head_hidden_size: int = 0

    def validate(self) -> None:
        if self.schema_version not in (
            "vision-encoder-v9",
            "vision-encoder-v10",
            "vision-encoder-v11",
        ):
            raise ValueError(
                "model.schema_version vision-encoder-v9, -v10 veya -v11 olmalı"
            )
        if self.risk_head_hidden_size < 0:
            raise ValueError("model.risk_head_hidden_size negatif olamaz")
        if self.risk_head_hidden_size > 0 and self.schema_version not in (
            "vision-encoder-v10",
            "vision-encoder-v11",
        ):
            raise ValueError(
                "risk_head_hidden_size > 0 yalnız vision-encoder-v10 ile "
                "kullanılabilir; v9 mimarisi tek doğrusal risk head'idir"
            )
        if min(self.input_width_px, self.input_height_px) < 16:
            raise ValueError("Encoder giriş boyutları en az 16 piksel olmalı")
        if self.latent_size <= 0:
            raise ValueError("model.latent_size pozitif olmalı")
        exposure_values = (
            self.adaptive_exposure_target_mean,
            self.adaptive_exposure_min_gain,
            self.adaptive_exposure_max_gain,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in exposure_values):
            raise ValueError("Adaptive exposure değerleri pozitif ve sonlu olmalı")
        if self.adaptive_exposure_target_mean > 1.0:
            raise ValueError("Adaptive exposure target mean [0, 1] içinde olmalı")
        if self.adaptive_exposure_min_gain > self.adaptive_exposure_max_gain:
            raise ValueError("Adaptive exposure minimum gain maximumu aşamaz")
        if not 0.0 < self.collision_risk_score_threshold < 1.0:
            raise ValueError("Risk score threshold (0, 1) aralığında olmalı")


@dataclass(frozen=True)
class VisionTrainingConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    target_visibility_weight: float
    target_center_weight: float
    free_space_weight: float
    target_mask_weight: float
    obstacle_mask_weight: float
    mask_positive_weight: float
    collision_risk_weight: float
    collision_risk_positive_weight: float
    collision_risk_threshold: float
    grayscale_augmentation_probability: float
    latency_warmup_runs: int
    latency_measurement_runs: int

    def validate(self) -> None:
        if self.seed < 0 or min(self.epochs, self.batch_size) <= 0:
            raise ValueError("Training seed negatif olamaz; epoch/batch pozitif olmalı")
        positive = (
            self.learning_rate,
            self.target_visibility_weight,
            self.target_center_weight,
            self.free_space_weight,
            self.target_mask_weight,
            self.obstacle_mask_weight,
            self.mask_positive_weight,
            self.collision_risk_weight,
            self.collision_risk_positive_weight,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Training oranları pozitif ve sonlu olmalı")
        if self.latency_warmup_runs < 0 or self.latency_measurement_runs <= 0:
            raise ValueError("Latency tekrar sayıları geçersiz")
        if not 0.0 < self.collision_risk_threshold < 1.0:
            raise ValueError("collision_risk_threshold (0, 1) aralığında olmalı")
        if not 0.0 <= self.grayscale_augmentation_probability <= 1.0:
            raise ValueError("grayscale augmentation olasılığı [0, 1] içinde olmalı")


@dataclass(frozen=True)
class PerceptionMetrics:
    target_precision: float
    target_recall: float
    target_center_mean_error: float
    free_space_mae: float
    free_space_mae_left: float
    free_space_mae_center: float
    free_space_mae_right: float
    risk_precision: float
    risk_recall: float
    risk_specificity: float
    risk_balanced_accuracy: float
    obstacle_map_iou: float
    obstacle_map_precision: float
    obstacle_map_recall: float
    sample_count: int


@dataclass(frozen=True)
class VisionTrainingArtifacts:
    checkpoint: Path
    metrics: Path


class VisionEncoderOutput(NamedTuple):
    latent: Tensor
    target_logit: Tensor
    target_center_xy: Tensor
    free_space: Tensor
    target_heatmap_logits: Tensor
    obstacle_heatmap_logits: Tensor
    collision_risk_logits: Tensor


VisionSample = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


def preprocess_actor_rgb(rgb: np.ndarray) -> Tensor:
    """Actor uint8 HWC RGB karesini tek kanonik CHW [0, 1] tensöre dönüştür."""

    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        raise ValueError("Actor RGB uint8 olmalı")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Actor RGB shape'i HWC ve üç kanallı olmalı")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("Actor RGB yüksekliği ve genişliği pozitif olmalı")
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)


def grayscale_actor_batch(images: Tensor) -> Tensor:
    """Renk kısa yolunu ölçmek için BCHW RGB batch'ini luminance'a indir."""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Grayscale ablation girdisi BCHW RGB olmalı")
    if not torch.is_floating_point(images):
        raise ValueError("Grayscale ablation girdisi floating point olmalı")
    weights = images.new_tensor((0.2126, 0.7152, 0.0722)).reshape(1, 3, 1, 1)
    luminance = (images * weights).sum(dim=1, keepdim=True)
    return luminance.expand(-1, 3, -1, -1)


class VisionNpzDataset(Dataset[VisionSample]):
    """Actor RGB ile ayrı privileged supervision arşivini eğitimde eşleştir."""

    def __init__(self, dataset_dir: str | Path, split: Split) -> None:
        self.root = Path(dataset_dir).expanduser().resolve()
        self.records = tuple(
            record
            for record in load_manifest(self.root / "manifest.jsonl")
            if record.split == split
        )
        if not self.records:
            raise ValueError(f"Dataset split'i boş: {split}")

    def __len__(self) -> int:
        return len(self.records)

    def _safe_path(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Dataset path kök dışına çıkıyor: {relative_path}")
        return path

    def __getitem__(self, index: int) -> VisionSample:
        record = self.records[index]
        with np.load(
            self._safe_path(record.actor_file), allow_pickle=False
        ) as actor_payload:
            rgb = np.asarray(actor_payload["rgb"])
        with np.load(
            self._safe_path(record.privileged_file), allow_pickle=False
        ) as privileged_payload:
            visible = bool(privileged_payload["target_visible"])
            center = np.asarray(
                privileged_payload["target_center_xy"], dtype=np.float32
            )
            free_space = np.asarray(privileged_payload["free_space"], dtype=np.float32)
            target_mask = np.asarray(
                privileged_payload["target_mask"], dtype=np.float32
            )
            obstacle_mask = np.asarray(
                privileged_payload["obstacle_mask"], dtype=np.float32
            )
        if center.shape != (2,) or free_space.shape != (3,):
            raise ValueError("Privileged yardımcı hedef shape'i geçersiz")
        if target_mask.shape != rgb.shape[:2] or obstacle_mask.shape != rgb.shape[:2]:
            raise ValueError("Privileged mask ve RGB boyutları eşleşmeli")
        image = preprocess_actor_rgb(rgb)
        return (
            image,
            torch.tensor(float(visible), dtype=torch.float32),
            torch.from_numpy(center.copy()),
            torch.from_numpy(free_space.copy()),
            torch.from_numpy(target_mask.copy()),
            torch.from_numpy(obstacle_mask.copy()),
        )


class TinyVisionEncoder(nn.Module):
    """Orin profiling öncesi küçük, doğrudan dışa aktarılabilir CNN baseline."""

    heatmap_x: Tensor
    heatmap_y: Tensor

    def __init__(self, config: VisionEncoderConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.features = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((3, 4))
        self.latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 3 * 4, config.latent_size),
            nn.ReLU(),
        )
        self.target_visibility_head = nn.Linear(config.latent_size + 2, 1)
        self.free_space_head = nn.Linear(config.latent_size, 3)
        self.collision_risk_head: nn.Module = (
            nn.Linear(config.latent_size, 3)
            if config.risk_head_hidden_size == 0
            else nn.Sequential(
                nn.Linear(config.latent_size, config.risk_head_hidden_size),
                nn.ReLU(),
                nn.Linear(config.risk_head_hidden_size, 3),
            )
        )
        self.target_heatmap_head = nn.Conv2d(32, 1, kernel_size=1)
        self.obstacle_heatmap_head = nn.Conv2d(32, 1, kernel_size=1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, HEATMAP_HEIGHT),
            torch.linspace(-1.0, 1.0, HEATMAP_WIDTH),
            indexing="ij",
        )
        self.register_buffer("heatmap_x", grid_x.flatten(), persistent=False)
        self.register_buffer("heatmap_y", grid_y.flatten(), persistent=False)

    def forward(self, images: Tensor) -> VisionEncoderOutput:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Encoder girdisi BCHW ve üç kanallı olmalı")
        if not torch.is_floating_point(images):
            raise ValueError("Encoder girdisi floating point olmalı")
        if not bool(torch.isfinite(images).all()):
            raise ValueError("Encoder girdisi sonlu olmalı")
        if images.numel() == 0 or bool(torch.any((images < 0.0) | (images > 1.0))):
            raise ValueError("Encoder girdisi [0, 1] aralığında olmalı")
        resized = functional.interpolate(
            images,
            size=(self.config.input_height_px, self.config.input_width_px),
            mode="bilinear",
            align_corners=False,
        )
        channel_mean = resized.mean(dim=(2, 3), keepdim=True)
        exposure_gain = torch.clamp(
            self.config.adaptive_exposure_target_mean / channel_mean.clamp_min(1e-4),
            min=self.config.adaptive_exposure_min_gain,
            max=self.config.adaptive_exposure_max_gain,
        )
        normalized = torch.clamp(resized * exposure_gain, 0.0, 1.0)
        feature_map = self.features(normalized)
        pooled_features = self.spatial_pool(feature_map)
        latent = self.latent(pooled_features)
        heatmap_features = functional.adaptive_avg_pool2d(
            feature_map, (HEATMAP_HEIGHT, HEATMAP_WIDTH)
        )
        target_heatmap_logits = self.target_heatmap_head(heatmap_features).squeeze(1)
        obstacle_heatmap_logits = self.obstacle_heatmap_head(heatmap_features).squeeze(
            1
        )
        target_presence_features = torch.stack(
            (
                target_heatmap_logits.amax(dim=(1, 2)),
                target_heatmap_logits.mean(dim=(1, 2)),
            ),
            dim=1,
        )
        target_logit = self.target_visibility_head(
            torch.cat((latent, target_presence_features), dim=1)
        ).squeeze(-1)
        target_weights = torch.softmax(target_heatmap_logits.flatten(1), dim=1)
        target_center_xy = torch.stack(
            (
                (target_weights * self.heatmap_x).sum(dim=1),
                (target_weights * self.heatmap_y).sum(dim=1),
            ),
            dim=1,
        )
        return VisionEncoderOutput(
            latent=latent,
            target_logit=target_logit,
            target_center_xy=target_center_xy,
            free_space=torch.sigmoid(self.free_space_head(latent)),
            target_heatmap_logits=target_heatmap_logits,
            obstacle_heatmap_logits=obstacle_heatmap_logits,
            collision_risk_logits=self.collision_risk_head(latent),
        )


def load_vision_encoder_config(
    path: str | Path = DEFAULT_VISION_ENCODER_CONFIG,
) -> tuple[VisionEncoderConfig, VisionTrainingConfig]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Vision encoder config bulunamadı: {config_path}")
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    model_raw = raw["model"]
    training_raw = raw["training"]
    model = VisionEncoderConfig(
        schema_version=str(model_raw["schema_version"]),
        input_width_px=int(model_raw["input_width_px"]),
        input_height_px=int(model_raw["input_height_px"]),
        latent_size=int(model_raw["latent_size"]),
        adaptive_exposure_target_mean=float(model_raw["adaptive_exposure_target_mean"]),
        adaptive_exposure_min_gain=float(model_raw["adaptive_exposure_min_gain"]),
        adaptive_exposure_max_gain=float(model_raw["adaptive_exposure_max_gain"]),
        collision_risk_score_threshold=float(
            model_raw["collision_risk_score_threshold"]
        ),
        risk_head_hidden_size=int(model_raw.get("risk_head_hidden_size", 0)),
    )
    training = VisionTrainingConfig(
        seed=int(training_raw["seed"]),
        epochs=int(training_raw["epochs"]),
        batch_size=int(training_raw["batch_size"]),
        learning_rate=float(training_raw["learning_rate"]),
        target_visibility_weight=float(training_raw["target_visibility_weight"]),
        target_center_weight=float(training_raw["target_center_weight"]),
        free_space_weight=float(training_raw["free_space_weight"]),
        target_mask_weight=float(training_raw["target_mask_weight"]),
        obstacle_mask_weight=float(training_raw["obstacle_mask_weight"]),
        mask_positive_weight=float(training_raw["mask_positive_weight"]),
        collision_risk_weight=float(training_raw["collision_risk_weight"]),
        collision_risk_positive_weight=float(
            training_raw["collision_risk_positive_weight"]
        ),
        collision_risk_threshold=float(training_raw["collision_risk_threshold"]),
        grayscale_augmentation_probability=float(
            training_raw["grayscale_augmentation_probability"]
        ),
        latency_warmup_runs=int(training_raw["latency_warmup_runs"]),
        latency_measurement_runs=int(training_raw["latency_measurement_runs"]),
    )
    model.validate()
    training.validate()
    return model, training


def perception_loss(
    output: VisionEncoderOutput,
    target_visible: Tensor,
    target_center_xy: Tensor,
    free_space: Tensor,
    target_mask: Tensor,
    obstacle_mask: Tensor,
    config: VisionTrainingConfig,
) -> Tensor:
    visibility_loss = functional.binary_cross_entropy_with_logits(
        output.target_logit, target_visible
    )
    visible_count = target_visible.sum().clamp_min(1.0)
    center_per_sample = functional.smooth_l1_loss(
        output.target_center_xy, target_center_xy, reduction="none"
    ).mean(dim=1)
    center_loss = (center_per_sample * target_visible).sum() / visible_count
    free_space_loss = functional.l1_loss(output.free_space, free_space)
    target_heatmap = functional.adaptive_max_pool2d(
        target_mask.unsqueeze(1), (HEATMAP_HEIGHT, HEATMAP_WIDTH)
    ).squeeze(1)
    obstacle_heatmap = functional.adaptive_max_pool2d(
        obstacle_mask.unsqueeze(1), (HEATMAP_HEIGHT, HEATMAP_WIDTH)
    ).squeeze(1)
    positive_weight = torch.as_tensor(
        config.mask_positive_weight,
        device=output.target_heatmap_logits.device,
        dtype=output.target_heatmap_logits.dtype,
    )
    target_mask_loss = functional.binary_cross_entropy_with_logits(
        output.target_heatmap_logits,
        target_heatmap,
        pos_weight=positive_weight,
    )
    obstacle_mask_loss = functional.binary_cross_entropy_with_logits(
        output.obstacle_heatmap_logits,
        obstacle_heatmap,
        pos_weight=positive_weight,
    )
    collision_risk = (free_space < config.collision_risk_threshold).to(
        output.collision_risk_logits.dtype
    )
    # Kaçırılan çarpışma riski (false negative), yanlış alarmdan (false
    # positive) daha pahalıdır; bu nedenle pozitif sınıf ağırlıklandırılır.
    # Ağırlık olmadan risk head'i eğitim prior'ını takip eder ve pozitif oran
    # düştüğünde recall çöker.
    collision_risk_loss = functional.binary_cross_entropy_with_logits(
        output.collision_risk_logits,
        collision_risk,
        pos_weight=torch.tensor(
            config.collision_risk_positive_weight,
            dtype=output.collision_risk_logits.dtype,
        ),
    )
    return (
        config.target_visibility_weight * visibility_loss
        + config.target_center_weight * center_loss
        + config.free_space_weight * free_space_loss
        + config.target_mask_weight * target_mask_loss
        + config.obstacle_mask_weight * obstacle_mask_loss
        + config.collision_risk_weight * collision_risk_loss
    )


def compute_metrics(
    target_scores: np.ndarray,
    predicted_centers: np.ndarray,
    predicted_free_space: np.ndarray,
    target_visible: np.ndarray,
    target_centers: np.ndarray,
    target_free_space: np.ndarray,
    predicted_risk_scores: np.ndarray | None = None,
    risk_threshold: float = 0.25,
    predicted_risk_threshold: float = 0.5,
    predicted_obstacle_map: np.ndarray | None = None,
    target_obstacle_map: np.ndarray | None = None,
    obstacle_map_threshold: float = 0.5,
) -> PerceptionMetrics:
    predicted_visible = np.asarray(target_scores) >= 0.5
    actual_visible = np.asarray(target_visible, dtype=bool)
    true_positive = int(np.count_nonzero(predicted_visible & actual_visible))
    false_positive = int(np.count_nonzero(predicted_visible & ~actual_visible))
    false_negative = int(np.count_nonzero(~predicted_visible & actual_visible))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    if np.any(actual_visible):
        center_error = float(
            np.linalg.norm(
                predicted_centers[actual_visible] - target_centers[actual_visible],
                axis=1,
            ).mean()
        )
    else:
        center_error = 0.0
    free_space_error = np.abs(predicted_free_space - target_free_space)
    free_space_mae_by_region = np.mean(free_space_error, axis=0)
    actual_risk = target_free_space < risk_threshold
    predicted_risk = (
        predicted_free_space < risk_threshold
        if predicted_risk_scores is None
        else predicted_risk_scores >= predicted_risk_threshold
    )
    risk_true_positive = int(np.count_nonzero(predicted_risk & actual_risk))
    risk_false_positive = int(np.count_nonzero(predicted_risk & ~actual_risk))
    risk_false_negative = int(np.count_nonzero(~predicted_risk & actual_risk))
    risk_true_negative = int(np.count_nonzero(~predicted_risk & ~actual_risk))
    risk_precision = risk_true_positive / max(
        risk_true_positive + risk_false_positive, 1
    )
    risk_recall = risk_true_positive / max(risk_true_positive + risk_false_negative, 1)
    risk_specificity = risk_true_negative / max(
        risk_true_negative + risk_false_positive, 1
    )
    map_iou = map_precision = map_recall = float("nan")
    if predicted_obstacle_map is not None and target_obstacle_map is not None:
        predicted_cells = np.asarray(predicted_obstacle_map) >= obstacle_map_threshold
        actual_cells = np.asarray(target_obstacle_map) >= 0.5
        intersection = float(np.sum(predicted_cells & actual_cells))
        union = float(np.sum(predicted_cells | actual_cells))
        predicted_total = float(np.sum(predicted_cells))
        actual_total = float(np.sum(actual_cells))
        map_iou = intersection / union if union > 0.0 else 1.0
        map_precision = intersection / predicted_total if predicted_total > 0.0 else 0.0
        map_recall = intersection / actual_total if actual_total > 0.0 else 1.0
    return PerceptionMetrics(
        target_precision=float(precision),
        target_recall=float(recall),
        target_center_mean_error=center_error,
        free_space_mae=float(np.mean(free_space_error)),
        free_space_mae_left=float(free_space_mae_by_region[0]),
        free_space_mae_center=float(free_space_mae_by_region[1]),
        free_space_mae_right=float(free_space_mae_by_region[2]),
        risk_precision=float(risk_precision),
        risk_recall=float(risk_recall),
        risk_specificity=float(risk_specificity),
        risk_balanced_accuracy=float((risk_recall + risk_specificity) / 2.0),
        obstacle_map_iou=float(map_iou),
        obstacle_map_precision=float(map_precision),
        obstacle_map_recall=float(map_recall),
        sample_count=int(actual_visible.size),
    )


def color_baseline_predict(
    images: Tensor, minimum_target_pixels: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Yeşil hedef için açıklanabilir renk eşiği; CNN karşılaştırma baseline'ı."""

    pixels = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    mask = (green > 0.25) & (green > 1.25 * red) & (green > 1.25 * blue)
    scores = np.zeros(images.shape[0], dtype=np.float32)
    centers = np.zeros((images.shape[0], 2), dtype=np.float32)
    for index, sample_mask in enumerate(mask):
        rows, columns = np.nonzero(sample_mask)
        if columns.size >= minimum_target_pixels:
            scores[index] = 1.0
            centers[index] = (
                2.0 * float(np.mean(columns)) / max(sample_mask.shape[1] - 1, 1) - 1.0,
                2.0 * float(np.mean(rows)) / max(sample_mask.shape[0] - 1, 1) - 1.0,
            )
    free_space = np.ones((images.shape[0], 3), dtype=np.float32)
    return scores, centers, free_space


def _evaluate_model(
    model: TinyVisionEncoder,
    loader: DataLoader[VisionSample],
    image_transform: Callable[[Tensor], Tensor] | None = None,
) -> PerceptionMetrics:
    model.eval()
    score_parts: list[np.ndarray] = []
    center_parts: list[np.ndarray] = []
    free_parts: list[np.ndarray] = []
    visible_parts: list[np.ndarray] = []
    target_center_parts: list[np.ndarray] = []
    target_free_parts: list[np.ndarray] = []
    risk_score_parts: list[np.ndarray] = []
    map_parts: list[np.ndarray] = []
    target_map_parts: list[np.ndarray] = []
    with torch.no_grad():
        for images, visible, centers, free_space, _, obstacle_mask in loader:
            if image_transform is not None:
                images = image_transform(images)
            output = model(images)
            map_parts.append(
                torch.sigmoid(output.obstacle_heatmap_logits).cpu().numpy()
            )
            # Eğitimdeki havuzlamanın aynısı (max-pool), yoksa etiket ile
            # tahmin farklı çözünürlükte karşılaştırılırdı.
            target_map_parts.append(
                functional.adaptive_max_pool2d(
                    obstacle_mask.unsqueeze(1), (HEATMAP_HEIGHT, HEATMAP_WIDTH)
                )
                .squeeze(1)
                .cpu()
                .numpy()
            )
            score_parts.append(torch.sigmoid(output.target_logit).cpu().numpy())
            center_parts.append(output.target_center_xy.cpu().numpy())
            free_parts.append(output.free_space.cpu().numpy())
            visible_parts.append(visible.cpu().numpy())
            target_center_parts.append(centers.cpu().numpy())
            target_free_parts.append(free_space.cpu().numpy())
            risk_score_parts.append(
                torch.sigmoid(output.collision_risk_logits).cpu().numpy()
            )
    return compute_metrics(
        np.concatenate(score_parts),
        np.concatenate(center_parts),
        np.concatenate(free_parts),
        np.concatenate(visible_parts),
        np.concatenate(target_center_parts),
        np.concatenate(target_free_parts),
        np.concatenate(risk_score_parts),
        predicted_risk_threshold=model.config.collision_risk_score_threshold,
        predicted_obstacle_map=np.concatenate(map_parts),
        target_obstacle_map=np.concatenate(target_map_parts),
    )


def _evaluate_color_baseline(
    loader: DataLoader[VisionSample],
) -> PerceptionMetrics:
    score_parts: list[np.ndarray] = []
    center_parts: list[np.ndarray] = []
    free_parts: list[np.ndarray] = []
    visible_parts: list[np.ndarray] = []
    target_center_parts: list[np.ndarray] = []
    target_free_parts: list[np.ndarray] = []
    risk_score_parts: list[np.ndarray] = []
    for images, visible, centers, free_space, _, _ in loader:
        scores, predicted_centers, predicted_free = color_baseline_predict(images)
        score_parts.append(scores)
        center_parts.append(predicted_centers)
        free_parts.append(predicted_free)
        visible_parts.append(visible.numpy())
        target_center_parts.append(centers.numpy())
        target_free_parts.append(free_space.numpy())
        risk_score_parts.append(np.zeros_like(predicted_free))
    return compute_metrics(
        np.concatenate(score_parts),
        np.concatenate(center_parts),
        np.concatenate(free_parts),
        np.concatenate(visible_parts),
        np.concatenate(target_center_parts),
        np.concatenate(target_free_parts),
        np.concatenate(risk_score_parts),
    )


def evaluate_vision_encoder(
    model: TinyVisionEncoder,
    dataset_dir: str | Path,
    split: Split,
    *,
    batch_size: int = 32,
) -> tuple[PerceptionMetrics, PerceptionMetrics]:
    """Dondurulmuş encoder ve renk baseline'ını seçilen split'te değerlendir."""

    if batch_size <= 0:
        raise ValueError("Evaluation batch_size pozitif olmalı")
    dataset = VisionNpzDataset(dataset_dir, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return _evaluate_model(model, loader), _evaluate_color_baseline(loader)


def evaluate_vision_encoder_ablation(
    model: TinyVisionEncoder,
    dataset_dir: str | Path,
    split: Split,
    *,
    batch_size: int = 32,
) -> PerceptionMetrics:
    """Encoder'ı hue bilgisi kaldırılmış RGB üzerinde değerlendir."""

    if batch_size <= 0:
        raise ValueError("Evaluation batch_size pozitif olmalı")
    dataset = VisionNpzDataset(dataset_dir, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return _evaluate_model(model, loader, grayscale_actor_batch)


def benchmark_latency_ms(
    model: TinyVisionEncoder,
    sample: Tensor,
    *,
    warmup_runs: int,
    measurement_runs: int,
) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(sample)
        started = time.perf_counter()
        for _ in range(measurement_runs):
            model(sample)
    return (time.perf_counter() - started) * 1000.0 / measurement_runs


def train_vision_encoder(
    dataset_dir: str | Path,
    output_dir: str | Path,
    model_config: VisionEncoderConfig,
    training_config: VisionTrainingConfig,
) -> VisionTrainingArtifacts:
    """Train/validation split'iyle encoder eğit ve sürümlü checkpoint yaz."""

    model_config.validate()
    training_config.validate()
    validate_dataset_leakage(dataset_dir)
    dataset_provenance = vision_dataset_provenance(dataset_dir)
    if dataset_provenance.schema_version != "vision-dataset-v4":
        raise ValueError("Encoder training dataset'i vision-dataset-v4 olmalı")
    torch.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)
    train_dataset = VisionNpzDataset(dataset_dir, "train")
    validation_dataset = VisionNpzDataset(dataset_dir, "validation")
    development_ood_dataset = VisionNpzDataset(dataset_dir, "development_ood")
    generator = torch.Generator().manual_seed(training_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    development_ood_loader = DataLoader(
        development_ood_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = TinyVisionEncoder(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    epoch_losses: list[float] = []
    for _ in range(training_config.epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for (
            images,
            visible,
            centers,
            free_space,
            target_mask,
            obstacle_mask,
        ) in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if training_config.grayscale_augmentation_probability > 0.0:
                grayscale = grayscale_actor_batch(images)
                ablation_mask = (
                    torch.rand(images.shape[0])
                    < training_config.grayscale_augmentation_probability
                ).reshape(-1, 1, 1, 1)
                images = torch.where(ablation_mask, grayscale, images)
            output = model(images)
            loss = perception_loss(
                output,
                visible,
                centers,
                free_space,
                target_mask,
                obstacle_mask,
                training_config,
            )
            loss.backward()
            optimizer.step()
            batch_size = int(images.shape[0])
            running_loss += float(loss.detach()) * batch_size
            sample_count += batch_size
        epoch_losses.append(running_loss / sample_count)
    validation_metrics = _evaluate_model(model, validation_loader)
    baseline_metrics = _evaluate_color_baseline(validation_loader)
    development_ood_metrics = _evaluate_model(model, development_ood_loader)
    baseline_development_ood_metrics = _evaluate_color_baseline(development_ood_loader)
    grayscale_validation_metrics = _evaluate_model(
        model, validation_loader, grayscale_actor_batch
    )
    grayscale_development_ood_metrics = _evaluate_model(
        model, development_ood_loader, grayscale_actor_batch
    )
    sample_image = validation_dataset[0][0].unsqueeze(0)
    latency_ms = benchmark_latency_ms(
        model,
        sample_image,
        warmup_runs=training_config.latency_warmup_runs,
        measurement_runs=training_config.latency_measurement_runs,
    )
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    # Dosya adı şemadan türetilir; v10 checkpoint'i v9 adıyla kaydedilmemelidir.
    checkpoint_path = (
        output_path / f"{model_config.schema_version.replace('-', '_')}.pt"
    )
    metrics_path = output_path / "metrics.json"
    if checkpoint_path.exists() or metrics_path.exists():
        raise FileExistsError(f"Encoder çıktısının üzerine yazılmadı: {output_path}")
    metrics = {
        "schema_version": model_config.schema_version,
        "epoch_losses": epoch_losses,
        "validation": asdict(validation_metrics),
        "color_baseline_validation": asdict(baseline_metrics),
        "grayscale_validation": asdict(grayscale_validation_metrics),
        "development_ood": asdict(development_ood_metrics),
        "color_baseline_development_ood": asdict(baseline_development_ood_metrics),
        "grayscale_development_ood": asdict(grayscale_development_ood_metrics),
        "cpu_latency_ms": latency_ms,
    }
    torch.save(
        {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "dataset_provenance": asdict(dataset_provenance),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return VisionTrainingArtifacts(checkpoint=checkpoint_path, metrics=metrics_path)


def load_vision_encoder_checkpoint(path: str | Path) -> TinyVisionEncoder:
    checkpoint = torch.load(
        Path(path).expanduser().resolve(), map_location="cpu", weights_only=True
    )
    if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Desteklenmeyen vision encoder checkpoint sürümü")
    config = VisionEncoderConfig(**checkpoint["model_config"])
    model = TinyVisionEncoder(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
