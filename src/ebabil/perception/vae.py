"""Etiketsiz görsel temsil: yalnız actor RGB'siyle eğitilen VAE (ADR-022).

Bu modül privileged arşivleri hiçbir koşulda açmaz; `DatasetRecord.privileged_file`
yolu burada yalnız "actor dosyasıyla aynı olmasın" denetiminde string olarak
karşılaştırılır. Etiket gerektiren bütün ölçümler `ebabil.perception.probes`
içindedir ve yalnız kalite kapısına hizmet eder.
"""

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
import numpy.typing as npt
import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ebabil.perception.dataset import (
    ACTOR_KEYS,
    Split,
    load_manifest,
    vision_dataset_provenance,
)
from ebabil.perception.model import grayscale_actor_batch, preprocess_actor_rgb

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VISUAL_VAE_CONFIG = PACKAGE_ROOT / "configs" / "training" / "visual_vae.toml"
VAE_CHECKPOINT_FORMAT_VERSION = "ebabil-vae-checkpoint-v1"
VAE_SCHEMA_VERSION = "visual-vae-v1"
#: Sayısal kararlılık için log-varyans kırpma sınırı.
LOG_VARIANCE_CLAMP = 10.0


@dataclass(frozen=True)
class VisualVAEConfig:
    """Kodlayıcı/kod çözücü mimarisi; hiçbir alanı etiket gerektirmez."""

    schema_version: str
    input_width_px: int
    input_height_px: int
    latent_size: int
    channel_widths: tuple[int, ...]
    adaptive_exposure_target_mean: float
    adaptive_exposure_min_gain: float
    adaptive_exposure_max_gain: float
    #: REV-049: kontrastif yardımcı kayıp için projeksiyon başlığı boyutu.
    #: `0` başlık yok demektir (temel mimari); pozitif değer yalnız eğitimde
    #: kullanılan `latent -> hidden -> proj` başlığı kurar. Runtime kodlayıcı
    #: değişmez.
    contrastive_projection_size: int = 0

    def validate(self) -> None:
        if self.schema_version != VAE_SCHEMA_VERSION:
            raise ValueError(f"model.schema_version {VAE_SCHEMA_VERSION} olmalı")
        if min(self.input_width_px, self.input_height_px) < 16:
            raise ValueError("VAE giriş boyutları en az 16 piksel olmalı")
        if isinstance(self.latent_size, bool) or self.latent_size <= 0:
            raise ValueError("model.latent_size pozitif olmalı")
        if len(self.channel_widths) < 2 or any(
            isinstance(width, bool) or width <= 0 for width in self.channel_widths
        ):
            raise ValueError("model.channel_widths en az iki pozitif genişlik içermeli")
        if min(self.input_width_px, self.input_height_px) < 2 ** len(
            self.channel_widths
        ):
            raise ValueError(
                "model.channel_widths katman sayısı giriş boyutunu sıfıra indirir"
            )
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
        if (
            isinstance(self.contrastive_projection_size, bool)
            or self.contrastive_projection_size < 0
        ):
            raise ValueError("model.contrastive_projection_size negatif olamaz")


@dataclass(frozen=True)
class VisualVAETrainingConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    #: `beta`: piksel toplamı üzerinden hesaplanan yeniden üretim hatasına göre
    #: KL ağırlığı. `1.0` standart VAE'dir.
    kl_weight: float
    #: KL ağırlığının sıfırdan `kl_weight`'e doğrusal ısındığı epoch sayısı;
    #: `0` ısınma yok demektir. Posterior collapse'a karşı ilk savunma.
    kl_warmup_epochs: int
    #: Boyut başına bu değerin altındaki KL cezalandırılmaz (free bits).
    #: `0.0` kapalıdır.
    free_bits_nats: float
    grayscale_augmentation_probability: float
    #: Girişe rastgele kanal kazancı, gamma, parlaklık ofseti ve gürültü
    #: uygulanır; kod çözücü ise TEMİZ görüntüyü yeniden üretmek zorundadır.
    #: Latent'i küresel ton/parlaklıktan bağımsızlaştırır (etiketsiz). `0.0`
    #: kapalıdır; REV-049 taramasında ölçülür.
    photometric_augmentation_probability: float
    #: REV-049: SimCLR/CURL tarzı kontrastif yardımcı kayıp ağırlığı. Aynı
    #: karenin iki rastgele kırpılmış + fotometrik bozulmuş görünümü
    #: birbirine, batch'teki diğer karelere uzak olmalı (InfoNCE). Etiketsiz.
    #: `0.0` kapalıdır; pozitif değer `contrastive_projection_size > 0` ister.
    contrastive_weight: float
    contrastive_temperature: float
    #: Rastgele kırpmada korunan alan oranının alt sınırı.
    contrastive_minimum_crop_fraction: float
    #: Ortalama KL'si bu eşiği aşan latent boyutu "etkin" sayılır.
    active_unit_kl_threshold_nats: float
    latency_warmup_runs: int
    latency_measurement_runs: int

    def validate(self) -> None:
        if self.seed < 0 or min(self.epochs, self.batch_size) <= 0:
            raise ValueError("Training seed negatif olamaz; epoch/batch pozitif olmalı")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("training.learning_rate pozitif ve sonlu olmalı")
        non_negative = (
            self.kl_weight,
            self.free_bits_nats,
            self.active_unit_kl_threshold_nats,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in non_negative):
            raise ValueError(
                "KL ağırlığı, free bits ve etkin boyut eşiği negatif olamaz"
            )
        if self.kl_warmup_epochs < 0:
            raise ValueError("training.kl_warmup_epochs negatif olamaz")
        if not 0.0 <= self.grayscale_augmentation_probability <= 1.0:
            raise ValueError("grayscale augmentation olasılığı [0, 1] içinde olmalı")
        if not 0.0 <= self.photometric_augmentation_probability <= 1.0:
            raise ValueError("photometric augmentation olasılığı [0, 1] içinde olmalı")
        if not math.isfinite(self.contrastive_weight) or self.contrastive_weight < 0.0:
            raise ValueError("training.contrastive_weight negatif olamaz")
        if (
            not math.isfinite(self.contrastive_temperature)
            or self.contrastive_temperature <= 0.0
        ):
            raise ValueError("training.contrastive_temperature pozitif olmalı")
        if not 0.0 < self.contrastive_minimum_crop_fraction <= 1.0:
            raise ValueError("contrastive_minimum_crop_fraction (0, 1] içinde olmalı")
        if self.latency_warmup_runs < 0 or self.latency_measurement_runs <= 0:
            raise ValueError("Latency tekrar sayıları geçersiz")


@dataclass(frozen=True)
class VAEMetrics:
    """Etiket gerektirmeyen temsil metrikleri."""

    reconstruction_mse: float
    psnr_db: float
    kl_nats: float
    active_latent_units: int
    active_latent_units_variance_criterion: int
    latent_size: int
    sample_count: int


@dataclass(frozen=True)
class VisualVAETrainingArtifacts:
    checkpoint: Path
    metrics: Path


class VisualVAEOutput(NamedTuple):
    mean: Tensor
    log_variance: Tensor
    latent: Tensor
    reconstruction: Tensor
    target: Tensor


class VAELossBreakdown(NamedTuple):
    total: Tensor
    reconstruction: Tensor
    kl: Tensor


class ActorOnlyNpzDataset(Dataset[Tensor]):
    """Yalnız actor arşivini okur; `privileged_file` hiçbir zaman açılmaz.

    `cache=True` decompress edilmiş `uint8` kareleri bellekte tutar; 6528
    karelik D1 dataset'i için `~1.5 GB`'dır ve epoch başına npz açma
    maliyetini kaldırır.
    """

    def __init__(
        self, dataset_dir: str | Path, split: Split, *, cache: bool = True
    ) -> None:
        self.root = Path(dataset_dir).expanduser().resolve()
        self.records = tuple(
            record
            for record in load_manifest(self.root / "manifest.jsonl")
            if record.split == split
        )
        if not self.records:
            raise ValueError(f"Dataset split'i boş: {split}")
        self._cache: dict[int, npt.NDArray[np.uint8]] | None = {} if cache else None

    def __len__(self) -> int:
        return len(self.records)

    def _actor_path(self, index: int) -> Path:
        record = self.records[index]
        path = (self.root / record.actor_file).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Dataset path kök dışına çıkıyor: {record.actor_file}")
        if "privileged" in Path(record.actor_file).parts:
            raise ValueError(f"Actor yolu privileged dizinini gösteriyor: {path}")
        return path

    def rgb(self, index: int) -> npt.NDArray[np.uint8]:
        if self._cache is not None and index in self._cache:
            return self._cache[index]
        with np.load(self._actor_path(index), allow_pickle=False) as payload:
            rgb = np.asarray(payload["rgb"])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Actor RGB uint8 HWC olmalı")
        if self._cache is not None:
            self._cache[index] = rgb
        return rgb

    def __getitem__(self, index: int) -> Tensor:
        return preprocess_actor_rgb(self.rgb(index))


def validate_actor_archives(dataset_dir: str | Path) -> None:
    """Manifest ve actor arşivlerini privileged dosyalara dokunmadan denetle."""

    root = Path(dataset_dir).expanduser().resolve()
    records = load_manifest(root / "manifest.jsonl")
    if not records:
        raise ValueError("Dataset manifest boş")
    seen: set[str] = set()
    scene_splits: dict[int, Split] = {}
    for record in records:
        if record.sample_id in seen:
            raise ValueError(f"Tekrarlanan sample_id: {record.sample_id}")
        seen.add(record.sample_id)
        previous = scene_splits.setdefault(record.scene_seed, record.split)
        if previous != record.split:
            raise ValueError(f"Scene seed split sızıntısı: {record.scene_seed}")
        actor_path = (root / record.actor_file).resolve()
        privileged_path = (root / record.privileged_file).resolve()
        if not actor_path.is_relative_to(root):
            raise ValueError(f"Dataset path kök dışına çıkıyor: {record.sample_id}")
        if actor_path == privileged_path:
            raise ValueError(f"Actor/privileged dosyaları aynı: {record.sample_id}")
        with np.load(actor_path, allow_pickle=False) as payload:
            if set(payload.files) != ACTOR_KEYS:
                raise ValueError(
                    f"Actor payload allowlist ihlali: {actor_path}: "
                    f"{sorted(payload.files)}"
                )
            if str(payload["sample_id"]) != record.sample_id:
                raise ValueError(f"Actor sample_id eşleşmiyor: {record.sample_id}")
            rgb = payload["rgb"]
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"Actor RGB şeması geçersiz: {record.sample_id}")


def _halved(size: int) -> int:
    return (size + 2 - 4) // 2 + 1


class VisualVAE(nn.Module):
    """Küçük konvolüsyonel VAE; kodlayıcı runtime'a, kod çözücü yalnız eğitime."""

    def __init__(self, config: VisualVAEConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        widths = tuple(config.channel_widths)
        sizes = [(config.input_height_px, config.input_width_px)]
        for _ in widths:
            height, width = sizes[-1]
            sizes.append((_halved(height), _halved(width)))
        encoder_layers: list[nn.Module] = []
        in_channels = 3
        for width in widths:
            encoder_layers.append(
                nn.Conv2d(in_channels, width, kernel_size=4, stride=2, padding=1)
            )
            encoder_layers.append(nn.ReLU())
            in_channels = width
        self.encoder = nn.Sequential(*encoder_layers)
        self._feature_shape = (widths[-1], sizes[-1][0], sizes[-1][1])
        flat = int(np.prod(self._feature_shape))
        self.mean_head = nn.Linear(flat, config.latent_size)
        self.log_variance_head = nn.Linear(flat, config.latent_size)
        self.decoder_input = nn.Linear(config.latent_size, flat)
        decoder_layers: list[nn.Module] = []
        out_channels_sequence = tuple(widths[-2::-1]) + (3,)
        in_channels = widths[-1]
        for index, out_channels in enumerate(out_channels_sequence):
            source = sizes[-1 - index]
            target = sizes[-2 - index]
            output_padding = (target[0] - 2 * source[0], target[1] - 2 * source[1])
            decoder_layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=output_padding,
                )
            )
            if index < len(out_channels_sequence) - 1:
                decoder_layers.append(nn.ReLU())
            in_channels = out_channels
        self.decoder = nn.Sequential(*decoder_layers)
        self.projection_head: nn.Module | None = (
            nn.Sequential(
                nn.Linear(config.latent_size, config.contrastive_projection_size),
                nn.ReLU(),
                nn.Linear(
                    config.contrastive_projection_size,
                    config.contrastive_projection_size,
                ),
            )
            if config.contrastive_projection_size > 0
            else None
        )

    def project(self, images: Tensor) -> Tensor:
        """Kontrastif kayıp için L2-normalize projeksiyon; yalnız eğitimde."""

        if self.projection_head is None:
            raise ValueError("Bu modelde kontrastif projeksiyon başlığı yok")
        mean, _ = self.encode(images)
        return functional.normalize(self.projection_head(mean), dim=1)

    def normalize(self, images: Tensor) -> Tensor:
        """Giriş doğrulaması, yeniden boyutlama ve kanal bazlı adaptive exposure."""

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("VAE girdisi BCHW ve üç kanallı olmalı")
        if not torch.is_floating_point(images):
            raise ValueError("VAE girdisi floating point olmalı")
        if not bool(torch.isfinite(images).all()):
            raise ValueError("VAE girdisi sonlu olmalı")
        if images.numel() == 0 or bool(torch.any((images < 0.0) | (images > 1.0))):
            raise ValueError("VAE girdisi [0, 1] aralığında olmalı")
        size = (self.config.input_height_px, self.config.input_width_px)
        resized = (
            images
            if tuple(images.shape[-2:]) == size
            else functional.interpolate(
                images, size=size, mode="bilinear", align_corners=False
            )
        )
        channel_mean = resized.mean(dim=(2, 3), keepdim=True)
        exposure_gain = torch.clamp(
            self.config.adaptive_exposure_target_mean / channel_mean.clamp_min(1e-4),
            min=self.config.adaptive_exposure_min_gain,
            max=self.config.adaptive_exposure_max_gain,
        )
        return torch.clamp(resized * exposure_gain, 0.0, 1.0)

    def _posterior(self, normalized: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encoder(normalized).flatten(1)
        mean = self.mean_head(features)
        log_variance = torch.clamp(
            self.log_variance_head(features), -LOG_VARIANCE_CLAMP, LOG_VARIANCE_CLAMP
        )
        return mean, log_variance

    def encode(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Posterior ortalama ve log-varyansı; runtime yalnız ortalamayı kullanır."""

        return self._posterior(self.normalize(images))

    def decode(self, latent: Tensor) -> Tensor:
        hidden = functional.relu(self.decoder_input(latent))
        return torch.sigmoid(self.decoder(hidden.reshape(-1, *self._feature_shape)))

    def forward(
        self, images: Tensor, target_images: Tensor | None = None
    ) -> VisualVAEOutput:
        """Kodla ve yeniden üret.

        `target_images` verilirse kod çözücünün hedefi ondan türetilir; eğitimde
        bozulmuş girişten temiz görüntüyü üretme (fotometrik değişmezlik) bu
        yolla kurulur. Verilmezse giriş kendi hedefidir.
        """

        normalized_input = self.normalize(images)
        target = (
            normalized_input if target_images is None else self.normalize(target_images)
        )
        if target.shape != normalized_input.shape:
            raise ValueError("target_images girişle aynı batch ve boyutta olmalı")
        mean, log_variance = self._posterior(normalized_input)
        if self.training:
            latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        else:
            latent = mean
        return VisualVAEOutput(
            mean=mean,
            log_variance=log_variance,
            latent=latent,
            reconstruction=self.decode(latent),
            target=target,
        )


def photometric_augment_batch(images: Tensor, generator: torch.Generator) -> Tensor:
    """Her kareye seed'li rastgele kanal kazancı, gamma, ofset ve gürültü uygula.

    Aralıklar dataset görsel alanlarının (`visual_randomization`) zarfına
    yakındır: kazanç `[0.7, 1.3]`, gamma `[0.8, 1.25]`, ofset `[-0.1, 0.1]`,
    gürültü `[0, 0.03]`. Çıktı `[0, 1]` içinde kırpılır.
    """

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Photometric augmentation girdisi BCHW RGB olmalı")
    batch = images.shape[0]
    gain = 0.7 + 0.6 * torch.rand((batch, 3, 1, 1), generator=generator)
    gamma = 0.8 + 0.45 * torch.rand((batch, 1, 1, 1), generator=generator)
    offset = -0.1 + 0.2 * torch.rand((batch, 1, 1, 1), generator=generator)
    noise_std = 0.03 * torch.rand((batch, 1, 1, 1), generator=generator)
    noise = torch.randn(images.shape, generator=generator) * noise_std
    augmented = torch.clamp(images * gain, 0.0, 1.0).pow(gamma) + offset + noise
    return torch.clamp(augmented, 0.0, 1.0)


def random_resized_crop_batch(
    images: Tensor, generator: torch.Generator, *, minimum_fraction: float
) -> Tensor:
    """Her kareden alan oranı `[minimum_fraction, 1]` olan pencere kırp, geri boyutla."""

    if images.ndim != 4:
        raise ValueError("Crop girdisi BCHW olmalı")
    batch, _, height, width = images.shape
    crops: list[Tensor] = []
    for index in range(batch):
        fraction = float(
            minimum_fraction
            + (1.0 - minimum_fraction) * torch.rand(1, generator=generator)
        )
        crop_height = max(int(round(height * math.sqrt(fraction))), 8)
        crop_width = max(int(round(width * math.sqrt(fraction))), 8)
        top = int(torch.randint(0, height - crop_height + 1, (1,), generator=generator))
        left = int(torch.randint(0, width - crop_width + 1, (1,), generator=generator))
        window = images[
            index : index + 1, :, top : top + crop_height, left : left + crop_width
        ]
        crops.append(
            functional.interpolate(
                window, size=(height, width), mode="bilinear", align_corners=False
            )
        )
    return torch.cat(crops, dim=0)


def info_nce_loss(first: Tensor, second: Tensor, *, temperature: float) -> Tensor:
    """Simetrik NT-Xent: `first[i]` ile `second[i]` pozitif çift, batch geri kalanı negatif."""

    if first.shape != second.shape or first.ndim != 2 or first.shape[0] < 2:
        raise ValueError("InfoNCE en az iki örnekli eşit boyutlu iki görünüm ister")
    logits = first @ second.T / temperature
    labels = torch.arange(first.shape[0], device=first.device)
    return 0.5 * (
        functional.cross_entropy(logits, labels)
        + functional.cross_entropy(logits.T, labels)
    )


def kl_per_dimension(mean: Tensor, log_variance: Tensor) -> Tensor:
    """Standart normal prior'a göre boyut bazlı KL, shape `(B, D)`."""

    return 0.5 * (mean.square() + log_variance.exp() - 1.0 - log_variance)


def vae_loss(
    output: VisualVAEOutput, *, kl_weight: float, free_bits_nats: float
) -> VAELossBreakdown:
    """Piksel toplamı yeniden üretim hatası + `beta * KL`; batch ortalaması."""

    reconstruction = (
        (output.reconstruction - output.target).square().sum(dim=(1, 2, 3)).mean()
    )
    kl_dims = kl_per_dimension(output.mean, output.log_variance)
    kl = kl_dims.sum(dim=1).mean()
    if free_bits_nats > 0.0:
        penalized = torch.clamp(kl_dims.mean(dim=0), min=free_bits_nats).sum()
    else:
        penalized = kl
    return VAELossBreakdown(
        total=reconstruction + kl_weight * penalized,
        reconstruction=reconstruction,
        kl=kl,
    )


def load_visual_vae_config(
    path: str | Path = DEFAULT_VISUAL_VAE_CONFIG,
) -> tuple[VisualVAEConfig, VisualVAETrainingConfig]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Visual VAE config bulunamadı: {config_path}")
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    model_raw = raw["model"]
    training_raw = raw["training"]
    model = VisualVAEConfig(
        schema_version=str(model_raw["schema_version"]),
        input_width_px=int(model_raw["input_width_px"]),
        input_height_px=int(model_raw["input_height_px"]),
        latent_size=int(model_raw["latent_size"]),
        channel_widths=tuple(int(width) for width in model_raw["channel_widths"]),
        adaptive_exposure_target_mean=float(model_raw["adaptive_exposure_target_mean"]),
        adaptive_exposure_min_gain=float(model_raw["adaptive_exposure_min_gain"]),
        adaptive_exposure_max_gain=float(model_raw["adaptive_exposure_max_gain"]),
        contrastive_projection_size=int(
            model_raw.get("contrastive_projection_size", 0)
        ),
    )
    training = VisualVAETrainingConfig(
        seed=int(training_raw["seed"]),
        epochs=int(training_raw["epochs"]),
        batch_size=int(training_raw["batch_size"]),
        learning_rate=float(training_raw["learning_rate"]),
        kl_weight=float(training_raw["kl_weight"]),
        kl_warmup_epochs=int(training_raw["kl_warmup_epochs"]),
        free_bits_nats=float(training_raw["free_bits_nats"]),
        grayscale_augmentation_probability=float(
            training_raw["grayscale_augmentation_probability"]
        ),
        photometric_augmentation_probability=float(
            training_raw.get("photometric_augmentation_probability", 0.0)
        ),
        contrastive_weight=float(training_raw.get("contrastive_weight", 0.0)),
        contrastive_temperature=float(training_raw.get("contrastive_temperature", 0.1)),
        contrastive_minimum_crop_fraction=float(
            training_raw.get("contrastive_minimum_crop_fraction", 0.6)
        ),
        active_unit_kl_threshold_nats=float(
            training_raw["active_unit_kl_threshold_nats"]
        ),
        latency_warmup_runs=int(training_raw["latency_warmup_runs"]),
        latency_measurement_runs=int(training_raw["latency_measurement_runs"]),
    )
    model.validate()
    training.validate()
    return model, training


def _make_loader(
    dataset: ActorOnlyNpzDataset,
    batch_size: int,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader[Tensor]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def evaluate_visual_vae(
    model: VisualVAE,
    dataset_dir: str | Path,
    split: Split,
    *,
    batch_size: int,
    active_unit_kl_threshold_nats: float,
    image_transform: Callable[[Tensor], Tensor] | None = None,
) -> VAEMetrics:
    """Etiketsiz metrikler: yeniden üretim, KL ve etkin latent boyutu."""

    if batch_size <= 0:
        raise ValueError("Evaluation batch_size pozitif olmalı")
    model.eval()
    loader = _make_loader(
        ActorOnlyNpzDataset(dataset_dir, split, cache=False), batch_size, shuffle=False
    )
    squared_error_sum = 0.0
    pixel_count = 0
    kl_dim_sum: Tensor | None = None
    means: list[Tensor] = []
    sample_count = 0
    with torch.no_grad():
        for images in loader:
            batch = image_transform(images) if image_transform is not None else images
            output = model(batch)
            squared_error_sum += float(
                (output.reconstruction - output.target).square().sum()
            )
            pixel_count += int(output.target.numel())
            kl_dims = kl_per_dimension(output.mean, output.log_variance)
            kl_dim_sum = (
                kl_dims.sum(dim=0)
                if kl_dim_sum is None
                else kl_dim_sum + kl_dims.sum(0)
            )
            means.append(output.mean)
            sample_count += int(images.shape[0])
    assert kl_dim_sum is not None
    mean_kl_per_dim = kl_dim_sum / sample_count
    mse = squared_error_sum / max(pixel_count, 1)
    psnr = 10.0 * math.log10(1.0 / mse) if mse > 0.0 else float("inf")
    all_means = torch.cat(means, dim=0)
    variance_criterion = int((all_means.var(dim=0, unbiased=False) > 0.01).sum())
    return VAEMetrics(
        reconstruction_mse=float(mse),
        psnr_db=float(psnr),
        kl_nats=float(mean_kl_per_dim.sum()),
        active_latent_units=int(
            (mean_kl_per_dim > active_unit_kl_threshold_nats).sum()
        ),
        active_latent_units_variance_criterion=variance_criterion,
        latent_size=int(model.config.latent_size),
        sample_count=sample_count,
    )


def benchmark_encoder_latency_ms(
    model: VisualVAE, sample: Tensor, *, warmup_runs: int, measurement_runs: int
) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            model.encode(sample)
        started = time.perf_counter()
        for _ in range(measurement_runs):
            model.encode(sample)
    return (time.perf_counter() - started) * 1000.0 / measurement_runs


def train_visual_vae(
    dataset_dir: str | Path,
    output_dir: str | Path,
    model_config: VisualVAEConfig,
    training_config: VisualVAETrainingConfig,
    *,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> VisualVAETrainingArtifacts:
    """Yalnız actor RGB ile VAE eğit; etiketsiz metrik ve sürümlü checkpoint yaz.

    Bu fonksiyon privileged arşivleri açmaz; `validate_actor_archives` ve
    `ActorOnlyNpzDataset` yalnız actor dosyalarını okur. Dataset provenance'ı
    (metadata/manifest/statistics hash'leri) approval zinciri için kaydedilir.
    """

    model_config.validate()
    training_config.validate()
    if (
        training_config.contrastive_weight > 0.0
        and model_config.contrastive_projection_size <= 0
    ):
        raise ValueError(
            "contrastive_weight > 0 için model.contrastive_projection_size > 0 gerekli"
        )
    validate_actor_archives(dataset_dir)
    dataset_provenance = vision_dataset_provenance(dataset_dir)
    if dataset_provenance.schema_version != "vision-dataset-v4":
        raise ValueError("VAE training dataset'i vision-dataset-v4 olmalı")
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / f"{VAE_SCHEMA_VERSION.replace('-', '_')}.pt"
    metrics_path = output_path / "metrics.json"
    if checkpoint_path.exists() or metrics_path.exists():
        raise FileExistsError(f"VAE çıktısının üzerine yazılmadı: {output_path}")
    torch.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)
    train_dataset = ActorOnlyNpzDataset(dataset_dir, "train")
    generator = torch.Generator().manual_seed(training_config.seed)
    augmentation_generator = torch.Generator().manual_seed(training_config.seed + 1)
    train_loader = _make_loader(
        train_dataset, training_config.batch_size, shuffle=True, generator=generator
    )
    model = VisualVAE(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    epoch_losses: list[dict[str, float]] = []
    for epoch in range(training_config.epochs):
        model.train()
        if training_config.kl_warmup_epochs > 0:
            warmup = min(1.0, (epoch + 1) / training_config.kl_warmup_epochs)
        else:
            warmup = 1.0
        kl_weight = training_config.kl_weight * warmup
        running = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0, "contrastive": 0.0}
        sample_count = 0
        for images in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if training_config.grayscale_augmentation_probability > 0.0:
                grayscale_images = grayscale_actor_batch(images)
                mask = (
                    torch.rand(images.shape[0])
                    < training_config.grayscale_augmentation_probability
                ).reshape(-1, 1, 1, 1)
                images = torch.where(mask, grayscale_images, images)
            inputs = images
            if training_config.photometric_augmentation_probability > 0.0:
                augmented = photometric_augment_batch(images, augmentation_generator)
                mask = (
                    torch.rand(images.shape[0], generator=augmentation_generator)
                    < training_config.photometric_augmentation_probability
                ).reshape(-1, 1, 1, 1)
                inputs = torch.where(mask, augmented, images)
            output = model(inputs, target_images=images)
            loss = vae_loss(
                output,
                kl_weight=kl_weight,
                free_bits_nats=training_config.free_bits_nats,
            )
            total = loss.total
            contrastive_value = 0.0
            if training_config.contrastive_weight > 0.0 and images.shape[0] >= 2:
                views = []
                for _ in range(2):
                    cropped = random_resized_crop_batch(
                        images,
                        augmentation_generator,
                        minimum_fraction=training_config.contrastive_minimum_crop_fraction,
                    )
                    views.append(
                        photometric_augment_batch(cropped, augmentation_generator)
                    )
                contrastive = info_nce_loss(
                    model.project(views[0]),
                    model.project(views[1]),
                    temperature=training_config.contrastive_temperature,
                )
                total = total + training_config.contrastive_weight * contrastive
                contrastive_value = float(contrastive.detach())
            total.backward()
            optimizer.step()
            batch_size = int(images.shape[0])
            running["total"] += float(total.detach()) * batch_size
            running["reconstruction"] += (
                float(loss.reconstruction.detach()) * batch_size
            )
            running["kl"] += float(loss.kl.detach()) * batch_size
            running["contrastive"] += contrastive_value * batch_size
            sample_count += batch_size
        epoch_losses.append(
            {name: value / sample_count for name, value in running.items()}
            | {"kl_weight": kl_weight}
        )
        if progress_callback is not None:
            progress_callback(
                epoch + 1, training_config.epochs, epoch_losses[-1]["total"]
            )
    metrics: dict[str, object] = {
        "schema_version": model_config.schema_version,
        "label_free": True,
        "epoch_losses": epoch_losses,
    }
    for split in ("validation", "development_ood"):
        nominal = evaluate_visual_vae(
            model,
            dataset_dir,
            split,  # type: ignore[arg-type]
            batch_size=training_config.batch_size,
            active_unit_kl_threshold_nats=training_config.active_unit_kl_threshold_nats,
        )
        grayscale_metrics = evaluate_visual_vae(
            model,
            dataset_dir,
            split,  # type: ignore[arg-type]
            batch_size=training_config.batch_size,
            active_unit_kl_threshold_nats=training_config.active_unit_kl_threshold_nats,
            image_transform=grayscale_actor_batch,
        )
        metrics[split] = asdict(nominal)
        metrics[f"grayscale_{split}"] = asdict(grayscale_metrics)
    sample_image = train_dataset[0].unsqueeze(0)
    metrics["cpu_latency_ms"] = benchmark_encoder_latency_ms(
        model,
        sample_image,
        warmup_runs=training_config.latency_warmup_runs,
        measurement_runs=training_config.latency_measurement_runs,
    )
    metrics["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
    metrics["encoder_parameter_count"] = int(
        sum(p.numel() for p in model.encoder.parameters())
        + sum(p.numel() for p in model.mean_head.parameters())
    )
    torch.save(
        {
            "checkpoint_format_version": VAE_CHECKPOINT_FORMAT_VERSION,
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
    return VisualVAETrainingArtifacts(checkpoint=checkpoint_path, metrics=metrics_path)


def load_visual_vae_checkpoint(path: str | Path) -> VisualVAE:
    checkpoint = torch.load(
        Path(path).expanduser().resolve(), map_location="cpu", weights_only=True
    )
    if checkpoint.get("checkpoint_format_version") != VAE_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Desteklenmeyen visual VAE checkpoint sürümü")
    raw_config = dict(checkpoint["model_config"])
    raw_config["channel_widths"] = tuple(int(w) for w in raw_config["channel_widths"])
    raw_config.setdefault("contrastive_projection_size", 0)
    config = VisualVAEConfig(**raw_config)
    model = VisualVAE(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def encode_actor_rgb(
    model: VisualVAE, rgb: npt.NDArray[np.uint8]
) -> npt.NDArray[np.float64]:
    """Tek actor karesini posterior ortalama vektörüne çevir (runtime yolu)."""

    with torch.inference_mode():
        mean, _ = model.encode(preprocess_actor_rgb(rgb).unsqueeze(0))
    latent = mean[0].detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(latent)):
        raise ValueError("VAE latent'i yalnız sonlu değerler içermeli")
    return latent


__all__ = [
    "DEFAULT_VISUAL_VAE_CONFIG",
    "LOG_VARIANCE_CLAMP",
    "VAE_CHECKPOINT_FORMAT_VERSION",
    "VAE_SCHEMA_VERSION",
    "ActorOnlyNpzDataset",
    "VAELossBreakdown",
    "VAEMetrics",
    "VisualVAE",
    "VisualVAEConfig",
    "VisualVAEOutput",
    "VisualVAETrainingArtifacts",
    "VisualVAETrainingConfig",
    "benchmark_encoder_latency_ms",
    "encode_actor_rgb",
    "evaluate_visual_vae",
    "kl_per_dimension",
    "load_visual_vae_checkpoint",
    "load_visual_vae_config",
    "info_nce_loss",
    "photometric_augment_batch",
    "random_resized_crop_batch",
    "train_visual_vae",
    "vae_loss",
    "validate_actor_archives",
]
