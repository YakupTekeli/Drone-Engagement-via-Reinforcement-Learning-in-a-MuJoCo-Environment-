"""Seed'li MuJoCo görünüm ve kamera etkisi randomization araçları."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional

FloatArray = npt.NDArray[np.float64]
UInt8Array = npt.NDArray[np.uint8]


def _validate_range(values: tuple[float, float], name: str) -> None:
    if (
        len(values) != 2
        or any(not math.isfinite(value) for value in values)
        or values[0] > values[1]
    ):
        raise ValueError(f"{name} sonlu ve artan iki değer içermeli")


def _validate_rgb(values: tuple[float, float, float], name: str) -> None:
    if len(values) != 3 or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values
    ):
        raise ValueError(f"{name} üç adet [0, 1] RGB değeri içermeli")


def _validate_gain_rgb(values: tuple[float, float, float], name: str) -> None:
    if len(values) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in values
    ):
        raise ValueError(f"{name} üç adet pozitif ve sonlu gain içermeli")


@dataclass(frozen=True)
class VisualDomainConfig:
    name: str
    light_scale: tuple[float, float]
    texture_tint_min: tuple[float, float, float]
    texture_tint_max: tuple[float, float, float]
    obstacle_rgb_min: tuple[float, float, float]
    obstacle_rgb_max: tuple[float, float, float]
    target_rgb_min: tuple[float, float, float]
    target_rgb_max: tuple[float, float, float]
    exposure_gain: tuple[float, float]
    gamma: tuple[float, float]
    noise_std: tuple[float, float]
    vignette_strength: tuple[float, float]
    blur_probability: float

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Visual domain adı boş olamaz")
        _validate_range(self.light_scale, "light_scale")
        _validate_range(self.exposure_gain, "exposure_gain")
        _validate_range(self.gamma, "gamma")
        _validate_range(self.noise_std, "noise_std")
        _validate_range(self.vignette_strength, "vignette_strength")
        _validate_gain_rgb(self.texture_tint_min, "texture_tint_min")
        _validate_gain_rgb(self.texture_tint_max, "texture_tint_max")
        _validate_rgb(self.obstacle_rgb_min, "obstacle_rgb_min")
        _validate_rgb(self.obstacle_rgb_max, "obstacle_rgb_max")
        _validate_rgb(self.target_rgb_min, "target_rgb_min")
        _validate_rgb(self.target_rgb_max, "target_rgb_max")
        for minimum, maximum, name in (
            (self.texture_tint_min, self.texture_tint_max, "texture_tint"),
            (self.obstacle_rgb_min, self.obstacle_rgb_max, "obstacle_rgb"),
            (self.target_rgb_min, self.target_rgb_max, "target_rgb"),
        ):
            if any(low > high for low, high in zip(minimum, maximum, strict=True)):
                raise ValueError(f"{name} min değerleri max değerlerini aşamaz")
        if self.light_scale[0] <= 0.0 or self.exposure_gain[0] <= 0.0:
            raise ValueError("Işık ve exposure gain pozitif olmalı")
        if self.gamma[0] <= 0.0 or self.noise_std[0] < 0.0:
            raise ValueError("Gamma pozitif, noise negatif olmayan olmalı")
        if not 0.0 <= self.vignette_strength[0] <= self.vignette_strength[1] <= 1.0:
            raise ValueError("Vignette aralığı [0, 1] içinde olmalı")
        if not 0.0 <= self.blur_probability <= 1.0:
            raise ValueError("blur_probability [0, 1] aralığında olmalı")


@dataclass(frozen=True)
class SceneVisualParameters:
    domain_name: str
    light_scale: float
    texture_tint: tuple[float, float, float]
    obstacle_rgb: tuple[float, float, float]
    target_rgb: tuple[float, float, float]


@dataclass(frozen=True)
class FrameVisualParameters:
    exposure_gain: float
    gamma: float
    noise_std: float
    vignette_strength: float
    apply_blur: bool


def sample_scene_visual_parameters(
    config: VisualDomainConfig, seed: int
) -> SceneVisualParameters:
    config.validate()
    if seed < 0:
        raise ValueError("Visual seed negatif olamaz")
    rng = np.random.default_rng(seed)
    texture_tint = rng.uniform(config.texture_tint_min, config.texture_tint_max)
    obstacle_rgb = rng.uniform(config.obstacle_rgb_min, config.obstacle_rgb_max)
    target_rgb = rng.uniform(config.target_rgb_min, config.target_rgb_max)
    return SceneVisualParameters(
        domain_name=config.name,
        light_scale=float(rng.uniform(*config.light_scale)),
        texture_tint=tuple(float(value) for value in texture_tint),  # type: ignore[arg-type]
        obstacle_rgb=tuple(float(value) for value in obstacle_rgb),  # type: ignore[arg-type]
        target_rgb=tuple(float(value) for value in target_rgb),  # type: ignore[arg-type]
    )


def sample_frame_visual_parameters(
    config: VisualDomainConfig, seed: int
) -> FrameVisualParameters:
    config.validate()
    if seed < 0:
        raise ValueError("Visual seed negatif olamaz")
    rng = np.random.default_rng(seed)
    return FrameVisualParameters(
        exposure_gain=float(rng.uniform(*config.exposure_gain)),
        gamma=float(rng.uniform(*config.gamma)),
        noise_std=float(rng.uniform(*config.noise_std)),
        vignette_strength=float(rng.uniform(*config.vignette_strength)),
        apply_blur=bool(rng.random() < config.blur_probability),
    )


def apply_scene_visual_parameters(
    model: mujoco.MjModel, parameters: SceneVisualParameters
) -> None:
    """Renderer kurulmadan model ışık, texture ve material renklerini değiştir."""

    model.light_diffuse[:] *= parameters.light_scale
    model.light_specular[:] *= parameters.light_scale
    model.vis.headlight.ambient[:] *= parameters.light_scale
    model.vis.headlight.diffuse[:] *= parameters.light_scale
    for texture_id in range(model.ntex):
        address = int(model.tex_adr[texture_id])
        size = (
            int(model.tex_width[texture_id])
            * int(model.tex_height[texture_id])
            * int(model.tex_nchannel[texture_id])
        )
        channels = int(model.tex_nchannel[texture_id])
        texture = model.tex_data[address : address + size].reshape(-1, channels)
        tinted = texture.astype(np.float64)
        color_channels = min(channels, 3)
        tinted[:, :color_channels] *= np.asarray(parameters.texture_tint)[
            :color_channels
        ]
        texture[:] = np.clip(np.rint(tinted), 0.0, 255.0).astype(np.uint8)
    obstacle_rgb = np.asarray(parameters.obstacle_rgb)
    target_rgb = np.asarray(parameters.target_rgb)
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name
        if name.startswith("obstacle_"):
            model.geom_rgba[geom_id, :3] = obstacle_rgb
        elif name.startswith("distractor_obstacle_"):
            model.geom_rgba[geom_id, :3] = obstacle_rgb
        elif name.startswith("distractor_target_"):
            model.geom_rgba[geom_id, :3] = target_rgb
        elif name == "target_area_geom":
            model.geom_rgba[geom_id, :3] = target_rgb
        elif name == "target_area_safe_region_geom":
            model.geom_rgba[geom_id, :3] = target_rgb * 0.55


def _box_blur(image: FloatArray) -> FloatArray:
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
    blurred = np.zeros_like(image)
    for row_offset in range(3):
        for column_offset in range(3):
            blurred += padded[
                row_offset : row_offset + image.shape[0],
                column_offset : column_offset + image.shape[1],
            ]
    return blurred / 9.0


def apply_frame_visual_parameters(
    rgb: UInt8Array,
    parameters: FrameVisualParameters,
    *,
    noise_seed: int,
) -> UInt8Array:
    """RGB actor frame'ine deterministik photometric kamera etkileri uygula."""

    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB uint8[H,W,3] olmalı")
    if noise_seed < 0:
        raise ValueError("noise_seed negatif olamaz")
    image = rgb.astype(np.float64) / 255.0
    image = np.power(
        np.clip(image * parameters.exposure_gain, 0.0, 1.0), parameters.gamma
    )
    if parameters.vignette_strength > 0.0:
        y = np.linspace(-1.0, 1.0, rgb.shape[0], dtype=np.float64)
        x = np.linspace(-1.0, 1.0, rgb.shape[1], dtype=np.float64)
        radius_squared = y[:, None] ** 2 + x[None, :] ** 2
        vignette = 1.0 - parameters.vignette_strength * np.clip(
            radius_squared / 2.0, 0.0, 1.0
        )
        image *= vignette[..., None]
    if parameters.apply_blur:
        image = _box_blur(image)
    if parameters.noise_std > 0.0:
        rng = np.random.default_rng(noise_seed)
        image += rng.normal(0.0, parameters.noise_std, size=image.shape)
    return np.clip(np.rint(image * 255.0), 0.0, 255.0).astype(np.uint8)


_VIGNETTE_BASE_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _vignette_base(height: int, width: int) -> torch.Tensor:
    """`clip(r^2 / 2, 0, 1)` haritası; şekle göre önbelleklenir."""

    key = (height, width)
    cached = _VIGNETTE_BASE_CACHE.get(key)
    if cached is None:
        y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
        x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
        cached = torch.clamp((y[:, None] ** 2 + x[None, :] ** 2) / 2.0, 0.0, 1.0)
        _VIGNETTE_BASE_CACHE[key] = cached
    return cached


def make_noise_bank(height: int, width: int, count: int, seed: int) -> torch.Tensor:
    """Birim normal gürültü bankası `(count, H, W, 3)`; seed'li, reset başına bir kez.

    Kare başına `randn` üretmek (`~2 ms`) bankadan seed'li bir dilim seçip
    dairesel kaydırmaktan (`~0.3 ms`) pahalıdır. Marjinal dağılım aynıdır;
    kareler arası gürültü bağımsızlığı augmentation için gerekmez.
    """

    if min(height, width, count) <= 0 or seed < 0:
        raise ValueError(
            "Gürültü bankası boyutları pozitif, seed negatif olmayan olmalı"
        )
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randn((count, height, width, 3), generator=generator)


def apply_frame_visual_parameters_torch(
    rgb: UInt8Array,
    parameters: FrameVisualParameters,
    *,
    noise_seed: int,
    noise_bank: torch.Tensor | None = None,
) -> UInt8Array:
    """`apply_frame_visual_parameters`'ın runtime için hızlı torch eşleniği.

    ADR-022 §6: RL ortamı her yakalanan kareye fotometrik etki uygular; NumPy
    sürümü `240 x 320` karede `~5 ms`. Bu sürüm kazanç + gamma'yı 256 girişli
    arama tablosuyla, vinyeti önbellekli haritayla, gürültüyü verilirse
    `noise_bank`'tan seed'li dilim + dairesel kaydırmayla uygular. Parametre
    anlamları aynıdır; `float32` kullandığı için NumPy sürümüyle gürültüsüz
    koşulda bir gri seviye içinde eşleşir. Aynı `noise_seed` (ve aynı banka)
    aynı çıktıyı verir.
    """

    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB uint8[H,W,3] olmalı")
    if noise_seed < 0:
        raise ValueError("noise_seed negatif olamaz")
    height, width = rgb.shape[0], rgb.shape[1]
    # Kazanç + gamma yalnız piksel değerinin fonksiyonu: tablo birebir aynıdır.
    levels = torch.arange(256, dtype=torch.float32) / 255.0
    lookup = torch.clamp(levels * parameters.exposure_gain, 0.0, 1.0).pow(
        parameters.gamma
    )
    image = lookup[torch.from_numpy(np.ascontiguousarray(rgb)).to(torch.int64)]
    if parameters.vignette_strength > 0.0:
        vignette = 1.0 - parameters.vignette_strength * _vignette_base(height, width)
        image = image * vignette[..., None]
    if parameters.apply_blur:
        padded = functional.pad(
            image.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode="replicate"
        )
        image = (
            functional.avg_pool2d(padded, kernel_size=3, stride=1)
            .squeeze(0)
            .permute(1, 2, 0)
        )
    if parameters.noise_std > 0.0:
        if noise_bank is not None:
            if noise_bank.ndim != 4 or tuple(noise_bank.shape[1:]) != (
                height,
                width,
                3,
            ):
                raise ValueError("noise_bank shape'i (count, H, W, 3) olmalı")
            picks = np.random.SeedSequence([int(noise_seed), 0xA5]).generate_state(
                3, dtype=np.uint32
            )
            field = noise_bank[int(picks[0]) % int(noise_bank.shape[0])]
            field = torch.roll(
                field,
                shifts=(int(picks[1]) % height, int(picks[2]) % width),
                dims=(0, 1),
            )
        else:
            generator = torch.Generator().manual_seed(int(noise_seed))
            field = torch.randn(image.shape, generator=generator)
        image = image + field * float(parameters.noise_std)
    return torch.clamp(torch.round(image * 255.0), 0.0, 255.0).to(torch.uint8).numpy()
