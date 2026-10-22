"""Rangefinder ve algılayıcı hata modeli konfigürasyonu."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SENSOR_CONFIG = PACKAGE_ROOT / "configs" / "environment" / "sensors.toml"


def _triple(raw: object, name: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{name} üç sonlu değer içermeli")
    values = float(raw[0]), float(raw[1]), float(raw[2])
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} üç sonlu değer içermeli")
    return values


@dataclass(frozen=True)
class RangefinderConfig:
    ray_count: int
    field_of_view_rad: float
    min_range_m: float
    max_range_m: float
    origin_body_m: tuple[float, float, float]
    geom_group: int

    def validate(self) -> None:
        if self.ray_count < 1:
            raise ValueError("rangefinder.ray_count pozitif olmalı")
        if not 0.0 < self.field_of_view_rad <= 2.0 * math.pi:
            raise ValueError("rangefinder.field_of_view_rad (0, 2*pi] içinde olmalı")
        if not 0.0 <= self.min_range_m < self.max_range_m:
            raise ValueError("Rangefinder mesafe sınırları geçersiz")
        _triple(self.origin_body_m, "rangefinder.origin_body_m")
        if not 0 <= self.geom_group <= 5:
            raise ValueError("rangefinder.geom_group [0, 5] içinde olmalı")


@dataclass(frozen=True)
class SensorImperfectionConfig:
    sample_period_s: float
    delay_s: float
    noise_std: float
    bias_std: float
    dropout_probability: float

    def validate(self) -> None:
        values = (
            self.sample_period_s,
            self.delay_s,
            self.noise_std,
            self.bias_std,
            self.dropout_probability,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Sensör hata parametreleri sonlu olmalı")
        if self.sample_period_s <= 0.0 or self.delay_s < 0.0:
            raise ValueError("Sample period pozitif, delay negatif olmayan olmalı")
        if self.noise_std < 0.0 or self.bias_std < 0.0:
            raise ValueError("Noise ve bias standart sapmaları negatif olamaz")
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("Dropout olasılığı [0, 1] içinde olmalı")


@dataclass(frozen=True)
class SensorSuiteConfig:
    rangefinder: RangefinderConfig
    imperfections: SensorImperfectionConfig

    def validate(self) -> None:
        self.rangefinder.validate()
        self.imperfections.validate()


def load_sensor_config(path: str | Path = DEFAULT_SENSOR_CONFIG) -> SensorSuiteConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Sensör konfigürasyonu bulunamadı: {config_path}")
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    rangefinder = raw["rangefinder"]
    imperfections = raw["imperfections"]
    config = SensorSuiteConfig(
        rangefinder=RangefinderConfig(
            ray_count=int(rangefinder["ray_count"]),
            field_of_view_rad=float(rangefinder["field_of_view_rad"]),
            min_range_m=float(rangefinder["min_range_m"]),
            max_range_m=float(rangefinder["max_range_m"]),
            origin_body_m=_triple(
                rangefinder["origin_body_m"], "rangefinder.origin_body_m"
            ),
            geom_group=int(rangefinder["geom_group"]),
        ),
        imperfections=SensorImperfectionConfig(
            sample_period_s=float(imperfections["sample_period_s"]),
            delay_s=float(imperfections["delay_s"]),
            noise_std=float(imperfections["noise_std"]),
            bias_std=float(imperfections["bias_std"]),
            dropout_probability=float(imperfections["dropout_probability"]),
        ),
    )
    config.validate()
    return config
