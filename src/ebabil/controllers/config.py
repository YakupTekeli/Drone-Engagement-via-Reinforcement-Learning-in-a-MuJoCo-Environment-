"""Kademeli uçuş kontrolcüsü konfigürasyonu."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROLLER_CONFIG = PACKAGE_ROOT / "configs" / "controller" / "cascaded.toml"


def _triple(
    raw: object, name: str, *, allow_zero: bool = True
) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{name} üç değer içermeli")
    values = float(raw[0]), float(raw[1]), float(raw[2])
    lower_bound = 0.0 if allow_zero else 1e-15
    if any(not math.isfinite(value) or value < lower_bound for value in values):
        qualifier = "negatif olmayan" if allow_zero else "pozitif"
        raise ValueError(f"{name} üç {qualifier} ve sonlu değer içermeli")
    return values


@dataclass(frozen=True)
class PositionControllerConfig:
    kp: tuple[float, float, float]
    kd: tuple[float, float, float]
    max_horizontal_acceleration_m_s2: float
    max_vertical_acceleration_m_s2: float
    max_tilt_rad: float

    def validate(self) -> None:
        _triple(self.kp, "position.kp")
        _triple(self.kd, "position.kd")
        scalars = {
            "max_horizontal_acceleration_m_s2": self.max_horizontal_acceleration_m_s2,
            "max_vertical_acceleration_m_s2": self.max_vertical_acceleration_m_s2,
            "max_tilt_rad": self.max_tilt_rad,
        }
        for name, value in scalars.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"position.{name} pozitif ve sonlu olmalı")
        if self.max_tilt_rad >= math.pi / 2.0:
            raise ValueError("position.max_tilt_rad pi/2 değerinden küçük olmalı")


@dataclass(frozen=True)
class AttitudeControllerConfig:
    kp: tuple[float, float, float]
    max_body_rate_rad_s: tuple[float, float, float]

    def validate(self) -> None:
        _triple(self.kp, "attitude.kp")
        _triple(
            self.max_body_rate_rad_s,
            "attitude.max_body_rate_rad_s",
            allow_zero=False,
        )


@dataclass(frozen=True)
class BodyRateControllerConfig:
    kp: tuple[float, float, float]
    ki: tuple[float, float, float]
    integral_limit_rad: tuple[float, float, float]
    max_torque_nm: tuple[float, float, float]

    def validate(self) -> None:
        _triple(self.kp, "body_rate.kp")
        _triple(self.ki, "body_rate.ki")
        _triple(
            self.integral_limit_rad,
            "body_rate.integral_limit_rad",
            allow_zero=False,
        )
        _triple(
            self.max_torque_nm,
            "body_rate.max_torque_nm",
            allow_zero=False,
        )


@dataclass(frozen=True)
class ControllerConfig:
    position: PositionControllerConfig
    attitude: AttitudeControllerConfig
    body_rate: BodyRateControllerConfig

    def validate(self) -> None:
        self.position.validate()
        self.attitude.validate()
        self.body_rate.validate()


def load_controller_config(
    path: str | Path = DEFAULT_CONTROLLER_CONFIG,
) -> ControllerConfig:
    """TOML dosyasından doğrulanmış kontrolcü konfigürasyonu yükle."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Kontrolcü konfigürasyonu bulunamadı: {config_path}")
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    position_raw = raw["position"]
    attitude_raw = raw["attitude"]
    body_rate_raw = raw["body_rate"]
    config = ControllerConfig(
        position=PositionControllerConfig(
            kp=_triple(position_raw["kp"], "position.kp"),
            kd=_triple(position_raw["kd"], "position.kd"),
            max_horizontal_acceleration_m_s2=float(
                position_raw["max_horizontal_acceleration_m_s2"]
            ),
            max_vertical_acceleration_m_s2=float(
                position_raw["max_vertical_acceleration_m_s2"]
            ),
            max_tilt_rad=float(position_raw["max_tilt_rad"]),
        ),
        attitude=AttitudeControllerConfig(
            kp=_triple(attitude_raw["kp"], "attitude.kp"),
            max_body_rate_rad_s=_triple(
                attitude_raw["max_body_rate_rad_s"],
                "attitude.max_body_rate_rad_s",
                allow_zero=False,
            ),
        ),
        body_rate=BodyRateControllerConfig(
            kp=_triple(body_rate_raw["kp"], "body_rate.kp"),
            ki=_triple(body_rate_raw["ki"], "body_rate.ki"),
            integral_limit_rad=_triple(
                body_rate_raw["integral_limit_rad"],
                "body_rate.integral_limit_rad",
                allow_zero=False,
            ),
            max_torque_nm=_triple(
                body_rate_raw["max_torque_nm"],
                "body_rate.max_torque_nm",
                allow_zero=False,
            ),
        ),
    )
    config.validate()
    return config
