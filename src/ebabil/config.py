"""Proje konfigürasyonlarını tipli veri yapılarına yükle."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_VEHICLE_CONFIG = PACKAGE_ROOT / "configs" / "vehicle" / "quadrotor.toml"


def _float_triple(raw: object, name: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{name} üç değer içermeli")
    return float(raw[0]), float(raw[1]), float(raw[2])


def _int_quadruple(raw: object, name: str) -> tuple[int, int, int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError(f"{name} dört değer içermeli")
    return int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])


@dataclass(frozen=True)
class RotorConfig:
    """Dört rotorlu propulsion modeli parametreleri."""

    count: int
    max_thrust_n: float
    max_speed_rad_s: float
    thrust_coefficient: float
    yaw_moment_ratio_m: float
    time_constant_up_s: float
    time_constant_down_s: float
    reaction_torque_signs: tuple[int, int, int, int]

    def validate(self) -> None:
        if self.count != 4:
            raise ValueError(f"İlk model tam dört rotor bekler, alınan: {self.count}")
        positive_values = {
            "max_thrust_n": self.max_thrust_n,
            "max_speed_rad_s": self.max_speed_rad_s,
            "thrust_coefficient": self.thrust_coefficient,
            "yaw_moment_ratio_m": self.yaw_moment_ratio_m,
            "time_constant_up_s": self.time_constant_up_s,
            "time_constant_down_s": self.time_constant_down_s,
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} pozitif ve sonlu olmalı, alınan: {value}")
        if len(self.reaction_torque_signs) != self.count:
            raise ValueError("reaction_torque_signs rotor sayısıyla eşleşmeli")
        if any(sign not in (-1, 1) for sign in self.reaction_torque_signs):
            raise ValueError("reaction_torque_signs yalnızca -1 veya +1 içerebilir")
        if sum(self.reaction_torque_signs) != 0:
            raise ValueError(
                "Nominal hover için reaction torque işaretleri dengeli olmalı"
            )

        thrust_at_max_speed = self.thrust_coefficient * self.max_speed_rad_s**2
        if not math.isclose(thrust_at_max_speed, self.max_thrust_n, rel_tol=1e-9):
            raise ValueError(
                "thrust_coefficient * max_speed_rad_s², max_thrust_n ile eşleşmeli"
            )


@dataclass(frozen=True)
class AerodynamicsConfig:
    """Gövde eksenlerindeki sadeleştirilmiş aerodinamik parametreler."""

    quadratic_drag_coefficients_n_s2_m2: tuple[float, float, float]

    def validate(self) -> None:
        coefficients = self.quadratic_drag_coefficients_n_s2_m2
        if len(coefficients) != 3 or any(
            not math.isfinite(value) or value < 0.0 for value in coefficients
        ):
            raise ValueError(
                "quadratic_drag_coefficients_n_s2_m2 üç sonlu, negatif olmayan "
                "değer içermeli"
            )


@dataclass(frozen=True)
class VehicleConfig:
    """Nominal quadrotor gövde ve propulsion parametreleri."""

    mass_kg: float
    inertia_diagonal_kg_m2: tuple[float, float, float]
    arm_length_m: float
    gravity_m_s2: float
    rotor: RotorConfig
    aerodynamics: AerodynamicsConfig

    @property
    def hover_thrust_per_rotor_n(self) -> float:
        return self.mass_kg * self.gravity_m_s2 / self.rotor.count

    @property
    def thrust_to_weight_ratio(self) -> float:
        return (
            self.rotor.count
            * self.rotor.max_thrust_n
            / (self.mass_kg * self.gravity_m_s2)
        )

    def validate(self) -> None:
        if not math.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise ValueError("mass_kg pozitif ve sonlu olmalı")
        if len(self.inertia_diagonal_kg_m2) != 3 or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.inertia_diagonal_kg_m2
        ):
            raise ValueError(
                "inertia_diagonal_kg_m2 üç pozitif ve sonlu değer içermeli"
            )
        if not math.isfinite(self.arm_length_m) or self.arm_length_m <= 0.0:
            raise ValueError("arm_length_m pozitif ve sonlu olmalı")
        if not math.isfinite(self.gravity_m_s2) or self.gravity_m_s2 <= 0.0:
            raise ValueError("gravity_m_s2 pozitif ve sonlu olmalı")
        self.rotor.validate()
        self.aerodynamics.validate()
        if self.hover_thrust_per_rotor_n >= self.rotor.max_thrust_n:
            raise ValueError("Nominal hover thrust'ı rotor limitinin altında olmalı")


def load_vehicle_config(path: str | Path = DEFAULT_VEHICLE_CONFIG) -> VehicleConfig:
    """TOML dosyasından doğrulanmış quadrotor konfigürasyonu yükle."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Araç konfigürasyonu bulunamadı: {config_path}")

    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    vehicle_raw = raw["vehicle"]
    rotor_raw = raw["rotor"]
    aerodynamics_raw = raw["aerodynamics"]
    rotor = RotorConfig(
        count=int(rotor_raw["count"]),
        max_thrust_n=float(rotor_raw["max_thrust_n"]),
        max_speed_rad_s=float(rotor_raw["max_speed_rad_s"]),
        thrust_coefficient=float(rotor_raw["thrust_coefficient"]),
        yaw_moment_ratio_m=float(rotor_raw["yaw_moment_ratio_m"]),
        time_constant_up_s=float(rotor_raw["time_constant_up_s"]),
        time_constant_down_s=float(rotor_raw["time_constant_down_s"]),
        reaction_torque_signs=_int_quadruple(
            rotor_raw["reaction_torque_signs"], "rotor.reaction_torque_signs"
        ),
    )
    config = VehicleConfig(
        mass_kg=float(vehicle_raw["mass_kg"]),
        inertia_diagonal_kg_m2=_float_triple(
            vehicle_raw["inertia_diagonal_kg_m2"],
            "vehicle.inertia_diagonal_kg_m2",
        ),
        arm_length_m=float(vehicle_raw["arm_length_m"]),
        gravity_m_s2=float(vehicle_raw["gravity_m_s2"]),
        rotor=rotor,
        aerodynamics=AerodynamicsConfig(
            quadratic_drag_coefficients_n_s2_m2=_float_triple(
                aerodynamics_raw["quadratic_drag_coefficients_n_s2_m2"],
                "aerodynamics.quadratic_drag_coefficients_n_s2_m2",
            )
        ),
    )
    config.validate()
    return config
