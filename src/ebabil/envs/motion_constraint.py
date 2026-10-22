"""Kamera konisi hareket kısıtı: "baktığın yöne uç" (ADR-024, REV-051).

Tek ileri bakan kamera yanı ve arkayı görmez. ADR-020 holonomik action bu kör
alana translasyonu ödüle bırakmıştı; REV-051'de çarpışmaların 13/14'ünün
kameranın görmediği yöne hareketle olduğu ölçüldü. Kısıt:

- Seyir/arama durumlarında gövde çerçevesindeki yatay hız komutu, kameranın
  optik ekseni etrafındaki yarım açısı `half_angle_deg` olan koniye Öklid
  izdüşümüyle taşınır. Koni içindeki komut değişmez; yana komutun koni
  kenarına izdüşümü kalır; geriye komut sıfırlanır. Yön değiştirmek için
  yaw kullanılır.
- Terminal durumlarda (`relaxed_states`, varsayılan settle/hold; ADR-024
  Revizyon 1 approach'ı çıkardı) hedef üstünde ince düzeltme gerekir (ADR-020
  gerekçesi): koni içi bileşen korunur, koni dışı bileşen
  `terminal_blind_speed_m_s` ile sınırlanır.

Dikey hız ve yaw-rate etkilenmez.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
HOVER_STATE_NAMES = (
    "transit",
    "search",
    "acquired",
    "approach",
    "settle",
    "hold",
    "success",
    "abort",
)


@dataclass(frozen=True)
class MotionConstraintConfig:
    enabled: bool = False
    half_angle_deg: float = 30.0
    terminal_blind_speed_m_s: float = 0.5
    relaxed_states: tuple[str, ...] = ("settle", "hold")

    def validate(self) -> None:
        if not 0.0 < self.half_angle_deg < 90.0:
            raise ValueError(
                "motion_constraint.half_angle_deg (0, 90) aralığında olmalı"
            )
        if self.terminal_blind_speed_m_s < 0.0 or not math.isfinite(
            self.terminal_blind_speed_m_s
        ):
            raise ValueError(
                "motion_constraint.terminal_blind_speed_m_s negatif olamaz"
            )
        unknown = sorted(set(self.relaxed_states) - set(HOVER_STATE_NAMES))
        if unknown:
            raise ValueError(
                f"motion_constraint.relaxed_states bilinmeyen durum: {unknown}"
            )

    @property
    def half_angle_rad(self) -> float:
        return math.radians(self.half_angle_deg)


def project_to_camera_cone(
    velocity_xy: npt.ArrayLike, half_angle_rad: float
) -> FloatArray:
    """`{x >= 0, |y| <= tan(a) x}` konisine Öklid izdüşümü (gövde çerçevesi, FLU)."""

    velocity = np.asarray(velocity_xy, dtype=np.float64)
    if velocity.shape != (2,) or not np.all(np.isfinite(velocity)):
        raise ValueError("velocity_xy sonlu ve shape (2,) olmalı")
    if not 0.0 < half_angle_rad < math.pi / 2.0:
        raise ValueError("half_angle_rad (0, pi/2) aralığında olmalı")
    speed = float(np.linalg.norm(velocity))
    if speed <= 1e-12:
        return np.zeros(2, dtype=np.float64)
    angle = math.atan2(float(velocity[1]), float(velocity[0]))
    if abs(angle) <= half_angle_rad:
        return velocity.copy()
    edge = np.array(
        [math.cos(half_angle_rad), math.copysign(math.sin(half_angle_rad), angle)],
        dtype=np.float64,
    )
    along = float(velocity @ edge)
    if along <= 0.0:
        return np.zeros(2, dtype=np.float64)
    return along * edge


def constrain_horizontal_velocity(
    velocity_xy: npt.ArrayLike,
    config: MotionConstraintConfig,
    *,
    relaxed: bool,
) -> FloatArray:
    """Kısıtlı yatay hız komutu; `relaxed` terminal durumlarda doğrudur."""

    velocity = np.asarray(velocity_xy, dtype=np.float64)
    inside = project_to_camera_cone(velocity, config.half_angle_rad)
    if not relaxed:
        return inside
    outside = velocity - inside
    magnitude = float(np.linalg.norm(outside))
    limit = float(config.terminal_blind_speed_m_s)
    if magnitude > limit:
        outside = outside * (limit / magnitude) if magnitude > 0.0 else outside
    return inside + outside


@dataclass(frozen=True)
class SpeedGovernorConfig:
    """ADR-024 Revizyon 2: menzil tabanlı ileri hız sınırlayıcı (dedektör kutuları).

    Kadrajın orta koridorunda (`|cx_norm| <= corridor_half_width_norm`) bir engel
    yuvası varsa yatay hız büyüklüğü `gain_per_s * (menzil - safe_range_m)` ile
    sınırlanır (`min_speed_m_s` tabanı). Menzili geçersiz (tabanı kadraj dışı,
    yani çok yakın) yuva `invalid_range_m` menzilinde sayılır.
    """

    enabled: bool = False
    corridor_half_width_norm: float = 0.4
    safe_range_m: float = 0.6
    gain_per_s: float = 1.0
    min_speed_m_s: float = 0.15
    invalid_range_m: float = 0.5
    range_max_m: float = 12.0
    #: Dönüş-hız bağlaması: yatay hız `max_speed x (1 - turn_penalty x |yaw| / yaw_max)`.
    #: 30 Eylül ölçümünde (0.8) yeniden eğitilmemiş politikalarda çarpışmayı
    #: ARTIRDI (seed 1 %5 -> %12.5, seed 2 %22.5 -> %45); varsayılan kapalı (0).
    turn_penalty: float = 0.0

    def validate(self) -> None:
        if not 0.0 < self.corridor_half_width_norm <= 1.0:
            raise ValueError("speed_governor.corridor_half_width_norm (0, 1] olmalı")
        if not 0.0 <= self.turn_penalty <= 1.0:
            raise ValueError("speed_governor.turn_penalty [0, 1] aralığında olmalı")
        for name in (
            "safe_range_m",
            "gain_per_s",
            "min_speed_m_s",
            "invalid_range_m",
            "range_max_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"speed_governor.{name} pozitif olmalı")


def governor_speed_limit(
    slots: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    config: SpeedGovernorConfig,
    *,
    obstacle_slots: int,
) -> float | None:
    """Öndeki en yakın engel yuvasından yatay hız sınırı; engel yoksa `None`.

    `slots` `vision-v4` vektörüdür (yuva başına 8 alan: present, cx, cy, w, h,
    conf, range_norm, range_valid); `cx` `[0, 1]`.
    """

    table = np.asarray(slots, dtype=np.float64).reshape(-1, 8)[:obstacle_slots]
    nearest: float | None = None
    for present, cx, _cy, _w, _h, _conf, range_norm, range_valid in table:
        if present < 0.5:
            continue
        # Yalnız orta koridordaki engeller yolu keser. Kenardaki yakın (menzili
        # geçersiz) engelleri de saymak 30 Eylül ölçümünde çarpışmayı ARTIRDI
        # (seed 1 %5 -> %12.5, seed 2 %22.5 -> %30); koridor kuralı korunur.
        if abs(2.0 * cx - 1.0) > config.corridor_half_width_norm:
            continue
        distance = (
            range_norm * config.range_max_m
            if range_valid > 0.5
            else config.invalid_range_m
        )
        nearest = distance if nearest is None else min(nearest, distance)
    if nearest is None:
        return None
    return max(
        config.gain_per_s * (nearest - config.safe_range_m), config.min_speed_m_s
    )


def turn_speed_limit(
    yaw_rate_rad_s: float,
    config: SpeedGovernorConfig,
    *,
    max_yaw_rate_rad_s: float,
    max_speed_m_s: float,
) -> float | None:
    """Dönüş-hız bağlaması; `turn_penalty = 0` veya yaw yoksa `None`."""

    if config.turn_penalty <= 0.0 or max_yaw_rate_rad_s <= 0.0:
        return None
    fraction = min(abs(float(yaw_rate_rad_s)) / max_yaw_rate_rad_s, 1.0)
    return max(
        max_speed_m_s * (1.0 - config.turn_penalty * fraction), config.min_speed_m_s
    )


def apply_speed_limit(velocity_xy: npt.ArrayLike, limit: float | None) -> FloatArray:
    velocity = np.asarray(velocity_xy, dtype=np.float64)
    if limit is None:
        return velocity.copy()
    speed = float(np.linalg.norm(velocity))
    if speed <= limit or speed <= 1e-12:
        return velocity.copy()
    return velocity * (limit / speed)


__all__ = [
    "HOVER_STATE_NAMES",
    "MotionConstraintConfig",
    "SpeedGovernorConfig",
    "apply_speed_limit",
    "constrain_horizontal_velocity",
    "governor_speed_limit",
    "project_to_camera_cone",
    "turn_speed_limit",
]
