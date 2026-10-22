"""Terminal hover yardımcısı: dedektör kutusuyla klasik yerleşme (ADR-025).

REV-043/REV-051: RL politikası hedefe ulaşıyor fakat beş hover koşulunu
birlikte sağlayacak biçimde yerleşemiyor. Bu modül terminal durumlarda
politikanın action'ını, yalnız deploy edilebilir girdilerle (dedektörün hedef
kutusu, uçuş kontrolcüsü attitude ve irtifa tahmini) hesaplanan bir hız
komutuyla değiştirir:

1. Hedef merkezi tahmini: kutunun üst kenarı (uzak kenar) zemin düzlemine
   izdüşürülür; hedef `2 x 2 m` olduğu için merkez, uzak kenardan bakış
   yönünde `half_extent_guess_m` (1.0 hizalı, 1.41 köşegen; varsayılan 1.2)
   geriye alınır. Kutunun alt kenarı kadraj içindeyse yakın kenar da
   izdüşürülür ve merkez iki kenarın ortasıdır. Yanal konum kutu merkez
   sütunundan gelir.
2. Komut: düzey (yaw) çerçevesinde `v = kp * offset` (`0.5 m/s` sınır),
   irtifa `kp_vertical * (hover_altitude - z)`, yaw hedefi yatayda ortalayacak
   kadar (`kyaw * cx_norm`). Hedef kutusu yoksa yerinde hover.

Gövde FLU, kamera geometrisi `detection_slots.CameraGeometry` ile aynı.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ebabil.perception.detection_slots import CameraGeometry
from ebabil.perception.detections import Detection

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TerminalAssistConfig:
    enabled: bool = False
    states: tuple[str, ...] = ("approach", "settle", "hold")
    kp_horizontal: float = 1.0
    kp_vertical: float = 1.0
    kyaw: float = 0.8
    max_horizontal_speed_m_s: float = 0.5
    max_vertical_speed_m_s: float = 0.3
    max_yaw_rate_rad_s: float = 0.5
    half_extent_guess_m: float = 1.2
    #: Kutunun alt kenarı görüntü altına bu kadar yakınsa yakın kenar kadraj
    #: dışı sayılır (dedektör kutusu kenardan birkaç piksel kısa kalabilir).
    bottom_margin_px: float = 4.0

    def validate(self) -> None:
        for name in (
            "kp_horizontal",
            "kp_vertical",
            "kyaw",
            "max_horizontal_speed_m_s",
            "max_vertical_speed_m_s",
            "max_yaw_rate_rad_s",
            "half_extent_guess_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"terminal_assist.{name} negatif olamaz")
        if self.bottom_margin_px < 0.0:
            raise ValueError("terminal_assist.bottom_margin_px negatif olamaz")
        if not self.states:
            raise ValueError("terminal_assist.states boş olamaz")


def level_frame(gravity_direction_body: npt.ArrayLike) -> FloatArray:
    """Gövde eksenlerinden yaw'ı koruyan düzey çerçeve; satırlar x, y, z (gövdede)."""

    gravity = np.asarray(gravity_direction_body, dtype=np.float64)
    norm = float(np.linalg.norm(gravity))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("Yerçekimi yönü sıfır olamaz")
    z_level = -gravity / norm
    x_body = np.array([1.0, 0.0, 0.0])
    x_level = x_body - float(x_body @ z_level) * z_level
    if float(np.linalg.norm(x_level)) <= 1e-9:
        raise ValueError("Gövde x ekseni düşey; düzey çerçeve tanımsız")
    x_level /= np.linalg.norm(x_level)
    y_level = np.cross(z_level, x_level)
    return np.stack([x_level, y_level, z_level])


def ground_point_level(
    geometry: CameraGeometry,
    u_edge: float,
    v_edge: float,
    *,
    gravity_direction_body: npt.ArrayLike,
    altitude_m: float,
) -> FloatArray | None:
    """Piksel ışınının zemini kestiği noktanın düzey çerçevede `(x, y)` konumu."""

    frame = level_frame(gravity_direction_body)
    ray = geometry.ray_body(u_edge, v_edge)
    ray_level = frame @ ray
    camera_level = frame @ np.asarray(geometry.position_body_m, dtype=np.float64)
    camera_height = float(altitude_m) + float(camera_level[2])
    if ray_level[2] >= -1e-6 or camera_height <= 0.0:
        return None
    scale = camera_height / (-float(ray_level[2]))
    point = camera_level + scale * ray_level
    return np.asarray(point[:2], dtype=np.float64)


def _point_on_column_at_range(
    geometry: CameraGeometry,
    u_edge: float,
    v_top: float,
    target_range_m: float,
    *,
    gravity_direction_body: npt.ArrayLike,
    altitude_m: float,
) -> FloatArray | None:
    """Sütun `u` üzerinde yatay menzili `target_range_m` olan satırın zemin noktası.

    Menzil satırla monoton azalır; `v_top` ile görüntü altı arasında ikiye bölme.
    Hedef menzil bu aralığın dışındaysa `None`.
    """

    def range_at(v: float) -> float | None:
        point = ground_point_level(
            geometry,
            u_edge,
            v,
            gravity_direction_body=gravity_direction_body,
            altitude_m=altitude_m,
        )
        return None if point is None else float(np.linalg.norm(point))

    low, high = float(v_top), float(geometry.height_px)
    range_low, range_high = range_at(low), range_at(high)
    if range_low is None or range_high is None:
        return None
    if not range_high <= target_range_m <= range_low:
        return None
    for _ in range(40):
        mid = 0.5 * (low + high)
        value = range_at(mid)
        if value is None:
            return None
        if value > target_range_m:
            low = mid
        else:
            high = mid
    return ground_point_level(
        geometry,
        u_edge,
        0.5 * (low + high),
        gravity_direction_body=gravity_direction_body,
        altitude_m=altitude_m,
    )


@dataclass(frozen=True)
class TargetOffsetEstimate:
    #: Düzey çerçevede hedef merkezine göreli konum (ileri, sol), metre.
    offset_xy_m: FloatArray
    near_edge_visible: bool
    center_x_norm: float


def estimate_target_offset(
    box: Detection,
    geometry: CameraGeometry,
    config: TerminalAssistConfig,
    *,
    gravity_direction_body: npt.ArrayLike,
    altitude_m: float,
) -> TargetOffsetEstimate | None:
    center_x, _ = box.center
    far = ground_point_level(
        geometry,
        center_x,
        box.y_min,
        gravity_direction_body=gravity_direction_body,
        altitude_m=altitude_m,
    )
    if far is None:
        return None
    near_visible = box.y_max < geometry.height_px - config.bottom_margin_px
    if near_visible:
        near = ground_point_level(
            geometry,
            center_x,
            box.y_max,
            gravity_direction_body=gravity_direction_body,
            altitude_m=altitude_m,
        )
        if near is None:
            near_visible = False
    if near_visible:
        assert near is not None
        center = 0.5 * (far + near)
    else:
        # Yakın kenar kadraj dışı: merkez menzili uzak kenardan geri alınır ve
        # yanal konum, aynı sütunda o menzile karşılık gelen satırdan okunur
        # (uzak satırdaki yanal okuma menzille ölçeklenip hata büyütür).
        center_range = max(
            float(np.linalg.norm(far)) - config.half_extent_guess_m, 0.05
        )
        column_point = _point_on_column_at_range(
            geometry,
            center_x,
            box.y_min,
            center_range,
            gravity_direction_body=gravity_direction_body,
            altitude_m=altitude_m,
        )
        if column_point is not None:
            center = column_point
        else:
            direction = far / max(float(np.linalg.norm(far)), 1e-9)
            center = far - config.half_extent_guess_m * direction
    center_x_norm = float(np.clip(2.0 * center_x / geometry.width_px - 1.0, -1.0, 1.0))
    return TargetOffsetEstimate(
        offset_xy_m=np.asarray(center, dtype=np.float64),
        near_edge_visible=near_visible,
        center_x_norm=center_x_norm,
    )


def assist_action(
    estimate: TargetOffsetEstimate | None,
    *,
    altitude_error_m: float,
    config: TerminalAssistConfig,
    max_horizontal_speed_m_s: float,
    max_vertical_speed_m_s: float,
    max_yaw_rate_rad_s: float,
) -> npt.NDArray[np.float32]:
    """Normalize action `[ileri, yanal, dikey, yaw]`; hedef yoksa yerinde hover."""

    if estimate is None:
        horizontal = np.zeros(2, dtype=np.float64)
        yaw_rate = 0.0
    else:
        horizontal = config.kp_horizontal * estimate.offset_xy_m
        speed = float(np.linalg.norm(horizontal))
        if speed > config.max_horizontal_speed_m_s:
            horizontal *= config.max_horizontal_speed_m_s / speed
        yaw_rate = float(
            np.clip(
                -config.kyaw * estimate.center_x_norm,
                -config.max_yaw_rate_rad_s,
                config.max_yaw_rate_rad_s,
            )
        )
    vertical = float(
        np.clip(
            config.kp_vertical * altitude_error_m,
            -config.max_vertical_speed_m_s,
            config.max_vertical_speed_m_s,
        )
    )
    action = np.array(
        [
            horizontal[0] / max_horizontal_speed_m_s,
            horizontal[1] / max_horizontal_speed_m_s,
            vertical / max_vertical_speed_m_s,
            yaw_rate / max_yaw_rate_rad_s,
        ],
        dtype=np.float64,
    )
    return np.clip(action, -1.0, 1.0).astype(np.float32)


__all__ = [
    "TargetOffsetEstimate",
    "TerminalAssistConfig",
    "assist_action",
    "estimate_target_offset",
    "ground_point_level",
    "level_frame",
]
