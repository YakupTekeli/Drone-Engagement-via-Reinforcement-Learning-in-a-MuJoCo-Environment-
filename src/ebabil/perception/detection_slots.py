"""`vision-v4` kutu yuvası kodlayıcısı ve zemin düzlemi menzil tahmini (ADR-023 §4).

Politikanın tek görsel girdisi dedektör kutularıdır. Her yuva sekiz alan
taşır: `present, cx, cy, w, h, conf, range_norm, range_valid`. `8` engel
yuvası + `1` hedef yuvası = `72` değer, hepsi `[0, 1]`.

Menzil tahmini: tek kamerada derinlik yoktur. Engeller zemine basar (sahne
üreticisi, REV-035), dolayısıyla kutunun alt kenarındaki ışının zemin
düzlemini kestiği nokta engelin tabanıdır. Işın gövde çerçevesinde kamera
montajından (pitch, konum) kurulur, dünyaya araç attitude'unun yerçekimi
yönüyle (`fc_state`'in kullandığı büyüklük) taşınır; yükseklik araç
irtifasıdır. İkisi de gerçek araçta uçuş kontrolcüsü tahminiyle mevcuttur.

Koordinat sözleşmesi: kutular yarı-açık piksel kenar koordinatındadır
(piksel `i` `[i, i + 1)` aralığını kaplar). Kamera config'inin asal noktası
piksel-merkez koordinatındadır (`(W - 1) / 2`); dönüşüm `- 0.5` iledir.
Gövde çerçevesi FLU (`+x` ileri, `+y` sol, `+z` yukarı).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ebabil.perception.config import CameraConfig
from ebabil.perception.detections import OBSTACLE_CLASS_ID, TARGET_CLASS_ID, Detection

SLOT_FIELDS: tuple[str, ...] = (
    "present",
    "cx",
    "cy",
    "w",
    "h",
    "conf",
    "range_norm",
    "range_valid",
)
SLOT_FIELD_COUNT = len(SLOT_FIELDS)
#: Boş yuva: yalnız `range_norm = 1.0` ("uzak"); diğer alanlar sıfır.
EMPTY_SLOT: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DetectionSlotConfig:
    obstacle_slots: int = 8
    range_max_m: float = 12.0
    obstacle_conf: float = 0.25
    target_conf: float = 0.50
    target_min_area_px: float = 16.0
    #: Kutunun alt kenarı görüntü alt sınırına bu kadar yakınsa taban kadraj
    #: dışıdır ve menzil geçersizdir.
    bottom_margin_px: float = 1.0

    def validate(self) -> None:
        if self.obstacle_slots <= 0:
            raise ValueError("obstacle_slots pozitif olmalı")
        if self.range_max_m <= 0.0 or not math.isfinite(self.range_max_m):
            raise ValueError("range_max_m pozitif ve sonlu olmalı")
        if not 0.0 <= self.obstacle_conf <= 1.0 or not 0.0 <= self.target_conf <= 1.0:
            raise ValueError("Güven eşikleri [0, 1] aralığında olmalı")
        if self.target_min_area_px < 0.0 or self.bottom_margin_px < 0.0:
            raise ValueError("target_min_area_px ve bottom_margin_px negatif olamaz")

    @property
    def slot_count(self) -> int:
        return self.obstacle_slots + 1

    @property
    def vector_size(self) -> int:
        return self.slot_count * SLOT_FIELD_COUNT


@dataclass(frozen=True)
class CameraGeometry:
    """Montaj ve iç parametreler; ışını gövde çerçevesinde üretir."""

    width_px: int
    height_px: int
    focal_px: float
    principal_point_px: tuple[float, float]
    pitch_down_rad: float
    position_body_m: tuple[float, float, float]

    @classmethod
    def from_camera_config(cls, config: CameraConfig) -> CameraGeometry:
        return cls(
            width_px=int(config.output_width_px),
            height_px=int(config.output_height_px),
            focal_px=float(config.focal_length_px),
            principal_point_px=(
                float(config.principal_point_px[0]),
                float(config.principal_point_px[1]),
            ),
            pitch_down_rad=float(config.pitch_down_rad),
            position_body_m=tuple(float(v) for v in config.position_body_m),  # type: ignore[arg-type]
        )

    def ray_body(self, u_edge: float, v_edge: float) -> FloatArray:
        """Kenar koordinatındaki `(u, v)` noktasından geçen ışın (gövde, normalize değil)."""

        pitch = self.pitch_down_rad
        axis = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
        right = np.array([0.0, -1.0, 0.0])
        down = np.array([-math.sin(pitch), 0.0, -math.cos(pitch)])
        cx, cy = self.principal_point_px
        du = (u_edge - 0.5 - cx) / self.focal_px
        dv = (v_edge - 0.5 - cy) / self.focal_px
        return np.asarray(axis + du * right + dv * down, dtype=np.float64)


def ground_range_m(
    geometry: CameraGeometry,
    u_edge: float,
    v_edge: float,
    *,
    gravity_direction_body: npt.ArrayLike,
    altitude_m: float,
) -> float | None:
    """Işının zemin düzlemini (`z = 0`) kestiği noktaya kameradan yatay mesafe.

    Işın aşağı bakmıyorsa veya kamera zeminin üstünde değilse `None`.
    """

    gravity = np.asarray(gravity_direction_body, dtype=np.float64)
    norm = float(np.linalg.norm(gravity))
    if not math.isfinite(norm) or norm <= 1e-9 or not math.isfinite(altitude_m):
        return None
    gravity = gravity / norm
    ray = geometry.ray_body(u_edge, v_edge)
    downward = float(ray @ gravity)
    if downward <= 1e-6:
        return None
    camera_height_m = float(altitude_m) - float(
        np.asarray(geometry.position_body_m) @ gravity
    )
    if camera_height_m <= 0.0:
        return None
    scale = camera_height_m / downward
    horizontal = math.sqrt(max(float(ray @ ray) - downward * downward, 0.0))
    return scale * horizontal


@dataclass(frozen=True)
class SlotEncoding:
    vector: npt.NDArray[np.float32]
    obstacle_detections: int
    obstacles_encoded: int
    range_valid_count: int
    target_present: bool
    #: `[-1, 1]`; hedef yoksa `0.0` (ADR-017 `target_alignment` girişi).
    target_center_x: float
    target_area_px: float
    #: Yuvaya yazılan hedef kutusu (terminal yardımcı için); yoksa None.
    target_box: Detection | None = None


def _slot(
    detection: Detection,
    geometry: CameraGeometry,
    config: DetectionSlotConfig,
    *,
    gravity_direction_body: npt.ArrayLike,
    altitude_m: float,
) -> tuple[list[float], float | None]:
    width, height = float(geometry.width_px), float(geometry.height_px)
    center_x, center_y = detection.center
    touches_bottom = detection.y_max >= height - config.bottom_margin_px
    distance = (
        None
        if touches_bottom
        else ground_range_m(
            geometry,
            center_x,
            detection.y_max,
            gravity_direction_body=gravity_direction_body,
            altitude_m=altitude_m,
        )
    )
    valid = distance is not None
    range_norm = (
        min(distance / config.range_max_m, 1.0) if distance is not None else 1.0
    )
    values = [
        1.0,
        center_x / width,
        center_y / height,
        detection.width / width,
        detection.height / height,
        detection.confidence,
        range_norm,
        1.0 if valid else 0.0,
    ]
    return [float(np.clip(value, 0.0, 1.0)) for value in values], distance


def encode_detection_slots(
    detections: Iterable[Detection],
    *,
    config: DetectionSlotConfig,
    geometry: CameraGeometry,
    gravity_direction_body: npt.ArrayLike,
    altitude_m: float,
) -> SlotEncoding:
    """Kutuları sabit uzunluklu vektöre çevir.

    Engel yuvaları: geçerli menzili olanlar artan menzille, ardından menzili
    geçersizler azalan alanla; eşitlikte `cx`. Fazlası düşer. Hedef yuvası:
    güveni `>= target_conf` ve alanı `>= target_min_area_px` olan en güvenli
    hedef kutusu.
    """

    config.validate()
    detections = tuple(detections)
    obstacles = [
        d
        for d in detections
        if d.class_id == OBSTACLE_CLASS_ID and d.confidence >= config.obstacle_conf
    ]
    targets = [
        d
        for d in detections
        if d.class_id == TARGET_CLASS_ID
        and d.confidence >= config.target_conf
        and d.area >= config.target_min_area_px
    ]
    encoded: list[tuple[tuple[int, float, float], list[float]]] = []
    for detection in obstacles:
        values, distance = _slot(
            detection,
            geometry,
            config,
            gravity_direction_body=gravity_direction_body,
            altitude_m=altitude_m,
        )
        key = (
            (0, distance, detection.center[0])
            if distance is not None
            else (1, -detection.area, detection.center[0])
        )
        encoded.append((key, values))
    encoded.sort(key=lambda item: item[0])
    kept = encoded[: config.obstacle_slots]
    vector = np.tile(np.asarray(EMPTY_SLOT, dtype=np.float32), config.slot_count)
    for index, (_, values) in enumerate(kept):
        start = index * SLOT_FIELD_COUNT
        vector[start : start + SLOT_FIELD_COUNT] = values
    target_present = bool(targets)
    target_center_x = 0.0
    target_area = 0.0
    target_box: Detection | None = None
    range_valid = sum(int(values[7] > 0.5) for _, values in kept)
    if targets:
        target = max(targets, key=lambda d: (d.confidence, d.area))
        values, _ = _slot(
            target,
            geometry,
            config,
            gravity_direction_body=gravity_direction_body,
            altitude_m=altitude_m,
        )
        start = config.obstacle_slots * SLOT_FIELD_COUNT
        vector[start : start + SLOT_FIELD_COUNT] = values
        target_center_x = float(
            np.clip(2.0 * target.center[0] / geometry.width_px - 1.0, -1.0, 1.0)
        )
        target_area = float(target.area)
        target_box = target
    return SlotEncoding(
        vector=vector,
        obstacle_detections=len(obstacles),
        obstacles_encoded=len(kept),
        range_valid_count=range_valid,
        target_present=target_present,
        target_center_x=target_center_x,
        target_area_px=target_area,
        target_box=target_box,
    )


__all__ = [
    "EMPTY_SLOT",
    "SLOT_FIELDS",
    "SLOT_FIELD_COUNT",
    "CameraGeometry",
    "DetectionSlotConfig",
    "SlotEncoding",
    "encode_detection_slots",
    "ground_range_m",
]
