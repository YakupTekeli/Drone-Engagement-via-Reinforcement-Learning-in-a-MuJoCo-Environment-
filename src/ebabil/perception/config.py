"""IMX477 ailesiyle eşleşen simülasyon kamera konfigürasyonu."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_CONFIG = PACKAGE_ROOT / "configs" / "environment" / "camera.toml"


def _float_tuple(raw: object, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) != size:
        raise ValueError(f"{name} {size} değer içermeli")
    values = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} yalnız sonlu değerler içermeli")
    return values


@dataclass(frozen=True)
class CameraConfig:
    """Deploy RGB kamera modeli; depth ve segmentation actor özelliği değildir."""

    name: str
    sensor_width_px: int
    sensor_height_px: int
    output_width_px: int
    output_height_px: int
    frame_rate_hz: float
    horizontal_fov_rad: float
    position_body_m: tuple[float, float, float]
    pitch_down_rad: float

    @property
    def output_aspect_ratio(self) -> float:
        return self.output_width_px / self.output_height_px

    @property
    def vertical_fov_rad(self) -> float:
        return 2.0 * math.atan(
            math.tan(self.horizontal_fov_rad / 2.0) / self.output_aspect_ratio
        )

    @property
    def vertical_half_fov_rad(self) -> float:
        return self.vertical_fov_rad / 2.0

    @property
    def minimum_pitch_for_nadir_view_rad(self) -> float:
        """Aracın tam altındaki noktayı görüntüye sokan en küçük montaj açısı."""

        return math.pi / 2.0 - self.vertical_half_fov_rad

    @property
    def nadir_in_view(self) -> bool:
        """Sabit montajlı kamera aracın tam altını görüyor mu?"""

        return self.pitch_down_rad >= self.minimum_pitch_for_nadir_view_rad

    def ground_swath_m(self, altitude_m: float) -> tuple[float, float]:
        """Düz uçuşta zemin izinin nadir'den yakın ve uzak sınırını ver.

        Yakın sınır `0.0` ise nadir görüş içindedir; uzak sınır `inf` ise ufuk
        görüş içindedir. Değerler irtifayla doğrusal ölçeklenir, bu nedenle
        yükseğe çıkmak nadir kör bölgesini kapatmaz.
        """

        if not math.isfinite(altitude_m) or altitude_m <= 0.0:
            raise ValueError("altitude_m pozitif ve sonlu olmalı")
        lower_edge_rad = self.pitch_down_rad + self.vertical_half_fov_rad
        upper_edge_rad = self.pitch_down_rad - self.vertical_half_fov_rad
        near_m = (
            0.0
            if lower_edge_rad >= math.pi / 2.0
            else altitude_m / math.tan(lower_edge_rad)
        )
        far_m = (
            math.inf if upper_edge_rad <= 0.0 else altitude_m / math.tan(upper_edge_rad)
        )
        return near_m, far_m

    @property
    def focal_length_px(self) -> float:
        return self.output_width_px / (2.0 * math.tan(self.horizontal_fov_rad / 2.0))

    @property
    def principal_point_px(self) -> tuple[float, float]:
        return (
            (self.output_width_px - 1) / 2.0,
            (self.output_height_px - 1) / 2.0,
        )

    @property
    def xyaxes_body(self) -> tuple[float, float, float, float, float, float]:
        """MuJoCo kamera +x(right), +y(up) eksenlerini FLU gövdede döndür."""

        sine = math.sin(self.pitch_down_rad)
        cosine = math.cos(self.pitch_down_rad)
        return (0.0, -1.0, 0.0, sine, 0.0, cosine)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("camera.name boş olamaz")
        dimensions = {
            "sensor_width_px": self.sensor_width_px,
            "sensor_height_px": self.sensor_height_px,
            "output_width_px": self.output_width_px,
            "output_height_px": self.output_height_px,
        }
        if any(value <= 0 for value in dimensions.values()):
            raise ValueError("Kamera sensör ve çıktı boyutları pozitif olmalı")
        if (
            self.output_width_px > self.sensor_width_px
            or self.output_height_px > self.sensor_height_px
        ):
            raise ValueError("Kamera çıktısı sensör çözünürlüğünü aşamaz")
        if not math.isfinite(self.frame_rate_hz) or self.frame_rate_hz <= 0.0:
            raise ValueError("camera.frame_rate_hz pozitif ve sonlu olmalı")
        if not 0.0 < self.horizontal_fov_rad < math.pi:
            raise ValueError("camera.horizontal_fov_rad (0, pi) aralığında olmalı")
        if not 0.0 <= self.pitch_down_rad < math.pi / 2.0:
            raise ValueError("camera.pitch_down_rad [0, pi/2) aralığında olmalı")
        _float_tuple(self.position_body_m, 3, "camera.position_body_m")


def load_camera_config(path: str | Path = DEFAULT_CAMERA_CONFIG) -> CameraConfig:
    """TOML dosyasından doğrulanmış kamera konfigürasyonu yükle."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Kamera konfigürasyonu bulunamadı: {config_path}")
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)["camera"]
    position = _float_tuple(raw["position_body_m"], 3, "camera.position_body_m")
    config = CameraConfig(
        name=str(raw["name"]),
        sensor_width_px=int(raw["sensor_width_px"]),
        sensor_height_px=int(raw["sensor_height_px"]),
        output_width_px=int(raw["output_width_px"]),
        output_height_px=int(raw["output_height_px"]),
        frame_rate_hz=float(raw["frame_rate_hz"]),
        horizontal_fov_rad=math.radians(float(raw["horizontal_fov_deg"])),
        position_body_m=(position[0], position[1], position[2]),
        pitch_down_rad=math.radians(float(raw["pitch_down_deg"])),
    )
    config.validate()
    return config
