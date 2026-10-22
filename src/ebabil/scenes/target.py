"""Dikdörtgen hedef alan ve güvenli hover iç bölgesi."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ebabil.scenes.generator import SceneGeneratorConfig, StaticScene

FloatArray = npt.NDArray[np.float64]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_AREA_CONFIG = (
    PACKAGE_ROOT / "configs" / "environment" / "target_area.toml"
)


def _vector3(value: npt.ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} sonlu ve shape (3,) olmalı")
    return vector


def _float_tuple(raw: object, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) != size:
        raise ValueError(f"{name} {size} değer içermeli")
    values = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} yalnız sonlu değerler içermeli")
    return values


@dataclass(frozen=True)
class TargetAreaConfig:
    """Hedef görseli ve hover başarı geometrisi için tek kaynak config."""

    name: str
    size_xy_m: tuple[float, float]
    surface_z_m: float
    thickness_m: float
    safety_margin_m: float
    hover_height_m: float
    altitude_tolerance_m: float
    maximum_linear_speed_m_s: float
    maximum_tilt_rad: float
    maximum_angular_rate_rad_s: float
    hold_seconds: float
    rgba: tuple[float, float, float, float]

    @property
    def half_size_xy_m(self) -> tuple[float, float]:
        return self.size_xy_m[0] / 2.0, self.size_xy_m[1] / 2.0

    @property
    def safe_half_size_xy_m(self) -> tuple[float, float]:
        half_x, half_y = self.half_size_xy_m
        return half_x - self.safety_margin_m, half_y - self.safety_margin_m

    def validate(self) -> None:
        if not self.name:
            raise ValueError("target_area.name boş olamaz")
        size = _float_tuple(self.size_xy_m, 2, "target_area.size_xy_m")
        if any(value <= 0.0 for value in size):
            raise ValueError("target_area.size_xy_m pozitif olmalı")
        positive = {
            "surface_z_m": self.surface_z_m,
            "thickness_m": self.thickness_m,
            "hover_height_m": self.hover_height_m,
            "altitude_tolerance_m": self.altitude_tolerance_m,
            "maximum_linear_speed_m_s": self.maximum_linear_speed_m_s,
            "maximum_tilt_rad": self.maximum_tilt_rad,
            "maximum_angular_rate_rad_s": self.maximum_angular_rate_rad_s,
            "hold_seconds": self.hold_seconds,
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in positive.values()):
            raise ValueError("Hedef alan fizik ve hover değerleri pozitif olmalı")
        if not math.isfinite(self.safety_margin_m) or self.safety_margin_m < 0.0:
            raise ValueError("target_area.safety_margin_m negatif olmayan olmalı")
        if any(value <= 0.0 for value in self.safe_half_size_xy_m):
            raise ValueError("Safety margin hedef alanın güvenli iç bölgesini siliyor")
        if self.maximum_tilt_rad >= math.pi / 2.0:
            raise ValueError(
                "target_area.maximum_tilt_rad pi/2 değerinden küçük olmalı"
            )
        rgba = _float_tuple(self.rgba, 4, "target_area.rgba")
        if any(value < 0.0 or value > 1.0 for value in rgba):
            raise ValueError("target_area.rgba değerleri [0, 1] aralığında olmalı")


@dataclass(frozen=True)
class HoverEnvelope:
    """Terminal hover'ın geçmesi gereken eksen hizalı hacim."""

    lower_world_m: FloatArray
    upper_world_m: FloatArray

    def validate(self) -> None:
        lower = _vector3(self.lower_world_m, "hover_envelope.lower_world_m")
        upper = _vector3(self.upper_world_m, "hover_envelope.upper_world_m")
        if np.any(lower >= upper):
            raise ValueError("Hover zarfı alt sınırı üst sınırından küçük olmalı")

    def inflated(self, margin_m: float) -> HoverEnvelope:
        """Zarfı her yönde verilen payla genişlet."""

        if not math.isfinite(margin_m) or margin_m < 0.0:
            raise ValueError("margin_m negatif olmayan ve sonlu olmalı")
        return HoverEnvelope(
            lower_world_m=self.lower_world_m - margin_m,
            upper_world_m=self.upper_world_m + margin_m,
        )

    def intersects_box(
        self, center_world_m: npt.ArrayLike, half_size_m: npt.ArrayLike
    ) -> bool:
        """Verilen eksen hizalı kutu bu zarfla kesişiyor mu?"""

        center = _vector3(center_world_m, "center_world_m")
        half_size = _vector3(half_size_m, "half_size_m")
        if np.any(half_size <= 0.0):
            raise ValueError("half_size_m pozitif olmalı")
        return bool(
            np.all(center - half_size < self.upper_world_m)
            and np.all(center + half_size > self.lower_world_m)
        )


@dataclass(frozen=True)
class TargetArea:
    """Dünya koordinatında hedef merkezi ve yapılandırılmış güvenli bölge."""

    center_xy_world_m: tuple[float, float]
    config: TargetAreaConfig

    def validate(self) -> None:
        _float_tuple(self.center_xy_world_m, 2, "target.center_xy_world_m")
        self.config.validate()

    @property
    def hover_anchor_world_m(self) -> FloatArray:
        return np.array(
            [
                self.center_xy_world_m[0],
                self.center_xy_world_m[1],
                self.config.surface_z_m + self.config.hover_height_m,
            ],
            dtype=np.float64,
        )

    @property
    def corners_world_m(self) -> FloatArray:
        half_x, half_y = self.config.half_size_xy_m
        center_x, center_y = self.center_xy_world_m
        return np.array(
            [
                [center_x - half_x, center_y - half_y, self.config.surface_z_m],
                [center_x + half_x, center_y - half_y, self.config.surface_z_m],
                [center_x + half_x, center_y + half_y, self.config.surface_z_m],
                [center_x - half_x, center_y + half_y, self.config.surface_z_m],
            ],
            dtype=np.float64,
        )

    @property
    def hover_envelope(self) -> HoverEnvelope:
        """Güvenli iç bölge ve irtifa toleransının kapsadığı hover hacmi."""

        safe_half_x, safe_half_y = self.config.safe_half_size_xy_m
        anchor = self.hover_anchor_world_m
        half_size = np.array(
            [safe_half_x, safe_half_y, self.config.altitude_tolerance_m],
            dtype=np.float64,
        )
        envelope = HoverEnvelope(
            lower_world_m=anchor - half_size,
            upper_world_m=anchor + half_size,
        )
        envelope.validate()
        return envelope

    def contains_safe_xy(self, point_world_m: npt.ArrayLike) -> bool:
        point = np.asarray(point_world_m, dtype=np.float64)
        if point.shape not in ((2,), (3,)) or not np.all(np.isfinite(point)):
            raise ValueError("point_world_m sonlu ve shape (2,) veya (3,) olmalı")
        safe_half_x, safe_half_y = self.config.safe_half_size_xy_m
        offset = np.abs(point[:2] - np.asarray(self.center_xy_world_m))
        return bool(offset[0] <= safe_half_x and offset[1] <= safe_half_y)


def believed_target_xy(
    true_goal_world_m: npt.ArrayLike,
    error_radius_m: float,
    seed: int,
    safe_half_size_xy_m: tuple[float, float] | None = None,
) -> FloatArray:
    """ADR-019 yaklaşık hedef konumunu deterministik üret.

    Araç hedefin gerçek konumunu hiçbir zaman öğrenmez; yalnız `error_radius_m`
    yarıçaplı bir diskten örneklenmiş inanılan konumu bilir. Hata episode içinde
    sabittir ve verilen seed'den türetilir, yani aynı sahne aynı hatayı üretir.

    `safe_half_size_xy_m` verilirse yarıçapın güvenli iç bölgeden büyük olması
    doğrulanır: daha küçük bir hata, aracın doğrudan inanılan konuma giderek
    görüntü kullanmadan hover başarısına ulaşmasına izin verir ve değişikliği
    anlamsız kılar.
    """

    goal = _vector3(true_goal_world_m, "true_goal_world_m")
    if not math.isfinite(error_radius_m) or error_radius_m < 0.0:
        raise ValueError("error_radius_m negatif olmayan ve sonlu olmalı")
    if safe_half_size_xy_m is not None and error_radius_m > 0.0:
        safe_half = min(safe_half_size_xy_m)
        if error_radius_m <= safe_half:
            raise ValueError(
                "target_error_radius_m güvenli iç bölgeden büyük olmalı; "
                f"{error_radius_m:.3f} m <= {safe_half:.3f} m ise görüntü "
                "kullanılmadan başarıya ulaşılabilir (ADR-019)"
            )
    if error_radius_m == 0.0:
        return goal[:2].copy()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xA19]))
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    # Disk üzerinde düzgün dağılım için yarıçapın karekökü alınır.
    radius = error_radius_m * math.sqrt(float(rng.uniform(0.0, 1.0)))
    return goal[:2] + np.array(
        [radius * math.cos(angle), radius * math.sin(angle)], dtype=np.float64
    )


def target_area_from_goal(
    goal_world_m: npt.ArrayLike,
    config: TargetAreaConfig | None = None,
) -> TargetArea:
    """Hedef alanı waypoint hedefinden türeten tek kaynaklı factory.

    Hedef merkezinin hedef noktasının XY'sinden geldiği kural yalnız burada
    tanımlıdır; MJCF üretimi, sahne açıklığı, reward ve başarı doğrulaması aynı
    türetmeyi kullanmalıdır. Hedef yüzeyi `surface_z_m` ile zemine sabitlendiği
    için `goal_world_m[2]` hedef alan geometrisine girmez.
    """

    target_config = config or load_target_area_config()
    target_config.validate()
    goal = _vector3(goal_world_m, "goal_world_m")
    target = TargetArea(
        center_xy_world_m=(float(goal[0]), float(goal[1])),
        config=target_config,
    )
    target.validate()
    return target


def target_area_from_scene(
    scene: StaticScene,
    config: TargetAreaConfig | None = None,
) -> TargetArea:
    """Sahne hedefinden hedef alanı türet; `target_area_from_goal`'a delege eder."""

    return target_area_from_goal(scene.goal_world_m, config)


def validate_hover_envelope(
    target_config: TargetAreaConfig,
    scene_config: SceneGeneratorConfig,
) -> None:
    """Hover irtifa bandının araç açıklığıyla erişilebilir olduğunu doğrula.

    Ortam konum referansını `workspace ± vehicle_clearance_m` içine kırptığı
    için, irtifa toleransının üst kenarı bu bandın dışında kalırsa hover hold
    koşulu hiçbir zaman sağlanamaz. Bu sessiz bir başarısızlık olduğundan
    fail-fast doğrulanır.
    """

    target_config.validate()
    scene_config.validate()
    anchor_z = target_config.surface_z_m + target_config.hover_height_m
    tolerance = target_config.altitude_tolerance_m
    clearance = scene_config.vehicle_clearance_m
    reachable_min_z = float(scene_config.workspace_min_m[2]) + clearance
    reachable_max_z = float(scene_config.workspace_max_m[2]) - clearance
    if anchor_z + tolerance > reachable_max_z:
        raise ValueError(
            "Hover irtifa toleransının üst kenarı erişilebilir workspace'i aşıyor: "
            f"{anchor_z + tolerance:.3f} m > {reachable_max_z:.3f} m"
        )
    if anchor_z - tolerance < reachable_min_z:
        raise ValueError(
            "Hover irtifa toleransının alt kenarı erişilebilir workspace altında: "
            f"{anchor_z - tolerance:.3f} m < {reachable_min_z:.3f} m"
        )


def reachable_safe_region_xy(
    target: TargetArea,
    scene_config: SceneGeneratorConfig,
) -> tuple[FloatArray, FloatArray]:
    """Güvenli iç bölgeyi erişilebilir workspace ile kesip sınırlarını ver.

    Hedef merkezi workspace kenarına yakınsa güvenli iç bölgenin bir kısmı araç
    açıklığı nedeniyle erişilemez olabilir. Kesişim boşsa hedef o araç ve
    workspace için geçersizdir.
    """

    target.validate()
    scene_config.validate()
    clearance = scene_config.vehicle_clearance_m
    safe_half = np.asarray(target.config.safe_half_size_xy_m, dtype=np.float64)
    center = np.asarray(target.center_xy_world_m, dtype=np.float64)
    lower = np.maximum(center - safe_half, scene_config.workspace_min_m[:2] + clearance)
    upper = np.minimum(center + safe_half, scene_config.workspace_max_m[:2] - clearance)
    if np.any(lower > upper):
        raise ValueError(
            "Hedefin güvenli iç bölgesi erişilebilir workspace ile kesişmiyor"
        )
    return lower, upper


def load_target_area_config(
    path: str | Path = DEFAULT_TARGET_AREA_CONFIG,
) -> TargetAreaConfig:
    """TOML dosyasından doğrulanmış hedef alan konfigürasyonu yükle."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Hedef alan konfigürasyonu bulunamadı: {config_path}")
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)["target_area"]
    size = _float_tuple(raw["size_xy_m"], 2, "target_area.size_xy_m")
    rgba = _float_tuple(raw["rgba"], 4, "target_area.rgba")
    config = TargetAreaConfig(
        name=str(raw["name"]),
        size_xy_m=(size[0], size[1]),
        surface_z_m=float(raw["surface_z_m"]),
        thickness_m=float(raw["thickness_m"]),
        safety_margin_m=float(raw["safety_margin_m"]),
        hover_height_m=float(raw["hover_height_m"]),
        altitude_tolerance_m=float(raw["altitude_tolerance_m"]),
        maximum_linear_speed_m_s=float(raw["maximum_linear_speed_m_s"]),
        maximum_tilt_rad=math.radians(float(raw["maximum_tilt_deg"])),
        maximum_angular_rate_rad_s=float(raw["maximum_angular_rate_rad_s"]),
        hold_seconds=float(raw["hold_seconds"]),
        rgba=(rgba[0], rgba[1], rgba[2], rgba[3]),
    )
    config.validate()
    return config
