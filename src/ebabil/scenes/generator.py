"""Curriculum seviyeli, seed tabanlı ve çözülebilir statik engel sahneleri."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import mujoco
import numpy as np
import numpy.typing as npt

from ebabil.models import DEFAULT_QUADROTOR_XML
from ebabil.scenes.target import (
    HoverEnvelope,
    TargetAreaConfig,
    load_target_area_config,
    target_area_from_goal,
    target_area_from_scene,
    validate_hover_envelope,
)

if TYPE_CHECKING:
    from ebabil.perception.config import CameraConfig

FloatArray = npt.NDArray[np.float64]
SceneMode = Literal["clear", "single_blocker", "random_obstacles"]
EndpointMode = Literal["fixed", "random_start", "random_goal", "random_both"]
ObstacleGeometry = Literal["box", "cylinder", "capsule", "sphere", "ellipsoid"]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_CONFIG = PACKAGE_ROOT / "configs" / "environment" / "static_scene.toml"


def _vector3(value: npt.ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} sonlu ve shape (3,) olmalı")
    return vector


@dataclass(frozen=True)
class Obstacle:
    name: str
    center_world_m: FloatArray
    half_size_m: FloatArray
    geometry: ObstacleGeometry = "box"

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Engel adı boş olamaz")
        _vector3(self.center_world_m, f"{self.name}.center_world_m")
        half_size = _vector3(self.half_size_m, f"{self.name}.half_size_m")
        if np.any(half_size <= 0.0):
            raise ValueError("Engel yarı boyutları pozitif olmalı")
        if self.geometry not in (
            "box",
            "cylinder",
            "capsule",
            "sphere",
            "ellipsoid",
        ):
            raise ValueError(f"Desteklenmeyen engel geometrisi: {self.geometry}")


@dataclass(frozen=True)
class VisualDistractor:
    """Privileged maskelere girmeyen, çarpışmasız görsel yanıltıcı nesne."""

    name: str
    center_world_m: FloatArray
    half_size_m: FloatArray
    geometry: ObstacleGeometry
    appearance: Literal["obstacle", "target"]

    def validate(self) -> None:
        if not self.name.startswith("distractor_"):
            raise ValueError("Distractor adı distractor_ ile başlamalı")
        _vector3(self.center_world_m, f"{self.name}.center_world_m")
        half_size = _vector3(self.half_size_m, f"{self.name}.half_size_m")
        if np.any(half_size <= 0.0):
            raise ValueError("Distractor yarı boyutları pozitif olmalı")
        if self.geometry not in (
            "box",
            "cylinder",
            "capsule",
            "sphere",
            "ellipsoid",
        ):
            raise ValueError(f"Desteklenmeyen distractor geometrisi: {self.geometry}")
        if self.appearance not in ("obstacle", "target"):
            raise ValueError("Distractor appearance obstacle veya target olmalı")


#: `single_blocker` engelinin rota üzerindeki konumu. `0.5` tam orta nokta,
#: `1.0` başlangıç noktasıdır. ADR-008 legacy sahnesi ve bütün D1 koşuları bu
#: sabit değerle üretildi; rastgeleleştirme config'ten opt-in'dir (REV-031).
LEGACY_BLOCKER_PATH_FRACTION = 0.5

#: MJCF'teki zemin düzleminin yarı-boyutu. ADR-008 legacy sahnesi ve onaylı
#: encoder zinciri bu değerle üretildiği için varsayılan olarak korunur;
#: büyütülmüş workspace'ler config'te `ground_half_size_m` verir (REV-030).
LEGACY_GROUND_HALF_SIZE_M = 5.0


@dataclass(frozen=True)
class StaticScene:
    seed: int
    start_world_m: FloatArray
    goal_world_m: FloatArray
    obstacles: tuple[Obstacle, ...]
    mode: str = "custom"

    def validate(self) -> None:
        _vector3(self.start_world_m, "start_world_m")
        _vector3(self.goal_world_m, "goal_world_m")
        if np.allclose(self.start_world_m, self.goal_world_m):
            raise ValueError("Başlangıç ve hedef farklı olmalı")
        names: set[str] = set()
        for obstacle in self.obstacles:
            obstacle.validate()
            if obstacle.name in names:
                raise ValueError(f"Tekrarlanan engel adı: {obstacle.name}")
            names.add(obstacle.name)


@dataclass(frozen=True)
class SceneGeneratorConfig:
    workspace_min_m: FloatArray
    workspace_max_m: FloatArray
    start_world_m: FloatArray
    goal_world_m: FloatArray
    random_endpoint_min_m: FloatArray
    random_endpoint_max_m: FloatArray
    minimum_endpoint_separation_m: float
    maximum_endpoint_separation_m: float
    obstacle_count: int
    obstacle_half_size_min_m: FloatArray
    obstacle_half_size_max_m: FloatArray
    endpoint_clearance_m: float
    obstacle_clearance_m: float
    vehicle_clearance_m: float
    path_grid_resolution_m: float
    max_sampling_attempts: int
    max_scene_attempts: int
    ground_half_size_m: float = LEGACY_GROUND_HALF_SIZE_M
    blocker_path_fraction_min: float = LEGACY_BLOCKER_PATH_FRACTION
    blocker_path_fraction_max: float = LEGACY_BLOCKER_PATH_FRACTION
    #: ADR-022 §6: RL sahnesinde engel geometrisi bu kümeden örneklenir
    #: (`randomize_obstacle_geometries`). Legacy ve `v0` ortamları yalnız
    #: kutu kullanır; alan varsayılan olarak `("box",)`'tur.
    obstacle_geometries: tuple[ObstacleGeometry, ...] = ("box",)

    def validate(self) -> None:
        supported_geometries = {"box", "cylinder", "capsule", "sphere", "ellipsoid"}
        if not self.obstacle_geometries or not set(self.obstacle_geometries).issubset(
            supported_geometries
        ):
            raise ValueError("obstacle_geometries boş veya geçersiz geometri içeriyor")
        if len(set(self.obstacle_geometries)) != len(self.obstacle_geometries):
            raise ValueError("obstacle_geometries yinelenen geometri içeriyor")
        workspace_min = _vector3(self.workspace_min_m, "workspace_min_m")
        workspace_max = _vector3(self.workspace_max_m, "workspace_max_m")
        start = _vector3(self.start_world_m, "start_world_m")
        goal = _vector3(self.goal_world_m, "goal_world_m")
        endpoint_min = _vector3(self.random_endpoint_min_m, "random_endpoint_min_m")
        endpoint_max = _vector3(self.random_endpoint_max_m, "random_endpoint_max_m")
        size_min = _vector3(self.obstacle_half_size_min_m, "obstacle_half_size_min_m")
        size_max = _vector3(self.obstacle_half_size_max_m, "obstacle_half_size_max_m")
        if np.any(workspace_min >= workspace_max):
            raise ValueError("Workspace min değerleri max değerlerinden küçük olmalı")
        if np.any(start < workspace_min) or np.any(start > workspace_max):
            raise ValueError("Başlangıç workspace dışında")
        if np.any(goal < workspace_min) or np.any(goal > workspace_max):
            raise ValueError("Hedef workspace dışında")
        if np.any(endpoint_min >= endpoint_max):
            raise ValueError("Rastgele endpoint min değerleri max'tan küçük olmalı")
        safe_min = workspace_min + self.vehicle_clearance_m
        safe_max = workspace_max - self.vehicle_clearance_m
        if np.any(endpoint_min < safe_min) or np.any(endpoint_max > safe_max):
            raise ValueError("Rastgele endpoint sınırları güvenli workspace dışında")
        diagonal = float(np.linalg.norm(endpoint_max - endpoint_min))
        if (
            not math.isfinite(self.minimum_endpoint_separation_m)
            or self.minimum_endpoint_separation_m <= 0.0
            or self.minimum_endpoint_separation_m > diagonal
        ):
            raise ValueError("minimum_endpoint_separation_m geçersiz")
        # `inf` üst sınır yok demektir ve legacy davranıştır; ADR-018 mesafe
        # seviyeleri bunu sonlu bir değerle sınırlar.
        if (
            math.isnan(self.maximum_endpoint_separation_m)
            or self.maximum_endpoint_separation_m < self.minimum_endpoint_separation_m
        ):
            raise ValueError(
                "maximum_endpoint_separation_m minimum değerden küçük olamaz"
            )
        if np.any(size_min <= 0.0) or np.any(size_min > size_max):
            raise ValueError("Engel boyut aralığı geçersiz")
        if (
            self.obstacle_count < 1
            or self.max_sampling_attempts <= 0
            or self.max_scene_attempts <= 0
        ):
            raise ValueError("Engel sayısı ve deneme sınırları pozitif olmalı")
        if not (
            math.isfinite(self.blocker_path_fraction_min)
            and math.isfinite(self.blocker_path_fraction_max)
        ):
            raise ValueError("blocker_path_fraction sınırları sonlu olmalı")
        if not (
            LEGACY_BLOCKER_PATH_FRACTION
            <= self.blocker_path_fraction_min
            <= self.blocker_path_fraction_max
            <= BLOCKER_MAXIMUM_PATH_FRACTION
        ):
            raise ValueError(
                "blocker_path_fraction aralığı "
                f"[{LEGACY_BLOCKER_PATH_FRACTION}, {BLOCKER_MAXIMUM_PATH_FRACTION}] "
                "içinde ve artan olmalı; alt sınır 0.5'in altına inemez çünkü "
                "0.5 hedefe en yakın, hattı hâlâ kapatan konumdur"
            )
        if self.ground_half_size_m <= 0.0 or not math.isfinite(self.ground_half_size_m):
            raise ValueError("ground_half_size_m pozitif ve sonlu olmalı")
        clearances = (
            self.endpoint_clearance_m,
            self.obstacle_clearance_m,
            self.vehicle_clearance_m,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in clearances):
            raise ValueError("Açıklık değerleri negatif olmayan ve sonlu olmalı")
        if (
            not math.isfinite(self.path_grid_resolution_m)
            or self.path_grid_resolution_m <= 0.0
        ):
            raise ValueError("path_grid_resolution_m pozitif ve sonlu olmalı")


def load_scene_config(
    path: str | Path = DEFAULT_SCENE_CONFIG,
) -> SceneGeneratorConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Sahne konfigürasyonu bulunamadı: {config_path}")
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)["scene"]
    config = SceneGeneratorConfig(
        workspace_min_m=_vector3(raw["workspace_min_m"], "workspace_min_m"),
        workspace_max_m=_vector3(raw["workspace_max_m"], "workspace_max_m"),
        start_world_m=_vector3(raw["start_world_m"], "start_world_m"),
        goal_world_m=_vector3(raw["goal_world_m"], "goal_world_m"),
        random_endpoint_min_m=_vector3(
            raw["random_endpoint_min_m"], "random_endpoint_min_m"
        ),
        random_endpoint_max_m=_vector3(
            raw["random_endpoint_max_m"], "random_endpoint_max_m"
        ),
        minimum_endpoint_separation_m=float(raw["minimum_endpoint_separation_m"]),
        maximum_endpoint_separation_m=float(
            raw.get("maximum_endpoint_separation_m", math.inf)
        ),
        obstacle_count=int(raw["obstacle_count"]),
        obstacle_half_size_min_m=_vector3(
            raw["obstacle_half_size_min_m"], "obstacle_half_size_min_m"
        ),
        obstacle_half_size_max_m=_vector3(
            raw["obstacle_half_size_max_m"], "obstacle_half_size_max_m"
        ),
        endpoint_clearance_m=float(raw["endpoint_clearance_m"]),
        obstacle_clearance_m=float(raw["obstacle_clearance_m"]),
        vehicle_clearance_m=float(raw["vehicle_clearance_m"]),
        path_grid_resolution_m=float(raw["path_grid_resolution_m"]),
        max_sampling_attempts=int(raw["max_sampling_attempts"]),
        max_scene_attempts=int(raw["max_scene_attempts"]),
        ground_half_size_m=float(
            raw.get("ground_half_size_m", LEGACY_GROUND_HALF_SIZE_M)
        ),
        blocker_path_fraction_min=float(
            raw.get("blocker_path_fraction_min", LEGACY_BLOCKER_PATH_FRACTION)
        ),
        blocker_path_fraction_max=float(
            raw.get("blocker_path_fraction_max", LEGACY_BLOCKER_PATH_FRACTION)
        ),
        obstacle_geometries=tuple(
            cast(ObstacleGeometry, str(value))
            for value in raw.get("obstacle_geometries", ["box"])
        ),
    )
    config.validate()
    return config


def randomize_obstacle_geometries(
    scene: StaticScene,
    geometries: tuple[ObstacleGeometry, ...],
    seed: int,
) -> StaticScene:
    """Engel geometrilerini scene seed'inden deterministik olarak yeniden örnekle.

    ADR-022 §6: kodlayıcı dataset'i kutu/silindir/capsule geometrileriyle
    eğitildi; RL sahnesi yalnız kutu gösterirse politika kodlayıcının
    dağılımının tek bir dilimini görür. Yalnız `geometry` alanı değişir;
    merkez ve yarı-boyutlar korunduğu için AABB tabanlı rota doğrulaması
    muhafazakâr kalır (bütün primitive'ler kendi AABB'lerinin içindedir).
    """

    if not geometries:
        raise ValueError("geometries boş olamaz")
    if seed < 0:
        raise ValueError("seed negatif olamaz")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x6E0]))
    obstacles = tuple(
        replace(obstacle, geometry=geometries[int(rng.integers(len(geometries)))])
        for obstacle in scene.obstacles
    )
    return replace(scene, obstacles=obstacles)


def required_ground_half_size_m(
    config: SceneGeneratorConfig, camera_config: CameraConfig | None = None
) -> float:
    """Kameranın workspace'in her yerinde zemini görebilmesi için gereken yarı-boyut.

    REV-030: MJCF'teki zemin düzlemi `size="5 5 0.1"`, yani görsel olarak
    `10 x 10 m`'dir. MuJoCo düzlemleri **çarpışmada sonsuz** olduğu için fizik
    etkilenmez, fakat **render** yalnız bu dikdörtgende yapılır. Dışında kamera
    boşluk görür. D1 workspace'i `20 x 20 m` olduğundan hedeflerin `%65`'i
    zeminin dışında kalıyordu.

    Gereken yarı-boyut, workspace köşesinden kameranın zemin izinin uzak
    sınırının (`3.71 * irtifa`, ADR-016) tavan irtifasında ulaştığı yerdir.
    """

    from ebabil.perception.config import load_camera_config

    camera_config = camera_config or load_camera_config()
    camera_config.validate()
    config.validate()
    half = float(
        np.max(
            np.maximum(
                np.abs(np.asarray(config.workspace_min_m[:2], dtype=np.float64)),
                np.abs(np.asarray(config.workspace_max_m[:2], dtype=np.float64)),
            )
        )
    )
    ceiling_m = float(config.workspace_max_m[2])
    _, far_m = camera_config.ground_swath_m(ceiling_m)
    return half + far_m


def _point_to_box_distance(point: FloatArray, obstacle: Obstacle) -> float:
    delta = np.maximum(
        np.abs(point - obstacle.center_world_m) - obstacle.half_size_m, 0.0
    )
    return float(np.linalg.norm(delta))


def _box_distance_xy(first: Obstacle, second: Obstacle) -> float:
    delta = np.maximum(
        np.abs(first.center_world_m[:2] - second.center_world_m[:2])
        - first.half_size_m[:2]
        - second.half_size_m[:2],
        0.0,
    )
    return float(np.linalg.norm(delta))


def _segment_intersects_aabb(
    start: FloatArray,
    goal: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> bool:
    """Sonlu bir segmentin kapalı axis-aligned kutuyla kesişimini slab yöntemiyle bul."""

    direction = goal - start
    minimum_parameter = 0.0
    maximum_parameter = 1.0
    epsilon = 1e-15
    for axis in range(start.size):
        if abs(float(direction[axis])) <= epsilon:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return False
            continue
        first = float((lower[axis] - start[axis]) / direction[axis])
        second = float((upper[axis] - start[axis]) / direction[axis])
        entry, exit_ = sorted((first, second))
        minimum_parameter = max(minimum_parameter, entry)
        maximum_parameter = min(maximum_parameter, exit_)
        if minimum_parameter > maximum_parameter:
            return False
    return True


def _segment_intersects_aabb_batch(
    start: FloatArray,
    goal: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> npt.NDArray[np.bool_]:
    """Tek segmentin `N` axis-aligned kutuyla kesişimini vektörize hesapla.

    `_segment_intersects_aabb` ile birebir aynı slab yöntemini uygular;
    tek fark Python döngüsü yerine numpy kullanmasıdır. `lower` ve `upper`
    `(N, eksen)` şeklindedir. Occupancy araması hücre başına bütün engelleri
    denetlediği için bu yol sıcak noktadır.
    """

    direction = goal - start
    count = lower.shape[0]
    minimum_parameter = np.zeros(count, dtype=np.float64)
    maximum_parameter = np.ones(count, dtype=np.float64)
    intersects = np.ones(count, dtype=np.bool_)
    epsilon = 1e-15
    for axis in range(start.size):
        component = float(direction[axis])
        if abs(component) <= epsilon:
            intersects &= (start[axis] >= lower[:, axis]) & (
                start[axis] <= upper[:, axis]
            )
            continue
        first = (lower[:, axis] - start[axis]) / component
        second = (upper[:, axis] - start[axis]) / component
        np.maximum(minimum_parameter, np.minimum(first, second), out=minimum_parameter)
        np.minimum(maximum_parameter, np.maximum(first, second), out=maximum_parameter)
        intersects &= minimum_parameter <= maximum_parameter
    return intersects


def _inflated_obstacle_bounds_xy(
    scene: StaticScene, clearance_m: float
) -> tuple[FloatArray, FloatArray]:
    """Engellerin araç açıklığıyla genişletilmiş XY sınırlarını bir kez üret."""

    if not scene.obstacles:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty
    centers = np.asarray(
        [obstacle.center_world_m[:2] for obstacle in scene.obstacles],
        dtype=np.float64,
    )
    half_sizes = np.asarray(
        [obstacle.half_size_m[:2] for obstacle in scene.obstacles],
        dtype=np.float64,
    )
    return centers - half_sizes - clearance_m, centers + half_sizes + clearance_m


def direct_path_is_blocked(scene: StaticScene, clearance_m: float = 0.0) -> bool:
    """Başlangıç-hedef 3B segmentinin genişletilmiş bir engelle kesişimini bildir."""

    scene.validate()
    if not math.isfinite(clearance_m) or clearance_m < 0.0:
        raise ValueError("clearance_m negatif olmayan ve sonlu olmalı")
    return any(
        _segment_intersects_aabb(
            scene.start_world_m,
            scene.goal_world_m,
            obstacle.center_world_m - obstacle.half_size_m - clearance_m,
            obstacle.center_world_m + obstacle.half_size_m + clearance_m,
        )
        for obstacle in scene.obstacles
    )


def has_navigable_xy_path(
    scene: StaticScene,
    config: SceneGeneratorConfig,
) -> bool:
    """Aracı genişleterek 8-komşulu occupancy grid üzerinde rota ara."""

    resolution = config.path_grid_resolution_m
    minimum = config.workspace_min_m[:2] + config.vehicle_clearance_m
    maximum = config.workspace_max_m[:2] - config.vehicle_clearance_m
    if (
        np.any(scene.start_world_m[:2] < minimum)
        or np.any(scene.start_world_m[:2] > maximum)
        or np.any(scene.goal_world_m[:2] < minimum)
        or np.any(scene.goal_world_m[:2] > maximum)
    ):
        return False
    shape = np.floor((maximum - minimum) / resolution).astype(int) + 1
    if np.any(shape < 2):
        return False

    def to_index(point: FloatArray) -> tuple[int, int]:
        index = np.rint((point[:2] - minimum) / resolution).astype(int)
        index = np.clip(index, 0, shape - 1)
        return int(index[0]), int(index[1])

    def grid_point(index: tuple[int, int]) -> FloatArray:
        return minimum + np.asarray(index, dtype=np.float64) * resolution

    obstacle_lower, obstacle_upper = _inflated_obstacle_bounds_xy(
        scene, config.vehicle_clearance_m
    )

    # Hücre doluluğu, genişletilmiş kutulara noktasal kapsanma testidir. Bütün
    # hücreler için tek seferde hesaplanır; hücre başına numpy çağrı yükü
    # büyük workspace'lerde aramanın baskın maliyetidir.
    occupancy = np.zeros((int(shape[0]), int(shape[1])), dtype=np.bool_)
    if obstacle_lower.shape[0] > 0:
        axis_x = minimum[0] + np.arange(int(shape[0]), dtype=np.float64) * resolution
        axis_y = minimum[1] + np.arange(int(shape[1]), dtype=np.float64) * resolution
        inside_x = (axis_x[:, None] >= obstacle_lower[None, :, 0]) & (
            axis_x[:, None] <= obstacle_upper[None, :, 0]
        )
        inside_y = (axis_y[:, None] >= obstacle_lower[None, :, 1]) & (
            axis_y[:, None] <= obstacle_upper[None, :, 1]
        )
        occupancy = np.einsum("xn,yn->xy", inside_x, inside_y).astype(np.bool_)

    def is_occupied(index: tuple[int, int]) -> bool:
        return bool(occupancy[index[0], index[1]])

    def segment_is_free(start: FloatArray, goal: FloatArray) -> bool:
        if obstacle_lower.shape[0] == 0:
            return True
        return not bool(
            _segment_intersects_aabb_batch(
                start, goal, obstacle_lower, obstacle_upper
            ).any()
        )

    start_index = to_index(scene.start_world_m)
    goal_index = to_index(scene.goal_world_m)
    if (
        is_occupied(start_index)
        or is_occupied(goal_index)
        or not segment_is_free(scene.start_world_m[:2], grid_point(start_index))
        or not segment_is_free(scene.goal_world_m[:2], grid_point(goal_index))
    ):
        return False
    frontier = [start_index]
    visited = {start_index}
    neighbor_offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    while frontier:
        current = frontier.pop()
        if current == goal_index:
            return True
        for offset_x, offset_y in neighbor_offsets:
            neighbor = (current[0] + offset_x, current[1] + offset_y)
            if not (0 <= neighbor[0] < shape[0] and 0 <= neighbor[1] < shape[1]):
                continue
            if neighbor in visited or is_occupied(neighbor):
                continue
            if not segment_is_free(grid_point(current), grid_point(neighbor)):
                continue
            if offset_x != 0 and offset_y != 0:
                horizontal = (current[0] + offset_x, current[1])
                vertical = (current[0], current[1] + offset_y)
                if is_occupied(horizontal) or is_occupied(vertical):
                    continue
            visited.add(neighbor)
            frontier.append(neighbor)
    return False


BLOCKER_MAXIMUM_PATH_FRACTION = 0.85
HOVER_ENVELOPE_SEPARATION_EPSILON_M = 1e-6


def _blocker_path_fraction(
    half_size_m: FloatArray,
    start_world_m: FloatArray,
    goal_world_m: FloatArray,
    hover_envelope: HoverEnvelope | None,
    blocker_top_z: float,
) -> float | None:
    """Blocker'ı hover hacminden çıkaran en küçük hat kesirini ver.

    Hedef merkezi hover zarfının XY merkezi olduğu için, hat üzerinde hedeften
    uzaklaşmak zarfı boşaltmanın en kısa yoludur. `0.5` orta nokta, `1.0`
    başlangıç noktasıdır. Zarf tanımsızsa veya blocker zarfın altında kalıyorsa
    kaydırma gerekmez. Gerekli kesir üst sınırı aşıyorsa `None` döner.
    """

    if hover_envelope is None:
        return 0.5
    if blocker_top_z <= float(hover_envelope.lower_world_m[2]):
        return 0.5
    envelope_half_xy = (
        hover_envelope.upper_world_m[:2] - hover_envelope.lower_world_m[:2]
    ) / 2.0
    direction_xy = (start_world_m - goal_world_m)[:2]
    required_fractions = []
    for axis in range(2):
        span = abs(float(direction_xy[axis]))
        if span <= 1e-9:
            continue
        # Tam teğet yerleşimde kesişim kararı yuvarlamaya kalır; kesin ayrım
        # için sabit bir metre payı eklenir.
        required_fractions.append(
            (
                float(envelope_half_xy[axis])
                + float(half_size_m[axis])
                + HOVER_ENVELOPE_SEPARATION_EPSILON_M
            )
            / span
        )
    if not required_fractions:
        return None
    fraction = max(0.5, min(required_fractions))
    if fraction > BLOCKER_MAXIMUM_PATH_FRACTION:
        return None
    return fraction


def _blocking_obstacle(
    rng: np.random.Generator,
    config: SceneGeneratorConfig,
    start_world_m: FloatArray,
    goal_world_m: FloatArray,
    hover_envelope: HoverEnvelope | None = None,
) -> Obstacle | None:
    """Doğrudan rotayı kapatan zorunlu engeli üret.

    `hover_envelope` verildiğinde engel hedefin hover hacmini boşaltacak kadar
    başlangıca doğru kaydırılır. Gerekli kaydırma hattın dışına çıkacaksa `None`
    döner ve çağıran yeni bir deneme yapar. Kaydırma sonrası doğrudan rota
    koşulu yine `direct_path_is_blocked` ile doğrulanır.
    """

    half_size = rng.uniform(
        config.obstacle_half_size_min_m,
        config.obstacle_half_size_max_m,
    )
    required_half_height = (
        max(start_world_m[2], goal_world_m[2])
        + config.vehicle_clearance_m
        - config.workspace_min_m[2]
    ) / 2.0
    if required_half_height > config.obstacle_half_size_max_m[2]:
        raise ValueError(
            "Engel yükseklik sınırı başlangıç-hedef hattını araç açıklığıyla "
            "kapatmaya yetmiyor"
        )
    half_size[2] = max(half_size[2], required_half_height)
    center_z = config.workspace_min_m[2] + half_size[2]
    required_fraction = _blocker_path_fraction(
        half_size,
        start_world_m,
        goal_world_m,
        hover_envelope,
        blocker_top_z=center_z + half_size[2],
    )
    if required_fraction is None:
        return None
    fraction = required_fraction
    if config.blocker_path_fraction_max > config.blocker_path_fraction_min:
        # REV-031: yalnız rastgeleleştirme açıkken rng'den çekilir; kapalıyken
        # akış ilerlemez ve legacy sahneler bit-bit korunur.
        sampled = float(
            rng.uniform(
                config.blocker_path_fraction_min,
                config.blocker_path_fraction_max,
            )
        )
        fraction = max(required_fraction, sampled)
        if fraction > BLOCKER_MAXIMUM_PATH_FRACTION:
            return None
    if fraction == LEGACY_BLOCKER_PATH_FRACTION:
        # Legacy ifadeyi bit-bit koru; kaydırma gerekmeyen yol değişmemelidir.
        center = (start_world_m + goal_world_m) / 2.0
    else:
        center = goal_world_m + fraction * (start_world_m - goal_world_m)
    center = center.copy()
    center[2] = center_z
    return Obstacle("obstacle_000", center, half_size)


def _sample_random_obstacle(
    rng: np.random.Generator,
    config: SceneGeneratorConfig,
    index: int,
) -> Obstacle:
    half_size = rng.uniform(
        config.obstacle_half_size_min_m,
        config.obstacle_half_size_max_m,
    )
    center = np.empty(3, dtype=np.float64)
    center[:2] = rng.uniform(
        config.workspace_min_m[:2] + half_size[:2],
        config.workspace_max_m[:2] - half_size[:2],
    )
    center[2] = config.workspace_min_m[2] + half_size[2]
    return Obstacle(f"obstacle_{index:03d}", center, half_size)


def _sample_endpoints(
    rng: np.random.Generator,
    config: SceneGeneratorConfig,
    endpoint_mode: EndpointMode,
) -> tuple[FloatArray, FloatArray]:
    randomize_start = endpoint_mode in ("random_start", "random_both")
    randomize_goal = endpoint_mode in ("random_goal", "random_both")
    if not randomize_start and not randomize_goal:
        return config.start_world_m.copy(), config.goal_world_m.copy()
    for _ in range(config.max_sampling_attempts):
        start = (
            rng.uniform(config.random_endpoint_min_m, config.random_endpoint_max_m)
            if randomize_start
            else config.start_world_m.copy()
        )
        goal = (
            rng.uniform(config.random_endpoint_min_m, config.random_endpoint_max_m)
            if randomize_goal
            else config.goal_world_m.copy()
        )
        separation = float(np.linalg.norm(goal - start))
        if (
            config.minimum_endpoint_separation_m
            <= separation
            <= config.maximum_endpoint_separation_m
        ):
            return start, goal
    raise RuntimeError(
        f"{endpoint_mode} endpoint'leri {config.max_sampling_attempts} "
        f"denemede {config.minimum_endpoint_separation_m:.2f}-"
        f"{config.maximum_endpoint_separation_m:.2f} m bandında üretilemedi"
    )


def _hover_envelope_for_goal(
    goal_world_m: FloatArray,
    config: SceneGeneratorConfig,
    target_config: TargetAreaConfig,
) -> HoverEnvelope:
    """Hedef hover hacmini araç açıklığıyla genişletilmiş biçimde ver."""

    target = target_area_from_goal(goal_world_m, target_config)
    return target.hover_envelope.inflated(config.vehicle_clearance_m)


def generate_static_scene(
    seed: int,
    config: SceneGeneratorConfig | None = None,
    mode: SceneMode = "random_obstacles",
    endpoint_mode: EndpointMode = "fixed",
    *,
    target_config: TargetAreaConfig | None = None,
) -> StaticScene:
    """Seed tabanlı çözülebilir statik sahne üret.

    `target_config` verildiğinde engeller ADR-012 hedef hover hacmini araç
    açıklığı payıyla boş bırakmak zorundadır. Varsayılan `None`, legacy
    `EbabilNavigation-v0` sahnelerini bit-bit korumak için bu kısıtı uygulamaz.
    """

    config = config or load_scene_config()
    config.validate()
    if mode not in ("clear", "single_blocker", "random_obstacles"):
        raise ValueError(f"Bilinmeyen sahne modu: {mode}")
    if endpoint_mode not in ("fixed", "random_start", "random_goal", "random_both"):
        raise ValueError(f"Bilinmeyen endpoint modu: {endpoint_mode}")
    if target_config is not None:
        target_config.validate()
        validate_hover_envelope(target_config, config)
    rng = np.random.default_rng(seed)
    start_world_m, goal_world_m = _sample_endpoints(rng, config, endpoint_mode)
    target_count = {
        "clear": 0,
        "single_blocker": 1,
        "random_obstacles": config.obstacle_count,
    }[mode]
    hover_envelope: HoverEnvelope | None = None
    if target_config is not None:
        hover_envelope = _hover_envelope_for_goal(goal_world_m, config, target_config)
    blocker_rejections = 0

    for _ in range(config.max_scene_attempts):
        obstacles: list[Obstacle] = []
        if target_count:
            blocker = _blocking_obstacle(
                rng, config, start_world_m, goal_world_m, hover_envelope
            )
            if blocker is None:
                blocker_rejections += 1
                continue
            if (
                hover_envelope is not None
                and _point_to_box_distance(start_world_m, blocker)
                < config.vehicle_clearance_m
            ):
                # Hover hacmini boşaltmak için kaydırılan blocker başlangıcı
                # yutmamalıdır; legacy orta nokta yerleşiminde bu risk yoktur.
                blocker_rejections += 1
                continue
            obstacles.append(blocker)
        sampling_attempts = 0
        while len(obstacles) < target_count:
            sampling_attempts += 1
            if sampling_attempts > config.max_sampling_attempts:
                break
            candidate = _sample_random_obstacle(rng, config, len(obstacles))
            if (
                _point_to_box_distance(start_world_m, candidate)
                < config.endpoint_clearance_m
                or _point_to_box_distance(goal_world_m, candidate)
                < config.endpoint_clearance_m
            ):
                continue
            if hover_envelope is not None and hover_envelope.intersects_box(
                candidate.center_world_m, candidate.half_size_m
            ):
                continue
            if any(
                _box_distance_xy(candidate, existing) < config.obstacle_clearance_m
                for existing in obstacles
            ):
                continue
            obstacles.append(candidate)
        if len(obstacles) != target_count:
            continue
        scene = StaticScene(
            seed=seed,
            start_world_m=start_world_m.copy(),
            goal_world_m=goal_world_m.copy(),
            obstacles=tuple(obstacles),
            mode=mode,
        )
        scene.validate()
        direct_path_condition = not target_count or direct_path_is_blocked(
            scene, config.vehicle_clearance_m
        )
        if direct_path_condition and has_navigable_xy_path(scene, config):
            return scene
    if hover_envelope is not None and target_count:
        separation_m = float(np.linalg.norm(goal_world_m - start_world_m))
        raise RuntimeError(
            f"{mode} sahnesi {config.max_scene_attempts} denemede hedef-farkında "
            f"üretilemedi: zorunlu blocker denemelerinin {blocker_rejections} "
            f"tanesi hedefin hover hacmini boşaltamadı veya başlangıcı yuttu "
            f"(endpoint ayrımı {separation_m:.2f} m). Endpoint ayrımını veya "
            "hedef hover parametrelerini gözden geçirin"
        )
    raise RuntimeError(
        f"{mode} sahnesi {config.max_scene_attempts} denemede çözülebilir üretilemedi"
    )


def build_scene_model(
    scene: StaticScene,
    *,
    camera_config: CameraConfig | None = None,
    target_config: TargetAreaConfig | None = None,
    include_legacy_goal_marker: bool = True,
    visual_distractors: tuple[VisualDistractor, ...] = (),
    ground_half_size_m: float | None = None,
) -> mujoco.MjModel:
    """Temel quadrotora engel, IMX477 kamera ve hedef alanı ekle.

    `ground_half_size_m` verilmezse MJCF'teki değer (`5.0 m`) korunur; ADR-008
    legacy sahnesi ve onaylı encoder zinciri bu değerle üretilmiştir ve
    bit-bit korunmak zorundadır. Büyütülmüş workspace'ler için değeri
    `SceneGeneratorConfig.ground_half_size_m` taşır (REV-030): MuJoCo düzlemi
    çarpışmada sonsuz fakat **render**'da bu dikdörtgenle sınırlıdır, yani
    dışında kamera boşluk görür.
    """

    from ebabil.perception.config import load_camera_config

    scene.validate()
    camera_config = camera_config or load_camera_config()
    target_config = target_config or load_target_area_config()
    camera_config.validate()
    target_config.validate()
    if ground_half_size_m is not None and (
        not math.isfinite(ground_half_size_m) or ground_half_size_m <= 0.0
    ):
        raise ValueError("ground_half_size_m pozitif ve sonlu olmalı")
    spec = mujoco.MjSpec.from_file(str(DEFAULT_QUADROTOR_XML))
    if ground_half_size_m is not None:
        ground = spec.geom("ground")
        ground.size = [
            float(ground_half_size_m),
            float(ground_half_size_m),
            float(ground.size[2]),
        ]
    quadrotor = spec.body("quadrotor")
    quadrotor.pos = scene.start_world_m
    quadrotor.add_camera(
        name=camera_config.name,
        pos=camera_config.position_body_m,
        xyaxes=camera_config.xyaxes_body,
        fovy=math.degrees(camera_config.vertical_fov_rad),
    )

    def add_geometry(
        *,
        name: str,
        center_world_m: FloatArray,
        half_size_m: FloatArray,
        geometry: ObstacleGeometry,
        rgba: tuple[float, float, float, float],
        colliding: bool,
    ) -> None:
        body = spec.worldbody.add_body(name=f"{name}_body", pos=center_world_m)
        if geometry == "box":
            geom_type = mujoco.mjtGeom.mjGEOM_BOX
            geom_size = tuple(float(value) for value in half_size_m)
        elif geometry == "cylinder":
            geom_type = mujoco.mjtGeom.mjGEOM_CYLINDER
            geom_size = (
                float(min(half_size_m[0], half_size_m[1])),
                float(half_size_m[2]),
                0.0,
            )
        elif geometry == "capsule":
            geom_type = mujoco.mjtGeom.mjGEOM_CAPSULE
            radius = float(min(half_size_m[0], half_size_m[1], half_size_m[2] / 2.0))
            geom_size = (radius, float(half_size_m[2] - radius), 0.0)
        elif geometry == "sphere":
            geom_type = mujoco.mjtGeom.mjGEOM_SPHERE
            geom_size = (float(np.min(half_size_m)), 0.0, 0.0)
        else:
            geom_type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
            geom_size = tuple(float(value) for value in half_size_m)
        body.add_geom(
            name=name,
            type=geom_type,
            size=geom_size,
            rgba=rgba,
            group=1,
            contype=1 if colliding else 0,
            conaffinity=1 if colliding else 0,
        )

    for obstacle in scene.obstacles:
        add_geometry(
            name=obstacle.name,
            center_world_m=obstacle.center_world_m,
            half_size_m=obstacle.half_size_m,
            geometry=obstacle.geometry,
            rgba=(0.55, 0.22, 0.12, 1.0),
            colliding=True,
        )
    for distractor in visual_distractors:
        distractor.validate()
        add_geometry(
            name=distractor.name,
            center_world_m=distractor.center_world_m,
            half_size_m=distractor.half_size_m,
            geometry=distractor.geometry,
            rgba=(0.1, 0.9, 0.25, 1.0)
            if distractor.appearance == "target"
            else (0.55, 0.22, 0.12, 1.0),
            colliding=False,
        )
    if include_legacy_goal_marker:
        target = spec.worldbody.add_body(
            name="target_marker", pos=scene.goal_world_m, mocap=True
        )
        target.add_geom(
            name="target_marker_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.12, 0.0, 0.0),
            rgba=(0.1, 0.9, 0.25, 0.55),
            group=2,
            contype=0,
            conaffinity=0,
        )
    target = target_area_from_scene(scene, target_config)
    half_x, half_y = target_config.half_size_xy_m
    target_area = spec.worldbody.add_body(
        name=f"{target_config.name}_body",
        pos=(
            target.center_xy_world_m[0],
            target.center_xy_world_m[1],
            target_config.surface_z_m - target_config.thickness_m / 2.0,
        ),
    )
    target_area.add_geom(
        name=f"{target_config.name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(half_x, half_y, target_config.thickness_m / 2.0),
        rgba=target_config.rgba,
        group=2,
        contype=0,
        conaffinity=0,
    )
    safe_half_x, safe_half_y = target_config.safe_half_size_xy_m
    target_area.add_geom(
        name=f"{target_config.name}_safe_region_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.0, target_config.thickness_m / 2.0 + 0.0005),
        size=(safe_half_x, safe_half_y, 0.0005),
        rgba=(0.05, 0.45, 0.10, 0.55),
        group=2,
        contype=0,
        conaffinity=0,
    )
    cross_height = target_config.thickness_m / 2.0 + 0.001
    cross_half_width = min(target_config.size_xy_m) * 0.035
    target_area.add_geom(
        name=f"{target_config.name}_cross_x_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.0, cross_height),
        size=(half_x * 0.45, cross_half_width, 0.001),
        rgba=(0.95, 0.95, 0.95, 1.0),
        group=2,
        contype=0,
        conaffinity=0,
    )
    target_area.add_geom(
        name=f"{target_config.name}_cross_y_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.0, cross_height),
        size=(cross_half_width, half_y * 0.45, 0.001),
        rgba=(0.95, 0.95, 0.95, 1.0),
        group=2,
        contype=0,
        conaffinity=0,
    )
    return spec.compile()
