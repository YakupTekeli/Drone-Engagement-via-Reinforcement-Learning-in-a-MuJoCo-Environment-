"""Seed ayrımlı vision dataset planlama, etiketleme ve depolama araçları."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import tomllib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

import mujoco
import numpy as np
import numpy.typing as npt

from ebabil.perception.camera import HeadlessCamera, PrivilegedCameraLabels
from ebabil.perception.config import CameraConfig, load_camera_config
from ebabil.perception.visual_randomization import (
    VisualDomainConfig,
    apply_frame_visual_parameters,
    apply_scene_visual_parameters,
    sample_frame_visual_parameters,
    sample_scene_visual_parameters,
)
from ebabil.scenes.generator import (
    EndpointMode,
    ObstacleGeometry,
    SceneGeneratorConfig,
    SceneMode,
    StaticScene,
    VisualDistractor,
    build_scene_model,
    generate_static_scene,
    load_scene_config,
)
from ebabil.scenes.target import TargetAreaConfig, load_target_area_config

Split = Literal["train", "validation", "development_ood", "audit"]
SPLITS: tuple[Split, ...] = (
    "train",
    "validation",
    "development_ood",
    "audit",
)
FloatArray = npt.NDArray[np.float64]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VISION_DATASET_CONFIG = (
    PACKAGE_ROOT / "configs" / "training" / "vision_dataset.toml"
)
ACTOR_KEYS = frozenset({"sample_id", "rgb", "timestamp_s"})
PRIVILEGED_KEYS = frozenset(
    {
        "sample_id",
        "depth_m",
        "segmentation",
        "target_mask",
        "obstacle_mask",
        "target_visible",
        "target_center_xy",
        "target_area_fraction",
        "free_space",
        "position_world_m",
        "quaternion_wxyz",
        "timestamp_s",
    }
)


@dataclass(frozen=True)
class VisionDatasetConfig:
    schema_version: str
    base_seed: int
    split_salt: str
    train_scene_count: int
    validation_scene_count: int
    development_ood_scene_count: int
    audit_scene_count: int
    samples_per_scene: int
    visual_distractor_count: int
    scene_mode: SceneMode
    endpoint_mode: EndpointMode
    minimum_progress: float
    maximum_progress: float
    maximum_lateral_offset_m: float
    minimum_altitude_m: float
    maximum_altitude_m: float
    maximum_yaw_jitter_rad: float
    target_negative_fraction: float
    hold_pose_fraction: float
    negative_yaw_min_rad: float
    negative_yaw_max_rad: float
    free_space_max_depth_m: float
    minimum_target_pixels: int
    train_visual_domain: VisualDomainConfig
    validation_visual_domain: VisualDomainConfig
    development_ood_visual_domain: VisualDomainConfig
    audit_visual_domain: VisualDomainConfig
    train_obstacle_geometries: tuple[ObstacleGeometry, ...]
    validation_obstacle_geometries: tuple[ObstacleGeometry, ...]
    development_ood_obstacle_geometries: tuple[ObstacleGeometry, ...]
    audit_obstacle_geometries: tuple[ObstacleGeometry, ...]

    def validate(self) -> None:
        if self.schema_version != "vision-dataset-v4":
            raise ValueError("dataset.schema_version vision-dataset-v4 olmalı")
        if self.base_seed < 0 or not self.split_salt:
            raise ValueError("dataset seed'i negatif olamaz ve split_salt boş olamaz")
        counts = (
            self.train_scene_count,
            self.validation_scene_count,
            self.development_ood_scene_count,
            self.audit_scene_count,
            self.samples_per_scene,
            self.minimum_target_pixels,
        )
        if any(isinstance(value, bool) or value <= 0 for value in counts):
            raise ValueError("Dataset sahne/örnek/piksel sayıları pozitif olmalı")
        if self.visual_distractor_count < 0:
            raise ValueError("visual_distractor_count negatif olamaz")
        if self.scene_mode not in ("clear", "single_blocker", "random_obstacles"):
            raise ValueError("dataset.scene_mode geçersiz")
        if self.endpoint_mode not in (
            "fixed",
            "random_start",
            "random_goal",
            "random_both",
        ):
            raise ValueError("dataset.endpoint_mode geçersiz")
        if not 0.0 <= self.minimum_progress < self.maximum_progress <= 1.0:
            raise ValueError("Dataset progress aralığı [0, 1] içinde olmalı")
        positive_values = (
            self.maximum_lateral_offset_m,
            self.minimum_altitude_m,
            self.maximum_altitude_m,
            self.maximum_yaw_jitter_rad,
            self.negative_yaw_min_rad,
            self.negative_yaw_max_rad,
            self.free_space_max_depth_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive_values):
            raise ValueError("Dataset mesafe/açı değerleri pozitif ve sonlu olmalı")
        if self.minimum_altitude_m >= self.maximum_altitude_m:
            raise ValueError("minimum_altitude_m maximum_altitude_m'den küçük olmalı")
        if not 0.0 <= self.target_negative_fraction < 1.0:
            raise ValueError("target_negative_fraction [0, 1) aralığında olmalı")
        if not 0.0 <= self.hold_pose_fraction < 1.0:
            raise ValueError("hold_pose_fraction [0, 1) aralığında olmalı")
        if self.hold_pose_fraction + self.target_negative_fraction >= 1.0:
            raise ValueError(
                "hold_pose_fraction ve target_negative_fraction toplamı 1'den "
                "küçük olmalı; rota pozlarına pay kalmıyor"
            )
        if self.negative_yaw_min_rad >= self.negative_yaw_max_rad:
            raise ValueError("negative yaw min değeri max değerinden küçük olmalı")
        visual_domains = (
            self.train_visual_domain,
            self.validation_visual_domain,
            self.development_ood_visual_domain,
            self.audit_visual_domain,
        )
        for visual_domain in visual_domains:
            visual_domain.validate()
        if len({domain.name for domain in visual_domains}) != len(visual_domains):
            raise ValueError("Bütün visual domain adları ayrık olmalı")
        geometry_domains = (
            self.train_obstacle_geometries,
            self.validation_obstacle_geometries,
            self.development_ood_obstacle_geometries,
            self.audit_obstacle_geometries,
        )
        supported = {"box", "cylinder", "capsule", "sphere", "ellipsoid"}
        if any(
            not geometries or not set(geometries).issubset(supported)
            for geometries in geometry_domains
        ):
            raise ValueError("Obstacle geometry domain'i boş veya geçersiz")
        if not set(self.audit_obstacle_geometries) - set(
            self.train_obstacle_geometries
        ):
            raise ValueError("Audit en az bir train-dışı engel geometrisi içermeli")


@dataclass(frozen=True)
class CaptureSpec:
    sample_id: str
    split: Split
    scene_seed: int
    sample_index: int
    position_world_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class DatasetRecord:
    sample_id: str
    split: Split
    scene_seed: int
    sample_index: int
    scene_mode: SceneMode
    actor_file: str
    privileged_file: str
    visual_domain: str = "legacy_nominal"
    scene_visual_seed: int = 0
    frame_visual_seed: int = 0


@dataclass(frozen=True)
class SupervisionTargets:
    target_visible: bool
    target_center_xy: tuple[float, float]
    target_area_fraction: float
    free_space: tuple[float, float, float]


@dataclass(frozen=True)
class SplitDatasetStatistics:
    record_count: int
    scene_count: int
    target_visible_count: int
    target_visible_fraction: float
    obstacle_visible_count: int
    obstacle_visible_fraction: float
    rgb_mean: tuple[float, float, float]
    rgb_std: tuple[float, float, float]
    free_space_mean: tuple[float, float, float]
    free_space_std: tuple[float, float, float]
    visual_domains: tuple[str, ...]


@dataclass(frozen=True)
class DatasetProvenance:
    schema_version: str
    config_sha256: str
    metadata_sha256: str
    manifest_sha256: str
    statistics_sha256: str


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vision_dataset_provenance(dataset_dir: str | Path) -> DatasetProvenance:
    root = Path(dataset_dir).expanduser().resolve()
    metadata_path = root / "metadata.json"
    manifest_path = root / "manifest.jsonl"
    statistics_path = root / "statistics.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Dataset metadata JSON nesnesi olmalı")
    schema_version = str(metadata.get("schema_version", ""))
    if schema_version != "vision-dataset-v4":
        raise ValueError("Dataset metadata schema_version vision-dataset-v4 olmalı")
    config_payload = metadata.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("Dataset metadata config nesnesi eksik")
    if str(config_payload.get("schema_version", "")) != schema_version:
        raise ValueError("Dataset config ve metadata schema sürümleri eşleşmiyor")
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    if not isinstance(statistics, dict) or set(statistics) != set(SPLITS):
        raise ValueError("Dataset statistics bütün v4 split'lerini içermeli")
    if not load_manifest(manifest_path):
        raise ValueError("Dataset manifest boş")
    config_sha256 = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return DatasetProvenance(
        schema_version=schema_version,
        config_sha256=config_sha256,
        metadata_sha256=file_sha256(metadata_path),
        manifest_sha256=file_sha256(manifest_path),
        statistics_sha256=file_sha256(statistics_path),
    )


def _float_pair(raw: object, name: str) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{name} iki değer içermeli")
    return float(raw[0]), float(raw[1])


def _float_rgb(raw: object, name: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{name} üç değer içermeli")
    return float(raw[0]), float(raw[1]), float(raw[2])


def _float_value(raw: object, name: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{name} sayısal olmalı")
    return float(raw)


def _obstacle_geometries(raw: object, name: str) -> tuple[ObstacleGeometry, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{name} boş olmayan bir liste olmalı")
    supported = {"box", "cylinder", "capsule", "sphere", "ellipsoid"}
    values = tuple(str(value) for value in raw)
    if len(set(values)) != len(values) or not set(values).issubset(supported):
        raise ValueError(f"{name} yinelenen veya geçersiz geometri içeriyor")
    return cast(tuple[ObstacleGeometry, ...], values)


def _load_visual_domain(raw: dict[str, object]) -> VisualDomainConfig:
    config = VisualDomainConfig(
        name=str(raw["name"]),
        light_scale=_float_pair(raw["light_scale"], "light_scale"),
        texture_tint_min=_float_rgb(raw["texture_tint_min"], "texture_tint_min"),
        texture_tint_max=_float_rgb(raw["texture_tint_max"], "texture_tint_max"),
        obstacle_rgb_min=_float_rgb(raw["obstacle_rgb_min"], "obstacle_rgb_min"),
        obstacle_rgb_max=_float_rgb(raw["obstacle_rgb_max"], "obstacle_rgb_max"),
        target_rgb_min=_float_rgb(raw["target_rgb_min"], "target_rgb_min"),
        target_rgb_max=_float_rgb(raw["target_rgb_max"], "target_rgb_max"),
        exposure_gain=_float_pair(raw["exposure_gain"], "exposure_gain"),
        gamma=_float_pair(raw["gamma"], "gamma"),
        noise_std=_float_pair(raw["noise_std"], "noise_std"),
        vignette_strength=_float_pair(raw["vignette_strength"], "vignette_strength"),
        blur_probability=_float_value(raw["blur_probability"], "blur_probability"),
    )
    config.validate()
    return config


def load_vision_dataset_config(
    path: str | Path = DEFAULT_VISION_DATASET_CONFIG,
) -> VisionDatasetConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Vision dataset config bulunamadı: {config_path}")
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)
    raw = raw_config["dataset"]
    visual = raw_config["visual"]
    config = VisionDatasetConfig(
        schema_version=str(raw["schema_version"]),
        base_seed=int(raw["base_seed"]),
        split_salt=str(raw["split_salt"]),
        train_scene_count=int(raw["train_scene_count"]),
        validation_scene_count=int(raw["validation_scene_count"]),
        development_ood_scene_count=int(raw["development_ood_scene_count"]),
        audit_scene_count=int(raw["audit_scene_count"]),
        samples_per_scene=int(raw["samples_per_scene"]),
        visual_distractor_count=int(raw["visual_distractor_count"]),
        scene_mode=cast(SceneMode, raw["scene_mode"]),
        endpoint_mode=cast(EndpointMode, raw["endpoint_mode"]),
        minimum_progress=float(raw["minimum_progress"]),
        maximum_progress=float(raw["maximum_progress"]),
        maximum_lateral_offset_m=float(raw["maximum_lateral_offset_m"]),
        minimum_altitude_m=float(raw["minimum_altitude_m"]),
        maximum_altitude_m=float(raw["maximum_altitude_m"]),
        maximum_yaw_jitter_rad=math.radians(float(raw["maximum_yaw_jitter_deg"])),
        target_negative_fraction=float(raw["target_negative_fraction"]),
        hold_pose_fraction=float(raw["hold_pose_fraction"]),
        negative_yaw_min_rad=math.radians(float(raw["negative_yaw_min_deg"])),
        negative_yaw_max_rad=math.radians(float(raw["negative_yaw_max_deg"])),
        free_space_max_depth_m=float(raw["free_space_max_depth_m"]),
        minimum_target_pixels=int(raw["minimum_target_pixels"]),
        train_visual_domain=_load_visual_domain(visual["train"]),
        validation_visual_domain=_load_visual_domain(visual["validation"]),
        development_ood_visual_domain=_load_visual_domain(visual["development_ood"]),
        audit_visual_domain=_load_visual_domain(visual["audit"]),
        train_obstacle_geometries=_obstacle_geometries(
            raw["train_obstacle_geometries"], "train_obstacle_geometries"
        ),
        validation_obstacle_geometries=_obstacle_geometries(
            raw["validation_obstacle_geometries"],
            "validation_obstacle_geometries",
        ),
        development_ood_obstacle_geometries=_obstacle_geometries(
            raw["development_ood_obstacle_geometries"],
            "development_ood_obstacle_geometries",
        ),
        audit_obstacle_geometries=_obstacle_geometries(
            raw["audit_obstacle_geometries"], "audit_obstacle_geometries"
        ),
    )
    config.validate()
    return config


def split_for_scene_seed(seed: int, config: VisionDatasetConfig) -> Split:
    """Salt'lı sabit hash ile bütün bir sahneyi tek split'e ata."""

    if seed < 0:
        raise ValueError("scene seed negatif olamaz")
    digest = hashlib.blake2b(
        f"{config.split_salt}:{seed}".encode(), digest_size=8
    ).digest()
    counts = (
        config.train_scene_count,
        config.validation_scene_count,
        config.development_ood_scene_count,
        config.audit_scene_count,
    )
    bucket = int.from_bytes(digest, "big") % sum(counts)
    cumulative = 0
    for split, count in zip(SPLITS, counts, strict=True):
        cumulative += count
        if bucket < cumulative:
            return split
    raise AssertionError("Split bucket kapsam dışında")


def select_scene_seeds(config: VisionDatasetConfig) -> dict[Split, tuple[int, ...]]:
    """İstenen sayıda ayrık sahne seed'ini deterministik seç."""

    config.validate()
    requested: dict[Split, int] = {
        "train": config.train_scene_count,
        "validation": config.validation_scene_count,
        "development_ood": config.development_ood_scene_count,
        "audit": config.audit_scene_count,
    }
    selected: dict[Split, list[int]] = {
        "train": [],
        "validation": [],
        "development_ood": [],
        "audit": [],
    }
    candidate = config.base_seed
    while any(len(selected[split]) < requested[split] for split in requested):
        split = split_for_scene_seed(candidate, config)
        if len(selected[split]) < requested[split]:
            selected[split].append(candidate)
        candidate += 1
    return {split: tuple(values) for split, values in selected.items()}


def visual_domain_for_split(
    config: VisionDatasetConfig, split: Split
) -> VisualDomainConfig:
    return {
        "train": config.train_visual_domain,
        "validation": config.validation_visual_domain,
        "development_ood": config.development_ood_visual_domain,
        "audit": config.audit_visual_domain,
    }[split]


def visual_seed(
    config: VisionDatasetConfig,
    scene_seed: int,
    sample_index: int,
    split: Split,
) -> int:
    split_code = {split_name: index for index, split_name in enumerate(SPLITS)}[split]
    seed_sequence = np.random.SeedSequence(
        [config.base_seed, scene_seed, sample_index, split_code, 0xEBAB1]
    )
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def obstacle_geometries_for_split(
    config: VisionDatasetConfig, split: Split
) -> tuple[ObstacleGeometry, ...]:
    return {
        "train": config.train_obstacle_geometries,
        "validation": config.validation_obstacle_geometries,
        "development_ood": config.development_ood_obstacle_geometries,
        "audit": config.audit_obstacle_geometries,
    }[split]


def _scene_with_perception_geometries(
    scene: StaticScene, split: Split, config: VisionDatasetConfig
) -> StaticScene:
    geometries = obstacle_geometries_for_split(config, split)
    rng = np.random.default_rng(visual_seed(config, scene.seed, 0xFFFE, split))
    obstacles = tuple(
        replace(obstacle, geometry=geometries[int(rng.integers(len(geometries)))])
        for obstacle in scene.obstacles
    )
    return replace(scene, obstacles=obstacles)


def _sample_visual_distractors(
    scene: StaticScene,
    split: Split,
    config: VisionDatasetConfig,
    scene_config: SceneGeneratorConfig,
) -> tuple[VisualDistractor, ...]:
    """Renk etiketinin semantik kestirme olmasını önleyen çarpışmasız nesneler üret."""

    if config.visual_distractor_count == 0:
        return ()
    rng = np.random.default_rng(visual_seed(config, scene.seed, 0xFFFD, split))
    geometries: tuple[ObstacleGeometry, ...] = (
        "box",
        "cylinder",
        "capsule",
        "sphere",
        "ellipsoid",
    )
    distractors: list[VisualDistractor] = []
    for index in range(config.visual_distractor_count):
        half_size = rng.uniform(
            scene_config.obstacle_half_size_min_m * 0.35,
            scene_config.obstacle_half_size_max_m * 0.75,
        )
        center = np.empty(3, dtype=np.float64)
        for _ in range(scene_config.max_sampling_attempts):
            center[:2] = rng.uniform(
                scene_config.workspace_min_m[:2] + half_size[:2],
                scene_config.workspace_max_m[:2] - half_size[:2],
            )
            if (
                np.linalg.norm(center[:2] - scene.start_world_m[:2]) > 0.75
                and np.linalg.norm(center[:2] - scene.goal_world_m[:2]) > 0.75
            ):
                break
        center[2] = scene_config.workspace_min_m[2] + half_size[2]
        appearance: Literal["obstacle", "target"] = (
            "target" if index % 2 == 0 else "obstacle"
        )
        distractors.append(
            VisualDistractor(
                name=f"distractor_{appearance}_{index:03d}",
                center_world_m=center.copy(),
                half_size_m=half_size,
                geometry=geometries[int(rng.integers(len(geometries)))],
                appearance=appearance,
            )
        )
    return tuple(distractors)


def _point_is_clear(
    point: FloatArray,
    scene: StaticScene,
    scene_config: SceneGeneratorConfig,
) -> bool:
    if np.any(point < scene_config.workspace_min_m) or np.any(
        point > scene_config.workspace_max_m
    ):
        return False
    for obstacle in scene.obstacles:
        expanded_half_size = obstacle.half_size_m + scene_config.vehicle_clearance_m
        if np.all(np.abs(point - obstacle.center_world_m) <= expanded_half_size):
            return False
    return True


def make_capture_spec(
    scene: StaticScene,
    split: Split,
    sample_index: int,
    config: VisionDatasetConfig,
    scene_config: SceneGeneratorConfig,
    *,
    target_config: TargetAreaConfig | None = None,
) -> CaptureSpec:
    """Sahne hattı çevresinde çarpışmasız, tekrar üretilebilir kamera pozu üret.

    Pozlar iki aileden gelir. Rota ailesi start-goal hattı boyunca `progress`,
    yanal offset ve irtifa örnekler. `hold_pose_fraction` olasılığıyla seçilen
    hold ailesi ise ADR-012 hover zarfının içinden örnekler: güvenli iç bölge
    XY'si ve hover anchor irtifası +- irtifa toleransı. Hold ailesi, terminal
    görevin `target_score` kapısının dayandığı poz bölgesini eğitim dağılımına
    sokar; rota ailesi tek başına bu bölgeyi yüzde bir civarında temsil eder.

    Hold ailesinde yaw rota yönünde tutulur ve negatif yaw dalı uygulanmaz;
    hedefin kadrajda olup olmadığı segmentation maskesinden dürüstçe
    etiketlenir. ADR-016 nadir kör bölgesi nedeniyle bu pozların bir kısmının
    negatif etiketlenmesi beklenen davranıştır.
    """

    if sample_index < 0:
        raise ValueError("sample_index negatif olamaz")
    target_config = target_config or load_target_area_config()
    target_config.validate()
    seed_sequence = np.random.SeedSequence([config.base_seed, scene.seed, sample_index])
    rng = np.random.default_rng(seed_sequence)
    direction_xy = scene.goal_world_m[:2] - scene.start_world_m[:2]
    norm = float(np.linalg.norm(direction_xy))
    if norm <= 1e-9:
        raise ValueError("Dataset pozu için yatay start-goal ayrımı gerekli")
    forward = direction_xy / norm
    left = np.array([-forward[1], forward[0]], dtype=np.float64)
    hold_pose = bool(rng.random() < config.hold_pose_fraction)
    position = scene.start_world_m.copy()
    if hold_pose:
        anchor = np.array(
            [
                float(scene.goal_world_m[0]),
                float(scene.goal_world_m[1]),
                target_config.surface_z_m + target_config.hover_height_m,
            ],
            dtype=np.float64,
        )
        safe_half_x, safe_half_y = target_config.safe_half_size_xy_m
        tolerance = target_config.altitude_tolerance_m
        for _ in range(100):
            position[0] = anchor[0] + rng.uniform(-safe_half_x, safe_half_x)
            position[1] = anchor[1] + rng.uniform(-safe_half_y, safe_half_y)
            position[2] = anchor[2] + rng.uniform(-tolerance, tolerance)
            if _point_is_clear(position, scene, scene_config):
                break
        else:
            position = anchor.copy()
    else:
        for _ in range(100):
            progress = rng.uniform(config.minimum_progress, config.maximum_progress)
            lateral = rng.uniform(
                -config.maximum_lateral_offset_m, config.maximum_lateral_offset_m
            )
            position[:2] = (
                scene.start_world_m[:2] + progress * direction_xy + lateral * left
            )
            position[2] = rng.uniform(
                config.minimum_altitude_m, config.maximum_altitude_m
            )
            if _point_is_clear(position, scene, scene_config):
                break
        else:
            position = scene.start_world_m.copy()
    route_yaw = math.atan2(float(forward[1]), float(forward[0]))
    if not hold_pose and rng.random() < config.target_negative_fraction:
        yaw_offset = rng.uniform(
            config.negative_yaw_min_rad, config.negative_yaw_max_rad
        ) * rng.choice((-1.0, 1.0))
    else:
        yaw_offset = rng.uniform(
            -config.maximum_yaw_jitter_rad, config.maximum_yaw_jitter_rad
        )
    yaw = route_yaw + yaw_offset
    quaternion = (
        math.cos(yaw / 2.0),
        0.0,
        0.0,
        math.sin(yaw / 2.0),
    )
    sample_id = f"{split}-s{scene.seed:010d}-p{sample_index:03d}"
    return CaptureSpec(
        sample_id=sample_id,
        split=split,
        scene_seed=scene.seed,
        sample_index=sample_index,
        position_world_m=tuple(float(value) for value in position),  # type: ignore[arg-type]
        quaternion_wxyz=quaternion,
    )


def derive_supervision(
    labels: PrivilegedCameraLabels,
    *,
    minimum_target_pixels: int,
    free_space_max_depth_m: float,
) -> SupervisionTargets:
    """Privileged render kanallarından deploy girdisine girmeyen hedefler türet.

    `target_center_xy`, hedef görünmediğinde `(0.0, 0.0)` sentinel'i alır ve bu
    değer "hedef tam görüntü merkezinde" durumundan ayırt edilemez. Bu nedenle
    merkez tek başına görünürlük kanıtı değildir: hem eğitim hem runtime
    tüketicileri görünürlüğü yalnız `target_visible` etiketi veya encoder'ın
    `target_logit` sigmoid skoru üzerinden kapılamalıdır.

    Hedef görünürken merkez, hedefin görüntüdeki **görünen** piksellerinin
    ağırlık merkezidir; hedefin gerçek geometrik merkezi değildir. Kamera
    nadir kör bölgesi nedeniyle hedef üstü hover'da merkez kadraj dışında
    kalabilir, bu durumda etiket görünen kenara doğru kayar.
    """

    if minimum_target_pixels <= 0 or free_space_max_depth_m <= 0.0:
        raise ValueError("Etiket eşikleri pozitif olmalı")
    height, width = labels.target_mask.shape
    if labels.obstacle_mask.shape != (height, width) or labels.depth_m.shape != (
        height,
        width,
    ):
        raise ValueError("Privileged mask/depth boyutları eşleşmeli")
    target_y, target_x = np.nonzero(labels.target_mask)
    target_visible = target_x.size >= minimum_target_pixels
    if target_visible:
        center = (
            2.0 * float(np.mean(target_x)) / max(width - 1, 1) - 1.0,
            2.0 * float(np.mean(target_y)) / max(height - 1, 1) - 1.0,
        )
    else:
        center = (0.0, 0.0)
    free_space: list[float] = []
    edges = np.linspace(0, width, 4, dtype=int)
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        obstacle_pixels = labels.obstacle_mask[:, left:right]
        obstacle_depth = labels.depth_m[:, left:right][obstacle_pixels]
        finite_depth = obstacle_depth[np.isfinite(obstacle_depth)]
        if finite_depth.size:
            nearest_robust_depth = float(np.quantile(finite_depth, 0.10))
            score = np.clip(nearest_robust_depth / free_space_max_depth_m, 0.0, 1.0)
        else:
            score = 1.0
        free_space.append(float(score))
    return SupervisionTargets(
        target_visible=target_visible,
        target_center_xy=center,
        target_area_fraction=float(np.mean(labels.target_mask)),
        free_space=(free_space[0], free_space[1], free_space[2]),
    )


def _atomic_npz(path: Path, **arrays: npt.ArrayLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz", delete=False
    ) as tmp:
        temporary_path = Path(tmp.name)
        np.savez_compressed(tmp, **arrays)  # type: ignore[arg-type]
    temporary_path.replace(path)


def write_sample(
    output_dir: Path,
    spec: CaptureSpec,
    actor_rgb: npt.NDArray[np.uint8],
    labels: PrivilegedCameraLabels,
    targets: SupervisionTargets,
    scene_mode: SceneMode,
    visual_domain: str = "legacy_nominal",
    scene_visual_seed: int = 0,
    frame_visual_seed: int = 0,
) -> DatasetRecord:
    """Actor ve privileged payload'larını fiziksel olarak ayrı arşivlere yaz."""

    actor_relative = Path("actor") / spec.split / f"{spec.sample_id}.npz"
    privileged_relative = Path("privileged") / spec.split / f"{spec.sample_id}.npz"
    sample_id = np.asarray(spec.sample_id)
    _atomic_npz(
        output_dir / actor_relative,
        sample_id=sample_id,
        rgb=actor_rgb,
        timestamp_s=np.asarray(labels.timestamp_s, dtype=np.float64),
    )
    _atomic_npz(
        output_dir / privileged_relative,
        sample_id=sample_id,
        depth_m=labels.depth_m,
        segmentation=labels.segmentation,
        target_mask=labels.target_mask,
        obstacle_mask=labels.obstacle_mask,
        target_visible=np.asarray(targets.target_visible),
        target_center_xy=np.asarray(targets.target_center_xy, dtype=np.float32),
        target_area_fraction=np.asarray(targets.target_area_fraction, dtype=np.float32),
        free_space=np.asarray(targets.free_space, dtype=np.float32),
        position_world_m=np.asarray(spec.position_world_m, dtype=np.float64),
        quaternion_wxyz=np.asarray(spec.quaternion_wxyz, dtype=np.float64),
        timestamp_s=np.asarray(labels.timestamp_s, dtype=np.float64),
    )
    return DatasetRecord(
        sample_id=spec.sample_id,
        split=spec.split,
        scene_seed=spec.scene_seed,
        sample_index=spec.sample_index,
        scene_mode=scene_mode,
        actor_file=actor_relative.as_posix(),
        privileged_file=privileged_relative.as_posix(),
        visual_domain=visual_domain,
        scene_visual_seed=scene_visual_seed,
        frame_visual_seed=frame_visual_seed,
    )


def _jsonable_camera_config(config: CameraConfig) -> dict[str, object]:
    raw = asdict(config)
    for key, value in tuple(raw.items()):
        if isinstance(value, tuple):
            raw[key] = list(value)
    return raw


def generate_vision_dataset(
    output_dir: str | Path,
    config: VisionDatasetConfig | None = None,
    *,
    camera_config: CameraConfig | None = None,
    scene_config: SceneGeneratorConfig | None = None,
    target_config: TargetAreaConfig | None = None,
    progress_callback: Callable[[int, int, Split, int], None] | None = None,
) -> tuple[DatasetRecord, ...]:
    """Seed'li MuJoCo sahnelerinden ayrık actor/privileged dataset üret."""

    config = config or load_vision_dataset_config()
    camera_config = camera_config or load_camera_config()
    scene_config = scene_config or load_scene_config()
    target_config = target_config or load_target_area_config()
    config.validate()
    output_path = Path(output_dir).expanduser().resolve()
    manifest_path = output_path / "manifest.jsonl"
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f"Dataset dizini boş değil; üzerine yazılmadı: {output_path}"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    records: list[DatasetRecord] = []
    selected_scene_seeds = select_scene_seeds(config)
    total_scenes = sum(len(values) for values in selected_scene_seeds.values())
    completed_scenes = 0
    for split, scene_seeds in selected_scene_seeds.items():
        domain = visual_domain_for_split(config, split)
        for scene_seed in scene_seeds:
            scene = generate_static_scene(
                scene_seed,
                scene_config,
                mode=config.scene_mode,
                endpoint_mode=config.endpoint_mode,
            )
            perception_scene = _scene_with_perception_geometries(scene, split, config)
            visual_distractors = _sample_visual_distractors(
                perception_scene, split, config, scene_config
            )
            model = build_scene_model(
                perception_scene,
                camera_config=camera_config,
                target_config=target_config,
                include_legacy_goal_marker=False,
                visual_distractors=visual_distractors,
                # REV-030: zemin düzlemi workspace'i kapsamazsa kamera boşluk
                # görür. Dataset tarafında da geçilmek zorunda, yoksa encoder
                # politikanın göreceğinden farklı bir dünyada eğitilir.
                ground_half_size_m=scene_config.ground_half_size_m,
            )
            scene_effect_seed = visual_seed(config, scene_seed, 0xFFFF, split)
            scene_effects = sample_scene_visual_parameters(domain, scene_effect_seed)
            apply_scene_visual_parameters(model, scene_effects)
            data = mujoco.MjData(model)
            with HeadlessCamera(model, camera_config) as camera:
                for sample_index in range(config.samples_per_scene):
                    spec = make_capture_spec(
                        scene,
                        split,
                        sample_index,
                        config,
                        scene_config,
                        target_config=target_config,
                    )
                    data.qpos[:3] = spec.position_world_m
                    data.qpos[3:7] = spec.quaternion_wxyz
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                    actor = camera.capture_actor(data)
                    privileged = camera.capture_privileged(data)
                    frame_effect_seed = visual_seed(
                        config, scene_seed, sample_index, split
                    )
                    frame_effects = sample_frame_visual_parameters(
                        domain, frame_effect_seed
                    )
                    actor_rgb = apply_frame_visual_parameters(
                        actor.rgb,
                        frame_effects,
                        noise_seed=frame_effect_seed,
                    )
                    targets = derive_supervision(
                        privileged,
                        minimum_target_pixels=config.minimum_target_pixels,
                        free_space_max_depth_m=config.free_space_max_depth_m,
                    )
                    records.append(
                        write_sample(
                            output_path,
                            spec,
                            actor_rgb,
                            privileged,
                            targets,
                            config.scene_mode,
                            domain.name,
                            scene_effect_seed,
                            frame_effect_seed,
                        )
                    )
            completed_scenes += 1
            if progress_callback is not None:
                progress_callback(completed_scenes, total_scenes, split, scene_seed)
    manifest_text = "".join(
        json.dumps(asdict(record), sort_keys=True) + "\n" for record in records
    )
    manifest_path.write_text(manifest_text, encoding="utf-8")
    metadata = {
        "schema_version": config.schema_version,
        "config": asdict(config),
        "camera": _jsonable_camera_config(camera_config),
        "record_count": len(records),
        "split_scene_seeds": selected_scene_seeds,
        "actor_keys": sorted(ACTOR_KEYS),
        "privileged_keys": sorted(PRIVILEGED_KEYS),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    statistics = summarize_vision_dataset(output_path, records)
    (output_path / "statistics.json").write_text(
        json.dumps(
            {split: asdict(values) for split, values in statistics.items()},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return tuple(records)


def load_manifest(path: str | Path) -> tuple[DatasetRecord, ...]:
    manifest_path = Path(path).expanduser().resolve()
    records: list[DatasetRecord] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        split = str(raw["split"])
        scene_mode = str(raw["scene_mode"])
        if split not in SPLITS:
            raise ValueError(f"Manifest split geçersiz: {split}")
        if scene_mode not in ("clear", "single_blocker", "random_obstacles"):
            raise ValueError(f"Manifest scene_mode geçersiz: {scene_mode}")
        records.append(
            DatasetRecord(
                sample_id=str(raw["sample_id"]),
                split=cast(Split, split),
                scene_seed=int(raw["scene_seed"]),
                sample_index=int(raw["sample_index"]),
                scene_mode=cast(SceneMode, scene_mode),
                actor_file=str(raw["actor_file"]),
                privileged_file=str(raw["privileged_file"]),
                visual_domain=str(raw.get("visual_domain", "legacy_nominal")),
                scene_visual_seed=int(raw.get("scene_visual_seed", 0)),
                frame_visual_seed=int(raw.get("frame_visual_seed", 0)),
            )
        )
    return tuple(records)


def iter_split_records(
    records: Sequence[DatasetRecord], split: Split
) -> Iterator[DatasetRecord]:
    return (record for record in records if record.split == split)


def summarize_vision_dataset(
    dataset_dir: str | Path,
    records: Sequence[DatasetRecord] | None = None,
) -> dict[Split, SplitDatasetStatistics]:
    """Split sınıf/görünürlük ve RGB/free-space dağılımlarını özetle."""

    root = Path(dataset_dir).expanduser().resolve()
    records = records or load_manifest(root / "manifest.jsonl")
    summaries: dict[Split, SplitDatasetStatistics] = {}
    for split in SPLITS:
        split_records = tuple(iter_split_records(records, split))
        if not split_records:
            raise ValueError(f"İstatistik için dataset split'i boş: {split}")
        rgb_sum = np.zeros(3, dtype=np.float64)
        rgb_square_sum = np.zeros(3, dtype=np.float64)
        rgb_pixel_count = 0
        free_space_values: list[FloatArray] = []
        target_visible_count = 0
        obstacle_visible_count = 0
        for record in split_records:
            actor_path = (root / record.actor_file).resolve()
            privileged_path = (root / record.privileged_file).resolve()
            if not actor_path.is_relative_to(
                root
            ) or not privileged_path.is_relative_to(root):
                raise ValueError(f"Dataset path kök dışına çıkıyor: {record.sample_id}")
            with np.load(actor_path, allow_pickle=False) as actor_payload:
                normalized_rgb = actor_payload["rgb"].astype(np.float64) / 255.0
            pixels = normalized_rgb.reshape(-1, 3)
            rgb_sum += pixels.sum(axis=0)
            rgb_square_sum += np.square(pixels).sum(axis=0)
            rgb_pixel_count += pixels.shape[0]
            with np.load(privileged_path, allow_pickle=False) as privileged_payload:
                target_visible_count += int(bool(privileged_payload["target_visible"]))
                obstacle_visible_count += int(
                    np.any(privileged_payload["obstacle_mask"])
                )
                free_space_values.append(
                    np.asarray(privileged_payload["free_space"], dtype=np.float64)
                )
        count = len(split_records)
        rgb_mean = rgb_sum / rgb_pixel_count
        rgb_variance = np.maximum(
            rgb_square_sum / rgb_pixel_count - np.square(rgb_mean), 0.0
        )
        free_space = np.vstack(free_space_values)
        summaries[split] = SplitDatasetStatistics(
            record_count=count,
            scene_count=len({record.scene_seed for record in split_records}),
            target_visible_count=target_visible_count,
            target_visible_fraction=target_visible_count / count,
            obstacle_visible_count=obstacle_visible_count,
            obstacle_visible_fraction=obstacle_visible_count / count,
            rgb_mean=tuple(float(value) for value in rgb_mean),  # type: ignore[arg-type]
            rgb_std=tuple(float(value) for value in np.sqrt(rgb_variance)),  # type: ignore[arg-type]
            free_space_mean=tuple(float(value) for value in free_space.mean(axis=0)),  # type: ignore[arg-type]
            free_space_std=tuple(float(value) for value in free_space.std(axis=0)),  # type: ignore[arg-type]
            visual_domains=tuple(
                sorted({record.visual_domain for record in split_records})
            ),
        )
    return summaries


def validate_dataset_leakage(dataset_dir: str | Path) -> None:
    """Actor arşiv anahtarlarını allowlist ile ve scene split'lerini denetle."""

    root = Path(dataset_dir).expanduser().resolve()
    records = load_manifest(root / "manifest.jsonl")
    if not records:
        raise ValueError("Dataset manifest boş")
    seen_samples: set[str] = set()
    scene_splits: dict[int, Split] = {}
    for record in records:
        if record.sample_id in seen_samples:
            raise ValueError(f"Tekrarlanan sample_id: {record.sample_id}")
        seen_samples.add(record.sample_id)
        previous_split = scene_splits.setdefault(record.scene_seed, record.split)
        if previous_split != record.split:
            raise ValueError(f"Scene seed split sızıntısı: {record.scene_seed}")
        actor_path = (root / record.actor_file).resolve()
        privileged_path = (root / record.privileged_file).resolve()
        if not actor_path.is_relative_to(root) or not privileged_path.is_relative_to(
            root
        ):
            raise ValueError(f"Dataset path kök dışına çıkıyor: {record.sample_id}")
        if actor_path == privileged_path:
            raise ValueError(f"Actor/privileged dosyaları aynı: {record.sample_id}")
        with np.load(actor_path, allow_pickle=False) as actor_payload:
            if set(actor_payload.files) != ACTOR_KEYS:
                raise ValueError(
                    f"Actor payload allowlist ihlali: {actor_path}: "
                    f"{sorted(actor_payload.files)}"
                )
            if str(actor_payload["sample_id"]) != record.sample_id:
                raise ValueError(f"Actor sample_id eşleşmiyor: {record.sample_id}")
            rgb = actor_payload["rgb"]
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"Actor RGB şeması geçersiz: {record.sample_id}")
            actor_timestamp = float(actor_payload["timestamp_s"])
        with np.load(privileged_path, allow_pickle=False) as privileged_payload:
            if set(privileged_payload.files) != PRIVILEGED_KEYS:
                raise ValueError(
                    f"Privileged payload şeması geçersiz: {privileged_path}"
                )
            if str(privileged_payload["sample_id"]) != record.sample_id:
                raise ValueError(f"Privileged sample_id eşleşmiyor: {record.sample_id}")
            if not np.isclose(
                actor_timestamp, float(privileged_payload["timestamp_s"])
            ):
                raise ValueError(
                    f"Actor/label timestamp eşleşmiyor: {record.sample_id}"
                )
