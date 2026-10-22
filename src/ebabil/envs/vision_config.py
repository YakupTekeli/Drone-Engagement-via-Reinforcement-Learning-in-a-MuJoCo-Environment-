"""Kamera tabanlı navigasyon ortamının sürümlü konfigürasyonu."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ebabil.envs.config import (
    ActionConfig,
    NavigationEnvConfig,
    ObservationConfig,
    RewardConfig,
    SimulationConfig,
    TerminationConfig,
)
from ebabil.envs.motion_constraint import MotionConstraintConfig, SpeedGovernorConfig
from ebabil.missions import TerminalHoverConfig
from ebabil.missions.terminal_assist import TerminalAssistConfig
from ebabil.perception.detection_slots import DetectionSlotConfig
from ebabil.perception.visual_randomization import VisualDomainConfig
from ebabil.rewards import TerminalHoverRewardConfig
from ebabil.scenes import TargetAreaConfig

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VISION_NAVIGATION_CONFIG = (
    PACKAGE_ROOT / "configs" / "environment" / "vision_navigation.toml"
)


def _positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} pozitif ve sonlu olmalı")
    return value


@dataclass(frozen=True)
class TargetUncertaintyConfig:
    """ADR-019 yaklaşık hedef konumu hatası."""

    error_radius_m: float

    def validate(self) -> None:
        if not math.isfinite(self.error_radius_m) or self.error_radius_m < 0.0:
            raise ValueError(
                "terminal_hover.target_error_radius_m negatif olmayan olmalı"
            )


@dataclass(frozen=True)
class VisionObservationConfig:
    latent_size: int
    latent_scale: float
    waypoint_distance_scale_m: float
    maximum_frame_age_s: float
    #: ADR-022 §4: `vision-v3` son `k` policy adımının latent'ini üst üste
    #: koyar. `v0`/`vision-v2` için `1`.
    latent_stack_size: int = 1

    def validate(self) -> None:
        if self.latent_size <= 0:
            raise ValueError("observation.latent_size pozitif olmalı")
        if isinstance(self.latent_stack_size, bool) or self.latent_stack_size < 1:
            raise ValueError("observation.latent_stack_size en az 1 olmalı")
        _positive(self.latent_scale, "observation.latent_scale")
        _positive(
            self.waypoint_distance_scale_m,
            "observation.waypoint_distance_scale_m",
        )
        _positive(self.maximum_frame_age_s, "observation.maximum_frame_age_s")


@dataclass(frozen=True)
class SegmentBudgetConfig:
    nominal_speed_m_s: float
    speed_floor_m_s: float
    time_factor: float
    fixed_seconds: float
    terminal_approach_seconds: float
    terminal_hold_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    maximum_planned_route_length_m: float

    def validate(self) -> None:
        for name, value in vars(self).items():
            _positive(float(value), f"segment_budget.{name}")
        if self.time_factor <= 1.0:
            raise ValueError("segment_budget.time_factor 1 değerinden büyük olmalı")
        if self.minimum_seconds > self.maximum_seconds:
            raise ValueError("Segment minimum süresi maximum süreyi aşamaz")


VISUAL_DOMAIN_SPLITS = ("train", "validation", "development_ood", "audit")
DATASET_CONFIG_ROOT = PACKAGE_ROOT / "configs" / "training"


@dataclass(frozen=True)
class VisualRandomizationConfig:
    """ADR-022 §6: reset ve kare başına görsel domain randomization.

    Görsel alanlar dataset config'inin `[visual.*]` bölümlerinden okunur;
    böylece politika kodlayıcının eğitildiği dağılımın aynısını görür.
    `split = "train"` eğitim alanıdır; evaluation held-out alan için
    `"validation"` verir.
    """

    enabled: bool = False
    domain_config: str = "vision_dataset_d1.toml"
    split: str = "train"

    def validate(self) -> None:
        if not self.domain_config:
            raise ValueError("visual_randomization.domain_config boş olamaz")
        if self.split not in VISUAL_DOMAIN_SPLITS:
            raise ValueError(
                "visual_randomization.split train/validation/development_ood/"
                "audit olmalı"
            )

    def domain_config_path(self) -> Path:
        candidate = Path(self.domain_config).expanduser()
        return (
            candidate.resolve()
            if candidate.is_absolute()
            else (DATASET_CONFIG_ROOT / candidate).resolve()
        )

    def load_domain(self, split: str | None = None) -> VisualDomainConfig:
        from ebabil.perception.dataset import (
            load_vision_dataset_config,
            visual_domain_for_split,
        )

        chosen = self.split if split is None else split
        if chosen not in VISUAL_DOMAIN_SPLITS:
            raise ValueError(f"Bilinmeyen görsel alan split'i: {chosen}")
        dataset_config = load_vision_dataset_config(self.domain_config_path())
        return visual_domain_for_split(dataset_config, chosen)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PrivilegedVisibilityConfig:
    """ADR-022 §5: hover kapısı için segmentation tabanlı hedef görünürlüğü."""

    minimum_target_pixels: int = 8

    def validate(self) -> None:
        if (
            isinstance(self.minimum_target_pixels, bool)
            or self.minimum_target_pixels <= 0
        ):
            raise ValueError(
                "privileged_visibility.minimum_target_pixels pozitif olmalı"
            )


@dataclass(frozen=True)
class DetectionObservationConfig:
    """ADR-023 §4-5: `vision-v4` kutu yuvaları, hover kapısı ve dedektör cihazı."""

    obstacle_slots: int = 8
    range_max_m: float = 12.0
    obstacle_conf: float = 0.25
    target_conf: float = 0.50
    target_min_area_px: float = 16.0
    bottom_margin_px: float = 1.0
    device: str = "mps"
    #: Segmentation tabanlı hedef görünürlüğü yalnız `info` tanısı için
    #: hesaplanır (kare başına ek render); reward/termination okumaz.
    privileged_diagnostics: bool = True

    def validate(self) -> None:
        self.slot_config().validate()
        if self.device not in ("mps", "cpu", "cuda"):
            raise ValueError("detections.device mps/cpu/cuda olmalı")

    def slot_config(self) -> DetectionSlotConfig:
        return DetectionSlotConfig(
            obstacle_slots=self.obstacle_slots,
            range_max_m=self.range_max_m,
            obstacle_conf=self.obstacle_conf,
            target_conf=self.target_conf,
            target_min_area_px=self.target_min_area_px,
            bottom_margin_px=self.bottom_margin_px,
        )


@dataclass(frozen=True)
class VisionNavigationEnvConfig:
    navigation: NavigationEnvConfig
    observation: VisionObservationConfig
    segment_budget: SegmentBudgetConfig
    terminal_hover: TerminalHoverConfig
    terminal_hover_reward: TerminalHoverRewardConfig
    target_uncertainty: TargetUncertaintyConfig
    visual_randomization: VisualRandomizationConfig = VisualRandomizationConfig()
    privileged_visibility: PrivilegedVisibilityConfig = PrivilegedVisibilityConfig()
    detections: DetectionObservationConfig = DetectionObservationConfig()
    motion_constraint: MotionConstraintConfig = MotionConstraintConfig()
    speed_governor: SpeedGovernorConfig = SpeedGovernorConfig()
    terminal_assist: TerminalAssistConfig = TerminalAssistConfig()

    def validate(self) -> None:
        self.navigation.validate()
        self.observation.validate()
        self.segment_budget.validate()
        self.terminal_hover.validate()
        self.terminal_hover_reward.validate()
        self.target_uncertainty.validate()
        self.visual_randomization.validate()
        self.privileged_visibility.validate()
        self.detections.validate()
        self.motion_constraint.validate()
        self.speed_governor.validate()
        self.terminal_assist.validate()

    def validate_against_target(self, target_config: TargetAreaConfig) -> None:
        """Hedef alan geometrisiyle çapraz tutarlılığı doğrula."""

        self.terminal_hover.validate_against_target(target_config)
        hover_altitude_m = target_config.surface_z_m + target_config.hover_height_m
        cruise_m = self.terminal_hover_reward.cruise_altitude_m
        if cruise_m <= hover_altitude_m:
            raise ValueError(
                "cruise_altitude_m hover irtifasının üzerinde olmalı; aksi halde "
                "seyir cezası terminal inişle çelişir: "
                f"{cruise_m:.3f} m <= {hover_altitude_m:.3f} m"
            )
        if self.target_uncertainty.error_radius_m > 0.0:
            safe_half = min(target_config.safe_half_size_xy_m)
            if self.target_uncertainty.error_radius_m <= safe_half:
                raise ValueError(
                    "target_error_radius_m güvenli iç bölgeden büyük olmalı; "
                    f"{self.target_uncertainty.error_radius_m:.3f} m <= "
                    f"{safe_half:.3f} m ise görüntü kullanılmadan başarıya "
                    "ulaşılabilir (ADR-019)"
                )
        if self.segment_budget.terminal_hold_seconds < target_config.hold_seconds:
            raise ValueError(
                "segment_budget.terminal_hold_seconds hedef hold süresinden "
                f"kısa olamaz: {self.segment_budget.terminal_hold_seconds:.2f} s "
                f"< {target_config.hold_seconds:.2f} s"
            )


def load_vision_navigation_config(
    path: str | Path = DEFAULT_VISION_NAVIGATION_CONFIG,
) -> VisionNavigationEnvConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Vision navigasyon konfigürasyonu bulunamadı: {config_path}"
        )
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    simulation = raw["simulation"]
    action = raw["action"]
    observation = raw["observation"]
    termination = raw["termination"]
    reward = raw["reward"]
    budget = raw["segment_budget"]
    hover = raw["terminal_hover"]
    hover_reward = raw["terminal_hover_reward"]
    randomization = raw.get("visual_randomization", {})
    visibility = raw.get("privileged_visibility", {})
    detections = raw.get("detections", {})
    motion = raw.get("motion_constraint", {})
    assist = raw.get("terminal_assist", {})
    governor = raw.get("speed_governor", {})
    navigation = NavigationEnvConfig(
        simulation=SimulationConfig(
            policy_frequency_hz=float(simulation["policy_frequency_hz"]),
            max_episode_seconds=float(budget["maximum_seconds"]),
        ),
        action=ActionConfig(
            max_horizontal_speed_m_s=float(action["max_forward_speed_m_s"]),
            max_vertical_speed_m_s=float(action["max_vertical_speed_m_s"]),
            max_yaw_rate_rad_s=float(action["max_yaw_rate_rad_s"]),
        ),
        observation=ObservationConfig(
            goal_distance_scale_m=float(observation["waypoint_distance_scale_m"]),
        ),
        termination=TerminationConfig(
            goal_radius_m=float(termination["goal_radius_m"]),
            goal_max_speed_m_s=float(termination["goal_max_speed_m_s"]),
            goal_hold_seconds=float(termination["goal_hold_seconds"]),
            minimum_altitude_m=float(termination["minimum_altitude_m"]),
            maximum_tilt_rad=math.radians(float(termination["maximum_tilt_deg"])),
        ),
        reward=RewardConfig(
            progress_weight=float(reward["progress_weight"]),
            goal_bonus=float(reward["goal_bonus"]),
            collision_penalty=float(reward["collision_penalty"]),
            failure_penalty=float(reward["failure_penalty"]),
            proximity_weight=float(reward["proximity_weight"]),
            proximity_threshold_m=float(reward["proximity_threshold_m"]),
            smoothness_weight=float(reward["smoothness_weight"]),
            attitude_weight=float(reward["attitude_weight"]),
            time_penalty=float(reward["time_penalty"]),
        ),
    )
    config = VisionNavigationEnvConfig(
        navigation=navigation,
        observation=VisionObservationConfig(
            latent_size=int(observation["latent_size"]),
            latent_scale=float(observation["latent_scale"]),
            waypoint_distance_scale_m=float(observation["waypoint_distance_scale_m"]),
            maximum_frame_age_s=float(observation["maximum_frame_age_s"]),
            latent_stack_size=int(observation.get("latent_stack_size", 1)),
        ),
        visual_randomization=VisualRandomizationConfig(
            enabled=bool(randomization.get("enabled", False)),
            domain_config=str(
                randomization.get("domain_config", "vision_dataset_d1.toml")
            ),
            split=str(randomization.get("split", "train")),
        ),
        privileged_visibility=PrivilegedVisibilityConfig(
            minimum_target_pixels=int(visibility.get("minimum_target_pixels", 8)),
        ),
        detections=DetectionObservationConfig(
            obstacle_slots=int(detections.get("obstacle_slots", 8)),
            range_max_m=float(detections.get("range_max_m", 12.0)),
            obstacle_conf=float(detections.get("obstacle_conf", 0.25)),
            target_conf=float(detections.get("target_conf", 0.50)),
            target_min_area_px=float(detections.get("target_min_area_px", 16.0)),
            bottom_margin_px=float(detections.get("bottom_margin_px", 1.0)),
            device=str(detections.get("device", "mps")),
            privileged_diagnostics=bool(detections.get("privileged_diagnostics", True)),
        ),
        motion_constraint=MotionConstraintConfig(
            enabled=bool(motion.get("enabled", False)),
            half_angle_deg=float(motion.get("half_angle_deg", 30.0)),
            terminal_blind_speed_m_s=float(motion.get("terminal_blind_speed_m_s", 0.5)),
            relaxed_states=tuple(
                str(name) for name in motion.get("relaxed_states", ("settle", "hold"))
            ),
        ),
        speed_governor=SpeedGovernorConfig(
            enabled=bool(governor.get("enabled", False)),
            corridor_half_width_norm=float(
                governor.get("corridor_half_width_norm", 0.4)
            ),
            safe_range_m=float(governor.get("safe_range_m", 0.6)),
            gain_per_s=float(governor.get("gain_per_s", 1.0)),
            min_speed_m_s=float(governor.get("min_speed_m_s", 0.15)),
            invalid_range_m=float(governor.get("invalid_range_m", 0.5)),
            range_max_m=float(governor.get("range_max_m", 12.0)),
            turn_penalty=float(governor.get("turn_penalty", 0.0)),
        ),
        terminal_assist=TerminalAssistConfig(
            enabled=bool(assist.get("enabled", False)),
            states=tuple(
                str(name)
                for name in assist.get("states", ("approach", "settle", "hold"))
            ),
            kp_horizontal=float(assist.get("kp_horizontal", 1.0)),
            kp_vertical=float(assist.get("kp_vertical", 1.0)),
            kyaw=float(assist.get("kyaw", 0.8)),
            max_horizontal_speed_m_s=float(assist.get("max_horizontal_speed_m_s", 0.5)),
            max_vertical_speed_m_s=float(assist.get("max_vertical_speed_m_s", 0.3)),
            max_yaw_rate_rad_s=float(assist.get("max_yaw_rate_rad_s", 0.5)),
            half_extent_guess_m=float(assist.get("half_extent_guess_m", 1.2)),
            bottom_margin_px=float(assist.get("bottom_margin_px", 4.0)),
        ),
        terminal_hover=TerminalHoverConfig(
            acquire_score=float(hover["acquire_score"]),
            loss_score=float(hover["loss_score"]),
            acquire_seconds=float(hover["acquire_seconds"]),
            grace_seconds=float(hover["grace_seconds"]),
            approach_radius_m=float(hover["approach_radius_m"]),
            search_entry_radius_m=float(hover["search_entry_radius_m"]),
            search_timeout_seconds=float(hover["search_timeout_seconds"]),
            far_loss_returns_to_transit=bool(
                hover.get("far_loss_returns_to_transit", False)
            ),
        ),
        target_uncertainty=TargetUncertaintyConfig(
            error_radius_m=float(hover["target_error_radius_m"]),
        ),
        terminal_hover_reward=TerminalHoverRewardConfig(
            target_progress_weight=float(hover_reward["target_progress_weight"]),
            target_visible_weight=float(hover_reward["target_visible_weight"]),
            cruise_altitude_weight=float(hover_reward["cruise_altitude_weight"]),
            cruise_altitude_m=float(hover_reward["cruise_altitude_m"]),
            bearing_alignment_weight=float(hover_reward["bearing_alignment_weight"]),
            target_alignment_weight=float(hover_reward["target_alignment_weight"]),
            settle_speed_weight=float(hover_reward["settle_speed_weight"]),
            settle_altitude_weight=float(hover_reward["settle_altitude_weight"]),
            settle_tilt_weight=float(hover_reward["settle_tilt_weight"]),
            settle_angular_rate_weight=float(
                hover_reward["settle_angular_rate_weight"]
            ),
            hover_hold_weight=float(hover_reward["hover_hold_weight"]),
            proximity_weight=float(hover_reward["proximity_weight"]),
            proximity_threshold_m=float(hover_reward["proximity_threshold_m"]),
            smoothness_weight=float(hover_reward["smoothness_weight"]),
            success_bonus=float(hover_reward["success_bonus"]),
            collision_penalty=float(hover_reward["collision_penalty"]),
            out_of_bounds_penalty=float(hover_reward["out_of_bounds_penalty"]),
            abort_penalty=float(hover_reward["abort_penalty"]),
            time_penalty=float(hover_reward["time_penalty"]),
        ),
        segment_budget=SegmentBudgetConfig(
            nominal_speed_m_s=float(budget["nominal_speed_m_s"]),
            speed_floor_m_s=float(budget["speed_floor_m_s"]),
            time_factor=float(budget["time_factor"]),
            fixed_seconds=float(budget["fixed_seconds"]),
            terminal_approach_seconds=float(budget["terminal_approach_seconds"]),
            terminal_hold_seconds=float(budget["terminal_hold_seconds"]),
            minimum_seconds=float(budget["minimum_seconds"]),
            maximum_seconds=float(budget["maximum_seconds"]),
            maximum_planned_route_length_m=float(
                budget["maximum_planned_route_length_m"]
            ),
        ),
    )
    config.validate()
    return config
