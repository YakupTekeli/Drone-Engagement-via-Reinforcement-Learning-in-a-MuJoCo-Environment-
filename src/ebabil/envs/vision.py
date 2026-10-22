"""Onaylı monoküler encoder kullanan GPS-denied yerel navigasyon ortamı."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Protocol

import mujoco
import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces

from ebabil.controllers import FlightState
from ebabil.envs.navigation import (
    EbabilNavigationEnv,
    TerminalDecision,
    VelocityCommand,
)
from ebabil.envs.vision_config import (
    DEFAULT_VISION_NAVIGATION_CONFIG,
    SegmentBudgetConfig,
    VisionNavigationEnvConfig,
    load_vision_navigation_config,
)
from ebabil.missions import (
    HoverConditionReading,
    HoverState,
    HoverStateMachine,
    HoverStateUpdate,
)
from ebabil.perception import (
    ActorCameraFrame,
    CameraConfig,
    HeadlessCamera,
    TinyVisionEncoder,
    load_approved_vision_encoder,
    load_camera_config,
    preprocess_actor_rgb,
)
from ebabil.rewards import (
    TerminalHoverRewardBreakdown,
    compute_terminal_hover_reward,
)
from ebabil.scenes import (
    EndpointMode,
    SceneGeneratorConfig,
    SceneMode,
    StaticScene,
    TargetArea,
    TargetAreaConfig,
    believed_target_xy,
    load_target_area_config,
    reachable_safe_region_xy,
    target_area_from_scene,
    validate_hover_envelope,
)
from ebabil.scenes.generator import build_scene_model
from ebabil.sensors import NavigationSensorReading

FloatArray = npt.NDArray[np.float64]
VisionObservation = dict[str, npt.NDArray[np.float32]]
VISION_ENVIRONMENT_ID = "EbabilVisionNavigation-v0"
VISION_OBSERVATION_VERSION = "vision-v2"
#: REV-037: `obstacle_map` kanalının ızgara boyutu. Encoder zaten bu
#: çözünürlükte bir engel ısı haritası üretiyordu fakat politikaya
#: verilmiyordu. Kamera 45 derece aşağı baktığı için görüntü SATIRLARI
#: zemin MESAFESİNE, KOLONLARI yön açısına karşılık gelir; yani bu bir
#: yerel mesafe x yön doluluk ızgarasıdır. Önceki `vision-v1` yalnız üç
#: dikey dilimlik `free_space` veriyordu ve koridorun NEREDE olduğunu
#: taşıyamıyordu.
#: REV-038: `vision_aux` artık `collision_risk` taşımaz. O kanal tanım
#: gereği `free_space < collision_risk_threshold`'dı, yani politikanın
#: zaten sahip olduğu `free_space`'in eşiklenmiş kopyası. `obstacle_map`
#: (108 hücre) aynı bilgiyi kat kat daha yüksek çözünürlükte taşıyor.
#: Kanal sıralaması: [target_score, center_x, center_y, free_L, free_C, free_R]
VISION_AUX_SIZE = 6
OBSTACLE_MAP_SHAPE = (9, 12)
OBSTACLE_MAP_SIZE = OBSTACLE_MAP_SHAPE[0] * OBSTACLE_MAP_SHAPE[1]
#: ADR-020: action sözleşmesi. `v1` ileri-yalnız 3 boyutluydu; `v2` gerçek
#: quadrotorun yapabildiği tam holonomik yatay hareketi verir. Observation
#: sözleşmesi DEĞİŞMEZ, bu yüzden onaylı encoder zinciri etkilenmez.
VISION_ACTION_VERSION = "vision-action-v2"

#: REV-033: bu sebeplerle biten episode toplanan `target_progress`'i geri
#: verir. Hover abort'u ve `time_limit` listede DEĞİLDİR: onlar erken
#: öğrenme için gereken yaklaşma gradyanını korur. Ölçülmüş defekt "hedefe
#: git sonra yere çak" stratejisinin net +7 ödül vermesiydi.
PHYSICAL_FAILURE_REASONS = (
    "obstacle_collision",
    "ground_collision",
    "out_of_bounds",
    "minimum_altitude",
    "maximum_tilt",
)
#: Sıralı tutulur; `_observation` bunu birebir karşılaştırır.
VISION_OBSERVATION_KEYS = (
    "fc_state",
    "obstacle_map",
    "timing",
    "vision_aux",
    "vision_latent",
    "waypoint",
)


class ActorCamera(Protocol):
    def capture_actor(self, data: mujoco.MjData) -> ActorCameraFrame: ...

    def close(self) -> None: ...


CameraFactory = Callable[[mujoco.MjModel], ActorCamera]


@dataclass(frozen=True)
class SegmentHorizon:
    planned_route_length_m: float
    budget_seconds: float
    maximum_steps: int
    final_segment: bool


class SimulationDeadlineScheduler:
    """Periyodik sensörü simülasyon deadline'larına bağlayan scheduler."""

    def __init__(self, frequency_hz: float) -> None:
        if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise ValueError("Scheduler frequency pozitif ve sonlu olmalı")
        self.frequency_hz = float(frequency_hz)
        self.period_s = 1.0 / self.frequency_hz
        self._next_deadline_s: float | None = None

    def reset(self, timestamp_s: float) -> None:
        if not math.isfinite(timestamp_s):
            raise ValueError("Scheduler reset zamanı sonlu olmalı")
        self._next_deadline_s = float(timestamp_s) + self.period_s

    def sample_due(self, timestamp_s: float) -> bool:
        if self._next_deadline_s is None:
            raise RuntimeError("Scheduler sample öncesinde reset edilmeli")
        if not math.isfinite(timestamp_s):
            raise ValueError("Scheduler zamanı sonlu olmalı")
        tolerance = 1e-12
        if timestamp_s + tolerance < self._next_deadline_s:
            return False
        while timestamp_s + tolerance >= self._next_deadline_s:
            self._next_deadline_s += self.period_s
        return True


def compute_segment_horizon(
    planned_route_length_m: float,
    policy_frequency_hz: float,
    config: SegmentBudgetConfig,
    *,
    final_segment: bool,
) -> SegmentHorizon:
    """ADR-013 rota uzunluğu formülünü deterministik policy adımına çevir."""

    config.validate()
    if (
        isinstance(planned_route_length_m, bool)
        or not isinstance(planned_route_length_m, Real)
        or not math.isfinite(float(planned_route_length_m))
        or float(planned_route_length_m) <= 0.0
    ):
        raise ValueError("planned_route_length_m pozitif ve sonlu olmalı")
    if float(planned_route_length_m) > config.maximum_planned_route_length_m:
        raise ValueError("planned_route_length_m yerel segment üst sınırını aşıyor")
    if not math.isfinite(policy_frequency_hz) or policy_frequency_hz <= 0.0:
        raise ValueError("policy_frequency_hz pozitif ve sonlu olmalı")
    speed = max(config.nominal_speed_m_s, config.speed_floor_m_s)
    terminal_seconds = (
        config.terminal_approach_seconds + config.terminal_hold_seconds
        if final_segment
        else 0.0
    )
    raw_seconds = (
        config.fixed_seconds
        + config.time_factor * float(planned_route_length_m) / speed
        + terminal_seconds
    )
    budget_seconds = float(
        np.clip(raw_seconds, config.minimum_seconds, config.maximum_seconds)
    )
    return SegmentHorizon(
        planned_route_length_m=float(planned_route_length_m),
        budget_seconds=budget_seconds,
        maximum_steps=math.ceil(budget_seconds * policy_frequency_hz),
        final_segment=final_segment,
    )


class EbabilVisionNavigationEnv(EbabilNavigationEnv):
    """Frozen vision encoder ile yerel waypoint takibi ve engelden kaçınma."""

    metadata = {"render_modes": [], "render_fps": 20}
    ACTION_SIZE = 4
    observation_version = VISION_OBSERVATION_VERSION
    action_version = VISION_ACTION_VERSION
    #: `_observation` bu sıralı anahtar listesiyle birebir karşılaştırır. Alt
    #: sınıflar (ADR-022 `vision-v3`) kendi listelerini verir.
    observation_keys: tuple[str, ...] = VISION_OBSERVATION_KEYS

    def __init__(
        self,
        *,
        approval_manifest: str | Path,
        scene_mode: SceneMode = "clear",
        endpoint_mode: EndpointMode = "fixed",
        config: VisionNavigationEnvConfig | None = None,
        scene_config: SceneGeneratorConfig | None = None,
        camera_config: CameraConfig | None = None,
        target_config: TargetAreaConfig | None = None,
        camera_factory: CameraFactory | None = None,
        record_physics_trace: bool = False,
        render_mode: str | None = None,
    ) -> None:
        self.vision_config = config or load_vision_navigation_config()
        self.vision_config.validate()
        self.camera_config = camera_config or load_camera_config()
        self.camera_config.validate()
        self.target_config = target_config or load_target_area_config()
        self.target_config.validate()
        self.approval_manifest = Path(approval_manifest).expanduser().resolve()
        self._load_perception()
        self._camera_factory = camera_factory
        self._camera: ActorCamera | None = None
        self._camera_scheduler = SimulationDeadlineScheduler(
            self.camera_config.frame_rate_hz
        )
        self._camera_frame_timestamp_s = 0.0
        self._camera_frame_count = 0
        self._allocate_perception_buffers()
        self._horizon = SegmentHorizon(1.0, 1.0, 1, False)
        self._route_length_source = "unset"
        self._reset_options: dict[str, Any] = {}
        super().__init__(
            scene_mode=scene_mode,
            endpoint_mode=endpoint_mode,
            sensor_mode="ideal",
            config=self.vision_config.navigation,
            scene_config=scene_config,
            record_physics_trace=record_physics_trace,
            render_mode=render_mode,
        )
        validate_hover_envelope(self.target_config, self.scene_config)
        self.vision_config.validate_against_target(self.target_config)
        self.hover_state_machine = HoverStateMachine(
            self.vision_config.terminal_hover,
            self.target_config,
            self.config.simulation.policy_frequency_hz,
        )
        self._previous_anchor_distance_m = 0.0
        self._banked_progress = 0.0
        self._believed_target_world_m = np.zeros(3, dtype=np.float64)
        self.observation_space = spaces.Dict(
            {
                "fc_state": spaces.Box(-1.0, 1.0, shape=(10,), dtype=np.float32),
                "timing": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
                **self._perception_spaces(),
                "waypoint": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
            }
        )

    # -- Algı kancaları -----------------------------------------------------
    # ADR-022: `vision-v3` alt sınıfı yalnız bu kancaları değiştirir; zamanlama,
    # horizon, hover durum makinesi ve reward yolu ortaktır. `v0` davranışı
    # bit-bit korunur.

    def _load_perception(self) -> None:
        """Onaylı etiketli encoder'ı yükle (`vision-v2`)."""

        self.encoder: TinyVisionEncoder = load_approved_vision_encoder(
            self.approval_manifest
        )
        if (
            self.encoder.config.latent_size
            != self.vision_config.observation.latent_size
        ):
            raise ValueError(
                "Approved encoder latent boyutu "
                f"{VISION_OBSERVATION_VERSION} ile eşleşmiyor"
            )
        self.encoder.eval()
        self.encoder.requires_grad_(False)

    def _allocate_perception_buffers(self) -> None:
        self._vision_latent = np.zeros(
            self.vision_config.observation.latent_size, dtype=np.float64
        )
        self._vision_aux = np.zeros(VISION_AUX_SIZE, dtype=np.float64)
        self._obstacle_map = np.zeros(OBSTACLE_MAP_SIZE, dtype=np.float64)

    def _perception_spaces(self) -> dict[str, spaces.Space[Any]]:
        return {
            "vision_aux": spaces.Box(
                -1.0, 1.0, shape=(VISION_AUX_SIZE,), dtype=np.float32
            ),
            "obstacle_map": spaces.Box(
                0.0, 1.0, shape=(OBSTACLE_MAP_SIZE,), dtype=np.float32
            ),
            "vision_latent": spaces.Box(
                -1.0,
                1.0,
                shape=(self.vision_config.observation.latent_size,),
                dtype=np.float32,
            ),
        }

    def _create_actor_camera(self) -> ActorCamera:
        assert self.model is not None
        return (
            self._camera_factory(self.model)
            if self._camera_factory is not None
            else HeadlessCamera(self.model, self.camera_config)
        )

    def _encode_actor_frame(self, frame: ActorCameraFrame) -> None:
        """Doğrulanmış actor karesini algı tamponlarına işle."""

        image = preprocess_actor_rgb(frame.rgb).unsqueeze(0)
        with torch.inference_mode():
            output = self.encoder(image)
        target_score = torch.sigmoid(output.target_logit)[0]
        obstacle_map = (
            torch.sigmoid(output.obstacle_heatmap_logits)[0]
            .reshape(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        latent = output.latent[0].detach().cpu().numpy().astype(np.float64)
        auxiliary = np.concatenate(
            (
                np.asarray([float(target_score)], dtype=np.float64),
                output.target_center_xy[0].detach().cpu().numpy(),
                output.free_space[0].detach().cpu().numpy(),
            )
        ).astype(np.float64)
        if (
            latent.shape != self._vision_latent.shape
            or auxiliary.shape != (VISION_AUX_SIZE,)
            or obstacle_map.shape != (OBSTACLE_MAP_SIZE,)
        ):
            raise ValueError("Encoder çıktısı vision-v2 shape'iyle eşleşmiyor")
        if (
            not np.all(np.isfinite(latent))
            or not np.all(np.isfinite(auxiliary))
            or not np.all(np.isfinite(obstacle_map))
        ):
            raise ValueError("Encoder çıktısı yalnız sonlu değerler içermeli")
        self._vision_latent[:] = latent
        self._vision_aux[:] = auxiliary
        self._obstacle_map[:] = obstacle_map

    def _perception_observation(self) -> dict[str, npt.NDArray[np.float32]]:
        latent = np.clip(
            self._vision_latent / self.vision_config.observation.latent_scale,
            -1.0,
            1.0,
        )
        vision_aux = self._vision_aux.copy()
        # target_score ve free_space sigmoid çıktısı `[0, 1]`; center_xy
        # zaten `[-1, 1]`. Yalnız ilkler kaydırılır.
        vision_aux[[0, 3, 4, 5]] = 2.0 * vision_aux[[0, 3, 4, 5]] - 1.0
        return {
            "vision_aux": np.clip(vision_aux, -1.0, 1.0).astype(np.float32),
            # Sigmoid çıktısı zaten [0, 1]; kanal kendi space'inde ham kalır
            # ve `vision_aux` gibi [-1, 1]'e kaydırılmaz.
            "obstacle_map": np.clip(self._obstacle_map, 0.0, 1.0).astype(np.float32),
            "vision_latent": latent.astype(np.float32),
        }

    def _target_visibility_score(self) -> float:
        """Durum makinesi görsel kapısı: encoder'ın ham sigmoid skoru.

        `_vision_aux[0]` `[0, 1]` aralığındadır. `[-1, 1]` kaydırması yalnız
        `_observation` içindeki kopyaya uygulanır; burada ham skor kullanılır.
        """

        return float(self._vision_aux[0])

    def _target_center_x(self) -> float:
        """`target_alignment` bileşeninin kullandığı yatay hedef merkezi."""

        return float(self._vision_aux[1])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[VisionObservation, dict[str, Any]]:
        self._reset_options = dict(options or {})
        observation, info = super().reset(seed=seed, options=self._reset_options)
        if not isinstance(observation, dict):
            raise RuntimeError("vision-v1 observation Dict olmalı")
        info.update(self._hover_update.as_info())
        info["hover_reading_valid"] = False
        return observation, info

    def _build_scene_model(self, scene: StaticScene) -> mujoco.MjModel:
        return build_scene_model(
            scene,
            camera_config=self.camera_config,
            target_config=self.target_config,
            include_legacy_goal_marker=False,
            ground_half_size_m=self.scene_config.ground_half_size_m,
        )

    def _scene_target_config(self) -> TargetAreaConfig:
        """Hedef üstü hover görevi sahne açıklığı kısıtını zorunlu kılar."""

        return self.target_config

    @property
    def target_area(self) -> TargetArea:
        """Güncel sahnenin hedef alanı; tek kaynaklı factory üzerinden türetilir."""

        if self.scene is None:
            raise RuntimeError("target_area öncesinde reset çağrılmalı")
        return target_area_from_scene(self.scene, self.target_config)

    def _after_simulation_reset(self) -> None:
        assert self.model is not None
        assert self.data is not None
        assert self.scene is not None
        self._close_camera()
        reachable_safe_region_xy(self.target_area, self.scene_config)
        direct_distance_m = float(
            np.linalg.norm(self.scene.goal_world_m - self.scene.start_world_m)
        )
        route_value = self._reset_options.get("planned_route_length_m")
        if route_value is None:
            planned_route_length_m = direct_distance_m
            self._route_length_source = "direct_distance_fallback"
        else:
            if (
                isinstance(route_value, bool)
                or not isinstance(route_value, Real)
                or not math.isfinite(float(route_value))
            ):
                raise ValueError("planned_route_length_m pozitif ve sonlu olmalı")
            planned_route_length_m = float(route_value)
            if planned_route_length_m + 1e-9 < direct_distance_m:
                raise ValueError(
                    "planned_route_length_m düz çizgi mesafesinden kısa olamaz"
                )
            self._route_length_source = "reset_option"
        if "final_segment" in self._reset_options:
            raise ValueError(
                "final_segment reset option'ı kaldırıldı; ADR-018 tek hedefli "
                "görevde her episode nihaidir ve terminal rezervi her zaman "
                "uygulanır"
            )
        # ADR-018: ara waypoint yoktur, yani her episode nihai episode'dur.
        # Terminal yaklaşma ve hover hold rezervi bu nedenle koşulsuz eklenir;
        # aksi halde araç hem rotayı uçup hem 3 s hold'u tamamlamak için
        # sistematik olarak yetersiz bütçe alır.
        self._horizon = compute_segment_horizon(
            planned_route_length_m,
            self.config.simulation.policy_frequency_hz,
            self.vision_config.segment_budget,
            final_segment=True,
        )
        self.max_episode_steps = self._horizon.maximum_steps
        self._camera = self._create_actor_camera()
        self._camera_frame_count = 0
        self._camera_frame_timestamp_s = float(self.data.time)
        self._camera_scheduler.reset(float(self.data.time))
        self._capture_actor_frame()
        self.hover_state_machine.reset()
        # ADR-019: araç hedefin yalnız yaklaşık konumunu bilir. İnanılan konum
        # scene seed'inden deterministik üretilir ve episode içinde sabittir.
        # Gerçek konum yalnız privileged başarı doğrulamasında kullanılır.
        believed_xy = believed_target_xy(
            self.scene.goal_world_m,
            self.vision_config.target_uncertainty.error_radius_m,
            self._scene_seed,
            self.target_config.safe_half_size_xy_m,
        )
        self._believed_target_world_m = np.array(
            [
                believed_xy[0],
                believed_xy[1],
                self.target_config.surface_z_m + self.target_config.hover_height_m,
            ],
            dtype=np.float64,
        )
        anchor = self.target_area.hover_anchor_world_m
        self._previous_anchor_distance_m = float(
            np.linalg.norm(self.scene.start_world_m - anchor)
        )
        self._banked_progress = 0.0
        self._hover_update = HoverStateUpdate(
            state=HoverState.TRANSIT,
            previous_state=HoverState.TRANSIT,
            success=False,
            abort_reason=None,
            hold_steps=0,
            hold_seconds=0.0,
            hold_progress=0.0,
            acquire_steps=0,
            search_steps=0,
            search_seconds=0.0,
            loss_steps=0,
            loss_seconds=0.0,
            hover_conditions_met=False,
            target_visible=False,
        )

    def _after_physics_step(self) -> None:
        assert self.data is not None
        if self._camera_scheduler.sample_due(float(self.data.time)):
            self._capture_actor_frame()

    def _capture_actor_frame(self) -> None:
        assert self._camera is not None
        assert self.data is not None
        frame = self._camera.capture_actor(self.data)
        expected_shape = (
            self.camera_config.output_height_px,
            self.camera_config.output_width_px,
            3,
        )
        if frame.rgb.shape != expected_shape or frame.rgb.dtype != np.uint8:
            raise ValueError("Actor kamera karesi config shape/uint8 ile eşleşmiyor")
        timestamp_s = float(frame.timestamp_s)
        if (
            not math.isfinite(timestamp_s)
            or timestamp_s + 1e-12 < self._camera_frame_timestamp_s
            or timestamp_s > float(self.data.time) + 1e-12
        ):
            raise ValueError("Actor kamera timestamp'i geçersiz")
        self._encode_actor_frame(frame)
        self._camera_frame_timestamp_s = timestamp_s
        self._camera_frame_count += 1

    def _observation(self, reading: NavigationSensorReading) -> VisionObservation:
        assert self.scene is not None
        rotation_flat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation_flat, reading.ideal.orientation_wxyz)
        rotation = rotation_flat.reshape(3, 3)
        # ADR-019: waypoint inanılan hedeften türetilir. Gerçek hedef konumu
        # hiçbir actor kanalında bulunmaz.
        waypoint_world = self._believed_target_world_m - reading.ideal.position_world_m
        waypoint_body = rotation.T @ waypoint_world
        distance_m = float(np.linalg.norm(waypoint_body))
        direction_body = (
            waypoint_body / distance_m if distance_m > 1e-9 else np.zeros(3)
        )
        distance_fraction = min(
            math.log1p(distance_m)
            / math.log1p(self.vision_config.observation.waypoint_distance_scale_m),
            1.0,
        )
        waypoint = np.concatenate((direction_body, [2.0 * distance_fraction - 1.0]))
        velocity_scale = np.array(
            [
                self.config.action.max_horizontal_speed_m_s,
                self.config.action.max_horizontal_speed_m_s,
                self.config.action.max_vertical_speed_m_s,
            ],
            dtype=np.float64,
        )
        angular_scale = np.asarray(
            self.controller_config.attitude.max_body_rate_rad_s, dtype=np.float64
        )
        gravity_direction_body = rotation.T @ np.array([0.0, 0.0, -1.0])
        altitude_fraction = (
            reading.ideal.position_world_m[2] - self.scene_config.workspace_min_m[2]
        ) / (
            self.scene_config.workspace_max_m[2] - self.scene_config.workspace_min_m[2]
        )
        fc_state = np.concatenate(
            (
                reading.ideal.linear_velocity_body_m_s / velocity_scale,
                reading.ideal.angular_velocity_body_rad_s / angular_scale,
                gravity_direction_body,
                [2.0 * altitude_fraction - 1.0],
            )
        )
        remaining_fraction = float(
            np.clip(
                (self.max_episode_steps - self._episode_steps) / self.max_episode_steps,
                0.0,
                1.0,
            )
        )
        frame_age_fraction = min(
            self._camera_frame_age_s()
            / self.vision_config.observation.maximum_frame_age_s,
            1.0,
        )
        observation = {
            "fc_state": np.clip(fc_state, -1.0, 1.0).astype(np.float32),
            "timing": np.asarray(
                [remaining_fraction, frame_age_fraction], dtype=np.float32
            ),
            **self._perception_observation(),
            "waypoint": np.clip(waypoint, -1.0, 1.0).astype(np.float32),
        }
        if tuple(sorted(observation)) != self.observation_keys:
            raise RuntimeError(
                f"{self.observation_version} observation anahtarları değişti"
            )
        return observation

    def _velocity_command(
        self, action: FloatArray, state: FlightState
    ) -> VelocityCommand:
        yaw = self._yaw_from_rotation(state.rotation_body_to_world)
        maximum_horizontal = float(self.config.action.max_horizontal_speed_m_s)
        horizontal_body = np.asarray(action[:2], dtype=np.float64) * maximum_horizontal
        # Gerçek bir quadrotor yatay düzlemde izotropiktir: hız sınırı yöne göre
        # değişmez. Bileşenleri tek tek kırpmak çapraz komutta `sqrt(2)` kat
        # hızlı uçmaya izin verirdi, bu yüzden büyüklük kırpılır (ADR-020).
        speed = float(np.linalg.norm(horizontal_body))
        if speed > maximum_horizontal:
            horizontal_body *= maximum_horizontal / speed
        horizontal_body = self._constrain_horizontal_body(horizontal_body)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        velocity_world = np.array(
            [
                cosine * horizontal_body[0] - sine * horizontal_body[1],
                sine * horizontal_body[0] + cosine * horizontal_body[1],
                action[2] * self.config.action.max_vertical_speed_m_s,
            ],
            dtype=np.float64,
        )
        return VelocityCommand(
            velocity_world_m_s=velocity_world,
            yaw_rate_rad_s=float(action[3] * self.config.action.max_yaw_rate_rad_s),
        )

    def _constrain_horizontal_body(self, horizontal_body: FloatArray) -> FloatArray:
        """Gövde yatay hız komutu kancası; `v0`/`v1`'de değişmez (ADR-020)."""

        return horizontal_body

    def _base_info(self, state: FlightState) -> dict[str, Any]:
        info = super()._base_info(state)
        info.update(
            {
                "observation_version": VISION_OBSERVATION_VERSION,
                "planned_route_length_m": self._horizon.planned_route_length_m,
                "planned_route_length_source": self._route_length_source,
                "segment_budget_seconds": self._horizon.budget_seconds,
                "segment_maximum_steps": self._horizon.maximum_steps,
                "segment_final": self._horizon.final_segment,
                "remaining_budget_fraction": max(
                    0.0,
                    (self.max_episode_steps - self._episode_steps)
                    / self.max_episode_steps,
                ),
                "budget_used_seconds": (
                    self._episode_steps / self.config.simulation.policy_frequency_hz
                ),
                "target_error_radius_m": (
                    self.vision_config.target_uncertainty.error_radius_m
                ),
                "target_error_m": float(
                    np.linalg.norm(
                        self._believed_target_world_m[:2] - self.scene.goal_world_m[:2]
                    )
                )
                if self.scene is not None
                else 0.0,
                "camera_frame_count": self._camera_frame_count,
                "camera_frame_timestamp_s": self._camera_frame_timestamp_s,
                "camera_frame_age_s": self._camera_frame_age_s(),
                "camera_frame_stale": self._camera_frame_age_s()
                > self.vision_config.observation.maximum_frame_age_s,
            }
        )
        return info

    def _hover_reading(self, state: FlightState) -> HoverConditionReading:
        """Privileged ground truth'tan durum makinesi girdisini üret.

        `target_score` actor yolundaki encoder çıktısıdır; geometrik alanlar
        ADR-012 uyarınca yalnız başarı/reward kararında kullanılır ve actor
        observation'ına girmez.
        """

        anchor = self.target_area.hover_anchor_world_m
        position = state.position_world_m
        target_score = self._target_visibility_score()
        believed = self._believed_target_world_m
        return HoverConditionReading(
            target_score=float(np.clip(target_score, 0.0, 1.0)),
            frame_stale=self._camera_frame_age_s()
            > self.vision_config.observation.maximum_frame_age_s,
            horizontal_distance_m=float(np.linalg.norm(position[:2] - anchor[:2])),
            believed_horizontal_distance_m=float(
                np.linalg.norm(position[:2] - believed[:2])
            ),
            inside_safe_region=self.target_area.contains_safe_xy(position),
            altitude_error_m=abs(float(position[2] - anchor[2])),
            speed_m_s=float(np.linalg.norm(state.velocity_world_m_s)),
            tilt_rad=math.acos(
                float(np.clip(state.rotation_body_to_world[2, 2], -1.0, 1.0))
            ),
            angular_rate_rad_s=float(np.linalg.norm(state.angular_velocity_body_rad_s)),
        )

    def _terminal_decision(
        self,
        state: FlightState,
        current_distance_m: float,
        failure_reason: str | None,
    ) -> TerminalDecision:
        """ADR-012 hover state machine'iyle başarı ve terminal sebebi kararı."""

        del current_distance_m
        reading = self._hover_reading(state)
        if failure_reason is not None:
            # Fiziksel failure hiçbir durumda başarı üretmez (ADR-012).
            # Durum makinesi ilerletilmez; son durum raporlanır.
            self._hover_update = HoverStateUpdate(
                state=self.hover_state_machine.state,
                previous_state=self.hover_state_machine.state,
                success=False,
                abort_reason=None,
                hold_steps=0,
                hold_seconds=0.0,
                hold_progress=0.0,
                acquire_steps=0,
                search_steps=0,
                search_seconds=0.0,
                loss_steps=0,
                loss_seconds=0.0,
                hover_conditions_met=self.hover_state_machine.hover_conditions_met(
                    reading
                ),
                target_visible=reading.target_score
                >= self.vision_config.terminal_hover.acquire_score,
            )
            return TerminalDecision(
                success=False,
                termination_reason=failure_reason,
                info=self._hover_update.as_info()
                | {
                    "hover_reading_valid": True,
                    "hover_horizontal_distance_m": reading.horizontal_distance_m,
                },
            )
        update = self.hover_state_machine.update(reading)
        self._hover_update = update
        termination_reason: str | None = None
        if update.success:
            termination_reason = "hover_success"
        elif update.abort_reason is not None:
            termination_reason = f"hover_abort_{update.abort_reason}"
        return TerminalDecision(
            success=update.success,
            termination_reason=termination_reason,
            info=update.as_info()
            | {
                "hover_reading_valid": True,
                # Kademe 1 `arrival_rate` metriği (ADR-022 §10) bu alandan
                # hesaplanır; gerçek hover anchor'a yatay mesafe, actor'a girmez.
                "hover_horizontal_distance_m": reading.horizontal_distance_m,
            },
        )

    def _compute_step_reward(
        self,
        *,
        state: FlightState,
        action: FloatArray,
        current_distance_m: float,
        minimum_range_m: float,
        success: bool,
        collision: bool,
        terminated: bool,
        truncated: bool,
        termination_reason: str | None,
    ) -> TerminalHoverRewardBreakdown:
        """ADR-012 bileşenlerini ayrı üreten terminal hover reward'ı."""

        del current_distance_m, truncated
        update = self._hover_update
        out_of_bounds = termination_reason == "out_of_bounds"
        anchor = self.target_area.hover_anchor_world_m
        anchor_distance_m = float(np.linalg.norm(state.position_world_m - anchor))
        aborted = terminated and not success and not collision
        reading = self._hover_reading(state)
        breakdown = compute_terminal_hover_reward(
            self.vision_config.terminal_hover_reward,
            state=update.state,
            previous_anchor_distance_m=self._previous_anchor_distance_m,
            anchor_distance_m=anchor_distance_m,
            target_center_x=self._target_center_x(),
            target_visible=update.target_visible,
            inside_safe_region=self.target_area.contains_safe_xy(
                state.position_world_m
            ),
            speed_m_s=float(np.linalg.norm(state.velocity_world_m_s)),
            minimum_range_m=minimum_range_m,
            current_action=action,
            previous_action=self._previous_action,
            success=success,
            collision=collision,
            out_of_bounds=out_of_bounds,
            aborted=aborted and not out_of_bounds,
            altitude_m=float(state.position_world_m[2]),
            believed_bearing_error_rad=self._believed_bearing_error_rad(state),
            speed_allowance_m_s=self._speed_allowance_m_s(anchor_distance_m),
            # REV-046: hold koşullarının beşi de şekillendirilir. Eşikler
            # hedef alan config'inden gelir, yani ödül ile başarı ölçütü
            # tek kaynaktan beslenir ve ayrışamaz.
            altitude_error_m=reading.altitude_error_m,
            altitude_allowance_m=self._altitude_allowance_m(anchor_distance_m),
            tilt_rad=reading.tilt_rad,
            maximum_tilt_rad=self.target_config.maximum_tilt_rad,
            angular_rate_rad_s=reading.angular_rate_rad_s,
            maximum_angular_rate_rad_s=(self.target_config.maximum_angular_rate_rad_s),
            physical_failure=termination_reason in PHYSICAL_FAILURE_REASONS,
            banked_progress=self._banked_progress,
        )
        self._previous_anchor_distance_m = anchor_distance_m
        self._banked_progress += breakdown.target_progress
        return breakdown

    def _speed_allowance_m_s(self, anchor_distance_m: float) -> float:
        """Mesafeyle ölçeklenen hız payı: yaklaşırken yavaşlamayı şekillendirir.

        REV-042: `approach_radius_m` mesafesinde seyir hızı, güvenli iç bölge
        kenarında hold hız sınırı. Aradaki geçiş doğrusaldır. Böylece araç
        güvenli bölgeye **girmeden** yavaşlamayı öğrenebilir; eski kurulumda
        ceza yalnız bölge içinde açıktı ve oraya hızlı gelen araç hiç
        giremiyordu.
        """

        hold_limit = float(self.target_config.maximum_linear_speed_m_s)
        cruise = float(self.config.action.max_horizontal_speed_m_s)
        safe_half = float(min(self.target_config.safe_half_size_xy_m))
        radius = float(self.vision_config.terminal_hover.approach_radius_m)
        span = radius - safe_half
        if span <= 0.0:
            return hold_limit
        fraction = (float(anchor_distance_m) - safe_half) / span
        fraction = float(np.clip(fraction, 0.0, 1.0))
        return hold_limit + (cruise - hold_limit) * fraction

    def _altitude_allowance_m(self, anchor_distance_m: float) -> float:
        """Mesafeyle ölçeklenen irtifa payı: yaklaşırken alçalmayı şekillendirir.

        REV-046: sabit bir pay (yalnız `altitude_tolerance_m`) aracı hedefi
        görür görmez hold bandına çekiyordu, hedef `10 m` uzakta olsa bile.
        Ölçülen sonuç `minimum_altitude` terminasyonunun `%70`'e çıkmasıydı.
        Pay artık `approach_radius_m`'de seyir irtifasına kadar açıktır ve
        güvenli iç bölgede hold toleransına iner; `settle_speed` payının
        aynı deseni.
        """

        tolerance_m = float(self.target_config.altitude_tolerance_m)
        hover_altitude_m = float(
            self.target_config.surface_z_m + self.target_config.hover_height_m
        )
        cruise_m = float(self.vision_config.terminal_hover_reward.cruise_altitude_m)
        far_allowance_m = max(tolerance_m, abs(cruise_m - hover_altitude_m))
        safe_half = float(min(self.target_config.safe_half_size_xy_m))
        radius = float(self.vision_config.terminal_hover.approach_radius_m)
        span = radius - safe_half
        if span <= 0.0:
            return tolerance_m
        fraction = float(
            np.clip((float(anchor_distance_m) - safe_half) / span, 0.0, 1.0)
        )
        return tolerance_m + (far_allowance_m - tolerance_m) * fraction

    def _believed_bearing_error_rad(self, state: FlightState) -> float:
        """Kameranın bakış yönü ile inanılan hedef yönü arasındaki açı.

        REV-041: **inanılan** hedeften türetilir, yani ayrıcalıklı değildir;
        `waypoint` kanalı bu yönü zaten taşır. Gerçek hedef kullanılmaz.
        """

        delta = self._believed_target_world_m[:2] - state.position_world_m[:2]
        if float(np.linalg.norm(delta)) <= 1e-9:
            return 0.0
        yaw = self._yaw_from_rotation(state.rotation_body_to_world)
        error = math.atan2(float(delta[1]), float(delta[0])) - yaw
        return float(abs((error + math.pi) % (2.0 * math.pi) - math.pi))

    def _camera_frame_age_s(self) -> float:
        if self.data is None:
            return 0.0
        return max(0.0, float(self.data.time) - self._camera_frame_timestamp_s)

    def _close_camera(self) -> None:
        if self._camera is not None:
            self._camera.close()
            self._camera = None

    def _close_additional_resources(self) -> None:
        self._close_camera()


__all__ = [
    "DEFAULT_VISION_NAVIGATION_CONFIG",
    "VISION_ENVIRONMENT_ID",
    "VISION_OBSERVATION_KEYS",
    "VISION_OBSERVATION_VERSION",
    "EbabilVisionNavigationEnv",
    "SegmentHorizon",
    "SimulationDeadlineScheduler",
    "VisionNavigationEnvConfig",
    "compute_segment_horizon",
    "load_vision_navigation_config",
]
