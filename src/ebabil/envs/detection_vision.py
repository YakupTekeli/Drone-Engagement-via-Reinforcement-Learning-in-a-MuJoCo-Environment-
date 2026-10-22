"""`vision-v4`: YOLO dedektör kutularıyla GPS-denied yerel navigasyon (ADR-023).

`EbabilVisionNavigationEnv` (`v0`) ile zamanlama, horizon, hover durum
makinesi ve reward yolunu paylaşır; yalnız algı kancalarını değiştirir:

- Onaylı dedektör (`load_approved_detector`) her yakalanan actor karesinde
  çalışır; observation'a yalnız kutu yuvaları (`detections`, 72 değer) girer.
  Engel/hedef koordinatı, depth veya segmentation actor'a girmez.
- Yuvaların menzil alanı kutu alt kenarından zemin düzlemi geometrisiyle
  hesaplanır; attitude ve irtifa `fc_state`'in kullandığı büyüklüklerdir.
- Hover durum makinesinin görsel kapısı dedektörün hedef kutusudur.
  Segmentation tabanlı gözlemci yalnız `info` tanısı içindir.
- Sahne ve kare düzeyinde görsel randomization `v1` ile aynıdır.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from ebabil.controllers import FlightState, read_flight_state
from ebabil.envs.latent_vision import (
    RandomizedActorCamera,
    TargetObserverFactory,
    build_randomized_scene,
)
from ebabil.envs.motion_constraint import (
    apply_speed_limit,
    constrain_horizontal_velocity,
    governor_speed_limit,
    turn_speed_limit,
)
from ebabil.envs.navigation import TerminalDecision
from ebabil.envs.vision import ActorCamera, CameraFactory, EbabilVisionNavigationEnv
from ebabil.envs.vision_config import (
    VisionNavigationEnvConfig,
    load_vision_navigation_config,
)
from ebabil.missions.terminal_assist import assist_action, estimate_target_offset
from ebabil.perception import (
    ActorCameraFrame,
    CameraConfig,
    PrivilegedTargetObserver,
    TargetObserver,
    TargetVisibility,
    VisualDomainConfig,
)
from ebabil.perception.detection_slots import (
    CameraGeometry,
    SlotEncoding,
    encode_detection_slots,
)
from ebabil.perception.detections import Detection
from ebabil.scenes import (
    EndpointMode,
    SceneGeneratorConfig,
    SceneMode,
    StaticScene,
    TargetAreaConfig,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETECTION_VISION_CONFIG = (
    PACKAGE_ROOT / "configs" / "environment" / "vision_navigation_v4.toml"
)
DETECTION_VISION_ENVIRONMENT_ID = "EbabilVisionNavigation-v2"
DETECTION_VISION_OBSERVATION_VERSION = "vision-v4"
#: Sıralı tutulur; `_observation` bunu birebir karşılaştırır.
DETECTION_VISION_OBSERVATION_KEYS = ("detections", "fc_state", "timing", "waypoint")
#: ADR-024: kamera konisi hareket kısıtı açıkken action anlamı değişir.
CONSTRAINED_ACTION_VERSION = "vision-action-v3"


class FrameDetector(Protocol):
    """Actor RGB karesinden kutu listesi üreten nesne (YOLO veya test sahtesi)."""

    def detect(
        self, rgb: npt.NDArray[np.uint8], *, conf: float
    ) -> tuple[Detection, ...]: ...


DetectorFactory = Callable[[], FrameDetector]


class EbabilDetectionVisionNavigationEnv(EbabilVisionNavigationEnv):
    """Dedektör kutu yuvalarıyla yerel waypoint takibi ve kaçınma."""

    observation_version = DETECTION_VISION_OBSERVATION_VERSION
    observation_keys = DETECTION_VISION_OBSERVATION_KEYS

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
        detector_factory: DetectorFactory | None = None,
        target_observer_factory: TargetObserverFactory | None = None,
        visual_domain_split: str | None = None,
        record_physics_trace: bool = False,
        render_mode: str | None = None,
    ) -> None:
        resolved_config = config or load_vision_navigation_config(
            DEFAULT_DETECTION_VISION_CONFIG
        )
        self._detector_factory = detector_factory
        self._target_observer_factory = target_observer_factory
        self._target_observer: TargetObserver | None = None
        self._privileged_visibility = TargetVisibility(False, 0, 0.0, 0.0, 0.0)
        self._visual_domain: VisualDomainConfig | None = None
        self._scene_visual_parameters: dict[str, Any] | None = None
        self._frames_encoded = 0
        self._range_valid_total = 0
        self._obstacles_encoded_total = 0
        self._motion_calls = 0
        self._motion_changed = 0
        self._assist_active = False
        self._assist_steps = 0
        self._policy_steps = 0
        self._governor_limit: float | None = None
        self._last_yaw_rate_command = 0.0
        if resolved_config.motion_constraint.enabled:
            self.action_version = CONSTRAINED_ACTION_VERSION
        randomization = resolved_config.visual_randomization
        if randomization.enabled:
            self._visual_domain = randomization.load_domain(visual_domain_split)
        elif visual_domain_split is not None:
            raise ValueError(
                "visual_domain_split verildi fakat görsel randomization kapalı"
            )
        super().__init__(
            approval_manifest=approval_manifest,
            scene_mode=scene_mode,
            endpoint_mode=endpoint_mode,
            config=resolved_config,
            scene_config=scene_config,
            camera_config=camera_config,
            target_config=target_config,
            camera_factory=camera_factory,
            record_physics_trace=record_physics_trace,
            render_mode=render_mode,
        )
        # Yanlış yapılandırma ilk reset'i beklemeden kurulumda yakalanır.
        self._require_approved_geometries()

    # -- Algı kancaları (ADR-023) ------------------------------------------

    def _load_perception(self) -> None:
        detections = self.vision_config.detections
        self._approved_geometries: tuple[str, ...] | None = None
        if self._detector_factory is not None:
            self.detector: FrameDetector = self._detector_factory()
        else:
            from ebabil.perception.detector import YoloDetector
            from ebabil.perception.detector_quality import verify_detector_approval

            approved = verify_detector_approval(self.approval_manifest)
            self._approved_geometries = approved.obstacle_geometries
            self.detector = YoloDetector(
                approved.weights,
                imgsz=approved.config.imgsz,
                device=detections.device,
                iou=approved.config.inference_iou,
                max_det=approved.config.max_det,
            )
        self._slot_config = detections.slot_config()
        self._detector_conf = min(detections.obstacle_conf, detections.target_conf)

    def _allocate_perception_buffers(self) -> None:
        self._geometry = CameraGeometry.from_camera_config(self.camera_config)
        self._encoding = self._empty_encoding()
        self._last_detections: tuple[Detection, ...] = ()

    def _empty_encoding(self) -> SlotEncoding:
        return encode_detection_slots(
            (),
            config=self._slot_config,
            geometry=self._geometry,
            gravity_direction_body=(0.0, 0.0, -1.0),
            altitude_m=1.0,
        )

    def _perception_spaces(self) -> dict[str, spaces.Space[Any]]:
        return {
            "detections": spaces.Box(
                0.0, 1.0, shape=(self._slot_config.vector_size,), dtype=np.float32
            )
        }

    def _create_actor_camera(self) -> ActorCamera:
        camera = super()._create_actor_camera()
        if self._visual_domain is None:
            return camera
        return RandomizedActorCamera(
            camera, self._visual_domain, scene_seed=self._scene_seed
        )

    def _require_approved_geometries(self) -> None:
        """REV-050: şekil kısıtlı onayda sahne yalnız onaylı şekilleri kullanır."""

        if self._approved_geometries is None:
            return
        requested = set(self.scene_config.obstacle_geometries)
        outside = sorted(requested - set(self._approved_geometries))
        if outside:
            raise ValueError(
                "Dedektör bu engel şekilleri için onaylı değil (REV-050): "
                f"{outside}; onaylı küme {list(self._approved_geometries)}"
            )

    def _build_scene_model(self, scene: StaticScene) -> mujoco.MjModel:
        self._require_approved_geometries()
        self.scene, model, self._scene_visual_parameters = build_randomized_scene(
            scene,
            scene_config=self.scene_config,
            camera_config=self.camera_config,
            target_config=self.target_config,
            visual_domain=self._visual_domain,
        )
        return model

    def _after_simulation_reset(self) -> None:
        assert self.model is not None
        self._close_target_observer()
        if self.vision_config.detections.privileged_diagnostics:
            self._target_observer = (
                self._target_observer_factory(self.model)
                if self._target_observer_factory is not None
                else PrivilegedTargetObserver(
                    self.model,
                    self.camera_config,
                    minimum_pixels=(
                        self.vision_config.privileged_visibility.minimum_target_pixels
                    ),
                )
            )
        self._frames_encoded = 0
        self._range_valid_total = 0
        self._obstacles_encoded_total = 0
        self._motion_calls = 0
        self._motion_changed = 0
        self._assist_active = False
        self._assist_steps = 0
        self._policy_steps = 0
        super()._after_simulation_reset()
        self._observe_privileged_target()

    def _attitude_and_altitude(self) -> tuple[npt.NDArray[np.float64], float]:
        """`fc_state` ile aynı büyüklükler: gövdede yerçekimi yönü ve irtifa."""

        assert self.model is not None and self.data is not None
        state = read_flight_state(self.model, self.data)
        rotation = np.asarray(state.rotation_body_to_world, dtype=np.float64)
        gravity_body = rotation.T @ np.array([0.0, 0.0, -1.0])
        return gravity_body, float(state.position_world_m[2])

    # -- Terminal hover yardımcısı (ADR-025) ------------------------------

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        assist = self.vision_config.terminal_assist
        applied = np.asarray(action, dtype=np.float32)
        self._assist_active = False
        if assist.enabled and str(self.hover_state_machine.state) in assist.states:
            applied = self._terminal_assist_action()
            self._assist_active = True
            self._assist_steps += 1
        self._policy_steps += 1
        self._last_yaw_rate_command = float(
            np.clip(applied[3], -1.0, 1.0) * self.config.action.max_yaw_rate_rad_s
        )
        return super().step(applied)

    def _terminal_assist_action(self) -> npt.NDArray[np.float32]:
        """Dedektör kutusu + FC attitude/irtifa ile klasik yerleşme komutu."""

        assist = self.vision_config.terminal_assist
        gravity_body, altitude_m = self._attitude_and_altitude()
        box = self._encoding.target_box
        estimate = (
            estimate_target_offset(
                box,
                self._geometry,
                assist,
                gravity_direction_body=gravity_body,
                altitude_m=altitude_m,
            )
            if box is not None
            else None
        )
        hover_altitude = float(
            self.target_config.surface_z_m + self.target_config.hover_height_m
        )
        return assist_action(
            estimate,
            altitude_error_m=hover_altitude - altitude_m,
            config=assist,
            max_horizontal_speed_m_s=float(self.config.action.max_horizontal_speed_m_s),
            max_vertical_speed_m_s=float(self.config.action.max_vertical_speed_m_s),
            max_yaw_rate_rad_s=float(self.config.action.max_yaw_rate_rad_s),
        )

    def _constrain_horizontal_body(
        self, horizontal_body: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """ADR-024: seyir/aramada kamera konisine izdüşüm, terminalde sınırlı kör hız."""

        motion = self.vision_config.motion_constraint
        governor = self.vision_config.speed_governor
        if not motion.enabled and not governor.enabled:
            return horizontal_body
        constrained = horizontal_body
        if motion.enabled:
            relaxed = str(self.hover_state_machine.state) in motion.relaxed_states
            constrained = constrain_horizontal_velocity(
                constrained, motion, relaxed=relaxed
            )
        if governor.enabled:
            limit = governor_speed_limit(
                self._encoding.vector,
                governor,
                obstacle_slots=self._slot_config.obstacle_slots,
            )
            turn_limit = turn_speed_limit(
                self._last_yaw_rate_command,
                governor,
                max_yaw_rate_rad_s=float(self.config.action.max_yaw_rate_rad_s),
                max_speed_m_s=float(self.config.action.max_horizontal_speed_m_s),
            )
            if turn_limit is not None:
                limit = turn_limit if limit is None else min(limit, turn_limit)
            self._governor_limit = limit
            constrained = apply_speed_limit(constrained, limit)
        self._motion_calls += 1
        self._motion_changed += int(
            not np.allclose(constrained, horizontal_body, atol=1e-9)
        )
        return constrained

    def _encode_actor_frame(self, frame: ActorCameraFrame) -> None:
        detections = self.detector.detect(frame.rgb, conf=self._detector_conf)
        gravity_body, altitude_m = self._attitude_and_altitude()
        encoding = encode_detection_slots(
            detections,
            config=self._slot_config,
            geometry=self._geometry,
            gravity_direction_body=gravity_body,
            altitude_m=altitude_m,
        )
        if encoding.vector.shape != (self._slot_config.vector_size,) or not np.all(
            np.isfinite(encoding.vector)
        ):
            raise ValueError("Kutu yuvası vektörü vision-v4 sözleşmesini bozuyor")
        self._last_detections = detections
        self._encoding = encoding
        self._frames_encoded += 1
        self._range_valid_total += encoding.range_valid_count
        self._obstacles_encoded_total += encoding.obstacles_encoded

    def _perception_observation(self) -> dict[str, npt.NDArray[np.float32]]:
        return {"detections": self._encoding.vector.copy()}

    def _target_visibility_score(self) -> float:
        """Dedektörün hedef kutusu: `{0, 1}` skoru (ADR-023 §5).

        ADR-017 histerezisi (`0.60 / 0.40`) bu skorda inerttir.
        """

        return 1.0 if self._encoding.target_present else 0.0

    def _target_center_x(self) -> float:
        return float(self._encoding.target_center_x)

    def _observe_privileged_target(self) -> None:
        if self._target_observer is None:
            return
        assert self.data is not None
        self._privileged_visibility = self._target_observer.observe(self.data)

    def _terminal_decision(
        self,
        state: FlightState,
        current_distance_m: float,
        failure_reason: str | None,
    ) -> TerminalDecision:
        # Yalnız tanı: reward ve termination bu okumayı kullanmaz.
        self._observe_privileged_target()
        return super()._terminal_decision(state, current_distance_m, failure_reason)

    def _base_info(self, state: FlightState) -> dict[str, Any]:
        info = super()._base_info(state)
        diagnostics = self._target_observer is not None
        info.update(
            {
                "observation_version": DETECTION_VISION_OBSERVATION_VERSION,
                "action_version": self.action_version,
                "terminal_assist_active": bool(self._assist_active),
                "speed_governor_limit_m_s": self._governor_limit,
                "terminal_assist_fraction": (
                    self._assist_steps / self._policy_steps
                    if self._policy_steps
                    else 0.0
                ),
                "motion_constrained_fraction": (
                    self._motion_changed / self._motion_calls
                    if self._motion_calls
                    else 0.0
                ),
                "detector_obstacles": int(self._encoding.obstacle_detections),
                "detector_obstacles_encoded": int(self._encoding.obstacles_encoded),
                "target_visible_detector": bool(self._encoding.target_present),
                "target_visible_privileged": (
                    bool(self._privileged_visibility.visible) if diagnostics else None
                ),
                "target_pixel_count": (
                    int(self._privileged_visibility.pixel_count)
                    if diagnostics
                    else None
                ),
                "range_valid_fraction": (
                    self._range_valid_total / self._obstacles_encoded_total
                    if self._obstacles_encoded_total
                    else 0.0
                ),
                "visual_domain": (
                    self._visual_domain.name
                    if self._visual_domain is not None
                    else None
                ),
                "scene_visual_parameters": self._scene_visual_parameters,
                "obstacle_geometries": (
                    sorted({obstacle.geometry for obstacle in self.scene.obstacles})
                    if self.scene is not None
                    else []
                ),
            }
        )
        return info

    def _close_target_observer(self) -> None:
        if self._target_observer is not None:
            self._target_observer.close()
            self._target_observer = None

    def _close_additional_resources(self) -> None:
        super()._close_additional_resources()
        self._close_target_observer()


__all__ = [
    "DEFAULT_DETECTION_VISION_CONFIG",
    "DETECTION_VISION_ENVIRONMENT_ID",
    "DETECTION_VISION_OBSERVATION_KEYS",
    "DETECTION_VISION_OBSERVATION_VERSION",
    "DetectorFactory",
    "EbabilDetectionVisionNavigationEnv",
    "FrameDetector",
]
