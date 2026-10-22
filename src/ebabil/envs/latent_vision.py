"""`vision-v3`: etiketsiz VAE latent'iyle GPS-denied yerel navigasyon (ADR-022).

`EbabilVisionNavigationEnv` (`v0`, `vision-v2`) ile zamanlama, horizon, hover
durum makinesi ve reward yolunu paylaşır; yalnız algı kancalarını değiştirir:

- Onaylı etiketsiz VAE (`load_approved_visual_vae`), donmuş; observation'a
  yalnız posterior ortalamasının son `k` policy adımındaki yığını girer.
- Hover durum makinesinin görsel kapısı, ortamın sahip olduğu ayrıcalıklı
  segmentation gözlemcisinden gelir; actor bu bilgiyi görmez.
- Reset'te sahne görünümü ve her karede fotometrik etkiler randomize edilir;
  engel geometrisi sahne config'indeki kümeden örneklenir.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces

from ebabil.controllers import FlightState
from ebabil.envs.navigation import TerminalDecision
from ebabil.envs.vision import (
    ActorCamera,
    CameraFactory,
    EbabilVisionNavigationEnv,
)
from ebabil.envs.vision_config import (
    VisionNavigationEnvConfig,
    load_vision_navigation_config,
)
from ebabil.perception import (
    ActorCameraFrame,
    CameraConfig,
    PrivilegedTargetObserver,
    TargetObserver,
    TargetVisibility,
    VisualDomainConfig,
    VisualVAE,
    apply_frame_visual_parameters_torch,
    apply_scene_visual_parameters,
    load_approved_visual_vae,
    make_noise_bank,
    preprocess_actor_rgb,
    sample_frame_visual_parameters,
    sample_scene_visual_parameters,
)
from ebabil.scenes import (
    EndpointMode,
    SceneGeneratorConfig,
    SceneMode,
    StaticScene,
    TargetAreaConfig,
    randomize_obstacle_geometries,
)
from ebabil.scenes.generator import build_scene_model

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LATENT_VISION_CONFIG = (
    PACKAGE_ROOT / "configs" / "environment" / "vision_navigation_v3.toml"
)
LATENT_VISION_ENVIRONMENT_ID = "EbabilVisionNavigation-v1"
LATENT_VISION_OBSERVATION_VERSION = "vision-v3"
#: Sıralı tutulur; `_observation` bunu birebir karşılaştırır.
LATENT_VISION_OBSERVATION_KEYS = ("fc_state", "timing", "vision_latent", "waypoint")
TargetObserverFactory = Callable[[mujoco.MjModel], TargetObserver]
#: Reset başına üretilen seed'li gürültü alanı sayısı (`240 x 320 x 3` x 16 ~ 15 MB).
NOISE_BANK_SIZE = 16


def build_randomized_scene(
    scene: StaticScene,
    *,
    scene_config: SceneGeneratorConfig,
    camera_config: CameraConfig,
    target_config: TargetAreaConfig,
    visual_domain: VisualDomainConfig | None,
) -> tuple[StaticScene, mujoco.MjModel, dict[str, Any] | None]:
    """ADR-022 §6 sahne çeşitliliği: engel geometrisi + sahne görünümü.

    `v1` ve `v2` ortamları ortak kullanır. Seed'ler `(scene seed, 0x6E0)`
    (geometri, `randomize_obstacle_geometries` içinde) ve `(scene seed, 0xDA)`
    (görünüm) olarak türetilir; aynı seed aynı modeli verir.
    """

    varied = randomize_obstacle_geometries(
        scene, scene_config.obstacle_geometries, scene.seed
    )
    model = build_scene_model(
        varied,
        camera_config=camera_config,
        target_config=target_config,
        include_legacy_goal_marker=False,
        ground_half_size_m=scene_config.ground_half_size_m,
    )
    parameters_payload: dict[str, Any] | None = None
    if visual_domain is not None:
        seed = int(
            np.random.SeedSequence([scene.seed, 0xDA]).generate_state(
                1, dtype=np.uint32
            )[0]
        )
        parameters = sample_scene_visual_parameters(visual_domain, seed)
        # Renderer kurulmadan uygulanmalı; texture ve rgba model üzerinde değişir.
        apply_scene_visual_parameters(model, parameters)
        parameters_payload = asdict(parameters)
    return varied, model, parameters_payload


class RandomizedActorCamera:
    """Actor kamerasını sarar ve her kareye seed'li fotometrik etki uygular.

    Yalnız `capture_actor` sunar; ayrıcalıklı yakalama yolu yoktur. Kare seed'i
    `(scene seed, kare indeksi)`'nden türetilir, yani aynı seed aynı baytları
    verir.
    """

    def __init__(
        self, inner: ActorCamera, domain: VisualDomainConfig, *, scene_seed: int
    ) -> None:
        domain.validate()
        if scene_seed < 0:
            raise ValueError("scene_seed negatif olamaz")
        self._inner = inner
        self._domain = domain
        self._scene_seed = int(scene_seed)
        self._frame_index = 0
        self._noise_bank: torch.Tensor | None = None

    def capture_actor(self, data: mujoco.MjData) -> ActorCameraFrame:
        frame = self._inner.capture_actor(data)
        seed = int(
            np.random.SeedSequence(
                [self._scene_seed, self._frame_index, 0xF0]
            ).generate_state(1, dtype=np.uint32)[0]
        )
        parameters = sample_frame_visual_parameters(self._domain, seed)
        if self._noise_bank is None:
            bank_seed = int(
                np.random.SeedSequence([self._scene_seed, 0xB0]).generate_state(
                    1, dtype=np.uint32
                )[0]
            )
            self._noise_bank = make_noise_bank(
                frame.rgb.shape[0], frame.rgb.shape[1], NOISE_BANK_SIZE, bank_seed
            )
        rgb = apply_frame_visual_parameters_torch(
            frame.rgb, parameters, noise_seed=seed, noise_bank=self._noise_bank
        )
        self._frame_index += 1
        return ActorCameraFrame(rgb=rgb, timestamp_s=frame.timestamp_s)

    def close(self) -> None:
        self._inner.close()


class EbabilLatentVisionNavigationEnv(EbabilVisionNavigationEnv):
    """Donmuş etiketsiz VAE latent yığınıyla yerel waypoint takibi ve kaçınma."""

    observation_version = LATENT_VISION_OBSERVATION_VERSION
    observation_keys = LATENT_VISION_OBSERVATION_KEYS

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
        target_observer_factory: TargetObserverFactory | None = None,
        visual_domain_split: str | None = None,
        record_physics_trace: bool = False,
        render_mode: str | None = None,
    ) -> None:
        resolved_config = config or load_vision_navigation_config(
            DEFAULT_LATENT_VISION_CONFIG
        )
        self._target_observer_factory = target_observer_factory
        self._target_observer: TargetObserver | None = None
        self._target_visibility = TargetVisibility(False, 0, 0.0, 0.0, 0.0)
        self._visual_domain: VisualDomainConfig | None = None
        self._visual_domain_split = visual_domain_split
        self._scene_visual_parameters: dict[str, Any] | None = None
        self._latent_clip_count = 0
        self._latent_clip_total = 0
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

    # -- Algı kancaları (ADR-022) ------------------------------------------

    def _load_perception(self) -> None:
        self.vae: VisualVAE = load_approved_visual_vae(self.approval_manifest)
        if self.vae.config.latent_size != self.vision_config.observation.latent_size:
            raise ValueError(
                "Onaylı VAE latent boyutu "
                f"{LATENT_VISION_OBSERVATION_VERSION} config'iyle eşleşmiyor: "
                f"{self.vae.config.latent_size} != "
                f"{self.vision_config.observation.latent_size}"
            )
        self.vae.eval()
        self.vae.requires_grad_(False)

    def _allocate_perception_buffers(self) -> None:
        observation = self.vision_config.observation
        self._latent_current = np.zeros(observation.latent_size, dtype=np.float64)
        self._latent_stack = np.zeros(
            (observation.latent_stack_size, observation.latent_size), dtype=np.float64
        )

    def _perception_spaces(self) -> dict[str, spaces.Space[Any]]:
        observation = self.vision_config.observation
        return {
            "vision_latent": spaces.Box(
                -1.0,
                1.0,
                shape=(observation.latent_stack_size * observation.latent_size,),
                dtype=np.float32,
            )
        }

    def _create_actor_camera(self) -> ActorCamera:
        camera = super()._create_actor_camera()
        if self._visual_domain is None:
            return camera
        return RandomizedActorCamera(
            camera, self._visual_domain, scene_seed=self._scene_seed
        )

    def _build_scene_model(self, scene: StaticScene) -> mujoco.MjModel:
        # Tabanın sakladığı sahne, MJCF'e giden sahneyle aynı olmalı.
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
        super()._after_simulation_reset()
        # Yığın ilk latent ile KOPYALANARAK dolar (ADR-022 §4); sıfırla doldurmak
        # sahte bir "hareket" sinyali üretirdi.
        self._latent_stack[:] = self._latent_current
        self._latent_clip_count = 0
        self._latent_clip_total = 0
        self._observe_target()

    def _encode_actor_frame(self, frame: ActorCameraFrame) -> None:
        image = preprocess_actor_rgb(frame.rgb).unsqueeze(0)
        with torch.inference_mode():
            mean, _ = self.vae.encode(image)
        latent = mean[0].detach().cpu().numpy().astype(np.float64)
        if latent.shape != self._latent_current.shape:
            raise ValueError("VAE latent'i vision-v3 shape'iyle eşleşmiyor")
        if not np.all(np.isfinite(latent)):
            raise ValueError("VAE latent'i yalnız sonlu değerler içermeli")
        self._latent_current[:] = latent

    def _perception_observation(self) -> dict[str, npt.NDArray[np.float32]]:
        # Her reset/step'te tam bir kez çağrılır: en yeni latent 0. sıraya girer.
        self._latent_stack[1:] = self._latent_stack[:-1]
        self._latent_stack[0] = self._latent_current
        scaled = self._latent_stack / self.vision_config.observation.latent_scale
        self._latent_clip_count += int(np.count_nonzero(np.abs(scaled) > 1.0))
        self._latent_clip_total += int(scaled.size)
        return {
            "vision_latent": np.clip(scaled, -1.0, 1.0).reshape(-1).astype(np.float32)
        }

    def _target_visibility_score(self) -> float:
        """Ayrıcalıklı gözlemcinin boolean görünürlüğü; `{0, 1}` skoru.

        ADR-017 histerezisi (`0.60 / 0.40`) bu skorda inerttir: `1.0` her zaman
        acquire eşiğinin üstünde, `0.0` her zaman loss eşiğinin altındadır.
        """

        return self._target_visibility.score

    def _target_center_x(self) -> float:
        return float(self._target_visibility.center_x)

    def _observe_target(self) -> None:
        assert self._target_observer is not None
        assert self.data is not None
        self._target_visibility = self._target_observer.observe(self.data)

    def _terminal_decision(
        self,
        state: FlightState,
        current_distance_m: float,
        failure_reason: str | None,
    ) -> TerminalDecision:
        # Policy adımı başına tek segmentation render'ı; reward yolu aynı
        # okumayı kullanır.
        self._observe_target()
        return super()._terminal_decision(state, current_distance_m, failure_reason)

    def _base_info(self, state: FlightState) -> dict[str, Any]:
        info = super()._base_info(state)
        info.update(
            {
                "observation_version": LATENT_VISION_OBSERVATION_VERSION,
                "latent_stack_size": self.vision_config.observation.latent_stack_size,
                "latent_clip_fraction": (
                    self._latent_clip_count / self._latent_clip_total
                    if self._latent_clip_total
                    else 0.0
                ),
                "target_visible_privileged": bool(self._target_visibility.visible),
                "target_pixel_count": int(self._target_visibility.pixel_count),
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
    "DEFAULT_LATENT_VISION_CONFIG",
    "LATENT_VISION_ENVIRONMENT_ID",
    "LATENT_VISION_OBSERVATION_KEYS",
    "LATENT_VISION_OBSERVATION_VERSION",
    "EbabilLatentVisionNavigationEnv",
    "RandomizedActorCamera",
    "build_randomized_scene",
    "TargetObserverFactory",
]
