"""Ortamın sahip olduğu ayrıcalıklı hedef görünürlük gözlemcisi (ADR-022 §5).

`vision-v3` ortamında hover durum makinesinin görsel kapısı etiketsiz
kodlayıcıdan değil, bu gözlemcinin segmentation maskesinden hesaplanan hedef
piksel sayısından gelir. Gözlemci actor kamerasından **ayrı** bir nesnedir,
yalnız reward/termination yolunda kullanılır ve actor observation'ına
girmez; `ActorCameraFrame` yalnız `("rgb", "timestamp_s")` taşımaya devam
eder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np

from ebabil.perception.camera import CameraRenderUnavailable
from ebabil.perception.config import CameraConfig, load_camera_config


@dataclass(frozen=True)
class TargetVisibility:
    """Tek policy adımının ayrıcalıklı hedef görünürlük okuması."""

    visible: bool
    pixel_count: int
    #: Görünen hedef piksellerinin ağırlık merkezi, `[-1, 1]` görüntü
    #: koordinatı; görünmüyorsa `(0.0, 0.0)` sentinel'i (`derive_supervision`
    #: ile aynı kural).
    center_x: float
    center_y: float
    timestamp_s: float

    @property
    def score(self) -> float:
        """Durum makinesi girdisi: boolean görünürlüğün `{0, 1}` skoru."""

        return 1.0 if self.visible else 0.0


class TargetObserver(Protocol):
    def observe(self, data: mujoco.MjData) -> TargetVisibility: ...

    def close(self) -> None: ...


class PrivilegedTargetObserver:
    """Segmentation render'ından hedef piksel sayısı; actor yolundan erişilemez."""

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_config: CameraConfig | None = None,
        *,
        minimum_pixels: int,
        target_geom_prefix: str = "target_area",
    ) -> None:
        if minimum_pixels <= 0:
            raise ValueError("minimum_pixels pozitif olmalı")
        self.config = camera_config or load_camera_config()
        self.config.validate()
        self.minimum_pixels = int(minimum_pixels)
        model.camera(self.config.name)
        self._target_geom_ids = np.array(
            [
                index
                for index in range(model.ngeom)
                if model.geom(index).name.startswith(target_geom_prefix)
            ],
            dtype=np.int32,
        )
        if self._target_geom_ids.size == 0:
            raise ValueError(f"Hedef geom prefix'i bulunamadı: {target_geom_prefix}")
        try:
            self._renderer = mujoco.Renderer(
                model,
                height=self.config.output_height_px,
                width=self.config.output_width_px,
            )
        except Exception as error:
            raise CameraRenderUnavailable(
                "MuJoCo offscreen renderer başlatılamadı; macOS grafik testi "
                "sandbox dışında veya mjpython süreciyle çalıştırılmalı"
            ) from error
        self._renderer.enable_segmentation_rendering()

    def observe(self, data: mujoco.MjData) -> TargetVisibility:
        self._renderer.update_scene(data, camera=self.config.name)
        segmentation = np.asarray(self._renderer.render(), dtype=np.int32)
        mask = (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(
            segmentation[..., 0], self._target_geom_ids
        )
        rows, columns = np.nonzero(mask)
        count = int(columns.size)
        visible = count >= self.minimum_pixels
        if visible:
            height, width = mask.shape
            center_x = 2.0 * float(np.mean(columns)) / max(width - 1, 1) - 1.0
            center_y = 2.0 * float(np.mean(rows)) / max(height - 1, 1) - 1.0
        else:
            center_x = center_y = 0.0
        return TargetVisibility(
            visible=visible,
            pixel_count=count,
            center_x=center_x,
            center_y=center_y,
            timestamp_s=float(data.time),
        )

    def close(self) -> None:
        self._renderer.close()


__all__ = ["PrivilegedTargetObserver", "TargetObserver", "TargetVisibility"]
