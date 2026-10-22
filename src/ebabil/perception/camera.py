"""MuJoCo IMX477 projection ve actor/privileged capture sınırı."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Self

import mujoco
import numpy as np
import numpy.typing as npt

from ebabil.perception.config import CameraConfig, load_camera_config

FloatArray = npt.NDArray[np.float64]
UInt8Array = npt.NDArray[np.uint8]
Int32Array = npt.NDArray[np.int32]
BoolArray = npt.NDArray[np.bool_]


class CameraRenderUnavailable(RuntimeError):
    """Makinede offscreen grafik bağlamı kurulamadığında üretilir."""


@dataclass(frozen=True)
class ActorCameraFrame:
    """Deploy actor'a verilebilen kamera yükü; privileged alan içermez."""

    rgb: UInt8Array
    timestamp_s: float

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(cls))


@dataclass(frozen=True)
class PrivilegedCameraLabels:
    """Yalnız simülasyon teacher, reward ve değerlendirme etiketleri."""

    depth_m: FloatArray
    segmentation: Int32Array
    target_mask: BoolArray
    obstacle_mask: BoolArray
    timestamp_s: float


@dataclass(frozen=True)
class ProjectionResult:
    pixels_xy: FloatArray
    depth_m: FloatArray
    in_front: BoolArray
    inside_image: BoolArray

    @property
    def visible_fraction(self) -> float:
        return float(np.mean(self.inside_image)) if self.inside_image.size else 0.0


def project_world_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    points_world_m: npt.ArrayLike,
    config: CameraConfig | None = None,
) -> ProjectionResult:
    """Dünya noktalarını MuJoCo camera frame üzerinden piksele projekte et."""

    config = config or load_camera_config()
    config.validate()
    points = np.asarray(points_world_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points_world_m sonlu ve shape (N, 3) olmalı")
    camera_id = model.camera(config.name).id
    rotation_camera_to_world = data.cam_xmat[camera_id].reshape(3, 3)
    camera_position_world = data.cam_xpos[camera_id]
    points_camera = (points - camera_position_world) @ rotation_camera_to_world
    depth = -points_camera[:, 2]
    in_front = depth > 1e-9
    pixels = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    focal = config.focal_length_px
    center_x, center_y = config.principal_point_px
    pixels[in_front, 0] = (
        center_x + focal * points_camera[in_front, 0] / depth[in_front]
    )
    pixels[in_front, 1] = (
        center_y - focal * points_camera[in_front, 1] / depth[in_front]
    )
    inside = (
        in_front
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < config.output_width_px)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < config.output_height_px)
    )
    return ProjectionResult(pixels, depth, in_front, inside)


class HeadlessCamera:
    """RGB actor frame'ini privileged render kanallarından ayrı üretir."""

    def __init__(
        self,
        model: mujoco.MjModel,
        config: CameraConfig | None = None,
        *,
        target_geom_prefix: str = "target_area",
        obstacle_geom_prefix: str = "obstacle_",
    ) -> None:
        self.model = model
        self.config = config or load_camera_config()
        self.config.validate()
        model.camera(self.config.name)
        if any(
            model.geom(index).name == "target_marker_geom"
            for index in range(model.ngeom)
        ):
            raise ValueError(
                "Vision actor modelinde legacy target_marker_geom bulunamaz"
            )
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
        self._obstacle_geom_ids = np.array(
            [
                index
                for index in range(model.ngeom)
                if model.geom(index).name.startswith(obstacle_geom_prefix)
            ],
            dtype=np.int32,
        )
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

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._renderer.close()

    def capture_actor(self, data: mujoco.MjData) -> ActorCameraFrame:
        """Yalnız RGB ve zaman damgası içeren deploy edilebilir frame üret."""

        self._renderer.disable_depth_rendering()
        self._renderer.disable_segmentation_rendering()
        self._renderer.update_scene(data, camera=self.config.name)
        rgb = np.asarray(self._renderer.render(), dtype=np.uint8).copy()
        return ActorCameraFrame(rgb=rgb, timestamp_s=float(data.time))

    def capture_privileged(self, data: mujoco.MjData) -> PrivilegedCameraLabels:
        """Depth, raw segmentation ve hedef maskesini actor dışında üret."""

        self._renderer.disable_segmentation_rendering()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(data, camera=self.config.name)
        depth = np.asarray(self._renderer.render(), dtype=np.float64).copy()
        self._renderer.disable_depth_rendering()
        self._renderer.enable_segmentation_rendering()
        self._renderer.update_scene(data, camera=self.config.name)
        segmentation = np.asarray(self._renderer.render(), dtype=np.int32).copy()
        self._renderer.disable_segmentation_rendering()
        if segmentation.shape != (
            self.config.output_height_px,
            self.config.output_width_px,
            2,
        ):
            raise RuntimeError(f"Beklenmeyen segmentation shape: {segmentation.shape}")
        geom_ids = segmentation[..., 0]
        object_types = segmentation[..., 1]
        target_mask = (object_types == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(
            geom_ids, self._target_geom_ids
        )
        obstacle_mask = (object_types == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(
            geom_ids, self._obstacle_geom_ids
        )
        return PrivilegedCameraLabels(
            depth_m=depth,
            segmentation=segmentation,
            target_mask=target_mask,
            obstacle_mask=obstacle_mask,
            timestamp_s=float(data.time),
        )
