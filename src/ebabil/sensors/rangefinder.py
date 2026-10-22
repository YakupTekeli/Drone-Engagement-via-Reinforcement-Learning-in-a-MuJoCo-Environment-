"""MuJoCo ray casting kullanan yatay düzlem rangefinder."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt

from ebabil.sensors.config import RangefinderConfig

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RangefinderReading:
    distances_m: FloatArray
    normalized_distances: FloatArray
    hit_geom_ids: npt.NDArray[np.int32]
    ray_origins_world_m: FloatArray
    ray_directions_world: FloatArray


class PlanarRangefinder:
    """Body `xy` düzleminde eşit açılı ışınlar üretir; ilk ışın body `+x`'tir."""

    def __init__(self, config: RangefinderConfig) -> None:
        config.validate()
        self.config = config
        if config.ray_count == 1:
            angles = np.zeros(1, dtype=np.float64)
        else:
            angles = np.linspace(
                -config.field_of_view_rad / 2.0,
                config.field_of_view_rad / 2.0,
                config.ray_count,
                endpoint=config.field_of_view_rad < 2.0 * np.pi,
                dtype=np.float64,
            )
            if config.field_of_view_rad == 2.0 * np.pi:
                angles = np.arange(config.ray_count, dtype=np.float64) * (
                    2.0 * np.pi / config.ray_count
                )
        self.directions_body = np.column_stack(
            (np.cos(angles), np.sin(angles), np.zeros_like(angles))
        )
        self.geom_group_mask = np.zeros(6, dtype=np.uint8)
        self.geom_group_mask[config.geom_group] = 1

    def read(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        body_name: str = "quadrotor",
    ) -> RangefinderReading:
        body_id = model.body(body_name).id
        rotation = data.xmat[body_id].reshape(3, 3)
        origin = data.xpos[body_id] + rotation @ np.asarray(
            self.config.origin_body_m, dtype=np.float64
        )
        directions_world = (rotation @ self.directions_body.T).T
        distances = np.empty(self.config.ray_count, dtype=np.float64)
        hit_geom_ids = np.full(self.config.ray_count, -1, dtype=np.int32)
        for index, direction in enumerate(directions_world):
            geom_id = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(
                model,
                data,
                origin,
                direction,
                self.geom_group_mask,
                True,
                body_id,
                geom_id,
            )
            if distance < 0.0 or distance > self.config.max_range_m:
                distances[index] = self.config.max_range_m
            else:
                distances[index] = np.clip(
                    distance, self.config.min_range_m, self.config.max_range_m
                )
                hit_geom_ids[index] = geom_id[0]
        normalized = distances / self.config.max_range_m
        origins = np.repeat(origin[np.newaxis, :], self.config.ray_count, axis=0)
        return RangefinderReading(
            distances_m=distances,
            normalized_distances=normalized,
            hit_geom_ids=hit_geom_ids,
            ray_origins_world_m=origins,
            ray_directions_world=directions_world,
        )
