#!/usr/bin/env python3
"""Kamera pitch adaylarında hedef köşe görünürlüğünü headless hesapla."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace

import mujoco
import numpy as np

from ebabil.perception import CameraConfig, load_camera_config, project_world_points
from ebabil.scenes import (
    build_scene_model,
    generate_static_scene,
    load_target_area_config,
    target_area_from_scene,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument(
        "--pitch-deg", type=float, nargs="+", default=[30, 40, 45, 50, 60]
    )
    parser.add_argument(
        "--approach-distance-m", type=float, nargs="+", default=[6, 4, 2, 0]
    )
    parser.add_argument("--minimum-visible-fraction", type=float, default=0.25)
    parser.add_argument(
        "--require-hold-center",
        action="store_true",
        help=(
            "Hedef merkezinin hover anchor üzerinde de kadrajda olmasını kapıya "
            "dahil et. Sabit montajlı kamerada bu ancak "
            "pitch >= 90 - vfov/2 iken sağlanabilir."
        ),
    )
    return parser.parse_args()


def _nadir_pitch_deg(camera_config: CameraConfig) -> float:
    return math.degrees(camera_config.minimum_pitch_for_nadir_view_rad)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.minimum_visible_fraction <= 1.0:
        raise ValueError("--minimum-visible-fraction [0, 1] aralığında olmalı")
    base_camera = load_camera_config()
    target_config = load_target_area_config()
    scene = generate_static_scene(args.seed, mode="clear")
    target = target_area_from_scene(scene, target_config)
    print(
        "pitch_deg,distance_m,visible_fraction,center_visible,"
        "nadir_in_view,swath_near_m,swath_far_m"
    )
    passing: list[float] = []
    hold_center_pitch: list[float] = []
    hover_altitude_m = float(target.hover_anchor_world_m[2])
    for pitch_deg in args.pitch_deg:
        camera_config = replace(base_camera, pitch_down_rad=math.radians(pitch_deg))
        camera_config.validate()
        near_m, far_m = camera_config.ground_swath_m(hover_altitude_m)
        model = build_scene_model(
            scene,
            camera_config=camera_config,
            include_legacy_goal_marker=False,
        )
        data = mujoco.MjData(model)
        fractions: list[float] = []
        hold_center_visible = True
        for distance_m in args.approach_distance_m:
            data.qpos[:3] = target.hover_anchor_world_m
            data.qpos[0] -= distance_m
            data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
            mujoco.mj_forward(model, data)
            corners = project_world_points(
                model, data, target.corners_world_m, camera_config
            )
            center = project_world_points(
                model,
                data,
                np.array(
                    [
                        [
                            target.center_xy_world_m[0],
                            target.center_xy_world_m[1],
                            target_config.surface_z_m,
                        ]
                    ]
                ),
                camera_config,
            )
            center_visible = bool(center.inside_image[0])
            fractions.append(corners.visible_fraction)
            if distance_m == 0.0:
                hold_center_visible = center_visible
            print(
                f"{pitch_deg:.1f},{distance_m:.2f},"
                f"{corners.visible_fraction:.2f},{center_visible},"
                f"{camera_config.nadir_in_view},{near_m:.3f},{far_m:.3f}"
            )
        if min(fractions) >= args.minimum_visible_fraction:
            passing.append(float(pitch_deg))
            if hold_center_visible:
                hold_center_pitch.append(float(pitch_deg))
    print(f"minimum_pitch_for_nadir_view_deg={_nadir_pitch_deg(base_camera):.2f}")
    print(f"passing_pitch_deg={passing}")
    print(f"passing_pitch_deg_with_hold_center={hold_center_pitch}")
    if not passing:
        raise SystemExit("Hiçbir pitch adayı minimum görünürlük kapısını geçmedi")
    if args.require_hold_center and not hold_center_pitch:
        raise SystemExit(
            "Hiçbir pitch adayı hover anchor üzerinde hedef merkezini kadrajda "
            "tutamadı; REV-014 uyarınca hold koşulu merkez görünürlüğüne "
            "dayandırılamaz"
        )


if __name__ == "__main__":
    main()
