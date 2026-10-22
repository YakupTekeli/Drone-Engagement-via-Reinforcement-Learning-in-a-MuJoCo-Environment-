#!/usr/bin/env mjpython
"""Hedef alanı overview veya drone üzerindeki IMX477 kameradan göster."""

from __future__ import annotations

import argparse
import os
import platform
import time

import mujoco
import numpy as np

from ebabil.perception import load_camera_config, project_world_points
from ebabil.scenes import (
    build_scene_model,
    generate_static_scene,
    load_target_area_config,
    target_area_from_scene,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--pose", choices=("start", "hover"), default="hover")
    parser.add_argument("--camera", choices=("overview", "onboard"), default="overview")
    parser.add_argument(
        "--dry-run", action="store_true", help="Modeli hazırla fakat GUI viewer açma"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        not args.dry_run
        and platform.system() == "Darwin"
        and "MJPYTHON_BIN" not in os.environ
    ):
        raise RuntimeError(
            "macOS'ta `scripts/run_mjpython.sh scripts/view_target_camera.py` kullanın."
        )
    camera_config = load_camera_config()
    target_config = load_target_area_config()
    scene = generate_static_scene(args.seed, mode="clear")
    target = target_area_from_scene(scene, target_config)
    model = build_scene_model(
        scene,
        camera_config=camera_config,
        target_config=target_config,
        include_legacy_goal_marker=False,
    )
    data = mujoco.MjData(model)
    if args.pose == "hover":
        data.qpos[:3] = target.hover_anchor_world_m
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)
    projection = project_world_points(
        model, data, target.corners_world_m, camera_config
    )
    print(
        f"pose={args.pose}, camera={args.camera}, "
        f"visible_fraction={projection.visible_fraction:.2f}, "
        f"safe_half_xy={target_config.safe_half_size_xy_m}"
    )
    if args.dry_run:
        return

    from mujoco import viewer as mj_viewer

    with mj_viewer.launch_passive(model, data) as viewer:
        if args.camera == "onboard":
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = model.camera(camera_config.name).id
        else:
            viewer.cam.lookat[:] = target.hover_anchor_world_m
            viewer.cam.distance = 5.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -35.0
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
