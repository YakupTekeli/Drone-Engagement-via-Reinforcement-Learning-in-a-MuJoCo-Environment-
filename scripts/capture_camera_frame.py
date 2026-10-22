#!/usr/bin/env python3
"""IMX477 simülasyonundan actor RGB ve privileged etiket smoke capture al."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from ebabil.perception import HeadlessCamera, load_camera_config, project_world_points
from ebabil.scenes import (
    build_scene_model,
    generate_static_scene,
    load_target_area_config,
    target_area_from_scene,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--pose", choices=("start", "hover"), default="start")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--save-prefix",
        type=Path,
        help="Ayrı <prefix>.actor.npz ve <prefix>.privileged.npz yaz",
    )
    output_group.add_argument(
        "--save-npz",
        dest="save_prefix",
        type=Path,
        help="Geriye uyumlu alias; birleşik değil iki ayrı NPZ yazar",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Renderer açmadan projection özeti üret"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        f"pose={args.pose}, resolution={camera_config.output_width_px}x"
        f"{camera_config.output_height_px}, "
        f"visible_fraction={projection.visible_fraction:.2f}"
    )
    if args.dry_run:
        return
    with HeadlessCamera(model, camera_config) as camera:
        actor = camera.capture_actor(data)
        privileged = camera.capture_privileged(data)
    print(
        f"rgb_shape={actor.rgb.shape}, depth_shape={privileged.depth_m.shape}, "
        f"target_pixels={int(np.count_nonzero(privileged.target_mask))}"
    )
    if args.save_prefix is not None:
        output = args.save_prefix.expanduser().resolve()
        if output.suffix == ".npz":
            output = output.with_suffix("")
        actor_output = output.with_name(output.name + ".actor.npz")
        privileged_output = output.with_name(output.name + ".privileged.npz")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            actor_output,
            sample_id=np.asarray(f"smoke-{args.seed}-{args.pose}"),
            rgb=actor.rgb,
            timestamp_s=np.array(actor.timestamp_s),
        )
        np.savez_compressed(
            privileged_output,
            sample_id=np.asarray(f"smoke-{args.seed}-{args.pose}"),
            depth_m=privileged.depth_m,
            segmentation=privileged.segmentation,
            target_mask=privileged.target_mask,
            obstacle_mask=privileged.obstacle_mask,
            timestamp_s=np.array(privileged.timestamp_s),
        )
        print(f"actor_saved={actor_output}")
        print(f"privileged_saved={privileged_output}")


if __name__ == "__main__":
    main()
