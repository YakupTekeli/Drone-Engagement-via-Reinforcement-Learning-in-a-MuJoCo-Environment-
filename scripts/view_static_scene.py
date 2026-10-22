#!/usr/bin/env mjpython
"""Seed tabanlı engel sahnesini MuJoCo viewer'da göster."""

from __future__ import annotations

import argparse
import os
import platform
import time

import mujoco
import numpy as np

from ebabil.scenes import (
    build_scene_model,
    generate_static_scene,
    load_scene_config,
    load_target_area_config,
    target_area_from_scene,
)
from ebabil.sensors import NavigationSensorSuite, load_sensor_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--mode",
        choices=("clear", "single_blocker", "random_obstacles"),
        default="single_blocker",
        help="Gösterilecek curriculum sahne seviyesi",
    )
    parser.add_argument(
        "--scene-config",
        default=None,
        help=(
            "Sahne config yolu. Verilmezse legacy workspace kullanılır "
            "(6 x 5 x 2.5 m). D1 için: "
            "src/ebabil/configs/environment/static_scene_d1.toml"
        ),
    )
    parser.add_argument(
        "--endpoint-mode",
        choices=("fixed", "random_start", "random_goal", "random_both"),
        default="fixed",
        help="D1 eğitimi random_both kullanır",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Kaç ardışık seed gösterilsin; viewer'da N tuşuyla sonrakine geç",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sahneyi hazırla fakat GUI viewer açma",
    )
    return parser.parse_args()


def _describe(scene: object, target_config: object) -> None:
    """Sahnenin gözle kıyaslanabilir sayılarını yazdır."""

    import numpy as np

    route_m = float(np.linalg.norm(scene.goal_world_m - scene.start_world_m))
    target = target_area_from_scene(scene, target_config)
    print(
        f"seed={scene.seed} mode={scene.mode} "
        f"obstacle_count={len(scene.obstacles)} rota={route_m:.2f} m"
    )
    print(
        f"  start={np.round(scene.start_world_m, 2)} "
        f"goal={np.round(scene.goal_world_m, 2)}"
    )
    print(
        f"  hedef alanı yarı-boyut={target.config.half_size_xy_m} "
        f"güvenli iç bölge={target.config.safe_half_size_xy_m} "
        f"hover anchor={np.round(target.hover_anchor_world_m, 2)}"
    )
    if scene.obstacles:
        rota = scene.goal_world_m[:2] - scene.start_world_m[:2]
        uzunluk = float(np.linalg.norm(rota))
        birim = rota / uzunluk if uzunluk > 0 else rota
        print("  engeller (rota eksenine göre):")
        for i, ob in enumerate(scene.obstacles):
            merkez = np.asarray(ob.center_world_m[:2], dtype=np.float64)
            delta = merkez - scene.start_world_m[:2]
            boyuna = float(np.dot(delta, birim))
            enine = float(np.linalg.norm(delta - boyuna * birim))
            hedefe = float(np.linalg.norm(merkez - scene.goal_world_m[:2]))
            isaret = ""
            if enine <= 1.0 and 0.0 <= boyuna <= uzunluk:
                isaret = "  <- rota üzerinde"
            elif hedefe <= 2.5:
                isaret = "  <- hedefe yakın"
            print(
                f"    {i:>2d}: yarı-boyut={np.round(ob.half_size_m, 2)} "
                f"rota boyunca={boyuna:+.2f} m, rotadan sapma={enine:.2f} m, "
                f"hedefe={hedefe:.2f} m{isaret}"
            )


def main() -> None:
    args = parse_args()
    if (
        not args.dry_run
        and platform.system() == "Darwin"
        and "MJPYTHON_BIN" not in os.environ
    ):
        raise RuntimeError(
            "macOS'ta `scripts/run_mjpython.sh scripts/view_static_scene.py` kullanın."
        )
    scene_config = (
        load_scene_config(args.scene_config)
        if args.scene_config is not None
        else load_scene_config()
    )
    target_config = load_target_area_config()
    scenes = [
        generate_static_scene(
            args.seed + offset,
            scene_config,
            mode=args.mode,
            endpoint_mode=args.endpoint_mode,
            target_config=target_config,
        )
        for offset in range(max(1, args.seeds))
    ]
    for scene in scenes:
        _describe(scene, target_config)
        print()

    scene = scenes[0]
    model = build_scene_model(scene, ground_half_size_m=scene_config.ground_half_size_m)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sensor_suite = NavigationSensorSuite(load_sensor_config())
    reading = sensor_suite.reset(model, data, seed=args.seed)
    print(f"rangefinder_m={np.round(reading.rangefinder_m, 3)}")
    if args.dry_run:
        return

    from mujoco import viewer as mj_viewer

    workspace = np.asarray(scene_config.workspace_max_m, dtype=np.float64) - np.asarray(
        scene_config.workspace_min_m, dtype=np.float64
    )
    with mj_viewer.launch_passive(model, data) as viewer:
        midpoint = (scene.start_world_m + scene.goal_world_m) / 2.0
        viewer.cam.lookat[:] = midpoint
        viewer.cam.distance = float(1.2 * max(workspace[0], workspace[1]))
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -50.0
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
