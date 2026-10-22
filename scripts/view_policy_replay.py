#!/usr/bin/env mjpython
"""Kaydedilmiş bir PPO evaluation episode'unu MuJoCo viewer'da oynat."""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path
from typing import cast

import mujoco
import numpy as np

from ebabil.evaluation import load_replay
from ebabil.perception import load_camera_config
from ebabil.scenes import (
    EndpointMode,
    SceneMode,
    build_scene_model,
    generate_static_scene,
    load_scene_config,
    load_target_area_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="1.0 gerçek zaman, 0.5 yarı hız, 2.0 iki kat hız",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--camera",
        default="track",
        help=(
            "Başlangıç kamerası: `track` (aracı izleyen serbest kamera) veya "
            "modeldeki bir kamera adı (`overview`, `imx477`). `imx477` "
            "drone'un kendi kamerasıdır (ADR-011, 45 derece aşağı). Viewer "
            "içinde `[` / `]` ile de geçilebilir; Türkçe klavyede bu tuşlar "
            "`ğ` ve `ü` üzerindedir."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Replay ve sahneyi doğrula fakat GUI viewer açma",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.playback_speed <= 0.0:
        raise ValueError("--playback-speed pozitif olmalı")
    if (
        not args.dry_run
        and platform.system() == "Darwin"
        and "MJPYTHON_BIN" not in os.environ
    ):
        raise RuntimeError(
            "macOS MuJoCo viewer ana thread ister. Komutu "
            "`scripts/run_mjpython.sh scripts/view_policy_replay.py <replay>` "
            "olarak çalıştırın."
        )

    replay, metadata = load_replay(args.replay)
    scene_mode_value = metadata.get("scene_mode")
    if scene_mode_value not in ("clear", "single_blocker", "random_obstacles"):
        raise ValueError("Replay metadata scene_mode değeri geçersiz")
    scene_seed = metadata.get("scene_seed")
    if isinstance(scene_seed, bool) or not isinstance(scene_seed, int):
        raise ValueError("Replay metadata scene_seed değeri geçersiz")
    scene_mode = cast(SceneMode, scene_mode_value)
    endpoint_mode_value = metadata.get("endpoint_mode", "fixed")
    if endpoint_mode_value not in (
        "fixed",
        "random_start",
        "random_goal",
        "random_both",
    ):
        raise ValueError("Replay metadata endpoint_mode değeri geçersiz")
    endpoint_mode = cast(EndpointMode, endpoint_mode_value)
    model_variant = metadata.get("model_variant", "legacy")
    if model_variant not in ("legacy", "vision"):
        raise ValueError("Replay metadata model_variant değeri geçersiz")
    target_config = load_target_area_config() if model_variant == "vision" else None
    # Replay, kaydedildiği sahne config'iyle yeniden üretilmek ZORUNDA. Aksi
    # halde varsayılan (legacy 6 x 5 m) workspace çizilir, araç kadraj dışında
    # kalır ve engel/hedef yerleşimi yanlış görünür.
    scene_config_path = metadata.get("scene_config")
    if scene_config_path is not None and not isinstance(scene_config_path, str):
        raise ValueError("Replay metadata scene_config değeri geçersiz")
    scene_config = (
        load_scene_config(scene_config_path)
        if scene_config_path
        else load_scene_config()
    )
    scene = generate_static_scene(
        scene_seed,
        scene_config,
        mode=scene_mode,
        endpoint_mode=endpoint_mode,
        target_config=target_config,
    )
    if model_variant == "vision":
        # Vision görevinde hedef zemindeki alandır; legacy nokta işaretçisi
        # yanıltıcı olacağı için çizilmez.
        model = build_scene_model(
            scene,
            camera_config=load_camera_config(),
            include_legacy_goal_marker=False,
            ground_half_size_m=scene_config.ground_half_size_m,
        )
    else:
        model = build_scene_model(
            scene, ground_half_size_m=scene_config.ground_half_size_m
        )
    data = mujoco.MjData(model)
    if replay.qpos.shape[1] != model.nq or replay.qvel.shape[1] != model.nv:
        raise ValueError("Replay state boyutları sahne modeliyle uyumsuz")
    # Yanlış sahne config'iyle oynatmayı sessizce geçirmemek için: kaydedilen
    # yörünge bu workspace'in içinde olmalı.
    minimum = np.min(replay.qpos[:, :3], axis=0)
    maximum = np.max(replay.qpos[:, :3], axis=0)
    lower = np.asarray(scene_config.workspace_min_m, dtype=np.float64) - 1.0
    upper = np.asarray(scene_config.workspace_max_m, dtype=np.float64) + 1.0
    if np.any(minimum < lower) or np.any(maximum > upper):
        raise ValueError(
            "Replay yörüngesi bu sahne config'inin workspace'i dışında: "
            f"yörünge {np.round(minimum, 2)}..{np.round(maximum, 2)}, "
            f"workspace {scene_config.workspace_min_m}..{scene_config.workspace_max_m}. "
            "Replay farklı bir scene_config ile kaydedilmiş olabilir"
        )

    camera_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
        for index in range(model.ncam)
    ]
    if args.camera != "track" and args.camera not in camera_names:
        raise ValueError(
            f"Bilinmeyen kamera: {args.camera}. "
            f"Seçenekler: track, {', '.join(str(n) for n in camera_names)}"
        )

    if args.dry_run:
        print(
            f"replay_frames={replay.time_s.size}, nq={model.nq}, nv={model.nv}, "
            f"mode={scene_mode}, variant={model_variant}, "
            f"scene_seed={scene_seed}, "
            f"endpoint_mode={endpoint_mode}, "
            f"camera={args.camera}, "
            f"reason={metadata.get('termination_reason')}"
        )
        return

    from mujoco import viewer as mj_viewer

    with mj_viewer.launch_passive(model, data) as viewer:
        if args.camera == "track":
            viewer.cam.trackbodyid = model.body("quadrotor").id
            viewer.cam.distance = 5.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -25.0
        else:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera
            )
        while viewer.is_running():
            start_wall_time = time.perf_counter()
            replay_start_time = float(replay.time_s[0])
            for index, simulation_time in enumerate(replay.time_s):
                if not viewer.is_running():
                    break
                data.qpos[:] = replay.qpos[index]
                data.qvel[:] = replay.qvel[index]
                data.time = float(simulation_time)
                mujoco.mj_forward(model, data)
                viewer.sync()
                elapsed_simulation = float(simulation_time) - replay_start_time
                target_wall_time = (
                    start_wall_time + elapsed_simulation / args.playback_speed
                )
                time.sleep(max(0.0, target_wall_time - time.perf_counter()))
            if not args.loop:
                break


if __name__ == "__main__":
    main()
