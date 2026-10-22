#!/usr/bin/env mjpython
"""Headless üretilmiş hover manevrasını MuJoCo viewer'da yeniden oynat."""

from __future__ import annotations

import argparse
import math
import os
import platform
import time

import mujoco

from ebabil.models import load_quadrotor
from ebabil.simulation import run_hover_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--roll-deg", type=float, default=12.0)
    parser.add_argument("--offset-x", type=float, default=0.5)
    parser.add_argument("--wind-x", type=float, default=0.0)
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="1.0 gerçek zaman, 0.5 yarı hız, 2.0 iki kat hız",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Replay'i hazırla fakat GUI viewer açma",
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
            "`scripts/run_mjpython.sh scripts/view_hover_demo.py` olarak çalıştırın."
        )

    print("Headless manevra hesaplanıyor...")
    result = run_hover_simulation(
        duration_s=args.duration,
        initial_position_world_m=(args.offset_x, 0.0, 1.0),
        initial_rpy_rad=(math.radians(args.roll_deg), 0.0, 0.0),
        wind_world_m_s=(args.wind_x, 0.0, 0.0),
    )
    model = load_quadrotor()
    data = mujoco.MjData(model)
    if args.dry_run:
        print(f"replay_frames={result.time_s.size}, nq={model.nq}, nv={model.nv}")
        return

    from mujoco import viewer as mj_viewer

    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.cam.trackbodyid = model.body("quadrotor").id
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0
        while viewer.is_running():
            start_wall_time = time.perf_counter()
            for index, simulation_time in enumerate(result.time_s):
                if not viewer.is_running():
                    break
                data.qpos[:] = result.qpos[index]
                data.qvel[:] = result.qvel[index]
                data.time = float(simulation_time)
                mujoco.mj_forward(model, data)
                viewer.sync()
                target_wall_time = (
                    start_wall_time + simulation_time / args.playback_speed
                )
                time.sleep(max(0.0, target_wall_time - time.perf_counter()))
            if not args.loop:
                break


if __name__ == "__main__":
    main()
