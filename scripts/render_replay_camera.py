#!/usr/bin/env python3
"""Kaydedilmiş bir replay boyunca DRONE KAMERASINI render et.

`scripts/view_policy_replay.py` sahneyi dışarıdan gösterir. Bu script ise
politikanın gerçekten gördüğü şeyi üretir: actor RGB karesi. İsteğe bağlı
olarak ayrıcalıklı derinlik ve engel maskesi de yazılır; bunlar **yalnız
insan incelemesi** içindir, actor observation'ına girmez.

Kamera geometrisi ADR-011/ADR-016'dır: gövdeye sabit, 45 derece aşağı,
yatay 75 / dikey 60 derece. Görüntünün alt satırları aracın yakınındaki
zemine, üst satırları ileriye karşılık gelir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from ebabil.evaluation import load_replay
from ebabil.perception import HeadlessCamera, load_camera_config
from ebabil.scenes import (
    build_scene_model,
    generate_static_scene,
    load_scene_config,
    load_target_area_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--frames",
        type=int,
        default=12,
        help="Eşit aralıklı kaç kare render edilsin",
    )
    parser.add_argument(
        "--privileged",
        action="store_true",
        help="Derinlik ve engel maskesini de yaz (yalnız insan incelemesi)",
    )
    parser.add_argument(
        "--contact-sheet",
        action="store_true",
        help="Kareleri tek bir ızgara görüntüde birleştir",
    )
    return parser.parse_args()


def _to_rgb(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")


def _depth_to_rgb(depth_m: np.ndarray, maximum_m: float = 8.0) -> Image.Image:
    """Derinliği gri tonlamaya çevir: yakın açık, uzak koyu."""

    finite = np.where(np.isfinite(depth_m), depth_m, maximum_m)
    scaled = 1.0 - np.clip(finite / maximum_m, 0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    return Image.fromarray(np.stack([gray] * 3, axis=-1), mode="RGB")


def _mask_to_rgb(mask: np.ndarray) -> Image.Image:
    value = (np.asarray(mask, dtype=bool) * 255).astype(np.uint8)
    return Image.fromarray(np.stack([value, value // 3, value // 3], -1), mode="RGB")


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames pozitif olmalı")
    replay, metadata = load_replay(args.replay)

    scene_config_path = metadata.get("scene_config")
    if scene_config_path is not None and not isinstance(scene_config_path, str):
        raise ValueError("Replay metadata scene_config değeri geçersiz")
    scene_config = (
        load_scene_config(scene_config_path)
        if scene_config_path
        else load_scene_config()
    )
    target_config = load_target_area_config()
    camera_config = load_camera_config()
    scene = generate_static_scene(
        int(metadata["scene_seed"]),
        scene_config,
        mode=str(metadata["scene_mode"]),
        endpoint_mode=str(metadata.get("endpoint_mode", "fixed")),
        target_config=target_config,
    )
    model = build_scene_model(
        scene,
        camera_config=camera_config,
        target_config=target_config,
        include_legacy_goal_marker=False,
        ground_half_size_m=scene_config.ground_half_size_m,
    )
    if replay.qpos.shape[1] != model.nq:
        raise ValueError("Replay state boyutları sahne modeliyle uyumsuz")

    data = mujoco.MjData(model)
    goal_xy = np.asarray(scene.goal_world_m[:2], dtype=np.float64)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, replay.qpos.shape[0] - 1, args.frames).astype(int)
    tiles: list[Image.Image] = []

    with HeadlessCamera(model, camera_config) as camera:
        for order, index in enumerate(indices):
            data.qpos[:] = replay.qpos[index]
            data.qvel[:] = replay.qvel[index]
            mujoco.mj_forward(model, data)
            frame = camera.capture_actor(data)
            image = _to_rgb(frame.rgb)
            image.save(output / f"{order:02d}_actor.png")
            tiles.append(image)
            distance_m = float(np.linalg.norm(data.qpos[:2] - goal_xy))
            print(
                f"kare {order:02d}: t={replay.time_s[index]:6.2f} s "
                f"irtifa={float(data.qpos[2]):.2f} m hedefe={distance_m:5.2f} m"
            )
            if args.privileged:
                labels = camera.capture_privileged(data)
                _depth_to_rgb(labels.depth_m).save(output / f"{order:02d}_depth.png")
                _mask_to_rgb(labels.obstacle_mask).save(
                    output / f"{order:02d}_obstacles.png"
                )

    if args.contact_sheet and tiles:
        columns = min(4, len(tiles))
        rows = (len(tiles) + columns - 1) // columns
        width, height = tiles[0].size
        sheet = Image.new("RGB", (columns * width, rows * height), (0, 0, 0))
        for order, tile in enumerate(tiles):
            sheet.paste(tile, ((order % columns) * width, (order // columns) * height))
        sheet.save(output / "contact_sheet.png")
        print(f"contact_sheet={output / 'contact_sheet.png'}")

    print(f"output={output}")
    print(f"scene_config={scene_config_path or '(legacy varsayilan)'}")
    print(
        f"frames={len(indices)} camera={camera_config.output_width_px}x"
        f"{camera_config.output_height_px}"
    )


if __name__ == "__main__":
    main()
