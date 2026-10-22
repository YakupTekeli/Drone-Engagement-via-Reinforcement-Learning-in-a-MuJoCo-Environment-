"""Bir episode'u üç görünümle videoya çevir: drone kamerası (dedektör kutularıyla),
dış takip kamerası ve üstten harita (ADR-023/024/025 demo aracı).

Yalnız inceleme içindir; observation'a giren hiçbir şeyi değiştirmez.

Kullanım:
    .venv/bin/python scripts/render_demo_video.py <checkpoint> <approval_manifest> <output.mp4> --episode-seed 310015 --scene-config src/ebabil/configs/environment/static_scene_d1a.toml --visual-domain-split validation
"""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from ebabil.controllers import read_flight_state
from ebabil.envs import (
    DEFAULT_DETECTION_VISION_CONFIG,
    EbabilDetectionVisionNavigationEnv,
    load_vision_navigation_config,
)
from ebabil.perception.detections import OBSTACLE_CLASS_ID, TARGET_CLASS_ID, draw_boxes
from ebabil.scenes import load_scene_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("approval_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episode-seed", type=int, default=310015)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--visual-domain-split", default="validation")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--terminal-assist", choices=("on", "off"), default="on")
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


FONT = _font(17)
SMALL = _font(14)


def minimap(
    env: EbabilDetectionVisionNavigationEnv, trail: list[np.ndarray], size: int = 480
) -> np.ndarray:
    scene_config = env.scene_config
    lo = np.asarray(scene_config.workspace_min_m[:2])
    hi = np.asarray(scene_config.workspace_max_m[:2])
    scale = size / float(max(hi - lo))

    def to_px(point: np.ndarray) -> tuple[float, float]:
        x = (float(point[0]) - lo[0]) * scale
        y = size - (float(point[1]) - lo[1]) * scale
        return x, y

    image = Image.new("RGB", (size, size), (22, 26, 24))
    draw = ImageDraw.Draw(image)
    assert env.scene is not None
    for obstacle in env.scene.obstacles:
        cx, cy = to_px(obstacle.center_world_m)
        hx, hy = (
            float(obstacle.half_size_m[0]) * scale,
            float(obstacle.half_size_m[1]) * scale,
        )
        if obstacle.geometry == "box":
            draw.rectangle([cx - hx, cy - hy, cx + hx, cy + hy], fill=(150, 60, 50))
        else:
            r = max(hx, hy)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(150, 60, 50))
    area = env.target_area
    gx, gy = to_px(area.hover_anchor_world_m)
    half = 1.0 * scale
    draw.rectangle(
        [gx - half, gy - half, gx + half, gy + half], outline=(40, 200, 90), width=2
    )
    bx, by = to_px(env._believed_target_world_m)
    draw.line([bx - 6, by - 6, bx + 6, by + 6], fill=(230, 200, 60), width=2)
    draw.line([bx - 6, by + 6, bx + 6, by - 6], fill=(230, 200, 60), width=2)
    if len(trail) > 1:
        draw.line([to_px(p) for p in trail], fill=(90, 160, 255), width=2)
    state = read_flight_state(env.model, env.data)
    px, py = to_px(state.position_world_m)
    yaw = math.atan2(
        state.rotation_body_to_world[1, 0], state.rotation_body_to_world[0, 0]
    )
    tip = (px + 12 * math.cos(yaw), py - 12 * math.sin(yaw))
    left = (px + 7 * math.cos(yaw + 2.5), py - 7 * math.sin(yaw + 2.5))
    right = (px + 7 * math.cos(yaw - 2.5), py - 7 * math.sin(yaw - 2.5))
    draw.polygon([tip, left, right], fill=(255, 255, 255))
    draw.text(
        (8, 8),
        "Üstten görünüm: kırmızı engel, yeşil hedef, sarı x inanılan hedef",
        fill=(200, 200, 200),
        font=SMALL,
    )
    return np.asarray(image)


def main() -> None:
    args = parse_args()
    config = load_vision_navigation_config(DEFAULT_DETECTION_VISION_CONFIG)
    config = replace(
        config,
        detections=replace(config.detections, privileged_diagnostics=True),
        terminal_assist=replace(
            config.terminal_assist, enabled=args.terminal_assist == "on"
        ),
    )
    env = EbabilDetectionVisionNavigationEnv(
        approval_manifest=args.approval_manifest,
        scene_mode="random_obstacles",
        endpoint_mode="random_both",
        scene_config=load_scene_config(args.scene_config),
        config=config,
        visual_domain_split=args.visual_domain_split,
    )
    model = PPO.load(str(args.checkpoint), device="cpu")
    captured: list[tuple[np.ndarray, tuple]] = []
    inner = env.detector.detect

    def spy(rgb: np.ndarray, *, conf: float) -> tuple:
        out = inner(rgb, conf=conf)
        captured.append((rgb.copy(), out))
        return out

    env.detector.detect = spy  # type: ignore[method-assign]
    obs, info = env.reset(seed=args.episode_seed)
    chase = mujoco.Renderer(env.model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    frames: list[np.ndarray] = []
    trail: list[np.ndarray] = []

    def compose(info: dict, step: int) -> None:
        rgb, dets = captured[-1]
        keep = [
            d
            for d in dets
            if (
                d.class_id == OBSTACLE_CLASS_ID
                and d.confidence >= config.detections.obstacle_conf
            )
            or (
                d.class_id == TARGET_CLASS_ID
                and d.confidence >= config.detections.target_conf
            )
        ]
        cam = draw_boxes(rgb, keep, thickness=2)
        cam = np.repeat(np.repeat(cam, 2, axis=0), 2, axis=1)
        state = read_flight_state(env.model, env.data)
        trail.append(np.asarray(state.position_world_m[:2]).copy())
        yaw = math.atan2(
            state.rotation_body_to_world[1, 0], state.rotation_body_to_world[0, 0]
        )
        camera.lookat[:] = state.position_world_m
        camera.distance = 4.5
        camera.azimuth = math.degrees(yaw) + 180.0
        camera.elevation = -28.0
        chase.update_scene(env.data, camera=camera)
        chase_rgb = np.asarray(chase.render(), dtype=np.uint8).copy()
        mini = minimap(env, trail)
        frame = np.concatenate(
            [cam, chase_rgb, mini[:480, :480] if mini.shape[0] >= 480 else mini], axis=1
        )
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        speed = float(np.linalg.norm(state.velocity_world_m_s[:2]))
        distance = info.get("hover_horizontal_distance_m")
        limit = info.get("speed_governor_limit_m_s")
        lines = [
            f"adım {step}  durum: {info.get('hover_state', 'transit')}  hedefe yatay mesafe: "
            + ("-" if distance is None else f"{distance:.2f} m"),
            f"hız {speed:.2f} m/s  engel kutusu {info.get('detector_obstacles', 0)}  hedef görüyor: "
            + ("evet" if info.get("target_visible_detector") else "hayır"),
            f"terminal yardımcı: {'AKTİF' if info.get('terminal_assist_active') else 'kapalı'}"
            + "  hız sınırı: "
            + ("-" if limit is None else f"{limit:.2f} m/s")
            + f"  hold {info.get('hover_hold_seconds', 0.0):.1f} s",
        ]
        for i, text in enumerate(lines):
            draw.text((10, 8 + 20 * i), text, fill=(255, 255, 255), font=FONT)
        draw.text(
            (10, 456),
            "sol: drone kamerası + YOLO kutuları (kırmızı engel, yeşil hedef)   orta: dış takip kamerası",
            fill=(220, 220, 220),
            font=SMALL,
        )
        frames.append(np.asarray(image))

    compose(info, 0)
    for step in range(1, args.max_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        compose(info, step)
        if terminated or truncated:
            break
    result = info.get("termination_reason")
    for _ in range(args.fps):
        frames.append(frames[-1])
    chase.close()
    env.close()
    height, width = frames[0].shape[:2]
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        writer = cv2.VideoWriter(
            str(raw), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(raw),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "28",
                "-movflags",
                "+faststart",
                str(args.output),
            ],
            check=True,
        )
    print(
        f"episode_seed={args.episode_seed} steps={len(frames) - args.fps - 1} termination={result} success={info.get('is_success')}"
    )
    print(
        f"video={args.output} frames={len(frames)} size_mb={args.output.stat().st_size / 1e6:.2f}"
    )


if __name__ == "__main__":
    main()
