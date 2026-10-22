#!/usr/bin/env python3
"""`vision-v4` ortamında check_env, random rollout, PPO ve deterministik evaluation smoke'u.

Onaylı YOLO dedektör manifestiyle çalışır (ADR-023). Ortam hızını (policy
adımı/s) ve dedektör-gerçek hedef görünürlük uyumunu raporlar; ADR-023
Dilim C kapısı `v0`'ın en az `0.5` katıdır.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ebabil.envs import (
    DETECTION_VISION_OBSERVATION_VERSION,
    EbabilDetectionVisionNavigationEnv,
)
from ebabil.scenes import load_scene_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("approval_manifest", type=Path)
    parser.add_argument(
        "--mode",
        choices=("clear", "single_blocker", "random_obstacles"),
        default="random_obstacles",
    )
    parser.add_argument(
        "--endpoint-mode",
        choices=("fixed", "random_start", "random_goal", "random_both"),
        default="random_both",
    )
    parser.add_argument("--scene-config", type=Path)
    parser.add_argument("--visual-domain-split", default=None)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--random-steps", type=int, default=200)
    parser.add_argument("--ppo-timesteps", type=int, default=64)
    parser.add_argument("--evaluation-steps", type=int, default=80)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> EbabilDetectionVisionNavigationEnv:
    return EbabilDetectionVisionNavigationEnv(
        approval_manifest=args.approval_manifest,
        scene_mode=args.mode,
        endpoint_mode=args.endpoint_mode,
        scene_config=(
            load_scene_config(args.scene_config)
            if args.scene_config is not None
            else None
        ),
        visual_domain_split=args.visual_domain_split,
    )


def random_policy_smoke(args: argparse.Namespace) -> dict[str, object]:
    env = make_env(args)
    rng = np.random.default_rng(args.seed)
    episode_count = 0
    completed_steps = 0
    observed_states: Counter[str] = Counter()
    range_fractions: list[float] = []
    obstacle_counts: list[int] = []
    agreements: list[bool] = []
    last_info: dict[str, object] = {}
    try:
        check_env(env, warn=True, skip_render_check=True)
        observation, last_info = env.reset(seed=args.seed)
        if not env.observation_space.contains(observation):
            raise RuntimeError("Reset observation space dışında")
        started = time.perf_counter()
        for step in range(args.random_steps):
            action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(
                np.float32
            )
            observation, reward, terminated, truncated, last_info = env.step(action)
            if not env.observation_space.contains(observation):
                raise RuntimeError("Step observation space dışında")
            if not np.isfinite(reward):
                raise RuntimeError("Random-policy reward sonlu değil")
            completed_steps += 1
            observed_states[str(last_info["hover_state"])] += 1
            obstacle_counts.append(int(last_info["detector_obstacles"]))
            if last_info["target_visible_privileged"] is not None:
                agreements.append(
                    bool(last_info["target_visible_detector"])
                    == bool(last_info["target_visible_privileged"])
                )
            if terminated or truncated:
                episode_count += 1
                range_fractions.append(float(last_info["range_valid_fraction"]))
                observation, last_info = env.reset(seed=args.seed + step + 1)
        elapsed = time.perf_counter() - started
        range_fractions.append(float(last_info["range_valid_fraction"]))
        return {
            "completed_steps": completed_steps,
            "completed_episodes": episode_count,
            "steps_per_second": completed_steps / elapsed,
            "last_camera_frame_count": int(last_info["camera_frame_count"]),
            "range_valid_fraction_mean": float(np.mean(range_fractions)),
            "detector_obstacles_mean": float(np.mean(obstacle_counts)),
            "target_visibility_agreement": (
                float(np.mean(agreements)) if agreements else None
            ),
            "visual_domain": last_info["visual_domain"],
            "obstacle_geometries": last_info["obstacle_geometries"],
            "hover_state_counts": dict(sorted(observed_states.items())),
        }
    finally:
        env.close()


def ppo_and_evaluation_smoke(
    args: argparse.Namespace,
) -> tuple[PPO, dict[str, object]]:
    def monitored_env() -> Monitor:
        return Monitor(make_env(args))

    vector_env = DummyVecEnv([monitored_env])
    model = PPO(
        "MultiInputPolicy",
        vector_env,
        seed=args.seed,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        policy_kwargs={"net_arch": [32, 32]},
        device="cpu",
        verbose=0,
    )
    try:
        model.learn(total_timesteps=args.ppo_timesteps)
    finally:
        vector_env.close()

    evaluation_env = make_env(args)
    episode_return = 0.0
    try:
        observation, info = evaluation_env.reset(seed=args.seed + 100_000)
        terminated = truncated = False
        executed_steps = 0
        for _ in range(args.evaluation_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = evaluation_env.step(
                action
            )
            episode_return += reward
            executed_steps += 1
            if terminated or truncated:
                break
        return model, {
            "executed_steps": executed_steps,
            "episode_return": episode_return,
            "terminated": terminated,
            "truncated": truncated,
            "termination_reason": info["termination_reason"],
            "hover_state": str(info["hover_state"]),
            "hover_horizontal_distance_m": float(info["hover_horizontal_distance_m"]),
            "is_success": bool(info["is_success"]),
        }
    finally:
        evaluation_env.close()


def main() -> None:
    args = parse_args()
    if min(args.random_steps, args.ppo_timesteps, args.evaluation_steps) <= 0:
        raise ValueError("Smoke adım sayıları pozitif olmalı")
    started_s = time.perf_counter()
    print("stage=random_policy_and_check_env started", flush=True)
    random_result = random_policy_smoke(args)
    print(f"stage=random_policy_and_check_env completed {random_result}", flush=True)
    print("stage=ppo_and_deterministic_evaluation started", flush=True)
    model, evaluation_result = ppo_and_evaluation_smoke(args)
    summary = {
        "schema_version": "detection-vision-navigation-smoke-v1",
        "observation_version": DETECTION_VISION_OBSERVATION_VERSION,
        "approval_manifest": str(args.approval_manifest.resolve()),
        "scene_mode": args.mode,
        "seed": args.seed,
        "elapsed_seconds": time.perf_counter() - started_s,
        "random_policy": random_result,
        "deterministic_evaluation": evaluation_result,
    }
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save(output_dir / "smoke_model")
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
