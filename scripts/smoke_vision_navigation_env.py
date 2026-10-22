#!/usr/bin/env python3
"""Vision navigation ortamında random, PPO ve deterministic evaluation smoke'u."""

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

from ebabil.envs import VISION_OBSERVATION_VERSION, EbabilVisionNavigationEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("approval_manifest", type=Path)
    parser.add_argument(
        "--mode",
        choices=("clear", "single_blocker", "random_obstacles"),
        default="random_obstacles",
    )
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--random-steps", type=int, default=100)
    parser.add_argument("--ppo-timesteps", type=int, default=64)
    parser.add_argument("--evaluation-steps", type=int, default=80)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> EbabilVisionNavigationEnv:
    return EbabilVisionNavigationEnv(
        approval_manifest=args.approval_manifest,
        scene_mode=args.mode,
    )


def random_policy_smoke(args: argparse.Namespace) -> dict[str, object]:
    env = make_env(args)
    rng = np.random.default_rng(args.seed)
    episode_count = 0
    completed_steps = 0
    observed_states: Counter[str] = Counter()
    last_info: dict[str, object] = {}
    try:
        check_env(env, warn=True, skip_render_check=True)
        observation, last_info = env.reset(seed=args.seed)
        if not env.observation_space.contains(observation):
            raise RuntimeError("Reset observation space dışında")
        for step in range(args.random_steps):
            # Boyut action space'ten okunur; ADR-020 ile 3 -> 4 oldu.
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
            if terminated or truncated:
                episode_count += 1
                observation, last_info = env.reset(seed=args.seed + step + 1)
        return {
            "completed_steps": completed_steps,
            "completed_episodes": episode_count,
            "last_camera_frame_count": int(last_info["camera_frame_count"]),
            "last_camera_frame_age_s": float(last_info["camera_frame_age_s"]),
            "hover_state_counts": dict(sorted(observed_states.items())),
        }
    finally:
        env.close()


def ppo_and_evaluation_smoke(args: argparse.Namespace) -> tuple[PPO, dict[str, object]]:
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
    component_totals: dict[str, float] = {}
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
            for name, value in info["reward_components"].items():
                component_totals[name] = component_totals.get(name, 0.0) + float(value)
            if terminated or truncated:
                break
        return model, {
            "executed_steps": executed_steps,
            "episode_return": episode_return,
            "terminated": terminated,
            "truncated": truncated,
            "termination_reason": info["termination_reason"],
            "distance_to_goal_m": float(info["distance_to_goal_m"]),
            "evaluation_cap_reached": not terminated and not truncated,
            "hover_state": str(info["hover_state"]),
            "hover_hold_seconds": float(info["hover_hold_seconds"]),
            "hover_abort_reason": info["hover_abort_reason"],
            "is_success": bool(info["is_success"]),
            "reward_component_totals": {
                name: round(value, 6)
                for name, value in sorted(component_totals.items())
            },
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
    random_elapsed_s = time.perf_counter() - started_s
    print(
        f"stage=random_policy_and_check_env completed elapsed_s={random_elapsed_s:.3f}",
        flush=True,
    )
    ppo_started_s = time.perf_counter()
    print("stage=ppo_and_deterministic_evaluation started", flush=True)
    model, evaluation_result = ppo_and_evaluation_smoke(args)
    ppo_elapsed_s = time.perf_counter() - ppo_started_s
    print(
        f"stage=ppo_and_deterministic_evaluation completed elapsed_s={ppo_elapsed_s:.3f}",
        flush=True,
    )
    summary = {
        "schema_version": "vision-navigation-smoke-v1",
        "observation_version": VISION_OBSERVATION_VERSION,
        "approval_manifest": str(args.approval_manifest.resolve()),
        "scene_mode": args.mode,
        "seed": args.seed,
        "ppo_timesteps": args.ppo_timesteps,
        "elapsed_seconds": {
            "random_policy_and_check_env": random_elapsed_s,
            "ppo_and_deterministic_evaluation": ppo_elapsed_s,
            "total": time.perf_counter() - started_s,
        },
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
