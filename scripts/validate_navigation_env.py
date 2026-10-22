#!/usr/bin/env python3
"""Uzun random-action rollout ile Gym ortamının sayısal kararlılığını doğrula."""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np

from ebabil.envs import EbabilNavigationEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--mode",
        choices=("clear", "single_blocker", "random_obstacles"),
        default="random_obstacles",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps pozitif olmalı")
    env = EbabilNavigationEnv(scene_mode=args.mode)
    action_rng = np.random.default_rng(args.seed)
    episode_index = 0
    reasons: Counter[str] = Counter()
    observation, _ = env.reset(seed=args.seed, options={"scene_seed": args.seed})
    start_time = time.perf_counter()
    try:
        for _ in range(args.steps):
            action = action_rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            if not env.observation_space.contains(observation):
                raise RuntimeError("Observation space dışı veya geçersiz observation")
            if not np.isfinite(reward):
                raise RuntimeError("Reward sonlu değil")
            if terminated or truncated:
                reasons[str(info["termination_reason"])] += 1
                episode_index += 1
                episode_seed = args.seed + episode_index
                observation, _ = env.reset(
                    seed=episode_seed,
                    options={"scene_seed": episode_seed},
                )
    finally:
        env.close()
    elapsed_s = time.perf_counter() - start_time
    print(f"mode={args.mode}, seed={args.seed}")
    print(f"policy_steps={args.steps}, episodes_completed={sum(reasons.values())}")
    print(f"termination_reasons={dict(sorted(reasons.items()))}")
    print(f"elapsed_s={elapsed_s:.6f}, policy_steps_per_s={args.steps / elapsed_s:.2f}")
    print("finite_observations_and_rewards=true")


if __name__ == "__main__":
    main()
