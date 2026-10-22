#!/usr/bin/env python3
"""Navigasyon ortamında kısa, headless scripted-policy koşusu."""

from __future__ import annotations

import argparse

import numpy as np

from ebabil.envs import EbabilNavigationEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("clear", "single_blocker", "random_obstacles"),
        default="clear",
    )
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument(
        "--sensor-mode", choices=("ideal", "imperfect"), default="ideal"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = EbabilNavigationEnv(
        scene_mode=args.mode,
        sensor_mode=args.sensor_mode,
        max_episode_steps=args.steps,
    )
    observation, info = env.reset(seed=args.seed, options={"scene_seed": args.seed})
    episode_return = 0.0
    terminated = truncated = False
    try:
        for _ in range(args.steps):
            assert env.scene is not None
            assert env.data is not None
            delta = env.scene.goal_world_m - env.data.qpos[:3]
            action = np.array(
                [
                    np.clip(0.4 * delta[0], -1.0, 1.0),
                    np.clip(0.4 * delta[1], -1.0, 1.0),
                    np.clip(0.6 * delta[2], -1.0, 1.0),
                    0.0,
                ],
                dtype=np.float32,
            )
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            if terminated or truncated:
                break
        print(f"observation_shape={observation.shape}")
        print(f"scene_mode={info['scene_mode']}, scene_seed={info['scene_seed']}")
        print(f"episode_steps={info['episode_step']}")
        print(f"episode_return={episode_return:.6f}")
        print(f"distance_to_goal_m={info['distance_to_goal_m']:.6f}")
        print(f"terminated={terminated}, truncated={truncated}")
        print(f"termination_reason={info['termination_reason']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
