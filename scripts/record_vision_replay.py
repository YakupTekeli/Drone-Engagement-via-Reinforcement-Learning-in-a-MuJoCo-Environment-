#!/usr/bin/env python3
"""Eğitilmiş vision politikasından MuJoCo viewer için replay kaydı üret.

Kayıt `500 Hz` fizik çözünürlüğündedir ve `scripts/view_policy_replay.py` ile
oynatılır. Terminal hover metrikleri metadata'ya yazılır.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from ebabil.envs import (
    DEFAULT_DETECTION_VISION_CONFIG,
    EbabilDetectionVisionNavigationEnv,
    EbabilLatentVisionNavigationEnv,
    EbabilVisionNavigationEnv,
    load_vision_navigation_config,
)
from ebabil.evaluation import save_replay
from ebabil.evaluation.replay import EpisodeReplay
from ebabil.scenes import load_scene_config

TERMINAL_HOVER_COMPONENT_NAMES = (
    "target_progress",
    "target_alignment",
    "settle_speed",
    "hover_hold",
    "proximity",
    "smoothness",
    "collision",
    "out_of_bounds",
    "success",
    "abort",
    "time",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("approval_manifest", type=Path)
    parser.add_argument("output", type=Path, help="`.npz` yolu")
    parser.add_argument("--seed", type=int, default=310001)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Stokastik policy kullan (varsayılan deterministik)",
    )
    parser.add_argument(
        "--scene-config",
        default=None,
        help=(
            "Sahne config yolu; verilmezse legacy workspace. D1 için: "
            "src/ebabil/configs/environment/static_scene_d1.toml"
        ),
    )
    parser.add_argument(
        "--scene-mode",
        choices=("clear", "single_blocker", "random_obstacles"),
        default="clear",
    )
    parser.add_argument(
        "--endpoint-mode",
        choices=("fixed", "random_start", "random_goal", "random_both"),
        default="fixed",
    )
    parser.add_argument("--env-version", choices=("v0", "v1", "v2"), default="v0")
    parser.add_argument(
        "--terminal-assist",
        choices=("on", "off"),
        default=None,
        help="Yalnız v2: ADR-025 terminal hover yardımcısını config'ten bağımsız aç/kapat",
    )
    parser.add_argument("--visual-domain-split", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps pozitif olmalı")
    model = PPO.load(str(args.checkpoint.expanduser().resolve()), device=args.device)
    scene_config = (
        load_scene_config(args.scene_config) if args.scene_config is not None else None
    )
    env: EbabilVisionNavigationEnv
    if args.env_version == "v2":
        v4_config = load_vision_navigation_config(DEFAULT_DETECTION_VISION_CONFIG)
        # Evaluation dedektör-gerçek hedef görünürlük uyumunu raporlar (ADR-023 §10).
        v4_config = replace(
            v4_config,
            detections=replace(v4_config.detections, privileged_diagnostics=True),
            terminal_assist=replace(
                v4_config.terminal_assist,
                enabled=(
                    v4_config.terminal_assist.enabled
                    if args.terminal_assist is None
                    else args.terminal_assist == "on"
                ),
            ),
        )
        env = EbabilDetectionVisionNavigationEnv(
            config=v4_config,
            approval_manifest=args.approval_manifest,
            scene_mode=args.scene_mode,
            endpoint_mode=args.endpoint_mode,
            scene_config=scene_config,
            visual_domain_split=args.visual_domain_split,
            record_physics_trace=True,
        )
    elif args.env_version == "v1":
        env = EbabilLatentVisionNavigationEnv(
            approval_manifest=args.approval_manifest,
            scene_mode=args.scene_mode,
            endpoint_mode=args.endpoint_mode,
            scene_config=scene_config,
            visual_domain_split=args.visual_domain_split,
            record_physics_trace=True,
        )
    else:
        env = EbabilVisionNavigationEnv(
            approval_manifest=args.approval_manifest,
            scene_mode=args.scene_mode,
            endpoint_mode=args.endpoint_mode,
            scene_config=scene_config,
            record_physics_trace=True,
        )
    try:
        observation, info = env.reset(seed=args.seed)
        policy_time_s: list[float] = []
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        distances: list[float] = [float(info["distance_to_goal_m"])]
        minimum_ranges: list[float] = []
        components: list[list[float]] = []
        hover_states: list[str] = []
        terminated = truncated = False
        for _ in range(args.max_steps):
            action, _ = model.predict(observation, deterministic=not args.stochastic)
            clipped = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
            assert env.data is not None
            policy_time_s.append(float(env.data.time))
            observation, reward, terminated, truncated, info = env.step(action)
            actions.append(clipped)
            rewards.append(float(reward))
            distances.append(float(info["distance_to_goal_m"]))
            minimum_ranges.append(float(info["minimum_range_m"]))
            components.append(
                [
                    float(info["reward_components"][name])
                    for name in TERMINAL_HOVER_COMPONENT_NAMES
                ]
            )
            hover_states.append(str(info["hover_state"]))
            if terminated or truncated:
                break

        trace = env.physics_trace()
        replay = EpisodeReplay(
            time_s=trace.time_s,
            qpos=trace.qpos,
            qvel=trace.qvel,
            policy_time_s=np.asarray(policy_time_s, dtype=np.float64),
            actions=np.asarray(actions, dtype=np.float64).reshape(-1, env.ACTION_SIZE),
            rewards=np.asarray(rewards, dtype=np.float64),
            distance_to_goal_m=np.asarray(distances, dtype=np.float64),
            minimum_range_m=np.asarray(minimum_ranges, dtype=np.float64),
            reward_components=np.asarray(components, dtype=np.float64).reshape(
                -1, len(TERMINAL_HOVER_COMPONENT_NAMES)
            ),
            rotor_thrusts_n=trace.rotor_thrusts_n,
            velocity_command_world_m_s=trace.velocity_command_world_m_s,
            yaw_rate_command_rad_s=trace.yaw_rate_command_rad_s,
            mixer_saturated=trace.mixer_saturated,
            torque_saturated=trace.torque_saturated,
        )
        replay_file, metadata_file = save_replay(
            args.output,
            replay,
            {
                "schema_version": 1,
                "model_variant": "vision",
                "observation_version": env.observation_version,
                "env_version": args.env_version,
                "visual_domain_split": args.visual_domain_split,
                "scene_mode": args.scene_mode,
                "endpoint_mode": args.endpoint_mode,
                "scene_config": args.scene_config,
                "scene_seed": int(info["scene_seed"]),
                "episode_seed": args.seed,
                "deterministic": not args.stochastic,
                "checkpoint": str(args.checkpoint),
                "approval_manifest": str(args.approval_manifest),
                "termination_reason": info["termination_reason"],
                "is_success": bool(info["is_success"]),
                "hover_state": str(info["hover_state"]),
                "hover_hold_seconds": float(info["hover_hold_seconds"]),
                "hover_abort_reason": info["hover_abort_reason"],
                "policy_steps": len(actions),
                "episode_return": float(sum(rewards)),
                "reward_component_names": list(TERMINAL_HOVER_COMPONENT_NAMES),
                "hover_state_sequence_tail": hover_states[-10:],
            },
        )
    finally:
        env.close()

    print(f"termination_reason={info['termination_reason']}")
    print(f"is_success={bool(info['is_success'])}")
    print(f"hover_hold_seconds={float(info['hover_hold_seconds']):.2f}")
    print(f"policy_steps={len(actions)}  physics_frames={trace.time_s.size}")
    print(f"episode_return={float(sum(rewards)):+.3f}")
    print(f"replay={replay_file}")
    print(f"metadata={metadata_file}")


if __name__ == "__main__":
    main()
