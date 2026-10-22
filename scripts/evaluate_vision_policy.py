#!/usr/bin/env python3
"""Eğitilmiş vision politikasını ADR-017 terminal hover metrikleriyle değerlendir.

İki değerlendirme kipi vardır ve ikisi farklı şey ölçer:

`in_distribution`
    Eğitim sahnesinin kendisi. `clear` + `fixed` sahnesi seed'den bağımsız
    olarak aynıdır, bu nedenle bu kip bir genelleme ölçümü **değildir**;
    politikanın terminal hover görevini öğrenip öğrenmediğini ölçer.
    Episode'lar arası fark yalnız stokastik policy örneklemesinden gelir.

`encoder_ood`
    Değişken başlangıç noktası. Hedef aynı dünya XY'sinde kalır fakat yaklaşma
    yönü değişir. Onaylı `vision-encoder-v9` poz örneklemesi tek rota etrafında
    `+-1.25 m` yanal offset ve `+-35 derece` yaw jitter kapsadığı için bu kip
    kısmen encoder'ın eğitim dağılımı dışındadır. Bir kalite kapısı değil,
    yetkinlik zarfının nereye kadar uzandığını ölçen tanısal bir problardır.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from ebabil.envs import (
    DEFAULT_DETECTION_VISION_CONFIG,
    EbabilDetectionVisionNavigationEnv,
    EbabilLatentVisionNavigationEnv,
    EbabilVisionNavigationEnv,
    load_vision_navigation_config,
)
from ebabil.envs.vision import PHYSICAL_FAILURE_REASONS
from ebabil.scenes import load_scene_config

SCHEMA_VERSION = "vision-policy-evaluation-v2"
MODES = ("in_distribution", "encoder_ood")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("approval_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=MODES, default="in_distribution")
    parser.add_argument(
        "--scene-config",
        type=Path,
        help="Sahne config yolu (D1 ve sonrası için zorunlu; yoksa legacy sahne)",
    )
    parser.add_argument(
        "--scene-mode",
        choices=("clear", "single_blocker", "random_obstacles"),
        default="clear",
    )
    parser.add_argument(
        "--endpoint-mode",
        choices=("fixed", "random_start", "random_goal", "random_both"),
        help="Verilmezse kip'e göre seçilir",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=310001)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--env-version",
        choices=("v0", "v1", "v2"),
        default="v0",
        help=(
            "v0: EbabilVisionNavigation-v0 (vision-v2); v1: vision-v3 (ADR-022); "
            "v2: vision-v4, YOLO dedektör (ADR-023)"
        ),
    )
    parser.add_argument(
        "--terminal-assist",
        choices=("on", "off"),
        default=None,
        help="Yalnız v2: ADR-025 terminal hover yardımcısını config'ten bağımsız aç/kapat",
    )
    parser.add_argument(
        "--visual-domain-split",
        default=None,
        help="Yalnız v1/v2: görsel randomization alanı (train | validation | ...)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Deterministik policy çıktısı kullan (varsayılan stokastik)",
    )
    return parser.parse_args()


def _endpoint_mode(mode: str) -> str:
    return "fixed" if mode == "in_distribution" else "random_start"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("episodes ve max-steps pozitif olmalı")
    model = PPO.load(str(args.checkpoint.expanduser().resolve()), device=args.device)
    scene_config = (
        load_scene_config(args.scene_config) if args.scene_config is not None else None
    )
    endpoint_mode = args.endpoint_mode or _endpoint_mode(args.mode)
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
            endpoint_mode=endpoint_mode,
            scene_config=scene_config,
            visual_domain_split=args.visual_domain_split,
        )
    elif args.env_version == "v1":
        env = EbabilLatentVisionNavigationEnv(
            approval_manifest=args.approval_manifest,
            scene_mode=args.scene_mode,
            endpoint_mode=endpoint_mode,
            scene_config=scene_config,
            visual_domain_split=args.visual_domain_split,
        )
    else:
        if args.visual_domain_split is not None:
            raise ValueError("--visual-domain-split yalnız --env-version v1/v2 ile")
        env = EbabilVisionNavigationEnv(
            approval_manifest=args.approval_manifest,
            scene_mode=args.scene_mode,
            endpoint_mode=endpoint_mode,
            scene_config=scene_config,
        )
    approach_radius_m = float(env.vision_config.terminal_hover.approach_radius_m)
    rows: list[dict[str, Any]] = []
    try:
        for index in range(args.episodes):
            episode_seed = args.seed + index
            observation, info = env.reset(seed=episode_seed)
            episode_return = 0.0
            steps = 0
            settle_step: int | None = None
            first_hold_step: int | None = None
            minimum_range_m = float("inf")
            minimum_anchor_distance_m = float("inf")
            states: Counter[str] = Counter()
            terminated = truncated = False
            for step in range(args.max_steps):
                action, _ = model.predict(observation, deterministic=args.deterministic)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                steps = step + 1
                state = str(info["hover_state"])
                states[state] += 1
                if state == "settle" and settle_step is None:
                    settle_step = steps
                if state == "hold" and first_hold_step is None:
                    first_hold_step = steps
                minimum_range_m = min(minimum_range_m, float(info["minimum_range_m"]))
                minimum_anchor_distance_m = min(
                    minimum_anchor_distance_m,
                    float(info["hover_horizontal_distance_m"]),
                )
                if terminated or truncated:
                    break
            # ADR-022 §10: varış = gerçek hover anchor'a yatay mesafe yaklaşma
            # yarıçapına inmiş VE episode fiziksel failure ile bitmemiş.
            arrived = (
                minimum_anchor_distance_m <= approach_radius_m
                and info["termination_reason"] not in PHYSICAL_FAILURE_REASONS
            )
            rows.append(
                {
                    "episode": index,
                    "episode_seed": episode_seed,
                    "steps": steps,
                    "episode_return": episode_return,
                    "is_success": bool(info["is_success"]),
                    "termination_reason": info["termination_reason"],
                    "hover_state": str(info["hover_state"]),
                    "hover_abort_reason": info["hover_abort_reason"],
                    "hover_hold_seconds": float(info["hover_hold_seconds"]),
                    "hover_search_seconds": float(info["hover_search_seconds"]),
                    "target_error_m": float(info["target_error_m"]),
                    "route_length_m": float(info["planned_route_length_m"]),
                    "minimum_range_m": minimum_range_m,
                    "minimum_anchor_distance_m": minimum_anchor_distance_m,
                    "arrived": bool(arrived),
                    "settle_step": settle_step,
                    "first_hold_step": first_hold_step,
                    "step_cap_reached": not terminated and not truncated,
                    "state_counts": dict(sorted(states.items())),
                }
            )
    finally:
        env.close()

    successes = [row for row in rows if row["is_success"]]
    reasons = Counter(str(row["termination_reason"]) for row in rows)
    aborts = Counter(
        str(row["hover_abort_reason"])
        for row in rows
        if row["hover_abort_reason"] is not None
    )
    settle_steps = [row["settle_step"] for row in rows if row["settle_step"]]
    hold_steps = [row["first_hold_step"] for row in rows if row["first_hold_step"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "env_version": args.env_version,
        "observation_version": env.observation_version,
        "visual_domain_split": args.visual_domain_split,
        "mode": args.mode,
        "scene_mode": args.scene_mode,
        "scene_config": str(args.scene_config) if args.scene_config else None,
        "endpoint_mode": endpoint_mode,
        "generalization_claim": args.mode == "encoder_ood",
        "deterministic": bool(args.deterministic),
        "checkpoint": str(args.checkpoint),
        "approval_manifest": str(args.approval_manifest),
        "episodes": args.episodes,
        "aggregate": {
            "success_rate": len(successes) / len(rows),
            "arrival_rate": sum(1 for row in rows if row["arrived"]) / len(rows),
            "collision_rate": sum(
                1
                for row in rows
                if row["termination_reason"]
                in ("obstacle_collision", "ground_collision")
            )
            / len(rows),
            "timeout_rate": reasons["time_limit"] / len(rows),
            "abort_rate": sum(aborts.values()) / len(rows),
            "step_cap_rate": sum(1 for row in rows if row["step_cap_reached"])
            / len(rows),
            "mean_episode_return": float(
                np.mean([row["episode_return"] for row in rows])
            ),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
            "minimum_range_m": min(row["minimum_range_m"] for row in rows),
            "mean_settle_step": float(np.mean(settle_steps)) if settle_steps else None,
            "mean_first_hold_step": (
                float(np.mean(hold_steps)) if hold_steps else None
            ),
            "termination_reasons": dict(sorted(reasons.items())),
            "abort_reasons": dict(sorted(aborts.items())),
            "mean_search_seconds": float(
                np.mean([row["hover_search_seconds"] for row in rows])
            ),
            "mean_target_error_m": float(
                np.mean([row["target_error_m"] for row in rows])
            ),
            "mean_route_length_m": float(
                np.mean([row["route_length_m"] for row in rows])
            ),
        },
        "episodes_detail": rows,
    }


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    aggregate = summary["aggregate"]
    print(
        f"mode={summary['mode']} scene_mode={summary['scene_mode']} "
        f"endpoint_mode={summary['endpoint_mode']}"
    )
    print(f"success_rate={aggregate['success_rate']:.3f}")
    print(f"arrival_rate={aggregate['arrival_rate']:.3f}")
    print(f"collision_rate={aggregate['collision_rate']:.3f}")
    print(f"abort_rate={aggregate['abort_rate']:.3f}")
    print(f"timeout_rate={aggregate['timeout_rate']:.3f}")
    print(f"termination_reasons={aggregate['termination_reasons']}")
    print(f"abort_reasons={aggregate['abort_reasons']}")
    print(f"mean_episode_return={aggregate['mean_episode_return']:.3f}")
    print(f"mean_route_length_m={aggregate['mean_route_length_m']:.2f}")
    print(f"mean_target_error_m={aggregate['mean_target_error_m']:.3f}")
    print(f"mean_search_seconds={aggregate['mean_search_seconds']:.2f}")
    print(f"summary={output_path}")
    if summary["mode"] == "in_distribution":
        print(
            "not: in_distribution kipi tek deterministik sahnedir ve genelleme "
            "ölçümü değildir"
        )


if __name__ == "__main__":
    main()
