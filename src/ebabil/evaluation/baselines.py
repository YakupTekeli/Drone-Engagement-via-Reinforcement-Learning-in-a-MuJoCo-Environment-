"""PPO kalite kapısı için random ve doğrudan-hedef baseline'ları."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from ebabil.controllers import read_flight_state
from ebabil.envs import EbabilNavigationEnv
from ebabil.training.config import TrainingConfig, require_legacy_environment

BaselineName = Literal["random", "direct_goal"]


class BaselinePolicy(Protocol):
    def reset(self, seed: int) -> None: ...

    def predict(
        self,
        env: EbabilNavigationEnv,
        observation: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]: ...


class RandomPolicy:
    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def predict(
        self,
        env: EbabilNavigationEnv,
        observation: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        del env, observation
        return self._rng.uniform(-1.0, 1.0, size=4).astype(np.float32)


class DirectGoalPolicy:
    """Privileged world-state kullanan, engelden kaçınmayan P-controller."""

    def __init__(self, position_gain_s: float = 0.6) -> None:
        if not np.isfinite(position_gain_s) or position_gain_s <= 0.0:
            raise ValueError("position_gain_s pozitif ve sonlu olmalı")
        self.position_gain_s = position_gain_s

    def reset(self, seed: int) -> None:
        del seed

    def predict(
        self,
        env: EbabilNavigationEnv,
        observation: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        del observation
        if env.model is None or env.data is None or env.scene is None:
            raise RuntimeError("DirectGoalPolicy öncesinde environment reset gerekli")
        state = read_flight_state(env.model, env.data)
        delta_world = env.scene.goal_world_m - state.position_world_m
        delta_body = state.rotation_body_to_world.T @ delta_world
        action_config = env.config.action
        action = np.array(
            [
                self.position_gain_s
                * delta_body[0]
                / action_config.max_horizontal_speed_m_s,
                self.position_gain_s
                * delta_body[1]
                / action_config.max_horizontal_speed_m_s,
                self.position_gain_s
                * delta_world[2]
                / action_config.max_vertical_speed_m_s,
                0.0,
            ],
            dtype=np.float32,
        )
        return np.clip(action, -1.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class BaselineArtifacts:
    output_directory: Path
    summary: Path
    metrics: dict[str, Any]


def make_baseline_policy(name: BaselineName) -> BaselinePolicy:
    if name == "random":
        return RandomPolicy()
    if name == "direct_goal":
        return DirectGoalPolicy()
    raise ValueError(f"Bilinmeyen baseline: {name}")


def evaluate_baseline(
    name: BaselineName,
    config: TrainingConfig,
    output_directory: str | Path,
    episode_seeds: tuple[int, ...],
) -> BaselineArtifacts:
    """Bir baseline'ı verilen kesin held-out seed listesinde değerlendir."""

    config.validate()
    if not episode_seeds or len(set(episode_seeds)) != len(episode_seeds):
        raise ValueError("episode_seeds boş olmayan benzersiz bir liste olmalı")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in episode_seeds
    ):
        raise ValueError("episode seed'leri negatif olmayan tamsayı olmalı")
    output_path = Path(output_directory).expanduser().resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Baseline dizini boş değil: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    policy = make_baseline_policy(name)
    sensor_mode, max_episode_steps = require_legacy_environment(config)
    env = EbabilNavigationEnv(
        scene_mode=config.environment.scene_mode,
        endpoint_mode=config.environment.endpoint_mode,
        sensor_mode=sensor_mode,
        max_episode_steps=max_episode_steps,
    )
    episode_rows: list[dict[str, Any]] = []
    try:
        for episode_index, episode_seed in enumerate(episode_seeds):
            observation, reset_info = env.reset(seed=episode_seed)
            policy.reset(episode_seed)
            assert env.data is not None
            positions = [env.data.qpos[:3].copy()]
            rewards: list[float] = []
            minimum_ranges: list[float] = []
            terminated = truncated = False
            final_info = reset_info
            while not (terminated or truncated):
                action = policy.predict(env, observation)
                observation, reward, terminated, truncated, final_info = env.step(
                    action
                )
                assert env.data is not None
                positions.append(env.data.qpos[:3].copy())
                rewards.append(float(reward))
                minimum_ranges.append(float(final_info["minimum_range_m"]))
            path = np.asarray(positions, dtype=np.float64)
            episode_rows.append(
                {
                    "episode_index": episode_index,
                    "episode_seed": episode_seed,
                    "scene_seed": int(reset_info["scene_seed"]),
                    "sensor_seed": int(reset_info["sensor_seed"]),
                    "return": float(sum(rewards)),
                    "length": len(rewards),
                    "success": bool(final_info["is_success"]),
                    "termination_reason": str(final_info["termination_reason"]),
                    "final_distance_to_goal_m": float(final_info["distance_to_goal_m"]),
                    "minimum_clearance_m": float(min(minimum_ranges)),
                    "path_length_m": float(
                        np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
                    ),
                }
            )
    finally:
        env.close()

    returns = np.asarray([row["return"] for row in episode_rows], dtype=np.float64)
    lengths = np.asarray([row["length"] for row in episode_rows], dtype=np.float64)
    successes = np.asarray([row["success"] for row in episode_rows], dtype=np.bool_)
    collision_count = sum(
        row["termination_reason"] in ("obstacle_collision", "ground_collision")
        for row in episode_rows
    )
    termination_counts = Counter(str(row["termination_reason"]) for row in episode_rows)
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "policy": name,
        "deterministic": True,
        "episode_seeds": list(episode_seeds),
        "episode_count": len(episode_seeds),
        "success_rate": float(np.mean(successes)),
        "collision_rate": collision_count / len(episode_seeds),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "termination_counts": dict(sorted(termination_counts.items())),
        "episodes": episode_rows,
    }
    summary_path = output_path / "summary.json"
    summary_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return BaselineArtifacts(output_path, summary_path, metrics)


def evaluate_baseline_suite(
    config: TrainingConfig,
    output_directory: str | Path,
    episode_seeds: tuple[int, ...],
) -> Path:
    """İki baseline'ı aynı seed'lerde çalıştırıp karşılaştırma JSON'u yaz."""

    output_path = Path(output_directory).expanduser().resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Baseline suite dizini boş değil: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    results = {
        name: evaluate_baseline(name, config, output_path / name, episode_seeds).metrics
        for name in ("random", "direct_goal")
    }
    comparison = {
        "schema_version": 1,
        "episode_seeds": list(episode_seeds),
        "policies": results,
    }
    comparison_path = output_path / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return comparison_path
