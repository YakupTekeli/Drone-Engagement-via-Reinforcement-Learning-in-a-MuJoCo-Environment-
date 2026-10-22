"""PPO politikası için deterministic evaluation ve episode kaydı."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ebabil.envs import OBSERVATION_VERSION, EbabilNavigationEnv
from ebabil.evaluation.replay import EpisodeReplay, save_replay
from ebabil.training.config import TrainingConfig, require_legacy_environment

REWARD_COMPONENT_NAMES = (
    "progress",
    "goal",
    "collision",
    "failure",
    "proximity",
    "smoothness",
    "attitude",
    "time",
)


@dataclass(frozen=True)
class EvaluationArtifacts:
    output_directory: Path
    summary: Path
    replay_files: tuple[Path, ...]
    metrics: dict[str, Any]


def _observation_normalizer(
    config: TrainingConfig,
    env: EbabilNavigationEnv,
    normalization_path: str | Path | None,
) -> VecNormalize | None:
    if not config.normalization.observations:
        return None
    if normalization_path is None:
        raise ValueError(
            "Observation normalization açık; evaluation için normalization kaydı gerekli"
        )
    vector_env = DummyVecEnv([lambda: env])
    normalizer = VecNormalize.load(
        str(Path(normalization_path).expanduser().resolve()), vector_env
    )
    normalizer.training = False
    normalizer.norm_reward = False
    return normalizer


def _policy_observation(
    observation: np.ndarray,
    normalizer: VecNormalize | None,
) -> np.ndarray:
    if normalizer is None:
        return observation
    normalized = normalizer.normalize_obs(observation[np.newaxis])
    if not isinstance(normalized, np.ndarray):
        raise TypeError("Ebabil observation normalization sonucu ndarray olmalı")
    return normalized[0]


def _episode_replay(
    *,
    physics_time_s: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    policy_time_s: list[float],
    actions: list[np.ndarray],
    rewards: list[float],
    distances: list[float],
    minimum_ranges: list[float],
    reward_components: list[list[float]],
    rotor_thrusts_n: np.ndarray,
    velocity_commands: np.ndarray,
    yaw_rate_commands: np.ndarray,
    mixer_saturated: np.ndarray,
    torque_saturated: np.ndarray,
) -> EpisodeReplay:
    return EpisodeReplay(
        time_s=physics_time_s,
        qpos=qpos,
        qvel=qvel,
        policy_time_s=np.asarray(policy_time_s, dtype=np.float64),
        actions=np.asarray(actions, dtype=np.float64).reshape(-1, 4),
        rewards=np.asarray(rewards, dtype=np.float64),
        distance_to_goal_m=np.asarray(distances, dtype=np.float64),
        minimum_range_m=np.asarray(minimum_ranges, dtype=np.float64),
        reward_components=np.asarray(reward_components, dtype=np.float64).reshape(
            -1, 8
        ),
        rotor_thrusts_n=rotor_thrusts_n,
        velocity_command_world_m_s=velocity_commands,
        yaw_rate_command_rad_s=yaw_rate_commands,
        mixer_saturated=mixer_saturated,
        torque_saturated=torque_saturated,
    )


def evaluate_policy(
    checkpoint: str | Path,
    config: TrainingConfig,
    output_directory: str | Path,
    *,
    episodes: int = 20,
    seed: int = 20260910,
    episode_seeds: tuple[int, ...] | None = None,
    record_replays: int = 3,
    normalization_path: str | Path | None = None,
    device: str = "auto",
) -> EvaluationArtifacts:
    """Politikayı sabit seed dizisinde ölç ve seçili episode'ları kaydet."""

    config.validate()
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes pozitif bir tamsayı olmalı")
    resolved_episode_seeds = (
        tuple(seed + index for index in range(episodes))
        if episode_seeds is None
        else episode_seeds
    )
    if (
        not resolved_episode_seeds
        or len(set(resolved_episode_seeds)) != len(resolved_episode_seeds)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in resolved_episode_seeds
        )
    ):
        raise ValueError("episode_seeds benzersiz, negatif olmayan tamsayılar olmalı")
    episode_count = len(resolved_episode_seeds)
    if (
        isinstance(record_replays, bool)
        or not isinstance(record_replays, int)
        or not 0 <= record_replays <= episode_count
    ):
        raise ValueError("record_replays 0 ile episodes arasında bir tamsayı olmalı")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed negatif olmayan bir tamsayı olmalı")

    output_path = Path(output_directory).expanduser().resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Evaluation dizini boş değil: {output_path}")
    replay_path = output_path / "replays"
    replay_path.mkdir(parents=True, exist_ok=True)

    sensor_mode, max_episode_steps = require_legacy_environment(config)
    env = EbabilNavigationEnv(
        scene_mode=config.environment.scene_mode,
        endpoint_mode=config.environment.endpoint_mode,
        sensor_mode=sensor_mode,
        max_episode_steps=max_episode_steps,
        record_physics_trace=record_replays > 0,
    )
    normalizer = _observation_normalizer(config, env, normalization_path)
    model = PPO.load(str(Path(checkpoint).expanduser().resolve()), device=device)
    episode_rows: list[dict[str, Any]] = []
    replay_files: list[Path] = []
    try:
        for episode_index, episode_seed in enumerate(resolved_episode_seeds):
            observation, reset_info = env.reset(seed=episode_seed)
            assert env.data is not None
            policy_times: list[float] = []
            policy_positions = [env.data.qpos[:3].copy()]
            actions: list[np.ndarray] = []
            rewards: list[float] = []
            distances = [float(reset_info["distance_to_goal_m"])]
            minimum_ranges: list[float] = []
            reward_components: list[list[float]] = []
            terminated = truncated = False
            final_info = reset_info

            while not (terminated or truncated):
                policy_observation = _policy_observation(observation, normalizer)
                action, _ = model.predict(policy_observation, deterministic=True)
                action = np.asarray(action, dtype=np.float32)
                observation, reward, terminated, truncated, final_info = env.step(
                    action
                )
                assert env.data is not None
                policy_times.append(float(env.data.time))
                policy_positions.append(env.data.qpos[:3].copy())
                actions.append(action.copy())
                rewards.append(float(reward))
                distances.append(float(final_info["distance_to_goal_m"]))
                minimum_ranges.append(float(final_info["minimum_range_m"]))
                components = final_info["reward_components"]
                reward_components.append(
                    [float(components[name]) for name in REWARD_COMPONENT_NAMES]
                )

            path_positions = np.asarray(policy_positions, dtype=np.float64)
            path_length = float(
                np.sum(np.linalg.norm(np.diff(path_positions, axis=0), axis=1))
            )
            reason = str(final_info["termination_reason"])
            episode_row = {
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "scene_seed": int(reset_info["scene_seed"]),
                "sensor_seed": int(reset_info["sensor_seed"]),
                "return": float(sum(rewards)),
                "length": len(rewards),
                "success": bool(final_info["is_success"]),
                "termination_reason": reason,
                "final_distance_to_goal_m": float(distances[-1]),
                "minimum_clearance_m": float(min(minimum_ranges)),
                "path_length_m": path_length,
            }
            episode_rows.append(episode_row)

            if episode_index < record_replays:
                physics_trace = env.physics_trace()
                replay = _episode_replay(
                    physics_time_s=physics_trace.time_s,
                    qpos=physics_trace.qpos,
                    qvel=physics_trace.qvel,
                    policy_time_s=policy_times,
                    actions=actions,
                    rewards=rewards,
                    distances=distances,
                    minimum_ranges=minimum_ranges,
                    reward_components=reward_components,
                    rotor_thrusts_n=physics_trace.rotor_thrusts_n,
                    velocity_commands=physics_trace.velocity_command_world_m_s,
                    yaw_rate_commands=physics_trace.yaw_rate_command_rad_s,
                    mixer_saturated=physics_trace.mixer_saturated,
                    torque_saturated=physics_trace.torque_saturated,
                )
                replay_file, _ = save_replay(
                    replay_path / f"episode_{episode_index:03d}.npz",
                    replay,
                    {
                        "schema_version": 1,
                        "observation_version": OBSERVATION_VERSION,
                        "frame_rate": "physics_steps_500_hz",
                        "reward_component_names": list(REWARD_COMPONENT_NAMES),
                        "scene_mode": config.environment.scene_mode,
                        "endpoint_mode": config.environment.endpoint_mode,
                        "sensor_mode": config.environment.sensor_mode,
                        **episode_row,
                    },
                )
                replay_files.append(replay_file)
    finally:
        if normalizer is not None:
            normalizer.close()
        else:
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
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "normalization": str(Path(normalization_path).expanduser().resolve())
        if normalization_path
        else None,
        "deterministic": True,
        "base_seed": seed if episode_seeds is None else None,
        "episode_seeds": list(resolved_episode_seeds),
        "episode_count": episode_count,
        "success_rate": float(np.mean(successes)),
        "collision_rate": collision_count / episode_count,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "termination_counts": dict(sorted(termination_counts.items())),
        "episodes": episode_rows,
    }
    summary_path = output_path / "summary.json"
    summary_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return EvaluationArtifacts(
        output_directory=output_path,
        summary=summary_path,
        replay_files=tuple(replay_files),
        metrics=metrics,
    )
