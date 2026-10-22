"""Stable-Baselines3 için tekrar üretilebilir vektör ortam fabrikası."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
)

from ebabil.envs import (
    EbabilDetectionVisionNavigationEnv,
    EbabilLatentVisionNavigationEnv,
    EbabilNavigationEnv,
    EbabilVisionNavigationEnv,
)
from ebabil.scenes import load_scene_config, load_target_area_config
from ebabil.training.config import TrainingConfig, VectorBackend

MONITOR_INFO_KEYS = (
    "is_success",
    "termination_reason",
    "distance_to_goal_m",
    "scene_seed",
    "sensor_seed",
)
#: Vision ortamında ADR-017 terminal hover metrikleri de episode kaydına girer.
VISION_MONITOR_INFO_KEYS = (
    "is_success",
    "termination_reason",
    "distance_to_goal_m",
    "scene_seed",
    "sensor_seed",
    "hover_state",
    "hover_abort_reason",
    "hover_hold_seconds",
    "hover_hold_progress",
    "camera_frame_stale",
    "segment_budget_seconds",
    "hover_horizontal_distance_m",
)


def monitor_info_keys(config: TrainingConfig) -> tuple[str, ...]:
    """Ortam türüne göre Monitor'a yazılacak info anahtarları."""

    if config.environment.kind in ("vision", "latent_vision", "detection_vision"):
        return VISION_MONITOR_INFO_KEYS
    return MONITOR_INFO_KEYS


def _environment_factory(
    config: TrainingConfig,
    rank: int,
    monitor_dir: Path | None,
) -> Callable[[], gym.Env]:
    def create() -> gym.Env:
        env: gym.Env
        scene_config = (
            load_scene_config(config.environment.scene_config_path)
            if config.environment.scene_config_path is not None
            else None
        )
        if config.environment.kind == "detection_vision":
            assert config.environment.approval_manifest is not None
            env = EbabilDetectionVisionNavigationEnv(
                approval_manifest=config.environment.approval_manifest,
                scene_mode=config.environment.scene_mode,
                endpoint_mode=config.environment.endpoint_mode,
                scene_config=scene_config,
                target_config=(
                    load_target_area_config(config.environment.target_config_path)
                    if config.environment.target_config_path is not None
                    else None
                ),
                visual_domain_split=config.environment.visual_domain_split,
            )
        elif config.environment.kind == "latent_vision":
            assert config.environment.approval_manifest is not None
            env = EbabilLatentVisionNavigationEnv(
                approval_manifest=config.environment.approval_manifest,
                scene_mode=config.environment.scene_mode,
                endpoint_mode=config.environment.endpoint_mode,
                scene_config=scene_config,
                target_config=(
                    load_target_area_config(config.environment.target_config_path)
                    if config.environment.target_config_path is not None
                    else None
                ),
                visual_domain_split=config.environment.visual_domain_split,
            )
        elif config.environment.kind == "vision":
            assert config.environment.approval_manifest is not None
            # REV-045: hold süresi kademelendirmesi. Kademeler yalnız eğitim
            # protokolüdür; evaluation varsayılan config'i kullanır ve D1
            # kapısı `hold_seconds = 3.0` ile ölçülür.
            target_config = (
                load_target_area_config(config.environment.target_config_path)
                if config.environment.target_config_path is not None
                else None
            )
            env = EbabilVisionNavigationEnv(
                approval_manifest=config.environment.approval_manifest,
                scene_mode=config.environment.scene_mode,
                endpoint_mode=config.environment.endpoint_mode,
                scene_config=scene_config,
                target_config=target_config,
            )
        else:
            assert config.environment.sensor_mode is not None
            env = EbabilNavigationEnv(
                scene_mode=config.environment.scene_mode,
                endpoint_mode=config.environment.endpoint_mode,
                sensor_mode=config.environment.sensor_mode,
                max_episode_steps=config.environment.max_episode_steps,
                scene_config=scene_config,
            )
        monitor_file = None
        if monitor_dir is not None:
            monitor_dir.mkdir(parents=True, exist_ok=True)
            monitor_file = str(monitor_dir / f"env_{rank}")
        return Monitor(
            env, filename=monitor_file, info_keywords=monitor_info_keys(config)
        )

    return create


def make_vector_env(
    config: TrainingConfig,
    *,
    seed: int | None = None,
    n_envs: int | None = None,
    vector_backend: VectorBackend | None = None,
    monitor_dir: str | Path | None = None,
    training: bool = True,
) -> VecEnv:
    """Config'e göre seed'lenmiş SB3 vektör ortamı oluştur."""

    count = config.run.n_envs if n_envs is None else n_envs
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("n_envs pozitif bir tamsayı olmalı")
    backend = config.run.vector_backend if vector_backend is None else vector_backend
    if backend not in ("dummy", "subproc"):
        raise ValueError("vector_backend 'dummy' veya 'subproc' olmalı")
    base_seed = config.run.seed if seed is None else seed
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("seed negatif olmayan bir tamsayı olmalı")

    output_dir = Path(monitor_dir).expanduser().resolve() if monitor_dir else None
    factories = [
        _environment_factory(config, rank, output_dir) for rank in range(count)
    ]
    if backend == "subproc" and count > 1:
        vec_env: VecEnv = SubprocVecEnv(factories, start_method="spawn")
    else:
        vec_env = DummyVecEnv(factories)
    vec_env.seed(base_seed)

    normalization = config.normalization
    if normalization.observations or normalization.rewards:
        vec_env = VecNormalize(
            vec_env,
            training=training,
            norm_obs=normalization.observations,
            norm_reward=normalization.rewards if training else False,
            clip_obs=normalization.clip_observation,
            clip_reward=normalization.clip_reward,
            gamma=config.ppo.gamma,
        )
    return vec_env
