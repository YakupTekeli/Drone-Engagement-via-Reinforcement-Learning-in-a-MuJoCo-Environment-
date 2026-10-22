"""Headless PPO eğitim, checkpoint ve resume akışı."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from ebabil.training.config import TrainingConfig
from ebabil.training.env_factory import make_vector_env


@dataclass(frozen=True)
class TrainingArtifacts:
    run_directory: Path
    final_model: Path
    best_model: Path | None
    best_normalization: Path | None
    normalization: Path | None
    metadata: Path
    progress_csv: Path
    actual_total_timesteps: int


class _SaveBestNormalizationCallback(BaseCallback):
    def __init__(self, path: Path) -> None:
        super().__init__(verbose=0)
        self.path = path

    def _on_step(self) -> bool:
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is None:
            raise RuntimeError("Best normalization callback'i VecNormalize bekler")
        vec_normalize.save(str(self.path))
        return True


def default_run_name(seed: int) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"ppo_clear_seed_{seed}_{timestamp}"


def _uses_normalization(config: TrainingConfig) -> bool:
    return config.normalization.observations or config.normalization.rewards


def _load_normalization(
    config: TrainingConfig,
    env: VecEnv,
    path: Path,
    *,
    training: bool,
) -> VecNormalize:
    if not isinstance(env, VecNormalize):
        raise ValueError("Normalization kapalıyken normalization kaydı yüklenemez")
    loaded = VecNormalize.load(str(path), env.venv)
    loaded.training = training
    loaded.norm_reward = config.normalization.rewards if training else False
    return loaded


def _metadata(config: TrainingConfig, *, status: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "ebabil_mujoco": version("ebabil-mujoco"),
            "gymnasium": version("gymnasium"),
            "mujoco": version("mujoco"),
            "numpy": version("numpy"),
            "stable_baselines3": version("stable-baselines3"),
            "torch": version("torch"),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def train_ppo(
    config: TrainingConfig,
    run_directory: str | Path,
    *,
    device: str = "auto",
    resume_checkpoint: str | Path | None = None,
    resume_normalization: str | Path | None = None,
) -> TrainingArtifacts:
    """PPO eğit, ara/final artifact'leri güvenli bir koşu dizinine yaz."""

    config.validate()
    run_path = Path(run_directory).expanduser().resolve()
    if run_path.exists() and any(run_path.iterdir()):
        raise FileExistsError(f"Koşu dizini boş değil: {run_path}")
    run_path.mkdir(parents=True, exist_ok=True)
    checkpoints_path = run_path / "checkpoints"
    best_path = run_path / "best"
    logs_path = run_path / "logs"
    evaluation_path = run_path / "evaluation"
    for directory in (checkpoints_path, best_path, logs_path, evaluation_path):
        directory.mkdir()

    metadata_path = run_path / "metadata.json"
    metadata = _metadata(config, status="running")
    metadata["resume"] = {
        "checkpoint": str(Path(resume_checkpoint).resolve())
        if resume_checkpoint
        else None,
        "normalization": str(Path(resume_normalization).resolve())
        if resume_normalization
        else None,
    }
    _write_json(metadata_path, metadata)

    if config.logging.tensorboard and not importlib.util.find_spec("tensorboard"):
        raise RuntimeError(
            "TensorBoard logu istendi fakat paket .venv içinde kurulu değil"
        )
    if resume_normalization is not None and not _uses_normalization(config):
        raise ValueError("Resume normalization verildi fakat normalization kapalı")
    if (
        resume_checkpoint is not None
        and _uses_normalization(config)
        and resume_normalization is None
    ):
        raise ValueError(
            "Normalization açık bir resume koşusunda --resume-normalization zorunlu"
        )

    train_env: VecEnv | None = None
    eval_env: VecEnv | None = None
    try:
        train_env = make_vector_env(
            config,
            monitor_dir=logs_path / "train_monitor",
            training=True,
        )
        if resume_normalization is not None:
            train_env = _load_normalization(
                config,
                train_env,
                Path(resume_normalization).expanduser().resolve(),
                training=True,
            )

        eval_env = make_vector_env(
            config,
            seed=config.run.seed + 1_000_000,
            n_envs=1,
            vector_backend="dummy",
            monitor_dir=logs_path / "eval_monitor",
            training=False,
        )
        if (
            config.normalization.observations
            and isinstance(train_env, VecNormalize)
            and isinstance(eval_env, VecNormalize)
        ):
            eval_env.obs_rms = train_env.obs_rms

        save_frequency = max(
            config.checkpoint.save_frequency_timesteps // config.run.n_envs,
            1,
        )
        evaluation_frequency = max(
            config.checkpoint.evaluation_frequency_timesteps // config.run.n_envs,
            1,
        )
        callbacks = CallbackList(
            [
                CheckpointCallback(
                    save_freq=save_frequency,
                    save_path=str(checkpoints_path),
                    name_prefix="ppo",
                    save_vecnormalize=_uses_normalization(config),
                ),
                EvalCallback(
                    eval_env,
                    callback_on_new_best=(
                        _SaveBestNormalizationCallback(best_path / "vecnormalize.pkl")
                        if _uses_normalization(config)
                        else None
                    ),
                    best_model_save_path=str(best_path),
                    log_path=str(evaluation_path),
                    eval_freq=evaluation_frequency,
                    n_eval_episodes=config.checkpoint.evaluation_episodes,
                    deterministic=True,
                    render=False,
                    verbose=1,
                ),
            ]
        )

        formats = ["stdout", "csv"]
        if config.logging.tensorboard:
            formats.append("tensorboard")
        logger = configure(str(logs_path), formats)
        if resume_checkpoint is None:
            policy = (
                "MultiInputPolicy"
                if config.environment.kind
                in ("vision", "latent_vision", "detection_vision")
                else "MlpPolicy"
            )
            model = PPO(
                policy,
                train_env,
                learning_rate=config.ppo.learning_rate,
                n_steps=config.ppo.n_steps,
                batch_size=config.ppo.batch_size,
                n_epochs=config.ppo.n_epochs,
                gamma=config.ppo.gamma,
                gae_lambda=config.ppo.gae_lambda,
                clip_range=config.ppo.clip_range,
                ent_coef=config.ppo.ent_coef,
                vf_coef=config.ppo.vf_coef,
                max_grad_norm=config.ppo.max_grad_norm,
                policy_kwargs={"net_arch": list(config.ppo.network_arch)},
                seed=config.run.seed,
                device=device,
                verbose=1,
            )
        else:
            model = PPO.load(
                Path(resume_checkpoint).expanduser().resolve(),
                env=train_env,
                device=device,
            )
        model.set_logger(logger)
        model.learn(
            total_timesteps=config.run.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=resume_checkpoint is None,
            progress_bar=False,
        )

        final_model = run_path / "final_model"
        model.save(final_model)
        normalization_path: Path | None = None
        if isinstance(train_env, VecNormalize):
            normalization_path = run_path / "vecnormalize.pkl"
            train_env.save(str(normalization_path))
        best_model_candidate = best_path / "best_model.zip"
        best_model_path: Path | None = (
            best_model_candidate if best_model_candidate.is_file() else None
        )
        best_normalization_candidate = best_path / "vecnormalize.pkl"
        best_normalization_path: Path | None = (
            best_normalization_candidate
            if best_normalization_candidate.is_file()
            else None
        )
        metadata["status"] = "completed"
        metadata["completed_at_utc"] = datetime.now(UTC).isoformat()
        metadata["actual_total_timesteps"] = model.num_timesteps
        metadata["artifacts"] = {
            "final_model": str(final_model.with_suffix(".zip")),
            "best_model": str(best_model_path) if best_model_path else None,
            "best_normalization": (
                str(best_normalization_path) if best_normalization_path else None
            ),
            "normalization": str(normalization_path) if normalization_path else None,
            "progress_csv": str(logs_path / "progress.csv"),
        }
        _write_json(metadata_path, metadata)
    finally:
        training_error = sys.exception()
        if training_error is not None:
            metadata["status"] = "failed"
            metadata["failed_at_utc"] = datetime.now(UTC).isoformat()
            metadata["error"] = {
                "type": type(training_error).__name__,
                "message": str(training_error),
            }
            _write_json(metadata_path, metadata)
        if eval_env is not None:
            eval_env.close()
        if train_env is not None:
            train_env.close()

    return TrainingArtifacts(
        run_directory=run_path,
        final_model=final_model.with_suffix(".zip"),
        best_model=best_model_path,
        best_normalization=best_normalization_path,
        normalization=normalization_path,
        metadata=metadata_path,
        progress_csv=logs_path / "progress.csv",
        actual_total_timesteps=model.num_timesteps,
    )
