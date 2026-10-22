"""Tipli PPO eğitim konfigürasyonu."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ebabil.envs.navigation import SensorMode
from ebabil.scenes import EndpointMode, SceneMode

VectorBackend = Literal["dummy", "subproc"]
EnvironmentKind = Literal["legacy", "vision", "latent_vision", "detection_vision"]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PPO_CONFIG = PACKAGE_ROOT / "configs" / "training" / "ppo_baseline.toml"


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} pozitif bir tamsayı olmalı")
    return value


def _bounded_float(
    value: int | float,
    name: str,
    *,
    lower: float = 0.0,
    upper: float | None = None,
    allow_lower: bool = False,
) -> float:
    number = float(value)
    lower_valid = number >= lower if allow_lower else number > lower
    upper_valid = upper is None or number <= upper
    if not math.isfinite(number) or not lower_valid or not upper_valid:
        interval = f"{lower}{' dahil' if allow_lower else ' hariç'}"
        if upper is not None:
            interval += f" ile {upper} dahil"
        raise ValueError(f"{name} {interval} aralığında ve sonlu olmalı")
    return number


@dataclass(frozen=True)
class RunConfig:
    total_timesteps: int
    seed: int
    n_envs: int
    vector_backend: VectorBackend

    def validate(self) -> None:
        _positive_int(self.total_timesteps, "run.total_timesteps")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("run.seed negatif olmayan bir tamsayı olmalı")
        _positive_int(self.n_envs, "run.n_envs")
        if self.vector_backend not in ("dummy", "subproc"):
            raise ValueError("run.vector_backend 'dummy' veya 'subproc' olmalı")


@dataclass(frozen=True)
class EnvironmentConfig:
    """Eğitim ortamı seçimi.

    `kind = "legacy"` ADR-008 `EbabilNavigation-v0` ortamını kullanır ve
    `sensor_mode` ile `max_episode_steps` gerektirir. `kind = "vision"` ADR-015
    `EbabilVisionNavigation-v0` ortamını kullanır; sensör modu her zaman ideal
    adapter'dır ve episode sınırı ADR-013 dinamik segment horizon'ından gelir,
    bu nedenle iki alan da verilmemelidir. `kind = "latent_vision"` ADR-022
    `EbabilVisionNavigation-v1` (`vision-v3`) ortamıdır; `approval_manifest`
    etiketsiz VAE zincirinin manifest'idir ve `visual_domain_split` görsel
    randomization alanını seçer. `kind = "detection_vision"` ADR-023
    `EbabilVisionNavigation-v2` (`vision-v4`) ortamıdır; `approval_manifest`
    YOLO dedektör zincirinin (`kind = "yolo-detector"`) manifest'idir.
    """

    scene_mode: SceneMode
    endpoint_mode: EndpointMode
    kind: EnvironmentKind = "legacy"
    sensor_mode: SensorMode | None = None
    max_episode_steps: int | None = None
    approval_manifest: str | None = None
    scene_config_path: str | None = None
    #: REV-045: hold süresi kademelendirmesi için hedef alan config'i.
    #: `None` global varsayılanı kullanır. Kademeler YALNIZ eğitim
    #: protokolüdür; D1 kapısı `hold_seconds = 3.0` ile ölçülür ve
    #: evaluation varsayılan config'i kullanır.
    target_config_path: str | None = None
    #: ADR-022 §6: yalnız `latent_vision`; `None` ortam config'inin alanını
    #: kullanır (`train`), evaluation `validation` verir.
    visual_domain_split: str | None = None

    def validate(self) -> None:
        if self.scene_mode not in ("clear", "single_blocker", "random_obstacles"):
            raise ValueError("environment.scene_mode geçersiz")
        if self.endpoint_mode not in (
            "fixed",
            "random_start",
            "random_goal",
            "random_both",
        ):
            raise ValueError("environment.endpoint_mode geçersiz")
        if self.kind not in ("legacy", "vision", "latent_vision", "detection_vision"):
            raise ValueError(
                "environment.kind 'legacy', 'vision', 'latent_vision' veya "
                "'detection_vision' olmalı"
            )
        if self.visual_domain_split is not None and self.kind not in (
            "latent_vision",
            "detection_vision",
        ):
            raise ValueError(
                "visual_domain_split yalnız latent_vision/detection_vision "
                "ortamında anlamlıdır"
            )
        if self.scene_config_path is not None and not self.scene_config_path:
            raise ValueError("environment.scene_config_path boş olamaz")
        if self.target_config_path is not None and not self.target_config_path:
            raise ValueError("environment.target_config_path boş olamaz")
        if self.target_config_path is not None and self.kind == "legacy":
            raise ValueError("target_config_path yalnız vision ortamında anlamlıdır")
        if self.kind == "legacy":
            if self.sensor_mode not in ("ideal", "imperfect"):
                raise ValueError("environment.sensor_mode geçersiz")
            _positive_int(self.max_episode_steps, "environment.max_episode_steps")
            if self.approval_manifest is not None:
                raise ValueError(
                    "environment.approval_manifest yalnız vision ortamında geçerli"
                )
            return
        if not self.approval_manifest:
            raise ValueError(
                "vision ortamı environment.approval_manifest gerektirir; "
                "encoder yalnız ADR-014 approval zincirinden yüklenir"
            )
        if self.sensor_mode is not None:
            raise ValueError(
                "vision ortamında sensor_mode verilemez; ideal adapter zorunludur"
            )
        if self.max_episode_steps is not None:
            raise ValueError(
                "vision ortamında max_episode_steps verilemez; episode sınırı "
                "ADR-013 dinamik segment horizon'ından gelir"
            )


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    network_arch: tuple[int, ...]

    def validate(self, n_envs: int) -> None:
        _bounded_float(self.learning_rate, "ppo.learning_rate")
        _positive_int(self.n_steps, "ppo.n_steps")
        _positive_int(self.batch_size, "ppo.batch_size")
        _positive_int(self.n_epochs, "ppo.n_epochs")
        _bounded_float(self.gamma, "ppo.gamma", upper=1.0)
        _bounded_float(self.gae_lambda, "ppo.gae_lambda", upper=1.0)
        _bounded_float(self.clip_range, "ppo.clip_range", upper=1.0)
        _bounded_float(self.ent_coef, "ppo.ent_coef", allow_lower=True)
        _bounded_float(self.vf_coef, "ppo.vf_coef", allow_lower=True)
        _bounded_float(self.max_grad_norm, "ppo.max_grad_norm")
        if not self.network_arch or any(
            _positive_int(width, "ppo.network_arch") <= 0 for width in self.network_arch
        ):
            raise ValueError("ppo.network_arch boş olamaz")
        rollout_size = self.n_steps * n_envs
        if self.batch_size > rollout_size or rollout_size % self.batch_size != 0:
            raise ValueError(
                "ppo.batch_size, n_steps*n_envs değerini kalansız bölmeli ve "
                "ondan büyük olmamalı"
            )


@dataclass(frozen=True)
class NormalizationConfig:
    observations: bool
    rewards: bool
    clip_observation: float
    clip_reward: float

    def validate(self) -> None:
        if not isinstance(self.observations, bool) or not isinstance(
            self.rewards, bool
        ):
            raise ValueError("normalization bayrakları boolean olmalı")
        _bounded_float(self.clip_observation, "normalization.clip_observation")
        _bounded_float(self.clip_reward, "normalization.clip_reward")


@dataclass(frozen=True)
class CheckpointConfig:
    save_frequency_timesteps: int
    evaluation_frequency_timesteps: int
    evaluation_episodes: int

    def validate(self) -> None:
        _positive_int(
            self.save_frequency_timesteps, "checkpoint.save_frequency_timesteps"
        )
        _positive_int(
            self.evaluation_frequency_timesteps,
            "checkpoint.evaluation_frequency_timesteps",
        )
        _positive_int(self.evaluation_episodes, "checkpoint.evaluation_episodes")


@dataclass(frozen=True)
class LoggingConfig:
    tensorboard: bool

    def validate(self) -> None:
        if not isinstance(self.tensorboard, bool):
            raise ValueError("logging.tensorboard boolean olmalı")


@dataclass(frozen=True)
class TrainingConfig:
    run: RunConfig
    environment: EnvironmentConfig
    ppo: PPOConfig
    normalization: NormalizationConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig

    def validate(self) -> None:
        self.run.validate()
        self.environment.validate()
        self.ppo.validate(self.run.n_envs)
        self.normalization.validate()
        self.checkpoint.validate()
        self.logging.validate()


def require_legacy_environment(config: TrainingConfig) -> tuple[SensorMode, int]:
    """Legacy evaluation yolunun gerektirdiği alanları doğrulayarak çıkar.

    ADR-008 legacy değerlendirme akışı `EbabilNavigation-v0` içindir. Vision
    ortamı için ayrı bir evaluation akışı gerekir; sessizce legacy ortama
    düşmek yerine açık hata verilir.
    """

    environment = config.environment
    if environment.kind != "legacy":
        raise ValueError(
            "Bu evaluation akışı yalnız legacy EbabilNavigation-v0 ortamını "
            f"destekler; environment.kind={environment.kind}"
        )
    if environment.sensor_mode is None or environment.max_episode_steps is None:
        raise ValueError(
            "Legacy evaluation sensor_mode ve max_episode_steps gerektirir"
        )
    return environment.sensor_mode, environment.max_episode_steps


def load_training_config(path: str | Path = DEFAULT_PPO_CONFIG) -> TrainingConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"PPO konfigürasyonu bulunamadı: {config_path}")
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    run = raw["run"]
    environment = raw["environment"]
    ppo = raw["ppo"]
    normalization = raw["normalization"]
    checkpoint = raw["checkpoint"]
    logging = raw["logging"]
    config = TrainingConfig(
        run=RunConfig(
            total_timesteps=run["total_timesteps"],
            seed=run["seed"],
            n_envs=run["n_envs"],
            vector_backend=run["vector_backend"],
        ),
        environment=EnvironmentConfig(
            scene_mode=environment["scene_mode"],
            endpoint_mode=environment["endpoint_mode"],
            kind=environment.get("kind", "legacy"),
            sensor_mode=environment.get("sensor_mode"),
            max_episode_steps=environment.get("max_episode_steps"),
            approval_manifest=environment.get("approval_manifest"),
            scene_config_path=environment.get("scene_config_path"),
            target_config_path=environment.get("target_config_path"),
            visual_domain_split=environment.get("visual_domain_split"),
        ),
        ppo=PPOConfig(
            learning_rate=float(ppo["learning_rate"]),
            n_steps=ppo["n_steps"],
            batch_size=ppo["batch_size"],
            n_epochs=ppo["n_epochs"],
            gamma=float(ppo["gamma"]),
            gae_lambda=float(ppo["gae_lambda"]),
            clip_range=float(ppo["clip_range"]),
            ent_coef=float(ppo["ent_coef"]),
            vf_coef=float(ppo["vf_coef"]),
            max_grad_norm=float(ppo["max_grad_norm"]),
            network_arch=tuple(ppo["network_arch"]),
        ),
        normalization=NormalizationConfig(
            observations=normalization["observations"],
            rewards=normalization["rewards"],
            clip_observation=float(normalization["clip_observation"]),
            clip_reward=float(normalization["clip_reward"]),
        ),
        checkpoint=CheckpointConfig(
            save_frequency_timesteps=checkpoint["save_frequency_timesteps"],
            evaluation_frequency_timesteps=checkpoint["evaluation_frequency_timesteps"],
            evaluation_episodes=checkpoint["evaluation_episodes"],
        ),
        logging=LoggingConfig(tensorboard=logging["tensorboard"]),
    )
    config.validate()
    return config
