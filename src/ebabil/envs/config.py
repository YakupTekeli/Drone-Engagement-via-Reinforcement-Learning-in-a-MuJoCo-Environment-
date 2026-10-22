"""Gymnasium navigasyon ortamı konfigürasyonu."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAVIGATION_CONFIG = PACKAGE_ROOT / "configs" / "environment" / "navigation.toml"


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not math.isfinite(value) or not lower_ok:
        qualifier = "negatif olmayan" if allow_zero else "pozitif"
        raise ValueError(f"{name} {qualifier} ve sonlu olmalı")
    return value


@dataclass(frozen=True)
class SimulationConfig:
    policy_frequency_hz: float
    max_episode_seconds: float

    def validate(self) -> None:
        _positive(self.policy_frequency_hz, "simulation.policy_frequency_hz")
        _positive(self.max_episode_seconds, "simulation.max_episode_seconds")


@dataclass(frozen=True)
class ActionConfig:
    max_horizontal_speed_m_s: float
    max_vertical_speed_m_s: float
    max_yaw_rate_rad_s: float

    def validate(self) -> None:
        _positive(self.max_horizontal_speed_m_s, "action.max_horizontal_speed_m_s")
        _positive(self.max_vertical_speed_m_s, "action.max_vertical_speed_m_s")
        _positive(self.max_yaw_rate_rad_s, "action.max_yaw_rate_rad_s")


@dataclass(frozen=True)
class ObservationConfig:
    goal_distance_scale_m: float

    def validate(self) -> None:
        _positive(self.goal_distance_scale_m, "observation.goal_distance_scale_m")


@dataclass(frozen=True)
class TerminationConfig:
    goal_radius_m: float
    goal_max_speed_m_s: float
    goal_hold_seconds: float
    minimum_altitude_m: float
    maximum_tilt_rad: float

    def validate(self) -> None:
        _positive(self.goal_radius_m, "termination.goal_radius_m")
        _positive(self.goal_max_speed_m_s, "termination.goal_max_speed_m_s")
        _positive(self.goal_hold_seconds, "termination.goal_hold_seconds")
        _positive(
            self.minimum_altitude_m,
            "termination.minimum_altitude_m",
            allow_zero=True,
        )
        _positive(self.maximum_tilt_rad, "termination.maximum_tilt_rad")
        if self.maximum_tilt_rad >= math.pi:
            raise ValueError("termination.maximum_tilt_rad pi değerinden küçük olmalı")


@dataclass(frozen=True)
class RewardConfig:
    progress_weight: float
    goal_bonus: float
    collision_penalty: float
    failure_penalty: float
    proximity_weight: float
    proximity_threshold_m: float
    smoothness_weight: float
    attitude_weight: float
    time_penalty: float

    def validate(self) -> None:
        for name, value in vars(self).items():
            _positive(value, f"reward.{name}", allow_zero=True)
        _positive(self.proximity_threshold_m, "reward.proximity_threshold_m")


@dataclass(frozen=True)
class NavigationEnvConfig:
    simulation: SimulationConfig
    action: ActionConfig
    observation: ObservationConfig
    termination: TerminationConfig
    reward: RewardConfig

    def validate(self) -> None:
        self.simulation.validate()
        self.action.validate()
        self.observation.validate()
        self.termination.validate()
        self.reward.validate()


def load_navigation_config(
    path: str | Path = DEFAULT_NAVIGATION_CONFIG,
) -> NavigationEnvConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Navigasyon konfigürasyonu bulunamadı: {config_path}")
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    simulation = raw["simulation"]
    action = raw["action"]
    observation = raw["observation"]
    termination = raw["termination"]
    reward = raw["reward"]
    config = NavigationEnvConfig(
        simulation=SimulationConfig(
            policy_frequency_hz=float(simulation["policy_frequency_hz"]),
            max_episode_seconds=float(simulation["max_episode_seconds"]),
        ),
        action=ActionConfig(
            max_horizontal_speed_m_s=float(action["max_horizontal_speed_m_s"]),
            max_vertical_speed_m_s=float(action["max_vertical_speed_m_s"]),
            max_yaw_rate_rad_s=float(action["max_yaw_rate_rad_s"]),
        ),
        observation=ObservationConfig(
            goal_distance_scale_m=float(observation["goal_distance_scale_m"]),
        ),
        termination=TerminationConfig(
            goal_radius_m=float(termination["goal_radius_m"]),
            goal_max_speed_m_s=float(termination["goal_max_speed_m_s"]),
            goal_hold_seconds=float(termination["goal_hold_seconds"]),
            minimum_altitude_m=float(termination["minimum_altitude_m"]),
            maximum_tilt_rad=float(termination["maximum_tilt_rad"]),
        ),
        reward=RewardConfig(
            progress_weight=float(reward["progress_weight"]),
            goal_bonus=float(reward["goal_bonus"]),
            collision_penalty=float(reward["collision_penalty"]),
            failure_penalty=float(reward["failure_penalty"]),
            proximity_weight=float(reward["proximity_weight"]),
            proximity_threshold_m=float(reward["proximity_threshold_m"]),
            smoothness_weight=float(reward["smoothness_weight"]),
            attitude_weight=float(reward["attitude_weight"]),
            time_penalty=float(reward["time_penalty"]),
        ),
    )
    config.validate()
    return config
