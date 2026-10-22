"""Faz 6 train/held-out seed ve scenario protokolü."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ebabil.envs.navigation import SensorMode
from ebabil.scenes import EndpointMode, SceneMode

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE6_PROTOCOL = (
    PACKAGE_ROOT / "configs" / "evaluation" / "phase6_protocol.toml"
)


@dataclass(frozen=True)
class EvaluationScenario:
    name: str
    scene_mode: SceneMode
    endpoint_mode: EndpointMode
    sensor_mode: SensorMode

    def validate(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("scenario.name boş olmayan güvenli bir ad olmalı")
        if self.scene_mode not in ("clear", "single_blocker", "random_obstacles"):
            raise ValueError(f"{self.name}.scene_mode geçersiz")
        if self.endpoint_mode not in (
            "fixed",
            "random_start",
            "random_goal",
            "random_both",
        ):
            raise ValueError(f"{self.name}.endpoint_mode geçersiz")
        if self.sensor_mode not in ("ideal", "imperfect"):
            raise ValueError(f"{self.name}.sensor_mode geçersiz")


@dataclass(frozen=True)
class EvaluationThresholds:
    minimum_success_rate: float
    maximum_collision_rate: float

    def validate(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"thresholds.{name} [0,1] aralığında olmalı")


@dataclass(frozen=True)
class Phase6Protocol:
    training_seeds: tuple[int, ...]
    held_out_episode_seeds: tuple[int, ...]
    pilot_timesteps: int
    full_timesteps: int
    primary_scenario: str
    thresholds: EvaluationThresholds
    scenarios: tuple[EvaluationScenario, ...]

    def validate(self) -> None:
        if len(self.training_seeds) < 3:
            raise ValueError("En az üç training seed gerekli")
        if len(self.held_out_episode_seeds) < 20:
            raise ValueError("En az yirmi held-out episode seed gerekli")
        all_seeds = self.training_seeds + self.held_out_episode_seeds
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in all_seeds
        ):
            raise ValueError("Tüm seed'ler negatif olmayan tamsayı olmalı")
        if len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("Training seed'leri benzersiz olmalı")
        if len(set(self.held_out_episode_seeds)) != len(self.held_out_episode_seeds):
            raise ValueError("Held-out seed'leri benzersiz olmalı")
        if set(self.training_seeds) & set(self.held_out_episode_seeds):
            raise ValueError("Training ve held-out seed kümeleri ayrık olmalı")
        if (
            isinstance(self.pilot_timesteps, bool)
            or not isinstance(self.pilot_timesteps, int)
            or self.pilot_timesteps <= 0
            or isinstance(self.full_timesteps, bool)
            or not isinstance(self.full_timesteps, int)
            or self.full_timesteps < self.pilot_timesteps
        ):
            raise ValueError("Pilot/full timestep sınırları geçersiz")
        if not self.scenarios:
            raise ValueError("En az bir evaluation scenario gerekli")
        for scenario in self.scenarios:
            scenario.validate()
        names = [scenario.name for scenario in self.scenarios]
        if len(set(names)) != len(names):
            raise ValueError("Scenario adları benzersiz olmalı")
        if self.primary_scenario not in names:
            raise ValueError("primary_scenario scenario listesinde bulunmalı")
        self.thresholds.validate()

    def scenario(self, name: str | None = None) -> EvaluationScenario:
        requested = self.primary_scenario if name is None else name
        for scenario in self.scenarios:
            if scenario.name == requested:
                return scenario
        raise KeyError(f"Bilinmeyen evaluation scenario: {requested}")


def load_phase6_protocol(
    path: str | Path = DEFAULT_PHASE6_PROTOCOL,
) -> Phase6Protocol:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Faz 6 protokolü bulunamadı: {config_path}")
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    experiment = raw["experiment"]
    thresholds = raw["thresholds"]
    protocol = Phase6Protocol(
        training_seeds=tuple(experiment["training_seeds"]),
        held_out_episode_seeds=tuple(experiment["held_out_episode_seeds"]),
        pilot_timesteps=experiment["pilot_timesteps"],
        full_timesteps=experiment["full_timesteps"],
        primary_scenario=experiment["primary_scenario"],
        thresholds=EvaluationThresholds(
            minimum_success_rate=float(thresholds["minimum_success_rate"]),
            maximum_collision_rate=float(thresholds["maximum_collision_rate"]),
        ),
        scenarios=tuple(
            EvaluationScenario(
                name=scenario["name"],
                scene_mode=scenario["scene_mode"],
                endpoint_mode=scenario["endpoint_mode"],
                sensor_mode=scenario["sensor_mode"],
            )
            for scenario in raw["scenarios"]
        ),
    )
    protocol.validate()
    return protocol
