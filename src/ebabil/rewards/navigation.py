"""Ayrı loglanabilir navigasyon reward hesabı."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ebabil.envs.config import RewardConfig


@dataclass(frozen=True)
class RewardBreakdown:
    progress: float
    goal: float
    collision: float
    failure: float
    proximity: float
    smoothness: float
    attitude: float
    time: float

    @property
    def total(self) -> float:
        return float(sum(vars(self).values()))

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in vars(self).items()}


def compute_navigation_reward(
    config: RewardConfig,
    *,
    previous_distance_m: float,
    current_distance_m: float,
    minimum_range_m: float,
    current_action: npt.ArrayLike,
    previous_action: npt.ArrayLike,
    body_z_world_z: float,
    success: bool,
    collision: bool,
    other_failure: bool,
) -> RewardBreakdown:
    """İşaretleri açık, bileşen bazlı tek policy-step reward üret."""

    action = np.asarray(current_action, dtype=np.float64)
    prior_action = np.asarray(previous_action, dtype=np.float64)
    progress = config.progress_weight * (previous_distance_m - current_distance_m)
    proximity_ratio = np.clip(
        (config.proximity_threshold_m - minimum_range_m) / config.proximity_threshold_m,
        0.0,
        1.0,
    )
    return RewardBreakdown(
        progress=float(progress),
        goal=config.goal_bonus if success else 0.0,
        collision=-config.collision_penalty if collision else 0.0,
        failure=-config.failure_penalty if other_failure else 0.0,
        proximity=-config.proximity_weight * float(proximity_ratio**2),
        smoothness=-config.smoothness_weight
        * float(np.sum(np.square(action - prior_action))),
        attitude=-config.attitude_weight
        * float(np.square(1.0 - np.clip(body_z_world_z, -1.0, 1.0))),
        time=-config.time_penalty,
    )
