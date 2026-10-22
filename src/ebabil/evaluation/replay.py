"""Taşınabilir, pickle gerektirmeyen episode replay sözleşmesi."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class EpisodeReplay:
    time_s: FloatArray
    qpos: FloatArray
    qvel: FloatArray
    policy_time_s: FloatArray
    actions: FloatArray
    rewards: FloatArray
    distance_to_goal_m: FloatArray
    minimum_range_m: FloatArray
    reward_components: FloatArray
    rotor_thrusts_n: FloatArray
    velocity_command_world_m_s: FloatArray
    yaw_rate_command_rad_s: FloatArray
    mixer_saturated: BoolArray
    torque_saturated: BoolArray

    @property
    def action_size(self) -> int:
        """Action boyutu veriden türetilir.

        Legacy `EbabilNavigation-v0` ve `vision-action-v2` (ADR-020) dört
        boyutludur; `vision-action-v1` üç boyutluydu ve kayıtları hâlâ
        oynatılabilir olduğu için boyut sabitlenmez.
        """

        if self.actions.ndim != 2 or self.actions.shape[1] <= 0:
            raise ValueError("Replay actions shape'i (adım, action) olmalı")
        return int(self.actions.shape[1])

    @property
    def reward_component_count(self) -> int:
        """Reward bileşen sayısı veriden türetilir: legacy `8`, terminal hover `11`."""

        if self.reward_components.ndim != 2 or self.reward_components.shape[1] <= 0:
            raise ValueError("Replay reward_components shape'i (adım, bileşen) olmalı")
        return int(self.reward_components.shape[1])

    def validate(self) -> None:
        frame_count = self.time_s.shape[0]
        step_count = self.actions.shape[0]
        physics_step_count = frame_count - 1
        action_size = self.action_size
        component_count = self.reward_component_count
        if self.time_s.ndim != 1 or frame_count < 2:
            raise ValueError("Replay en az iki fizik frame'i içermeli")
        if self.qpos.ndim != 2 or self.qpos.shape[0] != frame_count:
            raise ValueError("Replay qpos shape'i geçersiz")
        if self.qvel.ndim != 2 or self.qvel.shape[0] != frame_count:
            raise ValueError("Replay qvel shape'i geçersiz")
        expected_shapes = {
            "policy_time_s": (step_count,),
            "actions": (step_count, action_size),
            "rewards": (step_count,),
            "distance_to_goal_m": (step_count + 1,),
            "minimum_range_m": (step_count,),
            "reward_components": (step_count, component_count),
            "rotor_thrusts_n": (physics_step_count, 4),
            "velocity_command_world_m_s": (physics_step_count, 3),
            "yaw_rate_command_rad_s": (physics_step_count,),
            "mixer_saturated": (physics_step_count,),
            "torque_saturated": (physics_step_count,),
        }
        for name, expected in expected_shapes.items():
            if getattr(self, name).shape != expected:
                raise ValueError(f"Replay {name} shape'i geçersiz")
        numeric_arrays = (
            self.time_s,
            self.qpos,
            self.qvel,
            self.policy_time_s,
            self.actions,
            self.rewards,
            self.distance_to_goal_m,
            self.minimum_range_m,
            self.reward_components,
            self.rotor_thrusts_n,
            self.velocity_command_world_m_s,
            self.yaw_rate_command_rad_s,
        )
        if not all(np.all(np.isfinite(array)) for array in numeric_arrays):
            raise ValueError("Replay sonlu olmayan değer içeriyor")
        if frame_count > 1 and np.any(np.diff(self.time_s) <= 0.0):
            raise ValueError("Replay zamanları kesin artan olmalı")


def save_replay(
    path: str | Path,
    replay: EpisodeReplay,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    replay.validate()
    output_path = Path(path).expanduser().resolve().with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".json")
    np.savez_compressed(
        output_path,
        time_s=replay.time_s,
        qpos=replay.qpos,
        qvel=replay.qvel,
        policy_time_s=replay.policy_time_s,
        actions=replay.actions,
        rewards=replay.rewards,
        distance_to_goal_m=replay.distance_to_goal_m,
        minimum_range_m=replay.minimum_range_m,
        reward_components=replay.reward_components,
        rotor_thrusts_n=replay.rotor_thrusts_n,
        velocity_command_world_m_s=replay.velocity_command_world_m_s,
        yaw_rate_command_rad_s=replay.yaw_rate_command_rad_s,
        mixer_saturated=replay.mixer_saturated,
        torque_saturated=replay.torque_saturated,
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path, metadata_path


def load_replay(path: str | Path) -> tuple[EpisodeReplay, dict[str, Any]]:
    input_path = Path(path).expanduser().resolve()
    metadata_path = input_path.with_suffix(".json")
    with np.load(input_path, allow_pickle=False) as archive:
        replay = EpisodeReplay(
            time_s=archive["time_s"].astype(np.float64, copy=False),
            qpos=archive["qpos"].astype(np.float64, copy=False),
            qvel=archive["qvel"].astype(np.float64, copy=False),
            policy_time_s=archive["policy_time_s"].astype(np.float64, copy=False),
            actions=archive["actions"].astype(np.float64, copy=False),
            rewards=archive["rewards"].astype(np.float64, copy=False),
            distance_to_goal_m=archive["distance_to_goal_m"].astype(
                np.float64, copy=False
            ),
            minimum_range_m=archive["minimum_range_m"].astype(np.float64, copy=False),
            reward_components=archive["reward_components"].astype(
                np.float64, copy=False
            ),
            rotor_thrusts_n=archive["rotor_thrusts_n"].astype(np.float64, copy=False),
            velocity_command_world_m_s=archive["velocity_command_world_m_s"].astype(
                np.float64, copy=False
            ),
            yaw_rate_command_rad_s=archive["yaw_rate_command_rad_s"].astype(
                np.float64, copy=False
            ),
            mixer_saturated=archive["mixer_saturated"].astype(np.bool_, copy=False),
            torque_saturated=archive["torque_saturated"].astype(np.bool_, copy=False),
        )
    replay.validate()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("Desteklenmeyen replay metadata sürümü")
    return replay, metadata
