"""Faz 6 çoklu-seed eğitim, evaluation ve kalite kapısı orkestrasyonu."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ebabil.evaluation import (
    Phase6Protocol,
    evaluate_baseline_suite,
    evaluate_policy,
)
from ebabil.training import TrainingConfig, train_ppo

ExperimentStage = Literal["pilot", "full"]


@dataclass(frozen=True)
class Phase6ExperimentArtifacts:
    output_directory: Path
    summary: Path
    metrics: dict[str, Any]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_phase6_experiment(
    config: TrainingConfig,
    protocol: Phase6Protocol,
    output_directory: str | Path,
    *,
    stage: ExperimentStage,
    training_seeds: tuple[int, ...] | None = None,
    held_out_episode_seeds: tuple[int, ...] | None = None,
    record_replays_per_seed: int = 1,
    device: str = "auto",
) -> Phase6ExperimentArtifacts:
    """Baseline'ları ve PPO seed'lerini çalıştırıp aggregate kalite raporu üret."""

    config.validate()
    protocol.validate()
    if stage not in ("pilot", "full"):
        raise ValueError("stage 'pilot' veya 'full' olmalı")
    selected_training_seeds = training_seeds or protocol.training_seeds
    selected_held_out_seeds = held_out_episode_seeds or protocol.held_out_episode_seeds
    if not selected_training_seeds or not selected_held_out_seeds:
        raise ValueError("Training ve held-out seed listeleri boş olamaz")
    if record_replays_per_seed < 0 or record_replays_per_seed > len(
        selected_held_out_seeds
    ):
        raise ValueError("record_replays_per_seed held-out sınırları içinde olmalı")
    timesteps = (
        protocol.pilot_timesteps if stage == "pilot" else protocol.full_timesteps
    )
    scenario = protocol.scenario()
    experiment_config = replace(
        config,
        environment=replace(
            config.environment,
            scene_mode=scenario.scene_mode,
            endpoint_mode=scenario.endpoint_mode,
            sensor_mode=scenario.sensor_mode,
        ),
    )
    output_path = Path(output_directory).expanduser().resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Deney dizini boş değil: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    status_path = output_path / "experiment.json"
    status: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "stage": stage,
        "timesteps_per_seed": timesteps,
        "scenario": asdict(scenario),
        "training_seeds": list(selected_training_seeds),
        "held_out_episode_seeds": list(selected_held_out_seeds),
        "completed_training_seeds": [],
    }
    _write_json(status_path, status)

    try:
        baseline_comparison_path = evaluate_baseline_suite(
            experiment_config,
            output_path / "baselines",
            selected_held_out_seeds,
        )
        baseline_comparison = json.loads(
            baseline_comparison_path.read_text(encoding="utf-8")
        )
        seed_rows: list[dict[str, Any]] = []
        for training_seed in selected_training_seeds:
            seed_config = replace(
                experiment_config,
                run=replace(
                    experiment_config.run,
                    seed=training_seed,
                    total_timesteps=timesteps,
                ),
            )
            seed_root = output_path / "ppo" / f"seed_{training_seed}"
            training = train_ppo(
                seed_config,
                seed_root / "training",
                device=device,
            )
            checkpoint = training.best_model or training.final_model
            normalization = (
                training.best_normalization
                if training.best_model is not None
                else training.normalization
            )
            evaluation = evaluate_policy(
                checkpoint,
                seed_config,
                seed_root / "held_out",
                episodes=len(selected_held_out_seeds),
                episode_seeds=selected_held_out_seeds,
                record_replays=record_replays_per_seed,
                normalization_path=normalization,
                device=device,
            )
            row = {
                "training_seed": training_seed,
                "actual_total_timesteps": training.actual_total_timesteps,
                "checkpoint": str(checkpoint),
                "normalization": str(normalization) if normalization else None,
                "success_rate": evaluation.metrics["success_rate"],
                "collision_rate": evaluation.metrics["collision_rate"],
                "mean_return": evaluation.metrics["mean_return"],
                "mean_episode_length": evaluation.metrics["mean_episode_length"],
                "summary": str(evaluation.summary),
            }
            seed_rows.append(row)
            status["completed_training_seeds"] = [
                item["training_seed"] for item in seed_rows
            ]
            _write_json(status_path, status)

        success_rates = np.asarray(
            [row["success_rate"] for row in seed_rows], dtype=np.float64
        )
        collision_rates = np.asarray(
            [row["collision_rate"] for row in seed_rows], dtype=np.float64
        )
        mean_returns = np.asarray(
            [row["mean_return"] for row in seed_rows], dtype=np.float64
        )
        random_return = float(baseline_comparison["policies"]["random"]["mean_return"])
        direct_goal_return = float(
            baseline_comparison["policies"]["direct_goal"]["mean_return"]
        )
        performance_checks = {
            "every_seed_meets_success_target": bool(
                np.all(success_rates >= protocol.thresholds.minimum_success_rate)
            ),
            "every_seed_meets_collision_target": bool(
                np.all(collision_rates <= protocol.thresholds.maximum_collision_rate)
            ),
            "mean_ppo_return_exceeds_random": bool(
                float(np.mean(mean_returns)) > random_return
            ),
            "mean_ppo_return_exceeds_direct_goal": bool(
                float(np.mean(mean_returns)) > direct_goal_return
            ),
        }
        protocol_coverage = {
            "full_stage": stage == "full",
            "all_training_seeds": set(selected_training_seeds)
            == set(protocol.training_seeds),
            "all_held_out_episode_seeds": set(selected_held_out_seeds)
            == set(protocol.held_out_episode_seeds),
        }
        performance_checks_passed = all(performance_checks.values())
        metrics: dict[str, Any] = {
            "schema_version": 1,
            "status": "completed",
            "stage": stage,
            "timesteps_per_seed": timesteps,
            "scenario": asdict(scenario),
            "thresholds": asdict(protocol.thresholds),
            "training_seeds": list(selected_training_seeds),
            "held_out_episode_seeds": list(selected_held_out_seeds),
            "ppo_seeds": seed_rows,
            "aggregate": {
                "mean_success_rate": float(np.mean(success_rates)),
                "std_success_rate": float(np.std(success_rates)),
                "mean_collision_rate": float(np.mean(collision_rates)),
                "mean_return": float(np.mean(mean_returns)),
            },
            "baselines": baseline_comparison["policies"],
            "performance_checks": performance_checks,
            "performance_checks_passed": performance_checks_passed,
            "protocol_coverage": protocol_coverage,
            "quality_gate_passed": performance_checks_passed
            and all(protocol_coverage.values()),
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        summary_path = output_path / "summary.json"
        _write_json(summary_path, metrics)
        status.update(
            {
                "status": "completed",
                "completed_at_utc": metrics["completed_at_utc"],
                "summary": str(summary_path),
            }
        )
        _write_json(status_path, status)
        return Phase6ExperimentArtifacts(output_path, summary_path, metrics)
    except Exception as error:
        status["status"] = "failed"
        status["failed_at_utc"] = datetime.now(UTC).isoformat()
        status["error"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(status_path, status)
        raise
