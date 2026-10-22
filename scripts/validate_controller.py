#!/usr/bin/env python3
"""Seed'li başlangıç dağılımında klasik kontrolcü doğrulaması."""

from __future__ import annotations

import argparse
import math

import numpy as np

from ebabil.simulation import run_hover_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=20260910)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples pozitif olmalı")

    rng = np.random.default_rng(args.seed)
    metrics: list[tuple[bool, float, float, float, float, float, float]] = []
    for _ in range(args.samples):
        position = np.array(
            [
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.5, 0.5),
                rng.uniform(0.8, 1.2),
            ]
        )
        rpy = np.radians(
            [
                rng.uniform(-15.0, 15.0),
                rng.uniform(-15.0, 15.0),
                rng.uniform(-30.0, 30.0),
            ]
        )
        result = run_hover_simulation(
            duration_s=args.duration,
            initial_position_world_m=position,
            initial_rpy_rad=rpy,
        )
        minimum_altitude = float(np.min(result.positions_world_m[:, 2]))
        success = (
            result.final_position_error_m < 0.05
            and result.final_speed_m_s < 0.08
            and result.final_attitude_error_rad < math.radians(2.0)
            and minimum_altitude > 0.10
        )
        metrics.append(
            (
                success,
                result.final_position_error_m,
                result.final_speed_m_s,
                math.degrees(result.final_attitude_error_rad),
                result.mixer_saturation_fraction,
                result.torque_saturation_fraction,
                minimum_altitude,
            )
        )

    values = np.asarray(metrics, dtype=np.float64)
    successes = int(np.sum(values[:, 0]))
    print(f"seed={args.seed}")
    print(f"success={successes}/{args.samples}")
    print(f"success_rate={successes / args.samples:.6f}")
    print(f"worst_final_position_error_m={np.max(values[:, 1]):.6f}")
    print(f"worst_final_speed_m_s={np.max(values[:, 2]):.6f}")
    print(f"worst_final_attitude_error_deg={np.max(values[:, 3]):.6f}")
    print(f"worst_mixer_saturation_fraction={np.max(values[:, 4]):.6f}")
    print(f"worst_torque_saturation_fraction={np.max(values[:, 5]):.6f}")
    print(f"minimum_altitude_m={np.min(values[:, 6]):.6f}")


if __name__ == "__main__":
    main()
