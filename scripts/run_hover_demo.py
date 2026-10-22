#!/usr/bin/env python3
"""Kademeli kontrolcü için headless hover gösterimi."""

from __future__ import annotations

import argparse
import math

import numpy as np

from ebabil.simulation import run_hover_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=5.0, help="Süre [s]")
    parser.add_argument(
        "--roll-deg", type=float, default=12.0, help="Başlangıç roll açısı [deg]"
    )
    parser.add_argument(
        "--offset-x", type=float, default=0.5, help="Başlangıç x konum hatası [m]"
    )
    parser.add_argument(
        "--wind-x", type=float, default=0.0, help="Dünya +x yönünde rüzgâr [m/s]"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_hover_simulation(
        duration_s=args.duration,
        initial_position_world_m=(args.offset_x, 0.0, 1.0),
        initial_rpy_rad=(math.radians(args.roll_deg), 0.0, 0.0),
        wind_world_m_s=(args.wind_x, 0.0, 0.0),
    )
    print(f"final_position_m={result.positions_world_m[-1]}")
    print(f"final_position_error_m={result.final_position_error_m:.6f}")
    print(f"final_speed_m_s={result.final_speed_m_s:.6f}")
    print(
        f"final_attitude_error_deg={math.degrees(result.final_attitude_error_rad):.6f}"
    )
    print(f"mixer_saturation_fraction={result.mixer_saturation_fraction:.6f}")
    print(f"torque_saturation_fraction={result.torque_saturation_fraction:.6f}")
    print(f"peak_collective_thrust_n={np.max(result.collective_thrust_n):.6f}")
    print(f"peak_body_torque_nm={np.max(np.abs(result.body_torque_nm), axis=0)}")


if __name__ == "__main__":
    main()
