#!/usr/bin/env python3
"""Mixer ve rotor step-response parametrelerini yazdır."""

from __future__ import annotations

import numpy as np

from ebabil.config import DEFAULT_VEHICLE_CONFIG, load_vehicle_config
from ebabil.dynamics import RotorDynamics, XQuadrotorMixer


def main() -> None:
    config = load_vehicle_config()
    mixer = XQuadrotorMixer.from_config(config)
    rotors = RotorDynamics(config.rotor)
    hover = config.hover_thrust_per_rotor_n

    print(f"config: {DEFAULT_VEHICLE_CONFIG}")
    print(f"hover_thrust_per_rotor_n: {hover:.6f}")
    print(
        "hover_rotor_speed_rad_s: "
        f"{np.sqrt(hover / config.rotor.thrust_coefficient):.6f}"
    )
    print(f"thrust_to_weight_ratio: {config.thrust_to_weight_ratio:.6f}")
    print("allocation_matrix_[Fz_tauX_tauY_tauZ]:")
    print(mixer.allocation_matrix)

    target = np.full(config.rotor.count, hover, dtype=np.float64)
    for step in range(1, 251):
        output = rotors.step(target, dt_s=0.002)
        if step in (1, 15, 50, 250):
            print(
                f"t={step * 0.002:.3f}s "
                f"speed={output.rotor_speeds_rad_s[0]:.3f}rad/s "
                f"thrust={output.actual_thrusts_n[0]:.6f}N"
            )


if __name__ == "__main__":
    main()
