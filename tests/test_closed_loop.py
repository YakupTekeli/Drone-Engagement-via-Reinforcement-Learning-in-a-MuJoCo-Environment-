"""MuJoCo üzerinde kademeli kontrolcü kapalı çevrim testleri."""

from __future__ import annotations

import math
import unittest

from ebabil.simulation import run_hover_simulation


class ClosedLoopTests(unittest.TestCase):
    def test_level_hover_remains_stationary(self) -> None:
        result = run_hover_simulation(duration_s=2.0)

        self.assertEqual(result.qpos.shape, (result.time_s.size, 7))
        self.assertEqual(result.qvel.shape, (result.time_s.size, 6))
        self.assertLess(result.final_position_error_m, 1e-6)
        self.assertLess(result.final_speed_m_s, 1e-6)
        self.assertLess(result.final_attitude_error_rad, 1e-6)

    def test_recovers_from_position_and_roll_perturbation(self) -> None:
        result = run_hover_simulation(
            duration_s=6.0,
            initial_position_world_m=(0.5, -0.3, 1.2),
            initial_rpy_rad=(math.radians(12.0), math.radians(-8.0), 0.0),
        )

        self.assertLess(result.final_position_error_m, 0.04)
        self.assertLess(result.final_speed_m_s, 0.06)
        self.assertLess(result.final_attitude_error_rad, math.radians(2.0))

    def test_constant_wind_disturbance_stays_bounded(self) -> None:
        result = run_hover_simulation(
            duration_s=8.0,
            wind_world_m_s=(1.0, 0.0, 0.0),
        )

        self.assertLess(result.final_position_error_m, 0.08)
        self.assertLess(result.final_speed_m_s, 0.05)
        self.assertLess(result.mixer_saturation_fraction, 0.05)

    def test_tracks_yaw_reference(self) -> None:
        result = run_hover_simulation(
            duration_s=4.0,
            target_yaw_rad=math.radians(45.0),
        )

        self.assertLess(result.final_position_error_m, 0.01)
        self.assertLess(result.final_attitude_error_rad, math.radians(1.0))


if __name__ == "__main__":
    unittest.main()
