"""Kademeli uçuş kontrolcüsü için analitik birim testleri."""

from __future__ import annotations

import math
import unittest

import numpy as np

from ebabil.config import load_vehicle_config
from ebabil.controllers import (
    AttitudeController,
    BodyRateController,
    FlightState,
    FlightTarget,
    PositionVelocityController,
    load_controller_config,
)


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vehicle = load_vehicle_config()
        cls.config = load_controller_config()

    @staticmethod
    def level_state() -> FlightState:
        return FlightState(
            position_world_m=np.array([0.0, 0.0, 1.0]),
            velocity_world_m_s=np.zeros(3),
            rotation_body_to_world=np.eye(3),
            angular_velocity_body_rad_s=np.zeros(3),
        )

    def test_level_target_requests_hover_thrust(self) -> None:
        controller = PositionVelocityController(self.config, self.vehicle)
        output = controller.compute(
            self.level_state(), FlightTarget(position_world_m=np.array([0.0, 0.0, 1.0]))
        )

        np.testing.assert_allclose(output.desired_rotation_body_to_world, np.eye(3))
        np.testing.assert_allclose(output.force_command_world_n[:2], np.zeros(2))
        self.assertAlmostEqual(
            output.collective_thrust_n,
            self.vehicle.mass_kg * self.vehicle.gravity_m_s2,
        )

    def test_positive_x_error_tilts_thrust_toward_positive_x(self) -> None:
        controller = PositionVelocityController(self.config, self.vehicle)
        output = controller.compute(
            self.level_state(), FlightTarget(position_world_m=np.array([2.0, 0.0, 1.0]))
        )

        body_z = output.desired_rotation_body_to_world[:, 2]
        tilt = math.acos(float(np.clip(body_z[2], -1.0, 1.0)))
        self.assertGreater(body_z[0], 0.0)
        self.assertAlmostEqual(body_z[1], 0.0, places=12)
        self.assertLessEqual(tilt, self.config.position.max_tilt_rad + 1e-12)

    def test_positive_roll_error_requests_negative_roll_rate(self) -> None:
        controller = AttitudeController(self.config)
        angle = math.radians(10.0)
        current = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(angle), -math.sin(angle)],
                [0.0, math.sin(angle), math.cos(angle)],
            ]
        )
        output = controller.compute(current, np.eye(3))

        self.assertLess(output.desired_body_rate_rad_s[0], 0.0)
        self.assertAlmostEqual(output.desired_body_rate_rad_s[1], 0.0, places=12)

    def test_positive_rate_error_requests_positive_torque(self) -> None:
        controller = BodyRateController(self.config, self.vehicle)
        output = controller.compute(
            desired_body_rate_rad_s=np.array([1.0, 0.0, 0.0]),
            current_body_rate_rad_s=np.zeros(3),
            dt_s=0.002,
        )

        self.assertGreater(output.applied_torque_nm[0], 0.0)
        self.assertAlmostEqual(output.applied_torque_nm[1], 0.0, places=12)
        self.assertAlmostEqual(output.applied_torque_nm[2], 0.0, places=12)

    def test_body_rate_integrator_does_not_wind_up_on_saturated_axis(self) -> None:
        controller = BodyRateController(self.config, self.vehicle)
        for _ in range(100):
            output = controller.compute(
                desired_body_rate_rad_s=np.array([100.0, 0.0, 0.0]),
                current_body_rate_rad_s=np.zeros(3),
                dt_s=0.01,
            )

        self.assertTrue(output.torque_saturated)
        self.assertAlmostEqual(output.integral_error_rad[0], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
