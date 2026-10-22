"""İlk quadrotor MJCF modeli için yapısal ve fiziksel smoke testleri."""

from __future__ import annotations

import math
import unittest

import mujoco
import numpy as np

from ebabil.models import DEFAULT_QUADROTOR_XML, load_quadrotor

ROTOR_SITE_NAMES = (
    "rotor_front_left",
    "rotor_front_right",
    "rotor_rear_right",
    "rotor_rear_left",
)


class QuadrotorModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_quadrotor()

    def make_data(self) -> mujoco.MjData:
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        return data

    def test_model_file_exists_and_compiles(self) -> None:
        self.assertTrue(DEFAULT_QUADROTOR_XML.is_file())
        self.assertEqual(self.model.nq, 7)
        self.assertEqual(self.model.nv, 6)
        self.assertEqual(self.model.nu, 4)
        self.assertEqual(self.model.nsensor, 5)

    def test_mass_inertia_and_gravity(self) -> None:
        body_id = self.model.body("quadrotor").id
        self.assertAlmostEqual(float(self.model.body_mass[body_id]), 1.0, places=12)
        np.testing.assert_allclose(
            self.model.body_inertia[body_id],
            np.array([0.005, 0.005, 0.009]),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            self.model.opt.gravity,
            np.array([0.0, 0.0, -9.80665]),
            rtol=0.0,
            atol=1e-12,
        )

    def test_rotor_sites_follow_flu_x_configuration(self) -> None:
        expected_xy = {
            "rotor_front_left": (+1, +1),
            "rotor_front_right": (+1, -1),
            "rotor_rear_right": (-1, -1),
            "rotor_rear_left": (-1, +1),
        }
        expected_arm_length = 0.23

        for name in ROTOR_SITE_NAMES:
            site_id = self.model.site(name).id
            x, y, z = self.model.site_pos[site_id]
            expected_x_sign, expected_y_sign = expected_xy[name]
            self.assertEqual(int(math.copysign(1, x)), expected_x_sign)
            self.assertEqual(int(math.copysign(1, y)), expected_y_sign)
            self.assertAlmostEqual(float(z), 0.0, places=12)
            self.assertAlmostEqual(
                math.hypot(float(x), float(y)),
                expected_arm_length,
                delta=1e-6,
            )

    def test_free_fall_reduces_altitude(self) -> None:
        data = self.make_data()
        initial_z = float(data.qpos[2])
        for _ in range(100):
            mujoco.mj_step(self.model, data)
        self.assertLess(float(data.qpos[2]), initial_z)

    def test_free_fall_settles_on_ground_without_penetration(self) -> None:
        data = self.make_data()
        for _ in range(1_500):
            mujoco.mj_step(self.model, data)

        self.assertGreater(float(data.qpos[2]), 0.03)
        self.assertLess(float(data.qpos[2]), 0.05)
        self.assertLess(float(np.linalg.norm(data.qvel)), 1e-6)
        self.assertGreater(data.ncon, 0)

    def test_equal_hover_thrust_balances_gravity(self) -> None:
        data = self.make_data()
        hover_thrust_per_rotor = 9.80665 / 4.0
        data.ctrl[:] = hover_thrust_per_rotor
        mujoco.mj_forward(self.model, data)

        np.testing.assert_allclose(data.qacc, np.zeros(6), rtol=0.0, atol=1e-10)

        for _ in range(250):
            mujoco.mj_step(self.model, data)
        self.assertAlmostEqual(float(data.qpos[2]), 1.0, places=9)
        np.testing.assert_allclose(data.qvel, np.zeros(6), rtol=0.0, atol=1e-9)

    def test_actuator_yaw_signs_cancel_in_hover(self) -> None:
        expected = np.array([+0.02, -0.02, +0.02, -0.02])
        np.testing.assert_allclose(
            self.model.actuator_gear[:, 5],
            expected,
            rtol=0.0,
            atol=1e-12,
        )
        self.assertAlmostEqual(float(expected.sum()), 0.0, places=12)

    def test_differential_thrust_has_expected_body_moment_signs(self) -> None:
        hover = 9.80665 / 4.0
        delta = 0.2

        roll_data = self.make_data()
        roll_data.ctrl[:] = np.array(
            [hover + delta, hover - delta, hover - delta, hover + delta]
        )
        mujoco.mj_forward(self.model, roll_data)
        self.assertGreater(float(roll_data.qacc[3]), 0.0)
        self.assertAlmostEqual(float(roll_data.qacc[4]), 0.0, delta=1e-10)

        pitch_data = self.make_data()
        pitch_data.ctrl[:] = np.array(
            [hover + delta, hover + delta, hover - delta, hover - delta]
        )
        mujoco.mj_forward(self.model, pitch_data)
        self.assertLess(float(pitch_data.qacc[4]), 0.0)
        self.assertAlmostEqual(float(pitch_data.qacc[3]), 0.0, delta=1e-10)

        yaw_data = self.make_data()
        yaw_data.ctrl[:] = np.array(
            [hover + delta, hover - delta, hover + delta, hover - delta]
        )
        mujoco.mj_forward(self.model, yaw_data)
        self.assertGreater(float(yaw_data.qacc[5]), 0.0)

    def test_sensor_layout_is_stable(self) -> None:
        expected_dimensions = {
            "position": 3,
            "orientation": 4,
            "linear_velocity_body": 3,
            "angular_velocity_body": 3,
            "specific_force_body": 3,
        }
        self.assertEqual(self.model.nsensordata, sum(expected_dimensions.values()))
        for name, dimension in expected_dimensions.items():
            sensor_id = self.model.sensor(name).id
            self.assertEqual(int(self.model.sensor_dim[sensor_id]), dimension)


if __name__ == "__main__":
    unittest.main()
