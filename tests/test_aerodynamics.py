"""Quadratic gövde sürüklemesi ve rüzgâr arayüzü testleri."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from ebabil.config import load_vehicle_config
from ebabil.dynamics import QuadraticBodyDrag
from ebabil.models import load_quadrotor


class QuadraticBodyDragTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_vehicle_config()
        self.drag = QuadraticBodyDrag.from_config(self.config)

    def test_zero_relative_air_velocity_has_zero_force(self) -> None:
        velocity = np.array([2.0, -1.0, 0.5])
        np.testing.assert_allclose(
            self.drag.force_body(velocity, velocity),
            np.zeros(3),
            rtol=0.0,
            atol=0.0,
        )

    def test_drag_opposes_relative_air_velocity(self) -> None:
        velocity = np.array([1.0, -2.0, 0.5])
        force = self.drag.force_body(velocity, np.zeros(3))
        expected = np.array([-0.08, 0.32, -0.03])
        np.testing.assert_allclose(force, expected, rtol=0.0, atol=1e-12)
        self.assertTrue(np.all(force * velocity <= 0.0))

    def test_stationary_vehicle_is_pushed_with_wind(self) -> None:
        model = load_quadrotor()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        output = self.drag.apply(model, data, wind_world_m_s=[2.0, 0.0, 0.0])

        self.assertGreater(float(output.force_world_n[0]), 0.0)
        body_id = model.body("quadrotor").id
        np.testing.assert_allclose(
            data.xfrc_applied[body_id, :3],
            output.force_world_n,
            rtol=0.0,
            atol=1e-12,
        )

    def test_drag_reduces_horizontal_speed_in_mujoco(self) -> None:
        model = load_quadrotor()
        model.opt.gravity[:] = 0.0
        data = mujoco.MjData(model)
        data.qvel[0] = 2.0

        initial_speed = float(data.qvel[0])
        for _ in range(500):
            data.xfrc_applied[:] = 0.0
            mujoco.mj_forward(model, data)
            self.drag.apply(model, data, wind_world_m_s=np.zeros(3))
            mujoco.mj_step(model, data)

        self.assertGreater(float(data.qvel[0]), 0.0)
        self.assertLess(float(data.qvel[0]), initial_speed)


if __name__ == "__main__":
    unittest.main()
