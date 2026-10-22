"""Rangefinder ve sensör hata modeli testleri."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from ebabil.scenes import Obstacle, StaticScene, build_scene_model
from ebabil.sensors import (
    NavigationSensorSuite,
    PlanarRangefinder,
    RangefinderConfig,
    SampledSensor,
    SensorImperfectionConfig,
    load_sensor_config,
)


def rangefinder_config(ray_count: int = 4) -> RangefinderConfig:
    return RangefinderConfig(
        ray_count=ray_count,
        field_of_view_rad=2.0 * np.pi,
        min_range_m=0.05,
        max_range_m=5.0,
        origin_body_m=(0.0, 0.0, 0.0),
        geom_group=1,
    )


def single_obstacle_scene(center: tuple[float, float, float]) -> StaticScene:
    return StaticScene(
        seed=0,
        start_world_m=np.array([0.0, 0.0, 1.0]),
        goal_world_m=np.array([4.0, 0.0, 1.0]),
        obstacles=(
            Obstacle(
                name="test_obstacle",
                center_world_m=np.asarray(center, dtype=np.float64),
                half_size_m=np.array([0.5, 0.5, 1.0]),
            ),
        ),
    )


class RangefinderTests(unittest.TestCase):
    def test_default_configuration_is_valid(self) -> None:
        config = load_sensor_config()
        config.validate()
        self.assertEqual(config.rangefinder.ray_count, 16)

    def test_front_ray_returns_analytic_box_distance(self) -> None:
        model = build_scene_model(single_obstacle_scene((2.0, 0.0, 1.0)))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        reading = PlanarRangefinder(rangefinder_config()).read(model, data)

        self.assertAlmostEqual(reading.distances_m[0], 1.5, places=10)
        self.assertEqual(model.geom(int(reading.hit_geom_ids[0])).name, "test_obstacle")
        np.testing.assert_allclose(reading.distances_m[1:], np.full(3, 5.0))

    def test_body_yaw_rotates_ray_directions_into_world(self) -> None:
        model = build_scene_model(single_obstacle_scene((0.0, 2.0, 1.0)))
        data = mujoco.MjData(model)
        yaw_90_wxyz = np.array([np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)])
        data.qpos[3:7] = yaw_90_wxyz
        mujoco.mj_forward(model, data)

        reading = PlanarRangefinder(rangefinder_config(ray_count=1)).read(model, data)

        np.testing.assert_allclose(
            reading.ray_directions_world[0], np.array([0.0, 1.0, 0.0]), atol=1e-12
        )
        self.assertAlmostEqual(reading.distances_m[0], 1.5, places=10)

    def test_no_hit_is_encoded_as_max_range(self) -> None:
        scene = StaticScene(
            seed=0,
            start_world_m=np.array([0.0, 0.0, 1.0]),
            goal_world_m=np.array([2.0, 0.0, 1.0]),
            obstacles=(),
        )
        model = build_scene_model(scene)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        reading = PlanarRangefinder(rangefinder_config(ray_count=1)).read(model, data)

        self.assertEqual(reading.distances_m[0], 5.0)
        self.assertEqual(reading.normalized_distances[0], 1.0)
        self.assertEqual(reading.hit_geom_ids[0], -1)


class SampledSensorTests(unittest.TestCase):
    @staticmethod
    def ideal_config(*, dropout: float = 0.0) -> SensorImperfectionConfig:
        return SensorImperfectionConfig(
            sample_period_s=0.01,
            delay_s=0.02,
            noise_std=0.0,
            bias_std=0.0,
            dropout_probability=dropout,
        )

    def test_sample_and_delay_timing(self) -> None:
        sensor = SampledSensor(self.ideal_config())
        sensor.reset([1.0], seed=4)

        first = sensor.update([2.0], 0.01)
        second = sensor.update([3.0], 0.01)
        third = sensor.update([4.0], 0.01)

        self.assertEqual(first.value[0], 1.0)
        self.assertEqual(second.value[0], 1.0)
        self.assertEqual(third.value[0], 2.0)
        self.assertAlmostEqual(third.measurement_time_s, 0.01)

    def test_dropout_holds_last_measurement(self) -> None:
        sensor = SampledSensor(self.ideal_config(dropout=1.0))
        sensor.reset([1.0], seed=8)

        output = sensor.update([9.0], 0.05)

        self.assertTrue(output.sampled)
        self.assertTrue(output.dropped_out)
        self.assertEqual(output.value[0], 1.0)

    def test_same_seed_reproduces_bias_noise_and_dropout(self) -> None:
        config = SensorImperfectionConfig(
            sample_period_s=0.01,
            delay_s=0.0,
            noise_std=0.1,
            bias_std=0.2,
            dropout_probability=0.25,
        )
        first = SampledSensor(config)
        second = SampledSensor(config)
        first_values = [first.reset([1.0, 2.0], seed=42).value]
        second_values = [second.reset([1.0, 2.0], seed=42).value]
        for index in range(10):
            true_value = np.array([index, -index], dtype=np.float64)
            first_values.append(first.update(true_value, 0.01).value)
            second_values.append(second.update(true_value, 0.01).value)

        np.testing.assert_allclose(first_values, second_values, rtol=0.0, atol=0.0)


class NavigationSensorSuiteTests(unittest.TestCase):
    def test_named_mujoco_sensors_and_rangefinder_share_one_reading(self) -> None:
        scene = StaticScene(
            seed=0,
            start_world_m=np.array([0.0, 0.0, 1.0]),
            goal_world_m=np.array([2.0, 0.0, 1.0]),
            obstacles=(),
        )
        model = build_scene_model(scene)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        suite = NavigationSensorSuite(load_sensor_config())

        reading = suite.reset(model, data, seed=12)

        np.testing.assert_allclose(reading.ideal.position_world_m, scene.start_world_m)
        np.testing.assert_allclose(
            reading.ideal.orientation_wxyz, np.array([1.0, 0.0, 0.0, 0.0])
        )
        self.assertEqual(reading.ideal.linear_velocity_body_m_s.shape, (3,))
        self.assertEqual(reading.ideal.angular_velocity_body_rad_s.shape, (3,))
        self.assertEqual(reading.ideal.specific_force_body_m_s2.shape, (3,))
        self.assertEqual(reading.rangefinder_m.shape, (16,))
        self.assertTrue(np.all(0.0 <= reading.rangefinder_normalized))
        self.assertTrue(np.all(reading.rangefinder_normalized <= 1.0))


if __name__ == "__main__":
    unittest.main()
