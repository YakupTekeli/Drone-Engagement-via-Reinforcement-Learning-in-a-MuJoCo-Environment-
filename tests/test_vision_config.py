"""Kamera ve hedef alan konfigürasyonu/geometri testleri."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np

from ebabil.perception import load_camera_config
from ebabil.scenes import TargetArea, load_target_area_config


class CameraConfigTests(unittest.TestCase):
    def test_default_imx477_config_is_valid(self) -> None:
        config = load_camera_config()

        self.assertEqual(config.name, "imx477")
        self.assertEqual(
            (config.sensor_width_px, config.sensor_height_px), (4056, 3040)
        )
        self.assertEqual((config.output_width_px, config.output_height_px), (320, 240))
        self.assertAlmostEqual(math.degrees(config.horizontal_fov_rad), 75.0)
        self.assertAlmostEqual(math.degrees(config.pitch_down_rad), 40.0)
        self.assertGreater(config.focal_length_px, 0.0)

    def test_camera_axes_are_orthonormal_and_look_forward_down(self) -> None:
        config = load_camera_config()
        x_axis = np.asarray(config.xyaxes_body[:3])
        y_axis = np.asarray(config.xyaxes_body[3:])
        z_axis = np.cross(x_axis, y_axis)
        optical_axis = -z_axis

        np.testing.assert_allclose(np.linalg.norm(x_axis), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(y_axis), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.dot(x_axis, y_axis), 0.0, atol=1e-12)
        self.assertGreater(optical_axis[0], 0.0)
        self.assertLess(optical_axis[2], 0.0)

    def test_invalid_camera_output_is_rejected(self) -> None:
        config = load_camera_config()

        with self.assertRaisesRegex(ValueError, "sensör çözünürlüğünü"):
            replace(config, output_width_px=config.sensor_width_px + 1).validate()


class TargetAreaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_target_area_config()
        self.target = TargetArea((2.0, -1.0), self.config)
        self.target.validate()

    def test_hover_anchor_uses_target_surface(self) -> None:
        np.testing.assert_allclose(
            self.target.hover_anchor_world_m,
            np.array([2.0, -1.0, 1.11]),
            atol=1e-12,
        )

    def test_safe_inner_region_includes_boundary(self) -> None:
        half_x, half_y = self.config.safe_half_size_xy_m

        self.assertTrue(
            self.target.contains_safe_xy((2.0 + half_x, -1.0 - half_y, 4.0))
        )
        self.assertFalse(self.target.contains_safe_xy((2.0 + half_x + 1e-6, -1.0, 2.0)))

    def test_target_corners_are_ordered_on_surface(self) -> None:
        corners = self.target.corners_world_m

        self.assertEqual(corners.shape, (4, 3))
        np.testing.assert_allclose(corners[:, 2], self.config.surface_z_m)
        np.testing.assert_allclose(corners.mean(axis=0)[:2], np.array([2.0, -1.0]))

    def test_margin_cannot_remove_safe_region(self) -> None:
        with self.assertRaisesRegex(ValueError, "güvenli iç bölgesini siliyor"):
            replace(self.config, safety_margin_m=1.0).validate()


if __name__ == "__main__":
    unittest.main()
