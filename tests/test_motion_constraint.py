"""ADR-024 kamera konisi hareket kısıtı: geometri ve config."""

from __future__ import annotations

import math
import unittest

import numpy as np

from ebabil.envs.motion_constraint import (
    MotionConstraintConfig,
    constrain_horizontal_velocity,
    project_to_camera_cone,
)

HALF = math.radians(30.0)


class ConeProjectionTests(unittest.TestCase):
    def test_inside_cone_is_unchanged(self) -> None:
        for velocity in ((1.5, 0.0), (1.0, 0.5), (1.0, -0.57)):
            np.testing.assert_allclose(project_to_camera_cone(velocity, HALF), velocity)

    def test_lateral_command_projects_onto_cone_edge(self) -> None:
        projected = project_to_camera_cone((0.0, 1.0), HALF)
        edge = np.array([math.cos(HALF), math.sin(HALF)])
        np.testing.assert_allclose(projected, 0.5 * edge, atol=1e-12)
        mirrored = project_to_camera_cone((0.0, -1.0), HALF)
        np.testing.assert_allclose(
            mirrored, [0.5 * edge[0], -0.5 * edge[1]], atol=1e-12
        )

    def test_backward_command_is_zeroed(self) -> None:
        # Kenar ışınından 90 dereceden fazla uzak (açı > 30 + 90) komutlar sıfırlanır.
        for velocity in ((-1.0, 0.0), (-1.0, 0.3), (-1.0, -0.5)):
            np.testing.assert_allclose(
                project_to_camera_cone(velocity, HALF), (0.0, 0.0)
            )

    def test_lateral_slightly_backward_projects_forward_onto_edge(self) -> None:
        projected = project_to_camera_cone((-0.2, -1.0), HALF)
        self.assertGreater(projected[0], 0.0)
        self.assertAlmostEqual(math.atan2(projected[1], projected[0]), -HALF, places=12)
        self.assertLess(np.linalg.norm(projected), 0.4)

    def test_projection_never_increases_speed_and_is_idempotent(self) -> None:
        rng = np.random.default_rng(0)
        for velocity in rng.uniform(-1.5, 1.5, size=(200, 2)):
            projected = project_to_camera_cone(velocity, HALF)
            self.assertLessEqual(
                np.linalg.norm(projected), np.linalg.norm(velocity) + 1e-12
            )
            angle = math.atan2(projected[1], projected[0]) if np.any(projected) else 0.0
            self.assertLessEqual(abs(angle), HALF + 1e-9)
            np.testing.assert_allclose(
                project_to_camera_cone(projected, HALF), projected
            )


class ConstraintModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MotionConstraintConfig(enabled=True)

    def test_relaxed_mode_limits_blind_component(self) -> None:
        backward = constrain_horizontal_velocity((-1.5, 0.0), self.config, relaxed=True)
        np.testing.assert_allclose(backward, (-0.5, 0.0), atol=1e-12)
        slow = constrain_horizontal_velocity((-0.2, 0.1), self.config, relaxed=True)
        np.testing.assert_allclose(slow, (-0.2, 0.1), atol=1e-12)
        strict = constrain_horizontal_velocity((-0.2, 0.1), self.config, relaxed=False)
        np.testing.assert_allclose(strict, (0.0, 0.0))

    def test_config_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "half_angle"):
            MotionConstraintConfig(half_angle_deg=90.0).validate()
        with self.assertRaisesRegex(ValueError, "negatif"):
            MotionConstraintConfig(terminal_blind_speed_m_s=-0.1).validate()
        with self.assertRaisesRegex(ValueError, "bilinmeyen"):
            MotionConstraintConfig(relaxed_states=("hover",)).validate()


if __name__ == "__main__":
    unittest.main()
