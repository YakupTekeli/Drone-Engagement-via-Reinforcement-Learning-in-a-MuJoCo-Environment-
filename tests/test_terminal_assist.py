"""ADR-025 terminal hover yardımcısı: düzey çerçeve, hedef merkezi tahmini, komut yönleri."""

from __future__ import annotations

import math
import unittest

import mujoco
import numpy as np

from ebabil.controllers import read_flight_state
from ebabil.missions.terminal_assist import (
    TargetOffsetEstimate,
    TerminalAssistConfig,
    assist_action,
    estimate_target_offset,
    level_frame,
)
from ebabil.perception import load_camera_config
from ebabil.perception.camera import project_world_points
from ebabil.perception.detection_slots import CameraGeometry
from ebabil.perception.detections import TARGET_CLASS_ID, Detection
from ebabil.scenes import (
    generate_static_scene,
    load_scene_config,
    load_target_area_config,
    target_area_from_scene,
)
from ebabil.scenes.generator import build_scene_model

LEVEL = np.array([0.0, 0.0, -1.0])


def _rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


class LevelFrameTests(unittest.TestCase):
    def test_level_vehicle_gives_identity(self) -> None:
        np.testing.assert_allclose(level_frame(LEVEL), np.eye(3), atol=1e-12)

    def test_tilted_vehicle_keeps_horizontal_axes(self) -> None:
        rotation = _rotation(math.radians(5.0), math.radians(-8.0), 0.0)
        frame = level_frame(rotation.T @ LEVEL)
        np.testing.assert_allclose(
            frame[2], rotation.T @ np.array([0, 0, 1.0]), atol=1e-12
        )
        self.assertAlmostEqual(float(frame[0] @ frame[2]), 0.0, places=12)
        self.assertAlmostEqual(
            float(np.linalg.norm(np.cross(frame[0], frame[1]) - frame[2])),
            0.0,
            places=12,
        )


class TargetOffsetEstimateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = load_camera_config()
        self.geometry = CameraGeometry.from_camera_config(self.camera)
        self.target_config = load_target_area_config()
        self.scene = generate_static_scene(20260930, load_scene_config(), mode="clear")
        self.model = build_scene_model(
            self.scene,
            camera_config=self.camera,
            target_config=self.target_config,
            include_legacy_goal_marker=False,
        )
        self.area = target_area_from_scene(self.scene, self.target_config)
        self.config = TerminalAssistConfig(enabled=True)

    def _box_at(
        self,
        offset_world: np.ndarray,
        yaw: float,
        roll: float = 0.0,
        pitch: float = 0.0,
    ):
        data = mujoco.MjData(self.model)
        free = next(
            j
            for j in range(self.model.njnt)
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        )
        address = int(self.model.jnt_qposadr[free])
        anchor = self.area.hover_anchor_world_m
        position = np.array(
            [anchor[0] - offset_world[0], anchor[1] - offset_world[1], anchor[2]]
        )
        rotation = _rotation(roll, pitch, yaw)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
        data.qpos[address : address + 3] = position
        data.qpos[address + 3 : address + 7] = quaternion
        mujoco.mj_forward(self.model, data)
        # Dedektör kutusu görünür piksellerden gelir: hedef yüzeyinden örneklenen
        # noktaların kameranın önünde ve kadraj içinde olanlarının kutusu.
        corners = self.area.corners_world_m
        grid = np.linspace(0.0, 1.0, 61)
        points = np.array(
            [
                corners[0]
                + a * (corners[1] - corners[0])
                + b * (corners[3] - corners[0])
                for a in grid
                for b in grid
            ]
        )
        projection = project_world_points(self.model, data, points, self.camera)
        visible = projection.inside_image
        self.assertGreater(int(visible.sum()), 50)
        pixels = projection.pixels_xy[visible]
        x0 = float(pixels[:, 0].min() + 0.5)
        x1 = float(pixels[:, 0].max() + 0.5)
        y0 = float(pixels[:, 1].min() + 0.5)
        y1 = float(pixels[:, 1].max() + 0.5)
        box = Detection(TARGET_CLASS_ID, x0, y0, x1, y1, 0.9)
        state = read_flight_state(self.model, data)
        gravity = np.asarray(state.rotation_body_to_world).T @ LEVEL
        truth_world = self.area.hover_anchor_world_m[:2] - position[:2]
        truth_level = _rotation(0.0, 0.0, yaw)[:2, :2].T @ truth_world
        return box, gravity, float(state.position_world_m[2]), truth_level

    def test_estimate_tracks_true_offset_from_several_poses(self) -> None:
        # Toleranslar 29-30 Eylül ölçümünden: güvenli bölge içinde (<= 0.65 m)
        # hata p90 0.42 / max 0.59 m; uzakta yanal hata büyür, yaklaşınca düzelir.
        cases = [
            ((1.5, 0.0), 0.0, 0.9),
            ((1.8, -0.6), math.radians(35.0), 0.9),
            ((0.0, 0.0), 0.0, 0.6),
            ((0.3, 0.4), math.radians(-70.0), 0.6),
            ((0.5, -0.2), math.radians(120.0), 0.6),
        ]
        for offset, yaw, tolerance in cases:
            with self.subTest(offset=offset, yaw=round(math.degrees(yaw))):
                direction = _rotation(0.0, 0.0, yaw)[:2, :2] @ np.array(offset)
                box, gravity, altitude, truth = self._box_at(direction, yaw)
                estimate = estimate_target_offset(
                    box,
                    self.geometry,
                    self.config,
                    gravity_direction_body=gravity,
                    altitude_m=altitude,
                )
                assert estimate is not None
                error = float(np.linalg.norm(estimate.offset_xy_m - truth))
                self.assertLess(
                    error,
                    tolerance,
                    msg=f"tahmin {estimate.offset_xy_m}, gerçek {truth}",
                )

    def test_estimate_inside_safe_region_keeps_vehicle_inside(self) -> None:
        # Rastgele 60 pozda (<= 0.65 m, her yön ve yaw): 30 Eylül ölçümü 397 pozda
        # p50 0.24 / p90 0.50 / max 0.77 m; yaw kontrolü hedefi öne alınca hata küçülür.
        rng = np.random.default_rng(1)
        errors = []
        while len(errors) < 60:
            radius = rng.uniform(0.0, 0.65)
            angle = rng.uniform(-math.pi, math.pi)
            yaw = rng.uniform(-math.pi, math.pi)
            offset = np.array([radius * math.cos(angle), radius * math.sin(angle)])
            box, gravity, altitude, truth = self._box_at(
                _rotation(0.0, 0.0, yaw)[:2, :2] @ offset, yaw
            )
            estimate = estimate_target_offset(
                box,
                self.geometry,
                self.config,
                gravity_direction_body=gravity,
                altitude_m=altitude,
            )
            assert estimate is not None
            errors.append(float(np.linalg.norm(estimate.offset_xy_m - truth)))
        self.assertLess(max(errors), 0.85)
        self.assertLess(float(np.percentile(errors, 90)), 0.6)

    def test_estimate_with_tilt_stays_within_safe_region_margin(self) -> None:
        direction = np.array([0.4, -0.3])
        box, gravity, altitude, truth = self._box_at(
            direction, 0.0, roll=math.radians(4.0), pitch=math.radians(6.0)
        )
        estimate = estimate_target_offset(
            box,
            self.geometry,
            self.config,
            gravity_direction_body=gravity,
            altitude_m=altitude,
        )
        assert estimate is not None
        self.assertLess(float(np.linalg.norm(estimate.offset_xy_m - truth)), 0.45)


class AssistActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TerminalAssistConfig(enabled=True)
        self.scales = dict(
            max_horizontal_speed_m_s=1.5,
            max_vertical_speed_m_s=1.0,
            max_yaw_rate_rad_s=1.5,
        )

    def test_action_points_toward_target_and_is_capped(self) -> None:
        estimate = TargetOffsetEstimate(np.array([2.0, 1.0]), True, 0.5)
        action = assist_action(
            estimate, altitude_error_m=0.4, config=self.config, **self.scales
        )
        horizontal = action[:2] * 1.5
        self.assertAlmostEqual(float(np.linalg.norm(horizontal)), 0.5, places=5)
        self.assertGreater(horizontal[0], 0.0)
        self.assertGreater(horizontal[1], 0.0)
        self.assertAlmostEqual(float(action[2] * 1.0), 0.3, places=6)  # dikey sınır
        self.assertLess(action[3], 0.0)  # hedef sağda -> sağa dön (negatif yaw-rate)

    def test_small_offset_gives_proportional_slow_command(self) -> None:
        estimate = TargetOffsetEstimate(np.array([-0.2, 0.1]), True, 0.0)
        action = assist_action(
            estimate, altitude_error_m=-0.05, config=self.config, **self.scales
        )
        np.testing.assert_allclose(action[:2] * 1.5, (-0.2, 0.1), atol=1e-6)
        self.assertAlmostEqual(float(action[2]), -0.05, places=6)
        self.assertEqual(float(action[3]), 0.0)

    def test_missing_target_hovers_in_place(self) -> None:
        action = assist_action(
            None, altitude_error_m=0.0, config=self.config, **self.scales
        )
        np.testing.assert_array_equal(action, np.zeros(4, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
