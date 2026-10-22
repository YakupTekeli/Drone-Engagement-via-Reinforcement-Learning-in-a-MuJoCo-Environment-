"""MuJoCo kamera projection ve actor/privileged capture testleri."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from ebabil.perception import (
    ActorCameraFrame,
    CameraRenderUnavailable,
    HeadlessCamera,
    load_camera_config,
    project_world_points,
)
from ebabil.scenes import (
    TargetArea,
    build_scene_model,
    generate_static_scene,
    load_target_area_config,
)


class CameraModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.camera_config = load_camera_config()
        cls.target_config = load_target_area_config()
        cls.scene = generate_static_scene(20260911, mode="clear")
        cls.target = TargetArea(
            tuple(cls.scene.goal_world_m[:2]),  # type: ignore[arg-type]
            cls.target_config,
        )
        cls.model = build_scene_model(cls.scene, include_legacy_goal_marker=False)

    def make_data(self) -> mujoco.MjData:
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        return data

    def test_scene_model_contains_camera_and_noncolliding_target(self) -> None:
        camera_id = self.model.camera(self.camera_config.name).id
        target_geom_ids = [
            index
            for index in range(self.model.ngeom)
            if self.model.geom(index).name.startswith(self.target_config.name)
        ]

        self.assertGreaterEqual(camera_id, 0)
        self.assertEqual(len(target_geom_ids), 4)
        self.assertFalse(
            any(
                self.model.geom(index).name == "target_marker_geom"
                for index in range(self.model.ngeom)
            )
        )
        for geom_id in target_geom_ids:
            self.assertEqual(int(self.model.geom_contype[geom_id]), 0)
            self.assertEqual(int(self.model.geom_conaffinity[geom_id]), 0)

    def test_optical_axis_projects_to_principal_point(self) -> None:
        data = self.make_data()
        camera_id = self.model.camera(self.camera_config.name).id
        rotation = data.cam_xmat[camera_id].reshape(3, 3)
        point = data.cam_xpos[camera_id] - rotation[:, 2] * 3.0

        result = project_world_points(
            self.model, data, point.reshape(1, 3), self.camera_config
        )

        np.testing.assert_allclose(
            result.pixels_xy[0],
            np.asarray(self.camera_config.principal_point_px),
            atol=1e-10,
        )
        self.assertAlmostEqual(float(result.depth_m[0]), 3.0, places=12)
        self.assertTrue(result.inside_image[0])

    def test_target_is_partly_visible_at_start_and_hover(self) -> None:
        data = self.make_data()
        start_projection = project_world_points(
            self.model, data, self.target.corners_world_m, self.camera_config
        )
        data.qpos[:3] = self.target.hover_anchor_world_m
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(self.model, data)
        hover_projection = project_world_points(
            self.model, data, self.target.corners_world_m, self.camera_config
        )

        self.assertGreaterEqual(start_projection.visible_fraction, 0.25)
        self.assertGreaterEqual(hover_projection.visible_fraction, 0.25)

    def test_actor_frame_schema_excludes_privileged_channels(self) -> None:
        self.assertEqual(ActorCameraFrame.field_names(), ("rgb", "timestamp_s"))

    def test_capture_rejects_legacy_goal_marker_leakage(self) -> None:
        model = build_scene_model(self.scene, include_legacy_goal_marker=True)

        with self.assertRaisesRegex(ValueError, "legacy target_marker_geom"):
            HeadlessCamera(model, self.camera_config)

    def test_offscreen_capture_is_deterministic_and_separated(self) -> None:
        data = self.make_data()
        try:
            camera = HeadlessCamera(self.model, self.camera_config)
        except CameraRenderUnavailable as error:
            self.skipTest(str(error))
        with camera:
            first = camera.capture_actor(data)
            second = camera.capture_actor(data)
            privileged = camera.capture_privileged(data)

        np.testing.assert_array_equal(first.rgb, second.rgb)
        self.assertEqual(
            first.rgb.shape,
            (
                self.camera_config.output_height_px,
                self.camera_config.output_width_px,
                3,
            ),
        )
        self.assertEqual(first.rgb.dtype, np.uint8)
        self.assertEqual(
            privileged.depth_m.shape,
            (
                self.camera_config.output_height_px,
                self.camera_config.output_width_px,
            ),
        )
        self.assertEqual(privileged.target_mask.shape, privileged.depth_m.shape)
        self.assertEqual(privileged.obstacle_mask.shape, privileged.depth_m.shape)
        self.assertTrue(np.all(np.isfinite(privileged.depth_m)))
        self.assertTrue(np.any(privileged.target_mask))
        self.assertEqual(first.timestamp_s, privileged.timestamp_s)


if __name__ == "__main__":
    unittest.main()
