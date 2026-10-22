"""REV-014–REV-017 hover geometrisi, hedef bağlama ve sahne açıklığı testleri."""

from __future__ import annotations

import hashlib
import math
import unittest
from dataclasses import replace

import numpy as np

from ebabil.perception import load_camera_config
from ebabil.perception.dataset import PrivilegedCameraLabels, derive_supervision
from ebabil.scenes import (
    HoverEnvelope,
    load_scene_config,
    load_target_area_config,
    reachable_safe_region_xy,
    target_area_from_goal,
    target_area_from_scene,
    validate_hover_envelope,
)
from ebabil.scenes.generator import (
    BLOCKER_MAXIMUM_PATH_FRACTION,
    StaticScene,
    _point_to_box_distance,
    direct_path_is_blocked,
    generate_static_scene,
)

# 3 sahne modu x 4 endpoint modu x 20 seed (20260900-20260919) için legacy
# üretim parmak izi. Hedef-farkında açıklığın eklenmesi bu diziyi
# değiştirmemelidir. Aynı doğrulama 60 seed (720 sahne) ile de çalıştırıldı ve
# `75274526eae51f52475ea4bf7ae0d0f44f12d7c2f4cc3f6f84418a93c390dfe6` değerini
# korudu; test süresini kısa tutmak için burada 240 sahne kullanılır.
LEGACY_SCENE_FINGERPRINT = (
    "3a9da4aa00debb7e00d9c1c6f5d68ae283865d22f6677d0659ca891527ff1231"
)


def _scene_fingerprint() -> str:
    config = load_scene_config()
    digest = hashlib.sha256()
    for mode in ("clear", "single_blocker", "random_obstacles"):
        for endpoint_mode in ("fixed", "random_start", "random_goal", "random_both"):
            for seed in range(20260900, 20260920):
                scene = generate_static_scene(
                    seed, config, mode=mode, endpoint_mode=endpoint_mode
                )
                digest.update(np.round(scene.start_world_m, 12).tobytes())
                digest.update(np.round(scene.goal_world_m, 12).tobytes())
                for obstacle in scene.obstacles:
                    digest.update(obstacle.name.encode())
                    digest.update(np.round(obstacle.center_world_m, 12).tobytes())
                    digest.update(np.round(obstacle.half_size_m, 12).tobytes())
                    digest.update(obstacle.geometry.encode())
    return digest.hexdigest()


class CameraNadirGeometryTests(unittest.TestCase):
    """REV-014: sabit montajlı kameranın nadir kör bölgesi."""

    def setUp(self) -> None:
        self.camera = load_camera_config()
        self.target = load_target_area_config()

    def test_nadir_threshold_matches_vertical_field_of_view(self) -> None:
        expected_rad = math.pi / 2.0 - self.camera.vertical_fov_rad / 2.0

        self.assertAlmostEqual(
            self.camera.minimum_pitch_for_nadir_view_rad, expected_rad, places=12
        )
        self.assertAlmostEqual(
            math.degrees(self.camera.minimum_pitch_for_nadir_view_rad),
            60.08,
            places=2,
        )

    def test_shipped_mount_angle_cannot_see_nadir(self) -> None:
        self.assertFalse(self.camera.nadir_in_view)
        self.assertLess(
            self.camera.pitch_down_rad,
            self.camera.minimum_pitch_for_nadir_view_rad,
        )

    def test_nadir_enters_view_only_at_or_above_threshold(self) -> None:
        threshold_rad = self.camera.minimum_pitch_for_nadir_view_rad
        below = replace(self.camera, pitch_down_rad=threshold_rad - 1e-9)
        at_threshold = replace(self.camera, pitch_down_rad=threshold_rad)

        self.assertFalse(below.nadir_in_view)
        self.assertTrue(at_threshold.nadir_in_view)
        self.assertAlmostEqual(at_threshold.ground_swath_m(2.0)[0], 0.0, places=9)

    def test_ground_swath_scales_linearly_so_altitude_cannot_close_blind_spot(
        self,
    ) -> None:
        near_low, _ = self.camera.ground_swath_m(1.0)
        near_high, _ = self.camera.ground_swath_m(3.0)

        self.assertGreater(near_low, 0.0)
        self.assertAlmostEqual(near_high, 3.0 * near_low, places=9)

    def test_target_center_is_outside_frame_at_hover_anchor(self) -> None:
        """Hold pozunda hedef merkezi kadraj dışıdır; hold merkeze bağlanamaz."""

        hover_altitude_m = self.target.surface_z_m + self.target.hover_height_m
        near_m, _ = self.camera.ground_swath_m(hover_altitude_m)
        target_half_x, _ = self.target.half_size_xy_m

        self.assertAlmostEqual(near_m, 0.406, places=3)
        self.assertGreater(near_m, 0.0)
        self.assertLess(near_m, target_half_x)

    def test_center_visible_hold_band_fits_inside_safe_region(self) -> None:
        """REV-023: hold halkası hem var olmalı hem de uygulanabilir olmalı.

        Halka, hedefin görülebildiği en yakın zemin mesafesi (`near_m`) ile
        hold koşulunun izin verdiği en uzak mesafe (`safe_half_x`) arasındadır.
        REV-023: `hover_height_m = 2.0` ile halka `0.108 m` idi. REV-044: pitch
        `45 -> 40` derece olunca yakın sınır uzaklaştı ve halkayı korumak
        için hover yüksekliği `1.5 -> 1.1 m` indirilmek zorunda kaldı.
        Pitch ve hover yüksekliği tek bir geometrik karardır.
        """

        hover_altitude_m = self.target.surface_z_m + self.target.hover_height_m
        near_m, _ = self.camera.ground_swath_m(hover_altitude_m)
        safe_half_x, _ = self.target.safe_half_size_xy_m
        band_m = safe_half_x - near_m

        self.assertLess(near_m, safe_half_x)
        self.assertAlmostEqual(band_m, 0.244, places=3)
        self.assertGreater(band_m, 0.20)

    def test_lowering_hover_height_widens_the_hold_band(self) -> None:
        """REV-023 düzeltmesinin yönü: irtifa düştükçe halka genişler."""

        safe_half_x, _ = self.target.safe_half_size_xy_m
        previous_band_m = -math.inf
        for hover_height_m in (2.0, 1.75, 1.5, 1.25, 1.1):
            near_m, _ = self.camera.ground_swath_m(
                self.target.surface_z_m + hover_height_m
            )
            band_m = safe_half_x - near_m
            self.assertGreater(band_m, previous_band_m)
            previous_band_m = band_m

    def test_ground_swath_rejects_invalid_altitude(self) -> None:
        with self.assertRaisesRegex(ValueError, "altitude_m"):
            self.camera.ground_swath_m(0.0)
        with self.assertRaisesRegex(ValueError, "altitude_m"):
            self.camera.ground_swath_m(math.nan)


class HoverEnvelopeTests(unittest.TestCase):
    """REV-015: hedef alan tek kaynaklı türetme ve hover zarfı doğrulaması."""

    def setUp(self) -> None:
        self.scene_config = load_scene_config()
        self.target_config = load_target_area_config()

    def test_target_center_comes_from_goal_xy_and_ignores_goal_altitude(self) -> None:
        low = target_area_from_goal(np.array([1.5, -0.5, 0.8]), self.target_config)
        high = target_area_from_goal(np.array([1.5, -0.5, 1.75]), self.target_config)

        self.assertEqual(low.center_xy_world_m, (1.5, -0.5))
        self.assertEqual(low.center_xy_world_m, high.center_xy_world_m)
        np.testing.assert_allclose(
            low.hover_anchor_world_m, high.hover_anchor_world_m, atol=1e-12
        )

    def test_scene_factory_delegates_to_goal_factory(self) -> None:
        scene = StaticScene(
            seed=1,
            start_world_m=np.array([-2.0, 0.0, 1.0]),
            goal_world_m=np.array([2.0, 0.5, 1.3]),
            obstacles=(),
        )

        from_scene = target_area_from_scene(scene, self.target_config)
        from_goal = target_area_from_goal(scene.goal_world_m, self.target_config)

        self.assertEqual(from_scene.center_xy_world_m, from_goal.center_xy_world_m)

    def test_hover_envelope_covers_safe_region_and_altitude_tolerance(self) -> None:
        target = target_area_from_goal(np.array([0.0, 0.0, 1.0]), self.target_config)
        envelope = target.hover_envelope
        safe_half_x, safe_half_y = self.target_config.safe_half_size_xy_m
        anchor_z = self.target_config.surface_z_m + self.target_config.hover_height_m

        np.testing.assert_allclose(
            envelope.lower_world_m,
            np.array(
                [
                    -safe_half_x,
                    -safe_half_y,
                    anchor_z - self.target_config.altitude_tolerance_m,
                ]
            ),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            envelope.upper_world_m,
            np.array(
                [
                    safe_half_x,
                    safe_half_y,
                    anchor_z + self.target_config.altitude_tolerance_m,
                ]
            ),
            atol=1e-12,
        )

    def test_envelope_intersection_is_strict_and_inflation_widens(self) -> None:
        envelope = HoverEnvelope(
            lower_world_m=np.array([-1.0, -1.0, 1.0]),
            upper_world_m=np.array([1.0, 1.0, 2.0]),
        )
        envelope.validate()

        self.assertTrue(envelope.intersects_box([0.0, 0.0, 1.5], [0.1, 0.1, 0.1]))
        self.assertFalse(envelope.intersects_box([2.0, 0.0, 1.5], [0.9, 0.1, 0.1]))
        self.assertTrue(
            envelope.inflated(0.2).intersects_box([2.0, 0.0, 1.5], [0.9, 0.1, 0.1])
        )

    def test_shipped_hover_band_is_reachable(self) -> None:
        validate_hover_envelope(self.target_config, self.scene_config)

    def test_unreachable_hover_band_fails_fast(self) -> None:
        too_high = replace(self.target_config, hover_height_m=2.1)

        with self.assertRaisesRegex(ValueError, "üst kenarı erişilebilir"):
            validate_hover_envelope(too_high, self.scene_config)

    def test_safe_region_is_clipped_at_workspace_edge(self) -> None:
        edge_x = float(self.scene_config.workspace_max_m[0])
        target = target_area_from_goal(np.array([edge_x, 0.0, 1.0]), self.target_config)

        lower, upper = reachable_safe_region_xy(target, self.scene_config)

        self.assertAlmostEqual(
            float(upper[0]), edge_x - self.scene_config.vehicle_clearance_m, places=12
        )
        self.assertLess(float(upper[0]), edge_x)

    def test_safe_region_outside_workspace_is_rejected(self) -> None:
        outside_x = float(self.scene_config.workspace_max_m[0]) + 5.0
        target = target_area_from_goal(
            np.array([outside_x, 0.0, 1.0]), self.target_config
        )

        with self.assertRaisesRegex(ValueError, "kesişmiyor"):
            reachable_safe_region_xy(target, self.scene_config)


class TargetAwareSceneClearanceTests(unittest.TestCase):
    """REV-016: hedef hover hacminin sahne üretiminde boş bırakılması."""

    SEEDS = range(20260911, 20260936)

    def setUp(self) -> None:
        self.scene_config = load_scene_config()
        self.target_config = load_target_area_config()

    def _inflated_envelope(self, goal_world_m: np.ndarray) -> HoverEnvelope:
        target = target_area_from_goal(goal_world_m, self.target_config)
        return target.hover_envelope.inflated(self.scene_config.vehicle_clearance_m)

    def test_legacy_scene_sequence_is_unchanged(self) -> None:
        self.assertEqual(_scene_fingerprint(), LEGACY_SCENE_FINGERPRINT)

    def test_legacy_default_still_allows_hover_envelope_intrusion(self) -> None:
        """Kısıt opt-in'dir; legacy sahnelerde ihlal gözlenmeye devam eder."""

        intrusions = 0
        for seed in self.SEEDS:
            scene = generate_static_scene(
                seed, self.scene_config, mode="random_obstacles", endpoint_mode="fixed"
            )
            envelope = self._inflated_envelope(scene.goal_world_m)
            if any(
                envelope.intersects_box(o.center_world_m, o.half_size_m)
                for o in scene.obstacles
            ):
                intrusions += 1

        self.assertGreater(intrusions, 0)

    def test_target_aware_scenes_keep_hover_envelope_clear(self) -> None:
        for mode in ("single_blocker", "random_obstacles"):
            for endpoint_mode in ("fixed", "random_both"):
                with self.subTest(mode=mode, endpoint_mode=endpoint_mode):
                    for seed in self.SEEDS:
                        try:
                            scene = generate_static_scene(
                                seed,
                                self.scene_config,
                                mode=mode,
                                endpoint_mode=endpoint_mode,
                                target_config=self.target_config,
                            )
                        except RuntimeError:
                            # Geometrik olarak çözülemeyen endpoint ayrımı
                            # fail-closed hata üretir; sessiz ihlal üretmez.
                            continue
                        envelope = self._inflated_envelope(scene.goal_world_m)
                        for obstacle in scene.obstacles:
                            self.assertFalse(
                                envelope.intersects_box(
                                    obstacle.center_world_m, obstacle.half_size_m
                                ),
                                f"{obstacle.name} hover hacmine giriyor (seed {seed})",
                            )

    def test_shifted_blocker_still_blocks_path_and_spares_start(self) -> None:
        for endpoint_mode in ("fixed", "random_both"):
            with self.subTest(endpoint_mode=endpoint_mode):
                for seed in self.SEEDS:
                    try:
                        scene = generate_static_scene(
                            seed,
                            self.scene_config,
                            mode="single_blocker",
                            endpoint_mode=endpoint_mode,
                            target_config=self.target_config,
                        )
                    except RuntimeError:
                        continue
                    self.assertTrue(
                        direct_path_is_blocked(
                            scene, self.scene_config.vehicle_clearance_m
                        ),
                        f"Kaydırılmış blocker rotayı kapatmıyor (seed {seed})",
                    )
                    for obstacle in scene.obstacles:
                        self.assertGreaterEqual(
                            _point_to_box_distance(scene.start_world_m, obstacle),
                            self.scene_config.vehicle_clearance_m,
                            f"Blocker başlangıcı yutuyor (seed {seed})",
                        )

    def test_blocker_stays_on_the_start_goal_segment(self) -> None:
        for seed in self.SEEDS:
            scene = generate_static_scene(
                seed,
                self.scene_config,
                mode="single_blocker",
                endpoint_mode="random_both",
                target_config=self.target_config,
            )
            blocker = scene.obstacles[0]
            direction = scene.start_world_m - scene.goal_world_m
            fraction = float(
                np.dot(
                    blocker.center_world_m[:2] - scene.goal_world_m[:2], direction[:2]
                )
                / np.dot(direction[:2], direction[:2])
            )

            self.assertGreaterEqual(fraction, 0.5 - 1e-9)
            self.assertLessEqual(fraction, BLOCKER_MAXIMUM_PATH_FRACTION + 1e-9)

    def test_unreachable_hover_config_is_rejected_before_sampling(self) -> None:
        too_high = replace(self.target_config, hover_height_m=2.1)

        with self.assertRaisesRegex(ValueError, "üst kenarı erişilebilir"):
            generate_static_scene(
                20260911,
                self.scene_config,
                mode="random_obstacles",
                target_config=too_high,
            )

    def test_target_aware_failure_message_names_the_cause(self) -> None:
        crowded = replace(
            self.scene_config,
            start_world_m=np.array([0.30, 0.0, 1.0]),
            goal_world_m=np.array([0.0, 0.0, 1.0]),
            minimum_endpoint_separation_m=0.30,
        )

        with self.assertRaisesRegex(RuntimeError, "hedef-farkında üretilemedi"):
            generate_static_scene(
                20260911,
                crowded,
                mode="single_blocker",
                endpoint_mode="fixed",
                target_config=self.target_config,
            )


class SupervisionSentinelTests(unittest.TestCase):
    """REV-017: görünmeyen hedef merkezinin belirsiz sentinel değeri."""

    def _labels(self, target_mask: np.ndarray) -> PrivilegedCameraLabels:
        height, width = target_mask.shape
        return PrivilegedCameraLabels(
            depth_m=np.full((height, width), 5.0, dtype=np.float64),
            segmentation=np.zeros((height, width), dtype=np.int32),
            target_mask=target_mask,
            obstacle_mask=np.zeros((height, width), dtype=bool),
            timestamp_s=0.0,
        )

    def test_invisible_target_center_equals_frame_center_sentinel(self) -> None:
        invisible = derive_supervision(
            self._labels(np.zeros((12, 16), dtype=bool)),
            minimum_target_pixels=8,
            free_space_max_depth_m=8.0,
        )
        centered_mask = np.zeros((12, 16), dtype=bool)
        centered_mask[5:7, 7:9] = True
        centered = derive_supervision(
            self._labels(centered_mask),
            minimum_target_pixels=1,
            free_space_max_depth_m=8.0,
        )

        self.assertFalse(invisible.target_visible)
        self.assertTrue(centered.target_visible)
        self.assertEqual(invisible.target_center_xy, (0.0, 0.0))
        np.testing.assert_allclose(centered.target_center_xy, (0.0, 0.0), atol=0.05)

    def test_visible_center_tracks_only_visible_pixels(self) -> None:
        """Hedefin bir kenarı kadraj dışıysa merkez görünen kenara kayar."""

        bottom_edge = np.zeros((12, 16), dtype=bool)
        bottom_edge[10:12, :] = True
        supervision = derive_supervision(
            self._labels(bottom_edge),
            minimum_target_pixels=8,
            free_space_max_depth_m=8.0,
        )

        self.assertTrue(supervision.target_visible)
        self.assertGreater(supervision.target_center_xy[1], 0.7)


if __name__ == "__main__":
    unittest.main()
