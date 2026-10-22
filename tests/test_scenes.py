"""Seed tabanlı statik sahne üretimi testleri."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import mujoco
import numpy as np

from ebabil.perception import load_camera_config
from ebabil.scenes import (
    Obstacle,
    StaticScene,
    VisualDistractor,
    build_scene_model,
    direct_path_is_blocked,
    generate_static_scene,
    has_navigable_xy_path,
    load_scene_config,
)
from ebabil.scenes.generator import (
    _inflated_obstacle_bounds_xy,
    _segment_intersects_aabb,
    _segment_intersects_aabb_batch,
    required_ground_half_size_m,
)
from ebabil.scenes.target import load_target_area_config
from ebabil.sensors import PlanarRangefinder, RangefinderConfig


class StaticSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_scene_config()

    def test_same_seed_reproduces_scene(self) -> None:
        first = generate_static_scene(1234, self.config, mode="random_obstacles")
        second = generate_static_scene(1234, self.config, mode="random_obstacles")

        np.testing.assert_allclose(first.start_world_m, second.start_world_m)
        np.testing.assert_allclose(first.goal_world_m, second.goal_world_m)
        self.assertEqual(len(first.obstacles), self.config.obstacle_count)
        for first_obstacle, second_obstacle in zip(
            first.obstacles, second.obstacles, strict=True
        ):
            self.assertEqual(first_obstacle.name, second_obstacle.name)
            np.testing.assert_allclose(
                first_obstacle.center_world_m, second_obstacle.center_world_m
            )
            np.testing.assert_allclose(
                first_obstacle.half_size_m, second_obstacle.half_size_m
            )

    def test_different_seed_changes_scene(self) -> None:
        first = generate_static_scene(1, self.config, mode="random_obstacles")
        second = generate_static_scene(2, self.config, mode="random_obstacles")
        first_centers = np.vstack(
            [obstacle.center_world_m for obstacle in first.obstacles]
        )
        second_centers = np.vstack(
            [obstacle.center_world_m for obstacle in second.obstacles]
        )
        self.assertFalse(np.allclose(first_centers, second_centers))

    def test_generated_model_compiles_and_contains_collision_obstacles(self) -> None:
        scene = generate_static_scene(99, self.config, mode="random_obstacles")
        model = build_scene_model(scene)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        self.assertEqual(model.nmocap, 1)
        self.assertEqual(
            sum(
                model.geom(index).name.startswith("obstacle_")
                for index in range(model.ngeom)
            ),
            self.config.obstacle_count,
        )
        np.testing.assert_allclose(
            data.xpos[model.body("quadrotor").id], scene.start_world_m
        )
        np.testing.assert_allclose(
            data.xpos[model.body("target_marker").id], scene.goal_world_m
        )
        for obstacle in scene.obstacles:
            geom_id = model.geom(obstacle.name).id
            self.assertEqual(int(model.geom_group[geom_id]), 1)
            self.assertNotEqual(int(model.geom_contype[geom_id]), 0)

    def test_model_supports_multiple_obstacle_primitives_and_visual_distractors(
        self,
    ) -> None:
        scene = StaticScene(
            seed=7,
            start_world_m=np.array([-2.0, 0.0, 1.0]),
            goal_world_m=np.array([2.0, 0.0, 1.0]),
            obstacles=tuple(
                Obstacle(
                    f"obstacle_{index:03d}",
                    np.array([-1.5 + index, 1.0, 0.6]),
                    np.array([0.25, 0.3, 0.6]),
                    geometry,
                )
                for index, geometry in enumerate(
                    ("box", "cylinder", "capsule", "sphere", "ellipsoid")
                )
            ),
        )
        distractor = VisualDistractor(
            "distractor_target_000",
            np.array([0.0, -1.0, 0.3]),
            np.array([0.2, 0.2, 0.3]),
            "cylinder",
            "target",
        )

        model = build_scene_model(
            scene,
            include_legacy_goal_marker=False,
            visual_distractors=(distractor,),
        )

        expected_types = (
            mujoco.mjtGeom.mjGEOM_BOX,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        )
        for index, expected_type in enumerate(expected_types):
            geom_id = model.geom(f"obstacle_{index:03d}").id
            self.assertEqual(int(model.geom_type[geom_id]), int(expected_type))
            self.assertNotEqual(int(model.geom_contype[geom_id]), 0)
        distractor_id = model.geom(distractor.name).id
        self.assertEqual(int(model.geom_contype[distractor_id]), 0)
        self.assertEqual(int(model.geom_conaffinity[distractor_id]), 0)

    def test_clear_mode_has_no_obstacles(self) -> None:
        scene = generate_static_scene(20260910, self.config, mode="clear")

        self.assertEqual(scene.obstacles, ())
        self.assertFalse(direct_path_is_blocked(scene, self.config.vehicle_clearance_m))
        self.assertTrue(has_navigable_xy_path(scene, self.config))

    def test_random_endpoint_modes_are_seeded_and_respect_bounds(self) -> None:
        for endpoint_mode in ("random_start", "random_goal", "random_both"):
            with self.subTest(endpoint_mode=endpoint_mode):
                first = generate_static_scene(
                    1234,
                    self.config,
                    mode="clear",
                    endpoint_mode=endpoint_mode,
                )
                second = generate_static_scene(
                    1234,
                    self.config,
                    mode="clear",
                    endpoint_mode=endpoint_mode,
                )

                np.testing.assert_array_equal(first.start_world_m, second.start_world_m)
                np.testing.assert_array_equal(first.goal_world_m, second.goal_world_m)
                self.assertGreaterEqual(
                    np.linalg.norm(first.goal_world_m - first.start_world_m),
                    self.config.minimum_endpoint_separation_m,
                )
                for endpoint in (first.start_world_m, first.goal_world_m):
                    self.assertTrue(np.all(endpoint >= self.config.workspace_min_m))
                    self.assertTrue(np.all(endpoint <= self.config.workspace_max_m))
                if endpoint_mode == "random_start":
                    np.testing.assert_array_equal(
                        first.goal_world_m, self.config.goal_world_m
                    )
                if endpoint_mode == "random_goal":
                    np.testing.assert_array_equal(
                        first.start_world_m, self.config.start_world_m
                    )

    def test_different_seed_changes_random_endpoints(self) -> None:
        first = generate_static_scene(
            1, self.config, mode="clear", endpoint_mode="random_both"
        )
        second = generate_static_scene(
            2, self.config, mode="clear", endpoint_mode="random_both"
        )

        self.assertFalse(np.array_equal(first.start_world_m, second.start_world_m))
        self.assertFalse(np.array_equal(first.goal_world_m, second.goal_world_m))

    def test_single_blocker_forces_avoidance_but_keeps_alternative_path(self) -> None:
        scene = generate_static_scene(20260910, self.config, mode="single_blocker")
        model = build_scene_model(scene)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        direction = scene.goal_world_m[:2] - scene.start_world_m[:2]
        yaw = np.arctan2(direction[1], direction[0])
        data.qpos[3:7] = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
        mujoco.mj_forward(model, data)
        sensor = PlanarRangefinder(
            RangefinderConfig(
                ray_count=1,
                field_of_view_rad=1.0,
                min_range_m=0.05,
                max_range_m=float(np.linalg.norm(direction)),
                origin_body_m=(0.0, 0.0, 0.0),
                geom_group=1,
            )
        )

        reading = sensor.read(model, data)

        self.assertEqual(len(scene.obstacles), 1)
        self.assertTrue(direct_path_is_blocked(scene, self.config.vehicle_clearance_m))
        self.assertTrue(has_navigable_xy_path(scene, self.config))
        self.assertNotEqual(reading.hit_geom_ids[0], -1)
        self.assertLess(reading.distances_m[0], sensor.config.max_range_m)

    def test_random_scenes_block_direct_path_and_remain_solvable(self) -> None:
        for seed in range(25):
            scene = generate_static_scene(seed, self.config, mode="random_obstacles")
            self.assertEqual(len(scene.obstacles), self.config.obstacle_count)
            self.assertTrue(
                direct_path_is_blocked(scene, self.config.vehicle_clearance_m),
                msg=f"seed={seed}",
            )
            self.assertTrue(
                has_navigable_xy_path(scene, self.config), msg=f"seed={seed}"
            )

    def test_direct_path_uses_exact_3d_box_geometry(self) -> None:
        offset_scene = StaticScene(
            seed=0,
            start_world_m=np.array([-2.0, 0.0, 1.0]),
            goal_world_m=np.array([2.0, 0.0, 1.0]),
            obstacles=(
                Obstacle(
                    "offset",
                    np.array([0.0, 0.5, 1.0]),
                    np.array([1.0, 0.1, 0.5]),
                ),
            ),
        )
        low_scene = StaticScene(
            seed=0,
            start_world_m=np.array([-2.0, 0.0, 2.0]),
            goal_world_m=np.array([2.0, 0.0, 2.0]),
            obstacles=(
                Obstacle(
                    "low",
                    np.array([0.0, 0.0, 0.4]),
                    np.array([0.4, 0.4, 0.4]),
                ),
            ),
        )

        self.assertFalse(direct_path_is_blocked(offset_scene))
        self.assertFalse(direct_path_is_blocked(low_scene, clearance_m=0.25))

    def test_blocker_height_must_cover_configured_flight_line(self) -> None:
        high_config = replace(
            self.config,
            start_world_m=np.array([-2.0, 0.0, 2.0]),
            goal_world_m=np.array([2.0, 0.0, 2.0]),
            obstacle_half_size_min_m=np.array([0.15, 0.15, 0.2]),
            obstacle_half_size_max_m=np.array([0.45, 0.45, 0.4]),
        )

        with self.assertRaisesRegex(ValueError, "yükseklik sınırı"):
            generate_static_scene(9, high_config, mode="single_blocker")

    def test_grid_edges_cannot_cross_an_obstacle_between_free_nodes(self) -> None:
        narrow_config = replace(
            self.config,
            workspace_min_m=np.array([0.0, 0.0, 0.0]),
            workspace_max_m=np.array([0.1, 0.1, 1.0]),
            start_world_m=np.array([0.0, 0.0, 0.5]),
            goal_world_m=np.array([0.1, 0.0, 0.5]),
            vehicle_clearance_m=0.0,
            path_grid_resolution_m=0.1,
        )
        scene = StaticScene(
            seed=0,
            start_world_m=narrow_config.start_world_m.copy(),
            goal_world_m=narrow_config.goal_world_m.copy(),
            obstacles=(
                Obstacle(
                    "between_nodes",
                    np.array([0.05, 0.05, 0.5]),
                    np.array([0.01, 0.06, 0.1]),
                ),
            ),
        )

        self.assertFalse(has_navigable_xy_path(scene, narrow_config))


if __name__ == "__main__":
    unittest.main()


class PathValidationVectorizationTests(unittest.TestCase):
    """Rota doğrulamasının vektörleştirilmesi davranışı değiştirmemelidir."""

    def test_batch_slab_test_matches_scalar_reference(self) -> None:
        rng = np.random.default_rng(20260912)
        for _ in range(200):
            start = rng.uniform(-5.0, 5.0, size=2)
            goal = start + rng.uniform(-4.0, 4.0, size=2)
            if rng.random() < 0.2:
                goal = start.copy()  # dejenere segment: noktasal kapsanma
            centers = rng.uniform(-5.0, 5.0, size=(6, 2))
            halves = rng.uniform(0.1, 1.5, size=(6, 2))
            lower, upper = centers - halves, centers + halves

            batch = _segment_intersects_aabb_batch(start, goal, lower, upper)
            scalar = np.array(
                [
                    _segment_intersects_aabb(start, goal, lower[i], upper[i])
                    for i in range(lower.shape[0])
                ]
            )

            np.testing.assert_array_equal(batch, scalar)

    def test_batch_test_handles_empty_obstacle_set(self) -> None:
        empty = np.empty((0, 2), dtype=np.float64)

        result = _segment_intersects_aabb_batch(np.zeros(2), np.ones(2), empty, empty)

        self.assertEqual(result.shape, (0,))
        self.assertFalse(bool(result.any()))

    def test_inflated_bounds_apply_vehicle_clearance(self) -> None:
        scene = StaticScene(
            seed=1,
            start_world_m=np.array([-2.0, 0.0, 1.0]),
            goal_world_m=np.array([2.0, 0.0, 1.0]),
            obstacles=(
                Obstacle(
                    "obstacle_000",
                    np.array([0.0, 0.5, 0.6]),
                    np.array([0.3, 0.4, 0.6]),
                ),
            ),
        )

        lower, upper = _inflated_obstacle_bounds_xy(scene, 0.25)

        np.testing.assert_allclose(lower, np.array([[-0.55, -0.15]]), atol=1e-12)
        np.testing.assert_allclose(upper, np.array([[0.55, 1.15]]), atol=1e-12)

    def test_empty_scene_has_no_inflated_bounds(self) -> None:
        scene = StaticScene(
            seed=1,
            start_world_m=np.array([-2.0, 0.0, 1.0]),
            goal_world_m=np.array([2.0, 0.0, 1.0]),
            obstacles=(),
        )

        lower, upper = _inflated_obstacle_bounds_xy(scene, 0.25)

        self.assertEqual(lower.shape, (0, 2))
        self.assertEqual(upper.shape, (0, 2))

    def test_navigable_path_result_is_unchanged_on_known_scenes(self) -> None:
        """Bilinen sahnelerde çözülebilirlik kararı sabit kalmalı."""

        config = load_scene_config()
        for mode in ("clear", "single_blocker", "random_obstacles"):
            with self.subTest(mode=mode):
                for seed in range(20260911, 20260921):
                    scene = generate_static_scene(seed, config, mode=mode)
                    self.assertTrue(has_navigable_xy_path(scene, config))

    def test_blocked_corridor_is_still_rejected(self) -> None:
        """Optimizasyon çözülemeyen sahneyi çözülebilir göstermemeli."""

        config = load_scene_config()
        span = config.workspace_max_m[1] - config.workspace_min_m[1]
        wall = Obstacle(
            "obstacle_000",
            np.array([0.0, 0.0, 0.6]),
            np.array([0.3, span, 0.6]),
        )
        scene = StaticScene(
            seed=1,
            start_world_m=config.start_world_m.copy(),
            goal_world_m=config.goal_world_m.copy(),
            obstacles=(wall,),
        )

        self.assertFalse(has_navigable_xy_path(scene, config))

    def test_d1_scale_generation_stays_within_budget(self) -> None:
        """D1 ölçeğinde sahne üretimi eğitim reset'ini domine etmemeli."""

        import time

        half = 10.0
        config = replace(
            load_scene_config(),
            workspace_min_m=np.array([-half, -half, 0.0]),
            workspace_max_m=np.array([half, half, 3.0]),
            random_endpoint_min_m=np.array([-half + 0.75, -half + 0.75, 0.75]),
            random_endpoint_max_m=np.array([half - 0.75, half - 0.75, 1.75]),
            obstacle_count=40,
        )
        config.validate()

        started = time.perf_counter()
        generate_static_scene(
            20260911, config, mode="random_obstacles", endpoint_mode="random_both"
        )
        elapsed = time.perf_counter() - started

        # Vektörleştirme öncesi bu config ~2.5 s sürüyordu.
        self.assertLess(elapsed, 1.0)


class EndpointSeparationBandTests(unittest.TestCase):
    """ADR-018: mesafe seviyeleri rota uzunluğunu bir banda sınırlar."""

    def test_legacy_config_has_no_upper_bound(self) -> None:
        config = load_scene_config()
        config.validate()

        self.assertEqual(config.maximum_endpoint_separation_m, float("inf"))

    def test_upper_bound_is_enforced_when_configured(self) -> None:
        half = 10.0
        config = replace(
            load_scene_config(),
            workspace_min_m=np.array([-half, -half, 0.0]),
            workspace_max_m=np.array([half, half, 3.0]),
            random_endpoint_min_m=np.array([-half + 0.75, -half + 0.75, 0.75]),
            random_endpoint_max_m=np.array([half - 0.75, half - 0.75, 1.75]),
            minimum_endpoint_separation_m=5.0,
            maximum_endpoint_separation_m=15.0,
            obstacle_count=10,
        )
        config.validate()

        separations = []
        for seed in range(20260911, 20260951):
            scene = generate_static_scene(
                seed, config, mode="clear", endpoint_mode="random_both"
            )
            separations.append(
                float(np.linalg.norm(scene.goal_world_m - scene.start_world_m))
            )

        self.assertGreaterEqual(min(separations), 5.0 - 1e-9)
        self.assertLessEqual(max(separations), 15.0 + 1e-9)

    def test_maximum_below_minimum_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "maximum_endpoint_separation_m"):
            replace(
                load_scene_config(),
                minimum_endpoint_separation_m=5.0,
                maximum_endpoint_separation_m=4.0,
            ).validate()

    def test_unreachable_band_raises_descriptive_error(self) -> None:
        config = replace(
            load_scene_config(),
            minimum_endpoint_separation_m=4.90,
            maximum_endpoint_separation_m=4.95,
            max_sampling_attempts=50,
        )
        config.validate()

        with self.assertRaisesRegex(RuntimeError, "bandında üretilemedi"):
            generate_static_scene(
                20260911, config, mode="clear", endpoint_mode="random_both"
            )


class GroundPlaneExtentTests(unittest.TestCase):
    """REV-030: görsel zemin düzlemi workspace'i kapsamak zorundadır."""

    def test_legacy_ground_size_is_frozen_when_unspecified(self) -> None:
        """ADR-008 legacy sahnesi ve onaylı encoder zinciri 5.0 m ile üretildi."""

        self.assertEqual(load_scene_config().ground_half_size_m, 5.0)
        scene = generate_static_scene(20260910, mode="single_blocker")
        model = build_scene_model(scene)
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

        np.testing.assert_allclose(model.geom_size[index][:2], [5.0, 5.0])

    def test_explicit_ground_size_is_applied(self) -> None:
        scene = generate_static_scene(20260910, mode="single_blocker")
        model = build_scene_model(scene, ground_half_size_m=21.13)
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

        np.testing.assert_allclose(model.geom_size[index][:2], [21.13, 21.13])
        for invalid in (0.0, -1.0, math.inf, math.nan):
            with self.assertRaisesRegex(ValueError, "ground_half_size_m"):
                build_scene_model(scene, ground_half_size_m=invalid)

    def test_d1_config_covers_its_workspace(self) -> None:
        """Büyütülmüş workspace'te kamera her köşede zemini görmeli."""

        config = load_scene_config(
            "src/ebabil/configs/environment/static_scene_d1.toml"
        )
        required_m = required_ground_half_size_m(config)

        self.assertGreaterEqual(config.ground_half_size_m, required_m - 1e-2)
        self.assertGreater(config.ground_half_size_m, 5.0)

    def test_required_size_derives_from_workspace_and_ceiling(self) -> None:
        config = load_scene_config(
            "src/ebabil/configs/environment/static_scene_d1.toml"
        )
        camera = load_camera_config()
        _, far_m = camera.ground_swath_m(float(config.workspace_max_m[2]))
        corner_m = float(
            np.max(
                np.maximum(
                    np.abs(np.asarray(config.workspace_min_m[:2])),
                    np.abs(np.asarray(config.workspace_max_m[:2])),
                )
            )
        )

        self.assertAlmostEqual(
            required_ground_half_size_m(config), corner_m + far_m, places=9
        )

    def test_plane_stays_infinite_for_collision(self) -> None:
        """Düzlemin görsel sınırı fiziği etkilemez; aksi halde araç düşerdi."""

        scene = generate_static_scene(20260910, mode="single_blocker")
        model = build_scene_model(scene)
        data = mujoco.MjData(model)
        ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

        for x_m in (0.0, 4.0, 9.0, 15.0):
            with self.subTest(x_m=x_m):
                mujoco.mj_resetData(model, data)
                data.qpos[0:3] = [x_m, 0.0, 0.02]
                mujoco.mj_forward(model, data)
                contacts = [data.contact[index].geom2 for index in range(data.ncon)] + [
                    data.contact[index].geom1 for index in range(data.ncon)
                ]

                self.assertIn(ground, contacts)


class BlockerPlacementTests(unittest.TestCase):
    """REV-031: engel konumu ezberlenebilir bir sabit olmamalı."""

    def setUp(self) -> None:
        self.config = load_scene_config(
            "src/ebabil/configs/environment/static_scene_d1.toml"
        )
        self.target = load_target_area_config()

    # D1 sahne üretimi ~165 ms; test süresini kısa tutmak için az seed.
    def _fractions_from_start(self, config, seeds=range(410001, 410011)):
        values = []
        for seed in seeds:
            scene = generate_static_scene(
                seed,
                config,
                mode="single_blocker",
                endpoint_mode="random_both",
                target_config=self.target,
            )
            start = np.asarray(scene.start_world_m[:2], dtype=np.float64)
            goal = np.asarray(scene.goal_world_m[:2], dtype=np.float64)
            route = goal - start
            length = float(np.linalg.norm(route))
            center = np.asarray(scene.obstacles[0].center_world_m[:2], dtype=np.float64)
            values.append(float(np.dot(center - start, route / length)) / length)
        return np.asarray(values)

    def test_default_keeps_the_blocker_at_the_route_midpoint(self) -> None:
        """Bütün D1 koşuları bu sabit yerleşimle yapıldı; varsayılan korunur."""

        self.assertEqual(self.config.blocker_path_fraction_min, 0.5)
        self.assertEqual(self.config.blocker_path_fraction_max, 0.5)

        fractions = self._fractions_from_start(self.config)

        np.testing.assert_allclose(fractions, 0.5, atol=1e-9)

    def test_randomization_spreads_the_blocker_toward_the_start(self) -> None:
        """Hedefin hover zarfı boş kalmak zorunda olduğu için kaydırma tek yönlü.

        Kod kesri hedeften ölçülür (`0.5` orta, `0.85` başlangıca yakın), bu
        yüzden başlangıçtan ölçülen konum `0.15` ile `0.50` arasına düşer.
        """

        randomized = replace(
            self.config,
            blocker_path_fraction_min=0.5,
            blocker_path_fraction_max=0.85,
        )
        randomized.validate()

        fractions = self._fractions_from_start(randomized)

        self.assertGreater(float(fractions.std()), 0.05)
        self.assertGreaterEqual(float(fractions.min()), 0.15 - 1e-6)
        self.assertLessEqual(float(fractions.max()), 0.5 + 1e-6)

    def test_randomization_never_intrudes_on_the_target(self) -> None:
        """Kaydırma yalnız başlangıca doğrudur; hedefe yaklaşmak yasak."""

        randomized = replace(
            self.config,
            blocker_path_fraction_min=0.5,
            blocker_path_fraction_max=0.85,
        )
        for seed in range(410001, 410006):
            scene = generate_static_scene(
                seed,
                randomized,
                mode="single_blocker",
                endpoint_mode="random_both",
                target_config=self.target,
            )
            with self.subTest(seed=seed):
                self.assertTrue(
                    direct_path_is_blocked(scene, randomized.vehicle_clearance_m)
                )

    def test_invalid_fraction_ranges_are_rejected(self) -> None:
        for lo, hi in ((0.4, 0.85), (0.5, 0.95), (0.7, 0.6), (math.nan, 0.85)):
            with self.subTest(lo=lo, hi=hi):
                with self.assertRaises(ValueError):
                    replace(
                        self.config,
                        blocker_path_fraction_min=lo,
                        blocker_path_fraction_max=hi,
                    ).validate()
