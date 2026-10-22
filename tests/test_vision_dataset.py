"""Vision dataset ayrımı, sızıntı koruması ve etiket testleri."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ebabil.perception import (
    ACTOR_KEYS,
    CaptureSpec,
    PrivilegedCameraLabels,
    apply_frame_visual_parameters,
    apply_scene_visual_parameters,
    derive_supervision,
    load_vision_dataset_config,
    make_capture_spec,
    sample_frame_visual_parameters,
    sample_scene_visual_parameters,
    select_scene_seeds,
    summarize_vision_dataset,
    validate_dataset_leakage,
    write_sample,
)
from ebabil.scenes import (
    HoverEnvelope,
    StaticScene,
    VisualDistractor,
    build_scene_model,
    generate_static_scene,
    load_scene_config,
    load_target_area_config,
    target_area_from_scene,
)


def _labels(height: int = 12, width: int = 15) -> PrivilegedCameraLabels:
    depth = np.full((height, width), 10.0, dtype=np.float64)
    target = np.zeros((height, width), dtype=np.bool_)
    target[2:5, 9:13] = True
    obstacle = np.zeros((height, width), dtype=np.bool_)
    obstacle[3:10, 5:12] = True
    depth[obstacle] = 2.0
    return PrivilegedCameraLabels(
        depth_m=depth,
        segmentation=np.zeros((height, width, 2), dtype=np.int32),
        target_mask=target,
        obstacle_mask=obstacle,
        timestamp_s=1.25,
    )


class VisionDatasetTests(unittest.TestCase):
    def test_visual_domains_are_named_and_training_envelops_heldout_ranges(
        self,
    ) -> None:
        config = load_vision_dataset_config()

        self.assertEqual(config.schema_version, "vision-dataset-v4")
        self.assertEqual(config.train_visual_domain.name, "train_broad_randomized")
        self.assertLessEqual(
            config.train_visual_domain.light_scale[0],
            config.validation_visual_domain.light_scale[0],
        )
        self.assertGreaterEqual(
            config.train_visual_domain.light_scale[1],
            config.development_ood_visual_domain.light_scale[1],
        )
        self.assertLessEqual(
            config.train_visual_domain.exposure_gain[0],
            config.validation_visual_domain.exposure_gain[0],
        )
        self.assertGreaterEqual(
            config.train_visual_domain.exposure_gain[1],
            config.development_ood_visual_domain.exposure_gain[1],
        )
        self.assertGreater(config.visual_distractor_count, 0)
        self.assertTrue(
            set(config.audit_obstacle_geometries)
            - set(config.train_obstacle_geometries)
        )
        for channel_minimum, channel_maximum in zip(
            config.train_visual_domain.obstacle_rgb_min,
            config.train_visual_domain.obstacle_rgb_max,
            strict=True,
        ):
            self.assertLessEqual(channel_minimum, 0.05)
            self.assertGreaterEqual(channel_maximum, 0.90)

    def test_visual_randomization_is_seeded_and_changes_actor_rgb(self) -> None:
        config = load_vision_dataset_config()
        domain = config.train_visual_domain
        parameters = sample_frame_visual_parameters(domain, 123)
        rgb = np.full((16, 20, 3), 120, dtype=np.uint8)

        first = apply_frame_visual_parameters(rgb, parameters, noise_seed=456)
        second = apply_frame_visual_parameters(rgb, parameters, noise_seed=456)
        different = apply_frame_visual_parameters(rgb, parameters, noise_seed=457)

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, rgb))
        if parameters.noise_std > 0.0:
            self.assertFalse(np.array_equal(first, different))

    def test_scene_visual_randomization_changes_only_visual_properties(self) -> None:
        config = load_vision_dataset_config()
        scene = generate_static_scene(55, mode="single_blocker")
        first = build_scene_model(scene, include_legacy_goal_marker=False)
        second = build_scene_model(scene, include_legacy_goal_marker=False)
        parameters = sample_scene_visual_parameters(config.train_visual_domain, 789)

        apply_scene_visual_parameters(first, parameters)
        apply_scene_visual_parameters(second, parameters)

        np.testing.assert_array_equal(first.tex_data, second.tex_data)
        np.testing.assert_allclose(first.light_diffuse, second.light_diffuse)
        np.testing.assert_allclose(
            first.geom_rgba[first.geom("obstacle_000").id, :3],
            parameters.obstacle_rgb,
        )
        np.testing.assert_array_equal(first.geom_contype, second.geom_contype)

    def test_target_colored_distractor_breaks_color_only_target_shortcut(self) -> None:
        config = load_vision_dataset_config()
        scene = generate_static_scene(55, mode="clear")
        distractor = VisualDistractor(
            "distractor_target_000",
            np.array([0.0, 1.0, 0.3]),
            np.array([0.2, 0.2, 0.3]),
            "cylinder",
            "target",
        )
        model = build_scene_model(
            scene,
            include_legacy_goal_marker=False,
            visual_distractors=(distractor,),
        )
        parameters = sample_scene_visual_parameters(config.train_visual_domain, 789)

        apply_scene_visual_parameters(model, parameters)

        distractor_id = model.geom(distractor.name).id
        target_id = model.geom("target_area_geom").id
        np.testing.assert_allclose(
            model.geom_rgba[distractor_id, :3],
            model.geom_rgba[target_id, :3],
        )
        self.assertEqual(int(model.geom_contype[distractor_id]), 0)

    def test_scene_seed_splits_are_deterministic_and_disjoint(self) -> None:
        config = load_vision_dataset_config()

        first = select_scene_seeds(config)
        second = select_scene_seeds(config)
        all_seeds = [seed for values in first.values() for seed in values]

        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), config.train_scene_count)
        self.assertEqual(len(first["validation"]), config.validation_scene_count)
        self.assertEqual(
            len(first["development_ood"]), config.development_ood_scene_count
        )
        self.assertEqual(len(first["audit"]), config.audit_scene_count)
        self.assertEqual(len(all_seeds), len(set(all_seeds)))

    def test_capture_pose_is_seeded_and_outside_obstacles(self) -> None:
        config = load_vision_dataset_config()
        scene_config = load_scene_config()
        scene = generate_static_scene(20260911, scene_config, mode="random_obstacles")

        first = make_capture_spec(scene, "train", 3, config, scene_config)
        second = make_capture_spec(scene, "train", 3, config, scene_config)

        self.assertEqual(first, second)
        point = np.asarray(first.position_world_m)
        for obstacle in scene.obstacles:
            expanded = obstacle.half_size_m + scene_config.vehicle_clearance_m
            self.assertFalse(
                np.all(np.abs(point - obstacle.center_world_m) <= expanded)
            )

    def test_supervision_uses_target_mask_and_obstacle_depth(self) -> None:
        targets = derive_supervision(
            _labels(), minimum_target_pixels=8, free_space_max_depth_m=8.0
        )

        self.assertTrue(targets.target_visible)
        self.assertGreater(targets.target_center_xy[0], 0.0)
        self.assertEqual(targets.free_space[0], 1.0)
        self.assertAlmostEqual(targets.free_space[1], 0.25)
        self.assertAlmostEqual(targets.free_space[2], 0.25)

    def test_actor_archive_is_allowlisted_and_separate(self) -> None:
        labels = _labels()
        targets = derive_supervision(
            labels, minimum_target_pixels=8, free_space_max_depth_m=8.0
        )
        spec = CaptureSpec(
            sample_id="train-s0000000001-p000",
            split="train",
            scene_seed=1,
            sample_index=0,
            position_world_m=(0.0, 0.0, 1.0),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = write_sample(
                root,
                spec,
                np.zeros((12, 15, 3), dtype=np.uint8),
                labels,
                targets,
                "random_obstacles",
            )
            (root / "manifest.jsonl").write_text(
                json.dumps(asdict(record)) + "\n", encoding="utf-8"
            )

            validate_dataset_leakage(root)
            with np.load(root / record.actor_file, allow_pickle=False) as payload:
                self.assertEqual(set(payload.files), ACTOR_KEYS)
                self.assertNotIn("depth_m", payload.files)
                self.assertNotIn("position_world_m", payload.files)

    def test_leakage_validator_rejects_privileged_actor_key(self) -> None:
        labels = _labels()
        targets = derive_supervision(
            labels, minimum_target_pixels=8, free_space_max_depth_m=8.0
        )
        spec = CaptureSpec(
            sample_id="train-s0000000001-p000",
            split="train",
            scene_seed=1,
            sample_index=0,
            position_world_m=(0.0, 0.0, 1.0),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = write_sample(
                root,
                spec,
                np.zeros((12, 15, 3), dtype=np.uint8),
                labels,
                targets,
                "random_obstacles",
            )
            (root / "manifest.jsonl").write_text(
                json.dumps(asdict(record)) + "\n", encoding="utf-8"
            )
            np.savez_compressed(
                root / record.actor_file,
                sample_id=np.asarray(spec.sample_id),
                rgb=np.zeros((12, 15, 3), dtype=np.uint8),
                timestamp_s=np.asarray(0.0),
                depth_m=labels.depth_m,
            )

            with self.assertRaisesRegex(ValueError, "allowlist"):
                validate_dataset_leakage(root)

    def test_scene_count_is_single_source_for_realized_split_ratios(self) -> None:
        config = load_vision_dataset_config()
        selected = select_scene_seeds(config)
        total = sum(len(values) for values in selected.values())

        self.assertEqual(
            {split: len(values) / total for split, values in selected.items()},
            {
                "train": config.train_scene_count / total,
                "validation": config.validation_scene_count / total,
                "development_ood": config.development_ood_scene_count / total,
                "audit": config.audit_scene_count / total,
            },
        )

    def test_dataset_statistics_report_class_and_rgb_distribution(self) -> None:
        labels = _labels()
        targets = derive_supervision(
            labels, minimum_target_pixels=8, free_space_max_depth_m=8.0
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for split_index, split in enumerate(
                ("train", "validation", "development_ood", "audit")
            ):
                spec = CaptureSpec(
                    sample_id=f"{split}-s{split_index:010d}-p000",
                    split=split,  # type: ignore[arg-type]
                    scene_seed=split_index,
                    sample_index=0,
                    position_world_m=(0.0, 0.0, 1.0),
                    quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                )
                records.append(
                    write_sample(
                        root,
                        spec,
                        np.full((12, 15, 3), 64 + split_index, dtype=np.uint8),
                        labels,
                        targets,
                        "random_obstacles",
                        visual_domain=f"domain_{split}",
                    )
                )

            summaries = summarize_vision_dataset(root, records)

        self.assertEqual(summaries["train"].record_count, 1)
        self.assertEqual(summaries["train"].scene_count, 1)
        self.assertEqual(summaries["train"].target_visible_fraction, 1.0)
        self.assertEqual(summaries["train"].obstacle_visible_fraction, 1.0)
        self.assertEqual(
            summaries["development_ood"].visual_domains,
            ("domain_development_ood",),
        )
        self.assertEqual(summaries["audit"].visual_domains, ("domain_audit",))


if __name__ == "__main__":
    unittest.main()


class HoldPoseSamplingTests(unittest.TestCase):
    """REV-019: ADR-012 hover zarfından örneklenen poz ailesi."""

    def setUp(self) -> None:
        self.config = load_vision_dataset_config()
        self.scene_config = load_scene_config()
        self.target_config = load_target_area_config()

    def _scene(self, seed: int) -> StaticScene:
        return generate_static_scene(
            seed,
            self.scene_config,
            mode=self.config.scene_mode,
            endpoint_mode=self.config.endpoint_mode,
        )

    def _specs(self, seeds: range) -> list[tuple[CaptureSpec, HoverEnvelope]]:
        specs = []
        for seed in seeds:
            scene = self._scene(seed)
            envelope = target_area_from_scene(scene, self.target_config).hover_envelope
            for index in range(self.config.samples_per_scene):
                specs.append(
                    (
                        make_capture_spec(
                            scene,
                            "train",
                            index,
                            self.config,
                            self.scene_config,
                            target_config=self.target_config,
                        ),
                        envelope,
                    )
                )
        return specs

    def test_shipped_config_samples_the_hover_envelope(self) -> None:
        """Terminal görevin target_score kapısı bu poz bölgesine dayanır."""

        specs = self._specs(range(20260911, 20260961))
        inside = sum(
            1
            for spec, envelope in specs
            if np.all(
                np.asarray(spec.position_world_m) >= envelope.lower_world_m - 1e-9
            )
            and np.all(
                np.asarray(spec.position_world_m) <= envelope.upper_world_m + 1e-9
            )
        )

        self.assertGreater(self.config.hold_pose_fraction, 0.0)
        self.assertGreater(inside / len(specs), 0.15)

    def test_hold_poses_are_disabled_when_fraction_is_zero(self) -> None:
        config = replace(self.config, hold_pose_fraction=0.0)
        scene = self._scene(20260911)
        envelope = target_area_from_scene(scene, self.target_config).hover_envelope

        inside = 0
        for index in range(64):
            spec = make_capture_spec(
                scene,
                "train",
                index,
                config,
                self.scene_config,
                target_config=self.target_config,
            )
            position = np.asarray(spec.position_world_m)
            if np.all(position >= envelope.lower_world_m) and np.all(
                position <= envelope.upper_world_m
            ):
                inside += 1

        self.assertLess(inside, 8)

    def test_hold_pose_sampling_is_deterministic(self) -> None:
        first = self._specs(range(20260911, 20260916))
        second = self._specs(range(20260911, 20260916))

        self.assertEqual([spec for spec, _ in first], [spec for spec, _ in second])

    def test_endpoint_mode_is_randomized_for_route_diversity(self) -> None:
        """REV-018: tek rota geometrisi encoder genellemesini doğrulayamaz."""

        separations = {
            round(
                float(
                    np.linalg.norm(
                        self._scene(seed).goal_world_m - self._scene(seed).start_world_m
                    )
                ),
                3,
            )
            for seed in range(20260911, 20260941)
        }

        self.assertNotEqual(self.config.endpoint_mode, "fixed")
        self.assertGreater(len(separations), 20)

    def test_fraction_sum_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "rota pozlarına pay kalmıyor"):
            replace(
                self.config,
                hold_pose_fraction=0.75,
                target_negative_fraction=0.30,
            ).validate()
