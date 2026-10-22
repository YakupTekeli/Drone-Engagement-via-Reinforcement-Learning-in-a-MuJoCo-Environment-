"""Küçük vision encoder, metrik ve checkpoint smoke testleri."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from ebabil.perception import (
    CaptureSpec,
    PrivilegedCameraLabels,
    TinyVisionEncoder,
    color_baseline_predict,
    compute_metrics,
    derive_supervision,
    grayscale_actor_batch,
    load_approved_vision_encoder,
    load_vision_encoder_checkpoint,
    load_vision_encoder_config,
    load_vision_quality_config,
    perception_loss,
    preprocess_actor_rgb,
    run_vision_audit,
    run_vision_quality,
    summarize_vision_dataset,
    train_vision_encoder,
    write_sample,
)
from ebabil.perception.quality import _require_gated_obstacle_map


def _write_synthetic_dataset(root: Path) -> None:
    records = []
    for split_index, split in enumerate(
        ("train", "validation", "development_ood", "audit")
    ):
        for sample_index in range(2):
            height, width = 24, 32
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            target_mask = np.zeros((height, width), dtype=np.bool_)
            if sample_index == 0:
                rgb[6:14, 18:28, 1] = 220
                target_mask[6:14, 18:28] = True
            obstacle_mask = np.zeros((height, width), dtype=np.bool_)
            obstacle_mask[5:20, :] = True
            depth = np.full((height, width), 10.0, dtype=np.float64)
            depth[obstacle_mask] = 0.25
            labels = PrivilegedCameraLabels(
                depth_m=depth,
                segmentation=np.zeros((height, width, 2), dtype=np.int32),
                target_mask=target_mask,
                obstacle_mask=obstacle_mask,
                timestamp_s=float(sample_index),
            )
            targets = derive_supervision(
                labels, minimum_target_pixels=8, free_space_max_depth_m=8.0
            )
            spec = CaptureSpec(
                sample_id=f"{split}-s{split_index:010d}-p{sample_index:03d}",
                split=split,  # type: ignore[arg-type]
                scene_seed=split_index,
                sample_index=sample_index,
                position_world_m=(0.0, 0.0, 1.0),
                quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
            records.append(
                write_sample(
                    root,
                    spec,
                    rgb,
                    labels,
                    targets,
                    "random_obstacles",
                )
            )
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(asdict(record)) + "\n" for record in records),
        encoding="utf-8",
    )
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "vision-dataset-v4",
                "config": {
                    "schema_version": "vision-dataset-v4",
                    "synthetic_fixture": True,
                },
            }
        ),
        encoding="utf-8",
    )
    statistics = summarize_vision_dataset(root, records)
    (root / "statistics.json").write_text(
        json.dumps(
            {split: asdict(values) for split, values in statistics.items()},
            sort_keys=True,
        ),
        encoding="utf-8",
    )


class VisionEncoderTests(unittest.TestCase):
    def test_actor_rgb_preprocessing_is_shared_and_strict(self) -> None:
        rgb = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)

        image = preprocess_actor_rgb(rgb)

        self.assertEqual(image.shape, (3, 24, 32))
        self.assertEqual(image.dtype, torch.float32)
        np.testing.assert_allclose(
            image.permute(1, 2, 0).numpy(), rgb.astype(np.float32) / 255.0
        )
        with self.assertRaisesRegex(ValueError, "uint8"):
            preprocess_actor_rgb(rgb.astype(np.float32))

    def test_encoder_rejects_wrong_dtype_range_and_nonfinite_input(self) -> None:
        model_config, _ = load_vision_encoder_config()
        model = TinyVisionEncoder(model_config)

        with self.assertRaisesRegex(ValueError, "floating point"):
            model(torch.zeros((1, 3, 24, 32), dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            model(torch.full((1, 3, 24, 32), 128.0))
        invalid = torch.zeros((1, 3, 24, 32))
        invalid[0, 0, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "sonlu"):
            model(invalid)

    def test_grayscale_ablation_removes_hue_but_preserves_shape(self) -> None:
        images = torch.tensor(
            [[[[1.0]], [[0.25]], [[0.0]]], [[[0.0]], [[0.5]], [[1.0]]]]
        )

        grayscale = grayscale_actor_batch(images)

        self.assertEqual(grayscale.shape, images.shape)
        torch.testing.assert_close(grayscale[:, 0], grayscale[:, 1])
        torch.testing.assert_close(grayscale[:, 1], grayscale[:, 2])

    def test_encoder_output_shapes_and_ranges(self) -> None:
        model_config, _ = load_vision_encoder_config()
        model = TinyVisionEncoder(model_config)

        output = model(torch.zeros((3, 3, 120, 160), dtype=torch.float32))

        self.assertEqual(output.latent.shape, (3, model_config.latent_size))
        self.assertEqual(output.target_logit.shape, (3,))
        self.assertEqual(output.target_center_xy.shape, (3, 2))
        self.assertEqual(output.free_space.shape, (3, 3))
        self.assertEqual(model_config.schema_version, "vision-encoder-v9")
        self.assertEqual(model.latent[1].in_features, 32 * 3 * 4)
        self.assertEqual(
            model.target_visibility_head.in_features, model_config.latent_size + 2
        )
        self.assertEqual(output.target_heatmap_logits.shape, (3, 9, 12))
        self.assertEqual(output.obstacle_heatmap_logits.shape, (3, 9, 12))
        self.assertEqual(output.collision_risk_logits.shape, (3, 3))
        self.assertTrue(torch.all(output.target_center_xy.abs() <= 1.0))
        self.assertTrue(torch.all(0.0 <= output.free_space))
        self.assertTrue(torch.all(output.free_space <= 1.0))

    def test_encoder_heatmaps_support_nondefault_configured_resolutions(self) -> None:
        model_config, _ = load_vision_encoder_config()
        for height, width in ((64, 64), (80, 128)):
            with self.subTest(height=height, width=width):
                configured = replace(
                    model_config,
                    input_height_px=height,
                    input_width_px=width,
                )
                output = TinyVisionEncoder(configured)(
                    torch.zeros((2, 3, 120, 160), dtype=torch.float32)
                )

                self.assertEqual(output.target_heatmap_logits.shape, (2, 9, 12))
                self.assertEqual(output.obstacle_heatmap_logits.shape, (2, 9, 12))
                self.assertEqual(output.target_center_xy.shape, (2, 2))

    def test_channel_adaptive_exposure_rejects_global_color_gain(self) -> None:
        model_config, _ = load_vision_encoder_config()
        model = TinyVisionEncoder(model_config)
        base = torch.linspace(0.08, 0.24, 24 * 32).reshape(1, 1, 24, 32)
        image = torch.cat((base, base * 0.8, base * 1.2), dim=1)
        color_shifted = image * torch.tensor([1.4, 0.75, 1.1]).reshape(1, 3, 1, 1)

        first = model(image)
        second = model(color_shifted)

        torch.testing.assert_close(first.latent, second.latent, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            first.collision_risk_logits,
            second.collision_risk_logits,
            atol=2e-6,
            rtol=2e-6,
        )

    def test_metrics_have_expected_values(self) -> None:
        metrics = compute_metrics(
            target_scores=np.array([0.9, 0.8, 0.1, 0.2]),
            predicted_centers=np.zeros((4, 2)),
            predicted_free_space=np.ones((4, 3)),
            target_visible=np.array([1, 0, 1, 0]),
            target_centers=np.zeros((4, 2)),
            target_free_space=np.full((4, 3), 0.5),
        )

        self.assertEqual(metrics.target_precision, 0.5)
        self.assertEqual(metrics.target_recall, 0.5)
        self.assertEqual(metrics.target_center_mean_error, 0.0)
        self.assertEqual(metrics.free_space_mae, 0.5)
        self.assertEqual(metrics.free_space_mae_left, 0.5)
        self.assertEqual(metrics.free_space_mae_center, 0.5)
        self.assertEqual(metrics.free_space_mae_right, 0.5)
        self.assertEqual(metrics.risk_precision, 0.0)
        self.assertEqual(metrics.risk_recall, 0.0)
        self.assertEqual(metrics.risk_specificity, 1.0)
        self.assertEqual(metrics.risk_balanced_accuracy, 0.5)

    def test_green_color_baseline_detects_target(self) -> None:
        images = torch.zeros((2, 3, 20, 20), dtype=torch.float32)
        images[0, 1, 5:10, 7:15] = 0.9

        scores, centers, free_space = color_baseline_predict(images)

        np.testing.assert_array_equal(scores, np.array([1.0, 0.0]))
        self.assertGreater(centers[0, 0], 0.0)
        np.testing.assert_array_equal(free_space, np.ones((2, 3)))

    def test_short_training_writes_reloadable_versioned_checkpoint(self) -> None:
        model_config, training_config = load_vision_encoder_config()
        training_config = replace(
            training_config,
            epochs=1,
            batch_size=2,
            latency_warmup_runs=0,
            latency_measurement_runs=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            _write_synthetic_dataset(dataset)

            artifacts = train_vision_encoder(
                dataset, root / "run", model_config, training_config
            )
            reloaded = load_vision_encoder_checkpoint(artifacts.checkpoint)
            metrics = json.loads(artifacts.metrics.read_text(encoding="utf-8"))

            self.assertTrue(artifacts.checkpoint.is_file())
            self.assertEqual(reloaded.config, model_config)
            self.assertIn("validation", metrics)
            self.assertIn("color_baseline_validation", metrics)
            self.assertIn("development_ood", metrics)
            self.assertIn("color_baseline_development_ood", metrics)
            self.assertIn("grayscale_validation", metrics)
            self.assertIn("grayscale_development_ood", metrics)
            self.assertNotIn("audit", metrics)
            output = reloaded(torch.zeros((1, 3, 24, 32)))
            self.assertEqual(output.latent.shape, (1, model_config.latent_size))

    def test_multi_seed_quality_runner_writes_aggregate_report(self) -> None:
        model_config, training_config = load_vision_encoder_config()
        quality_config = load_vision_quality_config()
        training_config = replace(
            training_config,
            epochs=1,
            batch_size=2,
            latency_warmup_runs=0,
            latency_measurement_runs=1,
        )
        quality_config = replace(quality_config, model_seeds=(123,))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            _write_synthetic_dataset(dataset)

            artifacts = run_vision_quality(
                dataset,
                root / "quality",
                model_config,
                training_config,
                quality_config,
            )
            report = json.loads(artifacts.report.read_text(encoding="utf-8"))

            self.assertTrue(artifacts.candidate_checkpoint.is_file())
            self.assertEqual(report["schema_version"], "vision-quality-report-v2")
            self.assertEqual(report["candidate_seed"], 123)
            self.assertFalse(Path(report["candidate_checkpoint"]).is_absolute())
            self.assertEqual(report["dataset"]["schema_version"], "vision-dataset-v4")
            self.assertEqual(len(report["dataset"]["config_sha256"]), 64)
            self.assertEqual(len(report["dataset"]["manifest_sha256"]), 64)
            self.assertEqual(len(report["candidate_checkpoint_sha256"]), 64)
            self.assertIn("validation", report["aggregate"])
            self.assertIn("development_ood", report["aggregate"])
            self.assertNotIn("audit", report["aggregate"])
            self.assertIn(
                "grayscale_free_space_mae",
                report["checks"]["123"]["validation"],
            )
            self.assertIsInstance(report["passed"], bool)

    def test_quality_cli_returns_nonzero_and_withholds_approval_on_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            _write_synthetic_dataset(dataset)

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_vision_quality.py",
                    str(dataset),
                    str(root / "quality"),
                    "--epochs",
                    "1",
                    "--seeds",
                    "123",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("quality_passed=False", completed.stdout)
            self.assertNotIn("approved_checkpoint=", completed.stdout)

    def test_audit_approves_frozen_candidate_and_loader_verifies_hash_chain(
        self,
    ) -> None:
        model_config, training_config = load_vision_encoder_config()
        quality_config = load_vision_quality_config()
        training_config = replace(
            training_config,
            epochs=1,
            batch_size=2,
            latency_warmup_runs=0,
            latency_measurement_runs=1,
        )
        quality_config = replace(
            quality_config,
            model_seeds=(123,),
            minimum_target_precision=0.0,
            minimum_target_recall=0.0,
            maximum_target_center_error=10.0,
            maximum_free_space_mae=1.0,
            minimum_free_space_improvement=0.0,
            minimum_risk_precision=0.0,
            minimum_risk_recall=0.0,
            minimum_risk_specificity=0.0,
            minimum_obstacle_map_iou=0.0,
            minimum_obstacle_map_precision=0.0,
            minimum_obstacle_map_recall=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            _write_synthetic_dataset(dataset)
            quality = run_vision_quality(
                dataset,
                root / "quality",
                model_config,
                training_config,
                quality_config,
            )
            self.assertTrue(quality.passed, quality.report.read_text(encoding="utf-8"))

            audit = run_vision_audit(dataset, quality.report, root / "audit")

            self.assertTrue(audit.passed)
            self.assertIsNotNone(audit.approval_manifest)
            assert audit.approval_manifest is not None
            approved = load_approved_vision_encoder(audit.approval_manifest)
            self.assertEqual(approved.config, model_config)
            audit_report = json.loads(audit.report.read_text(encoding="utf-8"))
            self.assertIn("grayscale_audit", audit_report["metrics"])
            with self.assertRaisesRegex(RuntimeError, "daha önce tüketti"):
                run_vision_audit(dataset, quality.report, root / "second-audit")

            manifest = json.loads(audit.approval_manifest.read_text(encoding="utf-8"))
            manifest["approved_checkpoint_sha256"] = "0" * 64
            audit.approval_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_approved_vision_encoder(audit.approval_manifest)

    def test_training_rejects_incompatible_dataset_schema(self) -> None:
        model_config, training_config = load_vision_encoder_config()
        training_config = replace(training_config, epochs=1, batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            _write_synthetic_dataset(dataset)
            (dataset / "metadata.json").write_text(
                json.dumps({"schema_version": "vision-dataset-v3"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "vision-dataset-v4"):
                train_vision_encoder(
                    dataset,
                    root / "run",
                    model_config,
                    training_config,
                )


if __name__ == "__main__":
    unittest.main()


class CollisionRiskPositiveWeightTests(unittest.TestCase):
    """REV-020: risk BCE'sinde pozitif sınıf ağırlığı."""

    def _batch(self) -> tuple[torch.Tensor, ...]:
        torch.manual_seed(20260911)
        model_config, training_config = load_vision_encoder_config()
        model = TinyVisionEncoder(model_config)
        images = torch.rand(
            6, 3, model_config.input_height_px, model_config.input_width_px
        )
        output = model(images)
        target_visible = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        target_center_xy = torch.zeros(6, 2)
        free_space = torch.tensor(
            [
                [0.10, 0.90, 0.50],
                [0.80, 0.85, 0.90],
                [0.05, 0.10, 0.20],
                [0.95, 0.95, 0.95],
                [0.15, 0.60, 0.30],
                [0.40, 0.20, 0.70],
            ]
        )
        return output, target_visible, target_center_xy, free_space, training_config

    def test_unit_weight_matches_unweighted_cross_entropy(self) -> None:
        """`1.0` ağırlıksız BCE ile aynı olmalı.

        Onaylı `vision-encoder-v9` ağırlıksız kayıpla eğitildi; gönderilen
        config'in onu tekrar üretebilmesi bu eşitliğe dayanır.
        """

        output, visible, center, free_space, training = self._batch()
        risk = (free_space < training.collision_risk_threshold).to(
            output.collision_risk_logits.dtype
        )
        expected = functional.binary_cross_entropy_with_logits(
            output.collision_risk_logits, risk
        )
        weighted = functional.binary_cross_entropy_with_logits(
            output.collision_risk_logits,
            risk,
            pos_weight=torch.tensor(1.0, dtype=output.collision_risk_logits.dtype),
        )

        self.assertAlmostEqual(
            float(expected.detach()), float(weighted.detach()), places=12
        )

    def test_shipped_weight_is_behaviour_neutral(self) -> None:
        _, training_config = load_vision_encoder_config()

        self.assertAlmostEqual(training_config.collision_risk_positive_weight, 1.0)

    def test_higher_weight_increases_loss_on_missed_positives(self) -> None:
        """Ağırlık artışı kaçırılan riskleri (false negative) daha pahalı yapar."""

        output, visible, center, free_space, training = self._batch()
        height, width = output.target_heatmap_logits.shape[-2:]
        masks = torch.zeros(6, height, width)
        masks[0, :2, :2] = 1.0
        losses = [
            float(
                perception_loss(
                    output,
                    visible,
                    center,
                    free_space,
                    masks,
                    masks,
                    replace(training, collision_risk_positive_weight=weight),
                ).detach()
            )
            for weight in (1.0, 2.0, 3.0)
        ]

        self.assertLess(losses[0], losses[1])
        self.assertLess(losses[1], losses[2])

    def test_non_positive_weight_is_rejected(self) -> None:
        _, training_config = load_vision_encoder_config()

        with self.assertRaisesRegex(ValueError, "pozitif"):
            replace(training_config, collision_risk_positive_weight=0.0).validate()


class RiskHeadCapacityTests(unittest.TestCase):
    """REV-021: risk head kapasitesi konfigüre edilebilir olmalı."""

    def test_shipped_v9_config_keeps_linear_risk_head(self) -> None:
        """Onaylı checkpoint'in yüklenebilirliği buna bağlıdır."""

        model_config, _ = load_vision_encoder_config()
        model = TinyVisionEncoder(model_config)

        self.assertEqual(model_config.schema_version, "vision-encoder-v9")
        self.assertEqual(model_config.risk_head_hidden_size, 0)
        self.assertIsInstance(model.collision_risk_head, torch.nn.Linear)

    def test_current_chain_loads_and_legacy_chain_is_refused(self) -> None:
        """ADR-021: `obstacle_map` kapısı olmayan zincir fail-closed reddedilir.

        `vision-v2` observation'ı engel haritasını actor'a verir. Eski zincirler
        (`vision_audit_v2`, v9) bu kanalı hiç ölçmemişti, bu yüzden artık
        yüklenmemeleri gerekir; aksi halde politikaya doğruluğu bilinmeyen bir
        kanal verilmiş olurdu.
        """

        current = Path("runs/vision_audit_v4_20260913/approval_manifest.json")
        legacy = Path("runs/vision_audit_v2_20260911/approval_manifest.json")
        if not current.is_file():
            self.skipTest("Güncel approval manifest bu ortamda yok")

        model = load_approved_vision_encoder(current)

        self.assertEqual(model.config.schema_version, "vision-encoder-v11")
        self.assertIsInstance(model.collision_risk_head, torch.nn.Sequential)

        if legacy.is_file():
            with self.assertRaisesRegex(ValueError, "engel haritası kapısı"):
                load_approved_vision_encoder(legacy)

    def test_v10_config_builds_hidden_risk_head(self) -> None:
        path = Path("src/ebabil/configs/training/vision_encoder_v10.toml")
        if not path.is_file():
            self.skipTest("v10 deney config'i yok")
        model_config, training_config = load_vision_encoder_config(path)
        model_config.validate()
        model = TinyVisionEncoder(model_config)

        self.assertEqual(model_config.schema_version, "vision-encoder-v10")
        self.assertGreater(model_config.risk_head_hidden_size, 0)
        self.assertIsInstance(model.collision_risk_head, torch.nn.Sequential)
        self.assertGreater(training_config.collision_risk_positive_weight, 1.0)

    def test_hidden_risk_head_requires_v10_schema(self) -> None:
        model_config, _ = load_vision_encoder_config()

        with self.assertRaisesRegex(ValueError, "yalnız vision-encoder-v10"):
            replace(model_config, risk_head_hidden_size=64).validate()

    def test_negative_hidden_size_is_rejected(self) -> None:
        model_config, _ = load_vision_encoder_config()

        with self.assertRaisesRegex(ValueError, "risk_head_hidden_size"):
            replace(model_config, risk_head_hidden_size=-1).validate()

    def test_risk_head_output_shape_is_unchanged(self) -> None:
        """Kapasite değişse de üç bölge çıktısı korunmalı."""

        for path in (None, Path("src/ebabil/configs/training/vision_encoder_v10.toml")):
            if path is not None and not path.is_file():
                continue
            with self.subTest(config=str(path)):
                model_config, _ = (
                    load_vision_encoder_config()
                    if path is None
                    else load_vision_encoder_config(path)
                )
                model = TinyVisionEncoder(model_config)
                images = torch.rand(
                    2, 3, model_config.input_height_px, model_config.input_width_px
                )

                with torch.inference_mode():
                    output = model(images)

                self.assertEqual(output.collision_risk_logits.shape, (2, 3))


class ObstacleMapApprovalGateTests(unittest.TestCase):
    """ADR-021: `obstacle_map` kanalı ölçülmüş bir zincirden gelmek zorunda."""

    def _quality(self, checks: dict[str, object]) -> dict[str, object]:
        return {"schema_version": "vision-quality-report-v2", "checks": checks}

    def test_missing_map_gate_is_refused(self) -> None:
        """Kapı hiç yoksa fail-closed: kanal ölçülmemiş demektir."""

        quality = self._quality({"20260911": {"validation": {"target_recall": True}}})
        with self.assertRaisesRegex(ValueError, "engel haritası kapısını içermiyor"):
            _require_gated_obstacle_map(quality)

    def test_failed_map_gate_is_refused(self) -> None:
        quality = self._quality(
            {
                "20260911": {
                    "validation": {
                        "obstacle_map_iou": True,
                        "obstacle_map_precision": False,
                        "obstacle_map_recall": True,
                    }
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "obstacle_map_precision"):
            _require_gated_obstacle_map(quality)

    def test_passing_map_gate_is_accepted(self) -> None:
        quality = self._quality(
            {
                "20260911": {
                    "validation": {
                        "obstacle_map_iou": True,
                        "obstacle_map_precision": True,
                        "obstacle_map_recall": True,
                    }
                }
            }
        )

        self.assertIsNone(_require_gated_obstacle_map(quality))

    def test_missing_checks_block_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "checks"):
            _require_gated_obstacle_map({"schema_version": "x"})

    def test_risk_gate_is_opt_in_and_defaults_to_enabled(self) -> None:
        """REV-038: varsayılan `True`, yani v9/v10 zincirinin anlamı korunur."""

        legacy = load_vision_quality_config(
            "src/ebabil/configs/training/vision_encoder_v10.toml"
        )
        current = load_vision_quality_config(
            "src/ebabil/configs/training/vision_encoder_v11.toml"
        )

        self.assertTrue(legacy.gate_collision_risk)
        self.assertFalse(current.gate_collision_risk)
