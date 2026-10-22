"""ADR-022: etiketsiz VAE, probe ölçümü ve onay zinciri testleri."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from ebabil.perception import (
    CaptureSpec,
    PrivilegedCameraLabels,
    ProbeConfig,
    ProbeLabels,
    VisualVAE,
    VisualVAEOutput,
    average_precision,
    binary_auroc,
    derive_supervision,
    encode_actor_rgb,
    evaluate_linear_probes,
    fit_linear_probes,
    kl_per_dimension,
    load_approved_visual_vae,
    load_visual_vae_checkpoint,
    load_visual_vae_config,
    load_visual_vae_quality_config,
    predict_linear_probes,
    run_visual_vae_audit,
    run_visual_vae_quality,
    summarize_vision_dataset,
    threshold_free_metrics,
    train_visual_vae,
    vae_loss,
    write_sample,
)
from ebabil.perception.vae_quality import REQUIRED_GATES, _require_gated_probes


def _write_synthetic_dataset(root: Path, samples_per_split: int = 4) -> None:
    """Küçük ama probe'lanabilir sentetik dataset: görünürlük ve engel derinliği değişir."""

    records = []
    rng = np.random.default_rng(20260922)
    for split_index, split in enumerate(
        ("train", "validation", "development_ood", "audit")
    ):
        for sample_index in range(samples_per_split):
            height, width = 24, 32
            rgb = rng.integers(0, 60, size=(height, width, 3), dtype=np.uint8)
            target_mask = np.zeros((height, width), dtype=np.bool_)
            if sample_index % 2 == 0:
                rgb[6:14, 18:28, 1] = 220
                target_mask[6:14, 18:28] = True
            obstacle_mask = np.zeros((height, width), dtype=np.bool_)
            near = sample_index % 4 < 2
            rows = slice(5, 20) if near else slice(16, 20)
            obstacle_mask[rows, :] = True
            rgb[rows, :, 0] = 200 if near else 90
            depth = np.full((height, width), 10.0, dtype=np.float64)
            depth[obstacle_mask] = 0.25 if near else 6.0
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
                write_sample(root, spec, rgb, labels, targets, "random_obstacles")
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


def _short_training_config():  # type: ignore[no-untyped-def]
    model_config, training_config = load_visual_vae_config()
    return model_config, replace(
        training_config,
        epochs=1,
        batch_size=2,
        latency_warmup_runs=0,
        latency_measurement_runs=1,
    )


def _relaxed_quality_config(seeds: tuple[int, ...] = (123,)):  # type: ignore[no-untyped-def]
    quality = load_visual_vae_quality_config()
    return replace(
        quality,
        model_seeds=seeds,
        minimum_active_latent_units=0,
        maximum_reconstruction_mse=1.0,
        maximum_free_space_mae=1.0,
        minimum_target_auroc=0.0,
        minimum_target_average_precision=0.0,
        minimum_obstacle_map_best_iou=0.0,
        probe=replace(quality.probe, logistic_iterations=20, batch_size=4),
    )


class VisualVAETests(unittest.TestCase):
    def test_default_config_loads_and_rejects_invalid_values(self) -> None:
        model_config, training_config = load_visual_vae_config()
        quality_config = load_visual_vae_quality_config()

        self.assertEqual(model_config.schema_version, "visual-vae-v1")
        self.assertEqual(model_config.latent_size, 128)
        self.assertEqual(
            (model_config.input_height_px, model_config.input_width_px), (72, 96)
        )
        self.assertEqual(training_config.kl_weight, 1.0)
        self.assertEqual(quality_config.minimum_active_latent_units, 32)
        self.assertEqual(quality_config.maximum_free_space_mae, 0.30)
        self.assertEqual(quality_config.minimum_target_auroc, 0.80)
        self.assertEqual(quality_config.minimum_target_average_precision, 0.85)
        self.assertEqual(quality_config.minimum_obstacle_map_best_iou, 0.45)
        self.assertNotIn("obstacle_map_iou", REQUIRED_GATES)
        with self.assertRaisesRegex(ValueError, "schema_version"):
            replace(model_config, schema_version="visual-vae-v0").validate()
        with self.assertRaisesRegex(ValueError, "channel_widths"):
            replace(model_config, channel_widths=(16,)).validate()
        with self.assertRaisesRegex(ValueError, "sıfıra indirir"):
            replace(model_config, channel_widths=(8,) * 8).validate()
        with self.assertRaisesRegex(ValueError, "KL ağırlığı"):
            replace(training_config, kl_weight=-1.0).validate()
        with self.assertRaisesRegex(ValueError, "olasılık"):
            replace(quality_config, minimum_target_auroc=1.5).validate()

    def test_output_shapes_ranges_and_input_validation(self) -> None:
        model_config, _ = load_visual_vae_config()
        model = VisualVAE(model_config)

        output = model(torch.rand((3, 3, 120, 160), dtype=torch.float32))

        self.assertIsInstance(output, VisualVAEOutput)
        self.assertEqual(output.mean.shape, (3, model_config.latent_size))
        self.assertEqual(output.log_variance.shape, (3, model_config.latent_size))
        self.assertEqual(output.reconstruction.shape, (3, 3, 72, 96))
        self.assertEqual(output.target.shape, (3, 3, 72, 96))
        self.assertTrue(torch.all(output.reconstruction >= 0.0))
        self.assertTrue(torch.all(output.reconstruction <= 1.0))
        with self.assertRaisesRegex(ValueError, "floating point"):
            model(torch.zeros((1, 3, 24, 32), dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            model(torch.full((1, 3, 24, 32), 128.0))
        invalid = torch.zeros((1, 3, 24, 32))
        invalid[0, 0, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "sonlu"):
            model(invalid)

    def test_decoder_mirrors_encoder_for_other_resolutions(self) -> None:
        model_config, _ = load_visual_vae_config()
        for height, width in ((64, 64), (80, 128), (36, 48)):
            with self.subTest(height=height, width=width):
                configured = replace(
                    model_config, input_height_px=height, input_width_px=width
                )
                output = VisualVAE(configured)(torch.rand((2, 3, 30, 40)))

                self.assertEqual(output.reconstruction.shape, (2, 3, height, width))

    def test_encode_uses_posterior_mean_and_is_deterministic(self) -> None:
        model_config, _ = load_visual_vae_config()
        model = VisualVAE(model_config).eval()
        images = torch.rand((2, 3, 24, 32))

        first, _ = model.encode(images)
        second, _ = model.encode(images)
        output = model(images)

        torch.testing.assert_close(first, second)
        torch.testing.assert_close(output.latent, output.mean)
        model.train()
        sampled = model(images)
        self.assertFalse(torch.allclose(sampled.latent, sampled.mean))

    def test_photometric_augmentation_keeps_clean_target(self) -> None:
        """REV-049: bozulmuş giriş, temiz hedef; augmentation seed'li ve [0, 1] içinde."""

        from ebabil.perception import photometric_augment_batch

        model_config, training_config = load_visual_vae_config()
        self.assertEqual(training_config.photometric_augmentation_probability, 0.0)
        with self.assertRaisesRegex(ValueError, "photometric"):
            replace(
                training_config, photometric_augmentation_probability=1.5
            ).validate()
        images = torch.rand((3, 3, 24, 32))
        first = photometric_augment_batch(images, torch.Generator().manual_seed(1))
        second = photometric_augment_batch(images, torch.Generator().manual_seed(1))
        third = photometric_augment_batch(images, torch.Generator().manual_seed(2))
        torch.testing.assert_close(first, second)
        self.assertFalse(torch.allclose(first, third))
        self.assertTrue(torch.all(first >= 0.0) and torch.all(first <= 1.0))
        model = VisualVAE(model_config).eval()
        output = model(first, target_images=images)
        torch.testing.assert_close(output.target, model.normalize(images))
        self.assertFalse(torch.allclose(output.target, model.normalize(first)))
        with self.assertRaisesRegex(ValueError, "aynı batch"):
            model(first, target_images=images[:2])

    def test_contrastive_head_crop_and_info_nce(self) -> None:
        """REV-049: kontrastif kaldıraç varsayılan kapalı; açılınca tutarlı çalışmalı."""

        from ebabil.perception import info_nce_loss, random_resized_crop_batch

        model_config, training_config = load_visual_vae_config()
        self.assertEqual(model_config.contrastive_projection_size, 0)
        self.assertEqual(training_config.contrastive_weight, 0.0)
        self.assertIsNone(VisualVAE(model_config).projection_head)
        with self.assertRaisesRegex(ValueError, "projeksiyon"):
            VisualVAE(model_config).project(torch.rand((2, 3, 24, 32)))
        with self.assertRaisesRegex(ValueError, "contrastive_temperature"):
            replace(training_config, contrastive_temperature=0.0).validate()

        images = torch.rand((4, 3, 24, 32))
        crops = random_resized_crop_batch(
            images, torch.Generator().manual_seed(3), minimum_fraction=0.5
        )
        self.assertEqual(crops.shape, images.shape)
        self.assertTrue(torch.all(crops >= 0.0) and torch.all(crops <= 1.0))

        aligned = torch.eye(4)
        shuffled = aligned[[1, 0, 3, 2]]
        self.assertLess(
            float(info_nce_loss(aligned, aligned, temperature=0.1)),
            float(info_nce_loss(aligned, shuffled, temperature=0.1)),
        )

        contrastive_model = VisualVAE(
            replace(model_config, contrastive_projection_size=16)
        ).eval()
        projection = contrastive_model.project(images)
        self.assertEqual(projection.shape, (4, 16))
        torch.testing.assert_close(
            projection.norm(dim=1), torch.ones(4), atol=1e-5, rtol=1e-5
        )

    def test_contrastive_training_runs_and_requires_projection_head(self) -> None:
        model_config, training_config = _short_training_config()
        contrastive_training = replace(
            training_config, contrastive_weight=0.5, batch_size=4
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            _write_synthetic_dataset(dataset)
            with self.assertRaisesRegex(ValueError, "contrastive_projection_size"):
                train_visual_vae(
                    dataset, root / "bad", model_config, contrastive_training
                )
            artifacts = train_visual_vae(
                dataset,
                root / "run",
                replace(model_config, contrastive_projection_size=16),
                contrastive_training,
            )
            metrics = json.loads(artifacts.metrics.read_text(encoding="utf-8"))
            self.assertIn("contrastive", metrics["epoch_losses"][0])
            reloaded = load_visual_vae_checkpoint(artifacts.checkpoint)
            self.assertIsNotNone(reloaded.projection_head)

    def test_threshold_free_metrics_definitions(self) -> None:
        """REV-049: AUROC/AP tanımları; mükemmel sıralama 1.0, tek sınıf 0.5."""

        labels = np.array([True, True, False, False])
        self.assertAlmostEqual(
            binary_auroc(np.array([0.9, 0.8, 0.2, 0.1]), labels), 1.0
        )
        self.assertAlmostEqual(
            binary_auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels), 0.0
        )
        self.assertAlmostEqual(
            binary_auroc(np.array([0.5, 0.5, 0.5, 0.5]), labels), 0.5
        )
        self.assertAlmostEqual(
            binary_auroc(np.array([0.5, 0.5]), np.array([True, True])), 0.5
        )
        self.assertAlmostEqual(
            average_precision(np.array([0.9, 0.8, 0.2, 0.1]), labels), 1.0
        )
        self.assertAlmostEqual(
            average_precision(np.array([0.9, 0.1, 0.8, 0.2]), labels),
            (1.0 + 2.0 / 4.0) / 2.0,
        )
        self.assertEqual(average_precision(np.array([0.3]), np.array([False])), 0.0)

    def test_loss_components_and_kl_identity(self) -> None:
        mean = torch.zeros((4, 8))
        log_variance = torch.zeros((4, 8))

        self.assertEqual(float(kl_per_dimension(mean, log_variance).sum()), 0.0)
        kl = kl_per_dimension(torch.ones((1, 1)), torch.zeros((1, 1)))
        self.assertAlmostEqual(float(kl), 0.5, places=6)

        target = torch.rand((2, 3, 8, 8))
        output = VisualVAEOutput(
            mean=torch.ones((2, 4)),
            log_variance=torch.zeros((2, 4)),
            latent=torch.ones((2, 4)),
            reconstruction=target.clone(),
            target=target,
        )
        breakdown = vae_loss(output, kl_weight=2.0, free_bits_nats=0.0)
        self.assertAlmostEqual(float(breakdown.reconstruction), 0.0, places=6)
        self.assertAlmostEqual(float(breakdown.kl), 2.0, places=6)
        self.assertAlmostEqual(float(breakdown.total), 4.0, places=6)
        free_bits = vae_loss(output, kl_weight=1.0, free_bits_nats=1.0)
        self.assertAlmostEqual(float(free_bits.total), 4.0, places=6)

    def test_trainer_never_opens_privileged_archives(self) -> None:
        """ADR-022 doğrulama kapısı 1: casus + privileged dizini silinmiş kopya."""

        model_config, training_config = _short_training_config()
        original_load = np.load
        opened: list[str] = []

        def spy(path, *args, **kwargs):  # type: ignore[no-untyped-def]
            opened.append(str(path))
            return original_load(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            _write_synthetic_dataset(dataset)
            with mock.patch("numpy.load", side_effect=spy):
                artifacts = train_visual_vae(
                    dataset, root / "run", model_config, training_config
                )
            self.assertGreater(len(opened), 0)
            self.assertFalse(
                [path for path in opened if "privileged" in Path(path).parts],
                "Trainer privileged arşiv açtı",
            )
            metrics = json.loads(artifacts.metrics.read_text(encoding="utf-8"))
            self.assertTrue(metrics["label_free"])
            self.assertNotIn("free_space_mae", json.dumps(metrics))

            stripped = root / "stripped"
            shutil.copytree(
                dataset, stripped, ignore=shutil.ignore_patterns("privileged")
            )
            self.assertFalse((stripped / "privileged").exists())
            stripped_artifacts = train_visual_vae(
                stripped, root / "stripped-run", model_config, training_config
            )
            self.assertTrue(stripped_artifacts.checkpoint.is_file())

    def test_checkpoint_roundtrip_and_actor_rgb_encoding(self) -> None:
        model_config, training_config = _short_training_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            _write_synthetic_dataset(dataset)
            artifacts = train_visual_vae(
                dataset, root / "run", model_config, training_config
            )
            reloaded = load_visual_vae_checkpoint(artifacts.checkpoint)
            metrics = json.loads(artifacts.metrics.read_text(encoding="utf-8"))

            self.assertEqual(reloaded.config, model_config)
            self.assertFalse(reloaded.training)
            for key in ("validation", "development_ood", "grayscale_validation"):
                self.assertIn(key, metrics)
                self.assertIn("active_latent_units", metrics[key])
                self.assertIn("reconstruction_mse", metrics[key])
            self.assertNotIn("audit", metrics)
            payload = torch.load(
                artifacts.checkpoint, map_location="cpu", weights_only=True
            )
            self.assertEqual(
                payload["checkpoint_format_version"], "ebabil-vae-checkpoint-v1"
            )
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            latent = encode_actor_rgb(reloaded, rgb)
            self.assertEqual(latent.shape, (model_config.latent_size,))
            self.assertTrue(np.all(np.isfinite(latent)))
            with self.assertRaisesRegex(ValueError, "uint8"):
                encode_actor_rgb(reloaded, rgb.astype(np.float32))
            with self.assertRaisesRegex(FileExistsError, "üzerine yazılmadı"):
                train_visual_vae(dataset, root / "run", model_config, training_config)

    def test_linear_probes_recover_synthetic_linear_structure(self) -> None:
        rng = np.random.default_rng(7)
        latents = rng.normal(size=(400, 8))
        free_space = np.clip(0.5 + 0.2 * latents[:, :3], 0.0, 1.0)
        visible = latents[:, 3] > 0.0
        centers = np.clip(
            np.stack((0.4 * latents[:, 4], -0.4 * latents[:, 5]), axis=1), -1, 1
        )
        obstacle_map = np.repeat(
            (latents[:, 6] > 0.0).astype(np.float64)[:, None, None], 9, axis=1
        ).repeat(12, axis=2)
        labels = ProbeLabels(visible, centers, free_space, obstacle_map)
        config = ProbeConfig(
            ridge_lambda=0.1, logistic_l2=1e-4, logistic_iterations=100, batch_size=8
        )

        probes = fit_linear_probes(latents[:300], _slice(labels, slice(0, 300)), config)
        metrics = evaluate_linear_probes(
            probes, latents[300:], _slice(labels, slice(300, None))
        )

        # Sınıf sınırı z = 0 üzerinde; sınıra çok yakın örneklerde karar
        # yanılabilir, bu yüzden eşikler mükemmel değil "yapıyı çözüyor" der.
        self.assertLess(metrics.free_space_mae, 0.05)
        self.assertGreater(metrics.target_precision, 0.90)
        self.assertGreater(metrics.target_recall, 0.90)
        self.assertLess(metrics.target_center_mean_error, 0.1)
        self.assertGreater(metrics.obstacle_map_iou, 0.90)
        threshold_free = threshold_free_metrics(
            predict_linear_probes(probes, latents[300:]),
            _slice(labels, slice(300, None)),
        )
        self.assertGreater(threshold_free.target_auroc, 0.97)
        self.assertGreater(threshold_free.target_average_precision, 0.97)
        self.assertGreater(threshold_free.obstacle_map_best_iou, 0.90)

    def test_quality_audit_chain_loader_and_hash_verification(self) -> None:
        model_config, training_config = _short_training_config()
        quality_config = _relaxed_quality_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            _write_synthetic_dataset(dataset)

            quality = run_visual_vae_quality(
                dataset, root / "quality", model_config, training_config, quality_config
            )
            report = json.loads(quality.report.read_text(encoding="utf-8"))

            self.assertTrue(quality.passed, quality.report.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], "visual-vae-quality-report-v1")
            self.assertEqual(report["kind"], "visual-vae")
            self.assertEqual(report["candidate_seed"], 123)
            self.assertFalse(Path(report["candidate_checkpoint"]).is_absolute())
            self.assertIn("probe_free_space_mae", report["aggregate"]["validation"])
            self.assertNotIn("audit", report["aggregate"])
            for gate in REQUIRED_GATES:
                self.assertIn(gate, report["checks"]["123"]["validation"])
            self.assertTrue(
                (root / "quality" / "seed_123" / "probe_metrics.json").is_file()
            )
            seed_metrics = json.loads(
                (root / "quality" / "seed_123" / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(seed_metrics["label_free"])

            audit = run_visual_vae_audit(dataset, quality.report, root / "audit")

            self.assertTrue(audit.passed)
            assert audit.approval_manifest is not None
            approved = load_approved_visual_vae(audit.approval_manifest)
            self.assertEqual(approved.config, model_config)
            self.assertFalse(approved.training)
            audit_report = json.loads(audit.report.read_text(encoding="utf-8"))
            self.assertIn("grayscale_audit", audit_report["metrics"]["probes"])
            self.assertIn("audit", audit_report["metrics"]["label_free"])
            with self.assertRaisesRegex(RuntimeError, "daha önce tüketti"):
                run_visual_vae_audit(dataset, quality.report, root / "second-audit")

            manifest = json.loads(audit.approval_manifest.read_text(encoding="utf-8"))
            manifest["approved_checkpoint_sha256"] = "0" * 64
            audit.approval_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_approved_visual_vae(audit.approval_manifest)

    def test_loader_refuses_supervised_encoder_manifest(self) -> None:
        """Etiketli zincirin manifest'i etiketsiz yükleyiciden geçemez."""

        supervised = Path("runs/vision_audit_v5_20260914/approval_manifest.json")
        if not supervised.is_file():
            self.skipTest("Etiketli approval manifest bu ortamda yok")
        with self.assertRaisesRegex(ValueError, "approval manifest"):
            load_approved_visual_vae(supervised)

    def test_gate_requirement_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "içermiyor"):
            _require_gated_probes({"checks": {"1": {"validation": {"psnr": True}}}})
        failed = {
            "checks": {"1": {"validation": {gate: True for gate in REQUIRED_GATES}}}
        }
        failed["checks"]["1"]["validation"]["target_auroc"] = False
        with self.assertRaisesRegex(ValueError, "target_auroc"):
            _require_gated_probes(failed)
        # ADR-022 Revizyon 1: eski (sabit eşikli) zincir eşikten bağımsız
        # kapıları içermediği için fail-closed reddedilir.
        legacy = {
            "checks": {
                "1": {
                    "validation": {
                        "active_latent_units": True,
                        "free_space_mae": True,
                        "target_recall": True,
                        "obstacle_map_iou": True,
                    }
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "target_auroc"):
            _require_gated_probes(legacy)
        passing = {
            "checks": {"1": {"validation": {gate: True for gate in REQUIRED_GATES}}}
        }
        self.assertIsNone(_require_gated_probes(passing))

    def test_quality_cli_returns_nonzero_and_withholds_approval_on_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            _write_synthetic_dataset(dataset)
            impossible = root / "impossible.toml"
            base = Path("src/ebabil/configs/training/visual_vae.toml").read_text(
                encoding="utf-8"
            )
            impossible.write_text(
                base.replace(
                    "minimum_active_latent_units = 32",
                    "minimum_active_latent_units = 100000",
                ).replace(
                    "probe_logistic_iterations = 200", "probe_logistic_iterations = 5"
                ),
                encoding="utf-8",
            )
            self.assertIn(
                "minimum_active_latent_units = 100000", impossible.read_text()
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_vae_quality.py",
                    str(dataset),
                    str(root / "quality"),
                    "--config",
                    str(impossible),
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

            self.assertNotEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("quality_passed=False", completed.stdout)
            self.assertNotIn("approved_checkpoint=", completed.stdout)


def _slice(labels: ProbeLabels, window: slice) -> ProbeLabels:
    return ProbeLabels(
        target_visible=labels.target_visible[window],
        target_center_xy=labels.target_center_xy[window],
        free_space=labels.free_space[window],
        obstacle_map=labels.obstacle_map[window],
    )


if __name__ == "__main__":
    unittest.main()
