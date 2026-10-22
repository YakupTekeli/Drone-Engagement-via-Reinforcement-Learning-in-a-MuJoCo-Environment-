"""ADR-023 §2 kutu etiketi türetme, YOLO formatı ve dışa aktarım testleri."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ebabil.perception.dataset import SPLITS
from ebabil.perception.detections import (
    CLASS_NAMES,
    OBSTACLE_CLASS_ID,
    TARGET_CLASS_ID,
    Detection,
    boxes_from_masks,
    draw_boxes,
    export_yolo_dataset,
    label_record,
    load_frame_records,
    parse_yolo_lines,
    to_yolo_lines,
)

GEOM = 5  # mujoco.mjtObj.mjOBJ_GEOM


def _synthetic_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """20 x 30: engel A (id 5), engel B (id 7), küçük engel (id 9), hedef (id 3) A ile örtülü."""

    height, width = 20, 30
    geom_ids = np.full((height, width), 1, dtype=np.int32)  # zemin
    geom_ids[6:10, 6:13] = 3  # hedef
    geom_ids[2:10, 3:9] = 5  # engel A hedefin sol kısmını örter
    geom_ids[12:20, 20:30] = 7  # engel B
    geom_ids[0:2, 28:30] = 9  # küçük engel (4 px)
    segmentation = np.stack([geom_ids, np.full_like(geom_ids, GEOM)], axis=-1)
    obstacle_mask = np.isin(geom_ids, (5, 7, 9))
    target_mask = geom_ids == 3
    depth = np.full((height, width), 9.0, dtype=np.float64)
    depth[target_mask] = 4.0
    return segmentation, obstacle_mask, target_mask, depth


class BoxDerivationTests(unittest.TestCase):
    def test_instances_target_occlusion_and_small_filter(self) -> None:
        segmentation, obstacle_mask, target_mask, _ = _synthetic_scene()
        labels = boxes_from_masks(segmentation, obstacle_mask, target_mask)

        boxes = {
            (d.class_id, d.x_min, d.y_min, d.x_max, d.y_max) for d in labels.detections
        }
        self.assertEqual(
            boxes,
            {
                (OBSTACLE_CLASS_ID, 3, 2, 9, 10),
                (OBSTACLE_CLASS_ID, 20, 12, 30, 20),
                (TARGET_CLASS_ID, 9, 6, 13, 10),  # yalnız görünür kısım
            },
        )
        self.assertEqual(labels.dropped_small, 1)

    def test_target_below_minimum_is_dropped_and_counted(self) -> None:
        segmentation, obstacle_mask, target_mask, _ = _synthetic_scene()
        labels = boxes_from_masks(
            segmentation, obstacle_mask, target_mask, min_area_px=17
        )
        self.assertFalse(any(d.class_id == TARGET_CLASS_ID for d in labels.detections))
        self.assertEqual(labels.dropped_small, 2)

    def test_shape_and_parameter_validation(self) -> None:
        segmentation, obstacle_mask, target_mask, _ = _synthetic_scene()
        with self.assertRaisesRegex(ValueError, "segmentation"):
            boxes_from_masks(segmentation[..., 0], obstacle_mask, target_mask)
        with self.assertRaisesRegex(ValueError, "Maskeler"):
            boxes_from_masks(segmentation, obstacle_mask[:10], target_mask)
        with self.assertRaisesRegex(ValueError, "pozitif"):
            boxes_from_masks(segmentation, obstacle_mask, target_mask, min_side_px=0)
        with self.assertRaisesRegex(ValueError, "sınıf"):
            Detection(2, 0, 0, 1, 1)
        with self.assertRaisesRegex(ValueError, "pozitif"):
            Detection(0, 5, 0, 5, 1)

    def test_label_record_reports_range_and_visible_instances(self) -> None:
        segmentation, obstacle_mask, target_mask, depth = _synthetic_scene()
        rgb = np.zeros((20, 30, 3), dtype=np.uint8)
        labels, target_range, visible = label_record(
            rgb,
            depth,
            segmentation,
            obstacle_mask,
            target_mask,
            min_side_px=3,
            min_area_px=16,
        )
        self.assertEqual(len(labels.detections), 3)
        self.assertAlmostEqual(target_range or 0.0, 4.0)
        self.assertEqual(visible, 3)  # küçük engel görünür örnek olarak sayılır


class YoloFormatTests(unittest.TestCase):
    def test_lines_round_trip_within_sub_pixel(self) -> None:
        segmentation, obstacle_mask, target_mask, _ = _synthetic_scene()
        detections = boxes_from_masks(
            segmentation, obstacle_mask, target_mask
        ).detections
        lines = to_yolo_lines(detections, 30, 20)
        self.assertEqual(len(lines), 3)
        self.assertTrue(all(len(line.split()) == 5 for line in lines))
        parsed = parse_yolo_lines(lines, 30, 20)
        for original, back in zip(detections, parsed, strict=True):
            self.assertEqual(original.class_id, back.class_id)
            for attribute in ("x_min", "y_min", "x_max", "y_max"):
                self.assertAlmostEqual(
                    getattr(original, attribute), getattr(back, attribute), places=3
                )
        with self.assertRaisesRegex(ValueError, "5 veya 6"):
            parse_yolo_lines(["0 0.5 0.5"], 30, 20)

    def test_draw_boxes_marks_only_border(self) -> None:
        rgb = np.zeros((20, 30, 3), dtype=np.uint8)
        drawn = draw_boxes(rgb, [Detection(TARGET_CLASS_ID, 5, 5, 15, 12)], thickness=1)
        self.assertTrue(np.all(rgb == 0))
        self.assertTrue(np.all(drawn[5, 5:15] == (32, 255, 96)))
        self.assertTrue(np.all(drawn[11, 5:15] == (32, 255, 96)))
        self.assertTrue(np.all(drawn[6:11, 6:14] == 0))
        self.assertTrue(np.all(drawn[5:12, 5] == (32, 255, 96)))


class ExportTests(unittest.TestCase):
    def _write_source(self, root: Path) -> None:
        segmentation, obstacle_mask, target_mask, depth = _synthetic_scene()
        rgb = np.full((20, 30, 3), 90, dtype=np.uint8)
        (root / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": "vision-dataset-v4",
                    "config": {"schema_version": "vision-dataset-v4"},
                }
            )
        )
        (root / "statistics.json").write_text(
            json.dumps({split: {} for split in SPLITS})
        )
        lines = []
        for index, split in enumerate(("train", "validation")):
            sample_id = f"{split}-s0000000001-p00{index}"
            actor = Path("actor") / split / f"{sample_id}.npz"
            privileged = Path("privileged") / split / f"{sample_id}.npz"
            (root / actor).parent.mkdir(parents=True, exist_ok=True)
            (root / privileged).parent.mkdir(parents=True, exist_ok=True)
            np.savez(root / actor, sample_id=sample_id, rgb=rgb, timestamp_s=0.0)
            np.savez(
                root / privileged,
                sample_id=sample_id,
                depth_m=depth,
                segmentation=segmentation,
                target_mask=target_mask,
                obstacle_mask=obstacle_mask,
                target_visible=True,
                timestamp_s=0.0,
            )
            lines.append(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "split": split,
                        "scene_seed": 1,
                        "sample_index": index,
                        "scene_mode": "random_obstacles",
                        "actor_file": str(actor),
                        "privileged_file": str(privileged),
                        "visual_domain": "test",
                    }
                )
            )
        (root / "manifest.jsonl").write_text("\n".join(lines) + "\n")

    def test_export_layout_summary_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src"
            source.mkdir()
            self._write_source(source)
            output = Path(tmp) / "yolo"
            summary = export_yolo_dataset(source, output)

            self.assertEqual(
                summary.split_counts,
                {s: (1 if s in ("train", "validation") else 0) for s in SPLITS},
            )
            self.assertEqual(summary.box_counts["train"], {"obstacle": 2, "target": 1})
            self.assertEqual(summary.dropped_small_total, 2)
            self.assertEqual((summary.image_width, summary.image_height), (30, 20))
            label = (output / "labels" / "train").glob("*.txt")
            self.assertEqual(len(next(label).read_text().splitlines()), 3)
            self.assertTrue((output / "images" / "train").glob("*.png"))
            yaml = (output / "dataset.yaml").read_text()
            self.assertIn("val: images/validation", yaml)
            self.assertIn("test: images/development_ood", yaml)
            self.assertIn("1: target", yaml)
            self.assertIn(
                "val: images/audit", (output / "dataset_audit.yaml").read_text()
            )
            frames = load_frame_records(output)
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0].obstacle_boxes, 2)
            self.assertTrue(frames[0].target_box)
            self.assertAlmostEqual(frames[0].target_range_m or 0.0, 4.0)
            payload = json.loads((output / "export_summary.json").read_text())
            self.assertEqual(payload["schema_version"], "yolo-dataset-v1")
            self.assertEqual(tuple(payload["class_names"]), CLASS_NAMES)
            self.assertIn("frames_sha256", payload)
            with self.assertRaises(FileExistsError):
                export_yolo_dataset(source, output)


if __name__ == "__main__":
    unittest.main()
