"""Segmentation'dan kutu etiketi türetme, YOLO formatı ve dataset dışa aktarımı.

ADR-023 §2: dedektör verisi elle etiketlenmez. Privileged NPZ'deki ham
`segmentation (H, W, 2)` (geom kimliği, nesne türü), `obstacle_mask` ve
`target_mask` alanlarından her engel geom'u için görünür piksel kutusu, hedef
için tek kutu türetilir. Görsel çeldiriciler bu maskelerde olmadığı için
etiketlenmez (negatif örnek). Kenarı `min_side_px`'ten veya görünür piksel
sayısı `min_area_px`'ten küçük kutular düşer ve sayılır.

Kutu koordinatları piksel cinsinden yarı-açık aralıktır: `x_min` en soldaki
piksel sütunu, `x_max` en sağdaki sütun + 1; genişlik `x_max - x_min`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ebabil.perception.dataset import (
    SPLITS,
    DatasetRecord,
    Split,
    load_manifest,
    vision_dataset_provenance,
)

CLASS_NAMES: tuple[str, ...] = ("obstacle", "target")
OBSTACLE_CLASS_ID = 0
TARGET_CLASS_ID = 1
YOLO_DATASET_SCHEMA_VERSION = "yolo-dataset-v1"
#: Kutu çizimi için sınıf renkleri (RGB).
CLASS_COLORS: tuple[tuple[int, int, int], ...] = ((255, 64, 32), (32, 255, 96))

UInt8Array = npt.NDArray[np.uint8]
BoolArray = npt.NDArray[np.bool_]
Int32Array = npt.NDArray[np.int32]


@dataclass(frozen=True)
class Detection:
    """Tek kutu: sınıf, piksel koordinatları (yarı-açık) ve güven."""

    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.class_id < 0 or self.class_id >= len(CLASS_NAMES):
            raise ValueError(f"Geçersiz sınıf kimliği: {self.class_id}")
        if not (self.x_max > self.x_min and self.y_max > self.y_min):
            raise ValueError("Kutu boyutları pozitif olmalı")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Güven [0, 1] aralığında olmalı")

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max))


@dataclass(frozen=True)
class BoxLabels:
    detections: tuple[Detection, ...]
    dropped_small: int


def _mask_box(mask: BoolArray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def boxes_from_masks(
    segmentation: Int32Array,
    obstacle_mask: BoolArray,
    target_mask: BoolArray,
    *,
    min_side_px: int = 3,
    min_area_px: int = 16,
) -> BoxLabels:
    """Engel geom'u başına ve hedef için görünür piksel kutuları üret.

    Engel örnekleri `segmentation[..., 0]` geom kimliğiyle ayrılır; hedef
    geom'ları tek nesnedir. Örtülme yalnız görünür kısmı kutular. Döndürülen
    liste engeller (kimlik sırasıyla) ve varsa hedef biçimindedir.
    """

    if segmentation.ndim != 3 or segmentation.shape[2] != 2:
        raise ValueError("segmentation (H, W, 2) olmalı")
    height, width = segmentation.shape[0], segmentation.shape[1]
    if obstacle_mask.shape != (height, width) or target_mask.shape != (height, width):
        raise ValueError("Maskeler segmentation ile aynı (H, W) olmalı")
    if min_side_px <= 0 or min_area_px <= 0:
        raise ValueError("min_side_px ve min_area_px pozitif olmalı")
    geom_ids = segmentation[..., 0]
    detections: list[Detection] = []
    dropped = 0
    for geom_id in np.unique(geom_ids[obstacle_mask]):
        instance = obstacle_mask & (geom_ids == geom_id)
        box = _mask_box(instance)
        if box is None:
            continue
        x_min, y_min, x_max, y_max = box
        if (
            int(instance.sum()) < min_area_px
            or x_max - x_min < min_side_px
            or y_max - y_min < min_side_px
        ):
            dropped += 1
            continue
        detections.append(Detection(OBSTACLE_CLASS_ID, x_min, y_min, x_max, y_max))
    target_box = _mask_box(target_mask)
    if target_box is not None:
        x_min, y_min, x_max, y_max = target_box
        if (
            int(target_mask.sum()) < min_area_px
            or x_max - x_min < min_side_px
            or y_max - y_min < min_side_px
        ):
            dropped += 1
        else:
            detections.append(Detection(TARGET_CLASS_ID, x_min, y_min, x_max, y_max))
    return BoxLabels(detections=tuple(detections), dropped_small=dropped)


def to_yolo_lines(
    detections: Iterable[Detection], width: int, height: int
) -> tuple[str, ...]:
    """`sınıf cx cy w h` satırları, `[0, 1]` normalize (Ultralytics formatı)."""

    if width <= 0 or height <= 0:
        raise ValueError("Görüntü boyutları pozitif olmalı")
    lines: list[str] = []
    for detection in detections:
        cx, cy = detection.center
        lines.append(
            f"{detection.class_id} {cx / width:.6f} {cy / height:.6f} "
            f"{detection.width / width:.6f} {detection.height / height:.6f}"
        )
    return tuple(lines)


def parse_yolo_lines(
    lines: Iterable[str], width: int, height: int
) -> tuple[Detection, ...]:
    """YOLO satırlarını piksel kutularına çevir (inceleme ve testler için)."""

    detections: list[Detection] = []
    for raw in lines:
        parts = raw.split()
        if not parts:
            continue
        if len(parts) not in (5, 6):
            raise ValueError(f"YOLO satırı 5 veya 6 alan içermeli: {raw!r}")
        class_id = int(parts[0])
        cx, cy, w, h = (float(value) for value in parts[1:5])
        confidence = float(parts[5]) if len(parts) == 6 else 1.0
        detections.append(
            Detection(
                class_id,
                (cx - 0.5 * w) * width,
                (cy - 0.5 * h) * height,
                (cx + 0.5 * w) * width,
                (cy + 0.5 * h) * height,
                confidence,
            )
        )
    return tuple(detections)


def draw_boxes(
    rgb: UInt8Array, detections: Iterable[Detection], *, thickness: int = 1
) -> UInt8Array:
    """Kutuları sınıf rengiyle çiz; girdi değişmez, kopya döner (NumPy, cv2 yok)."""

    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB uint8[H, W, 3] olmalı")
    if thickness <= 0:
        raise ValueError("thickness pozitif olmalı")
    canvas = rgb.copy()
    height, width = canvas.shape[0], canvas.shape[1]
    for detection in detections:
        color = np.array(CLASS_COLORS[detection.class_id], dtype=np.uint8)
        x0 = int(np.clip(np.floor(detection.x_min), 0, width - 1))
        y0 = int(np.clip(np.floor(detection.y_min), 0, height - 1))
        x1 = int(np.clip(np.ceil(detection.x_max), 1, width))
        y1 = int(np.clip(np.ceil(detection.y_max), 1, height))
        t = thickness
        canvas[y0 : min(y0 + t, height), x0:x1] = color
        canvas[max(y1 - t, 0) : y1, x0:x1] = color
        canvas[y0:y1, x0 : min(x0 + t, width)] = color
        canvas[y0:y1, max(x1 - t, 0) : x1] = color
    return canvas


@dataclass(frozen=True)
class FrameLabelRecord:
    """Dışa aktarılan her kare için kalite ölçümüne yarayan yan bilgiler."""

    sample_id: str
    split: str
    visual_domain: str
    scene_seed: int
    image_file: str
    label_file: str
    obstacle_boxes: int
    target_box: bool
    target_visible_privileged: bool
    target_range_m: float | None
    dropped_small: int
    obstacle_visible_instances: int


@dataclass(frozen=True)
class ExportSummary:
    schema_version: str
    source_dataset: str
    source_provenance: dict[str, str]
    class_names: tuple[str, ...]
    min_side_px: int
    min_area_px: int
    image_width: int
    image_height: int
    split_counts: dict[str, int]
    box_counts: dict[str, dict[str, int]]
    dropped_small_total: int
    frames_with_target: int
    frames_without_obstacle: int
    frames_file: str = "frames.jsonl"
    dataset_yaml: str = "dataset.yaml"
    audit_yaml: str = "dataset_audit.yaml"
    notes: tuple[str, ...] = field(default_factory=tuple)


def _write_png(path: Path, rgb: UInt8Array) -> None:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - kurulum hatası
        raise RuntimeError(
            "PNG yazımı için pillow gerekir: .venv/bin/pip install -e '.[detector]'"
        ) from error
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG", compress_level=6)


def _target_range_m(
    depth_m: npt.NDArray[np.float64], target_mask: BoolArray
) -> float | None:
    """Hedef piksellerinin medyan derinliği (kameradan eğik menzil); yoksa None."""

    if not target_mask.any():
        return None
    values = depth_m[target_mask]
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return None
    return float(np.median(values))


def label_record(
    rgb: UInt8Array,
    depth_m: npt.NDArray[np.float64],
    segmentation: Int32Array,
    obstacle_mask: BoolArray,
    target_mask: BoolArray,
    *,
    min_side_px: int,
    min_area_px: int,
) -> tuple[BoxLabels, float | None, int]:
    """Tek kare için kutular, hedef menzili ve görünür engel örneği sayısı."""

    if rgb.shape[:2] != segmentation.shape[:2]:
        raise ValueError("RGB ve segmentation boyutları eşleşmiyor")
    labels = boxes_from_masks(
        segmentation,
        obstacle_mask,
        target_mask,
        min_side_px=min_side_px,
        min_area_px=min_area_px,
    )
    visible_instances = int(np.unique(segmentation[..., 0][obstacle_mask]).size)
    return labels, _target_range_m(depth_m, target_mask), visible_instances


def _dataset_yaml(root: Path, *, val_split: str, test_split: str | None) -> str:
    lines = [
        f"# {YOLO_DATASET_SCHEMA_VERSION}; kaynak: ebabil vision-dataset-v4 segmentation etiketleri",
        f"path: {root}",
        "train: images/train",
        f"val: images/{val_split}",
    ]
    if test_split is not None:
        lines.append(f"test: images/{test_split}")
    lines.append("names:")
    for index, name in enumerate(CLASS_NAMES):
        lines.append(f"  {index}: {name}")
    return "\n".join(lines) + "\n"


def export_yolo_dataset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    min_side_px: int = 3,
    min_area_px: int = 16,
    splits: Sequence[Split] = SPLITS,
) -> ExportSummary:
    """NPZ dataset'inden YOLO dizin yapısı üret; sızıntı denetimi kaynak dataset'te.

    Çıktı: `images/<split>/<sample_id>.png`, `labels/<split>/<sample_id>.txt`,
    `frames.jsonl`, `dataset.yaml` (train / validation / development_ood) ve
    `dataset_audit.yaml` (val = audit, tek kullanımlık). Var olan çıktı
    dizinine yazılmaz.
    """

    source = Path(dataset_dir).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Çıktı dizini zaten var: {root}")
    for split in splits:
        if split not in SPLITS:
            raise ValueError(f"Geçersiz split: {split}")
    provenance = vision_dataset_provenance(source)
    records = load_manifest(source / "manifest.jsonl")
    root.mkdir(parents=True)
    split_counts: dict[str, int] = {split: 0 for split in splits}
    box_counts: dict[str, dict[str, int]] = {
        split: {name: 0 for name in CLASS_NAMES} for split in splits
    }
    dropped_total = 0
    frames_with_target = 0
    frames_without_obstacle = 0
    width = height = 0
    with (root / "frames.jsonl").open("w", encoding="utf-8") as frames_file:
        for record in records:
            if record.split not in splits:
                continue
            frame = _export_record(
                source, root, record, min_side_px=min_side_px, min_area_px=min_area_px
            )
            frames_file.write(json.dumps(asdict(frame[0]), sort_keys=True) + "\n")
            width, height = frame[1]
            split_counts[record.split] += 1
            box_counts[record.split][CLASS_NAMES[OBSTACLE_CLASS_ID]] += frame[
                0
            ].obstacle_boxes
            box_counts[record.split][CLASS_NAMES[TARGET_CLASS_ID]] += int(
                frame[0].target_box
            )
            dropped_total += frame[0].dropped_small
            frames_with_target += int(frame[0].target_box)
            frames_without_obstacle += int(frame[0].obstacle_boxes == 0)
    (root / "dataset.yaml").write_text(
        _dataset_yaml(root, val_split="validation", test_split="development_ood"),
        encoding="utf-8",
    )
    (root / "dataset_audit.yaml").write_text(
        _dataset_yaml(root, val_split="audit", test_split=None), encoding="utf-8"
    )
    summary = ExportSummary(
        schema_version=YOLO_DATASET_SCHEMA_VERSION,
        source_dataset=str(source),
        source_provenance=asdict(provenance),
        class_names=CLASS_NAMES,
        min_side_px=min_side_px,
        min_area_px=min_area_px,
        image_width=width,
        image_height=height,
        split_counts=split_counts,
        box_counts=box_counts,
        dropped_small_total=dropped_total,
        frames_with_target=frames_with_target,
        frames_without_obstacle=frames_without_obstacle,
        notes=(
            "Görsel çeldiriciler etiketlenmez (negatif örnek).",
            "audit bölümü tek kullanımlıktır; dataset_audit.yaml yalnız audit runner'ı okur.",
        ),
    )
    payload = asdict(summary)
    payload["frames_sha256"] = hashlib.sha256(
        (root / "frames.jsonl").read_bytes()
    ).hexdigest()
    (root / "export_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _export_record(
    source: Path,
    root: Path,
    record: DatasetRecord,
    *,
    min_side_px: int,
    min_area_px: int,
) -> tuple[FrameLabelRecord, tuple[int, int]]:
    actor = np.load(source / record.actor_file, allow_pickle=False)
    privileged = np.load(source / record.privileged_file, allow_pickle=False)
    rgb = np.asarray(actor["rgb"], dtype=np.uint8)
    labels, target_range, visible_instances = label_record(
        rgb,
        np.asarray(privileged["depth_m"], dtype=np.float64),
        np.asarray(privileged["segmentation"], dtype=np.int32),
        np.asarray(privileged["obstacle_mask"], dtype=bool),
        np.asarray(privileged["target_mask"], dtype=bool),
        min_side_px=min_side_px,
        min_area_px=min_area_px,
    )
    image_rel = Path("images") / record.split / f"{record.sample_id}.png"
    label_rel = Path("labels") / record.split / f"{record.sample_id}.txt"
    (root / image_rel).parent.mkdir(parents=True, exist_ok=True)
    (root / label_rel).parent.mkdir(parents=True, exist_ok=True)
    _write_png(root / image_rel, rgb)
    height, width = rgb.shape[0], rgb.shape[1]
    lines = to_yolo_lines(labels.detections, width, height)
    (root / label_rel).write_text(
        ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
    )
    frame = FrameLabelRecord(
        sample_id=record.sample_id,
        split=record.split,
        visual_domain=record.visual_domain,
        scene_seed=record.scene_seed,
        image_file=str(image_rel),
        label_file=str(label_rel),
        obstacle_boxes=sum(
            1 for d in labels.detections if d.class_id == OBSTACLE_CLASS_ID
        ),
        target_box=any(d.class_id == TARGET_CLASS_ID for d in labels.detections),
        target_visible_privileged=bool(privileged["target_visible"]),
        target_range_m=target_range,
        dropped_small=labels.dropped_small,
        obstacle_visible_instances=visible_instances,
    )
    return frame, (width, height)


def load_frame_records(yolo_dir: str | Path) -> tuple[FrameLabelRecord, ...]:
    path = Path(yolo_dir).expanduser().resolve() / "frames.jsonl"
    records: list[FrameLabelRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(FrameLabelRecord(**json.loads(line)))
    return tuple(records)
