"""Rastgele seçilen karelerde YOLO etiketlerini çizerek görsel kontrol dosyaları üret.

Etiketler diskteki `labels/<split>/<id>.txt` dosyasından yeniden okunur; böylece
yazılan dosyalar doğrulanır, bellekteki kutular değil.

Kullanım:
    .venv/bin/python scripts/review_yolo_labels.py runs/yolo_dataset_d1_20260926 --count 8 --output-dir runs/yolo_dataset_d1_20260926/review
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ebabil.perception.dataset import SPLITS
from ebabil.perception.detections import (
    CLASS_NAMES,
    draw_boxes,
    load_frame_records,
    parse_yolo_lines,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yolo_dir", type=Path)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.yolo_dir.resolve()
    records = load_frame_records(root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    per_split = max(1, args.count // len(SPLITS))
    chosen = []
    for split in SPLITS:
        pool = [record for record in records if record.split == split]
        picks = rng.choice(len(pool), size=min(per_split, len(pool)), replace=False)
        chosen.extend(pool[int(index)] for index in picks)
    tiles = []
    for record in chosen:
        rgb = np.asarray(
            Image.open(root / record.image_file).convert("RGB"), dtype=np.uint8
        )
        lines = (root / record.label_file).read_text(encoding="utf-8").splitlines()
        detections = parse_yolo_lines(lines, rgb.shape[1], rgb.shape[0])
        drawn = draw_boxes(rgb, detections, thickness=2)
        drawn = np.repeat(np.repeat(drawn, args.scale, axis=0), args.scale, axis=1)
        name = f"review_{record.split}_{record.sample_id}.png"
        Image.fromarray(drawn).save(args.output_dir / name)
        tiles.append(drawn)
        counts = Counter(CLASS_NAMES[d.class_id] for d in detections)
        print(
            f"{name}: obstacle={counts.get('obstacle', 0)} target={counts.get('target', 0)} "
            f"visible_instances={record.obstacle_visible_instances} dropped_small={record.dropped_small} "
            f"target_range_m={record.target_range_m if record.target_range_m is None else round(record.target_range_m, 2)}"
        )
    if tiles:
        columns = 2
        rows = (len(tiles) + columns - 1) // columns
        tile_h, tile_w = tiles[0].shape[0], tiles[0].shape[1]
        sheet = np.zeros((rows * tile_h, columns * tile_w, 3), dtype=np.uint8)
        for index, tile in enumerate(tiles):
            r, c = divmod(index, columns)
            sheet[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
        Image.fromarray(sheet).save(args.output_dir / "review_sheet.png")
        print(f"sheet={args.output_dir / 'review_sheet.png'}")


if __name__ == "__main__":
    main()
