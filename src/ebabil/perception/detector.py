"""YOLO dedektör: config, eğitim ve çıkarım sarmalayıcısı (ADR-023 §3).

Ultralytics yalnız bu modülün fonksiyonları içinde, ortam değişkenleri
ayarlandıktan sonra içe aktarılır: `YOLO_CONFIG_DIR` proje içindeki
`runs/ultralytics_config`'e, `YOLO_OFFLINE=1` ağ erişimini kapatır ve
telemetri senkronu kapalı tutulur. Böylece çekirdek paket ultralytics
kurulu olmadan da içe aktarılabilir ve testler geçer.

Renk sözleşmesi: ortam ve dataset RGB üretir; ultralytics NumPy girdisini
OpenCV gibi BGR varsayar. `YoloDetector.detect` dönüşümü kendisi yapar.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ebabil.perception.detections import CLASS_NAMES, Detection

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DETECTOR_CONFIG = (
    PROJECT_ROOT / "src" / "ebabil" / "configs" / "training" / "detector.toml"
)
DETECTOR_SCHEMA_VERSION = "yolo-detector-v1"
TRAINING_METADATA_SCHEMA = "yolo-detector-training-v1"
YOLO_DATASET_SCHEMA_VERSION = "yolo-dataset-v1"

UInt8Array = npt.NDArray[np.uint8]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DetectorConfig:
    schema_version: str = DETECTOR_SCHEMA_VERSION
    model: str = "yolo11n"
    pretrained_weights: str = "runs/weights/yolo11n.pt"
    pretrained_weights_sha256: str = ""
    imgsz: int = 320
    epochs: int = 60
    batch: int = 32
    workers: int = 4
    cache: str = "ram"
    device: str = "mps"
    fliplr: float = 0.5
    mosaic: float = 1.0
    close_mosaic: int = 10
    inference_iou: float = 0.7
    max_det: int = 32

    def validate(self) -> None:
        if self.schema_version != DETECTOR_SCHEMA_VERSION:
            raise ValueError(f"Dedektör config sürümü {DETECTOR_SCHEMA_VERSION} olmalı")
        if self.model != "yolo11n":
            raise ValueError("ADR-023: dedektör modeli yolo11n olmalı")
        if self.imgsz <= 0 or self.imgsz % 32 != 0:
            raise ValueError("imgsz pozitif ve 32'nin katı olmalı")
        if self.epochs <= 0 or self.batch <= 0 or self.workers < 0:
            raise ValueError("epochs/batch pozitif, workers negatif olmayan olmalı")
        if self.cache not in ("ram", "none"):
            raise ValueError("cache 'ram' veya 'none' olmalı")
        if not 0.0 <= self.fliplr <= 1.0 or not 0.0 <= self.mosaic <= 1.0:
            raise ValueError("fliplr ve mosaic [0, 1] aralığında olmalı")
        if not 0.0 < self.inference_iou < 1.0 or self.max_det <= 0:
            raise ValueError("inference_iou (0, 1), max_det pozitif olmalı")
        if len(self.pretrained_weights_sha256) != 64:
            raise ValueError("pretrained_weights_sha256 64 karakterlik SHA-256 olmalı")

    def pretrained_path(self) -> Path:
        path = Path(self.pretrained_weights)
        return path if path.is_absolute() else PROJECT_ROOT / path


def detector_config_from_raw(raw: dict[str, Any]) -> DetectorConfig:
    config = DetectorConfig(
        schema_version=str(raw.get("schema_version", DETECTOR_SCHEMA_VERSION)),
        model=str(raw.get("model", "yolo11n")),
        pretrained_weights=str(raw["pretrained_weights"]),
        pretrained_weights_sha256=str(raw["pretrained_weights_sha256"]),
        imgsz=int(raw.get("imgsz", 320)),
        epochs=int(raw.get("epochs", 60)),
        batch=int(raw.get("batch", 32)),
        workers=int(raw.get("workers", 4)),
        cache=str(raw.get("cache", "ram")),
        device=str(raw.get("device", "mps")),
        fliplr=float(raw.get("fliplr", 0.5)),
        mosaic=float(raw.get("mosaic", 1.0)),
        close_mosaic=int(raw.get("close_mosaic", 10)),
        inference_iou=float(raw.get("inference_iou", 0.7)),
        max_det=int(raw.get("max_det", 32)),
    )
    config.validate()
    return config


def load_detector_config(path: str | Path = DEFAULT_DETECTOR_CONFIG) -> DetectorConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    section = raw.get("detector")
    if not isinstance(section, dict):
        raise ValueError("Dedektör config'inde [detector] bölümü eksik")
    return detector_config_from_raw(section)


def configure_ultralytics_environment() -> None:
    """Ultralytics içe aktarılmadan önce çevrimdışı ve proje-yerel ayarları kur."""

    config_dir = PROJECT_ROOT / "runs" / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ["YOLO_OFFLINE"] = "1"
    from ultralytics import settings

    if settings.get("sync", True):
        settings.update({"sync": False})


def _ultralytics_version() -> str:
    import ultralytics

    return str(ultralytics.__version__)


def yolo_dataset_provenance(yolo_dir: str | Path) -> dict[str, Any]:
    """YOLO dışa aktarımının kimliği: özet SHA-256'sı ve kaynak dataset kimliği."""

    root = Path(yolo_dir).expanduser().resolve()
    summary_path = root / "export_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != YOLO_DATASET_SCHEMA_VERSION:
        raise ValueError(f"YOLO dataset sürümü {YOLO_DATASET_SCHEMA_VERSION} olmalı")
    if tuple(summary.get("class_names", ())) != CLASS_NAMES:
        raise ValueError("YOLO dataset sınıfları obstacle/target olmalı")
    frames_sha256 = file_sha256(root / "frames.jsonl")
    if frames_sha256 != summary.get("frames_sha256"):
        raise ValueError("frames.jsonl SHA-256 export özetiyle eşleşmiyor")
    return {
        "schema_version": YOLO_DATASET_SCHEMA_VERSION,
        "export_summary_sha256": file_sha256(summary_path),
        "frames_sha256": frames_sha256,
        "source_provenance": summary["source_provenance"],
        "min_side_px": summary["min_side_px"],
        "min_area_px": summary["min_area_px"],
    }


@dataclass(frozen=True)
class DetectorTrainingArtifacts:
    run_dir: Path
    weights: Path
    weights_sha256: str
    metadata: Path
    seed: int


def train_detector(
    yolo_dir: str | Path,
    output_dir: str | Path,
    config: DetectorConfig,
    seed: int,
    *,
    epochs_override: int | None = None,
) -> DetectorTrainingArtifacts:
    """YOLO11n'i önceden eğitilmiş COCO ağırlığından tek seed ile ince ayarla.

    Onay adayı `last.pt`'dir: `best.pt` validation bölümünde seçildiği için
    kalite kapısını aynı bölümde ölçmek seçim sızıntısı olurdu.
    """

    config.validate()
    provenance = yolo_dataset_provenance(yolo_dir)
    pretrained = config.pretrained_path()
    if file_sha256(pretrained) != config.pretrained_weights_sha256:
        raise ValueError("Önceden eğitilmiş ağırlık SHA-256 eşleşmiyor")
    output_path = Path(output_dir).expanduser().resolve()
    run_dir = output_path / f"seed_{seed}"
    if run_dir.exists():
        raise FileExistsError(f"Seed dizini zaten var: {run_dir}")
    output_path.mkdir(parents=True, exist_ok=True)
    configure_ultralytics_environment()
    from ultralytics import YOLO

    epochs = config.epochs if epochs_override is None else int(epochs_override)
    started = time.perf_counter()
    model = YOLO(str(pretrained))
    model.train(
        data=str(Path(yolo_dir).expanduser().resolve() / "dataset.yaml"),
        epochs=epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        workers=config.workers,
        cache="ram" if config.cache == "ram" else False,
        device=config.device,
        seed=int(seed),
        deterministic=True,
        amp=False,
        plots=False,
        verbose=False,
        fliplr=config.fliplr,
        flipud=0.0,
        mosaic=config.mosaic,
        close_mosaic=config.close_mosaic,
        project=str(output_path),
        name=f"seed_{seed}",
        exist_ok=False,
    )
    elapsed = time.perf_counter() - started
    weights = run_dir / "weights" / "last.pt"
    if not weights.is_file():
        raise RuntimeError(f"Eğitim last.pt üretmedi: {weights}")
    weights_sha256 = file_sha256(weights)
    metadata = {
        "schema_version": TRAINING_METADATA_SCHEMA,
        "seed": int(seed),
        "epochs": epochs,
        "config": asdict(config),
        "dataset": provenance,
        "pretrained_weights_sha256": config.pretrained_weights_sha256,
        "weights": weights.relative_to(run_dir).as_posix(),
        "weights_sha256": weights_sha256,
        "ultralytics_version": _ultralytics_version(),
        "elapsed_seconds": elapsed,
    }
    metadata_path = run_dir / "training_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return DetectorTrainingArtifacts(
        run_dir=run_dir,
        weights=weights,
        weights_sha256=weights_sha256,
        metadata=metadata_path,
        seed=int(seed),
    )


def _to_numpy(value: Any) -> npt.NDArray[Any]:
    """Ultralytics tensor veya ndarray çıktısını NumPy'a çevir."""

    if hasattr(value, "cpu"):
        return np.asarray(value.cpu().numpy())
    return np.asarray(value)


class YoloDetector:
    """Eğitilmiş ağırlıkla tek kare çıkarım; RGB girer, `Detection` listesi çıkar."""

    def __init__(
        self,
        weights: str | Path,
        *,
        imgsz: int = 320,
        device: str = "mps",
        iou: float = 0.7,
        max_det: int = 32,
    ) -> None:
        configure_ultralytics_environment()
        from ultralytics import YOLO

        self.weights = Path(weights).expanduser().resolve()
        self.imgsz = int(imgsz)
        self.device = str(device)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self._model = YOLO(str(self.weights))
        names = tuple(
            str(self._model.names[index]) for index in sorted(self._model.names)
        )
        if names != CLASS_NAMES:
            raise ValueError(f"Ağırlığın sınıfları {CLASS_NAMES} değil: {names}")

    def detect(self, rgb: UInt8Array, *, conf: float) -> tuple[Detection, ...]:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("RGB uint8[H, W, 3] olmalı")
        if not 0.0 <= conf <= 1.0:
            raise ValueError("conf [0, 1] aralığında olmalı")
        height, width = rgb.shape[0], rgb.shape[1]
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        results: Any = self._model.predict(
            bgr,
            imgsz=self.imgsz,
            conf=conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        boxes: Any = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return ()
        xyxy = _to_numpy(boxes.xyxy).astype(np.float64)
        classes = _to_numpy(boxes.cls).astype(np.int64)
        scores = _to_numpy(boxes.conf).astype(np.float64)
        detections: list[Detection] = []
        for (x0, y0, x1, y1), class_id, score in zip(
            xyxy, classes, scores, strict=True
        ):
            x0, x1 = float(np.clip(x0, 0.0, width)), float(np.clip(x1, 0.0, width))
            y0, y1 = float(np.clip(y0, 0.0, height)), float(np.clip(y1, 0.0, height))
            if x1 <= x0 or y1 <= y0:
                continue
            detections.append(
                Detection(
                    int(class_id), x0, y0, x1, y1, float(np.clip(score, 0.0, 1.0))
                )
            )
        return tuple(detections)


def measure_latency(
    detector: YoloDetector,
    frames: Sequence[UInt8Array],
    *,
    conf: float,
    warmup: int = 5,
) -> dict[str, float]:
    """Tek kare gecikmesi (ms): ısınmadan sonra medyan, p90, ortalama."""

    if not frames:
        raise ValueError("Gecikme ölçümü için en az bir kare gerekli")
    for index in range(warmup):
        detector.detect(frames[index % len(frames)], conf=conf)
    samples: list[float] = []
    for frame in frames:
        started = time.perf_counter()
        detector.detect(frame, conf=conf)
        samples.append((time.perf_counter() - started) * 1000.0)
    values = np.asarray(samples, dtype=np.float64)
    return {
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "mean_ms": float(values.mean()),
        "frames": float(values.size),
    }
