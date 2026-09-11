"""Run a trained Light detector directly on processed CCD feature images.

The input images are expected to be registered optical-intensity planes, such
as the 640x640 PNG files produced by tools/capture_ccd_video_frames.py.  This
script deliberately bypasses the Teacher/SLM simulation and does not apply
scene-image letterboxing or per-image contrast normalisation.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.class_display import class_name_for_id
from models.yolov8.config_v8 import ConfigYOLOv8Anchor as Config
from models.yolov8.detection_protocol import decode_detections
from models.yolov8.feature_adapter import prepare_detector_feature
from models.yolov8.head_v8 import build_detector_head


DEFAULT_CCD_DIR = Path("output/Tv2_dmd640_scratch/hardware_export_100/ccd")
DEFAULT_CHECKPOINT = Path("output/Tv2_dmd640_scratch/teacher_detector_best.pth")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect processed CCD optical feature images with the trained Light head only."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_CCD_DIR,
        help="Directory containing processed CCD feature images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint containing detector_state_dict.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/military/data.yaml"),
        help="Dataset YAML used to recover class names and class count.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Detection output directory (default: a ccd_detection sibling of --images).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda, cuda:0, or cpu (default: project configuration).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size (default: 4).",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="Detection confidence threshold (default: project inference configuration).",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=None,
        help="NMS IoU threshold (default: project inference configuration).",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=None,
        help="Maximum detections retained per image (default: project inference configuration).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N images, useful for a hardware test.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    for name in ("conf_threshold", "nms_threshold"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if args.max_det is not None and args.max_det < 1:
        raise ValueError("--max-det must be at least 1.")


def natural_image_key(path: Path) -> tuple[int, int | str, str]:
    """Sort numbered hardware IDs numerically while retaining other filenames."""
    if path.stem.isdigit():
        return 0, int(path.stem), path.name.casefold()
    return 1, path.stem.casefold(), path.name.casefold()


def find_images(images_dir: Path, limit: int | None) -> list[Path]:
    paths = sorted(
        (
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_image_key,
    )
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise RuntimeError(f"No supported CCD images found in: {images_dir}")
    return paths


def load_detector_checkpoint(path: Path, detector: torch.nn.Module, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Expected a checkpoint dictionary, got {type(checkpoint).__name__}.")
    if "detector_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing 'detector_state_dict': {path}")
    checkpoint_head = str(checkpoint.get("head_type", "light")).strip().lower()
    if checkpoint_head not in {"light", "yolo_light"}:
        raise ValueError(
            f"Checkpoint head_type is {checkpoint_head!r}; this script requires the Light detector head."
        )
    detector.load_state_dict(checkpoint["detector_state_dict"], strict=True)
    return checkpoint


def load_ccd_feature(path: Path, resolution_hw: tuple[int, int]) -> tuple[Image.Image, torch.Tensor, list[int]]:
    target_h, target_w = resolution_hw
    with Image.open(path) as source:
        original_size_wh = [int(source.width), int(source.height)]
        image = source.convert("L")
        if image.size != (target_w, target_h):
            image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
        image = image.copy()
    values = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(values).unsqueeze(0)
    return image, tensor, original_size_wh


def detection_records(detections: np.ndarray) -> list[dict]:
    records = []
    for cx, cy, width, height, confidence, class_id_value in detections:
        class_id = int(class_id_value)
        x1 = float(cx - width / 2.0)
        y1 = float(cy - height / 2.0)
        x2 = float(cx + width / 2.0)
        y2 = float(cy + height / 2.0)
        records.append(
            {
                "class_id": class_id,
                "class_name": str(Config.CLASS_NAMES.get(class_id, f"class_{class_id}")),
                "display_name": class_name_for_id(Config.CLASS_NAMES, class_id),
                "confidence": float(confidence),
                "xywh_pixels": [float(cx), float(cy), float(width), float(height)],
                "xyxy_pixels": [x1, y1, x2, y2],
            }
        )
    return records


def draw_detections(image: Image.Image, records: list[dict], sample_id: str) -> Image.Image:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    image_w, image_h = canvas.size
    for record in records:
        x1, y1, x2, y2 = record["xyxy_pixels"]
        x1 = min(max(x1, 0.0), image_w - 1.0)
        y1 = min(max(y1, 0.0), image_h - 1.0)
        x2 = min(max(x2, 0.0), image_w - 1.0)
        y2 = min(max(y2, 0.0), image_h - 1.0)
        label = f"{record['display_name']} {record['confidence']:.2f}"
        label_box = draw.textbbox((x1, y1), label, stroke_width=1)
        label_height = label_box[3] - label_box[1] + 4
        text_y = max(0.0, y1 - label_height)
        text_box = draw.textbbox((x1 + 2, text_y + 1), label, stroke_width=1)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.rectangle(
            (x1, text_y, min(float(image_w - 1), text_box[2] + 2), min(float(image_h - 1), text_box[3] + 2)),
            fill="black",
        )
        draw.text((x1 + 2, text_y + 1), label, fill="red", stroke_width=1, stroke_fill="black")

    status = f"{sample_id} | detections: {len(records)}"
    status_box = draw.textbbox((0, 0), status, stroke_width=1)
    draw.rectangle((0, 0, status_box[2] + 8, status_box[3] + 7), fill="black")
    draw.text((4, 3), status, fill="white", stroke_width=1, stroke_fill="black")
    return canvas


def main() -> None:
    args = parse_args()
    validate_args(args)

    images_dir = resolve_project_path(args.images)
    checkpoint_path = resolve_project_path(args.checkpoint)
    data_path = resolve_project_path(args.data)
    if not images_dir.is_dir():
        raise NotADirectoryError(f"CCD image directory not found: {images_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    output_dir = (
        resolve_project_path(args.output)
        if args.output is not None
        else images_dir.parent / f"{images_dir.name}_detection"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    Config.YAML_PATH = str(data_path)
    Config.initialize()
    if str(Config.DETECTOR_HEAD_TYPE).strip().lower() not in {"light", "yolo_light"}:
        raise ValueError("The current Config.DETECTOR_HEAD_TYPE must be 'light'.")

    device = torch.device(args.device or Config.DEVICE)
    detector = build_detector_head(Config, in_channels=1).to(device).eval()
    checkpoint = load_detector_checkpoint(checkpoint_path, detector, device)

    image_paths = find_images(images_dir, args.limit)
    conf_threshold = Config.CONF_THRESH if args.conf_threshold is None else args.conf_threshold
    nms_threshold = Config.NMS_THRESH if args.nms_threshold is None else args.nms_threshold
    max_det = Config.MAX_DET if args.max_det is None else args.max_det
    amp_enabled = bool(getattr(Config, "ENABLE_AMP", True)) and device.type == "cuda"
    amp_dtype_name = str(getattr(Config, "AMP_DTYPE", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16

    samples: list[dict] = []
    for start in tqdm(range(0, len(image_paths), args.batch_size), desc="Detecting CCD features"):
        batch_paths = image_paths[start : start + args.batch_size]
        batch_images: list[Image.Image] = []
        batch_tensors: list[torch.Tensor] = []
        original_sizes: list[list[int]] = []
        for path in batch_paths:
            image, tensor, original_size = load_ccd_feature(path, tuple(Config.RESOLUTION))
            batch_images.append(image)
            batch_tensors.append(tensor)
            original_sizes.append(original_size)

        features = torch.stack(batch_tensors, dim=0).to(device)
        amp_context = (
            torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), amp_context:
            predictions = detector(prepare_detector_feature(Config, features))
            batch_detections = decode_detections(
                Config,
                predictions,
                conf_thresh=conf_threshold,
                nms_thresh=nms_threshold,
                max_det=max_det,
            )

        for path, image, original_size, detections in zip(
            batch_paths, batch_images, original_sizes, batch_detections
        ):
            detections_array = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
            records = detection_records(detections_array)
            output_path = output_dir / f"{path.stem}.png"
            draw_detections(image, records, path.stem).save(output_path)
            samples.append(
                {
                    "id": path.stem,
                    "input": str(path),
                    "output": str(output_path),
                    "original_size_wh": original_size,
                    "detector_size_hw": [int(Config.RESOLUTION[0]), int(Config.RESOLUTION[1])],
                    "resized_for_detector": original_size
                    != [int(Config.RESOLUTION[1]), int(Config.RESOLUTION[0])],
                    "detection_count": len(records),
                    "detections": records,
                }
            )

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_map50": checkpoint.get("val_map50"),
        "head_type": checkpoint.get("head_type", Config.DETECTOR_HEAD_TYPE),
        "data_yaml": str(data_path),
        "class_names": {str(key): str(value) for key, value in Config.CLASS_NAMES.items()},
        "input_directory": str(images_dir),
        "output_directory": str(output_dir),
        "resolution_hw": [int(Config.RESOLUTION[0]), int(Config.RESOLUTION[1])],
        "preprocessing": "grayscale, optional bilinear resize, divide by 255; no letterbox or per-image normalization",
        "detector_invert_feature": bool(getattr(Config, "DETECTOR_INVERT_FEATURE", False)),
        "confidence_threshold": float(conf_threshold),
        "nms_iou_threshold": float(nms_threshold),
        "max_detections_per_image": int(max_det),
        "image_count": len(samples),
        "total_detection_count": sum(sample["detection_count"] for sample in samples),
        "samples": samples,
    }
    report_path = output_dir / "detection_results.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Detected {len(samples)} CCD images on {device}.")
    print(f"Detection images: {output_dir}")
    print(f"Detection records: {report_path}")


if __name__ == "__main__":
    main()
