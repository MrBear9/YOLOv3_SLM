"""Evaluate a Teacher V2 + light detector checkpoint on a labelled dataset split."""

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow direct execution with ``python src/evaluate_teacher_v2_dataset.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dataset import YOLODataset, identity_collate
from models.monitoring import (
    _render_confusion_matrix_image,
    _render_normalized_confusion_matrix_image,
    compute_detection_confusion_matrix,
)
from models.runtime import get_dataloader_kwargs, prepare_batch
from models.teacher import build_teacher
from models.yolov8.config_v8 import ConfigYOLOv8Anchor as Config
from models.yolov8.detection_protocol import build_detection_criterion, decode_detections
from models.yolov8.head_v8 import TeacherWithDetector, build_detector_head
from models.yolov8.metrics_anchor_v8 import evaluate_model_anchor_v8


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Teacher V2 + light detector on a labelled dataset split.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Teacher + detector checkpoint.")
    parser.add_argument("--data", type=Path, default=Path("data/military/data.yaml"), help="Dataset YAML file.")
    parser.add_argument("--split", choices=("val", "test"), default="test", help="Dataset split to evaluate.")
    parser.add_argument("--output", type=Path, default=Path("output/teacher_v2_dataset_eval"), help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch-size override.")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu (default: training config).")
    parser.add_argument("--conf-threshold", type=float, default=None, help="Confidence for the confusion matrices.")
    parser.add_argument("--nms-threshold", type=float, default=None, help="NMS IoU threshold for all metrics.")
    parser.add_argument("--iou-threshold", type=float, default=None, help="IoU threshold for mAP and confusion matching.")
    return parser.parse_args()


def load_checkpoint(path, teacher, detector, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Expected a checkpoint dictionary, got {type(checkpoint).__name__}.")
    for key, module in (("teacher_state_dict", teacher), ("detector_state_dict", detector)):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint is missing {key!r}: {path}")
        module.load_state_dict(checkpoint[key], strict=True)
    return checkpoint


def collect_detections(config, model, dataloader, device):
    """Collect detections and labels once for the two operating-point matrices."""
    model.eval()
    detections_all, targets_all = [], []
    amp_enabled = bool(getattr(config, "ENABLE_AMP", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(getattr(config, "AMP_DTYPE", "float16")).lower() in {"bf16", "bfloat16"} else torch.float16
    amp_context = torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled) if device.type == "cuda" else nullcontext()
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Collecting detections"):
            images, targets = prepare_batch(config, batch, device)
            with amp_context:
                _, predictions = model(images, return_feature=True)
            detections = decode_detections(
                config,
                predictions,
                conf_thresh=config.METRIC_CONF_THRESH,
                nms_thresh=config.METRIC_NMS_THRESH,
                max_det=config.METRIC_MAX_DET,
            )
            detections_all.extend(np.asarray(sample, dtype=np.float32).reshape(-1, 6) for sample in detections)
            targets_all.extend(target.cpu().numpy() for target in targets)
    return detections_all, targets_all


def write_results(output_dir, confusion, metrics, checkpoint, args, dataset_size):
    class_names = Config.CLASS_NAMES
    labels = [class_names[index] for index in range(Config.NUM_CLASSES)] + ["background"]
    normalized = confusion[:Config.NUM_CLASSES, :Config.NUM_CLASSES]
    normalized = np.divide(
        normalized * 100.0,
        normalized.sum(axis=1, keepdims=True),
        out=np.zeros_like(normalized),
        where=normalized.sum(axis=1, keepdims=True) > 0,
    )
    Image.fromarray(_render_confusion_matrix_image(
        confusion,
        class_names,
        title=f"{args.split} Detection Confusion Matrix (IoU >= {args.iou_threshold:.2f}, Conf >= {args.conf_threshold:.2f})",
    )).save(output_dir / "confusion_matrix_counts_with_background.png")
    Image.fromarray(_render_normalized_confusion_matrix_image(
        confusion,
        class_names,
        title=f"{args.split} Foreground Normalized Confusion Matrix (IoU >= {args.iou_threshold:.2f}, Conf >= {args.conf_threshold:.2f})",
    )).save(output_dir / "confusion_matrix_normalized_foreground_percent.png")
    np.savetxt(output_dir / "confusion_matrix_counts_with_background.csv", confusion.astype(np.int64), delimiter=",", fmt="%d")
    np.savetxt(output_dir / "confusion_matrix_normalized_foreground_percent.csv", normalized, delimiter=",", fmt="%.2f")
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_recorded_val_map50": checkpoint.get("val_map50"),
        "dataset_yaml": str(args.data.resolve()),
        "split": args.split,
        "images": dataset_size,
        "resolution_hw": list(Config.RESOLUTION),
        "metric_iou_threshold": args.iou_threshold,
        "matrix_confidence_threshold": args.conf_threshold,
        "metric_decode_confidence_threshold": Config.METRIC_CONF_THRESH,
        "metric_decode_nms_threshold": Config.METRIC_NMS_THRESH,
        "labels_counts_matrix": labels,
        "labels_normalized_matrix": labels[:-1],
        "metrics": metrics,
    }
    (output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    for name in ("conf_threshold", "nms_threshold", "iou_threshold"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")

    Config.YAML_PATH = str(args.data.resolve())
    Config.initialize()
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.nms_threshold is not None:
        Config.METRIC_NMS_THRESH = args.nms_threshold
    if args.iou_threshold is not None:
        Config.METRIC_IOU_THRESHOLD = args.iou_threshold
    args.conf_threshold = float(args.conf_threshold if args.conf_threshold is not None else Config.CONF_THRESH)
    args.iou_threshold = float(Config.METRIC_IOU_THRESHOLD)
    device = torch.device(args.device or Config.DEVICE)

    dataset = YOLODataset(Config, yaml_path=Config.YAML_PATH, split=args.split)
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset split {args.split!r} is empty.")
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, collate_fn=identity_collate, **get_dataloader_kwargs(Config))
    teacher = build_teacher(Config).to(device)
    detector = build_detector_head(Config, in_channels=1).to(device)
    checkpoint = load_checkpoint(args.checkpoint, teacher, detector, device)
    model = TeacherWithDetector(Config, teacher=teacher, detector=detector).to(device).eval()
    criterion = build_detection_criterion(Config)
    _, metrics = evaluate_model_anchor_v8(Config, model, dataloader, criterion, device)
    detections, targets = collect_detections(Config, model, dataloader, device)
    confusion, class_tp, class_fp, class_fn = compute_detection_confusion_matrix(
        detections, targets, Config.NUM_CLASSES,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
        image_size=Config.RESOLUTION,
    )

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_results(output_dir, confusion, metrics, checkpoint, args, len(dataset))
    print(f"Evaluated {len(dataset)} {args.split} images on {device}.")
    print(f"mAP50: {metrics['map50']:.4f}")
    print(f"Count matrix (with background): {output_dir / 'confusion_matrix_counts_with_background.png'}")
    print(f"Normalized foreground matrix: {output_dir / 'confusion_matrix_normalized_foreground_percent.png'}")
    print("per-class TP:", class_tp.astype(int))
    print("per-class FP:", class_fp.astype(int))
    print("per-class FN:", class_fn.astype(int))


if __name__ == "__main__":
    main()
