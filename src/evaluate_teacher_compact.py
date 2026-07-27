"""Evaluate a saved teacher + CompactOpticalDetector checkpoint on validation data."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from models.dataset import YOLODataset, identity_collate
from models.runtime import get_dataloader_kwargs
from models.teacher import build_teacher
from models.teacher_train_compact import Config, _evaluate_teacher_compact
from models.teacher_train_compact import _collect_compact_val_detections
from models.compact_detect.losses import CenterDetectionLoss
from models.monitoring import compute_detection_confusion_matrix, _render_confusion_matrix_image
from models.yolov8.head_v8 import build_detector_head


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a teacher + compact detector checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output/Tv1_compactv2/teacher_detector_best.pth"),
        help="Checkpoint containing teacher_state_dict and detector_state_dict.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Optional validation batch-size override.")
    parser.add_argument(
        "--conf-threshold", type=float, default=None,
        help="Operating-point confidence for precision/recall and confusion matrix (default: config value).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/Tv1_compactv2/confusion_matrix_eval.png"),
        help="Path for the operating-point confusion matrix PNG.",
    )
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
            raise KeyError(f"Checkpoint is missing '{key}': {path}")
        module.load_state_dict(checkpoint[key], strict=True)
    return checkpoint


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    Config.initialize()
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be at least 1.")
        Config.BATCH_SIZE = args.batch_size
    if args.conf_threshold is not None:
        if not 0.0 <= args.conf_threshold <= 1.0:
            raise ValueError("--conf-threshold must be between 0 and 1.")
        Config.COMPACT_CONF_THRESH = args.conf_threshold
    device = torch.device(Config.DEVICE)

    teacher = build_teacher(Config).to(device)
    detector = build_detector_head(Config, in_channels=1).to(device)
    checkpoint = load_checkpoint(checkpoint_path, teacher, detector, device)

    val_dataset = YOLODataset(Config, split="val")
    if not val_dataset:
        raise RuntimeError("Validation dataset is empty.")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        collate_fn=identity_collate,
        **get_dataloader_kwargs(Config),
    )
    losses, metrics = _evaluate_teacher_compact(
        teacher, detector, val_loader, CenterDetectionLoss(Config), device
    )
    detections, targets = _collect_compact_val_detections(
        teacher, detector, val_loader, device
    )
    conf_threshold = float(Config.COMPACT_CONF_THRESH)
    iou_threshold = float(Config.COMPACT_METRIC_IOU_THRESHOLD)
    confusion, class_tp, class_fp, class_fn = compute_detection_confusion_matrix(
        detections,
        targets,
        Config.NUM_CLASSES,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
        image_size=Config.RESOLUTION,
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = _render_confusion_matrix_image(
        confusion,
        Config.CLASS_NAMES,
        title=f"Detection Confusion Matrix (IoU>={iou_threshold}, Conf>={conf_threshold})",
    )
    Image.fromarray(image).save(output_path)

    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"checkpoint recorded mAP50: {checkpoint.get('val_map50', 'unknown')}")
    print(f"validation images: {len(val_dataset)}")
    print("losses:", {name: round(value, 6) for name, value in losses.items()})
    print(f"mAP candidate confidence: {Config.COMPACT_METRIC_CONF_THRESH}")
    print(f"operating-point confidence: {conf_threshold}")
    print("metrics:", {name: round(value, 6) for name, value in metrics.items()})
    labels = [Config.CLASS_NAMES.get(i, f"class_{i}") for i in range(Config.NUM_CLASSES)] + ["background"]
    print("confusion labels:", labels)
    print("confusion matrix (rows=ground truth, cols=predicted):")
    print(confusion.astype(int))
    print("per-class TP:", class_tp.astype(int))
    print("per-class FP:", class_fp.astype(int))
    print("per-class FN:", class_fn.astype(int))
    print(f"confusion matrix saved to: {output_path}")


if __name__ == "__main__":
    main()
