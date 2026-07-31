"""Evaluate a paired optical-student + light-detector checkpoint on val or test."""

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

# Allow direct execution with ``python src/evaluate_slm_light_dataset.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.SLM.config_slm import ConfigSLM as Config
from models.SLM.dataset_slm import SLMFeatureDataset, slm_collate_fn
from models.SLM.optical_layers import OpticalStudent
from models.SLM.utils_slm import load_student_detector_checkpoint
from models.geometry import bbox_iou_xywh
from models.monitoring import (
    _render_confusion_matrix_image,
    _render_normalized_confusion_matrix_image,
    compute_detection_confusion_matrix,
)
from models.runtime import get_dataloader_kwargs
from models.yolov8.detection_protocol import decode_detections
from models.yolov8.feature_adapter import prepare_slm_detector_feature
from models.yolov8.head_v8 import build_detector_head
from models.yolov8.metrics_anchor_v8 import compute_average_precision, compute_pr_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an optical student + light detector on a labelled dataset split.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Paired detector_best.pth with student and detector states.")
    parser.add_argument("--data", type=Path, default=Path("data/military/data.yaml"), help="Dataset YAML file.")
    parser.add_argument("--split", choices=("val", "test"), default="test", help="Dataset split to evaluate.")
    parser.add_argument("--output", type=Path, default=Path("output/slm_light_dataset_eval"), help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch-size override.")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu (default: student config).")
    parser.add_argument("--conf-threshold", type=float, default=None, help="Confidence for the count confusion matrix.")
    parser.add_argument("--nms-threshold", type=float, default=None, help="NMS IoU threshold for decoding.")
    parser.add_argument("--iou-threshold", type=float, default=None, help="IoU threshold for mAP and confusion matching.")
    return parser.parse_args()


def load_checkpoint_metadata(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_student(config, checkpoint):
    """Match a multi-head checkpoint automatically; optical settings remain config-owned."""
    checkpoint_heads = int(checkpoint.get("num_heads", 1)) if isinstance(checkpoint, dict) else 1
    if checkpoint_heads > 1:
        from models.SLM.multi_head_slm import MultiHeadOpticalStudent

        config.SLM_MULTI_HEAD_ENABLED = True
        config.SLM_MULTI_HEAD_NUM_HEADS = checkpoint_heads
        return MultiHeadOpticalStudent(config)
    config.SLM_MULTI_HEAD_ENABLED = False
    return OpticalStudent(config)


def evaluate_student(config, student, detector, dataloader, device):
    """Compute the same class-wise AP50 matching protocol used by SLM training."""
    student.eval()
    detector.eval()
    all_detections, all_targets = [], []
    storage = {class_id: [] for class_id in range(config.NUM_CLASSES)}
    gt_counts = {class_id: 0 for class_id in range(config.NUM_CLASSES)}
    amp_enabled = bool(getattr(config, "ENABLE_AMP", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(getattr(config, "AMP_DTYPE", "float16")).lower() in {"bf16", "bfloat16"} else torch.float16
    amp_context = torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled) if device.type == "cuda" else nullcontext()

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating SLM + light"):
            gray = batch["gray_tensor"].to(device, non_blocking=config.PIN_MEMORY)
            if config.ENABLE_CHANNELS_LAST and device.type == "cuda":
                gray = gray.contiguous(memory_format=torch.channels_last)
            with amp_context:
                feature = student(gray)
                predictions = detector(prepare_slm_detector_feature(config, feature))
            detections = decode_detections(
                config,
                predictions,
                conf_thresh=config.METRIC_CONF_THRESH,
                nms_thresh=config.METRIC_NMS_THRESH,
                max_det=config.METRIC_MAX_DET,
            )
            for targets, sample_detections in zip(batch["targets"], detections):
                targets_np = targets.cpu().numpy()
                detections_np = np.asarray(sample_detections, dtype=np.float32).reshape(-1, 6)
                all_targets.append(targets_np)
                all_detections.append(detections_np)
                gt_by_class = {}
                for target in targets_np:
                    if target.shape[0] < 5 or target[3] <= 0 or target[4] <= 0:
                        continue
                    class_id = int(target[0])
                    gt_by_class.setdefault(class_id, []).append(target[1:5] * np.asarray(
                        [config.RESOLUTION[1], config.RESOLUTION[0], config.RESOLUTION[1], config.RESOLUTION[0]], dtype=np.float32
                    ))
                    gt_counts[class_id] += 1
                for class_id in range(config.NUM_CLASSES):
                    class_detections = sorted(
                        (det for det in detections_np if int(det[5]) == class_id), key=lambda det: float(det[4]), reverse=True
                    )
                    matched_gt = set()
                    for detection in class_detections:
                        best_iou, best_index = 0.0, -1
                        for index, gt_box in enumerate(gt_by_class.get(class_id, ())):
                            if index in matched_gt:
                                continue
                            iou = float(bbox_iou_xywh(
                                torch.from_numpy(detection[:4]).float().unsqueeze(0),
                                torch.from_numpy(np.asarray(gt_box)).float().unsqueeze(0),
                            ).item())
                            if iou > best_iou:
                                best_iou, best_index = iou, index
                        is_tp = best_iou >= config.METRIC_IOU_THRESHOLD
                        storage[class_id].append((float(detection[4]), float(is_tp)))
                        if is_tp:
                            matched_gt.add(best_index)

    per_class, ap_values = {}, []
    for class_id in range(config.NUM_CLASSES):
        ap50 = compute_average_precision(storage[class_id], gt_counts[class_id])
        if ap50 is not None:
            ap_values.append(ap50)
        per_class[class_id] = {
            "class_name": config.CLASS_NAMES[class_id],
            "ap50": float(ap50 or 0.0),
            "gt_count": int(gt_counts[class_id]),
            **compute_pr_summary(storage[class_id], gt_counts[class_id]),
        }
    return all_detections, all_targets, {
        "map50": float(np.mean(ap_values)) if ap_values else 0.0,
        "per_class": per_class,
    }


def save_outputs(output_dir, confusion, metrics, checkpoint, restore_info, args, dataset_size):
    class_names = Config.CLASS_NAMES
    labels = [class_names[index] for index in range(Config.NUM_CLASSES)] + ["background"]
    foreground = confusion[:Config.NUM_CLASSES, :Config.NUM_CLASSES]
    percentages = np.divide(foreground * 100.0, foreground.sum(axis=1, keepdims=True), out=np.zeros_like(foreground), where=foreground.sum(axis=1, keepdims=True) > 0)
    Image.fromarray(_render_confusion_matrix_image(
        confusion, class_names,
        title=f"{args.split} SLM + light Confusion Matrix (IoU >= {args.iou_threshold:.2f}, Conf >= {args.conf_threshold:.2f})",
    )).save(output_dir / "confusion_matrix_counts_with_background.png")
    Image.fromarray(_render_normalized_confusion_matrix_image(
        confusion, class_names,
        title=f"{args.split} SLM + light Foreground Normalized Matrix (IoU >= {args.iou_threshold:.2f}, Conf >= {args.conf_threshold:.2f})",
    )).save(output_dir / "confusion_matrix_normalized_foreground_percent.png")
    np.savetxt(output_dir / "confusion_matrix_counts_with_background.csv", confusion.astype(np.int64), delimiter=",", fmt="%d")
    np.savetxt(output_dir / "confusion_matrix_normalized_foreground_percent.csv", percentages, delimiter=",", fmt="%.2f")
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_recorded_val_map50": checkpoint.get("val_map50"), "checkpoint_restore": restore_info,
        "dataset_yaml": str(args.data.resolve()), "split": args.split, "images": dataset_size,
        "resolution_hw": list(Config.RESOLUTION), "metric_iou_threshold": args.iou_threshold,
        "matrix_confidence_threshold": args.conf_threshold, "metric_decode_confidence_threshold": Config.METRIC_CONF_THRESH,
        "metric_decode_nms_threshold": Config.METRIC_NMS_THRESH, "labels_counts_matrix": labels,
        "labels_normalized_matrix": labels[:-1], "metrics": metrics,
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
        if (value := getattr(args, name)) is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")

    Config.YAML_PATH = str(args.data.resolve())
    Config.initialize()
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.nms_threshold is not None:
        Config.METRIC_NMS_THRESH = args.nms_threshold
    if args.iou_threshold is not None:
        Config.METRIC_IOU_THRESHOLD = args.iou_threshold
    args.conf_threshold = float(Config.CONF_THRESH if args.conf_threshold is None else args.conf_threshold)
    args.iou_threshold = float(Config.METRIC_IOU_THRESHOLD)
    device = torch.device(args.device or Config.DEVICE)
    checkpoint = load_checkpoint_metadata(args.checkpoint, device)
    if not isinstance(checkpoint, dict) or "student_state_dict" not in checkpoint or "detector_state_dict" not in checkpoint:
        raise KeyError("--checkpoint must contain paired student_state_dict and detector_state_dict.")

    student = build_student(Config, checkpoint).to(device)
    detector = build_detector_head(Config, in_channels=1).to(device)
    restore_info = load_student_detector_checkpoint(student, detector, str(args.checkpoint), device)
    dataset = SLMFeatureDataset(Config, split=args.split)
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset split {args.split!r} is empty.")
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, collate_fn=slm_collate_fn, **get_dataloader_kwargs(Config))
    detections, targets, metrics = evaluate_student(Config, student, detector, dataloader, device)
    confusion, class_tp, class_fp, class_fn = compute_detection_confusion_matrix(
        detections, targets, Config.NUM_CLASSES, iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold, image_size=Config.RESOLUTION,
    )
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_outputs(output_dir, confusion, metrics, checkpoint, restore_info, args, len(dataset))
    print(f"Evaluated {len(dataset)} {args.split} images on {device}.")
    print(f"mAP50: {metrics['map50']:.4f}")
    print(f"Count matrix (with background): {output_dir / 'confusion_matrix_counts_with_background.png'}")
    print(f"Normalized foreground matrix: {output_dir / 'confusion_matrix_normalized_foreground_percent.png'}")
    print("per-class TP:", class_tp.astype(int))
    print("per-class FP:", class_fp.astype(int))
    print("per-class FN:", class_fn.astype(int))


if __name__ == "__main__":
    main()
