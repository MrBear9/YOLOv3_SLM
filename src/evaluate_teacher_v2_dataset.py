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

from models.dataset import YOLODataset, identity_collate, letterbox_content_bounds
from models.geometry import bbox_iou_matrix_xywh
from models.detection_metrics_offline import add_multi_iou_metrics
from models.coco_offline import evaluate_coco, require_coco
from models.monitoring import (
    _render_confusion_matrix_image,
    _render_normalized_confusion_matrix_image,
    compute_detection_confusion_matrix,
)
from models.runtime import get_dataloader_kwargs, prepare_batch
from models.review_candidates import write_review_lists as write_shared_review_lists
from models.teacher import build_teacher, configure_teacher_checkpoint
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
    parser.add_argument("--iou-threshold", type=float, default=None, help="IoU for confusion/review matching only; AP50 and AP50:95 use fixed thresholds.")
    parser.add_argument("--skip-coco", action="store_true", help="Skip official COCO evaluation (enabled by default).")
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
    """Collect detections, labels, and their stable source image paths."""
    model.eval()
    detections_all, targets_all, source_paths = [], [], []
    dataset_paths = tuple(Path(path).resolve() for path in getattr(dataloader.dataset, "files", ()))
    sample_offset = 0
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
            batch_size = len(detections)
            batch_paths = dataset_paths[sample_offset:sample_offset + batch_size]
            if len(batch_paths) != batch_size:
                raise RuntimeError("Detection collection lost alignment with dataset.files.")
            detections_all.extend(np.asarray(sample, dtype=np.float32).reshape(-1, 6) for sample in detections)
            targets_all.extend(target.cpu().numpy() for target in targets)
            source_paths.extend(batch_paths)
            sample_offset += batch_size
    if sample_offset != len(dataset_paths):
        raise RuntimeError("Detection collection did not visit every dataset image exactly once.")
    return detections_all, targets_all, source_paths


def analyse_map_sample(config, detections, targets, device):
    """Mirror the evaluator's class-wise IoU matching for one image.

    The returned TP/FP/FN counts use ``METRIC_CONF_THRESH`` and
    ``METRIC_IOU_THRESHOLD``, i.e. the same threshold regime used for mAP50.
    """
    ground_truths = []
    gt_by_class = {}
    image_h, image_w = config.RESOLUTION
    for gt in targets:
        if len(gt) < 5 or gt[3] <= 0 or gt[4] <= 0:
            continue
        cls_id = int(gt[0])
        gt_box = [
            float(gt[1] * image_w), float(gt[2] * image_h),
            float(gt[3] * image_w), float(gt[4] * image_h),
        ]
        gt_index = len(ground_truths)
        ground_truths.append({
            "class_id": cls_id,
            "class_name": config.CLASS_NAMES.get(cls_id, str(cls_id)),
            "xywh_eval": gt_box,
            "status": "fn",
        })
        gt_by_class.setdefault(cls_id, []).append(gt_index)

    predictions = []
    det_by_class = {}
    for det in np.asarray(detections, dtype=np.float32).reshape(-1, 6):
        cls_id = int(det[5])
        prediction_index = len(predictions)
        predictions.append({
            "class_id": cls_id,
            "class_name": config.CLASS_NAMES.get(cls_id, str(cls_id)),
            "confidence": float(det[4]),
            "xywh_eval": [float(value) for value in det[:4]],
            "status": "fp",
        })
        det_by_class.setdefault(cls_id, []).append(prediction_index)

    total_tp = total_fp = total_fn = 0
    per_class = []
    for cls_id in sorted(set(gt_by_class) | set(det_by_class)):
        gt_indices = gt_by_class.get(cls_id, [])
        detection_indices = sorted(
            det_by_class.get(cls_id, []),
            key=lambda index: predictions[index]["confidence"],
            reverse=True,
        )
        gt_boxes_list = [ground_truths[index]["xywh_eval"] for index in gt_indices]
        dets = [predictions[index] for index in detection_indices]
        tp = fp = fn = 0
        if not gt_boxes_list:
            fp = len(dets)
        elif not dets:
            fn = len(gt_boxes_list)
        else:
            gt_boxes = torch.tensor(gt_boxes_list, dtype=torch.float32, device=device)
            det_boxes = torch.tensor([det["xywh_eval"] for det in dets], dtype=torch.float32, device=device)
            iou_matrix = bbox_iou_matrix_xywh(det_boxes, gt_boxes)
            matched_gt = set()
            for det_idx in range(len(dets)):
                ious = iou_matrix[det_idx].clone()
                for gt_idx in matched_gt:
                    ious[gt_idx] = -1.0
                best_iou, best_gt_idx = ious.max(dim=0)
                if float(best_iou.item()) >= config.METRIC_IOU_THRESHOLD:
                    tp += 1
                    matched_gt_index = int(best_gt_idx.item())
                    matched_gt.add(matched_gt_index)
                    prediction = predictions[detection_indices[det_idx]]
                    ground_truth = ground_truths[gt_indices[matched_gt_index]]
                    prediction["status"] = "tp"
                    prediction["matched_gt_index"] = gt_indices[matched_gt_index]
                    prediction["matched_iou"] = float(best_iou.item())
                    ground_truth["status"] = "tp"
                    ground_truth["matched_prediction_index"] = detection_indices[det_idx]
                    ground_truth["matched_iou"] = float(best_iou.item())
                else:
                    fp += 1
            fn = len(gt_boxes_list) - len(matched_gt)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_class.append({
            "class_id": cls_id,
            "class_name": config.CLASS_NAMES.get(cls_id, str(cls_id)),
            "ground_truth": len(gt_boxes_list),
            "detections": len(dets),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

    return {
        "ground_truth": sum(len(boxes) for boxes in gt_by_class.values()),
        "detections": sum(len(dets) for dets in det_by_class.values()),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_class": per_class,
        "ground_truth_boxes": ground_truths,
        "prediction_boxes": predictions,
    }


def eval_xywh_to_source(xywh_eval, source_size, resolution):
    """Map an evaluator-canvas box back onto its original image pixels."""
    source_w, source_h = source_size
    left, top, right, _ = letterbox_content_bounds(source_size, resolution)
    scale = (right - left) / source_w
    x, y, width, height = xywh_eval
    x = min(max((x - left) / scale, 0.0), float(source_w))
    y = min(max((y - top) / scale, 0.0), float(source_h))
    width = min(max(width / scale, 0.0), float(source_w))
    height = min(max(height / scale, 0.0), float(source_h))
    return [round(value, 3) for value in (x, y, width, height)]


def write_review_lists(output_dir, source_paths, detections, targets, config, device):
    """Write direct-to-review image lists plus explainable per-image JSONL."""
    records = []
    for source_path, sample_detections, sample_targets in zip(source_paths, detections, targets):
        summary = analyse_map_sample(config, sample_detections, sample_targets, device)
        no_true_positive = summary["ground_truth"] > 0 and summary["tp"] == 0
        affects_map = summary["fp"] > 0 or summary["fn"] > 0
        if not (no_true_positive or affects_map):
            continue
        reasons = []
        if no_true_positive:
            reasons.append("no_true_positive")
        if summary["fn"]:
            reasons.append("false_negative")
        if summary["fp"]:
            reasons.append("false_positive")
        with Image.open(source_path) as source_image:
            source_size = source_image.size
        for box_group in (summary["ground_truth_boxes"], summary["prediction_boxes"]):
            for box in box_group:
                box["xywh_original"] = eval_xywh_to_source(
                    box["xywh_eval"], source_size, config.RESOLUTION,
                )
        records.append({
            "source_image": str(source_path),
            "source_image_size": list(source_size),
            "reasons": reasons,
            "metric_confidence_threshold": float(config.METRIC_CONF_THRESH),
            "metric_iou_threshold": float(config.METRIC_IOU_THRESHOLD),
            **summary,
        })

    unrecognized = [record for record in records if "no_true_positive" in record["reasons"]]
    map_errors = [record for record in records if record["fp"] or record["fn"]]

    def write_path_list(path, selected):
        path.write_text(
            "".join(f"{record['source_image']}\n" for record in selected),
            encoding="utf-8",
        )

    write_path_list(output_dir / "review_unrecognized_images.txt", unrecognized)
    write_path_list(output_dir / "review_map_error_images.txt", map_errors)
    with (output_dir / "review_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "unrecognized_images": len(unrecognized),
        "map_error_images": len(map_errors),
        "candidates_jsonl": "review_candidates.jsonl",
        "unrecognized_list": "review_unrecognized_images.txt",
        "map_error_list": "review_map_error_images.txt",
    }


def write_results(output_dir, confusion, metrics, checkpoint, args, dataset_size, review_summary):
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
        "review_summary": review_summary,
    }
    (output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    if not args.skip_coco:
        require_coco()
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
    configure_teacher_checkpoint(Config, args.checkpoint)
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
    # Preserve historical AP50 and PR fields at their actual IoU=0.5.
    # The command-line IoU remains available for confusion/review analysis.
    review_iou = Config.METRIC_IOU_THRESHOLD
    try:
        Config.METRIC_IOU_THRESHOLD = 0.5
        _, metrics = evaluate_model_anchor_v8(Config, model, dataloader, criterion, device)
    finally:
        Config.METRIC_IOU_THRESHOLD = review_iou
    detections, targets, source_paths = collect_detections(Config, model, dataloader, device)
    add_multi_iou_metrics(metrics, detections, targets, Config)
    confusion, class_tp, class_fp, class_fn = compute_detection_confusion_matrix(
        detections, targets, Config.NUM_CLASSES,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
        image_size=Config.RESOLUTION,
    )

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_coco:
        metrics["coco"] = evaluate_coco(detections, targets, source_paths, Config, output_dir)
    review_summary = write_shared_review_lists(
        output_dir, source_paths, detections, targets, Config, device
    )
    write_results(output_dir, confusion, metrics, checkpoint, args, len(dataset), review_summary)
    print(f"Evaluated {len(dataset)} {args.split} images on {device}.")
    print(f"mAP50: {metrics['map50']:.4f}")
    print(f"mAP50:95 (101-point): {metrics['map50_95']:.4f}")
    print(f"mAP75 (101-point): {metrics['map75']:.4f}")
    print(f"Count matrix (with background): {output_dir / 'confusion_matrix_counts_with_background.png'}")
    print(f"Normalized foreground matrix: {output_dir / 'confusion_matrix_normalized_foreground_percent.png'}")
    print("per-class TP:", class_tp.astype(int))
    print("per-class FP:", class_fp.astype(int))
    print("per-class FN:", class_fn.astype(int))
    print(
        f"Review lists: {review_summary['unrecognized_images']} unrecognized -> "
        f"{output_dir / review_summary['unrecognized_list']}; "
        f"{review_summary['map_error_images']} mAP-error images -> "
        f"{output_dir / review_summary['map_error_list']}"
    )


if __name__ == "__main__":
    main()
