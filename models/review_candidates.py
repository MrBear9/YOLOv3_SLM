"""Shared per-image mAP review exports for teacher and optical-student evaluation."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.dataset import letterbox_content_bounds
from models.geometry import bbox_iou_matrix_xywh


def _class_name(config, class_id):
    class_names = getattr(config, "CLASS_NAMES", {})
    if isinstance(class_names, dict):
        return class_names.get(class_id, str(class_id))
    try:
        return class_names[class_id]
    except (IndexError, TypeError):
        return str(class_id)


def analyse_map_sample(config, detections, targets, device):
    """Return per-image class-wise TP/FP/FN details using the mAP50 protocol."""
    ground_truths = []
    gt_by_class = {}
    image_h, image_w = config.RESOLUTION
    for gt in targets:
        if len(gt) < 5 or gt[3] <= 0 or gt[4] <= 0:
            continue
        class_id = int(gt[0])
        gt_box = [
            float(gt[1] * image_w), float(gt[2] * image_h),
            float(gt[3] * image_w), float(gt[4] * image_h),
        ]
        gt_index = len(ground_truths)
        ground_truths.append({
            "class_id": class_id,
            "class_name": _class_name(config, class_id),
            "xywh_eval": gt_box,
            "status": "fn",
        })
        gt_by_class.setdefault(class_id, []).append(gt_index)

    predictions = []
    detections_by_class = {}
    for detection in np.asarray(detections, dtype=np.float32).reshape(-1, 6):
        class_id = int(detection[5])
        prediction_index = len(predictions)
        predictions.append({
            "class_id": class_id,
            "class_name": _class_name(config, class_id),
            "confidence": float(detection[4]),
            "xywh_eval": [float(value) for value in detection[:4]],
            "status": "fp",
        })
        detections_by_class.setdefault(class_id, []).append(prediction_index)

    total_tp = total_fp = total_fn = 0
    per_class = []
    for class_id in sorted(set(gt_by_class) | set(detections_by_class)):
        gt_indices = gt_by_class.get(class_id, [])
        detection_indices = sorted(
            detections_by_class.get(class_id, []),
            key=lambda index: predictions[index]["confidence"],
            reverse=True,
        )
        gt_boxes = [ground_truths[index]["xywh_eval"] for index in gt_indices]
        detections_for_class = [predictions[index] for index in detection_indices]
        tp = fp = fn = 0
        if not gt_boxes:
            fp = len(detections_for_class)
        elif not detections_for_class:
            fn = len(gt_boxes)
        else:
            gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32, device=device)
            detection_tensor = torch.tensor(
                [detection["xywh_eval"] for detection in detections_for_class],
                dtype=torch.float32,
                device=device,
            )
            iou_matrix = bbox_iou_matrix_xywh(detection_tensor, gt_tensor)
            matched_gt = set()
            for detection_position, prediction_index in enumerate(detection_indices):
                ious = iou_matrix[detection_position].clone()
                for gt_position in matched_gt:
                    ious[gt_position] = -1.0
                best_iou, best_gt_position = ious.max(dim=0)
                if float(best_iou.item()) >= config.METRIC_IOU_THRESHOLD:
                    tp += 1
                    gt_position = int(best_gt_position.item())
                    matched_gt.add(gt_position)
                    prediction = predictions[prediction_index]
                    ground_truth = ground_truths[gt_indices[gt_position]]
                    prediction["status"] = "tp"
                    prediction["matched_gt_index"] = gt_indices[gt_position]
                    prediction["matched_iou"] = float(best_iou.item())
                    ground_truth["status"] = "tp"
                    ground_truth["matched_prediction_index"] = prediction_index
                    ground_truth["matched_iou"] = float(best_iou.item())
                else:
                    fp += 1
            fn = len(gt_boxes) - len(matched_gt)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_class.append({
            "class_id": class_id,
            "class_name": _class_name(config, class_id),
            "ground_truth": len(gt_boxes),
            "detections": len(detections_for_class),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

    return {
        "ground_truth": len(ground_truths),
        "detections": len(predictions),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_class": per_class,
        "ground_truth_boxes": ground_truths,
        "prediction_boxes": predictions,
    }


def eval_xywh_to_source(xywh_eval, source_size, resolution):
    """Map an evaluator-canvas box back to original-image pixel coordinates."""
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
    """Write list files and detailed JSONL records consumable by the review UI."""
    if not (len(source_paths) == len(detections) == len(targets)):
        raise ValueError("Review export requires equally sized paths, detections, and targets.")

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

        source_path = Path(source_path).resolve()
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

    output_dir = Path(output_dir)
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
