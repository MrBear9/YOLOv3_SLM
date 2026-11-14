"""Class-aware IoU diagnostics for exported, simulated hardware sample detections.

Boxes are center-based xywh in the same letterboxed pixel coordinates. These
diagnostics describe exported detections, not camera measurements or dataset AP.
"""

from __future__ import annotations

import math


def summarize_detection_iou(records, threshold: float = 0.5) -> dict:
    """Pool exported TP pairs/GTs, rather than averaging per-image means."""
    if not 0 < threshold <= 1:
        raise ValueError("Match IoU threshold must be in (0, 1].")
    tp = fp = fn = images = 0
    iou_sums = []
    for record in records:
        metrics = record["iou_metrics"]
        if metrics["match_iou_threshold"] != threshold:
            raise ValueError("Cannot summarize samples with different matching thresholds.")
        images += 1
        tp += metrics["tp"]
        fp += metrics["fp"]
        fn += metrics["fn"]
        iou_sums.append(math.fsum(p["matched_iou"] for p in record["detections"] if p["is_tp"]))
    total_iou = math.fsum(iou_sums)
    return {
        "mean_matched_iou": total_iou / tp if tp else None,
        "mean_gt_iou": total_iou / (tp + fn) if tp + fn else None,
        "tp": tp, "fp": fp, "fn": fn,
        "exported_images": images,
        "prediction_count": tp + fp,
        "ground_truth_count": tp + fn,
        "match_iou_threshold": threshold,
        "prediction_source": "simulation",
        "scope": "exported images only, after confidence filtering and NMS; respects --limit",
        "aggregation": "sum of TP matched IoUs / total TP (mean_matched_iou) or total GT (mean_gt_iou); unmatched GTs count as zero",
        "note": "IoU values are in [0, 1]; zero denominator -> null; not AP or mAP",
    }


def box_iou_xywh(first, second) -> float:
    def corners(box):
        cx, cy, width, height = map(float, box)
        if not all(math.isfinite(value) for value in (cx, cy, width, height)):
            raise ValueError("Non-finite exported box coordinates.")
        if width < 0 or height < 0:
            raise ValueError("Exported box width/height must be nonnegative.")
        return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2

    ax1, ay1, ax2, ay2 = corners(first)
    bx1, by1, bx2, by2 = corners(second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def add_detection_iou(record: dict, threshold: float = 0.5) -> dict:
    """Enrich a sample in place; ties use original array order.

    Match predictions in descending confidence to the best still-unmatched GT
    of the same class, accepting IoU >= threshold. Best-overlap fields ignore
    assignment, so duplicate detections can have high best IoU but remain FP.
    mean_matched_iou averages TP pairs only (None with no TP). mean_gt_iou
    divides their IoU sum by all GTs, counting unmatched GTs as zero.
    """
    if not 0 < threshold <= 1:
        raise ValueError("Match IoU threshold must be in (0, 1].")
    predictions, targets = record["detections"], record["ground_truth"]
    overlaps = [
        [box_iou_xywh(pred["xywh_pixels"], gt["xywh_pixels"])
         if int(pred["class_id"]) == int(gt["class_id"]) else None for gt in targets]
        for pred in predictions
    ]
    for pred, row in zip(predictions, overlaps):
        candidates = [j for j, value in enumerate(row) if value is not None]
        best = max(candidates, key=lambda j: row[j]) if candidates else None
        pred.update(best_gt_index=best, best_same_class_iou=row[best] if best is not None else None,
                    matched_gt_index=None, matched_iou=None, is_tp=False)
    for gt in targets:
        gt.update(matched_prediction_index=None, matched_iou=None, is_matched=False)
    used = set()
    matched_ious = []
    for index in sorted(range(len(predictions)), key=lambda i: -float(predictions[i]["confidence"])):
        row = overlaps[index]
        available = [j for j, value in enumerate(row) if value is not None and j not in used]
        if not available:
            continue
        best = max(available, key=lambda j: row[j])
        value = row[best]
        if value < threshold:
            continue
        used.add(best)
        matched_ious.append(value)
        predictions[index].update(matched_gt_index=best, matched_iou=value, is_tp=True)
        targets[best].update(matched_prediction_index=index, matched_iou=value, is_matched=True)
    tp = len(matched_ious)
    record["iou_metrics"] = {
        "schema_version": 1,
        "prediction_source": "simulation",
        "box_format": "center_xywh_pixels_letterboxed",
        "match_iou_threshold": threshold,
        "matching_rule": "same_class_confidence_descending_greedy_best_unmatched_gt",
        "index_convention": "zero_based_original_array_order",
        "tp": tp, "fp": len(predictions) - tp, "fn": len(targets) - tp,
        "mean_matched_iou": sum(matched_ious) / tp if tp else None,
        "mean_gt_iou": sum(matched_ious) / len(targets) if targets else None,
        "unmatched_prediction_indices": [i for i, pred in enumerate(predictions) if not pred["is_tp"]],
        "unmatched_gt_indices": [j for j, gt in enumerate(targets) if not gt["is_matched"]],
        "annotation_metric": "mean_matched_iou",
        "annotation_note": "100 * mean IoU of TP matches only; no TP -> N/A; not AP or mAP",
    }
    return record
