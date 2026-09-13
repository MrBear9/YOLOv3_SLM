"""Multi-IoU box AP from cached predictions, without model inference.

Inputs: detections [cx, cy, w, h, score, class] in canvas pixels;
targets [class, cx, cy, w, h] normalized to the same canvas.
Uses 101-point interpolated AP, all areas, no crowd/ignore annotations.
The caller owns confidence filtering, NMS and the maximum detection count.
"""

from math import isfinite


IOU_THRESHOLDS = tuple((50 + 5 * i) / 100 for i in range(10))


def _iou(a, b):
    width = max(0.0, min(a[0] + a[2] / 2, b[0] + b[2] / 2)
                - max(a[0] - a[2] / 2, b[0] - b[2] / 2))
    height = max(0.0, min(a[1] + a[3] / 2, b[1] + b[3] / 2)
                 - max(a[1] - a[3] / 2, b[1] - b[3] / 2))
    intersection = width * height
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union > 0 else 0.0


def _ap101(records, total_gt):
    if not total_gt:
        return None
    records = sorted(records, key=lambda item: item[0], reverse=True)
    recalls, precisions = [], []
    tp = 0
    for rank, (_, matched) in enumerate(records, 1):
        tp += matched
        recalls.append(tp / total_gt)
        precisions.append(tp / rank)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    index, total = 0, 0.0
    for r in range(101):
        while index < len(recalls) and recalls[index] < r / 100:
            index += 1
        if index < len(precisions):
            total += precisions[index]
    return total / 101


def compute_multi_iou_metrics(detections, targets, num_classes, resolution):
    """Match independently at each IoU, then aggregate by class and threshold.

    Classes without GT are excluded from macro averages (AP=None). An empty
    evaluation returns zero aggregate scores and evaluated_classes=0.
    """
    if len(detections) != len(targets):
        raise ValueError("Predictions and targets must have the same image count.")
    if num_classes < 1 or len(resolution) != 2 or min(resolution) <= 0:
        raise ValueError("Positive class count and (height, width) are required.")
    height, width = resolution
    scales = (width, height, width, height)
    counts = [0] * num_classes
    records = [[[] for _ in IOU_THRESHOLDS] for _ in range(num_classes)]
    for sample_dets, sample_targets in zip(detections, targets):
        gt_by_class = [[] for _ in range(num_classes)]
        det_by_class = [[] for _ in range(num_classes)]
        for raw in sample_targets:
            row = [float(value) for value in raw]
            if len(row) != 5 or not all(isfinite(v) for v in row):
                raise ValueError("Targets must contain five finite values.")
            class_id = int(row[0])
            if row[0] != class_id or not 0 <= class_id < num_classes:
                raise ValueError("Target class is outside the configured classes.")
            if row[3] <= 0 or row[4] <= 0:
                continue
            gt_by_class[class_id].append([v * s for v, s in zip(row[1:], scales)])
            counts[class_id] += 1
        for raw in sample_dets:
            row = [float(value) for value in raw]
            if len(row) != 6 or not all(isfinite(v) for v in row):
                raise ValueError("Detections must contain six finite values.")
            class_id = int(row[5])
            if row[5] != class_id or not 0 <= class_id < num_classes:
                raise ValueError("Detection class is outside the configured classes.")
            det_by_class[class_id].append(row)
        for class_id in range(num_classes):
            ordered = sorted(det_by_class[class_id], key=lambda row: row[4], reverse=True)
            gt = gt_by_class[class_id]
            ious = [[_iou(det, box) for box in gt] for det in ordered]
            for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
                matched = set()
                for det, overlaps in zip(ordered, ious):
                    candidates = [j for j in range(len(gt)) if j not in matched]
                    best = max(candidates, key=lambda j: overlaps[j], default=None)
                    is_tp = best is not None and overlaps[best] >= threshold
                    if is_tp:
                        matched.add(best)
                    records[class_id][threshold_index].append((det[4], int(is_tp)))
    per_class = {}
    valid_aps = []
    for class_id, count in enumerate(counts):
        aps = [_ap101(items, count) for items in records[class_id]]
        if count:
            valid_aps.append(aps)
        per_class[class_id] = {
            "ap50_101": aps[0], "ap75": aps[5],
            "ap50_95": sum(aps) / len(aps) if count else None,
            "ap_by_iou": {f"{iou:.2f}": ap for iou, ap in zip(IOU_THRESHOLDS, aps)},
        }
    means = [sum(ap[i] for ap in valid_aps) / len(valid_aps) if valid_aps else 0.0
             for i in range(len(IOU_THRESHOLDS))]
    return {
        "map50_101": means[0], "map75": means[5],
        "map50_95": sum(means) / len(means),
        "map_by_iou": {f"{iou:.2f}": ap for iou, ap in zip(IOU_THRESHOLDS, means)},
        "multi_iou_protocol": {
            "iou_thresholds": list(IOU_THRESHOLDS), "recall_points": 101,
            "matching": "per-image, per-class, confidence-ordered greedy; independent per IoU",
            "area_range": "all", "crowd_ignore_supported": False,
            "prediction_filtering": "caller confidence, NMS and max_det settings",
            "evaluated_classes": len(valid_aps),
            "legacy_map50": "existing all-point PR integral, not the 101-point map50_101",
        },
        "per_class": per_class,
    }


def add_multi_iou_metrics(metrics, detections, targets, config):
    """Add new AP fields without replacing historical AP50/PR statistics."""
    result = compute_multi_iou_metrics(detections, targets, config.NUM_CLASSES, config.RESOLUTION)
    per_class = result.pop("per_class")
    for class_id, values in per_class.items():
        metrics.setdefault("per_class", {}).setdefault(class_id, {}).update(values)
    result["multi_iou_protocol"].update({
        "confidence_threshold": float(config.METRIC_CONF_THRESH),
        "nms_threshold": float(config.METRIC_NMS_THRESH),
        "max_detections_per_image": int(config.METRIC_MAX_DET),
    })
    metrics.update(result)
    return metrics
