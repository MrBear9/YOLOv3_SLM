import numpy as np
import torch
from tqdm import tqdm

from models.geometry import bbox_iou_xywh
from models.runtime import prepare_batch
# (guidance loss removed — pure detection-driven)
from .decode_anchor_v8 import decode_detections_anchor_v8


def compute_average_precision(detections, total_gt):
    if total_gt == 0:
        return None
    if len(detections) == 0:
        return 0.0
    detections = sorted(detections, key=lambda item: item[0], reverse=True)
    tp = np.array([item[1] for item in detections], dtype=np.float32)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


def evaluate_model_anchor_v8(config, model, dataloader, criterion, device):
    model.eval()
    metric_storage = {cls_id: [] for cls_id in range(config.NUM_CLASSES)}
    gt_counts = {cls_id: 0 for cls_id in range(config.NUM_CLASSES)}
    component_totals = {key: 0.0 for key in ("total", "box", "obj", "noobj", "cls")}
    total_tp = total_fp = total_fn = 0
    is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False, disable=not is_main):
            batch_images, batch_targets = prepare_batch(config, batch, device)
            teacher_features, predictions = model(batch_images, return_feature=True)
            loss, loss_stats = criterion(predictions, batch_targets)
            for key in ("box", "obj", "noobj", "cls"):
                component_totals[key] += loss_stats.get(key, 0.0)
            component_totals["total"] += float(loss.detach().item())
            detections = decode_detections_anchor_v8(
                config,
                predictions,
                conf_thresh=getattr(config, "METRIC_CONF_THRESH", config.CONF_THRESH),
                nms_thresh=getattr(config, "METRIC_NMS_THRESH", config.NMS_THRESH),
                max_det=getattr(config, "METRIC_MAX_DET", config.MAX_DET),
            )

            for sample_idx, sample_detections in enumerate(detections):
                gt_by_class = {}
                for gt in batch_targets[sample_idx]:
                    if gt.shape[0] < 5 or gt[3] <= 0 or gt[4] <= 0:
                        continue
                    cls_id = int(gt[0].item())
                    gt_box = [
                        float(gt[1].item() * config.IMG_SIZE),
                        float(gt[2].item() * config.IMG_SIZE),
                        float(gt[3].item() * config.IMG_SIZE),
                        float(gt[4].item() * config.IMG_SIZE),
                    ]
                    gt_by_class.setdefault(cls_id, []).append(gt_box)
                    gt_counts[cls_id] += 1
                matched = {cls_id: set() for cls_id in gt_by_class}
                for det in sorted(sample_detections, key=lambda item: item[4], reverse=True):
                    cls_id = int(det[5])
                    best_iou, best_gt_idx = 0.0, -1
                    for gt_idx, gt_box in enumerate(gt_by_class.get(cls_id, [])):
                        if gt_idx in matched.get(cls_id, set()):
                            continue
                        iou = float(bbox_iou_xywh(torch.tensor(det[:4]).float().unsqueeze(0), torch.tensor(gt_box).float().unsqueeze(0)).item())
                        if iou > best_iou:
                            best_iou, best_gt_idx = iou, gt_idx
                    is_tp = best_iou >= config.METRIC_IOU_THRESHOLD
                    metric_storage[cls_id].append((float(det[4]), 1.0 if is_tp else 0.0))
                    if is_tp:
                        total_tp += 1
                        matched.setdefault(cls_id, set()).add(best_gt_idx)
                    else:
                        total_fp += 1
                for cls_id, gt_boxes in gt_by_class.items():
                    total_fn += len(gt_boxes) - len(matched.get(cls_id, set()))
    num_batches = max(len(dataloader), 1)
    avg_losses = {key: value / num_batches for key, value in component_totals.items()}
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1_score = 2.0 * precision * recall / (precision + recall + 1e-6)
    ap_values = []
    for cls_id in range(config.NUM_CLASSES):
        ap = compute_average_precision(metric_storage[cls_id], gt_counts[cls_id])
        if ap is not None:
            ap_values.append(ap)
    return avg_losses, {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score),
        "map50": float(np.mean(ap_values)) if ap_values else 0.0,
    }
