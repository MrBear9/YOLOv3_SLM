import numpy as np
import torch
from contextlib import nullcontext
from tqdm import tqdm

from models.geometry import bbox_iou_xywh, bbox_iou_matrix_xywh
from models.runtime import prepare_batch
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
    total_tp_op = total_fp_op = total_fn_op = 0
    op_conf_thresh = float(getattr(config, "CONF_THRESH", 0.35))
    is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    amp_enabled = bool(getattr(config, "ENABLE_AMP", True)) and device.type == "cuda"
    amp_dtype_name = str(getattr(config, "AMP_DTYPE", "float16")).strip().lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False, disable=not is_main):
            batch_images, batch_targets = prepare_batch(config, batch, device)
            with amp_ctx:
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

                dets_by_class = {}
                for det in sample_detections:
                    cls_id = int(det[5])
                    dets_by_class.setdefault(cls_id, []).append(det)

                matched_coco = {}
                for cls_id in set(gt_by_class) | set(dets_by_class):
                    gt_boxes_list = gt_by_class.get(cls_id, [])
                    dets = dets_by_class.get(cls_id, [])
                    if not gt_boxes_list:
                        metric_storage[cls_id].extend((float(det[4]), 0.0) for det in dets)
                        total_fp += len(dets)
                        continue
                    if not dets:
                        total_fn += len(gt_boxes_list)
                        continue
                    gt_boxes = torch.tensor(gt_boxes_list, dtype=torch.float32, device=device)
                    dets_sorted = sorted(dets, key=lambda d: d[4], reverse=True)
                    det_boxes = torch.from_numpy(np.stack([d[:4] for d in dets_sorted])).to(device=device, dtype=torch.float32)
                    det_confs = [d[4] for d in dets_sorted]
                    iou_matrix = bbox_iou_matrix_xywh(det_boxes, gt_boxes)
                    matched_gt = set()
                    for det_idx in range(len(dets_sorted)):
                        ious = iou_matrix[det_idx].clone()
                        for m in matched_gt:
                            ious[m] = -1.0
                        best_iou, best_gt_idx = ious.max(dim=0)
                        best_iou = float(best_iou.item())
                        best_gt_idx = int(best_gt_idx.item())
                        is_tp = best_iou >= config.METRIC_IOU_THRESHOLD
                        metric_storage[cls_id].append((float(det_confs[det_idx]), 1.0 if is_tp else 0.0))
                        if is_tp:
                            total_tp += 1
                            matched_gt.add(best_gt_idx)
                        else:
                            total_fp += 1
                    matched_coco[cls_id] = matched_gt
                    total_fn += len(gt_boxes_list) - len(matched_gt)

                matched_op = {}
                for cls_id in set(gt_by_class) | set(dets_by_class):
                    gt_boxes_list = gt_by_class.get(cls_id, [])
                    dets = dets_by_class.get(cls_id, [])
                    op_dets = [d for d in dets if float(d[4]) >= op_conf_thresh]
                    if not gt_boxes_list:
                        total_fp_op += len(op_dets)
                        continue
                    if not op_dets:
                        total_fn_op += len(gt_boxes_list)
                        continue
                    gt_boxes = torch.tensor(gt_boxes_list, dtype=torch.float32, device=device)
                    op_dets_sorted = sorted(op_dets, key=lambda d: d[4], reverse=True)
                    op_det_boxes = torch.from_numpy(np.stack([d[:4] for d in op_dets_sorted])).to(device=device, dtype=torch.float32)
                    iou_matrix = bbox_iou_matrix_xywh(op_det_boxes, gt_boxes)
                    matched_gt = set()
                    for det_idx in range(len(op_dets_sorted)):
                        ious = iou_matrix[det_idx].clone()
                        for m in matched_gt:
                            ious[m] = -1.0
                        best_iou, best_gt_idx = ious.max(dim=0)
                        best_iou = float(best_iou.item())
                        best_gt_idx = int(best_gt_idx.item())
                        if best_iou >= config.METRIC_IOU_THRESHOLD:
                            total_tp_op += 1
                            matched_gt.add(best_gt_idx)
                        else:
                            total_fp_op += 1
                    matched_op[cls_id] = matched_gt
                    total_fn_op += len(gt_boxes_list) - len(matched_gt)

    num_batches = max(len(dataloader), 1)
    avg_losses = {key: value / num_batches for key, value in component_totals.items()}
    # COCO-style (conf_thresh=0.001) — used for mAP, precision_metric ≈ 0.004
    precision_metric = total_tp / (total_tp + total_fp + 1e-6)
    recall_metric = total_tp / (total_tp + total_fn + 1e-6)
    f1_metric = 2.0 * precision_metric * recall_metric / (precision_metric + recall_metric + 1e-6)
    # Operating-point (conf_thresh=0.35) — human-interpretable, should reach 0.7-0.9
    precision_op = total_tp_op / (total_tp_op + total_fp_op + 1e-6)
    recall_op = total_tp_op / (total_tp_op + total_fn_op + 1e-6)
    f1_op = 2.0 * precision_op * recall_op / (precision_op + recall_op + 1e-6)
    ap_values = []
    for cls_id in range(config.NUM_CLASSES):
        ap = compute_average_precision(metric_storage[cls_id], gt_counts[cls_id])
        if ap is not None:
            ap_values.append(ap)
    return avg_losses, {
        "precision": float(precision_metric),
        "recall": float(recall_metric),
        "f1": float(f1_metric),
        "map50": float(np.mean(ap_values)) if ap_values else 0.0,
        "precision_op": float(precision_op),
        "recall_op": float(recall_op),
        "f1_op": float(f1_op),
    }
