import numpy as np
import torch
import torch.distributed as dist
from contextlib import nullcontext
from tqdm import tqdm

from models.geometry import bbox_iou_matrix_xywh
from .decode import decode_center_detections


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


def _distributed_merge_metrics(metric_storage, gt_counts, loss_totals, total_tp, total_fp, total_fn, num_batches, device):
    if not dist.is_initialized():
        return metric_storage, gt_counts, loss_totals, total_tp, total_fp, total_fn, num_batches

    payload = {
        "metric_storage": metric_storage,
        "gt_counts": gt_counts,
    }
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, payload)

    merged_storage = {cls_id: [] for cls_id in metric_storage.keys()}
    merged_gt_counts = {cls_id: 0 for cls_id in gt_counts.keys()}
    for item in gathered:
        for cls_id, values in item["metric_storage"].items():
            merged_storage[cls_id].extend(values)
        for cls_id, value in item["gt_counts"].items():
            merged_gt_counts[cls_id] += int(value)

    stat_tensor = torch.tensor(
        [
            loss_totals["total"],
            loss_totals["heatmap"],
            loss_totals["wh"],
            loss_totals["offset"],
            float(total_tp),
            float(total_fp),
            float(total_fn),
            float(num_batches),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(stat_tensor, op=dist.ReduceOp.SUM)
    merged_losses = {
        "total": float(stat_tensor[0].item()),
        "heatmap": float(stat_tensor[1].item()),
        "wh": float(stat_tensor[2].item()),
        "offset": float(stat_tensor[3].item()),
    }
    return (
        merged_storage,
        merged_gt_counts,
        merged_losses,
        int(stat_tensor[4].item()),
        int(stat_tensor[5].item()),
        int(stat_tensor[6].item()),
        int(max(stat_tensor[7].item(), 1.0)),
    )


@torch.no_grad()
def evaluate_center_detector(config, student, detector, dataloader, criterion, device):
    student.eval()
    detector.eval()
    metric_storage = {cls_id: [] for cls_id in range(config.NUM_CLASSES)}
    gt_counts = {cls_id: 0 for cls_id in range(config.NUM_CLASSES)}
    loss_totals = {key: 0.0 for key in ("total", "heatmap", "wh", "offset")}
    total_tp = total_fp = total_fn = 0
    is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    amp_enabled = bool(getattr(config, "ENABLE_AMP", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(getattr(config, "AMP_DTYPE", "float16")).lower() in {"bf16", "bfloat16"} else torch.float16
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled) if device.type == "cuda" else nullcontext()

    for batch in tqdm(dataloader, desc="Validation", leave=False, disable=not is_main):
        images = batch["gray_tensor"].to(device, non_blocking=config.PIN_MEMORY)
        targets = [target.to(device, non_blocking=config.PIN_MEMORY) for target in batch["targets"]]
        with amp_ctx:
            features = student(images)
            pred = detector(features)
            loss, stats = criterion(pred, targets)
        for key in loss_totals:
            loss_totals[key] += stats.get(key, float(loss.detach().item()) if key == "total" else 0.0)
        detections = decode_center_detections(
            config,
            pred,
            conf_thresh=getattr(config, "METRIC_CONF_THRESH", config.CONF_THRESH),
            nms_thresh=getattr(config, "METRIC_NMS_THRESH", config.NMS_THRESH),
            max_det=getattr(config, "METRIC_MAX_DET", config.MAX_DET),
            pre_nms_topk=getattr(config, "METRIC_PRE_NMS_TOPK", None),
        )

        for sample_idx, sample_detections in enumerate(detections):
            gt_by_class = {}
            for gt in targets[sample_idx]:
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
                dets_by_class.setdefault(int(det[5]), []).append(det)

            for cls_id, gt_boxes_list in gt_by_class.items():
                gt_boxes = torch.tensor(gt_boxes_list, dtype=torch.float32, device=device)
                dets = sorted(dets_by_class.get(cls_id, []), key=lambda d: d[4], reverse=True)
                if not dets:
                    total_fn += len(gt_boxes_list)
                    continue
                det_boxes = torch.from_numpy(np.stack([d[:4] for d in dets])).to(device=device, dtype=torch.float32)
                iou_matrix = bbox_iou_matrix_xywh(det_boxes, gt_boxes)
                matched_gt = set()
                for det_idx, det in enumerate(dets):
                    ious = iou_matrix[det_idx].clone()
                    for matched_idx in matched_gt:
                        ious[matched_idx] = -1.0
                    best_iou, best_gt_idx = ious.max(dim=0)
                    is_tp = float(best_iou.item()) >= config.METRIC_IOU_THRESHOLD
                    metric_storage[cls_id].append((float(det[4]), 1.0 if is_tp else 0.0))
                    if is_tp:
                        total_tp += 1
                        matched_gt.add(int(best_gt_idx.item()))
                    else:
                        total_fp += 1
                total_fn += len(gt_boxes_list) - len(matched_gt)

    num_batches = max(len(dataloader), 1)
    metric_storage, gt_counts, loss_totals, total_tp, total_fp, total_fn, num_batches = _distributed_merge_metrics(
        metric_storage,
        gt_counts,
        loss_totals,
        total_tp,
        total_fp,
        total_fn,
        num_batches,
        device,
    )
    losses = {key: value / num_batches for key, value in loss_totals.items()}
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    ap_values = []
    for cls_id in range(config.NUM_CLASSES):
        ap = compute_average_precision(metric_storage[cls_id], gt_counts[cls_id])
        if ap is not None:
            ap_values.append(ap)
    return losses, {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "map50": float(np.mean(ap_values)) if ap_values else 0.0,
    }
