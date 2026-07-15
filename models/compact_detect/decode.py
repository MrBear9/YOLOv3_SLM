import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms, nms


def local_maximum_nms(heatmap, kernel=3):
    pad = (kernel - 1) // 2
    pooled = F.max_pool2d(heatmap, kernel, stride=1, padding=pad)
    return heatmap * (pooled == heatmap).to(heatmap.dtype)


def _xywh_to_xyxy_tensor(boxes):
    half_w = boxes[:, 2] / 2
    half_h = boxes[:, 3] / 2
    return torch.stack(
        [
            boxes[:, 0] - half_w,
            boxes[:, 1] - half_h,
            boxes[:, 0] + half_w,
            boxes[:, 1] + half_h,
        ],
        dim=1,
    )


def decode_center_detections(config, pred, conf_thresh=None, nms_thresh=None, max_det=None, pre_nms_topk=None):
    conf_thresh = float(getattr(config, "CONF_THRESH", 0.3) if conf_thresh is None else conf_thresh)
    nms_thresh = float(getattr(config, "NMS_THRESH", 0.45) if nms_thresh is None else nms_thresh)
    max_det = int(getattr(config, "MAX_DET", 100) if max_det is None else max_det)
    pre_nms_topk = int(getattr(config, "DECODE_PRE_NMS_TOPK", max_det * 4) if pre_nms_topk is None else pre_nms_topk)
    stride = int(getattr(config, "OUTPUT_STRIDE", 4))

    heatmap = local_maximum_nms(torch.sigmoid(pred["heatmap"]))
    wh = pred["wh"]
    offset = pred["offset"]
    batch_size, num_classes, out_h, out_w = heatmap.shape
    topk = min(max(pre_nms_topk, max_det), num_classes * out_h * out_w)
    flat_scores, flat_indices = torch.topk(heatmap.reshape(batch_size, -1), k=topk, dim=1)
    detections = []
    for batch_idx in range(batch_size):
        scores = flat_scores[batch_idx]
        keep = scores >= conf_thresh
        if not keep.any():
            detections.append(np.zeros((0, 6), dtype=np.float32))
            continue

        indices = flat_indices[batch_idx][keep]
        scores = scores[keep]
        cls_ids = torch.div(indices, out_h * out_w, rounding_mode="floor")
        rem = indices.remainder(out_h * out_w)
        gy = torch.div(rem, out_w, rounding_mode="floor")
        gx = rem.remainder(out_w)

        off_x = offset[batch_idx, 0, gy, gx]
        off_y = offset[batch_idx, 1, gy, gx]
        bw = wh[batch_idx, 0, gy, gx].clamp(min=1.0)
        bh = wh[batch_idx, 1, gy, gx].clamp(min=1.0)
        cx = (gx.to(offset.dtype) + off_x) * stride
        cy = (gy.to(offset.dtype) + off_y) * stride
        boxes_xywh = torch.stack([cx, cy, bw, bh], dim=1)
        boxes_xyxy = _xywh_to_xyxy_tensor(boxes_xywh)

        if bool(getattr(config, "AGNOSTIC_NMS", False)):
            kept = nms(boxes_xyxy, scores, nms_thresh)
        else:
            kept = batched_nms(boxes_xyxy, scores, cls_ids, nms_thresh)
        kept = kept[:max_det]
        sample = torch.cat(
            [
                boxes_xywh[kept],
                scores[kept, None],
                cls_ids[kept, None].to(boxes_xywh.dtype),
            ],
            dim=1,
        )
        detections.append(sample.detach().float().cpu().numpy())
    return detections


# ═══════════════════════════════════════════════════════════════════════════
# Decoupled decode  (Level-4 V2:  obj + cls + wh + offset)
# ═══════════════════════════════════════════════════════════════════════════

def decode_decoupled(config, pred, conf_thresh=None, nms_thresh=None,
                     max_det=None, pre_nms_topk=None):
    """Decode detections from decoupled obj+cls+wh+offset predictions.

    Differences from ``decode_center_detections``:
      - Peak-finding on ``pred["obj"]`` (class-agnostic binary map) instead
        of the per-class ``pred["heatmap"]``.
      - Class is determined via softmax on ``pred["cls"]`` at peak locations.
      - Final score = obj_score × cls_score  (decoupled confidence).

    All other logic (grid extraction, wh/offset, NMS) is identical to V1.
    """
    conf_thresh = float(getattr(config, "CONF_THRESH", 0.3) if conf_thresh is None else conf_thresh)
    nms_thresh = float(getattr(config, "NMS_THRESH", 0.45) if nms_thresh is None else nms_thresh)
    max_det = int(getattr(config, "MAX_DET", 100) if max_det is None else max_det)
    pre_nms_topk = int(getattr(config, "DECODE_PRE_NMS_TOPK", max_det * 4) if pre_nms_topk is None else pre_nms_topk)
    stride = int(getattr(config, "OUTPUT_STRIDE", 4))

    # Peak suppression on binary objectness map
    obj = local_maximum_nms(torch.sigmoid(pred["obj"]))   # (B, 1, H, W)
    cls_logits = pred["cls"]                               # (B, N, H, W)  raw
    wh = pred["wh"]
    offset = pred["offset"]

    batch_size, _, out_h, out_w = obj.shape
    topk = min(max(pre_nms_topk, max_det), out_h * out_w)
    flat_scores, flat_indices = torch.topk(obj.reshape(batch_size, -1), k=topk, dim=1)

    detections = []
    for batch_idx in range(batch_size):
        scores = flat_scores[batch_idx]
        keep = scores >= conf_thresh
        if not keep.any():
            detections.append(np.zeros((0, 6), dtype=np.float32))
            continue

        indices = flat_indices[batch_idx][keep]
        scores_obj = scores[keep]

        # Grid positions
        gy = torch.div(indices, out_w, rounding_mode="floor")
        gx = indices.remainder(out_w)

        # Class via softmax at peak locations
        # cls_at_peaks: (N_cls, num_keep)  — one logit per class per peak
        cls_at_peaks = cls_logits[batch_idx, :, gy, gx]
        cls_probs = F.softmax(cls_at_peaks, dim=0)           # softmax across classes
        cls_scores, cls_ids = cls_probs.max(dim=0)           # best class per peak
        final_scores = scores_obj * cls_scores

        # Box regression (same as V1)
        off_x = offset[batch_idx, 0, gy, gx]
        off_y = offset[batch_idx, 1, gy, gx]
        bw = wh[batch_idx, 0, gy, gx].clamp(min=1.0)
        bh = wh[batch_idx, 1, gy, gx].clamp(min=1.0)
        cx = (gx.to(offset.dtype) + off_x) * stride
        cy = (gy.to(offset.dtype) + off_y) * stride
        boxes_xywh = torch.stack([cx, cy, bw, bh], dim=1)
        boxes_xyxy = _xywh_to_xyxy_tensor(boxes_xywh)

        # NMS
        if bool(getattr(config, "AGNOSTIC_NMS", False)):
            kept = nms(boxes_xyxy, final_scores, nms_thresh)
        else:
            kept = batched_nms(boxes_xyxy, final_scores, cls_ids.to(boxes_xyxy.dtype), nms_thresh)
        kept = kept[:max_det]

        sample = torch.cat(
            [
                boxes_xywh[kept],
                final_scores[kept, None],
                cls_ids[kept, None].to(boxes_xywh.dtype),
            ],
            dim=1,
        )
        detections.append(sample.detach().float().cpu().numpy())
    return detections
