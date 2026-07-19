"""Lightweight YOLOv8-style anchor-free assignment, loss, and decoding."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms


def get_anchor_free_strides(config):
    return getattr(config, "ANCHOR_FREE_STRIDES", config.STRIDES)


def make_anchor_points(predictions, strides, dtype, device):
    points, stride_values = [], []
    for pred, stride in zip(predictions, strides):
        h, w = pred["cls"].shape[-2:]
        y, x = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype) + 0.5,
            torch.arange(w, device=device, dtype=dtype) + 0.5,
            indexing="ij",
        )
        points.append(torch.stack((x, y), dim=-1).reshape(-1, 2) * float(stride))
        stride_values.append(torch.full((h * w, 1), float(stride), device=device, dtype=dtype))
    return torch.cat(points), torch.cat(stride_values)


def flatten_predictions(predictions, reg_max):
    cls = torch.cat([p["cls"].flatten(2).permute(0, 2, 1) for p in predictions], dim=1)
    reg = torch.cat(
        [p["reg"].view(p["reg"].shape[0], 4, reg_max, -1).permute(0, 3, 1, 2) for p in predictions],
        dim=1,
    )
    return cls, reg


def decode_boxes(reg_logits, points, strides, reg_max):
    projection = torch.arange(reg_max, device=reg_logits.device, dtype=reg_logits.dtype)
    distances = reg_logits.softmax(-1).matmul(projection) * strides.unsqueeze(0)
    return torch.cat((points.unsqueeze(0) - distances[..., :2], points.unsqueeze(0) + distances[..., 2:]), dim=-1)


def pairwise_iou(boxes1, boxes2, eps=1e-7):
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(-1)
    return inter / (area1[:, None] + area2[None, :] - inter + eps)


def aligned_iou(boxes1, boxes2, eps=1e-7):
    lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(-1)
    return inter / (area1 + area2 - inter + eps)


def aligned_ciou(boxes1, boxes2, eps=1e-7):
    iou = aligned_iou(boxes1, boxes2, eps)
    center1, center2 = (boxes1[:, :2] + boxes1[:, 2:]) / 2, (boxes2[:, :2] + boxes2[:, 2:]) / 2
    center_distance = (center1 - center2).square().sum(-1)
    enclosing_lt, enclosing_rb = torch.minimum(boxes1[:, :2], boxes2[:, :2]), torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
    enclosing_diagonal = (enclosing_rb - enclosing_lt).square().sum(-1).clamp(min=eps)
    wh1, wh2 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=eps), (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=eps)
    aspect = (4.0 / (torch.pi ** 2)) * (torch.atan(wh2[:, 0] / wh2[:, 1]) - torch.atan(wh1[:, 0] / wh1[:, 1])).square()
    with torch.no_grad():
        alpha = aspect / (1.0 - iou + aspect + eps)
    return iou - center_distance / enclosing_diagonal - alpha * aspect


class TaskAlignedAssigner:
    def __init__(self, topk=10, alpha=0.5, beta=6.0):
        self.topk, self.alpha, self.beta = int(topk), float(alpha), float(beta)

    @torch.no_grad()
    def __call__(self, scores, boxes, points, gt_boxes, gt_classes):
        num_points, num_classes = scores.shape
        target_scores = scores.new_zeros((num_points, num_classes))
        target_boxes = boxes.new_zeros((num_points, 4))
        foreground = torch.zeros(num_points, device=scores.device, dtype=torch.bool)
        if gt_boxes.numel() == 0:
            return target_scores, target_boxes, foreground
        inside = (
            (points[:, None, 0] > gt_boxes[None, :, 0]) & (points[:, None, 0] < gt_boxes[None, :, 2])
            & (points[:, None, 1] > gt_boxes[None, :, 1]) & (points[:, None, 1] < gt_boxes[None, :, 3])
        )
        ious = pairwise_iou(boxes, gt_boxes).clamp(min=0)
        metric = scores[:, gt_classes].pow(self.alpha) * ious.pow(self.beta) * inside
        candidate = torch.zeros_like(inside)
        k = min(self.topk, num_points)
        top_values, top_indices = metric.topk(k, dim=0)
        candidate.scatter_(0, top_indices, top_values > 0)
        matched_iou = ious.masked_fill(~candidate, -1.0)
        best_iou, matched_gt = matched_iou.max(dim=1)
        foreground = best_iou >= 0
        if foreground.any():
            fg_gt = matched_gt[foreground]
            target_boxes[foreground] = gt_boxes[fg_gt]
            soft_labels = best_iou[foreground].clamp(min=0).to(dtype=target_scores.dtype)
            target_scores[foreground, gt_classes[fg_gt]] = soft_labels
        return target_scores, target_boxes, foreground


class AnchorFreeTALLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.reg_max = int(getattr(config, "ANCHOR_FREE_REG_MAX", 16))
        self.assigner = TaskAlignedAssigner(
            getattr(config, "TAL_TOPK", 10), getattr(config, "TAL_ALPHA", 0.5), getattr(config, "TAL_BETA", 6.0)
        )
        self.last_components = {"total": 0.0, "box": 0.0, "obj": 0.0, "noobj": 0.0, "cls": 0.0, "dfl": 0.0}

    def set_epoch_weights(self, epoch):
        return self.config.get_dynamic_weights(epoch)["phase"]

    @staticmethod
    def targets_to_xyxy(targets, image_size, device, dtype):
        if targets is None or targets.numel() == 0:
            return torch.empty((0, 4), device=device, dtype=dtype), torch.empty(0, device=device, dtype=torch.long)
        targets = targets.to(device=device, dtype=dtype)
        cx, cy = targets[:, 1] * image_size, targets[:, 2] * image_size
        w, h = targets[:, 3] * image_size, targets[:, 4] * image_size
        return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1), targets[:, 0].long()

    def dfl_loss(self, logits, target):
        # Keep bin construction in FP32. In FP16, 14.999 rounds to 15 for
        # reg_max=16, which previously made the right bin 16 (out of range).
        target = target.float().clamp(0, self.reg_max - 1)
        left = target.floor().long().clamp_(0, self.reg_max - 1)
        right = (left + 1).clamp_max_(self.reg_max - 1)
        wr = target - left.to(target.dtype)
        wr = torch.where(right == left, torch.zeros_like(wr), wr)
        flat = logits.reshape(-1, self.reg_max)
        return (
            F.cross_entropy(flat, left.reshape(-1), reduction="none").view_as(target) * (1 - wr)
            + F.cross_entropy(flat, right.reshape(-1), reduction="none").view_as(target) * wr
        ).mean(-1)

    def forward(self, predictions, targets):
        cls_logits, reg_logits = flatten_predictions(predictions, self.reg_max)
        points, strides = make_anchor_points(predictions, get_anchor_free_strides(self.config), cls_logits.dtype, cls_logits.device)
        decoded = decode_boxes(reg_logits, points, strides, self.reg_max)
        target_scores, target_boxes = torch.zeros_like(cls_logits), torch.zeros_like(decoded)
        foreground = torch.zeros(cls_logits.shape[:2], device=cls_logits.device, dtype=torch.bool)
        for b in range(cls_logits.shape[0]):
            gt_boxes, gt_classes = self.targets_to_xyxy(targets[b], self.config.IMG_SIZE, cls_logits.device, cls_logits.dtype)
            target_scores[b], target_boxes[b], foreground[b] = self.assigner(
                cls_logits[b].sigmoid(), decoded[b].detach(), points, gt_boxes, gt_classes
            )
        score_sum = target_scores.sum().clamp(min=1.0)
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, target_scores, reduction="sum") / score_sum
        if foreground.any():
            weights = target_scores.sum(-1)[foreground].clamp(min=1e-3)
            boxes = target_boxes[foreground]
            box_loss = ((1.0 - aligned_ciou(decoded[foreground], target_boxes[foreground])) * weights).sum() / weights.sum()
            batch_points = points.unsqueeze(0).expand(cls_logits.shape[0], -1, -1)[foreground]
            batch_strides = strides.unsqueeze(0).expand(cls_logits.shape[0], -1, -1)[foreground]
            distances = torch.cat((batch_points - boxes[:, :2], boxes[:, 2:] - batch_points), dim=-1) / batch_strides
            dfl_loss = (self.dfl_loss(reg_logits[foreground], distances) * weights).sum() / weights.sum()
        else:
            box_loss = reg_logits.sum() * 0.0
            dfl_loss = reg_logits.sum() * 0.0
        total = (
            box_loss * float(getattr(self.config, "ANCHOR_FREE_BOX_WEIGHT", 7.5))
            + cls_loss * float(getattr(self.config, "ANCHOR_FREE_CLS_WEIGHT", 0.5))
            + dfl_loss * float(getattr(self.config, "ANCHOR_FREE_DFL_WEIGHT", 1.5))
        )
        positive_area = ((target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])).clamp(min=0)
        stats = {
            "total": float(total.detach()), "box": float(box_loss.detach()), "obj": 0.0, "noobj": 0.0,
            "cls": float(cls_loss.detach()), "dfl": float(dfl_loss.detach()),
            "positive_total": float(foreground.sum().detach()),
            "positive_small": float((foreground & (positive_area < float(getattr(self.config, "SMALL_OBJ_AREA", 32 ** 2)))).sum().detach()),
        }
        positive_classes = target_scores.argmax(-1)
        for cls_id in range(int(getattr(self.config, "NUM_CLASSES", 0))):
            stats[f"positive_class_{cls_id}"] = float((foreground & (positive_classes == cls_id)).sum().detach())
        self.last_components = stats
        return total, stats


def decode_anchor_free(config, predictions, conf_thresh=None, nms_thresh=None, max_det=None):
    conf_thresh = float(config.CONF_THRESH if conf_thresh is None else conf_thresh)
    nms_thresh = float(config.NMS_THRESH if nms_thresh is None else nms_thresh)
    max_det = int(config.MAX_DET if max_det is None else max_det)
    reg_max = int(getattr(config, "ANCHOR_FREE_REG_MAX", 16))
    cls_logits, reg_logits = flatten_predictions(predictions, reg_max)
    points, strides = make_anchor_points(predictions, get_anchor_free_strides(config), cls_logits.dtype, cls_logits.device)
    boxes, scores = decode_boxes(reg_logits, points, strides, reg_max), cls_logits.sigmoid()
    detections, pre_topk = [], int(getattr(config, "ANCHOR_FREE_PRE_NMS_TOPK", 3000))
    for b in range(scores.shape[0]):
        flat = scores[b].flatten()
        values, indices = flat.topk(min(pre_topk, flat.numel()))
        keep = values >= conf_thresh
        values, indices = values[keep], indices[keep]
        if values.numel() == 0:
            detections.append(np.zeros((0, 6), dtype=np.float32))
            continue
        classes = indices.remainder(config.NUM_CLASSES)
        point_ids = torch.div(indices, config.NUM_CLASSES, rounding_mode="floor")
        selected = boxes[b, point_ids]
        kept = batched_nms(selected, values, classes, nms_thresh)[:max_det]
        selected, values, classes = selected[kept], values[kept], classes[kept]
        xywh = torch.cat(((selected[:, :2] + selected[:, 2:]) / 2, selected[:, 2:] - selected[:, :2]), dim=-1)
        detections.append(torch.cat((xywh, values[:, None], classes[:, None].to(xywh.dtype)), dim=-1).detach().float().cpu().numpy())
    return detections
