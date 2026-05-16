import torch
import torch.nn as nn
import torch.nn.functional as F

from models.geometry import bbox_iou_matrix_xywh, bbox_iou_xywh, weighted_mean
from models.losses import SigmoidFocalLoss


def decode_boxes_to_absolute(pred_boxes, anchors, stride, box_decode_range=2.0):
    """Decode predicted box offsets to absolute coordinates.

    box_decode_range controls the reach of each grid cell:
      1.0 → sigmoid       → [0, 1]      (legacy, single-cell)
      2.0 → sigmoid*2-0.5 → [-0.5, 1.5] (neighbor-cell compatible)
    """
    grid_h, grid_w = pred_boxes.shape[1], pred_boxes.shape[2]
    device = pred_boxes.device
    dtype = pred_boxes.dtype
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=dtype),
        torch.arange(grid_w, device=device, dtype=dtype),
        indexing="ij",
    )
    grid_x = grid_x.contiguous().view(1, grid_h, grid_w, 1)
    grid_y = grid_y.contiguous().view(1, grid_h, grid_w, 1)
    anchor_tensor = anchors.contiguous().view(1, 1, 1, 3, 2).to(device=device, dtype=dtype)
    half = (box_decode_range - 1.0) / 2.0
    x = (torch.sigmoid(pred_boxes[..., 0]) * box_decode_range - half + grid_x) * stride
    y = (torch.sigmoid(pred_boxes[..., 1]) * box_decode_range - half + grid_y) * stride
    w = torch.exp(torch.clamp(pred_boxes[..., 2], min=-8.0, max=8.0)) * anchor_tensor[..., 0]
    h = torch.exp(torch.clamp(pred_boxes[..., 3], min=-8.0, max=8.0)) * anchor_tensor[..., 1]
    return torch.stack([x, y, w, h], dim=-1)


def _focal_bce_unreduced(logits, targets, alpha, gamma):
    """Per-element focal BCE loss (for hard negative mining)."""
    bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    prob = torch.sigmoid(logits.float())
    p_t = targets.float() * prob + (1.0 - targets.float()) * (1.0 - prob)
    alpha_t = targets.float() * alpha + (1.0 - targets.float()) * (1.0 - alpha)
    return alpha_t * torch.pow(1.0 - p_t, gamma) * bce


class YOLOv3AnchorLossForV8Head(nn.Module):
    """Anchor-based YOLO loss with ratio matching, neighbor cells, and HNM."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.anchors = torch.tensor(config.ANCHORS, dtype=torch.float32)
        self.num_classes = config.NUM_CLASSES
        self.strides = config.STRIDES
        self.box_weight = config.BOX_WEIGHT_BASE
        self.obj_weight = config.OBJ_WEIGHT_BASE
        self.noobj_weight = config.NOOBJ_WEIGHT_BASE
        self.cls_weight = config.CLS_WEIGHT_BASE
        self.focal_alpha = config.FOCAL_ALPHA
        self.focal_gamma = config.FOCAL_GAMMA
        self.focal_loss = SigmoidFocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="mean")
        self.size_weights = {"small": 1.0, "medium": 1.0, "large": 1.0}
        self.last_components = {"total": 0.0, "box": 0.0, "obj": 0.0, "noobj": 0.0, "cls": 0.0}

    def set_epoch_weights(self, epoch):
        weights = self.config.get_dynamic_weights(epoch)
        self.box_weight = weights["box_weight"]
        self.obj_weight = weights["obj_weight"]
        self.noobj_weight = weights["noobj_weight"]
        self.cls_weight = weights["cls_weight"]
        self.size_weights = weights.get("size_weights", self.size_weights)
        return weights["phase"]

    def _get_size_weight(self, width, height):
        area = float(width * height)
        if area >= self.config.LARGE_OBJ_AREA:
            return self.size_weights["large"]
        if area >= self.config.SMALL_OBJ_AREA:
            return self.size_weights["medium"]
        return self.size_weights["small"]

    def _match_anchors_by_ratio(self, tw, th, prepared_scales):
        """Ratio-based anchor matching (Temple-style).

        Returns list of (ratio, scale_idx, anchor_idx) sorted best-first.
        """
        candidates = []
        for scale_idx, scale_data in enumerate(prepared_scales):
            for anchor_idx in range(3):
                aw, ah = scale_data["anchors"][anchor_idx]
                rw = max(tw / (aw + 1e-6), aw / (tw + 1e-6))
                rh = max(th / (ah + 1e-6), ah / (th + 1e-6))
                candidates.append((max(rw, rh), scale_idx, anchor_idx))
        candidates.sort(key=lambda item: item[0])
        return candidates

    def _neighbor_offsets(self, cx, cy, gw, gh):
        """Return grid offsets for neighbor-cell assignment."""
        gi, gj = int(cx), int(cy)
        offsets = [(gi, gj)]
        assign_neighbor = getattr(self.config, "ASSIGN_NEIGHBOR_CELLS", True)
        if not assign_neighbor:
            return offsets
        if cx - gi < 0.5 and gi > 0:
            offsets.append((gi - 1, gj))
        if (gi + 1) - cx < 0.5 and gi < gw - 1:
            offsets.append((gi + 1, gj))
        if cy - gj < 0.5 and gj > 0:
            offsets.append((gi, gj - 1))
        if (gj + 1) - cy < 0.5 and gj < gh - 1:
            offsets.append((gi, gj + 1))
        return offsets

    def forward(self, predictions, targets):
        config = self.config
        device = predictions[0].device
        total_loss = torch.zeros((), device=device)
        component_sums = {name: torch.zeros((), device=device) for name in ("box", "obj", "noobj", "cls")}
        batch_size = predictions[0].shape[0]
        prepared_scales = []
        box_decode_range = float(getattr(config, "BOX_DECODE_RANGE", 2.0))
        ratio_thresh = float(getattr(config, "ANCHOR_MATCH_RATIO_THRESH", 3.5))
        hard_neg_ratio = int(getattr(config, "HARD_NEG_RATIO", 30))
        hard_neg_min = int(getattr(config, "HARD_NEG_MIN", 512))

        for i, pred in enumerate(predictions):
            _, _, grid_h, grid_w = pred.shape
            pred = pred.contiguous().permute(0, 2, 3, 1).contiguous().reshape(batch_size, grid_h, grid_w, 3, -1)
            prepared_scales.append(
                {
                    "pred_boxes": pred[..., :4],
                    "pred_obj": pred[..., 4],
                    "pred_cls": pred[..., 5:],
                    "target_boxes_abs": torch.zeros_like(pred[..., :4]),
                    "target_obj": torch.zeros_like(pred[..., 4]),
                    "target_cls": torch.zeros_like(pred[..., 5:]),
                    "target_match_ratio": torch.full_like(pred[..., 4], float("inf")),
                    "target_scale_weight": torch.ones_like(pred[..., 4]),
                    "grid_h": grid_h,
                    "grid_w": grid_w,
                    "stride": self.strides[i],
                    "anchors": self.anchors[i].to(device),
                }
            )

        gt_boxes_abs_by_batch = [[] for _ in range(batch_size)]
        for b in range(batch_size):
            if len(targets[b]) == 0:
                continue
            for target in targets[b].to(device):
                cls_id = int(target[0].item())
                tx = target[1]
                ty = target[2]
                tw = target[3] * config.IMG_SIZE
                th = target[4] * config.IMG_SIZE
                gt_boxes_abs_by_batch[b].append(torch.stack([tx * config.IMG_SIZE, ty * config.IMG_SIZE, tw, th]))
                size_weight = self._get_size_weight(tw.item(), th.item())

                candidates = self._match_anchors_by_ratio(tw, th, prepared_scales)
                for ratio, scale_idx, anchor_idx in candidates:
                    if ratio >= ratio_thresh and len([m for m in candidates if m[0] < ratio_thresh]) > 0:
                        continue
                    scale_data = prepared_scales[scale_idx]
                    gx = tx * scale_data["grid_w"]
                    gy = ty * scale_data["grid_h"]
                    gw, gh = scale_data["grid_w"], scale_data["grid_h"]

                    for grid_x, grid_y in self._neighbor_offsets(gx, gy, gw, gh):
                        if not (0 <= grid_x < gw and 0 <= grid_y < gh):
                            continue
                        if scale_data["target_match_ratio"][b, grid_y, grid_x, anchor_idx] <= ratio:
                            continue
                        scale_data["target_boxes_abs"][b, grid_y, grid_x, anchor_idx] = torch.stack(
                            [tx * config.IMG_SIZE, ty * config.IMG_SIZE, tw, th]
                        )
                        scale_data["target_obj"][b, grid_y, grid_x, anchor_idx] = 1.0
                        scale_data["target_cls"][b, grid_y, grid_x, anchor_idx, cls_id] = 1.0
                        scale_data["target_match_ratio"][b, grid_y, grid_x, anchor_idx] = ratio
                        scale_data["target_scale_weight"][b, grid_y, grid_x, anchor_idx] = size_weight

        gt_boxes_abs_by_batch = [
            torch.stack(sample_boxes).to(device=device, dtype=predictions[0].dtype)
            if sample_boxes
            else torch.zeros((0, 4), device=device, dtype=predictions[0].dtype)
            for sample_boxes in gt_boxes_abs_by_batch
        ]

        for scale_data in prepared_scales:
            pred_boxes = scale_data["pred_boxes"]
            pred_obj = scale_data["pred_obj"]
            pred_cls = scale_data["pred_cls"]
            target_obj = scale_data["target_obj"]
            target_cls = scale_data["target_cls"]
            target_boxes_abs = scale_data["target_boxes_abs"]
            target_scale_weight = scale_data["target_scale_weight"]

            pred_boxes_abs = decode_boxes_to_absolute(pred_boxes, scale_data["anchors"], scale_data["stride"], box_decode_range)
            obj_mask = target_obj > 0.5

            # Ignore mask: predictions with high IoU to any GT (not positive) are ignored
            ignore_mask = torch.zeros_like(target_obj, dtype=torch.bool)
            for b in range(batch_size):
                gt_boxes_abs = gt_boxes_abs_by_batch[b]
                if gt_boxes_abs.numel() == 0:
                    continue
                flat_pred_boxes = pred_boxes_abs[b].contiguous().reshape(-1, 4)
                max_iou = bbox_iou_matrix_xywh(flat_pred_boxes, gt_boxes_abs).max(dim=1).values
                ignore_mask[b] = max_iou.contiguous().view(scale_data["grid_h"], scale_data["grid_w"], 3) >= config.NOOBJ_IGNORE_IOU
            noobj_mask = (target_obj <= 0.5) & (~ignore_mask)

            # Positive losses
            if obj_mask.any():
                positive_weights = target_scale_weight[obj_mask]
                ciou = bbox_iou_xywh(pred_boxes_abs[obj_mask], target_boxes_abs[obj_mask], ciou=True)
                box_loss = weighted_mean(1.0 - ciou, positive_weights)
                obj_loss = self.focal_loss(pred_obj[obj_mask], target_obj[obj_mask], sample_weight=positive_weights)
                cls_loss = self.focal_loss(pred_cls[obj_mask], target_cls[obj_mask], sample_weight=positive_weights)
            else:
                box_loss = torch.zeros((), device=device)
                obj_loss = torch.zeros((), device=device)
                cls_loss = torch.zeros((), device=device)

            # Hard negative mining for noobj loss
            if noobj_mask.any():
                neg_losses = _focal_bce_unreduced(pred_obj[noobj_mask], target_obj[noobj_mask], self.focal_alpha, self.focal_gamma)
                pos_count = int(obj_mask.sum().item())
                k = max(hard_neg_min, hard_neg_ratio * max(pos_count, 1))
                k = min(k, neg_losses.numel())
                if k > 0:
                    neg_losses = torch.topk(neg_losses.flatten(), k=k, largest=True).values
                noobj_loss = neg_losses.mean()
            else:
                noobj_loss = torch.zeros((), device=device)

            scale_loss = self.box_weight * box_loss + self.obj_weight * obj_loss + self.noobj_weight * noobj_loss + self.cls_weight * cls_loss
            if torch.isfinite(scale_loss):
                total_loss = total_loss + scale_loss
                component_sums["box"] = component_sums["box"] + box_loss.detach()
                component_sums["obj"] = component_sums["obj"] + obj_loss.detach()
                component_sums["noobj"] = component_sums["noobj"] + noobj_loss.detach()
                component_sums["cls"] = component_sums["cls"] + cls_loss.detach()

        stats = {name: float(value.item()) for name, value in component_sums.items()}
        stats["total"] = float(total_loss.detach().item())
        self.last_components = stats
        return total_loss, stats
