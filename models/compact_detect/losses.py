import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian2d(radius, sigma, device, dtype):
    diameter = 2 * radius + 1
    coords = torch.arange(diameter, device=device, dtype=dtype) - radius
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))


def draw_gaussian(heatmap, center_x, center_y, radius):
    height, width = heatmap.shape[-2:]
    radius = int(max(radius, 0))
    sigma = max(diameter_sigma(radius), 1e-6)
    kernel = gaussian2d(radius, sigma, heatmap.device, heatmap.dtype)
    left, right = min(center_x, radius), min(width - center_x - 1, radius)
    top, bottom = min(center_y, radius), min(height - center_y - 1, radius)
    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return
    masked_heatmap = heatmap[center_y - top:center_y + bottom + 1, center_x - left:center_x + right + 1]
    masked_kernel = kernel[radius - top:radius + bottom + 1, radius - left:radius + right + 1]
    torch.maximum(masked_heatmap, masked_kernel, out=masked_heatmap)


def diameter_sigma(radius):
    return (2 * radius + 1) / 6.0


class CenterTargetBuilder:
    def __init__(self, config):
        self.config = config
        self.stride = int(getattr(config, "OUTPUT_STRIDE", 4))

    def __call__(self, targets, output_hw, device, dtype):
        batch_size = len(targets)
        out_h, out_w = output_hw
        num_classes = int(self.config.NUM_CLASSES)
        heatmap = torch.zeros((batch_size, num_classes, out_h, out_w), device=device, dtype=dtype)
        wh = torch.zeros((batch_size, 2, out_h, out_w), device=device, dtype=dtype)
        offset = torch.zeros((batch_size, 2, out_h, out_w), device=device, dtype=dtype)
        mask = torch.zeros((batch_size, 1, out_h, out_w), device=device, dtype=dtype)

        for batch_idx, sample_targets in enumerate(targets):
            if sample_targets is None or sample_targets.numel() == 0:
                continue
            sample_targets = sample_targets.to(device=device, dtype=dtype)
            for item in sample_targets:
                cls_id = int(item[0].item())
                if cls_id < 0 or cls_id >= num_classes:
                    continue
                cx = float(item[1].item()) * out_w
                cy = float(item[2].item()) * out_h
                bw = float(item[3].item()) * float(self.config.IMG_SIZE)
                bh = float(item[4].item()) * float(self.config.IMG_SIZE)
                if bw <= 0 or bh <= 0:
                    continue
                grid_x = min(max(int(cx), 0), out_w - 1)
                grid_y = min(max(int(cy), 0), out_h - 1)
                radius_base = min(bw / self.stride, bh / self.stride) * float(getattr(self.config, "HEATMAP_RADIUS_SCALE", 0.35))
                radius = max(int(math.floor(radius_base)), int(getattr(self.config, "MIN_HEATMAP_RADIUS", 1)))
                draw_gaussian(heatmap[batch_idx, cls_id], grid_x, grid_y, radius)
                existing_area = wh[batch_idx, 0, grid_y, grid_x] * wh[batch_idx, 1, grid_y, grid_x]
                current_area = bw * bh
                if mask[batch_idx, 0, grid_y, grid_x] == 0 or current_area < existing_area:
                    wh[batch_idx, :, grid_y, grid_x] = torch.tensor([bw, bh], device=device, dtype=dtype)
                    offset[batch_idx, :, grid_y, grid_x] = torch.tensor([cx - grid_x, cy - grid_y], device=device, dtype=dtype)
                    mask[batch_idx, 0, grid_y, grid_x] = 1.0

        return {"heatmap": heatmap, "wh": wh, "offset": offset, "mask": mask}


class CenterDetectionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.target_builder = CenterTargetBuilder(config)

    @staticmethod
    def focal_heatmap_loss(logits, target):
        pred = torch.sigmoid(logits).clamp(min=1e-4, max=1.0 - 1e-4)
        pos = target.eq(1).to(logits.dtype)
        neg = target.lt(1).to(logits.dtype)
        neg_weights = torch.pow(1.0 - target, 4)
        pos_loss = torch.log(pred) * torch.pow(1.0 - pred, 2) * pos
        neg_loss = torch.log(1.0 - pred) * torch.pow(pred, 2) * neg_weights * neg
        num_pos = pos.sum().clamp(min=1.0)
        return -(pos_loss.sum() + neg_loss.sum()) / num_pos

    @staticmethod
    def masked_l1(pred, target, mask):
        mask = mask.expand_as(pred)
        denom = mask.sum().clamp(min=1.0)
        return F.l1_loss(pred * mask, target * mask, reduction="sum") / denom

    def forward(self, pred, targets):
        _, _, out_h, out_w = pred["heatmap"].shape
        target = self.target_builder(targets, (out_h, out_w), pred["heatmap"].device, pred["heatmap"].dtype)
        heat_loss = self.focal_heatmap_loss(pred["heatmap"], target["heatmap"])
        wh_loss = self.masked_l1(pred["wh"], target["wh"], target["mask"])
        offset_loss = self.masked_l1(pred["offset"], target["offset"], target["mask"])
        total = (
            heat_loss * float(getattr(self.config, "HEATMAP_LOSS_WEIGHT", 1.0))
            + wh_loss * float(getattr(self.config, "WH_LOSS_WEIGHT", 0.08))
            + offset_loss * float(getattr(self.config, "OFFSET_LOSS_WEIGHT", 1.0))
        )
        return total, {
            "total": float(total.detach().item()),
            "heatmap": float(heat_loss.detach().item()),
            "wh": float(wh_loss.detach().item()),
            "offset": float(offset_loss.detach().item()),
        }
