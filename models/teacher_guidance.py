import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_teacher_target_map(config, targets, img_size, device, dtype):
    batch_size = len(targets)
    target_map = torch.zeros((batch_size, 1, img_size, img_size), device=device, dtype=dtype)
    foreground_mask = torch.zeros((batch_size, 1, img_size, img_size), device=device, dtype=torch.bool)
    sigma = max(config.FEATURE_HEATMAP_SIGMA, 1e-3)
    box_fill_value = float(np.clip(config.FEATURE_BOX_FILL_VALUE, 0.0, 1.0))
    core_fill_value = float(np.clip(config.FEATURE_CORE_FILL_VALUE, box_fill_value, 1.0))
    core_ratio = float(np.clip(config.FEATURE_CORE_RATIO, 0.1, 1.0))

    for batch_idx, sample_targets in enumerate(targets):
        if len(sample_targets) == 0:
            continue
        for target in sample_targets:
            if target.shape[0] < 5:
                continue
            x_center = float(target[1].item() * img_size)
            y_center = float(target[2].item() * img_size)
            width = max(float(target[3].item() * img_size), 1.0)
            height = max(float(target[4].item() * img_size), 1.0)
            box_fill = box_fill_value
            core_fill = core_fill_value
            local_core_ratio = core_ratio
            if width * height >= config.FEATURE_LARGE_OBJECT_AREA:
                box_fill = min(1.0, box_fill + config.FEATURE_LARGE_BOX_FILL_BOOST)
                core_fill = min(1.0, core_fill + config.FEATURE_LARGE_CORE_FILL_BOOST)
                local_core_ratio = min(1.0, local_core_ratio + config.FEATURE_LARGE_CORE_RATIO_BOOST)

            x1 = max(0, int(np.floor(x_center - width / 2)))
            y1 = max(0, int(np.floor(y_center - height / 2)))
            x2 = min(img_size, int(np.ceil(x_center + width / 2)))
            y2 = min(img_size, int(np.ceil(y_center + height / 2)))
            if x2 <= x1 or y2 <= y1:
                continue
            target_map[batch_idx, :, y1:y2, x1:x2] = torch.maximum(
                target_map[batch_idx, :, y1:y2, x1:x2],
                torch.full_like(target_map[batch_idx, :, y1:y2, x1:x2], box_fill),
            )
            foreground_mask[batch_idx, :, y1:y2, x1:x2] = True

            core_w = max(width * local_core_ratio, 1.0)
            core_h = max(height * local_core_ratio, 1.0)
            cx1 = max(0, int(np.floor(x_center - core_w / 2)))
            cy1 = max(0, int(np.floor(y_center - core_h / 2)))
            cx2 = min(img_size, int(np.ceil(x_center + core_w / 2)))
            cy2 = min(img_size, int(np.ceil(y_center + core_h / 2)))
            if cx2 > cx1 and cy2 > cy1:
                target_map[batch_idx, :, cy1:cy2, cx1:cx2] = torch.maximum(
                    target_map[batch_idx, :, cy1:cy2, cx1:cx2],
                    torch.full_like(target_map[batch_idx, :, cy1:cy2, cx1:cx2], core_fill),
                )

            yy, xx = torch.meshgrid(
                torch.arange(y1, y2, device=device, dtype=dtype),
                torch.arange(x1, x2, device=device, dtype=dtype),
                indexing="ij",
            )
            gaussian = torch.exp(-(((xx - x_center) / max(width * sigma, 1.0)) ** 2 + ((yy - y_center) / max(height * sigma, 1.0)) ** 2) / 2.0)
            target_map[batch_idx, 0, y1:y2, x1:x2] = torch.maximum(target_map[batch_idx, 0, y1:y2, x1:x2], gaussian)
    return target_map.clamp(0.0, 1.0), foreground_mask


def compute_teacher_guidance_loss(config, teacher_feature, targets, stage_settings=None):
    target_map, foreground_mask = build_teacher_target_map(config, targets, teacher_feature.shape[-1], teacher_feature.device, teacher_feature.dtype)
    background_mask = ~foreground_mask
    heatmap_loss = F.binary_cross_entropy(teacher_feature.clamp(1e-4, 1.0 - 1e-4), target_map)
    fg_mean = teacher_feature[foreground_mask].mean() if foreground_mask.any() else torch.zeros((), device=teacher_feature.device, dtype=teacher_feature.dtype)
    bg_mean = teacher_feature[background_mask].mean() if background_mask.any() else torch.zeros((), device=teacher_feature.device, dtype=teacher_feature.dtype)
    contrast_loss = F.relu(config.FEATURE_FOREGROUND_TARGET - fg_mean) + F.relu(bg_mean - config.FEATURE_BACKGROUND_TARGET)
    sparsity_loss = bg_mean
    tv_loss = torch.abs(teacher_feature[:, :, 1:, :] - teacher_feature[:, :, :-1, :]).mean() + torch.abs(teacher_feature[:, :, :, 1:] - teacher_feature[:, :, :, :-1]).mean()
    weights = config.get_teacher_guidance_weights(stage_settings=stage_settings)
    total = weights["heatmap"] * heatmap_loss + weights["contrast"] * contrast_loss + weights["sparsity"] * sparsity_loss + weights["tv"] * tv_loss
    stats = {
        "feature_heatmap": float(heatmap_loss.detach().item()),
        "feature_contrast": float(contrast_loss.detach().item()),
        "feature_sparsity": float(sparsity_loss.detach().item()),
        "feature_tv": float(tv_loss.detach().item()),
        "feature_total": float(total.detach().item()),
    }
    return total, stats


def enhance_feature_for_display(feature_map):
    feature_map = np.asarray(feature_map, dtype=np.float32)
    low = np.percentile(feature_map, 2)
    high = np.percentile(feature_map, 98)
    if high - low < 1e-6:
        return np.zeros_like(feature_map)
    feature_map = np.clip((feature_map - low) / (high - low), 0.0, 1.0)
    return np.power(feature_map, 0.8)


def build_foreground_mask_from_targets(config, targets, img_size, device, dtype):
    """Build a binary foreground mask from detection targets (no Gaussian).

    Returns (foreground_mask, target_map) where target_map is a simple
    box-fill + Gaussian heatmap used for lightweight region-aware losses.
    """
    batch_size = len(targets)
    foreground_mask = torch.zeros((batch_size, 1, img_size, img_size), device=device, dtype=torch.bool)
    target_map = torch.zeros((batch_size, 1, img_size, img_size), device=device, dtype=dtype)

    for batch_idx, sample_targets in enumerate(targets):
        if len(sample_targets) == 0:
            continue
        for target in sample_targets:
            if target.shape[0] < 5:
                continue
            x_center = float(target[1].item() * img_size)
            y_center = float(target[2].item() * img_size)
            width = max(float(target[3].item() * img_size), 1.0)
            height = max(float(target[4].item() * img_size), 1.0)

            x1 = max(0, int(np.floor(x_center - width / 2)))
            y1 = max(0, int(np.floor(y_center - height / 2)))
            x2 = min(img_size, int(np.ceil(x_center + width / 2)))
            y2 = min(img_size, int(np.ceil(y_center + height / 2)))
            if x2 <= x1 or y2 <= y1:
                continue

            foreground_mask[batch_idx, :, y1:y2, x1:x2] = True
            target_map[batch_idx, :, y1:y2, x1:x2] = 1.0

    return foreground_mask, target_map


def compute_teacher_guidance_loss_v3(config, det_feature, gray_image, targets):
    """V3 guidance loss for residual+gate teacher output.

    The V3 teacher outputs ``det_feature = gray + scale * gate * residual``.
    This loss enforces two constraints:

    1. **Background identity**: in regions without objects,
       ``det_feature`` should equal ``gray`` — the teacher must not
       modify the background.
    2. **Gradient consistency**: in background regions, the spatial
       gradients of ``det_feature`` should match those of ``gray``,
       preserving edges and texture for the detector.

    Both losses are applied *outside* object boxes only, leaving the
    teacher free to enhance foreground regions.
    """
    img_size = det_feature.shape[-1]
    device = det_feature.device
    dtype = det_feature.dtype

    foreground_mask, _ = build_foreground_mask_from_targets(config, targets, img_size, device, dtype)
    bg_mask = (~foreground_mask).float()

    # 1. Background identity: |det_feature - gray| weighted by background mask
    bg_diff = (det_feature - gray_image).abs()
    bg_identity = (bg_diff * bg_mask).sum() / bg_mask.sum().clamp(min=1.0)

    # 2. Gradient consistency in background
    dx_f = det_feature[:, :, :, 1:] - det_feature[:, :, :, :-1]
    dx_g = gray_image[:, :, :, 1:] - gray_image[:, :, :, :-1]
    dy_f = det_feature[:, :, 1:, :] - det_feature[:, :, :-1, :]
    dy_g = gray_image[:, :, 1:, :] - gray_image[:, :, :-1, :]

    fg_mask = foreground_mask.float()
    wx = 1.0 - 0.5 * torch.maximum(fg_mask[:, :, :, 1:], fg_mask[:, :, :, :-1])
    wy = 1.0 - 0.5 * torch.maximum(fg_mask[:, :, 1:, :], fg_mask[:, :, :-1, :])

    grad_consistency = (dx_f - dx_g).abs().mul(wx).mean() + (dy_f - dy_g).abs().mul(wy).mean()

    bg_weight = float(getattr(config, "TEACHER_V3_BG_IDENTITY_WEIGHT", 0.04))
    grad_weight = float(getattr(config, "TEACHER_V3_GRAD_CONSISTENCY_WEIGHT", 0.02))

    total = bg_weight * bg_identity + grad_weight * grad_consistency
    stats = {
        "feature_heatmap": 0.0,
        "feature_contrast": 0.0,
        "feature_sparsity": 0.0,
        "feature_tv": 0.0,
        "feature_bg_identity": float(bg_identity.detach().item()),
        "feature_grad_consistency": float(grad_consistency.detach().item()),
        "feature_total": float(total.detach().item()),
    }
    return total, stats


class FeatureDistillationLoss(nn.Module):
    """Multi-scale feature distillation from teacher to detector.

    Projects teacher intermediate features (feat_scale2/4/8) to match
    detector FPN features (s8/s16/s32) via learnable 1x1 convolutions,
    then computes MSE at each scale.

    This is a TRAINING-ONLY module.  At deployment the detector works
    from 1-channel input alone.
    """

    def __init__(self, teacher_feat_channels, detector_feat_channels):
        super().__init__()
        t8, t4, t2 = teacher_feat_channels
        d8, d16, d32 = detector_feat_channels
        self.proj_s8 = nn.Conv2d(t8, d8, 1)
        self.proj_s16 = nn.Conv2d(t4, d16, 1)
        self.proj_s32 = nn.Conv2d(t2, d32, 1)

    def forward(self, teacher_aux, det_features):
        loss_s8 = F.mse_loss(
            self.proj_s8(teacher_aux["feat_scale8"]),
            det_features["s8"],
        )
        t_s16 = F.adaptive_avg_pool2d(teacher_aux["feat_scale4"], det_features["s16"].shape[-2:])
        loss_s16 = F.mse_loss(self.proj_s16(t_s16), det_features["s16"])
        t_s32 = F.adaptive_avg_pool2d(teacher_aux["feat_scale2"], det_features["s32"].shape[-2:])
        loss_s32 = F.mse_loss(self.proj_s32(t_s32), det_features["s32"])
        total = loss_s8 + loss_s16 + loss_s32
        stats = {
            "distill_s8": float(loss_s8.detach().item()),
            "distill_s16": float(loss_s16.detach().item()),
            "distill_s32": float(loss_s32.detach().item()),
            "distill_total": float(total.detach().item()),
        }
        return total, stats


def build_feature_distillation_loss(config):
    """Create FeatureDistillationLoss with channels inferred from config."""
    arch = str(getattr(config, "TEACHER_ARCH", "convteacher_v2")).strip().lower()

    # Teacher feature channels  {feat_scale8, feat_scale4, feat_scale2}
    if arch in {"convteacher_v3", "v3"}:
        c = int(getattr(config, "TEACHER_V3_BASE_CHANNELS", 24))
        teacher_chs = (c, c * 2, c)       # feat_scale8: c, feat_scale4: 2c, feat_scale2: c
    elif arch in {"convteacher_unet", "unet"}:
        c = int(getattr(config, "TEACHER_UNET_BASE_CHANNELS", 16))
        teacher_chs = (c * 8, c * 4, c * 2)  # feat_scale8: 8c, feat_scale4: 4c, feat_scale2: 2c
    else:
        # V2 — uses bottleneck features
        c = int(getattr(config, "TEACHER_V2_BASE_CHANNELS", 24))
        teacher_chs = (c, c * 2, c)

    # Detector feature channels  {s8, s16, s32}
    head_type = str(getattr(config, "DETECTOR_HEAD_TYPE", "yolov8_anchor")).strip().lower()
    if head_type in {"light", "yolo_light"}:
        lc = int(getattr(config, "YOLO_LIGHT_BASE_CH", 16))
        detector_chs = (lc * 8, lc * 8, lc * 8)  # all use 8c at different spatial sizes
    else:
        dc = int(getattr(config, "YOLOV8_BASE_CHANNELS", 32))
        detector_chs = (dc * 8, dc * 8, dc * 8)

    return FeatureDistillationLoss(teacher_chs, detector_chs)
