import numpy as np
import torch
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
