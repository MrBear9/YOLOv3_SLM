import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _max_normalize(x, eps):
    return x / (x.amax(dim=(2, 3), keepdim=True) + eps)


def _pearson_abs(a, b, eps):
    batch_size = a.shape[0]
    a_flat = a.reshape(batch_size, -1)
    b_flat = b.reshape(batch_size, -1)
    a_centered = a_flat - a_flat.mean(dim=1, keepdim=True)
    b_centered = b_flat - b_flat.mean(dim=1, keepdim=True)
    a_std = a_centered.std(dim=1, keepdim=True, unbiased=False)
    b_std = b_centered.std(dim=1, keepdim=True, unbiased=False)
    corr = (a_centered * b_centered).mean(dim=1, keepdim=True) / (a_std * b_std + eps)
    return corr.abs().mean()


def _ssim_similarity(a, b, eps):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_a = F.avg_pool2d(a, 3, 1, 1)
    mu_b = F.avg_pool2d(b, 3, 1, 1)
    sigma_a = F.avg_pool2d(a * a, 3, 1, 1) - mu_a * mu_a
    sigma_b = F.avg_pool2d(b * b, 3, 1, 1) - mu_b * mu_b
    sigma_ab = F.avg_pool2d(a * b, 3, 1, 1) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2) + eps
    )
    return torch.clamp(ssim_map.mean(), min=0.0, max=1.0)


def teacher_cipher_loss(config, teacher_aux):
    weight = float(getattr(config, "TEACHER_CIPHER_LOSS_WEIGHT", 0.0))
    if weight <= 0 or teacher_aux is None:
        device = teacher_aux["det_feature"].device if teacher_aux is not None else "cpu"
        zero = torch.zeros((), device=device)
        return zero, {"cipher": 0.0, "cipher_corr": 0.0, "cipher_ssim": 0.0, "cipher_std": 0.0, "cipher_grad": 0.0}

    eps = float(getattr(config, "OPTICAL_FIELD_EPS", 1e-8))
    norm_eps = float(getattr(config, "OPTICAL_NORM_EPS", 1e-6))
    feature = teacher_aux["det_feature"].float()
    gray = teacher_aux["gray"].float().clamp(min=0)
    feature_view = _max_normalize(feature, norm_eps)
    gray_view = _max_normalize(gray, norm_eps)

    corr_abs = _pearson_abs(feature_view, gray_view, eps)
    ssim_sim = _ssim_similarity(feature_view, gray_view, eps)
    corr_target = float(getattr(config, "TEACHER_CIPHER_CORR_TARGET", 0.18))
    ssim_target = float(getattr(config, "TEACHER_CIPHER_SSIM_TARGET", 0.24))
    privacy_raw = F.relu(corr_abs - corr_target) + F.relu(ssim_sim - ssim_target)

    feat_std = feature_view.std(dim=(2, 3), unbiased=False).mean()
    grad_x = (feature_view[:, :, :, 1:] - feature_view[:, :, :, :-1]).abs().mean()
    grad_y = (feature_view[:, :, 1:, :] - feature_view[:, :, :-1, :]).abs().mean()
    grad_mean = 0.5 * (grad_x + grad_y)
    std_floor = float(getattr(config, "TEACHER_CIPHER_STD_FLOOR", 0.08))
    grad_floor = float(getattr(config, "TEACHER_CIPHER_GRAD_FLOOR", 0.015))
    structure_weight = float(getattr(config, "TEACHER_CIPHER_STRUCTURE_WEIGHT", 0.25))
    structure_raw = F.relu(std_floor - feat_std) + F.relu(grad_floor - grad_mean)

    raw = privacy_raw + structure_weight * structure_raw
    return raw * weight, {
        "cipher": float(raw.detach().item()),
        "cipher_corr": float(corr_abs.detach().item()),
        "cipher_ssim": float(ssim_sim.detach().item()),
        "cipher_std": float(feat_std.detach().item()),
        "cipher_grad": float(grad_mean.detach().item()),
    }


def enhance_feature_for_display(feature_map):
    feature_map = np.asarray(feature_map, dtype=np.float32)
    low = np.percentile(feature_map, 2)
    high = np.percentile(feature_map, 98)
    if high - low < 1e-6:
        return np.zeros_like(feature_map)
    feature_map = np.clip((feature_map - low) / (high - low), 0.0, 1.0)
    return np.power(feature_map, 0.8)


# =========================================================================
# Feature distillation (teacher → detector)
# =========================================================================


class FeatureDistillationLoss(nn.Module):
    """Multi-scale feature distillation from teacher to detector.

    Projects teacher intermediate features (feat_scale2/4/8) to match
    detector FPN features (s8/s16/s32) via learnable 1×1 convolutions,
    then computes MSE at each scale.  Training-only.
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
        teacher_chs = (c, c * 2, c)
    else:
        # ConvTeacher / ConvTeacherV2 (deep projection)
        # feat_scale8: c1,  feat_scale4/2: c3 = c*4
        c = int(getattr(config, "TEACHER_V2_BASE_CHANNELS", 24))
        teacher_chs = (c, c * 4, c * 4)

    # Detector feature channels  {s8, s16, s32}
    head_type = str(getattr(config, "DETECTOR_HEAD_TYPE", "yolov8_anchor")).strip().lower()
    if head_type in {"light", "yolo_light"}:
        lc = int(getattr(config, "YOLO_LIGHT_BASE_CH", 16))
        detector_chs = (lc * 8, lc * 8, lc * 8)
    elif head_type in {"light_branch", "branch_light", "yolo_light_branch"}:
        lc = int(getattr(config, "YOLO_LIGHT_BASE_CH", 16))
        detector_chs = (lc * 8, lc * 8, lc * 8)
    else:
        dc = int(getattr(config, "YOLOV8_BASE_CHANNELS", 32))
        detector_chs = (dc * 8, dc * 8, dc * 8)

    return FeatureDistillationLoss(teacher_chs, detector_chs)
