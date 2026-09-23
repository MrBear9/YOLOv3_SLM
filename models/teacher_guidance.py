import numpy as np
import torch
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


def _blur_feature(x, kernel_size):
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def _neighbor_total_variation(x):
    grad_x = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    grad_y = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    return 0.5 * (grad_x + grad_y)


def teacher_slm_cipher_loss(config, teacher_aux):
    weight = float(getattr(config, "TEACHER_SLM_CIPHER_LOSS_WEIGHT", 0.0))
    if weight <= 0 or teacher_aux is None:
        device = teacher_aux["det_feature"].device if teacher_aux is not None else "cpu"
        zero = torch.zeros((), device=device)
        return zero, {
            "slm_cipher": 0.0,
            "slm_tv": 0.0,
            "slm_hf": 0.0,
            "slm_range": 0.0,
            "slm_mean": 0.0,
            "slm_peak": 0.0,
            "slm_edge": 0.0,
        }

    eps = float(getattr(config, "OPTICAL_NORM_EPS", 1e-6))
    feature = teacher_aux["det_feature"].float()
    feature_view = _max_normalize(feature.clamp(min=0), eps)
    kernel = int(getattr(config, "TEACHER_SLM_CIPHER_BLUR_KERNEL", 15))
    low_freq = _blur_feature(feature_view, kernel)
    high_freq = feature_view - low_freq

    tv = _neighbor_total_variation(feature_view)
    hf = high_freq.abs().mean()
    tv_target = float(getattr(config, "TEACHER_SLM_CIPHER_TV_TARGET", 0.026))
    hf_target = float(getattr(config, "TEACHER_SLM_CIPHER_HF_TARGET", 0.045))
    range_floor = float(getattr(config, "TEACHER_SLM_CIPHER_RANGE_FLOOR", 0.28))
    mean_floor = float(getattr(config, "TEACHER_SLM_CIPHER_MEAN_FLOOR", 0.52))
    peak_limit = float(getattr(config, "TEACHER_SLM_CIPHER_PEAK_LIMIT", 0.88))
    edge_limit = float(getattr(config, "TEACHER_SLM_CIPHER_EDGE_LIMIT", 0.62))
    tv_weight = float(getattr(config, "TEACHER_SLM_CIPHER_TV_WEIGHT", 1.0))
    hf_weight = float(getattr(config, "TEACHER_SLM_CIPHER_HF_WEIGHT", 1.2))
    range_weight = float(getattr(config, "TEACHER_SLM_CIPHER_RANGE_WEIGHT", 0.6))
    mean_weight = float(getattr(config, "TEACHER_SLM_CIPHER_MEAN_WEIGHT", 1.4))
    peak_weight = float(getattr(config, "TEACHER_SLM_CIPHER_PEAK_WEIGHT", 0.8))
    edge_weight = float(getattr(config, "TEACHER_SLM_CIPHER_EDGE_WEIGHT", 0.4))

    spatial_range = feature_view.amax(dim=(2, 3), keepdim=True) - feature_view.amin(dim=(2, 3), keepdim=True)
    mean_val = feature_view.mean()
    peak_mean = feature_view.flatten(2).topk(k=max(1, feature_view.shape[-1] * feature_view.shape[-2] // 100), dim=2).values.mean()

    edge = max(2, int(round(min(feature_view.shape[-2:]) * 0.025)))
    edge_parts = [
        feature_view[:, :, :edge, :],
        feature_view[:, :, -edge:, :],
        feature_view[:, :, :, :edge],
        feature_view[:, :, :, -edge:],
    ]
    edge_mean = torch.cat([part.flatten(1) for part in edge_parts], dim=1).mean()

    raw = (
        tv_weight * F.relu(tv - tv_target)
        + hf_weight * F.relu(hf - hf_target)
        + range_weight * F.relu(range_floor - spatial_range.mean())
        + mean_weight * F.relu(mean_floor - mean_val)
        + peak_weight * F.relu(peak_mean - peak_limit)
        + edge_weight * F.relu(edge_mean - edge_limit)
    )
    return raw * weight, {
        "slm_cipher": raw.detach(),
        "slm_tv": tv.detach(),
        "slm_hf": hf.detach(),
        "slm_range": spatial_range.mean().detach(),
        "slm_mean": mean_val.detach(),
        "slm_peak": peak_mean.detach(),
        "slm_edge": edge_mean.detach(),
    }


def enhance_feature_for_display(feature_map):
    feature_map = np.asarray(feature_map, dtype=np.float32)
    low = np.percentile(feature_map, 2)
    high = np.percentile(feature_map, 98)
    span = high - low
    if span < 1e-6:
        # Low-contrast fallback (e.g. multi-head mean fusion): use min-max.
        f_min, f_max = feature_map.min(), feature_map.max()
        span = f_max - f_min
        if span < 1e-8:
            return np.zeros_like(feature_map)
        low, high = f_min, f_max
    feature_map = np.clip((feature_map - low) / span, 0.0, 1.0)
    return np.power(feature_map, 0.8)


# =========================================================================
# Static-phase transfer constraint for V4
# =========================================================================


def _circular_total_variation(phase):
    dx = phase[..., :, 1:] - phase[..., :, :-1]
    dy = phase[..., 1:, :] - phase[..., :-1, :]
    return 0.5 * ((1.0 - torch.cos(dx)).mean() + (1.0 - torch.cos(dy)).mean())


def teacher_phase_transfer_loss(config, teacher_aux):
    """Keep V4's conditional residual small enough for a static SLM student.

    The residual penalty is circular, so phases that differ by 2π remain
    physically identical.  It replaces the old CNN-to-detector projection
    loss whose projection parameters were outside both DDP and the optimizer.
    """
    residuals = () if teacher_aux is None else teacher_aux.get("phase_residuals", ())
    weight = float(getattr(config, "TEACHER_V4_TRANSFER_LOSS_WEIGHT", 0.0))
    if weight <= 0 or not residuals:
        device = teacher_aux["det_feature"].device if teacher_aux is not None else "cpu"
        zero = torch.zeros((), device=device)
        static_only = teacher_aux.get("static_only_batch") if teacher_aux is not None else None
        scale = teacher_aux.get("dynamic_phase_scale_rad") if teacher_aux is not None else None
        return zero, {
            "phase_transfer": 0.0,
            "phase_residual_energy": 0.0,
            "phase_residual_tv": 0.0,
            "phase_residual_scale_rad": scale.detach() if scale is not None else 0.0,
            "static_batch_fraction": static_only.detach() if static_only is not None else 0.0,
        }

    energy = torch.stack([(1.0 - torch.cos(value)).mean() for value in residuals]).mean()
    residual_tv = torch.stack([_circular_total_variation(value) for value in residuals]).mean()
    tv_weight = float(getattr(config, "TEACHER_V4_RESIDUAL_TV_WEIGHT", 0.1))
    raw = energy + tv_weight * residual_tv
    scale = teacher_aux.get("dynamic_phase_scale_rad")
    static_only = teacher_aux.get("static_only_batch")
    return raw * weight, {
        "phase_transfer": raw.detach(),
        "phase_residual_energy": energy.detach(),
        "phase_residual_tv": residual_tv.detach(),
        "phase_residual_scale_rad": scale.detach() if scale is not None else 0.0,
        "static_batch_fraction": static_only.detach() if static_only is not None else 0.0,
    }
