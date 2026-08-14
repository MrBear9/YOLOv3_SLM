import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.yolov8.feature_adapter import prepare_slm_detector_feature


def _interpolate_preserve_layout(x, *args, **kwargs):
    channels_last = x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last)
    out = F.interpolate(x, *args, **kwargs)
    if out.dim() != 4:
        return out
    if channels_last:
        return out.contiguous(memory_format=torch.channels_last)
    return out.contiguous()


class CompositeOpticalFeatureLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pool1 = nn.AvgPool2d(8)
        self.pool2 = nn.AvgPool2d(32)
        self.avg_pool = nn.AvgPool2d(3, 1, 1)

    def ssim_loss(self, student_feature, teacher_feature):
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu_s = self.avg_pool(student_feature)
        mu_t = self.avg_pool(teacher_feature)
        sigma_s = self.avg_pool(student_feature * student_feature) - mu_s * mu_s
        sigma_t = self.avg_pool(teacher_feature * teacher_feature) - mu_t * mu_t
        sigma_st = self.avg_pool(student_feature * teacher_feature) - mu_s * mu_t
        ssim_map = ((2 * mu_s * mu_t + c1) * (2 * sigma_st + c2)) / (
            (mu_s * mu_s + mu_t * mu_t + c1) * (sigma_s + sigma_t + c2) + self.config.OPTICAL_FIELD_EPS
        )
        return torch.clamp((1.0 - ssim_map.mean()) * 0.5, min=0.0)

    def gradient_loss(self, student_feature, teacher_feature):
        grad_s_x = student_feature[:, :, :, 1:] - student_feature[:, :, :, :-1]
        grad_t_x = teacher_feature[:, :, :, 1:] - teacher_feature[:, :, :, :-1]
        grad_s_y = student_feature[:, :, 1:, :] - student_feature[:, :, :-1, :]
        grad_t_y = teacher_feature[:, :, 1:, :] - teacher_feature[:, :, :-1, :]
        return F.l1_loss(grad_s_x, grad_t_x) + F.l1_loss(grad_s_y, grad_t_y)

    def frequency_loss(self, student_feature, teacher_feature):
        freq_s = torch.fft.fft2(student_feature.squeeze(1), norm="ortho")
        freq_t = torch.fft.fft2(teacher_feature.squeeze(1), norm="ortho")
        return F.l1_loss(torch.log1p(torch.abs(freq_s)), torch.log1p(torch.abs(freq_t)))

    def pearson_loss(self, student_feature, teacher_feature):
        batch_size = student_feature.shape[0]
        student_flat = student_feature.reshape(batch_size, -1)
        teacher_flat = teacher_feature.reshape(batch_size, -1)
        student_centered = student_flat - student_flat.mean(dim=1, keepdim=True)
        teacher_centered = teacher_flat - teacher_flat.mean(dim=1, keepdim=True)
        student_std = student_centered.std(dim=1, keepdim=True, unbiased=False)
        teacher_std = teacher_centered.std(dim=1, keepdim=True, unbiased=False)
        corr = (student_centered * teacher_centered).mean(dim=1, keepdim=True) / (
            student_std * teacher_std + self.config.OPTICAL_FIELD_EPS
        )
        return torch.clamp((1.0 - corr.mean()) * 0.5, min=0.0)

    def domain_align(self, student_feature, teacher_feature):
        if not bool(getattr(self.config, "ENABLE_FEATURE_DOMAIN_ALIGNMENT", False)):
            return student_feature, teacher_feature
        mode = str(getattr(self.config, "FEATURE_DOMAIN_ALIGN_MODE", "mean_std")).strip().lower()
        if mode in {"none", "off", "false"}:
            return student_feature, teacher_feature
        eps = self.config.OPTICAL_NORM_EPS
        if mode in {"minmax", "min_max"}:
            s_min = student_feature.amin(dim=(2, 3), keepdim=True)
            t_min = teacher_feature.amin(dim=(2, 3), keepdim=True)
            s_span = student_feature.amax(dim=(2, 3), keepdim=True) - s_min
            t_span = teacher_feature.amax(dim=(2, 3), keepdim=True) - t_min
            return (student_feature - s_min) / (s_span + eps), (teacher_feature - t_min) / (t_span + eps)
        s_mean = student_feature.mean(dim=(2, 3), keepdim=True)
        t_mean = teacher_feature.mean(dim=(2, 3), keepdim=True)
        s_std = student_feature.std(dim=(2, 3), keepdim=True, unbiased=False)
        t_std = teacher_feature.std(dim=(2, 3), keepdim=True, unbiased=False)
        return (student_feature - s_mean) / (s_std + eps), (teacher_feature - t_mean) / (t_std + eps)

    def target_roi_mask(self, targets, height, width, device, dtype):
        """Build a soft target-and-context mask from normalized YOLO boxes."""
        mask = torch.zeros((len(targets), 1, height, width), device=device, dtype=dtype)
        context_scale = max(float(getattr(self.config, "TARGET_ROI_CONTEXT_SCALE", 1.0)), 1.0)
        class_weights = getattr(self.config, "TARGET_ROI_CLASS_WEIGHTS", {})
        for batch_idx, sample_targets in enumerate(targets):
            if sample_targets is None or sample_targets.numel() == 0:
                continue
            for target in sample_targets:
                if target.numel() < 5 or target[3] <= 0 or target[4] <= 0:
                    continue
                class_id = int(target[0].item())
                weight = float(class_weights.get(class_id, 1.0))
                cx, cy = float(target[1]), float(target[2])
                half_w = float(target[3]) * context_scale * 0.5
                half_h = float(target[4]) * context_scale * 0.5
                x0 = max(0, min(width, int((cx - half_w) * width)))
                x1 = max(0, min(width, int(math.ceil((cx + half_w) * width))))
                y0 = max(0, min(height, int((cy - half_h) * height)))
                y1 = max(0, min(height, int(math.ceil((cy + half_h) * height))))
                if x1 > x0 and y1 > y0:
                    mask[batch_idx, :, y0:y1, x0:x1] = torch.maximum(
                        mask[batch_idx, :, y0:y1, x0:x1],
                        mask.new_full((1, y1 - y0, x1 - x0), weight),
                    )
        feather_kernel = max(int(getattr(self.config, "TARGET_ROI_FEATHER_KERNEL", 1)), 1)
        if feather_kernel > 1:
            if feather_kernel % 2 == 0:
                feather_kernel += 1
            mask = F.max_pool2d(mask, feather_kernel, stride=1, padding=feather_kernel // 2)
        return mask

    @staticmethod
    def masked_mse(student_feature, teacher_feature, mask):
        if mask is None or not torch.any(mask > 0):
            return student_feature.new_zeros(())
        squared_error = (student_feature - teacher_feature).square()
        return (squared_error * mask).sum() / (mask.sum() * student_feature.shape[1] + 1e-8)

    @staticmethod
    def _iter_slm_layers(student):
        """Yield (index, slm_layer) for all SLM layers across all heads.

        Handles both single-head (``OpticalStudent``) and multi-head
        (``MultiHeadOpticalStudent``) via the ``all_slm_layers()`` interface.
        """
        if hasattr(student, "all_slm_layers"):
            for idx, (_, slm) in enumerate(student.all_slm_layers()):
                yield idx, slm
        else:
            yield 0, student.slm1
            yield 1, student.slm2

    def prefilter_feature(self, x):
        kernel_size = int(getattr(self.config, "FEATURE_LOSS_PREFILTER_KERNEL", 1))
        if kernel_size <= 1:
            return x
        if kernel_size % 2 == 0:
            kernel_size += 1
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, student_feature, teacher_feature, student, stage_name=None, targets=None):
        filtered_student = self.prefilter_feature(student_feature)
        filtered_teacher = self.prefilter_feature(teacher_feature)
        aligned_student, aligned_teacher = self.domain_align(filtered_student, filtered_teacher)
        loss_full = F.mse_loss(aligned_student, aligned_teacher)
        loss_low1 = F.mse_loss(self.pool1(aligned_student), self.pool1(aligned_teacher))
        loss_low2 = F.mse_loss(self.pool2(aligned_student), self.pool2(aligned_teacher))
        loss_ssim = self.ssim_loss(aligned_student, aligned_teacher)
        loss_grad = self.gradient_loss(aligned_student, aligned_teacher)
        loss_freq = self.frequency_loss(aligned_student, aligned_teacher)
        loss_pearson = self.pearson_loss(filtered_student, filtered_teacher)
        global_total = (
            loss_full * self.config.LOSS_FULL_WEIGHT
            + loss_low1 * self.config.LOSS_LOW1_WEIGHT
            + loss_low2 * self.config.LOSS_LOW2_WEIGHT
            + loss_ssim * self.config.LOSS_SSIM_WEIGHT
            + loss_grad * self.config.LOSS_GRAD_WEIGHT
            + loss_freq * self.config.LOSS_FREQ_WEIGHT
            + loss_pearson * self.config.LOSS_PEARSON_WEIGHT
        )
        roi_loss = student_feature.new_zeros(())
        background_loss = student_feature.new_zeros(())
        roi_weight = float(getattr(self.config, "LOSS_TARGET_ROI_WEIGHT", 0.0))
        if bool(getattr(self.config, "ENABLE_TARGET_ROI_FEATURE_LOSS", False)) and targets is not None and roi_weight > 0:
            roi_mask = self.target_roi_mask(targets, student_feature.shape[-2], student_feature.shape[-1], student_feature.device, student_feature.dtype)
            roi_loss = self.masked_mse(aligned_student, aligned_teacher, roi_mask)
            background_loss = self.masked_mse(aligned_student, aligned_teacher, (roi_mask <= 0).to(dtype=roi_mask.dtype))
        total = global_total + roi_weight * roi_loss
        stats = {
            "feature_total": float(total.detach().item()),
            "feature_global": float(global_total.detach().item()),
            "feature_roi": float(roi_loss.detach().item()),
            "feature_background": float(background_loss.detach().item()),
            "full": float(loss_full.detach().item()),
            "low1": float(loss_low1.detach().item()),
            "low2": float(loss_low2.detach().item()),
            "ssim": float(loss_ssim.detach().item()),
            "grad": float(loss_grad.detach().item()),
            "freq": float(loss_freq.detach().item()),
            "pearson": float(loss_pearson.detach().item()),
        }
        return total, stats


def phase_regularization_loss(config, student):
    """Regularize the physical complex phase, rather than unwrapped radians."""
    layers = student.all_slm_layers() if hasattr(student, "all_slm_layers") else ()
    terms = []
    for _, slm in layers:
        phase = slm._raw_phase()
        phase_complex = torch.polar(torch.ones_like(phase), phase)
        dx = phase_complex[..., :, 1:] - phase_complex[..., :, :-1]
        dy = phase_complex[..., 1:, :] - phase_complex[..., :-1, :]
        tv = dx.abs().mean() + dy.abs().mean()
        kernel = max(int(getattr(config, "PHASE_HIGH_FREQ_KERNEL", 5)), 1)
        if kernel % 2 == 0:
            kernel += 1
        real_smooth = F.avg_pool2d(phase_complex.real, kernel, stride=1, padding=kernel // 2)
        imag_smooth = F.avg_pool2d(phase_complex.imag, kernel, stride=1, padding=kernel // 2)
        high_freq = (phase_complex.real - real_smooth).square().mean() + (phase_complex.imag - imag_smooth).square().mean()
        circular_variance = 1.0 - phase_complex.mean(dim=(2, 3), keepdim=True).abs()
        modulation = F.relu(float(config.PHASE_TARGET_CIRCULAR_VARIANCE) - circular_variance).square().mean()
        terms.append((tv, high_freq, modulation, circular_variance.mean()))
    if not terms:
        zero = torch.zeros((), device=next(student.parameters()).device)
        return zero, {"phase_regularization": 0.0, "phase_tv": 0.0, "phase_high_freq": 0.0, "phase_circular_variance": 0.0}
    tv = torch.stack([term[0] for term in terms]).mean()
    high_freq = torch.stack([term[1] for term in terms]).mean()
    modulation = torch.stack([term[2] for term in terms]).mean()
    circular_variance = torch.stack([term[3] for term in terms]).mean()
    total = (
        float(config.PHASE_CIRCULAR_TV_WEIGHT) * tv
        + float(config.PHASE_CIRCULAR_HIGH_FREQ_WEIGHT) * high_freq
        + float(config.PHASE_CIRCULAR_VARIANCE_WEIGHT) * modulation
    )
    return total, {
        "phase_regularization": float(total.detach().item()),
        "phase_tv": float(tv.detach().item()),
        "phase_high_freq": float(high_freq.detach().item()),
        "phase_circular_variance": float(circular_variance.detach().item()),
    }


def prediction_response_tensor(config, preds):
    if isinstance(preds, dict):
        preds = (preds,)
    response_maps = []
    for pred in preds:
        if isinstance(pred, dict):
            if "heatmap" in pred:
                response = torch.sigmoid(pred["heatmap"]).amax(dim=1, keepdim=True)
            elif "cls" in pred:
                # Current YOLOLightHead is anchor-free TAL: each scale emits
                # [B, num_classes, H, W] class logits and has no objectness
                # branch.  Its strongest class probability is the response.
                response = torch.sigmoid(pred["cls"]).amax(dim=1, keepdim=True)
                if "obj" in pred:
                    response = response * torch.sigmoid(pred["obj"])
            else:
                raise ValueError(f"Unsupported detector prediction keys: {sorted(pred)}")
        else:
            grid_h, grid_w = pred.shape[2], pred.shape[3]
            pred = pred.contiguous().permute(0, 2, 3, 1).contiguous().reshape(pred.shape[0], grid_h, grid_w, 3, -1)
            obj_conf = torch.sigmoid(pred[..., 4])
            cls_conf = torch.sigmoid(pred[..., 5:]).max(dim=-1).values
            response = (obj_conf * cls_conf).max(dim=-1).values.unsqueeze(1)
        response = _interpolate_preserve_layout(
            response,
            size=config.RESOLUTION,
            mode="bilinear",
            align_corners=False,
        )
        response_maps.append(response)
    return torch.stack(response_maps, dim=0).max(dim=0).values


def detection_response_loss(config, detector, student_feature, teacher_feature):
    if detector is None:
        zero = torch.zeros((), device=student_feature.device, dtype=student_feature.dtype)
        return zero, {"response": 0.0}
    student_response = prediction_response_tensor(
        config, detector(prepare_slm_detector_feature(config, student_feature))
    )
    with torch.no_grad():
        teacher_response = prediction_response_tensor(
            config, detector(prepare_slm_detector_feature(config, teacher_feature.detach()))
        )
    student_response = student_response / (student_response.amax(dim=(2, 3), keepdim=True) + config.OPTICAL_NORM_EPS)
    teacher_response = teacher_response / (teacher_response.amax(dim=(2, 3), keepdim=True) + config.OPTICAL_NORM_EPS)
    raw = F.mse_loss(student_response, teacher_response)
    return raw, {"response": float(raw.detach().item())}
