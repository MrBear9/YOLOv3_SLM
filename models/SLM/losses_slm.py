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
    avg_pool = F.avg_pool2d
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_a = avg_pool(a, 3, 1, 1)
    mu_b = avg_pool(b, 3, 1, 1)
    sigma_a = avg_pool(a * a, 3, 1, 1) - mu_a * mu_a
    sigma_b = avg_pool(b * b, 3, 1, 1) - mu_b * mu_b
    sigma_ab = avg_pool(a * b, 3, 1, 1) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2) + eps
    )
    return torch.clamp(ssim_map.mean(), min=0.0, max=1.0)


def input_privacy_loss(config, student_feature, input_gray):
    student_view = _max_normalize(student_feature.float(), config.OPTICAL_NORM_EPS)
    gray_view = _max_normalize(input_gray.float().clamp(min=0), config.OPTICAL_NORM_EPS)
    corr_abs = _pearson_abs(student_view, gray_view, config.OPTICAL_FIELD_EPS)
    ssim_sim = _ssim_similarity(student_view, gray_view, config.OPTICAL_FIELD_EPS)
    corr_target = float(getattr(config, "PRIVACY_CORR_TARGET", 0.15))
    ssim_target = float(getattr(config, "PRIVACY_SSIM_TARGET", 0.20))
    raw = F.relu(corr_abs - corr_target) + F.relu(ssim_sim - ssim_target)
    return raw, {
        "privacy": float(raw.detach().item()),
        "privacy_corr": float(corr_abs.detach().item()),
        "privacy_ssim": float(ssim_sim.detach().item()),
    }


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

    def forward(self, student_feature, teacher_feature, student, stage_name=None):
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
        total = (
            loss_full * self.config.LOSS_FULL_WEIGHT
            + loss_low1 * self.config.LOSS_LOW1_WEIGHT
            + loss_low2 * self.config.LOSS_LOW2_WEIGHT
            + loss_ssim * self.config.LOSS_SSIM_WEIGHT
            + loss_grad * self.config.LOSS_GRAD_WEIGHT
            + loss_freq * self.config.LOSS_FREQ_WEIGHT
            + loss_pearson * self.config.LOSS_PEARSON_WEIGHT
        )
        stats = {
            "feature_total": float(total.detach().item()),
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
    """Keep phase maps expressive while suppressing non-deployable pixel noise."""
    layers = student.all_slm_layers() if hasattr(student, "all_slm_layers") else ()
    terms = []
    for _, slm in layers:
        phase = slm._raw_phase()
        dx = phase[..., :, 1:] - phase[..., :, :-1]
        dy = phase[..., 1:, :] - phase[..., :-1, :]
        tv = dx.abs().mean() + dy.abs().mean()
        kernel = max(int(getattr(config, "PHASE_HIGH_FREQ_KERNEL", 5)), 1)
        if kernel % 2 == 0:
            kernel += 1
        smooth = F.avg_pool2d(phase, kernel, stride=1, padding=kernel // 2)
        high_freq = (phase - smooth).square().mean()
        modulation = F.relu(float(config.PHASE_TARGET_STD_RAD) - phase.std(unbiased=False)).square()
        terms.append((tv, high_freq, modulation))
    if not terms:
        zero = torch.zeros((), device=next(student.parameters()).device)
        return zero, {"phase_regularization": 0.0, "phase_tv": 0.0, "phase_high_freq": 0.0, "phase_modulation": 0.0}
    tv = torch.stack([term[0] for term in terms]).mean()
    high_freq = torch.stack([term[1] for term in terms]).mean()
    modulation = torch.stack([term[2] for term in terms]).mean()
    total = (
        float(config.PHASE_TV_WEIGHT) * tv
        + float(config.PHASE_HIGH_FREQ_WEIGHT) * high_freq
        + float(config.PHASE_STD_WEIGHT) * modulation
    )
    return total, {
        "phase_regularization": float(total.detach().item()),
        "phase_tv": float(tv.detach().item()),
        "phase_high_freq": float(high_freq.detach().item()),
        "phase_modulation": float(modulation.detach().item()),
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
