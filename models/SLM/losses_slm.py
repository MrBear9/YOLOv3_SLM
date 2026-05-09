import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def phase_smoothness_loss(self, student):
        terms = []
        for slm_layer in (student.slm1, student.slm2):
            wrapped = slm_layer.wrapped_phase()
            cos_phase = torch.cos(wrapped)
            sin_phase = torch.sin(wrapped)
            terms.append(torch.abs(cos_phase[:, :, :, 1:] - cos_phase[:, :, :, :-1]).mean())
            terms.append(torch.abs(cos_phase[:, :, 1:, :] - cos_phase[:, :, :-1, :]).mean())
            terms.append(torch.abs(sin_phase[:, :, :, 1:] - sin_phase[:, :, :, :-1]).mean())
            terms.append(torch.abs(sin_phase[:, :, 1:, :] - sin_phase[:, :, :-1, :]).mean())
        return sum(terms) / max(len(terms), 1)

    def phase_diversity_loss(self, student):
        penalties = []
        stds = []
        spans = []
        circular_stds = []
        near_ratios = []
        for slm_layer in (student.slm1, student.slm2):
            wrapped = slm_layer.wrapped_phase()
            centered = slm_layer.centered_phase()
            flat = centered.flatten(1)
            std = flat.std(dim=1, unbiased=False).mean()
            span = (flat.amax(dim=1) - flat.amin(dim=1)).mean()
            wrapped_flat = wrapped.flatten(1)
            resultant = torch.sqrt(torch.mean(torch.cos(wrapped_flat), dim=1) ** 2 + torch.mean(torch.sin(wrapped_flat), dim=1) ** 2)
            circular_std = torch.sqrt(torch.clamp(-2.0 * torch.log(torch.clamp(resultant, min=1e-8)), min=0.0)).mean()
            near_boundary = ((wrapped_flat < self.config.PHASE_NEAR_BOUNDARY_EPS) | (wrapped_flat > 2 * torch.pi - self.config.PHASE_NEAR_BOUNDARY_EPS)).float().mean(dim=1).mean()
            penalties.append(F.relu(self.config.PHASE_STD_TARGET - std))
            penalties.append(F.relu(self.config.PHASE_SPAN_TARGET - span))
            penalties.append(F.relu(self.config.PHASE_CIRCULAR_STD_TARGET - circular_std))
            penalties.append(F.relu(near_boundary - self.config.PHASE_NEAR_BOUNDARY_LIMIT))
            stds.append(std.detach())
            spans.append(span.detach())
            circular_stds.append(circular_std.detach())
            near_ratios.append(near_boundary.detach())
        return (
            sum(penalties) / max(len(penalties), 1),
            sum(stds) / len(stds),
            sum(spans) / len(spans),
            sum(circular_stds) / len(circular_stds),
            sum(near_ratios) / len(near_ratios),
        )

    def forward(self, student_feature, teacher_feature, student):
        loss_full = F.mse_loss(student_feature, teacher_feature)
        loss_low1 = F.mse_loss(self.pool1(student_feature), self.pool1(teacher_feature))
        loss_low2 = F.mse_loss(self.pool2(student_feature), self.pool2(teacher_feature))
        loss_ssim = self.ssim_loss(student_feature, teacher_feature)
        loss_grad = self.gradient_loss(student_feature, teacher_feature)
        loss_freq = self.frequency_loss(student_feature, teacher_feature)
        loss_pearson = self.pearson_loss(student_feature, teacher_feature)
        loss_smooth = self.phase_smoothness_loss(student)
        loss_div, std, span, circular_std, near_boundary = self.phase_diversity_loss(student)
        total = (
            loss_full * self.config.LOSS_FULL_WEIGHT
            + loss_low1 * self.config.LOSS_LOW1_WEIGHT
            + loss_low2 * self.config.LOSS_LOW2_WEIGHT
            + loss_ssim * self.config.LOSS_SSIM_WEIGHT
            + loss_grad * self.config.LOSS_GRAD_WEIGHT
            + loss_freq * self.config.LOSS_FREQ_WEIGHT
            + loss_pearson * self.config.LOSS_PEARSON_WEIGHT
            + loss_smooth * self.config.LOSS_PHASE_SMOOTH_WEIGHT
            + loss_div * self.config.LOSS_PHASE_DIVERSITY_WEIGHT
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
            "slm_smooth": float(loss_smooth.detach().item()),
            "slm_diversity": float(loss_div.detach().item()),
            "slm_std": float(std.item()),
            "slm_span": float(span.item()),
            "slm_circular_std": float(circular_std.item()),
            "slm_near_boundary": float(near_boundary.item()),
        }
        return total, stats


def prediction_response_tensor(config, preds):
    response_maps = []
    for pred in preds:
        grid_h, grid_w = pred.shape[2], pred.shape[3]
        pred = pred.contiguous().permute(0, 2, 3, 1).contiguous().reshape(pred.shape[0], grid_h, grid_w, 3, -1)
        obj_conf = torch.sigmoid(pred[..., 4])
        cls_conf = torch.sigmoid(pred[..., 5:]).max(dim=-1).values
        response = (obj_conf * cls_conf).max(dim=-1).values.unsqueeze(1)
        response = _interpolate_preserve_layout(response, size=(config.IMG_SIZE, config.IMG_SIZE), mode="bilinear", align_corners=False)
        response_maps.append(response)
    return torch.stack(response_maps, dim=0).max(dim=0).values


def detection_response_loss(config, detector, student_feature, teacher_feature):
    if detector is None or config.RESPONSE_LOSS_WEIGHT <= 0:
        zero = torch.zeros((), device=student_feature.device, dtype=student_feature.dtype)
        return zero, {"response": 0.0}
    student_response = prediction_response_tensor(config, detector(student_feature))
    with torch.no_grad():
        teacher_response = prediction_response_tensor(config, detector(teacher_feature.detach()))
    student_response = student_response / (student_response.amax(dim=(2, 3), keepdim=True) + config.OPTICAL_NORM_EPS)
    teacher_response = teacher_response / (teacher_response.amax(dim=(2, 3), keepdim=True) + config.OPTICAL_NORM_EPS)
    raw = F.mse_loss(student_response, teacher_response)
    return raw * config.RESPONSE_LOSS_WEIGHT, {"response": float(raw.detach().item())}
