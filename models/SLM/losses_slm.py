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


class RegionInvariantOpticalLoss(nn.Module):
    """Transfer target-local structure across unlike optical quantities.

    Teacher output and student intensity are normalized independently inside
    ground-truth regions.  The loss therefore ignores absolute amplitude and
    never aligns the full image.  Only local normalized structure, gradients,
    coarse ROI layout and ROI correlation are transferred.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.eps = float(getattr(config, "OPTICAL_NORM_EPS", 1e-6))
        self.structure_weight = float(
            getattr(config, "LOCAL_FEATURE_STRUCTURE_WEIGHT", 0.45)
        )
        self.gradient_weight = float(
            getattr(config, "LOCAL_FEATURE_GRADIENT_WEIGHT", 0.25)
        )
        self.pyramid_weight = float(
            getattr(config, "LOCAL_FEATURE_PYRAMID_WEIGHT", 0.15)
        )
        self.correlation_weight = float(
            getattr(config, "LOCAL_FEATURE_CORRELATION_WEIGHT", 0.15)
        )
        self.pyramid_scales = tuple(
            int(value)
            for value in getattr(config, "LOCAL_FEATURE_PYRAMID_SCALES", (2, 4))
        )
        if not self.pyramid_scales or min(self.pyramid_scales) < 1:
            raise ValueError("LOCAL_FEATURE_PYRAMID_SCALES must be positive.")

    def target_roi_mask(self, targets, height, width, device, dtype):
        """Rasterize class-weighted GT boxes with context and soft boundaries."""
        if targets is None:
            return torch.zeros((0, 1, height, width), device=device, dtype=dtype)
        mask = torch.zeros((len(targets), 1, height, width), device=device, dtype=dtype)
        context_scale = max(
            float(getattr(self.config, "TARGET_ROI_CONTEXT_SCALE", 1.0)), 1.0
        )
        class_weights = dict(getattr(self.config, "TARGET_ROI_CLASS_WEIGHTS", {}))
        max_class_id = max([int(key) for key in class_weights] + [0])
        class_weight_lut = torch.ones(max_class_id + 1, device=device, dtype=dtype)
        for class_id, weight in class_weights.items():
            class_weight_lut[int(class_id)] = float(weight)

        pixel_y = torch.arange(height, device=device).view(1, height, 1)
        pixel_x = torch.arange(width, device=device).view(1, 1, width)
        for batch_index, sample_targets in enumerate(targets):
            if sample_targets is None or sample_targets.numel() == 0:
                continue
            boxes = sample_targets.to(device=device, dtype=dtype)
            if boxes.ndim != 2 or boxes.shape[1] < 5:
                continue
            boxes = boxes[(boxes[:, 3] > 0) & (boxes[:, 4] > 0)]
            if boxes.numel() == 0:
                continue
            half_w = boxes[:, 3] * context_scale * 0.5
            half_h = boxes[:, 4] * context_scale * 0.5
            x0 = torch.floor((boxes[:, 1] - half_w) * width).long().clamp(0, width)
            x1 = torch.ceil((boxes[:, 1] + half_w) * width).long().clamp(0, width)
            y0 = torch.floor((boxes[:, 2] - half_h) * height).long().clamp(0, height)
            y1 = torch.ceil((boxes[:, 2] + half_h) * height).long().clamp(0, height)
            valid = (x1 > x0) & (y1 > y0)
            if not torch.any(valid):
                continue
            class_ids = boxes[valid, 0].long()
            weights = torch.ones_like(class_ids, dtype=dtype)
            known = (class_ids >= 0) & (class_ids < class_weight_lut.numel())
            weights[known] = class_weight_lut[class_ids[known]]
            inside = (
                (pixel_x >= x0[valid].view(-1, 1, 1))
                & (pixel_x < x1[valid].view(-1, 1, 1))
                & (pixel_y >= y0[valid].view(-1, 1, 1))
                & (pixel_y < y1[valid].view(-1, 1, 1))
            )
            mask[batch_index, 0] = (
                inside.to(dtype) * weights.view(-1, 1, 1)
            ).amax(dim=0)

        feather = max(int(getattr(self.config, "TARGET_ROI_FEATHER_KERNEL", 1)), 1)
        if feather > 1:
            if feather % 2 == 0:
                feather += 1
            expanded = F.max_pool2d(mask, feather, stride=1, padding=feather // 2)
            softened = F.avg_pool2d(
                expanded, feather, stride=1, padding=feather // 2
            )
            mask = torch.maximum(mask, softened)
        return mask

    def _prefilter(self, feature):
        kernel = max(
            int(getattr(self.config, "FEATURE_LOSS_PREFILTER_KERNEL", 1)), 1
        )
        if kernel <= 1:
            return feature.float()
        if kernel % 2 == 0:
            kernel += 1
        return F.avg_pool2d(
            feature.float(), kernel_size=kernel, stride=1, padding=kernel // 2
        )

    def _roi_standardize(self, feature, mask):
        """Remove independent ROI amplitude and offset from each modality."""
        positive = self._prefilter(feature).clamp_min(0.0)
        mass = mask.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        roi_scale = (positive * mask).sum(dim=(2, 3), keepdim=True) / mass
        relative = torch.log1p(positive / (roi_scale + self.eps))
        roi_mean = (relative * mask).sum(dim=(2, 3), keepdim=True) / mass
        centered = relative - roi_mean
        roi_var = (centered.square() * mask).sum(dim=(2, 3), keepdim=True) / mass
        return centered / torch.sqrt(roi_var + self.eps)

    def _masked_sample_mean(self, values, weights):
        mass = weights.sum(dim=(1, 2, 3))
        valid = mass > self.eps
        if not torch.any(valid):
            return values.sum() * 0.0
        per_sample = (values * weights).sum(dim=(1, 2, 3)) / mass.clamp_min(self.eps)
        return per_sample[valid].mean()

    def _structure_loss(self, student, teacher, mask):
        pointwise = F.smooth_l1_loss(student, teacher, reduction="none")
        return self._masked_sample_mean(pointwise, mask)

    def _gradient_loss(self, student, teacher, mask):
        student_x = student[..., :, 1:] - student[..., :, :-1]
        teacher_x = teacher[..., :, 1:] - teacher[..., :, :-1]
        student_y = student[..., 1:, :] - student[..., :-1, :]
        teacher_y = teacher[..., 1:, :] - teacher[..., :-1, :]
        mask_x = torch.minimum(mask[..., :, 1:], mask[..., :, :-1])
        mask_y = torch.minimum(mask[..., 1:, :], mask[..., :-1, :])
        loss_x = self._masked_sample_mean(
            F.smooth_l1_loss(student_x, teacher_x, reduction="none"), mask_x
        )
        loss_y = self._masked_sample_mean(
            F.smooth_l1_loss(student_y, teacher_y, reduction="none"), mask_y
        )
        return 0.5 * (loss_x + loss_y)

    def _pyramid_loss(self, student, teacher, mask):
        losses = []
        for scale in self.pyramid_scales:
            if scale == 1:
                pooled_mask = mask
                pooled_student = student
                pooled_teacher = teacher
            else:
                pooled_mask = F.avg_pool2d(mask, scale, stride=scale)
                pooled_student = F.avg_pool2d(student * mask, scale, stride=scale)
                pooled_teacher = F.avg_pool2d(teacher * mask, scale, stride=scale)
                pooled_student = pooled_student / (pooled_mask + self.eps)
                pooled_teacher = pooled_teacher / (pooled_mask + self.eps)
            losses.append(
                self._masked_sample_mean(
                    F.smooth_l1_loss(
                        pooled_student, pooled_teacher, reduction="none"
                    ),
                    pooled_mask,
                )
            )
        return torch.stack(losses).mean()

    def _correlation_loss(self, student, teacher, mask):
        weighted_student = student * torch.sqrt(mask.clamp_min(0.0))
        weighted_teacher = teacher * torch.sqrt(mask.clamp_min(0.0))
        numerator = (weighted_student * weighted_teacher).sum(dim=(1, 2, 3))
        denominator = torch.sqrt(
            weighted_student.square().sum(dim=(1, 2, 3))
            * weighted_teacher.square().sum(dim=(1, 2, 3))
            + self.eps
        )
        valid = mask.sum(dim=(1, 2, 3)) > self.eps
        if not torch.any(valid):
            return student.sum() * 0.0
        correlation = (numerator / denominator.clamp_min(self.eps)).clamp(-1.0, 1.0)
        return (1.0 - correlation[valid]).mean()

    def forward(
        self,
        student_feature,
        teacher_feature,
        student,
        stage_name=None,
        targets=None,
    ):
        del student, stage_name
        if targets is None:
            raise ValueError(
                "RegionInvariantOpticalLoss requires GT targets; full-frame "
                "teacher/student alignment is intentionally unsupported."
            )
        mask = self.target_roi_mask(
            targets,
            student_feature.shape[-2],
            student_feature.shape[-1],
            student_feature.device,
            torch.float32,
        )
        if mask.shape[0] != student_feature.shape[0]:
            raise ValueError(
                "Target batch size does not match optical feature batch size."
            )
        student_local = self._roi_standardize(student_feature, mask)
        teacher_local = self._roi_standardize(teacher_feature.detach(), mask)
        structure = self._structure_loss(student_local, teacher_local, mask)
        gradient = self._gradient_loss(student_local, teacher_local, mask)
        pyramid = self._pyramid_loss(student_local, teacher_local, mask)
        correlation = self._correlation_loss(student_local, teacher_local, mask)
        total = (
            self.structure_weight * structure
            + self.gradient_weight * gradient
            + self.pyramid_weight * pyramid
            + self.correlation_weight * correlation
        )
        stats = {
            "feature_total": float(total.detach().item()),
            "feature_global": 0.0,
            "feature_roi": float(total.detach().item()),
            "feature_background": 0.0,
            "roi_structure": float(structure.detach().item()),
            "roi_gradient": float(gradient.detach().item()),
            "roi_pyramid": float(pyramid.detach().item()),
            "roi_correlation": float(correlation.detach().item()),
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
        real_smooth = F.avg_pool2d(
            phase_complex.real, kernel, stride=1, padding=kernel // 2
        )
        imag_smooth = F.avg_pool2d(
            phase_complex.imag, kernel, stride=1, padding=kernel // 2
        )
        high_freq = (
            (phase_complex.real - real_smooth).square().mean()
            + (phase_complex.imag - imag_smooth).square().mean()
        )
        circular_variance = 1.0 - phase_complex.mean(
            dim=(2, 3), keepdim=True
        ).abs()
        modulation = F.relu(
            float(config.PHASE_TARGET_CIRCULAR_VARIANCE) - circular_variance
        ).square().mean()
        terms.append((tv, high_freq, modulation, circular_variance.mean()))
    if not terms:
        zero = torch.zeros((), device=next(student.parameters()).device)
        return zero, {
            "phase_regularization": 0.0,
            "phase_tv": 0.0,
            "phase_high_freq": 0.0,
            "phase_circular_variance": 0.0,
        }
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
                response = torch.sigmoid(pred["cls"]).amax(dim=1, keepdim=True)
                if "obj" in pred:
                    response = response * torch.sigmoid(pred["obj"])
            else:
                raise ValueError(
                    f"Unsupported detector prediction keys: {sorted(pred)}"
                )
        else:
            grid_h, grid_w = pred.shape[2], pred.shape[3]
            pred = (
                pred.contiguous()
                .permute(0, 2, 3, 1)
                .contiguous()
                .reshape(pred.shape[0], grid_h, grid_w, 3, -1)
            )
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
        zero = torch.zeros(
            (), device=student_feature.device, dtype=student_feature.dtype
        )
        return zero, {"response": 0.0}
    student_response = prediction_response_tensor(
        config, detector(prepare_slm_detector_feature(config, student_feature))
    )
    with torch.no_grad():
        teacher_response = prediction_response_tensor(
            config,
            detector(
                prepare_slm_detector_feature(config, teacher_feature.detach())
            ),
        )
    student_response = student_response / (
        student_response.amax(dim=(2, 3), keepdim=True) + config.OPTICAL_NORM_EPS
    )
    teacher_response = teacher_response / (
        teacher_response.amax(dim=(2, 3), keepdim=True) + config.OPTICAL_NORM_EPS
    )
    raw = F.mse_loss(student_response, teacher_response)
    return raw, {"response": float(raw.detach().item())}
