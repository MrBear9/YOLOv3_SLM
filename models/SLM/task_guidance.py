"""Training-only task guidance for a compact optical detector.

The guide and trainable detector may use different channel widths.  Alignment
therefore happens on normalized spatial energy maps rather than raw channels.
Ground-truth boxes separate target, transition, and background regions, while
the letterbox mask prevents padding from becoming a distillation target.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTaskGuidance(nn.Module):
    """Align the four prediction scales without adding inference parameters."""

    def __init__(self, config):
        super().__init__()
        self.eps = float(getattr(config, "OPTICAL_NORM_EPS", 1e-6))
        self.level_weights = tuple(
            float(value)
            for value in getattr(config, "TASK_GUIDANCE_SCALE_WEIGHTS", (1.40, 1.00, 0.70, 0.40))
        )
        if len(self.level_weights) != 4 or not all(
            math.isfinite(value) and value >= 0.0 for value in self.level_weights
        ):
            raise ValueError("TASK_GUIDANCE_SCALE_WEIGHTS must contain four finite non-negative values.")
        self.target_weight = float(getattr(config, "TASK_GUIDANCE_TARGET_WEIGHT", 1.0))
        self.background_weight = float(getattr(config, "TASK_GUIDANCE_BACKGROUND_WEIGHT", 0.15))
        self.structure_weight = float(getattr(config, "TASK_GUIDANCE_STRUCTURE_WEIGHT", 0.05))
        self.target_context = float(getattr(config, "TASK_GUIDANCE_TARGET_CONTEXT", 1.15))
        self.transition_context = float(getattr(config, "TASK_GUIDANCE_TRANSITION_CONTEXT", 1.80))
        self.hard_background_gain = float(getattr(config, "TASK_GUIDANCE_HARD_BACKGROUND_GAIN", 1.5))
        self.class_weights = dict(getattr(config, "TARGET_ROI_CLASS_WEIGHTS", {}))
        if self.target_context <= 0.0 or self.transition_context < self.target_context:
            raise ValueError("Task-guidance context scales are invalid.")

    @staticmethod
    def _prediction_scales(feature_bundle):
        if not isinstance(feature_bundle, dict) or "prediction_scales" not in feature_bundle:
            raise KeyError("Detector feature bundle does not contain four prediction_scales.")
        scales = tuple(feature_bundle["prediction_scales"])
        if len(scales) != 4:
            raise ValueError(f"Expected four prediction scales, received {len(scales)}.")
        return scales

    def _normalized_energy(self, feature, valid):
        feature = feature.float()
        energy = torch.sqrt(feature.square().mean(dim=1, keepdim=True) + self.eps)
        valid_count = valid.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        mean_energy = (energy * valid).sum(dim=(2, 3), keepdim=True) / valid_count
        # log1p limits isolated optical peaks while retaining their spatial order.
        return torch.log1p(energy / (mean_energy + self.eps))

    @staticmethod
    def _box_bounds(box, height, width, context):
        center_x, center_y, box_w, box_h = (float(value) for value in box)
        half_w = 0.5 * box_w * context
        half_h = 0.5 * box_h * context
        x1 = max(0, min(width - 1, math.floor((center_x - half_w) * width)))
        y1 = max(0, min(height - 1, math.floor((center_y - half_h) * height)))
        x2 = max(x1 + 1, min(width, math.ceil((center_x + half_w) * width)))
        y2 = max(y1 + 1, min(height, math.ceil((center_y + half_h) * height)))
        return x1, y1, x2, y2

    def _region_masks(self, targets, valid_canvas, feature):
        batch_size, _, height, width = feature.shape
        if valid_canvas is None:
            valid = feature.new_ones((batch_size, 1, height, width))
        else:
            valid = F.interpolate(valid_canvas.to(device=feature.device, dtype=torch.float32),
                                  size=(height, width), mode="nearest")
        target_mask = feature.new_zeros((batch_size, 1, height, width), dtype=torch.float32)
        transition_exclusion = torch.zeros(
            (batch_size, 1, height, width), device=feature.device, dtype=torch.bool
        )
        for batch_index, sample_targets in enumerate(targets):
            sample_targets = sample_targets.to(device=feature.device)
            for row in sample_targets:
                if row.numel() < 5 or float(row[3]) <= 0.0 or float(row[4]) <= 0.0:
                    continue
                class_id = int(row[0].item())
                region_weight = float(self.class_weights.get(class_id, 1.0))
                x1, y1, x2, y2 = self._box_bounds(
                    row[1:5], height, width, self.target_context
                )
                current = target_mask[batch_index, 0, y1:y2, x1:x2]
                target_mask[batch_index, 0, y1:y2, x1:x2] = torch.maximum(
                    current, current.new_full(current.shape, region_weight)
                )
                x1, y1, x2, y2 = self._box_bounds(
                    row[1:5], height, width, self.transition_context
                )
                transition_exclusion[batch_index, 0, y1:y2, x1:x2] = True
        target_mask = target_mask * valid
        background_mask = valid * (~transition_exclusion).to(valid.dtype)
        return valid, target_mask, background_mask

    @staticmethod
    def _masked_smooth_l1(student, guide, weights):
        denominator = weights.sum().clamp(min=1.0)
        pointwise = F.smooth_l1_loss(student, guide, reduction="none")
        return (pointwise * weights).sum() / denominator

    @staticmethod
    def _batch_structure_loss(student, guide, valid):
        if student.shape[0] < 2:
            return student.sum() * 0.0
        student_desc = F.normalize((student * valid).flatten(1), dim=1, eps=1e-6)
        guide_desc = F.normalize((guide * valid).flatten(1), dim=1, eps=1e-6)
        student_relation = student_desc @ student_desc.transpose(0, 1)
        guide_relation = guide_desc @ guide_desc.transpose(0, 1)
        off_diagonal = ~torch.eye(student.shape[0], device=student.device, dtype=torch.bool)
        return F.smooth_l1_loss(
            student_relation[off_diagonal], guide_relation[off_diagonal], reduction="mean"
        )

    def forward(self, student_bundle, guide_bundle, targets, valid_canvas=None):
        student_scales = self._prediction_scales(student_bundle)
        guide_scales = self._prediction_scales(guide_bundle)
        total = student_scales[0].sum() * 0.0
        weighted_target = total
        weighted_background = total
        weighted_structure = total
        weight_sum = max(sum(self.level_weights), self.eps)
        stats = {}

        for index, (student_feature, guide_feature, level_weight) in enumerate(
            zip(student_scales, guide_scales, self.level_weights)
        ):
            if student_feature.shape[-2:] != guide_feature.shape[-2:]:
                raise ValueError(
                    "Guide and student prediction scales must have equal spatial sizes: "
                    f"{student_feature.shape[-2:]} != {guide_feature.shape[-2:]}"
                )
            valid, target_mask, background_mask = self._region_masks(
                targets, valid_canvas, student_feature
            )
            student_energy = self._normalized_energy(student_feature, valid)
            guide_energy = self._normalized_energy(guide_feature.detach(), valid)
            target_loss = self._masked_smooth_l1(student_energy, guide_energy, target_mask)
            hard_background = background_mask * (
                1.0 + self.hard_background_gain * guide_energy.detach().clamp(max=4.0)
            )
            background_loss = self._masked_smooth_l1(
                student_energy, guide_energy, hard_background
            )
            structure_loss = self._batch_structure_loss(
                student_energy, guide_energy, valid
            )
            level_loss = (
                self.target_weight * target_loss
                + self.background_weight * background_loss
                + self.structure_weight * structure_loss
            )
            total = total + level_weight * level_loss
            weighted_target = weighted_target + level_weight * target_loss
            weighted_background = weighted_background + level_weight * background_loss
            weighted_structure = weighted_structure + level_weight * structure_loss
            stride = 2 ** (index + 2)
            stats[f"stride{stride}_target"] = float(target_loss.detach().item())
            stats[f"stride{stride}_background"] = float(background_loss.detach().item())

        total = total / weight_sum
        stats.update({
            "total": float(total.detach().item()),
            "target": float((weighted_target / weight_sum).detach().item()),
            "background": float((weighted_background / weight_sum).detach().item()),
            "structure": float((weighted_structure / weight_sum).detach().item()),
        })
        return total, stats
