import torch.nn as nn
import torch.nn.functional as F


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, sample_weight=None):
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = logits.sigmoid()
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * loss

        if sample_weight is not None:
            sample_weight = sample_weight.to(device=loss.device, dtype=loss.dtype)
            while sample_weight.dim() < loss.dim():
                sample_weight = sample_weight.unsqueeze(-1)
            loss = loss * sample_weight
            if self.reduction == "sum":
                return loss.sum()
            if self.reduction == "none":
                return loss
            return loss.sum() / sample_weight.expand_as(loss).sum().clamp(min=1e-6)

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()
