"""Plan D: Multi-Head Virtual SLM — training-time capacity boost.

Creates K parallel virtual SLM pairs during training.  Each head learns its own
independent phase patterns; outputs are fused before the detector.

Fusion modes
------------
``"mean"`` (default)
    ``final = (out_1 + ... + out_K) / K`` — uniform average.
    All heads contribute equally; gradients flow to every head on every sample.
    Simple, stable, no extra parameters.  Best as a default / baseline.

``"learned_gate"``
    ``final = sum_k (w_k * out_k)`` where ``w = GateNetwork(intensity)``.
    A tiny conv network predicts per-sample, per-head weights via softmax.
    This lets heads **specialise**: the gate can route easy samples to one head
    and hard samples to another, or let different heads dominate different
    spatial-frequency regimes.  Adds ~100 parameters.

    This is the "选择最优" (select-best) mode — instead of blind averaging,
    the model learns *which head is best for each input*.

Principle difference
--------------------
- **mean**: stateless fusion, all heads forced to be generalists.  Capacity gain
  comes purely from the ensemble average reducing variance.
- **learned_gate**: input-conditional fusion, heads can specialise.  Capacity
  gain comes from both ensemble averaging AND functional specialisation
  (akin to mixture-of-experts with soft routing).

Reference
---------
``docs/相位层修改意见.md`` (original Plan D proposal)
``docs/SLMIdea/光学层临时分析意见.md`` (Plan D analysis)

Integration
-----------
Set ``SLM_MULTI_HEAD_ENABLED = True`` and choose ``SLM_MULTI_HEAD_FUSION``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SLM.optical_layers import SLMLayer, ASMPropagation, OpticalStudent, _resolve_prop_distance
from models.SLM.slm_modulation import complex_intensity


# ═══════════════════════════════════════════════════════════════════════════
# Gate network — input-dependent head selection
# ═══════════════════════════════════════════════════════════════════════════

class GateNetwork(nn.Module):
    """Tiny conv network that predicts per-sample head weights from intensity.

    Architecture (deliberately minimal, ~100 params)::

        intensity (B,1,H,W)
          → Conv2d(1→4, k3, s2) → ReLU      # cheap spatial downsampling
          → AdaptiveAvgPool2d(1)             # global context
          → Flatten → Linear(4 → K) → Softmax  # per-head weights

    The spatial downsampling gives the gate a hint about low-frequency image
    statistics (brightness distribution, coarse structure) without adding
    meaningful compute cost.
    """

    def __init__(self, num_heads, hidden=4):
        super().__init__()
        self.conv = nn.Conv2d(1, hidden, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden, num_heads)

    def forward(self, intensity):
        """Return per-sample head weights  (B, K)."""
        x = self.conv(intensity)          # (B, hidden, H/2, W/2)
        x = F.relu(x)
        x = self.pool(x)                   # (B, hidden, 1, 1)
        x = x.flatten(1)                   # (B, hidden)
        return F.softmax(self.fc(x), dim=-1)  # (B, K)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Head Optical Student
# ═══════════════════════════════════════════════════════════════════════════

class MultiHeadOpticalStudent(nn.Module):
    """Optical student with K parallel virtual SLM N-layer paths (Plan D).

    Each head is an independent optical path (SLM1→Prop1→SLM2→Prop2→...→SLM_N→Prop_N)
    with its own phase parameters.  Propagation layers (ASM) are shared
    across all heads since they have no learnable parameters.

    Parameters
    ----------
    config : ConfigSLM
        Reads ``SLM_MULTI_HEAD_NUM_HEADS``, ``SLM_MULTI_HEAD_FUSION``,
        and ``NUM_LAYERS``.
    enable_norm : bool or None
        Override student output normalisation (default: from config).
    """

    def __init__(self, config, enable_norm=None):
        super().__init__()
        self.config = config
        self.num_heads = int(getattr(config, "SLM_MULTI_HEAD_NUM_HEADS", 4))
        self.num_layers = int(getattr(config, "NUM_LAYERS", 2))
        fusion_mode = str(getattr(config, "SLM_MULTI_HEAD_FUSION", "mean")).lower()
        assert fusion_mode in {"mean", "learned_gate"}, (
            f"Unknown SLM_MULTI_HEAD_FUSION: {fusion_mode}"
        )
        self._fusion_mode = fusion_mode

        # --- K independent heads per layer ---
        for layer_idx in range(1, self.num_layers + 1):
            heads = nn.ModuleList([
                SLMLayer(config, layer_index=layer_idx)
                for _ in range(self.num_heads)
            ])
            setattr(self, f"slm{layer_idx}_heads", heads)

        # --- Shared propagation per layer (fixed, no learnable parameters) ---
        for layer_idx in range(1, self.num_layers + 1):
            dist = _resolve_prop_distance(config, layer_idx)
            sampling_pitch = (
                config.sampling_pitch(layer_idx)
                if hasattr(config, "sampling_pitch") else config.PIXEL_SIZE
            )
            setattr(
                self,
                f"prop{layer_idx}",
                ASMPropagation(config, dist, pixel_size=sampling_pitch),
            )

        # --- Gate network (only for learned_gate mode) ---
        if self._fusion_mode == "learned_gate":
            self.gate = GateNetwork(self.num_heads)
        else:
            self.gate = None

        self.enable_norm = (
            config.ENABLE_STUDENT_NORM if enable_norm is None else enable_norm
        )

    # ── backward-compatible accessors ──────────────────────────────────────

    @property
    def slm1(self):
        return self.slm1_heads[0]

    @property
    def slm2(self):
        return self.slm2_heads[0]

    # Note: prop1/prop2 are set as regular attributes by __init__ via setattr,
    # so they do NOT need @property wrappers (which would cause recursion).

    # ── iteration interfaces ───────────────────────────────────────────────

    def all_slm_layers(self):
        """Yield ``(name, slm_layer)`` for every layer of every head."""
        for layer_idx in range(1, self.num_layers + 1):
            heads = getattr(self, f"slm{layer_idx}_heads")
            for k in range(self.num_heads):
                yield f"slm{layer_idx}_head{k}", heads[k]

    def all_slm_pairs(self):
        """Yield 2-tuples of slm layers per head (backward compat)."""
        for k in range(self.num_heads):
            yield tuple(
                getattr(self, f"slm{layer_idx}_heads")[k]
                for layer_idx in range(1, self.num_layers + 1)
            )

    # ── forward ────────────────────────────────────────────────────────────

    def forward(self, intensity):
        amp = torch.sqrt(intensity.clamp(min=0) + self.config.OPTICAL_FIELD_EPS)
        field = torch.complex(amp, torch.zeros_like(amp))

        outputs = []
        for k in range(self.num_heads):
            f = field
            for layer_idx in range(1, self.num_layers + 1):
                f = getattr(self, f"slm{layer_idx}_heads")[k](f)
                f = getattr(self, f"prop{layer_idx}")(f)
            out = complex_intensity(f)

            blur_kernel = int(getattr(self.config, "STUDENT_OUTPUT_BLUR_KERNEL", 1))
            if blur_kernel > 1:
                if blur_kernel % 2 == 0:
                    blur_kernel += 1
                out = F.avg_pool2d(out, kernel_size=blur_kernel, stride=1,
                                   padding=blur_kernel // 2)
            if self.enable_norm:
                out = self._apply_norm(out)
            outputs.append(out)

        stacked = torch.stack(outputs, dim=1)  # (B, K, 1, H, W)
        if self._fusion_mode == "learned_gate":
            weights = self.gate(intensity)
            weights = weights.view(-1, self.num_heads, 1, 1, 1)
            return (stacked * weights).sum(dim=1)
        return stacked.mean(dim=1)

    def _apply_norm(self, out):
        norm_mode = str(getattr(self.config, "STUDENT_NORM_MODE", "mean")).lower()
        if norm_mode == "max":
            scale = out.amax(dim=[2, 3], keepdim=True)
        elif norm_mode == "percentile":
            flat = out.flatten(2)
            q = float(getattr(self.config, "STUDENT_NORM_PERCENTILE", 0.995))
            scale = torch.quantile(flat, q, dim=2, keepdim=True).view(
                out.shape[0], out.shape[1], 1, 1)
        elif norm_mode == "none":
            scale = torch.ones_like(out.mean(dim=[2, 3], keepdim=True))
        else:
            scale = out.mean(dim=[2, 3], keepdim=True)
        out = out / (scale + self.config.OPTICAL_NORM_EPS)
        clamp_max = float(getattr(self.config, "STUDENT_OUTPUT_CLAMP_MAX", 0.0))
        if clamp_max > 0:
            out = out.clamp(max=clamp_max)
        return out

    # ── distillation ───────────────────────────────────────────────────────

    def to_single_student(self, head_idx=0):
        """Extract one head as a standalone ``OpticalStudent`` for deployment."""
        single = OpticalStudent(self.config, enable_norm=self.enable_norm)
        for layer_idx in range(1, self.num_layers + 1):
            src = getattr(self, f"slm{layer_idx}_heads")[head_idx]
            dst = getattr(single, f"slm{layer_idx}")
            dst.load_state_dict(src.state_dict(), strict=False)
        return single

    def select_best_head(self, dataloader, loss_fn, device="cuda", max_batches=50):
        """Evaluate each head independently; return index of the best."""
        was_training = self.training
        self.eval()
        head_losses = torch.zeros(self.num_heads, device=device)
        head_counts = torch.zeros(self.num_heads, device=device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                intensity = batch["intensity"].to(device)
                teacher_feat = batch["teacher_feature"].to(device)
                amp = torch.sqrt(intensity.clamp(min=0) + self.config.OPTICAL_FIELD_EPS)
                field = torch.complex(amp, torch.zeros_like(amp))

                for k in range(self.num_heads):
                    f = field
                    for layer_idx in range(1, self.num_layers + 1):
                        f = getattr(self, f"slm{layer_idx}_heads")[k](f)
                        f = getattr(self, f"prop{layer_idx}")(f)
                    out = torch.abs(f) ** 2

                    blur_kernel = int(getattr(self.config, "STUDENT_OUTPUT_BLUR_KERNEL", 1))
                    if blur_kernel > 1:
                        if blur_kernel % 2 == 0:
                            blur_kernel += 1
                        out = F.avg_pool2d(out, kernel_size=blur_kernel,
                                           stride=1, padding=blur_kernel // 2)
                    if self.enable_norm:
                        out = self._apply_norm(out)
                    head_losses[k] += loss_fn(out, teacher_feat).item()
                    head_counts[k] += 1

        if was_training:
            self.train()
        return int((head_losses / head_counts.clamp(min=1)).argmin().item())

    def mean_teacher_phase(self):
        """Average unwrapped phase across heads. Returns {slm1: ..., slm2: ...}."""
        result = {}
        for layer_idx in range(1, self.num_layers + 1):
            phases = []
            for head in getattr(self, f"slm{layer_idx}_heads"):
                phases.append(head._raw_phase().detach())
            result[f"slm{layer_idx}"] = torch.stack(phases).mean(dim=0)
        return result
