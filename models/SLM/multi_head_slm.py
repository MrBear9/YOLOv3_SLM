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

from models.SLM.optical_layers import SLMLayer, ASMPropagation, OpticalStudent


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
    """Optical student with K parallel virtual SLM pairs (Plan D).

    Each head is an independent optical path (SLM1 → prop → SLM2 → prop)
    with its own phase parameters.  The propagation layers (ASM) are shared
    across all heads since they have no learnable parameters.

    Parameters
    ----------
    config : ConfigSLM
        Configuration object.  Reads ``SLM_MULTI_HEAD_NUM_HEADS`` and
        ``SLM_MULTI_HEAD_FUSION``.
    enable_norm : bool or None
        Override student output normalisation (default: from config).
    """

    def __init__(self, config, enable_norm=None):
        super().__init__()
        self.config = config
        self.num_heads = int(getattr(config, "SLM_MULTI_HEAD_NUM_HEADS", 4))
        fusion_mode = str(getattr(config, "SLM_MULTI_HEAD_FUSION", "mean")).lower()
        assert fusion_mode in {"mean", "learned_gate"}, (
            f"Unknown SLM_MULTI_HEAD_FUSION: {fusion_mode}"
        )
        self._fusion_mode = fusion_mode

        # --- K independent SLM layers (each with own phase parameters) ---
        self.slm1_heads = nn.ModuleList([
            SLMLayer(config, layer_index=1) for _ in range(self.num_heads)
        ])
        self.slm2_heads = nn.ModuleList([
            SLMLayer(config, layer_index=2) for _ in range(self.num_heads)
        ])

        # --- Shared propagation (fixed, no learnable parameters) ---
        self.prop1 = ASMPropagation(config, config.PROP_DISTANCE_1)
        self.prop2 = ASMPropagation(config, config.PROP_DISTANCE_2)

        # --- Gate network (only for learned_gate mode) ---
        if self._fusion_mode == "learned_gate":
            self.gate = GateNetwork(self.num_heads)
        else:
            self.gate = None

        self.enable_norm = (
            config.ENABLE_STUDENT_NORM if enable_norm is None else enable_norm
        )

    # ── backward-compatible accessors (stats / viz code expects these) ──────

    @property
    def slm1(self):
        """First head's SLM1 — for backward compat with stats/viz code."""
        return self.slm1_heads[0]

    @property
    def slm2(self):
        """First head's SLM2 — for backward compat with stats/viz code."""
        return self.slm2_heads[0]

    # ── multi-head iteration ───────────────────────────────────────────────

    def all_slm_pairs(self):
        """Yield (slm1, slm2) tuples for every head.

        Useful for phase regularisation loops that should cover all heads.
        """
        for k in range(self.num_heads):
            yield self.slm1_heads[k], self.slm2_heads[k]

    # ── forward ────────────────────────────────────────────────────────────

    def forward(self, intensity):
        """Run all K heads and fuse outputs.

        Fusion strategy depends on ``SLM_MULTI_HEAD_FUSION``:

        ``"mean"``
            ``out = mean(out_1, ..., out_K)`` — equal weights.

        ``"learned_gate"``
            ``out = sum_k (gate_k(intensity) * out_k)`` — input-dependent
            soft selection via a lightweight conv network.

        Parameters
        ----------
        intensity : Tensor  (B, 1, H, W)
            Input intensity image.

        Returns
        -------
        Tensor  (B, 1, H, W)
            Fused intensity across all heads.
        """
        amp = torch.sqrt(
            intensity.clamp(min=0) + self.config.OPTICAL_FIELD_EPS
        )
        field = torch.complex(amp, torch.zeros_like(amp))

        outputs = []
        for k in range(self.num_heads):
            f = self.slm1_heads[k](field)
            f = self.prop1(f)
            f = self.slm2_heads[k](f)
            f = self.prop2(f)
            out = torch.abs(f) ** 2

            # --- blur (shared across heads) ---
            blur_kernel = int(getattr(self.config, "STUDENT_OUTPUT_BLUR_KERNEL", 1))
            if blur_kernel > 1:
                if blur_kernel % 2 == 0:
                    blur_kernel += 1
                out = F.avg_pool2d(
                    out, kernel_size=blur_kernel, stride=1,
                    padding=blur_kernel // 2,
                )

            # --- normalisation ---
            if self.enable_norm:
                out = self._apply_norm(out)
            outputs.append(out)

        # (K, B, 1, H, W) → (B, K, 1, H, W)
        stacked = torch.stack(outputs, dim=1)

        # --- fuse ---
        if self._fusion_mode == "learned_gate":
            weights = self.gate(intensity)               # (B, K)
            weights = weights.view(-1, self.num_heads, 1, 1, 1)
            fused = (stacked * weights).sum(dim=1)       # (B, 1, H, W)
        else:
            fused = stacked.mean(dim=1)                  # (B, 1, H, W)

        return fused

    def _apply_norm(self, out):
        """Per-sample normalisation (mirrors OpticalStudent)."""
        norm_mode = str(getattr(self.config, "STUDENT_NORM_MODE", "mean")).lower()
        if norm_mode == "max":
            scale = out.amax(dim=[2, 3], keepdim=True)
        elif norm_mode == "percentile":
            flat = out.flatten(2)
            q = float(getattr(self.config, "STUDENT_NORM_PERCENTILE", 0.995))
            scale = torch.quantile(flat, q, dim=2, keepdim=True).view(
                out.shape[0], out.shape[1], 1, 1
            )
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
        """Extract one head as a standalone ``OpticalStudent`` for deployment.

        This creates a **new** ``OpticalStudent`` whose SLM phases are copied
        from the selected head.  The returned module is independent — further
        training of the multi-head model will not affect it.

        When ``fusion="learned_gate"``, use ``select_best_head(loader)`` first
        to pick the head that performs best on a validation subset.

        Parameters
        ----------
        head_idx : int
            Which head to extract (default: 0, the first head).

        Returns
        -------
        OpticalStudent
            Single-head student suitable for checkpoint export or deployment.
        """
        single = OpticalStudent(self.config, enable_norm=self.enable_norm)
        src_slm1 = self.slm1_heads[head_idx]
        src_slm2 = self.slm2_heads[head_idx]
        single.slm1.load_state_dict(src_slm1.state_dict(), strict=False)
        single.slm2.load_state_dict(src_slm2.state_dict(), strict=False)
        return single

    def select_best_head(self, dataloader, loss_fn, device="cuda", max_batches=50):
        """Evaluate each head independently and return the index of the best.

        Runs each head solo on a subset of data and compares the feature-match
        loss.  Useful before calling ``to_single_student()`` when using
        ``learned_gate`` fusion — the gate might have learned to route
        different samples to different heads, but for deployment you need one.

        Parameters
        ----------
        dataloader : DataLoader
            Validation or training subset.
        loss_fn : callable
            Feature loss function ``(student_feature, teacher_feature) → scalar``.
        device : str
        max_batches : int
            Cap on batches to evaluate.

        Returns
        -------
        int
            Index of the best-performing head.
        """
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
                amp = torch.sqrt(
                    intensity.clamp(min=0) + self.config.OPTICAL_FIELD_EPS
                )
                field = torch.complex(amp, torch.zeros_like(amp))

                for k in range(self.num_heads):
                    f = self.slm1_heads[k](field)
                    f = self.prop1(f)
                    f = self.slm2_heads[k](f)
                    f = self.prop2(f)
                    out = torch.abs(f) ** 2

                    blur_kernel = int(getattr(self.config, "STUDENT_OUTPUT_BLUR_KERNEL", 1))
                    if blur_kernel > 1:
                        if blur_kernel % 2 == 0:
                            blur_kernel += 1
                        out = F.avg_pool2d(
                            out, kernel_size=blur_kernel, stride=1,
                            padding=blur_kernel // 2,
                        )
                    if self.enable_norm:
                        out = self._apply_norm(out)

                    loss_val = loss_fn(out, teacher_feat)
                    head_losses[k] += loss_val.item()
                    head_counts[k] += 1

        if was_training:
            self.train()

        # Average loss per head; pick the lowest
        avg_losses = head_losses / head_counts.clamp(min=1)
        best = int(avg_losses.argmin().item())
        return best

    def mean_teacher_phase(self):
        """Average the unwrapped phase across heads (experimental).

        Returns a dict ``{"slm1": phase_tensor, "slm2": phase_tensor}``
        with the mean unwrapped phase from all heads.  This can be used to
        seed a single-head student for distillation fine-tuning.

        Notes
        -----
        Averaging *unwrapped* phase is mathematically valid (unlike averaging
        wrapped phase), but there is no guarantee the mean phase produces the
        same optical output as the mean intensity.  Fine-tuning is recommended.
        """
        slm1_phases = []
        slm2_phases = []
        for slm1_head in self.slm1_heads:
            slm1_phases.append(slm1_head._raw_phase().detach())
        for slm2_head in self.slm2_heads:
            slm2_phases.append(slm2_head._raw_phase().detach())
        return {
            "slm1": torch.stack(slm1_phases).mean(dim=0),
            "slm2": torch.stack(slm2_phases).mean(dim=0),
        }
