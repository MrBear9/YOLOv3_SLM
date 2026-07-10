"""Teacher architecture variants.

Contains the three teacher model implementations:
  - ConvTeacher          : v1 deeper semantic projection teacher
  - ConvTeacherV3        : v3 residual+gate output teacher
  - CVOCAConvTeacherV2   : v2 CVOCA-style optical teacher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .building_blocks import (
    CVOCAStage,
    TeacherC2f,
    TeacherConvBNAct,
    TeacherResidualBlock,
    TeacherSPPF,
    _interpolate_preserve_layout,
)


class ConvTeacher(nn.Module):
    """Deeper semantic projection teacher.

    Compared to the v2 teacher family: larger base channels (32 vs 24), more C2f
    blocks per stage (3 vs 2), deeper context with two dilated residual
    blocks, and higher-capacity bottleneck (128 vs 96 channels at c3).

    The architecture follows the same pattern: multi-scale fusion at
    stride-8 followed by 1ch projection and bilinear upsampling, but
    with extra depth and width for richer feature extraction.
    """

    def __init__(self, base_channels=32, c2f_blocks=3):
        super().__init__()
        c1 = base_channels          # 32
        c2 = base_channels * 2      # 64
        c3 = base_channels * 4      # 128

        # --- Backbone ---
        self.stem = TeacherConvBNAct(1, c1, 3, 2)
        self.stage1 = TeacherC2f(c1, c1, c2f_blocks, shortcut=True)
        self.down2 = TeacherConvBNAct(c1, c2, 3, 2)
        self.stage2 = TeacherC2f(c2, c2, c2f_blocks + 1, shortcut=True)
        self.down3 = TeacherConvBNAct(c2, c3, 3, 2)
        self.stage3 = TeacherC2f(c3, c3, c2f_blocks + 2, shortcut=True)
        self.sppf = TeacherSPPF(c3, c3)

        # Skip connections to bottleneck
        self.skip1 = nn.Sequential(nn.Conv2d(c1, c3, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU())
        self.skip2 = nn.Sequential(nn.Conv2d(c2, c3, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU())

        # Deeper context: C2f + two dilated ResidualBlocks
        self.context = nn.Sequential(
            TeacherC2f(c3, c3, c2f_blocks + 1, shortcut=True),
            TeacherResidualBlock(c3, dilation=2),
            TeacherResidualBlock(c3, dilation=4),
        )

        # Multi-scale lateral connections projected to stride-8.
        self.lateral_s4 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, 2, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU(),
        )
        self.lateral_s2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 2, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU(),
            nn.Conv2d(c2, c3, 3, 2, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU(),
        )

        # Deep fusion of all three scales at stride-8
        self.deep_fuse = nn.Sequential(
            TeacherConvBNAct(c3 * 3, c3),
            TeacherC2f(c3, c3, c2f_blocks, shortcut=True),
        )
        self.dropout = nn.Dropout2d(0.1)

        # Refinement at stride-8
        self.refine = nn.Sequential(
            TeacherConvBNAct(c3, c2, 3),
            TeacherC2f(c2, c2, c2f_blocks, shortcut=True),
            TeacherConvBNAct(c2, c1, 1),
        )

        # 1-channel projection at stride-8
        self.proj_out = nn.Sequential(
            TeacherConvBNAct(c1, c1, 3),
            nn.Conv2d(c1, 1, 1),
        )

        # Learnable output affine lets the teacher adapt its output
        # distribution to the detector's needs (feature stabilization).
        # out_scale is passed through softplus so the scale is always
        # positive and bounded, preventing the output range from drifting.
        self.out_scale = nn.Parameter(torch.ones(1))
        self.out_bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _normalize_intensity(x):
        """Per-sample min-max normalisation to [0, 1].

        Gradients flow through amin/amax (concentrated at extreme pixels),
        which pushes the teacher to maintain strong spatial contrast.
        """
        low = x.amin(dim=(2, 3), keepdim=True)
        high = x.amax(dim=(2, 3), keepdim=True)
        return (x - low) / (high - low + 1e-6)

    def forward(self, x, return_aux=False):
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        gray = x

        x1 = self.stage1(self.stem(gray))                              # [B, c1, H/2, W/2]
        x2 = self.stage2(self.down2(x1))                               # [B, c2, H/4, W/4]
        x3 = self.stage3(self.down3(x2))                               # [B, c3, H/8, W/8]
        p3 = self.sppf(x3)                                             # [B, c3, H/8, W/8]

        skip1 = _interpolate_preserve_layout(self.skip1(x1), size=p3.shape[-2:], mode="bilinear", align_corners=False)
        skip2 = _interpolate_preserve_layout(self.skip2(x2), size=p3.shape[-2:], mode="bilinear", align_corners=False)
        f_s8 = self.context(p3 + skip1 + skip2)                        # [B, c3, H/8, W/8]

        f_s4 = self.lateral_s4(x2)                                     # [B, c3, H/8, W/8]
        f_s2 = self.lateral_s2(x1)                                     # [B, c3, H/8, W/8]
        f_fused = self.deep_fuse(torch.cat([f_s8, f_s4, f_s2], dim=1)) # [B, c3, H/8, W/8]
        f_fused = self.dropout(f_fused)

        f_refined = self.refine(f_fused)                               # [B, c1, H/8, W/8]
        feat_1ch = F.softplus(self.proj_out(f_refined))                # [B, 1, H/8, W/8]  — no saturation, ≥0
        feat_1ch = self._normalize_intensity(feat_1ch)                  # per-sample [0, 1]  — forces full dynamic range
        feat_1ch = torch.clamp(feat_1ch * F.softplus(self.out_scale) + self.out_bias, min=0.0)
        det_feature = _interpolate_preserve_layout(feat_1ch, size=gray.shape[-2:], mode="bilinear", align_corners=False)

        if return_aux:
            return {
                "det_feature": det_feature,
                "gray": gray,
                "feat_scale8": f_refined,
                "feat_scale4": f_s4,
                "feat_scale2": f_s2,
                "feat_raw_1ch": feat_1ch,
            }
        return det_feature



class ConvTeacherV3(nn.Module):
    """YOLOv8-style teacher with residual+gate output (V2 backbone, new output head).

    Shares the same C2f feedforward backbone shape as the v2 teacher family. The difference
    is in the output: instead of ``sigmoid(abs(bridge(refine)))``, a purely
    synthetic feature map, V3 produces::

        det_feature = gray + residual_scale * gate * residual

    where *residual* (tanh) learns what to add/subtract and *gate* (sigmoid)
    learns where to apply the modification.  The original gray signal is
    always preserved; the network only needs to learn sparse enhancements
    in object-relevant regions.

    Auxiliary heads (heat / box / edge) are provided for optional pretraining
    and are not used during joint detection training.
    """

    def __init__(self, base_channels=24, c2f_blocks=2, residual_scale=0.30):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.residual_scale = float(residual_scale)

        # --- Backbone (same shape as the v2 teacher family) ---
        self.stem = TeacherConvBNAct(1, c1, 3, 2)
        self.stage1 = TeacherC2f(c1, c1, c2f_blocks, shortcut=True)
        self.down2 = TeacherConvBNAct(c1, c2, 3, 2)
        self.stage2 = TeacherC2f(c2, c2, c2f_blocks + 1, shortcut=True)
        self.down3 = TeacherConvBNAct(c2, c3, 3, 2)
        self.stage3 = TeacherC2f(c3, c3, c2f_blocks + 1, shortcut=True)
        self.sppf = TeacherSPPF(c3, c3)
        self.skip1 = nn.Sequential(nn.Conv2d(c1, c3, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU())
        self.skip2 = nn.Sequential(nn.Conv2d(c2, c3, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU())
        self.context = nn.Sequential(
            TeacherC2f(c3, c3, c2f_blocks, shortcut=True),
            TeacherResidualBlock(c3, dilation=2),
        )
        self.refine = nn.Sequential(
            TeacherConvBNAct(c3, c2, 3),
            TeacherC2f(c2, c2, max(c2f_blocks, 1), shortcut=True),
            TeacherConvBNAct(c2, c1, 1),
        )

        # --- Output heads (replaces bridge + abs + sigmoid) ---
        self.residual_head = nn.Sequential(TeacherConvBNAct(c1, c1), nn.Conv2d(c1, 1, 1))
        self.gate_head = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Sigmoid())

        # --- Auxiliary pretraining heads (not used in joint training) ---
        self.heat_head = nn.Conv2d(c1, 1, 1)
        self.box_head = nn.Conv2d(c1, 1, 1)
        self.edge_head = nn.Conv2d(c1, 1, 1)

    def forward(self, x, return_aux=False):
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        gray = x.clamp(0.0, 1.0)

        # Backbone (same as V2 up to refine)
        x1 = self.stage1(self.stem(gray))
        x2 = self.stage2(self.down2(x1))
        x3 = self.stage3(self.down3(x2))
        p3 = self.sppf(x3)
        skip1 = _interpolate_preserve_layout(self.skip1(x1), size=p3.shape[-2:], mode="bilinear", align_corners=False)
        skip2 = _interpolate_preserve_layout(self.skip2(x2), size=p3.shape[-2:], mode="bilinear", align_corners=False)
        f = self.context(p3 + skip1 + skip2)
        f = self.refine(f)

        # Multi-scale features for distillation (before upsampling)
        feat_scale8 = f                                                # [B, c1, H/8, W/8] deepest fused

        # Upsample to original resolution before applying heads
        f = _interpolate_preserve_layout(f, size=gray.shape[-2:], mode="bilinear", align_corners=False)

        # Residual + gate output
        residual = torch.tanh(self.residual_head(f))
        gate = self.gate_head(f)
        det_feature = (gray + self.residual_scale * gate * residual).clamp(0.0, 1.0)

        if return_aux:
            return {
                "det_feature": det_feature,
                "gray": gray,
                "heat_logits": self.heat_head(f),
                "box_logits": self.box_head(f),
                "edge_logits": self.edge_head(f),
                "gate": gate,
                "residual": residual,
                "feat_raw_1ch": det_feature,
                # Multi-scale features for detector distillation (training only)
                "feat_scale2": x1,       # [B, c1, H/2, W/2] shallow texture
                "feat_scale4": x2,       # [B, c2, H/4, W/4] mid-level structure
                "feat_scale8": feat_scale8,  # [B, c1, H/8, W/8] deep semantics
            }
        return det_feature


class CVOCAConvTeacherV2(nn.Module):
    """CVOCA-style v2 teacher with optical processing at the semantic scale.

    High-resolution stages use ordinary lightweight conv blocks so 640x640
    training does not materialize multiple real/imag optical branches at
    320x320 or 160x160. CVOCA is kept at stride-8 where the feature map is
    small enough for practical training.
    """

    def __init__(self, base_channels=24, c2f_blocks=2, synthetic_wavelengths=3, complex_kernel_size=5):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        depth = max(int(c2f_blocks), 1)
        self.entry = TeacherConvBNAct(1, c1, 3, 2)
        self.stage_s2 = TeacherC2f(c1, c1, depth, shortcut=True)
        self.stage_s4 = nn.Sequential(
            TeacherConvBNAct(c1, c2, 3, 2),
            TeacherC2f(c2, c2, depth + 1, shortcut=True),
        )
        self.stage_s8 = nn.Sequential(
            TeacherConvBNAct(c2, c3, 3, 2),
            CVOCAStage(c3, c3, kernel_size=complex_kernel_size, num_wavelengths=synthetic_wavelengths, stride=1),
            TeacherResidualBlock(c3, dilation=2),
        )
        self.global_optical_context = nn.Sequential(
            CVOCAStage(c3, c3, kernel_size=complex_kernel_size, num_wavelengths=synthetic_wavelengths, stride=1),
            TeacherResidualBlock(c3, dilation=2),
            TeacherResidualBlock(c3, dilation=4),
        )
        self.s2_to_s8 = nn.Sequential(
            TeacherConvBNAct(c1, c2, 3, 2),
            TeacherConvBNAct(c2, c3, 3, 2),
        )
        self.s4_to_s8 = TeacherConvBNAct(c2, c3, 3, 2)
        self.fuse = nn.Sequential(
            TeacherConvBNAct(c3 * 3, c3, 1),
            CVOCAStage(c3, c3, kernel_size=complex_kernel_size, num_wavelengths=synthetic_wavelengths, stride=1),
            TeacherResidualBlock(c3, dilation=2),
            TeacherConvBNAct(c3, c1, 1),
        )
        self.semantic_gain = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Sigmoid())
        self.proj_out = nn.Sequential(
            CVOCAStage(c1, c1, kernel_size=complex_kernel_size, num_wavelengths=synthetic_wavelengths, stride=1),
            nn.Conv2d(c1, 1, 1),
        )
        self.out_scale = nn.Parameter(torch.ones(1))
        self.out_bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _normalize_intensity(x):
        low = x.amin(dim=(2, 3), keepdim=True)
        high = x.amax(dim=(2, 3), keepdim=True)
        return (x - low) / (high - low + 1e-6)

    def forward(self, x, return_aux=False):
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        gray = x.clamp(min=0.0)

        f_s2 = self.stage_s2(self.entry(gray))
        f_s4 = self.stage_s4(f_s2)
        f_s8 = self.stage_s8(f_s4)
        f_context = self.global_optical_context(f_s8)
        f_from_s2 = self.s2_to_s8(f_s2)
        f_from_s4 = self.s4_to_s8(f_s4)
        f_refined = self.fuse(torch.cat([f_context, f_from_s4, f_from_s2], dim=1))

        raw_cipher = F.softplus(self.proj_out(f_refined))
        raw_cipher = raw_cipher * (0.75 + 0.50 * self.semantic_gain(f_refined))
        feat_1ch = self._normalize_intensity(raw_cipher)
        feat_1ch = torch.clamp(feat_1ch * F.softplus(self.out_scale) + self.out_bias, min=0.0)
        det_feature = _interpolate_preserve_layout(feat_1ch, size=gray.shape[-2:], mode="bilinear", align_corners=False)

        if return_aux:
            return {
                "det_feature": det_feature,
                "gray": gray,
                "feat_scale8": f_refined,
                "feat_scale4": f_s4,
                "feat_scale2": f_s2,
                "feat_raw_1ch": feat_1ch,
                "optical_cipher_raw": raw_cipher,
            }
        return det_feature
