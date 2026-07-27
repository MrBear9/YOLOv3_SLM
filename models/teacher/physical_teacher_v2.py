"""Phase-predicting, physically constrained teacher V2."""

import math

import torch
import torch.nn as nn

from .building_blocks import (
    C2fCIB,
    FeedbackGuidance,
    RawImageBridge,
    TeacherConvBNAct,
    TeacherResidualBlock,
    TeacherSPPF,
    _interpolate_preserve_layout,
)
from .fourier_layers import FourierOpticalLayer
from .physical_simulator import PhysicalSLMSimulator


class PhasePredictionHead(nn.Module):
    """Predict one full-resolution phase map with a neutral optical start."""

    def __init__(self, channels):
        super().__init__()
        self.refine = TeacherConvBNAct(channels, channels, 3)
        self.phase = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.phase.weight)
        nn.init.zeros_(self.phase.bias)

    def forward(self, features):
        # Zero initialization maps to an all-pass SLM at the start of training.
        return math.pi * torch.tanh(self.phase(self.refine(features)))


class PhysicallyConstrainedTeacherV2(nn.Module):
    """V2 teacher that predicts phase maps and renders them through SLM physics.

    The CNN is only a phase predictor.  The detector feature is strictly the
    intensity produced by phase-only modulation and band-limited ASM
    propagation, so it can be represented by an equally configured SLM student.
    """

    def __init__(self, config, base_channels=32, c2f_blocks=3, fourier_bands=8, fourier_low_pass_sigma=0.5):
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4

        self.stem = TeacherConvBNAct(1, c1, 3, 2)
        self.stage1 = C2fCIB(c1, c1, num_blocks=c2f_blocks, shortcut=True)
        self.down2 = TeacherConvBNAct(c1, c2, 3, 2)
        self.stage2 = C2fCIB(c2, c2, num_blocks=c2f_blocks + 1, shortcut=True)
        self.down3 = TeacherConvBNAct(c2, c3, 3, 2)
        self.stage3 = C2fCIB(c3, c3, num_blocks=c2f_blocks + 2, shortcut=True)
        self.sppf = TeacherSPPF(c3, c3)

        self.skip1 = nn.Sequential(nn.Conv2d(c1, c3, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU())
        self.skip2 = nn.Sequential(nn.Conv2d(c2, c3, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU())
        self.context = nn.Sequential(
            C2fCIB(c3, c3, num_blocks=c2f_blocks + 1, shortcut=True),
            TeacherResidualBlock(c3, dilation=2),
            TeacherResidualBlock(c3, dilation=4),
        )
        self.fourier = FourierOpticalLayer(c3, num_bands=fourier_bands, init_low_pass_sigma=fourier_low_pass_sigma)
        self.feedback_guidance = FeedbackGuidance([c3, c3, c3], guide_channels=16)

        self.lateral_s4 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, 2, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU(),
        )
        self.lateral_s2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 2, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU(),
            nn.Conv2d(c2, c3, 3, 2, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU(),
        )
        self.deep_fuse = nn.Sequential(
            TeacherConvBNAct(c3 * 3, c3),
            C2fCIB(c3, c3, num_blocks=c2f_blocks, shortcut=True),
        )
        self.dropout = nn.Dropout2d(0.1)
        self.refine = nn.Sequential(
            TeacherConvBNAct(c3, c2, 3),
            C2fCIB(c2, c2, num_blocks=c2f_blocks, shortcut=True),
            TeacherConvBNAct(c2, c1, 1),
        )
        self.raw_bridge_s8 = RawImageBridge(out_channels=c1)
        self.fuse_bridge_s8 = nn.Conv2d(c1 * 2, c1, 1, bias=True)

        self.simulator = PhysicalSLMSimulator(config)
        self.phase_heads = nn.ModuleList(PhasePredictionHead(c1) for _ in range(self.simulator.num_layers))

    def forward(self, x, return_aux=False):
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        gray = x.clamp(0.0, 1.0)

        x1 = self.stage1(self.stem(gray))
        x2 = self.stage2(self.down2(x1))
        x3 = self.stage3(self.down3(x2))
        p3 = self.sppf(x3)
        skip1 = _interpolate_preserve_layout(self.skip1(x1), size=p3.shape[-2:], mode="bilinear", align_corners=False)
        skip2 = _interpolate_preserve_layout(self.skip2(x2), size=p3.shape[-2:], mode="bilinear", align_corners=False)
        f_context = self.fourier(self.context(p3 + skip1 + skip2))
        f_s4 = self.lateral_s4(x2)
        f_s2 = self.lateral_s2(x1)
        f_guided = self.feedback_guidance(f_context, p3, f_context, f_s4)
        feat_scale8 = self.refine(self.dropout(self.deep_fuse(torch.cat([f_guided, f_s4, f_s2], dim=1))))

        bridge_s8 = self.raw_bridge_s8(gray, feat_scale8.shape[-2:])
        phase_features = self.fuse_bridge_s8(torch.cat([feat_scale8, bridge_s8], dim=1))
        phase_features = _interpolate_preserve_layout(
            phase_features, size=gray.shape[-2:], mode="bilinear", align_corners=False
        )
        phase_maps = tuple(head(phase_features) for head in self.phase_heads)
        det_feature = self.simulator(gray, phase_maps)

        if not return_aux:
            return det_feature
        aux = {
            "det_feature": det_feature,
            "gray": gray,
            "feat_raw_1ch": det_feature,
            "feat_scale2": x1,
            "feat_scale4": x2,
            "feat_scale8": feat_scale8,
            "phase_maps": phase_maps,
            "physics": self.simulator.physics_metadata(),
        }
        for index, phase_map in enumerate(phase_maps, start=1):
            aux[f"phase_map_{index}"] = phase_map
        return aux
