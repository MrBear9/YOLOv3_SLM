"""Phase-predicting physical teacher with global/local gated context."""

import math

import torch
import torch.nn as nn

from models.yolov8.building_blocks import (
    DepthwisePointwiseConv,
    GlobalLocalGatedBlock,
    GlobalLocalStage,
)

from .building_blocks import RawImageBridge, TeacherSPPF, _interpolate_preserve_layout
from .fourier_layers import FourierOpticalLayer
from .physical_simulator import PhysicalSLMSimulator


class PhasePredictionHead(nn.Module):
    """Project shared full-resolution features to one phase map."""

    def __init__(self, channels):
        super().__init__()
        self.phase = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.phase.weight)
        nn.init.zeros_(self.phase.bias)

    def forward(self, features):
        # Zero initialization preserves an all-pass optical start.
        return math.pi * torch.tanh(self.phase(features))


class PhysicallyConstrainedTeacherV2(nn.Module):
    """Physical teacher with an efficient global/local phase-prediction trunk.

    The neural network predicts phase only.  Its detector feature remains the
    intensity rendered by phase-only modulation and band-limited ASM
    propagation.  Spatial convolution in the trunk is depthwise-separable;
    pointwise projections are retained only where channels must be mixed.
    """

    def __init__(
        self,
        config,
        base_channels=32,
        c2f_blocks=3,
        fourier_bands=8,
        fourier_low_pass_sigma=0.5,
        stage_depths=None,
    ):
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        context_grid = int(getattr(config, "GLOBAL_LOCAL_CONTEXT_GRID", 6))
        default_depths = (
            max(int(c2f_blocks) - 1, 1),
            max(int(c2f_blocks), 1),
            max(int(c2f_blocks), 1),
        )
        depths = tuple(
            int(value)
            for value in (stage_depths if stage_depths is not None else getattr(config, "GLOBAL_LOCAL_TEACHER_DEPTHS", default_depths))
        )
        if len(depths) != 3 or min(depths) < 1:
            raise ValueError(
                "GLOBAL_LOCAL_TEACHER_DEPTHS must contain three positive integers."
            )

        self.stem = DepthwisePointwiseConv(1, c1, 3, stride=2)
        self.stage1 = GlobalLocalStage(
            c1, c1, num_blocks=depths[0], context_grid=context_grid
        )
        self.stage2 = GlobalLocalStage(
            c1, c2, stride=2, num_blocks=depths[1], context_grid=context_grid
        )
        self.stage3 = GlobalLocalStage(
            c2, c3, stride=2, num_blocks=depths[2], context_grid=context_grid
        )
        self.sppf = TeacherSPPF(c3, c3)

        self.skip1 = DepthwisePointwiseConv(c1, c3, 1)
        self.skip2 = DepthwisePointwiseConv(c2, c3, 1)
        self.context = nn.Sequential(
            GlobalLocalGatedBlock(c3, context_grid=context_grid, dilation=2),
            GlobalLocalGatedBlock(c3, context_grid=context_grid, dilation=3),
        )
        # The explicit Fourier route complements the coarse spatial context
        # branch and preserves the physical teacher's long-range inductive bias.
        self.fourier = FourierOpticalLayer(
            c3,
            num_bands=fourier_bands,
            init_low_pass_sigma=fourier_low_pass_sigma,
        )

        self.lateral_s4 = DepthwisePointwiseConv(c2, c3, 3, stride=2)
        self.lateral_s2 = nn.Sequential(
            DepthwisePointwiseConv(c1, c2, 3, stride=2),
            DepthwisePointwiseConv(c2, c3, 3, stride=2),
        )
        self.deep_fuse = nn.Sequential(
            DepthwisePointwiseConv(c3 * 3, c3, 1),
            GlobalLocalGatedBlock(c3, context_grid=context_grid),
            GlobalLocalGatedBlock(c3, context_grid=context_grid, dilation=2),
        )
        self.dropout = nn.Dropout2d(0.1)
        self.refine = nn.Sequential(
            DepthwisePointwiseConv(c3, c2, 3),
            GlobalLocalGatedBlock(c2, context_grid=context_grid),
            DepthwisePointwiseConv(c2, c1, 1),
        )
        self.raw_bridge_s8 = RawImageBridge(out_channels=c1)
        self.fuse_bridge_s8 = DepthwisePointwiseConv(c1 * 2, c1, 1)

        self.simulator = PhysicalSLMSimulator(config)
        self.phase_refine = nn.Sequential(
            GlobalLocalGatedBlock(c1, context_grid=context_grid),
            DepthwisePointwiseConv(c1, c1, 3),
        )
        self.phase_heads = nn.ModuleList(
            PhasePredictionHead(c1)
            for _ in range(self.simulator.num_layers)
        )

    def refine_multiscale(self, x1, x2, x3):
        return x1, x2, x3

    def fuse_phase_features(self, coarse, detail, gray):
        return _interpolate_preserve_layout(coarse, size=gray.shape[-2:],
                                            mode="bilinear", align_corners=False)

    def forward(self, x, return_aux=False):
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        gray = x.clamp(0.0, 1.0)

        x1 = self.stage1(self.stem(gray))
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x1, x2, x3 = self.refine_multiscale(x1, x2, x3)
        p3 = self.sppf(x3)
        skip1 = _interpolate_preserve_layout(
            self.skip1(x1),
            size=p3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        skip2 = _interpolate_preserve_layout(
            self.skip2(x2),
            size=p3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        context = self.fourier(self.context(p3 + skip1 + skip2))
        lateral_s4 = self.lateral_s4(x2)
        lateral_s2 = self.lateral_s2(x1)
        feat_scale8 = self.refine(
            self.dropout(
                self.deep_fuse(torch.cat((context, lateral_s4, lateral_s2), dim=1))
            )
        )

        bridge_s8 = self.raw_bridge_s8(gray, feat_scale8.shape[-2:])
        phase_features = self.fuse_bridge_s8(
            torch.cat((feat_scale8, bridge_s8), dim=1)
        )
        phase_features = self.fuse_phase_features(phase_features, x1, gray)
        phase_features = self.phase_refine(phase_features)
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
