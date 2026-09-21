"""Deeper V2 encoder with top-down detail fusion; trained from scratch."""
import torch
from torch import nn
from models.yolov8.building_blocks import DepthwisePointwiseConv, GlobalLocalStage
from .building_blocks import TeacherSPPF, _interpolate_preserve_layout
from .physical_teacher_v2 import PhysicallyConstrainedTeacherV2


class PhysicallyConstrainedTeacherV4(PhysicallyConstrainedTeacherV2):
    architecture_revision = "physical_v4_multiscale_v1"

    def __init__(self, config):
        c = int(getattr(config, "TEACHER_V4_BASE_CHANNELS", 32))
        depths = tuple(getattr(config, "TEACHER_V4_DEPTHS", (3, 5, 5, 3)))
        if len(depths) != 4 or min(depths) < 1:
            raise ValueError("TEACHER_V4_DEPTHS requires four positive depths")
        super().__init__(config, base_channels=c, stage_depths=depths[:3],
                         fourier_bands=int(getattr(config, "TEACHER_V2_FOURIER_BANDS", 8)),
                         fourier_low_pass_sigma=float(getattr(config, "TEACHER_V2_FOURIER_LOW_PASS_SIGMA", .5)))
        grid = int(getattr(config, "GLOBAL_LOCAL_CONTEXT_GRID", 6))
        self.stage4 = GlobalLocalStage(c * 4, c * 8, stride=2,
                                      num_blocks=depths[3], context_grid=grid)
        self.deep_pool = TeacherSPPF(c * 8, c * 8)
        self.top8 = self.fusion(c * 12, c * 4, grid)
        self.top4 = self.fusion(c * 6, c * 2, grid)
        self.top2 = self.fusion(c * 3, c, grid)
        self.detail_projection = DepthwisePointwiseConv(c, c, 1)
        # Context processing happens on smaller maps; retain local refinement
        # on the full phase canvas, and reuse V2's physical simulator and heads.
        self.phase_refine = DepthwisePointwiseConv(c, c, 3)

    @staticmethod
    def fusion(cin, cout, grid):
        return nn.Sequential(DepthwisePointwiseConv(cin, cout, 1),
                             GlobalLocalStage(cout, cout, num_blocks=2, context_grid=grid))

    @staticmethod
    def up(x, target):
        return _interpolate_preserve_layout(x, size=target.shape[-2:],
                                            mode="bilinear", align_corners=False)

    def refine_multiscale(self, x1, x2, x3):
        deep = self.deep_pool(self.stage4(x3))
        y3 = self.top8(torch.cat((x3, self.up(deep, x3)), dim=1))
        y2 = self.top4(torch.cat((x2, self.up(y3, x2)), dim=1))
        y1 = self.top2(torch.cat((x1, self.up(y2, x1)), dim=1))
        return y1, y2, y3

    def fuse_phase_features(self, coarse, detail, gray):
        fused = self.up(coarse, detail) + self.detail_projection(detail)
        return self.up(fused, gray)
