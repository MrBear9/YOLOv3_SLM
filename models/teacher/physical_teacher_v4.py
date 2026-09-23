"""Transferable physical teacher with a shared phase base and small residual."""

import math
import random

import torch
from torch import nn

from models.yolov8.building_blocks import DepthwisePointwiseConv, GlobalLocalStage
from .building_blocks import TeacherSPPF, _interpolate_preserve_layout
from .physical_teacher_v2 import PhysicallyConstrainedTeacherV2


class PhysicallyConstrainedTeacherV4(PhysicallyConstrainedTeacherV2):
    """Multi-scale teacher whose dominant phase is directly student-loadable.

    The old V4 predicted an unrelated full phase map for every image.  That
    raised teacher accuracy but left a fixed-SLM student fitting mutually
    incompatible targets.  This revision uses

        phase(x) = static_phase + residual_scale * tanh(residual_head(x))

    and randomly removes the residual during training.  Detection gradients
    therefore have to build a useful shared phase while a bounded conditional
    residual retains part of the stronger teacher's per-image flexibility.
    """

    architecture_revision = "physical_v4_static_residual_v2"

    def __init__(self, config):
        c = int(getattr(config, "TEACHER_V4_BASE_CHANNELS", 32))
        depths = tuple(getattr(config, "TEACHER_V4_DEPTHS", (2, 3, 3, 2)))
        if len(depths) != 4 or min(depths) < 1:
            raise ValueError("TEACHER_V4_DEPTHS requires four positive depths")
        super().__init__(
            config,
            base_channels=c,
            stage_depths=depths[:3],
            fourier_bands=int(getattr(config, "TEACHER_V2_FOURIER_BANDS", 8)),
            fourier_low_pass_sigma=float(
                getattr(config, "TEACHER_V2_FOURIER_LOW_PASS_SIGMA", 0.5)
            ),
            create_phase_heads=False,
        )
        grid = int(getattr(config, "GLOBAL_LOCAL_CONTEXT_GRID", 6))
        fusion_depth = int(getattr(config, "TEACHER_V4_FUSION_DEPTH", 1))
        if fusion_depth < 1:
            raise ValueError("TEACHER_V4_FUSION_DEPTH must be positive")
        self.stage4 = GlobalLocalStage(c * 4, c * 8, stride=2,
                                      num_blocks=depths[3], context_grid=grid)
        self.deep_pool = TeacherSPPF(c * 8, c * 8)
        self.top8 = self.fusion(c * 12, c * 4, grid, fusion_depth)
        self.top4 = self.fusion(c * 6, c * 2, grid, fusion_depth)
        self.top2 = self.fusion(c * 3, c, grid, fusion_depth)
        self.detail_projection = DepthwisePointwiseConv(c, c, 1)
        # Context processing stays on smaller maps; only one local convolution
        # is evaluated on the full phase canvas.
        self.phase_refine = DepthwisePointwiseConv(c, c, 3)

        resolution = tuple(int(value) for value in config.RESOLUTION)
        init_range = float(
            getattr(config, "TEACHER_V4_STATIC_PHASE_INIT_RANGE_RAD", 0.5)
        )
        if init_range <= 0:
            raise ValueError("TEACHER_V4_STATIC_PHASE_INIT_RANGE_RAD must be positive")
        self.static_phases = nn.ParameterList()
        self.dynamic_phase_heads = nn.ModuleList()
        for _ in range(self.simulator.num_layers):
            phase = torch.empty(1, 1, *resolution).uniform_(-init_range, init_range)
            phase.sub_(phase.mean())
            self.static_phases.append(nn.Parameter(phase))
            head = nn.Conv2d(c, 1, 1)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.dynamic_phase_heads.append(head)

        self.residual_start_rad = float(
            getattr(config, "TEACHER_V4_RESIDUAL_START_RAD", 0.35)
        )
        self.residual_end_rad = float(
            getattr(config, "TEACHER_V4_RESIDUAL_END_RAD", 0.08)
        )
        self.static_batch_probability = float(
            getattr(config, "TEACHER_V4_STATIC_BATCH_PROB", 0.50)
        )
        self.static_eval = bool(getattr(config, "TEACHER_V4_STATIC_EVAL", True))
        if not 0.0 <= self.static_batch_probability <= 1.0:
            raise ValueError("TEACHER_V4_STATIC_BATCH_PROB must be in [0, 1]")
        if min(self.residual_start_rad, self.residual_end_rad) < 0:
            raise ValueError("V4 residual phase limits must be non-negative")
        self._residual_scale_rad = self.residual_start_rad
        self._force_static_batch = None

    @staticmethod
    def fusion(cin, cout, grid, depth):
        return nn.Sequential(
            DepthwisePointwiseConv(cin, cout, 1),
            GlobalLocalStage(cout, cout, num_blocks=depth, context_grid=grid),
        )

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

    def set_training_progress(self, epoch, total_epochs):
        """Cosine-anneal the maximum conditional phase excursion."""
        denominator = max(int(total_epochs) - 1, 1)
        progress = min(max(float(epoch) / denominator, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        self._residual_scale_rad = (
            self.residual_end_rad
            + (self.residual_start_rad - self.residual_end_rad) * cosine
        )

    def set_static_batch(self, enabled):
        """Select the same static/dynamic route on every DDP rank."""
        self._force_static_batch = bool(enabled)

    def predict_phase_maps(self, phase_features, gray):
        del gray
        scale = phase_features.new_tensor(self._residual_scale_rad)
        residual_gate = phase_features.new_ones(())
        applied_scale = scale

        static_maps = tuple(phase for phase in self.static_phases)
        residual_maps = tuple(
            applied_scale * torch.tanh(head(phase_features))
            for head in self.dynamic_phase_heads
        )
        phase_maps = tuple(
            static_phase + residual
            for static_phase, residual in zip(static_maps, residual_maps)
        )
        return phase_maps, {
            "static_phase_maps": static_maps,
            "phase_residuals": residual_maps,
            "dynamic_phase_scale_rad": applied_scale,
            "static_only_batch": 1.0 - residual_gate,
        }

    def _static_forward(self, x, return_aux):
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        gray = x.clamp(0.0, 1.0)
        batch_size = gray.shape[0]
        static_maps = tuple(
            phase.expand(batch_size, -1, -1, -1)
            for phase in self.static_phases
        )
        det_feature = self.simulator(gray, static_maps)
        if self.training:
            # DDP still sees every CNN/residual parameter on static batches,
            # while the zero-valued connection avoids evaluating the encoder.
            dummy = det_feature.new_zeros(())
            for name, parameter in self.named_parameters():
                if not name.startswith("static_phases") and parameter.requires_grad:
                    dummy = dummy + parameter[(0,) * parameter.ndim] * 0.0
            det_feature = det_feature + dummy
        if not return_aux:
            return det_feature
        aux = {
            "det_feature": det_feature,
            "gray": gray,
            "feat_raw_1ch": det_feature,
            "phase_maps": static_maps,
            "static_phase_maps": static_maps,
            "phase_residuals": (),
            "dynamic_phase_scale_rad": gray.new_tensor(self._residual_scale_rad),
            "static_only_batch": gray.new_ones(()),
            "physics": self.simulator.physics_metadata(),
        }
        for index, phase_map in enumerate(static_maps, start=1):
            aux[f"phase_map_{index}"] = phase_map
        return aux

    def forward(self, x, return_aux=False):
        if not self.training:
            use_static = self.static_eval
        elif self._force_static_batch is not None:
            use_static = self._force_static_batch
        else:
            use_static = (
                self.static_batch_probability > 0
                and random.random() < self.static_batch_probability
            )
        if use_static:
            return self._static_forward(x, return_aux)
        return super().forward(x, return_aux=return_aux)

    def export_static_phases(self):
        """Return raw full-resolution phases compatible with student phase_raw."""
        return tuple(phase.detach() for phase in self.static_phases)
