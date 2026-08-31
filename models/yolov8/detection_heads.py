"""Anchor-free four-scale detector for optical intensity inputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .building_blocks import (
    DepthwisePointwiseConv,
    GlobalLocalGatedBlock,
    GlobalLocalStage,
    SPPF,
)


class DepthwiseSeparableTower(nn.Module):
    """Low-cost spatial task tower with pointwise channel mixing."""

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.input_proj = DepthwisePointwiseConv(in_channels, hidden_channels, 1)
        self.spatial_mix = DepthwisePointwiseConv(hidden_channels, hidden_channels, 3)

    def forward(self, x):
        return self.spatial_mix(self.input_proj(x))


class AnchorFreeDetectBranch(nn.Module):
    """Decoupled TAL branch with DFL box regression and class logits."""

    def __init__(self, in_channels, num_classes, reg_max, hidden_channels):
        super().__init__()
        self.box_tower = DepthwiseSeparableTower(in_channels, hidden_channels)
        self.cls_tower = DepthwiseSeparableTower(in_channels, hidden_channels)
        self.box_pred = nn.Conv2d(hidden_channels, 4 * reg_max, 1)
        self.cls_pred = nn.Conv2d(hidden_channels, num_classes, 1)
        nn.init.constant_(self.cls_pred.bias, -4.6)

    def forward(self, x):
        return {
            "reg": self.box_pred(self.box_tower(x)),
            "cls": self.cls_pred(self.cls_tower(x)),
        }


class YOLOLightHead(nn.Module):
    """Four-scale detector with early global context and local-detail retention.

    The backbone, bidirectional pyramid and prediction towers use depthwise
    spatial mixing. Every resolution receives a gated global/local residual
    block, so coarse scene context does not have to wait for the deepest stage
    while the local branch preserves small or partially occluded targets.
    """

    def __init__(self, config, in_channels=1, base_ch=None):
        super().__init__()
        self.config = config
        c = base_ch if base_ch is not None else int(getattr(config, "YOLO_LIGHT_BASE_CH", 8))
        c2, c4, c8 = c * 2, c * 4, c * 8
        context_grid = int(getattr(config, "GLOBAL_LOCAL_CONTEXT_GRID", 6))
        depths = tuple(
            int(value)
            for value in getattr(config, "GLOBAL_LOCAL_DETECTOR_DEPTHS", (1, 1, 2))
        )
        if len(depths) != 3 or min(depths) < 1:
            raise ValueError(
                "GLOBAL_LOCAL_DETECTOR_DEPTHS must contain three positive integers."
            )

        self.use_coordconv = bool(getattr(config, "DETECTOR_USE_COORDCONV", False))
        stem_in_ch = in_channels + 2 if self.use_coordconv else in_channels

        # Complete detector backbone. All spatial kernels are depthwise.
        self.stem0 = DepthwisePointwiseConv(stem_in_ch, c, 3)
        self.stage2 = GlobalLocalStage(
            c, c2, stride=2, num_blocks=depths[0], context_grid=context_grid
        )
        self.stage4 = GlobalLocalStage(
            c2, c4, stride=2, num_blocks=depths[1], context_grid=context_grid
        )
        self.stage8 = GlobalLocalStage(
            c4, c8, stride=2, num_blocks=depths[2], context_grid=context_grid
        )
        self.stem_dropout = nn.Dropout2d(0.1)

        # Direct shallow-to-stride-8 paths retain optical edges and tiny targets.
        self.p3_from_s4 = DepthwisePointwiseConv(c4, c8, 3, stride=2)
        self.p3_from_s2 = nn.Sequential(
            DepthwisePointwiseConv(c2, c4, 3, stride=2),
            DepthwisePointwiseConv(c4, c8, 3, stride=2),
        )
        self.p3_fuse = nn.Sequential(
            DepthwisePointwiseConv(c8 * 3, c8, 1),
            GlobalLocalGatedBlock(c8, context_grid=context_grid),
        )

        self.p4_path = GlobalLocalStage(
            c8, c8, stride=2, num_blocks=1, context_grid=context_grid
        )
        self.p5_path = nn.Sequential(
            GlobalLocalStage(c8, c8, stride=2, num_blocks=1, context_grid=context_grid),
            SPPF(c8, c8),
            GlobalLocalGatedBlock(c8, context_grid=context_grid),
        )

        # Top-down path injects global semantics at high spatial resolution.
        self.p5_to_p4 = DepthwisePointwiseConv(c8, c4, 1)
        self.fuse_p4 = nn.Sequential(
            DepthwisePointwiseConv(c4 + c8, c4, 1),
            GlobalLocalGatedBlock(c4, context_grid=context_grid),
        )
        self.fuse_p3 = nn.Sequential(
            DepthwisePointwiseConv(c4 + c8, c2, 1),
            GlobalLocalGatedBlock(c2, context_grid=context_grid),
        )

        # Bottom-up path restores localization after semantic fusion.
        self.down_p3 = DepthwisePointwiseConv(c2, c4, 3, stride=2)
        self.pan_p4 = nn.Sequential(
            DepthwisePointwiseConv(c4 * 2, c4, 1),
            GlobalLocalGatedBlock(c4, context_grid=context_grid),
        )
        self.down_p4 = DepthwisePointwiseConv(c4, c8, 3, stride=2)
        self.pan_p5 = nn.Sequential(
            DepthwisePointwiseConv(c8 * 2, c8, 1),
            GlobalLocalGatedBlock(c8, context_grid=context_grid),
        )

        self.head_dropout = nn.Dropout2d(0.1)
        reg_max = int(getattr(config, "ANCHOR_FREE_REG_MAX", 16))
        head_ch = int(getattr(config, "ANCHOR_FREE_HEAD_CH", max(c2, 16)))
        p2_ch = int(getattr(config, "ANCHOR_FREE_P2_FUSION_CH", c2))
        p2_head_ch = int(getattr(config, "ANCHOR_FREE_P2_HEAD_CH", p2_ch))
        self.p2_from_s4 = DepthwisePointwiseConv(c4, p2_ch, 1)
        self.p2_from_p3 = DepthwisePointwiseConv(c2, p2_ch, 1)
        self.p2_fuse = nn.Sequential(
            DepthwisePointwiseConv(p2_ch * 2, p2_ch, 1),
            GlobalLocalGatedBlock(p2_ch, context_grid=context_grid),
        )
        self.anchor_free_heads = nn.ModuleList(
            [AnchorFreeDetectBranch(p2_ch, config.NUM_CLASSES, reg_max, p2_head_ch)]
            + [
                AnchorFreeDetectBranch(ch, config.NUM_CLASSES, reg_max, head_ch)
                for ch in (c2, c4, c8)
            ]
        )

    def _append_coordinates(self, x):
        if not self.use_coordconv:
            return x
        batch, _, height, width = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coordinates = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0)
        return torch.cat((x, coordinates.expand(batch, -1, -1, -1)), dim=1)

    def forward(self, x, return_features=False):
        x = self._append_coordinates(x)
        s1 = self.stem0(x)
        s2 = self.stem_dropout(self.stage2(s1))
        s4 = self.stem_dropout(self.stage4(s2))
        s8 = self.stem_dropout(self.stage8(s4))

        p3_feat = self.p3_fuse(
            torch.cat((s8, self.p3_from_s4(s4), self.p3_from_s2(s2)), dim=1)
        )
        p4_feat = self.p4_path(p3_feat)
        p5_feat = self.p5_path(p4_feat)

        p5_up = F.interpolate(
            self.p5_to_p4(p5_feat), size=p4_feat.shape[-2:], mode="nearest"
        )
        p4_fused = self.fuse_p4(torch.cat((p5_up, p4_feat), dim=1))
        p4_up = F.interpolate(p4_fused, size=p3_feat.shape[-2:], mode="nearest")
        p3_fused = self.fuse_p3(torch.cat((p4_up, p3_feat), dim=1))

        p3_down = self.down_p3(p3_fused)
        p4_pan = self.pan_p4(torch.cat((p3_down, p4_fused), dim=1))
        p4_down = self.down_p4(p4_pan)
        p5_pan = self.pan_p5(torch.cat((p4_down, p5_feat), dim=1))

        p3_up = F.interpolate(p3_fused, size=s4.shape[-2:], mode="nearest")
        p2_fused = self.p2_fuse(
            torch.cat((self.p2_from_s4(s4), self.p2_from_p3(p3_up)), dim=1)
        )
        prediction_features = tuple(
            self.head_dropout(feature)
            for feature in (p2_fused, p3_fused, p4_pan, p5_pan)
        )
        predictions = tuple(
            head(feature)
            for head, feature in zip(self.anchor_free_heads, prediction_features)
        )

        if return_features:
            return predictions, {"s8": p3_feat, "s16": p4_feat, "s32": p5_feat}
        return predictions
