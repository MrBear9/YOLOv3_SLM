"""YOLOv8 detection head variants — anchor-free TAL only.

Active head:
  - YOLOLightHead : lightweight, FPGA-friendly, 4-scale (P2-P5)

Legacy anchor heads (YOLOv8AnchorHead, EnhancedYOLOv8AnchorHead) were
removed together with the anchor protocol. See:
  docs/HeadIdea/deprecated-matching-strategies.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .building_blocks import ConvBNAct, ECABlock, SPPF


class DepthwiseSeparableTower(nn.Module):
    """Low-cost spatial task tower suitable for FPGA-oriented deployment."""

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.input_proj = ConvBNAct(in_channels, hidden_channels, 1)
        self.depthwise = ConvBNAct(hidden_channels, hidden_channels, 3, groups=hidden_channels)
        self.output_proj = ConvBNAct(hidden_channels, hidden_channels, 1)

    def forward(self, x):
        return self.output_proj(self.depthwise(self.input_proj(x)))


class AnchorFreeDetectBranch(nn.Module):
    """Decoupled anchor-free branch with DFL box regression and class logits."""

    def __init__(self, in_channels, num_classes, reg_max, hidden_channels):
        super().__init__()
        self.box_tower = DepthwiseSeparableTower(in_channels, hidden_channels)
        self.cls_tower = DepthwiseSeparableTower(in_channels, hidden_channels)
        self.box_pred = nn.Conv2d(hidden_channels, 4 * reg_max, 1)
        self.cls_pred = nn.Conv2d(hidden_channels, num_classes, 1)
        nn.init.constant_(self.cls_pred.bias, -4.6)

    def forward(self, x):
        return {"reg": self.box_pred(self.box_tower(x)), "cls": self.cls_pred(self.cls_tower(x))}


class YOLOLightHead(nn.Module):
    """Four-scale FPGA-oriented detector with decoupled depthwise task towers.

    The configurable base width controls the stem and feature pyramid.  The P2
    branch is intentionally retained because its small parameter cost is
    disproportionately valuable for small targets.
    """

    def __init__(self, config, in_channels=1, base_ch=None):
        super().__init__()
        self.config = config
        c = base_ch if base_ch is not None else int(getattr(config, "YOLO_LIGHT_BASE_CH", 8))
        c2, c4, c8 = c * 2, c * 4, c * 8

        # CoordConv: internally expand input channels with xy coordinate grid.
        # The caller always passes in_channels (normally 1); the head decides
        # whether to add 2 coord channels based on config alone.
        self.use_coordconv = bool(getattr(config, "DETECTOR_USE_COORDCONV", False))
        stem_in_ch = in_channels + 2 if self.use_coordconv else in_channels

        # Stem: pure conv chain stem_in_ch → c → c2 → c4 → c8  (no residual)
        self.stem0 = ConvBNAct(stem_in_ch, c, 3)
        self.down1 = ConvBNAct(c, c2, 3, 2)
        self.down2 = ConvBNAct(c2, c4, 3, 2)
        self.down3 = ConvBNAct(c4, c8, 3, 2)
        self.stem_dropout = nn.Dropout2d(0.1)

        # Fuse shallow optical texture with deeper semantics at the P3 scale.
        self.p3_main = ConvBNAct(c8, c4, 1)
        self.p3_from_s4 = ConvBNAct(c4, c4, 3, 2)
        self.p3_from_s2 = nn.Sequential(ConvBNAct(c2, c2, 3, 2), ConvBNAct(c2, c4, 3, 2))
        self.p3_fuse = nn.Sequential(ConvBNAct(c4 * 3, c8, 1), ECABlock(c8))

        # P4 / P5: pure stride-2 conv (no residual)
        self.p4_path = ConvBNAct(c8, c8, 3, 2)
        self.p5_path = nn.Sequential(ConvBNAct(c8, c8, 3, 2), SPPF(c8, c8))

        # Top-down FPN: 1×1 fusion conv + ECA channel attention
        self.fuse_p4 = nn.Sequential(ConvBNAct(c8 * 2, c4, 1), ECABlock(c4))
        self.fuse_p3 = nn.Sequential(ConvBNAct(c4 + c8, c2, 1), ECABlock(c2))

        # Bottom-up PAN (lightweight, no C2f): ECA after each fusion
        self.down_p3 = ConvBNAct(c2, c4, 3, 2)
        self.pan_p4 = nn.Sequential(ConvBNAct(c4 * 2, c4, 1), ECABlock(c4))
        self.down_p4 = ConvBNAct(c4, c8, 3, 2)
        self.pan_p5 = nn.Sequential(ConvBNAct(c8 * 2, c8, 1), ECABlock(c8))

        self.head_dropout = nn.Dropout2d(0.1)

        reg_max = int(getattr(config, "ANCHOR_FREE_REG_MAX", 16))
        head_ch = int(getattr(config, "ANCHOR_FREE_HEAD_CH", max(c2, 16)))
        p2_ch = int(getattr(config, "ANCHOR_FREE_P2_FUSION_CH", c2))
        p2_head_ch = int(getattr(config, "ANCHOR_FREE_P2_HEAD_CH", p2_ch))
        self.p2_from_s4 = ConvBNAct(c4, p2_ch, 1)
        self.p2_from_p3 = ConvBNAct(c2, p2_ch, 1)
        self.p2_fuse = nn.Sequential(
            ConvBNAct(p2_ch * 2, p2_ch, 1),
            ConvBNAct(p2_ch, p2_ch, 3, groups=p2_ch),
            ECABlock(p2_ch),
        )
        self.anchor_free_heads = nn.ModuleList(
            [AnchorFreeDetectBranch(p2_ch, config.NUM_CLASSES, reg_max, p2_head_ch)]
            + [AnchorFreeDetectBranch(ch, config.NUM_CLASSES, reg_max, head_ch) for ch in (c2, c4, c8)]
        )
        return

    def forward(self, x, return_features=False):
        # CoordConv: prepend normalized [-1, 1] coordinate channels.
        # The rest of the head sees stem_in_ch (= in_channels + 2) channels.
        if self.use_coordconv:
            B, _, H, W = x.shape
            device = x.device
            gy, gx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device),
                torch.linspace(-1, 1, W, device=device),
                indexing="ij",
            )
            coord = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
            x = torch.cat([x, coord], dim=1)

        # Stem: stem_in_ch → c → c2 → c4 → c8
        s1 = self.stem0(x)
        s2 = self.stem_dropout(self.down1(s1))
        s4 = self.stem_dropout(self.down2(s2))
        s8 = self.stem_dropout(self.down3(s4))
        p3_feat = self.p3_fuse(
            torch.cat([self.p3_main(s8), self.p3_from_s4(s4), self.p3_from_s2(s2)], dim=1)
        )
        p4_feat = self.p4_path(p3_feat)
        p5_feat = self.p5_path(p4_feat)

        # Top-down FPN
        p5_up = F.interpolate(p5_feat, size=p4_feat.shape[-2:], mode="nearest")
        p4_fused = self.fuse_p4(torch.cat([p5_up, p4_feat], dim=1))

        p4_up = F.interpolate(p4_fused, size=p3_feat.shape[-2:], mode="nearest")
        p3_fused = self.fuse_p3(torch.cat([p4_up, p3_feat], dim=1))

        # Bottom-up PAN
        p3_down = self.down_p3(p3_fused)
        p4_pan = self.pan_p4(torch.cat([p3_down, p4_fused], dim=1))
        p4_down = self.down_p4(p4_pan)
        p5_pan = self.pan_p5(torch.cat([p4_down, p5_feat], dim=1))

        # Detection heads — anchor-free TAL with DFL
        p3_up = F.interpolate(p3_fused, size=s4.shape[-2:], mode="nearest")
        p2_fused = self.p2_fuse(torch.cat([self.p2_from_s4(s4), self.p2_from_p3(p3_up)], dim=1))
        prediction_scales = (p2_fused, p3_fused, p4_pan, p5_pan)
        prediction_features = tuple(self.head_dropout(feature) for feature in prediction_scales)
        predictions = tuple(head(feat) for head, feat in zip(self.anchor_free_heads, prediction_features))

        if return_features:
            # Keep the original teacher-training taps and additionally expose
            # the four tensors feeding the actual prediction branches.  The
            # latter remain pre-dropout so guidance is deterministic.
            return predictions, {
                "s8": p3_feat,
                "s16": p4_feat,
                "s32": p5_feat,
                "prediction_scales": prediction_scales,
            }
        return predictions
