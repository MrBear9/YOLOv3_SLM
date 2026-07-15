"""YOLOv8 detection head variants.

Contains the three detection head implementations:
  - YOLOv8AnchorHead         : original C2f/PAN anchor head
  - EnhancedYOLOv8AnchorHead  : ECA + deeper branches
  - YOLOLightHead             : lightweight FPGA-friendly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.runtime import should_use_channels_last

from .building_blocks import (
    C2f,
    C2fECA,
    ConvBNAct,
    ECABlock,
    EnhancedDetectBranch,
    SPPF,
    YOLOv8AnchorDetectBranch,
)


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


class YOLOv8AnchorHead(nn.Module):
    """YOLOv8-style C2f/PAN head with legacy YOLOv3 anchor-formatted outputs."""

    def __init__(self, config, in_channels=1, out_channels=None, base_ch=None, c2f_blocks=None):
        super().__init__()
        self.config = config
        out_channels = config.get_detector_output_channels() if out_channels is None else out_channels
        base_ch = config.YOLOV8_BASE_CHANNELS if base_ch is None else base_ch
        c2f_blocks = config.YOLOV8_C2F_BLOCKS if c2f_blocks is None else c2f_blocks
        self.stem = ConvBNAct(in_channels, base_ch, 3)
        self.down1 = ConvBNAct(base_ch, base_ch * 2, 3, 2)
        self.c2f1 = C2f(base_ch * 2, base_ch * 2, c2f_blocks, shortcut=True)
        self.down2 = ConvBNAct(base_ch * 2, base_ch * 4, 3, 2)
        self.c2f2 = C2f(base_ch * 4, base_ch * 4, c2f_blocks, shortcut=True)
        self.down3 = ConvBNAct(base_ch * 4, base_ch * 8, 3, 2)
        self.c2f3 = C2f(base_ch * 8, base_ch * 8, c2f_blocks, shortcut=True)
        self.down4 = ConvBNAct(base_ch * 8, base_ch * 8, 3, 2)
        self.c2f4 = C2f(base_ch * 8, base_ch * 8, c2f_blocks, shortcut=True)
        self.down5 = ConvBNAct(base_ch * 8, base_ch * 8, 3, 2)
        self.sppf = SPPF(base_ch * 8, base_ch * 8)
        self.up_p5 = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse_p4 = C2f(base_ch * 16, base_ch * 4, c2f_blocks)
        self.up_p4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse_p3 = C2f(base_ch * 12, base_ch * 2, c2f_blocks)
        self.down_p3 = ConvBNAct(base_ch * 2, base_ch * 4, 3, 2)
        self.pan_p4 = C2f(base_ch * 8, base_ch * 4, c2f_blocks)
        self.down_p4 = ConvBNAct(base_ch * 4, base_ch * 8, 3, 2)
        self.pan_p5 = C2f(base_ch * 16, base_ch * 8, c2f_blocks)
        self.head_dropout = nn.Dropout2d(0.1)
        self.head_p3 = YOLOv8AnchorDetectBranch(base_ch * 2, out_channels)
        self.head_p4 = YOLOv8AnchorDetectBranch(base_ch * 4, out_channels)
        self.head_p5 = YOLOv8AnchorDetectBranch(base_ch * 8, out_channels)

    def _preserve_layout_after_resize(self, x):
        if x.is_floating_point() and x.dim() == 4 and should_use_channels_last(self.config):
            return x.contiguous(memory_format=torch.channels_last)
        return x.contiguous() if x.is_floating_point() else x

    def forward(self, x, return_features=False):
        x = self.stem(x)
        x320 = self.c2f1(self.down1(x))
        x160 = self.c2f2(self.down2(x320))
        x80 = self.c2f3(self.down3(x160))
        x40 = self.c2f4(self.down4(x80))
        p5 = self.sppf(self.down5(x40))
        p5_up = self._preserve_layout_after_resize(self.up_p5(p5))
        p4 = self.fuse_p4(torch.cat([p5_up, x40], dim=1))
        p4_up = self._preserve_layout_after_resize(self.up_p4(p4))
        p3 = self.fuse_p3(torch.cat([p4_up, x80], dim=1))
        p4 = self.pan_p4(torch.cat([self.down_p3(p3), p4], dim=1))
        p5 = self.pan_p5(torch.cat([self.down_p4(p4), p5], dim=1))
        preds = (self.head_p3(self.head_dropout(p3)), self.head_p4(self.head_dropout(p4)), self.head_p5(self.head_dropout(p5)))
        if return_features:
            return preds, {"s8": x80, "s16": x40, "s32": p5}
        return preds


class EnhancedYOLOv8AnchorHead(nn.Module):
    """Enhanced YOLOv8-style anchor head with ECA attention + deeper detection branches.

    Improvements over YOLOv8AnchorHead:
      - C2fECA (C2f + ECA channel attention) in backbone and neck
      - ECA at every FPN/PAN fusion point
      - Deeper detection branches with residual + ECA (EnhancedDetectBranch)
      - Better gradient flow and feature recalibration

    Output format is identical to YOLOv8AnchorHead — fully compatible with
    the existing loss (YOLOv3AnchorLossForV8Head) and decode functions.
    """

    def __init__(self, config, in_channels=1, out_channels=None, base_ch=None, c2f_blocks=None):
        super().__init__()
        self.config = config
        out_channels = config.get_detector_output_channels() if out_channels is None else out_channels
        base_ch = config.YOLOV8_BASE_CHANNELS if base_ch is None else base_ch
        c2f_blocks = config.YOLOV8_C2F_BLOCKS if c2f_blocks is None else c2f_blocks

        # Backbone with C2fECA
        self.stem = ConvBNAct(in_channels, base_ch, 3)
        self.down1 = ConvBNAct(base_ch, base_ch * 2, 3, 2)
        self.c2f1 = C2fECA(base_ch * 2, base_ch * 2, c2f_blocks, shortcut=True)
        self.down2 = ConvBNAct(base_ch * 2, base_ch * 4, 3, 2)
        self.c2f2 = C2fECA(base_ch * 4, base_ch * 4, c2f_blocks, shortcut=True)
        self.down3 = ConvBNAct(base_ch * 4, base_ch * 8, 3, 2)
        self.c2f3 = C2fECA(base_ch * 8, base_ch * 8, c2f_blocks, shortcut=True)
        self.down4 = ConvBNAct(base_ch * 8, base_ch * 8, 3, 2)
        self.c2f4 = C2fECA(base_ch * 8, base_ch * 8, c2f_blocks, shortcut=True)
        self.down5 = ConvBNAct(base_ch * 8, base_ch * 8, 3, 2)
        self.sppf = SPPF(base_ch * 8, base_ch * 8)

        # Neck — FPN top-down with ECA
        self.up_p5 = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse_p4 = nn.Sequential(C2fECA(base_ch * 16, base_ch * 4, c2f_blocks), ECABlock(base_ch * 4))
        self.up_p4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse_p3 = nn.Sequential(C2fECA(base_ch * 12, base_ch * 2, c2f_blocks), ECABlock(base_ch * 2))

        # Neck — PAN bottom-up with ECA
        self.down_p3 = ConvBNAct(base_ch * 2, base_ch * 4, 3, 2)
        self.pan_p4 = nn.Sequential(C2fECA(base_ch * 8, base_ch * 4, c2f_blocks), ECABlock(base_ch * 4))
        self.down_p4 = ConvBNAct(base_ch * 4, base_ch * 8, 3, 2)
        self.pan_p5 = nn.Sequential(C2fECA(base_ch * 16, base_ch * 8, c2f_blocks), ECABlock(base_ch * 8))

        self.head_dropout = nn.Dropout2d(0.1)

        # Enhanced detection branches
        self.head_p3 = EnhancedDetectBranch(base_ch * 2, out_channels)
        self.head_p4 = EnhancedDetectBranch(base_ch * 4, out_channels)
        self.head_p5 = EnhancedDetectBranch(base_ch * 8, out_channels)

    def _preserve_layout_after_resize(self, x):
        if x.is_floating_point() and x.dim() == 4 and should_use_channels_last(self.config):
            return x.contiguous(memory_format=torch.channels_last)
        return x.contiguous() if x.is_floating_point() else x

    def forward(self, x, return_features=False):
        # Backbone
        x = self.stem(x)
        x320 = self.c2f1(self.down1(x))
        x160 = self.c2f2(self.down2(x320))
        x80 = self.c2f3(self.down3(x160))
        x40 = self.c2f4(self.down4(x80))
        p5 = self.sppf(self.down5(x40))

        # FPN top-down
        p5_up = self._preserve_layout_after_resize(self.up_p5(p5))
        p4 = self.fuse_p4(torch.cat([p5_up, x40], dim=1))
        p4_up = self._preserve_layout_after_resize(self.up_p4(p4))
        p3 = self.fuse_p3(torch.cat([p4_up, x80], dim=1))

        # PAN bottom-up
        p4 = self.pan_p4(torch.cat([self.down_p3(p3), p4], dim=1))
        p5 = self.pan_p5(torch.cat([self.down_p4(p4), p5], dim=1))

        # Detection heads
        preds = (
            self.head_p3(self.head_dropout(p3)),
            self.head_p4(self.head_dropout(p4)),
            self.head_p5(self.head_dropout(p5)),
        )
        if return_features:
            return preds, {"s8": x80, "s16": x40, "s32": p5}
        return preds


class YOLOLightHead(nn.Module):
    """Lightweight FPGA-friendly detection head (方案 A — slimmed).

    Compared to the original YOLOLightHead:
      - Base channel halved: 16 → 8  (total params ~657K → ~166K, -75%)
      - Stem: 1→8→16→32→64 (pure serial, no residual)
      - Multi-scale P3 fusion preserved (s2/s4/s8 → fused)
      - FPN + lightweight PAN with ECA
      - Detection heads: direct 1×1 projections, NO shared 3×3 conv
        (FPN features already carry sufficient spatial context)

    Pure conv — no residual blocks, no skip connections, no shared
    spatial mixing before task projections.
    """

    def __init__(self, config, in_channels=1, out_channels=None, base_ch=None):
        super().__init__()
        self.config = config
        self.detection_protocol = str(getattr(config, "DETECTION_PROTOCOL", "anchor_free_tal")).strip().lower()
        out_channels = config.get_detector_output_channels() if out_channels is None else out_channels
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

        if self.detection_protocol == "anchor_free_tal":
            reg_max = int(getattr(config, "ANCHOR_FREE_REG_MAX", 16))
            head_ch = int(getattr(config, "ANCHOR_FREE_HEAD_CH", max(c2, 16)))
            self.anchor_free_heads = nn.ModuleList(
                AnchorFreeDetectBranch(ch, config.NUM_CLASSES, reg_max, head_ch) for ch in (c2, c4, c8)
            )
            return
        if self.detection_protocol != "anchor":
            raise ValueError("DETECTION_PROTOCOL must be 'anchor_free_tal' or 'anchor'.")

        # 方案A: direct 1×1 projections — no shared 3×3 conv.
        # FPN/PAN features already carry sufficient spatial context from
        # the stem and multi-scale fusion; each task branch reads directly
        # from its scale's features.
        # P3 head (in=c2, ~16ch)
        self.head_p3_box = nn.Conv2d(c2, 3 * 4, 1)
        self.head_p3_obj = nn.Conv2d(c2, 3 * 1, 1)
        self.head_p3_cls = nn.Conv2d(c2, out_channels - 3 * 5, 1)

        # P4 head (in=c4, ~32ch)
        self.head_p4_box = nn.Conv2d(c4, 3 * 4, 1)
        self.head_p4_obj = nn.Conv2d(c4, 3 * 1, 1)
        self.head_p4_cls = nn.Conv2d(c4, out_channels - 3 * 5, 1)

        # P5 head (in=c8, ~64ch)
        self.head_p5_box = nn.Conv2d(c8, 3 * 4, 1)
        self.head_p5_obj = nn.Conv2d(c8, 3 * 1, 1)
        self.head_p5_cls = nn.Conv2d(c8, out_channels - 3 * 5, 1)

    @staticmethod
    def _decode_head(box, obj, cls_conv, feat):
        """Direct 1×1 projection — no shared spatial mixing (方案A)."""
        b, _, h, w = feat.shape
        box_out = box(feat).contiguous().view(b, 3, 4, h, w)
        obj_out = obj(feat).contiguous().view(b, 3, 1, h, w)
        cls_out = cls_conv(feat).contiguous().view(b, 3, -1, h, w)
        return torch.cat([box_out, obj_out, cls_out], dim=2).contiguous().view(b, -1, h, w)

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

        # Detection heads — direct 1×1 per scale (方案A)
        prediction_features = (self.head_dropout(p3_fused), self.head_dropout(p4_pan), self.head_dropout(p5_pan))
        if self.detection_protocol == "anchor_free_tal":
            predictions = tuple(head(feat) for head, feat in zip(self.anchor_free_heads, prediction_features))
        else:
            predictions = (
                self._decode_head(self.head_p3_box, self.head_p3_obj, self.head_p3_cls, prediction_features[0]),
                self._decode_head(self.head_p4_box, self.head_p4_obj, self.head_p4_cls, prediction_features[1]),
                self._decode_head(self.head_p5_box, self.head_p5_obj, self.head_p5_cls, prediction_features[2]),
            )

        if return_features:
            return predictions, {"s8": p3_feat, "s16": p4_feat, "s32": p5_feat}
        return predictions
