import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Basic blocks
# ═══════════════════════════════════════════════════════════════════════════

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SeparableConvBNAct(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.depthwise = ConvBNAct(channels, channels, 3, dilation=dilation, groups=channels)
        self.pointwise = ConvBNAct(channels, channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ECABlock(nn.Module):
    """Efficient Channel Attention — 1D conv over channel dimension.

    Adds negligible parameters (~kernel_size) while significantly improving
    channel-wise feature discrimination.  Standalone copy so the compact
    module stays self-contained.
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class CompactFusionBlock(nn.Module):
    """Small multi-dilation block for decoding optical intensity features."""

    def __init__(self, channels, dilations=(1, 2, 4)):
        super().__init__()
        hidden = max(channels // 2, 8)
        self.pre = ConvBNAct(channels, hidden, 1)
        self.branches = nn.ModuleList(SeparableConvBNAct(hidden, dilation=d) for d in dilations)
        self.fuse = ConvBNAct(hidden * len(dilations), channels, 1)
        self.shortcut = nn.Identity()

    def forward(self, x):
        y = self.pre(x)
        y = torch.cat([branch(y) for branch in self.branches], dim=1)
        return self.fuse(y) + self.shortcut(x)


class CompactFusionBlockECA(nn.Module):
    """CompactFusionBlock with ECA channel attention after fusion."""

    def __init__(self, channels, dilations=(1, 2, 4)):
        super().__init__()
        hidden = max(channels // 2, 8)
        self.pre = ConvBNAct(channels, hidden, 1)
        self.branches = nn.ModuleList(SeparableConvBNAct(hidden, dilation=d) for d in dilations)
        self.fuse = ConvBNAct(hidden * len(dilations), channels, 1)
        self.eca = ECABlock(channels)
        self.shortcut = nn.Identity()

    def forward(self, x):
        y = self.pre(x)
        y = torch.cat([branch(y) for branch in self.branches], dim=1)
        return self.eca(self.fuse(y)) + self.shortcut(x)


# ═══════════════════════════════════════════════════════════════════════════
# Detector variants
# ═══════════════════════════════════════════════════════════════════════════

class CompactOpticalDetector(nn.Module):
    """Anchor-free detector with heatmap, box-size, and center-offset heads."""

    def __init__(self, config, in_channels=1):
        super().__init__()
        self.config = config
        base = int(getattr(config, "COMPACT_BASE_CH", 16))
        head_ch = int(getattr(config, "COMPACT_HEAD_CH", 32))
        dilations = tuple(getattr(config, "COMPACT_DILATIONS", (1, 2, 4)))
        c1, c2, c4 = base, base * 2, base * 4

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, c1, 3),
            CompactFusionBlock(c1, dilations=dilations),
            ConvBNAct(c1, c2, 3, stride=2),
            CompactFusionBlock(c2, dilations=dilations),
            ConvBNAct(c2, c4, 3, stride=2),
            CompactFusionBlock(c4, dilations=dilations),
            CompactFusionBlock(c4, dilations=dilations),
        )

        self.heatmap_head = nn.Sequential(
            ConvBNAct(c4, head_ch, 3),
            nn.Conv2d(head_ch, config.NUM_CLASSES, 1),
        )
        self.wh_head = nn.Sequential(
            ConvBNAct(c4, head_ch, 3),
            nn.Conv2d(head_ch, 2, 1),
        )
        self.offset_head = nn.Sequential(
            ConvBNAct(c4, head_ch, 3),
            nn.Conv2d(head_ch, 2, 1),
        )
        self._init_heads()

    def _init_heads(self):
        final_heat = self.heatmap_head[-1]
        nn.init.constant_(final_heat.bias, -2.19)
        for head in (self.wh_head[-1], self.offset_head[-1]):
            nn.init.normal_(head.weight, std=0.001)
            nn.init.zeros_(head.bias)

    def forward(self, x):
        features = self.stem(x)
        return {
            "heatmap": self.heatmap_head(features),
            "wh": F.softplus(self.wh_head(features)),
            "offset": self.offset_head(features),
        }


class CompactOpticalDetectorV2(nn.Module):
    """Improved anchor-free detector (V2).

    Compared to V1 (109 K params):
      - ECA channel attention in every fusion block      ≈  +0.8 K
      - Extra CompactFusionBlock at deepest level        ≈ +10 K
      - Lightweight FPN: stride-2 detail fused into
        stride-4 features via lateral projection          ≈  +4 K
      - Deeper detection heads (2× ConvBNAct)            ≈ +14 K
                                                         ≈ 138 K total  (+27 %)
    """

    def __init__(self, config, in_channels=1):
        super().__init__()
        self.config = config
        base = int(getattr(config, "COMPACT_BASE_CH", 16))
        head_ch = int(getattr(config, "COMPACT_HEAD_CH", 32))
        dilations = tuple(getattr(config, "COMPACT_DILATIONS", (1, 2, 4)))
        c1, c2, c4 = base, base * 2, base * 4

        # ── Stem (kept non-Sequential to recover intermediate features) ──
        self.stem0 = ConvBNAct(in_channels, c1, 3)
        self.fuse0 = CompactFusionBlockECA(c1, dilations=dilations)
        self.down1 = ConvBNAct(c1, c2, 3, stride=2)           # → H/2
        self.fuse1 = CompactFusionBlockECA(c2, dilations=dilations)
        self.down2 = ConvBNAct(c2, c4, 3, stride=2)           # → H/4
        self.fuse2 = CompactFusionBlockECA(c4, dilations=dilations)
        self.fuse3 = CompactFusionBlockECA(c4, dilations=dilations)  # extra depth

        # ── Lightweight FPN: bring stride-2 texture into stride-4 features ──
        self.lateral_s2 = ConvBNAct(c2, c4 // 2, 1)           # project s2 detail
        self.fpn_fuse = ConvBNAct(c4 + c4 // 2, c4, 1)       # fuse s4 + s2 detail

        # ── Deeper detection heads ──
        self.heatmap_head = nn.Sequential(
            ConvBNAct(c4, head_ch, 3),
            ConvBNAct(head_ch, head_ch, 3),
            nn.Conv2d(head_ch, config.NUM_CLASSES, 1),
        )
        self.wh_head = nn.Sequential(
            ConvBNAct(c4, head_ch, 3),
            ConvBNAct(head_ch, head_ch, 3),
            nn.Conv2d(head_ch, 2, 1),
        )
        self.offset_head = nn.Sequential(
            ConvBNAct(c4, head_ch, 3),
            ConvBNAct(head_ch, head_ch, 3),
            nn.Conv2d(head_ch, 2, 1),
        )
        self._init_heads()

    def _init_heads(self):
        final_heat = self.heatmap_head[-1]
        nn.init.constant_(final_heat.bias, -2.19)
        for head in (self.wh_head[-1], self.offset_head[-1]):
            nn.init.normal_(head.weight, std=0.001)
            nn.init.zeros_(head.bias)

    def forward(self, x):
        # Stem with intermediate features
        s2 = self.fuse1(self.down1(self.fuse0(self.stem0(x))))          # H/2, c2
        s4 = self.fuse2(self.down2(s2))                                 # H/4, c4
        s4 = self.fuse3(s4)                                             # H/4, c4  (extra)

        # FPN: inject stride-2 detail
        s2_proj = self.lateral_s2(s2)                                   # H/2, c4//2
        s2_to_s4 = F.avg_pool2d(s2_proj, kernel_size=2, stride=2)      # H/4, c4//2
        fused = self.fpn_fuse(torch.cat([s4, s2_to_s4], dim=1))        # H/4, c4

        return {
            "heatmap": self.heatmap_head(fused),
            "wh": F.softplus(self.wh_head(fused)),
            "offset": self.offset_head(fused),
        }


class OpticalCompactDetector(nn.Module):
    def __init__(self, student, detector):
        super().__init__()
        self.student = student
        self.detector = detector

    def forward(self, x, return_feature=False):
        feature = self.student(x)
        pred = self.detector(feature)
        if return_feature:
            return feature, pred
        return pred
