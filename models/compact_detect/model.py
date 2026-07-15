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
