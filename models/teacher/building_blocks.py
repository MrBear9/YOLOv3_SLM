"""Teacher building blocks.

Shared NN modules used by the various teacher architecture variants:
SqueezeExcite, TeacherResidualBlock, TeacherConvBNAct, TeacherBottleneck,
TeacherC2f, TeacherSPPF, CIB, C2fCIB, FeedbackGuidance, RawImageBridge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _interpolate_preserve_layout(x, *args, **kwargs):
    channels_last = x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last)
    out = F.interpolate(x, *args, **kwargs)
    if out.dim() != 4:
        return out
    if channels_last:
        return out.contiguous(memory_format=torch.channels_last)
    return out.contiguous()


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class SwiGLUGate(nn.Module):
    """SwiGLU-style channel gate: AdaptiveAvgPool → 1×1 Conv → Sigmoid.

    Drop-in replacement for SqueezeExcite.  Unlike SE which compresses
    channels (reduction=8), this keeps full channel resolution for
    finer-grained per-channel attention.
    """

    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(self.pool(x))


class FeedbackGuidance(nn.Module):
    """Deep-feature feedback gate for OCA output.

    Compresses multi-scale features into a spatial attention map, then
    multiplicatively modulates the OCA output.  ``alpha`` starts at 0 so
    the gate is identity at init and gradually learns a non-trivial
    modulation signal.

    Args:
        in_channels_list: channel count of each input feature map.
        guide_channels: internal compression channels.
    """

    def __init__(self, in_channels_list, guide_channels=16):
        super().__init__()
        self.compress = nn.ModuleList(
            nn.Conv2d(c, guide_channels, 1, bias=False) for c in in_channels_list
        )
        n_inputs = len(in_channels_list)
        self.fuse = nn.Sequential(
            nn.Conv2d(guide_channels * n_inputs, guide_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(guide_channels),
            nn.SiLU(),
            nn.Conv2d(guide_channels, 1, 1),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, oca_out, *feats):
        h, w = oca_out.shape[2:]
        compressed = [
            _interpolate_preserve_layout(c(f), (h, w), mode="bilinear", align_corners=False)
            for c, f in zip(self.compress, feats)
        ]
        gate = self.fuse(torch.cat(compressed, dim=1))
        return oca_out * (1.0 + self.alpha * gate)


class RawImageBridge(nn.Module):
    """Bridge raw image features to a target spatial scale.

    Extracts edge/texture information from the original image via a
    lightweight conv branch and outputs at ``target_size`` for fusion
    with refine-stage features.
    """

    def __init__(self, out_channels=1):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.SiLU(),
            nn.Conv2d(4, out_channels, 1, bias=False),
        )

    def forward(self, raw_img, target_size):
        x = _interpolate_preserve_layout(raw_img, target_size, mode="bilinear", align_corners=False)
        return self.edge_conv(x)


class TeacherResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gate = SwiGLUGate(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.gate(out)
        return self.act(out + identity)


class TeacherConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TeacherBottleneck(nn.Module):
    def __init__(self, channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden = max(int(channels * expansion), 8)
        self.cv1 = TeacherConvBNAct(channels, hidden, 1)
        self.cv2 = TeacherConvBNAct(hidden, channels, 3)
        self.gate = SwiGLUGate(channels)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.gate(self.cv2(self.cv1(x)))
        return x + y if self.shortcut else y


class TeacherC2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2, shortcut=True, expansion=0.5):
        super().__init__()
        hidden = max(int(out_channels * expansion), 8)
        self.cv1 = TeacherConvBNAct(in_channels, 2 * hidden, 1)
        self.blocks = nn.ModuleList(TeacherBottleneck(hidden, shortcut=shortcut, expansion=1.0) for _ in range(num_blocks))
        self.cv2 = TeacherConvBNAct((2 + num_blocks) * hidden, out_channels, 1)

    def forward(self, x):
        parts = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.cv2(torch.cat(parts, dim=1))


class CIB(nn.Module):
    """Conv Inverted Bottleneck (YOLOv10-style efficient block).

    1x1 expansion → 3x3 depthwise → 1x1 projection, with an optional residual
    shortcut.  Provides more capacity than a plain bottleneck while keeping the
    FLOPs low, making it suitable for the optical-teacher backbone.
    """

    def __init__(self, channels, expansion=2.0, shortcut=True):
        super().__init__()
        hidden = max(int(channels * expansion), 8)
        self.cv1 = TeacherConvBNAct(channels, hidden, 1)
        self.cv2 = TeacherConvBNAct(hidden, hidden, 3, groups=hidden)
        self.cv3 = nn.Sequential(
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU()
        self.shortcut = shortcut

    def forward(self, x):
        identity = x
        y = self.cv3(self.cv2(self.cv1(x)))
        if self.shortcut:
            y = self.act(y + identity)
        else:
            y = self.act(y)
        return y


class C2fCIB(nn.Module):
    """C2f block where each bottleneck is a CIB (YOLOv10-style).

    Used by the V2 phase-prediction backbone to replace heavier standard C2f
    stages without changing the V1/V3 code paths.
    """

    def __init__(self, in_channels, out_channels, num_blocks=2, shortcut=True, expansion=0.5, bottle_expansion=2.0):
        super().__init__()
        hidden = max(int(out_channels * expansion), 8)
        self.cv1 = TeacherConvBNAct(in_channels, 2 * hidden, 1)
        self.blocks = nn.ModuleList(
            CIB(hidden, expansion=bottle_expansion, shortcut=shortcut) for _ in range(num_blocks)
        )
        self.cv2 = TeacherConvBNAct((2 + num_blocks) * hidden, out_channels, 1)

    def forward(self, x):
        parts = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.cv2(torch.cat(parts, dim=1))


class TeacherSPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden = max(in_channels // 2, 8)
        self.cv1 = TeacherConvBNAct(in_channels, hidden, 1)
        self.cv2 = TeacherConvBNAct(hidden * 4, out_channels, 1)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


