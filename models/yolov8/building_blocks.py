"""YOLOv8 building blocks and detection branches.

Shared NN modules used by the various detection head variants:
ConvBNAct, ResBlock, ECABlock, Bottleneck, C2f, SPPF, C2fECA,
EnhancedDetectBranch, YOLOv8AnchorDetectBranch.
"""

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Lightweight residual block — simpler than C2f, FPGA-friendly."""

    def __init__(self, channels):
        super().__init__()
        self.cv1 = ConvBNAct(channels, channels, 3)
        self.cv2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, bias=False), nn.BatchNorm2d(channels))
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.cv2(self.cv1(x)))


class ECABlock(nn.Module):
    """Efficient Channel Attention — 1D conv over channel dimension."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class Bottleneck(nn.Module):
    def __init__(self, channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden = max(int(channels * expansion), 8)
        self.cv1 = ConvBNAct(channels, hidden, 1)
        self.cv2 = ConvBNAct(hidden, channels, 3)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut else y


class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2, shortcut=False, expansion=0.5):
        super().__init__()
        hidden = max(int(out_channels * expansion), 8)
        self.cv1 = ConvBNAct(in_channels, 2 * hidden, 1)
        self.blocks = nn.ModuleList(Bottleneck(hidden, shortcut=shortcut, expansion=1.0) for _ in range(num_blocks))
        self.cv2 = ConvBNAct((2 + num_blocks) * hidden, out_channels, 1)

    def forward(self, x):
        parts = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.cv2(torch.cat(parts, dim=1))


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden = max(in_channels // 2, 8)
        self.cv1 = ConvBNAct(in_channels, hidden, 1)
        self.cv2 = ConvBNAct(hidden * 4, out_channels, 1)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class C2fECA(nn.Module):
    """C2f block with ECA channel attention at output.

    Drop-in replacement for C2f — same interface, slightly more expressive.
    The ECA block adds negligible parameters (~1K per block) but improves
    channel-wise feature recalibration, which is critical for small-object
    detection on optical features.
    """

    def __init__(self, in_channels, out_channels, num_blocks=2, shortcut=False, expansion=0.5):
        super().__init__()
        hidden = max(int(out_channels * expansion), 8)
        self.cv1 = ConvBNAct(in_channels, 2 * hidden, 1)
        self.blocks = nn.ModuleList(Bottleneck(hidden, shortcut=shortcut, expansion=1.0) for _ in range(num_blocks))
        self.cv2 = ConvBNAct((2 + num_blocks) * hidden, out_channels, 1)
        self.eca = ECABlock(out_channels)

    def forward(self, x):
        parts = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.eca(self.cv2(torch.cat(parts, dim=1)))


class EnhancedDetectBranch(nn.Module):
    """Enhanced decoupled detection branch with residual + ECA.

    Compared to the original 2-layer branch:
      - 3 ConvBNAct layers with channel expansion (in → 2×hidden → hidden → out)
      - Residual shortcut from first layer to third layer
      - ECA attention after residual addition

    This design provides deeper feature extraction with better gradient flow,
    which is essential for improving localization and classification accuracy.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden = max(in_channels, 64)
        if out_channels % 3 != 0 or out_channels < 18:
            raise ValueError(f"out_channels must be 3 * (5 + num_classes), got {out_channels}")
        cls_channels = out_channels - 3 * 5

        # Shared feature extractor: 3-layer with residual + ECA
        self.shared = nn.Sequential(
            ConvBNAct(in_channels, hidden * 2, 3),
            ConvBNAct(hidden * 2, hidden, 3),
        )
        self.residual_proj = nn.Conv2d(in_channels, hidden, 1, bias=False)
        self.eca = ECABlock(hidden)

        # Decoupled heads: each is a single 1×1 conv on shared features
        self.box_head = nn.Conv2d(hidden, 3 * 4, 1)
        self.obj_head = nn.Conv2d(hidden, 3 * 1, 1)
        self.cls_head = nn.Conv2d(hidden, cls_channels, 1)

    def forward(self, x):
        b, _, h, w = x.shape
        feat = self.shared(x) + self.residual_proj(x)
        feat = self.eca(feat)
        box = self.box_head(feat).contiguous().view(b, 3, 4, h, w)
        obj = self.obj_head(feat).contiguous().view(b, 3, 1, h, w)
        cls = self.cls_head(feat).contiguous().view(b, 3, -1, h, w)
        return torch.cat([box, obj, cls], dim=2).contiguous().view(b, -1, h, w)


class YOLOv8AnchorDetectBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden = max(in_channels, 64)
        if out_channels % 3 != 0 or out_channels < 18:
            raise ValueError(f"out_channels must be 3 * (5 + num_classes), got {out_channels}")
        cls_channels = out_channels - 3 * 5
        self.box = nn.Sequential(ConvBNAct(in_channels, hidden, 3), ConvBNAct(hidden, hidden, 3), nn.Conv2d(hidden, 3 * 4, 1))
        self.obj = nn.Sequential(ConvBNAct(in_channels, hidden, 3), ConvBNAct(hidden, hidden, 3), nn.Conv2d(hidden, 3 * 1, 1))
        self.cls = nn.Sequential(ConvBNAct(in_channels, hidden, 3), ConvBNAct(hidden, hidden, 3), nn.Conv2d(hidden, cls_channels, 1))

    def forward(self, x):
        b, _, h, w = x.shape
        box = self.box(x).contiguous().view(b, 3, 4, h, w)
        obj = self.obj(x).contiguous().view(b, 3, 1, h, w)
        cls = self.cls(x).contiguous().view(b, 3, -1, h, w)
        return torch.cat([box, obj, cls], dim=2).contiguous().view(b, -1, h, w)
