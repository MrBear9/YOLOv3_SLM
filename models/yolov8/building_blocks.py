"""YOLOv8 building blocks.

Shared NN modules: ConvBNAct, ResBlock, ECABlock, Bottleneck, C2f, SPPF, C2fECA.
Legacy anchor detection branches (EnhancedDetectBranch, YOLOv8AnchorDetectBranch)
were removed together with the anchor protocol.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwisePointwiseConv(nn.Module):
    """Spatial depthwise convolution followed by pointwise channel mixing.

    A 1x1 kernel is treated as a pure pointwise projection.  This keeps the
    block useful at fusion boundaries without adding a redundant depthwise
    operation.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        activate=True,
    ):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("DepthwisePointwiseConv requires a positive odd kernel size.")
        if kernel_size == 1:
            self.spatial = nn.Identity()
        else:
            padding = dilation * (kernel_size // 2)
            self.spatial = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation=dilation,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(),
            )
        point_stride = stride if kernel_size == 1 else 1
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, point_stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU() if activate else nn.Identity(),
        )

    def forward(self, x):
        return self.pointwise(self.spatial(x))


class GlobalLocalGatedBlock(nn.Module):
    """Fuse fine local evidence and coarse global context through a residual gate.

    The local branch preserves small and partially occluded responses with a
    depthwise 3x3 path.  The global branch mixes a compact spatial grid and is
    interpolated back to the local resolution.  A learned per-pixel,
    per-channel gate selects their contribution before a layer-scaled residual
    update.  The block is identity-biased at initialization.
    """

    def __init__(self, channels, context_grid=6, dilation=1, residual_init=0.10):
        super().__init__()
        if channels < 1:
            raise ValueError("GlobalLocalGatedBlock channels must be positive.")
        self.context_grid = max(int(context_grid), 1)
        self.local_path = DepthwisePointwiseConv(
            channels, channels, 3, dilation=max(int(dilation), 1)
        )
        self.global_path = DepthwisePointwiseConv(channels, channels, 3)
        self.gate = nn.Sequential(
            nn.Conv2d(
                channels, channels, 1, groups=channels, bias=True
            ),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), float(residual_init))
        )
        self.out_act = nn.SiLU()

    def forward(self, x):
        local = self.local_path(x)
        grid_h = min(self.context_grid, x.shape[-2])
        grid_w = min(self.context_grid, x.shape[-1])
        global_context = F.adaptive_avg_pool2d(x, (grid_h, grid_w))
        global_context = self.global_path(global_context)
        global_context = F.interpolate(
            global_context, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        blend = self.gate(local + global_context)
        fused = blend * local + (1.0 - blend) * global_context
        return self.out_act(x + self.residual_scale * fused)


class GlobalLocalStage(nn.Module):
    """Efficient resize/projection followed by one or more gated context blocks."""

    def __init__(self, in_channels, out_channels, stride=1, num_blocks=1, context_grid=6):
        super().__init__()
        self.projection = DepthwisePointwiseConv(
            in_channels, out_channels, 3, stride=stride
        )
        self.blocks = nn.Sequential(
            *[
                GlobalLocalGatedBlock(
                    out_channels,
                    context_grid=context_grid,
                    dilation=1 + (index % 2),
                )
                for index in range(max(int(num_blocks), 1))
            ]
        )

    def forward(self, x):
        return self.blocks(self.projection(x))


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


