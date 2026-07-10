"""Teacher building blocks.

Shared NN modules used by the various teacher architecture variants:
SqueezeExcite, TeacherResidualBlock, TeacherConvBNAct, TeacherBottleneck,
TeacherC2f, TeacherSPPF, SyntheticWavelengthComplexConv, CVOCAStage.
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


class TeacherResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcite(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
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
        self.se = SqueezeExcite(channels)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.se(self.cv2(self.cv1(x)))
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


class SyntheticWavelengthComplexConv(nn.Module):
    """Trainable complex convolution bank with non-coherent intensity fusion."""

    def __init__(self, channels, kernel_size=5, num_wavelengths=3):
        super().__init__()
        padding = kernel_size // 2
        self.num_wavelengths = max(int(num_wavelengths), 1)
        self.real_filters = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False)
            for _ in range(self.num_wavelengths)
        )
        self.imag_filters = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False)
            for _ in range(self.num_wavelengths)
        )
        self.phase_offsets = nn.Parameter(torch.linspace(0.0, 3.141592653589793, self.num_wavelengths))
        self.branch_logits = nn.Parameter(torch.zeros(self.num_wavelengths))
        self.mix = nn.Sequential(
            nn.Conv2d(channels * self.num_wavelengths, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

    def forward(self, real, imag):
        intensities = []
        branch_weights = torch.softmax(self.branch_logits, dim=0)
        for idx, (real_filter, imag_filter) in enumerate(zip(self.real_filters, self.imag_filters)):
            phase = self.phase_offsets[idx]
            cos_p = torch.cos(phase)
            sin_p = torch.sin(phase)
            real_rot = real * cos_p - imag * sin_p
            imag_rot = real * sin_p + imag * cos_p
            out_real = real_filter(real_rot) - imag_filter(imag_rot)
            out_imag = real_filter(imag_rot) + imag_filter(real_rot)
            intensities.append(branch_weights[idx] * (out_real.square() + out_imag.square()))
        return self.mix(torch.cat(intensities, dim=1))


class CVOCAStage(nn.Module):
    """One optical feature stage: phase modulation, complex convolution, intensity readout."""

    def __init__(self, in_channels, out_channels, kernel_size=5, num_wavelengths=3, stride=1, residual=True):
        super().__init__()
        self.amp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        self.phase = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Tanh(),
        )
        self.optical_conv = SyntheticWavelengthComplexConv(
            out_channels,
            kernel_size=kernel_size,
            num_wavelengths=num_wavelengths,
        )
        self.post = nn.Sequential(
            TeacherConvBNAct(out_channels, out_channels, 1),
            TeacherResidualBlock(out_channels, dilation=1),
        )
        use_projection = residual and (stride != 1 or in_channels != out_channels)
        if residual and not use_projection:
            self.skip = nn.Identity()
        elif use_projection:
            self.skip = TeacherConvBNAct(in_channels, out_channels, 1, stride=stride)
        else:
            self.skip = None

    def forward(self, x):
        amp = F.softplus(self.amp(x))
        phase = 3.141592653589793 * self.phase(x)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        out = self.post(self.optical_conv(real, imag))
        if self.skip is not None:
            out = out + self.skip(x)
        return out
