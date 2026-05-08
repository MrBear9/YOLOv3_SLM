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


class ConvTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16), nn.SiLU())
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.SiLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.SiLU())
        self.stage1 = nn.Sequential(TeacherResidualBlock(16), TeacherResidualBlock(16))
        self.stage2 = nn.Sequential(TeacherResidualBlock(32), TeacherResidualBlock(32))
        self.stage3 = nn.Sequential(TeacherResidualBlock(64), TeacherResidualBlock(64), TeacherResidualBlock(64, dilation=2))
        self.skip1 = nn.Sequential(nn.Conv2d(16, 64, 1, bias=False), nn.BatchNorm2d(64), nn.SiLU())
        self.skip2 = nn.Sequential(nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64), nn.SiLU())
        self.context = nn.Sequential(
            TeacherResidualBlock(64),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.project = nn.Conv2d(32, 1, 1, bias=False)

    def forward(self, x):
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        x1 = self.stage1(self.conv1(x))
        x2 = self.stage2(self.conv2(x1))
        x3 = self.stage3(self.conv3(x2))
        skip1 = _interpolate_preserve_layout(self.skip1(x1), size=x3.shape[-2:], mode="bilinear", align_corners=False)
        skip2 = _interpolate_preserve_layout(self.skip2(x2), size=x3.shape[-2:], mode="bilinear", align_corners=False)
        f = self.context(x3 + skip1 + skip2)
        f = self.refine(f)
        f = torch.abs(self.project(f))
        f = _interpolate_preserve_layout(f, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(f)
