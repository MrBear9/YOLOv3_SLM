import torch
import torch.nn as nn

from models.runtime import prepare_conv_tensor
from models.teacher import ConvTeacher


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


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
        box = self.box(x).view(b, 3, 4, h, w)
        obj = self.obj(x).view(b, 3, 1, h, w)
        cls = self.cls(x).view(b, 3, -1, h, w)
        return torch.cat([box, obj, cls], dim=2).view(b, -1, h, w)


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
        self.head_p3 = YOLOv8AnchorDetectBranch(base_ch * 2, out_channels)
        self.head_p4 = YOLOv8AnchorDetectBranch(base_ch * 4, out_channels)
        self.head_p5 = YOLOv8AnchorDetectBranch(base_ch * 8, out_channels)

    def forward(self, x):
        x = self.stem(x)
        x320 = self.c2f1(self.down1(x))
        x160 = self.c2f2(self.down2(x320))
        x80 = self.c2f3(self.down3(x160))
        x40 = self.c2f4(self.down4(x80))
        p5 = self.sppf(self.down5(x40))
        p4 = self.fuse_p4(torch.cat([self.up_p5(p5), x40], dim=1))
        p3 = self.fuse_p3(torch.cat([self.up_p4(p4), x80], dim=1))
        p4 = self.pan_p4(torch.cat([self.down_p3(p3), p4], dim=1))
        p5 = self.pan_p5(torch.cat([self.down_p4(p4), p5], dim=1))
        return self.head_p3(p3), self.head_p4(p4), self.head_p5(p5)


class TeacherWithYOLOv8AnchorDetector(nn.Module):
    def __init__(self, config, teacher=None, detector=None):
        super().__init__()
        self.config = config
        self.teacher = ConvTeacher() if teacher is None else teacher
        self.detector = YOLOv8AnchorHead(config, in_channels=1, out_channels=config.get_detector_output_channels()) if detector is None else detector

    def forward(self, x, return_feature=False):
        x = prepare_conv_tensor(self.config, x)
        features = self.teacher(x)
        detections = self.detector(prepare_conv_tensor(self.config, features))
        if return_feature:
            return features, detections
        return detections
