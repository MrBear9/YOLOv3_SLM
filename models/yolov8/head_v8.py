import torch
import torch.nn as nn
import torch.nn.functional as F

from models.runtime import prepare_conv_tensor, should_use_channels_last
from models.teacher import build_teacher


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

    def _preserve_layout_after_resize(self, x):
        if x.is_floating_point() and x.dim() == 4 and should_use_channels_last(self.config):
            return x.contiguous(memory_format=torch.channels_last)
        return x.contiguous() if x.is_floating_point() else x

    def forward(self, x):
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
        return self.head_p3(p3), self.head_p4(p4), self.head_p5(p5)


class YOLOLightHead(nn.Module):
    """Lightweight FPGA-friendly detection head.

    Simplified compared to YOLOv8AnchorHead:
      - No C2f blocks → simple ResBlock instead
      - No PAN bottom-up path → top-down FPN only
      - No multi-branch detection heads → 1×1 Conv direct output
      - ECA attention for lightweight channel reweighting

    Output format is identical to YOLOv8AnchorHead (anchor-based).
    """

    def __init__(self, config, in_channels=1, out_channels=None, base_ch=None):
        super().__init__()
        self.config = config
        out_channels = config.get_detector_output_channels() if out_channels is None else out_channels
        c = base_ch if base_ch is not None else int(getattr(config, "YOLO_LIGHT_BASE_CH", 16))
        c2, c4, c8 = c * 2, c * 4, c * 8

        # Stem: 1 → c → c2 → c4 → c8  (down to stride 8)
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, c),
            ResBlock(c),
            ConvBNAct(c, c2, 3, 2),
            ResBlock(c2),
            ConvBNAct(c2, c4, 3, 2),
            ResBlock(c4),
            ConvBNAct(c4, c8, 3, 2),
            ResBlock(c8),
        )

        # P3 / P4 / P5 paths
        self.p4_path = nn.Sequential(ConvBNAct(c8, c8, 3, 2), ResBlock(c8))
        self.p5_path = nn.Sequential(ConvBNAct(c8, c8, 3, 2), ResBlock(c8), SPPF(c8, c8))

        # Top-down FPN with ECA
        self.fuse_p4 = nn.Sequential(ConvBNAct(c8 * 2, c4), ResBlock(c4), ECABlock(c4))
        self.fuse_p3 = nn.Sequential(ConvBNAct(c4 + c8, c2), ResBlock(c2), ECABlock(c2))

        # 1×1 Conv detection heads
        self.head_p3 = nn.Conv2d(c2, out_channels, 1)
        self.head_p4 = nn.Conv2d(c4, out_channels, 1)
        self.head_p5 = nn.Conv2d(c8, out_channels, 1)

    def forward(self, x, return_features=False):
        p3_feat = self.stem(x)                                           # [B, c8, 80, 80]
        p4_feat = self.p4_path(p3_feat)                                  # [B, c8, 40, 40]
        p5_feat = self.p5_path(p4_feat)                                  # [B, c8, 20, 20]

        p5_up = F.interpolate(p5_feat, size=p4_feat.shape[-2:], mode="nearest")
        p4_fused = self.fuse_p4(torch.cat([p5_up, p4_feat], dim=1))      # [B, c4, 40, 40]

        p4_up = F.interpolate(p4_fused, size=p3_feat.shape[-2:], mode="nearest")
        p3_fused = self.fuse_p3(torch.cat([p4_up, p3_feat], dim=1))      # [B, c2, 80, 80]

        preds = (self.head_p3(p3_fused), self.head_p4(p4_fused), self.head_p5(p5_feat))
        if return_features:
            return preds, {"s8": p3_feat, "s16": p4_feat, "s32": p5_feat}
        return preds


def build_detector_head(config, in_channels=1, out_channels=None):
    """Factory: build the configured detector head type."""
    head_type = str(getattr(config, "DETECTOR_HEAD_TYPE", "yolov8_anchor")).strip().lower()
    out_channels = config.get_detector_output_channels() if out_channels is None else out_channels
    if head_type in {"light", "yolo_light"}:
        base_ch = int(getattr(config, "YOLO_LIGHT_BASE_CH", 16))
        return YOLOLightHead(config, in_channels=in_channels, out_channels=out_channels, base_ch=base_ch)
    base_ch = int(getattr(config, "YOLOV8_BASE_CHANNELS", 32))
    c2f_blocks = int(getattr(config, "YOLOV8_C2F_BLOCKS", 3))
    return YOLOv8AnchorHead(config, in_channels=in_channels, out_channels=out_channels, base_ch=base_ch, c2f_blocks=c2f_blocks)


class TeacherWithDetector(nn.Module):
    """Teacher + Detector wrapper.

    Supports both YOLOv8AnchorHead and YOLOLightHead via
    ``build_detector_head(config)``.  The detector type is controlled by
    ``DETECTOR_HEAD_TYPE`` in config.
    """

    def __init__(self, config, teacher=None, detector=None):
        super().__init__()
        self.config = config
        self.teacher = build_teacher(config) if teacher is None else teacher
        self.detector = build_detector_head(config, in_channels=1) if detector is None else detector

    def forward(self, x, return_feature=False, return_teacher_aux=False, return_det_features=False):
        x = prepare_conv_tensor(self.config, x)
        need_aux = return_teacher_aux or return_det_features
        teacher_out = self.teacher(x, return_aux=need_aux)
        teacher_feature = teacher_out["det_feature"] if need_aux else teacher_out
        det_out = self.detector(prepare_conv_tensor(self.config, teacher_feature), return_features=return_det_features)
        if return_det_features:
            detections, det_features = det_out
        else:
            detections, det_features = det_out, None

        if not return_feature and not need_aux:
            return detections
        result = []
        if return_feature:
            result.append(teacher_feature)
        result.append(detections)
        if return_teacher_aux:
            result.append(teacher_out)
        if return_det_features:
            result.append(det_features)
        return tuple(result)


# Backward-compatible alias
TeacherWithYOLOv8AnchorDetector = TeacherWithDetector
