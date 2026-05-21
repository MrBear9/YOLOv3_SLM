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


class ConvTeacherV2(nn.Module):
    """Light YOLOv8-style optical teacher with a 1-channel feature bridge."""

    def __init__(self, base_channels=24, c2f_blocks=2):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.stem = TeacherConvBNAct(1, c1, 3, 2)
        self.stage1 = TeacherC2f(c1, c1, c2f_blocks, shortcut=True)
        self.down2 = TeacherConvBNAct(c1, c2, 3, 2)
        self.stage2 = TeacherC2f(c2, c2, c2f_blocks + 1, shortcut=True)
        self.down3 = TeacherConvBNAct(c2, c3, 3, 2)
        self.stage3 = TeacherC2f(c3, c3, c2f_blocks + 1, shortcut=True)
        self.sppf = TeacherSPPF(c3, c3)
        self.skip1 = nn.Sequential(nn.Conv2d(c1, c3, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU())
        self.skip2 = nn.Sequential(nn.Conv2d(c2, c3, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU())
        self.context = nn.Sequential(
            TeacherC2f(c3, c3, c2f_blocks, shortcut=True),
            TeacherResidualBlock(c3, dilation=2),
        )
        self.refine = nn.Sequential(
            TeacherConvBNAct(c3, c2, 3),
            TeacherC2f(c2, c2, max(c2f_blocks, 1), shortcut=True),
            TeacherConvBNAct(c2, c1, 1),
        )
        self.bridge = nn.Conv2d(c1, 1, 1, bias=False)

    def forward(self, x):
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        x1 = self.stage1(self.stem(x))
        x2 = self.stage2(self.down2(x1))
        x3 = self.stage3(self.down3(x2))
        p3 = self.sppf(x3)
        skip1 = _interpolate_preserve_layout(self.skip1(x1), size=p3.shape[-2:], mode="bilinear", align_corners=False)
        skip2 = _interpolate_preserve_layout(self.skip2(x2), size=p3.shape[-2:], mode="bilinear", align_corners=False)
        f = self.context(p3 + skip1 + skip2)
        f = self.refine(f)
        f = torch.abs(self.bridge(f))
        f = _interpolate_preserve_layout(f, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(f)


class ConvTeacherV3(nn.Module):
    """YOLOv8-style teacher with residual+gate output (V2 backbone, new output head).

    Shares the same C2f feedforward backbone as ConvTeacherV2.  The difference
    is in the output: instead of ``sigmoid(abs(bridge(refine)))`` — a purely
    synthetic feature map — V3 produces::

        det_feature = gray + residual_scale * gate * residual

    where *residual* (tanh) learns what to add/subtract and *gate* (sigmoid)
    learns where to apply the modification.  The original gray signal is
    always preserved; the network only needs to learn sparse enhancements
    in object-relevant regions.

    Auxiliary heads (heat / box / edge) are provided for optional pretraining
    and are not used during joint detection training.
    """

    def __init__(self, base_channels=24, c2f_blocks=2, residual_scale=0.30):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.residual_scale = float(residual_scale)

        # --- Backbone (identical to ConvTeacherV2) ---
        self.stem = TeacherConvBNAct(1, c1, 3, 2)
        self.stage1 = TeacherC2f(c1, c1, c2f_blocks, shortcut=True)
        self.down2 = TeacherConvBNAct(c1, c2, 3, 2)
        self.stage2 = TeacherC2f(c2, c2, c2f_blocks + 1, shortcut=True)
        self.down3 = TeacherConvBNAct(c2, c3, 3, 2)
        self.stage3 = TeacherC2f(c3, c3, c2f_blocks + 1, shortcut=True)
        self.sppf = TeacherSPPF(c3, c3)
        self.skip1 = nn.Sequential(nn.Conv2d(c1, c3, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU())
        self.skip2 = nn.Sequential(nn.Conv2d(c2, c3, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU())
        self.context = nn.Sequential(
            TeacherC2f(c3, c3, c2f_blocks, shortcut=True),
            TeacherResidualBlock(c3, dilation=2),
        )
        self.refine = nn.Sequential(
            TeacherConvBNAct(c3, c2, 3),
            TeacherC2f(c2, c2, max(c2f_blocks, 1), shortcut=True),
            TeacherConvBNAct(c2, c1, 1),
        )

        # --- Output heads (replaces bridge + abs + sigmoid) ---
        self.residual_head = nn.Sequential(TeacherConvBNAct(c1, c1), nn.Conv2d(c1, 1, 1))
        self.gate_head = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Sigmoid())

        # --- Auxiliary pretraining heads (not used in joint training) ---
        self.heat_head = nn.Conv2d(c1, 1, 1)
        self.box_head = nn.Conv2d(c1, 1, 1)
        self.edge_head = nn.Conv2d(c1, 1, 1)

    def forward(self, x, return_aux=False):
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        gray = x.clamp(0.0, 1.0)

        # Backbone (same as V2 up to refine)
        x1 = self.stage1(self.stem(gray))
        x2 = self.stage2(self.down2(x1))
        x3 = self.stage3(self.down3(x2))
        p3 = self.sppf(x3)
        skip1 = _interpolate_preserve_layout(self.skip1(x1), size=p3.shape[-2:], mode="bilinear", align_corners=False)
        skip2 = _interpolate_preserve_layout(self.skip2(x2), size=p3.shape[-2:], mode="bilinear", align_corners=False)
        f = self.context(p3 + skip1 + skip2)
        f = self.refine(f)

        # Multi-scale features for distillation (before upsampling)
        feat_scale8 = f                                                # [B, c1, H/8, W/8] — deepest fused

        # Upsample to original resolution before applying heads
        f = _interpolate_preserve_layout(f, size=gray.shape[-2:], mode="bilinear", align_corners=False)

        # Residual + gate output
        residual = torch.tanh(self.residual_head(f))
        gate = self.gate_head(f)
        det_feature = (gray + self.residual_scale * gate * residual).clamp(0.0, 1.0)

        if return_aux:
            return {
                "det_feature": det_feature,
                "gray": gray,
                "heat_logits": self.heat_head(f),
                "box_logits": self.box_head(f),
                "edge_logits": self.edge_head(f),
                "gate": gate,
                "residual": residual,
                # Multi-scale features for detector distillation (training only)
                "feat_scale2": x1,       # [B, c1, H/2, W/2] — shallow texture
                "feat_scale4": x2,       # [B, c2, H/4, W/4] — mid-level structure
                "feat_scale8": feat_scale8,  # [B, c1, H/8, W/8] — deep semantics
            }
        return det_feature


class ConvTeacherUNet(nn.Module):
    """U-Net encoder-decoder teacher with residual+gate output.

    Architecture derived from the Temple all_train_teacher.py ConvTeacher:

    - Encoder: 4 stages (1→16→32→64→128) with stride-2 downsampling
      and residual blocks at each level, SPPF at the bottleneck.
    - Decoder: 3 upsampling stages with U-Net skip connections from
      each encoder level, preserving fine spatial details.
    - Output: ``det_feature = gray + residual_scale * gate * residual``
      where *residual* (tanh) and *gate* (sigmoid) are learned from
      the decoder's output.  Auxiliary heads (heat/box/edge) are
      available for pretraining.

    Compared to ConvTeacherV2/V3, the U-Net skip connections at every
    encoder-decoder level preserve more spatial detail, which is
    critical for precise bounding-box regression.
    """

    def __init__(self, base_channels=16, residual_scale=0.30):
        super().__init__()
        c1 = base_channels          # 16
        c2 = base_channels * 2      # 32
        c3 = base_channels * 4      # 64
        c4 = base_channels * 8      # 128
        self.residual_scale = float(residual_scale)

        # --- Encoder ---
        self.enc1 = nn.Sequential(TeacherConvBNAct(1, c1), TeacherResidualBlock(c1))
        self.enc2 = nn.Sequential(TeacherConvBNAct(c1, c2, 3, 2), TeacherResidualBlock(c2))
        self.enc3 = nn.Sequential(TeacherConvBNAct(c2, c3, 3, 2), TeacherResidualBlock(c3))
        self.enc4 = nn.Sequential(
            TeacherConvBNAct(c3, c4, 3, 2),
            TeacherResidualBlock(c4),
            TeacherSPPF(c4, c4),
        )

        # --- Decoder (U-Net skip connections) ---
        self.dec3 = nn.Sequential(TeacherConvBNAct(c4 + c3, c3), TeacherResidualBlock(c3))
        self.dec2 = nn.Sequential(TeacherConvBNAct(c3 + c2, c2), TeacherResidualBlock(c2))
        self.dec1 = nn.Sequential(TeacherConvBNAct(c2 + c1, c1), TeacherResidualBlock(c1))

        # --- Output heads ---
        self.residual_head = nn.Sequential(TeacherConvBNAct(c1, c1), nn.Conv2d(c1, 1, 1))
        self.gate_head = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Sigmoid())

        # --- Auxiliary pretraining heads ---
        self.heat_head = nn.Conv2d(c1, 1, 1)
        self.box_head = nn.Conv2d(c1, 1, 1)
        self.edge_head = nn.Conv2d(c1, 1, 1)

    def forward(self, x, return_aux=False):
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        gray = x.clamp(0.0, 1.0)

        # Encoder
        e1 = self.enc1(gray)                                               # [B, c1, H,   W]
        e2 = self.enc2(e1)                                                 # [B, c2, H/2, W/2]
        e3 = self.enc3(e2)                                                 # [B, c3, H/4, W/4]
        e4 = self.enc4(e3)                                                 # [B, c4, H/8, W/8]

        # Decoder with U-Net skip connections at every level
        d3 = _interpolate_preserve_layout(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))                        # [B, c3, H/4, W/4]

        d2 = _interpolate_preserve_layout(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))                        # [B, c2, H/2, W/2]

        d1 = _interpolate_preserve_layout(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))                        # [B, c1, H,   W]

        # Residual + gate output
        residual = torch.tanh(self.residual_head(d1))
        gate = self.gate_head(d1)
        det_feature = (gray + self.residual_scale * gate * residual).clamp(0.0, 1.0)

        if return_aux:
            return {
                "det_feature": det_feature,
                "gray": gray,
                "heat_logits": self.heat_head(d1),
                "box_logits": self.box_head(d1),
                "edge_logits": self.edge_head(d1),
                "gate": gate,
                "residual": residual,
                # Multi-scale features for detector distillation (training only)
                "feat_scale2": d2,          # [B, c2, H/2, W/2] — decoder mid-level
                "feat_scale4": d3,          # [B, c3, H/4, W/4] — decoder upper
                "feat_scale8": e4,          # [B, c4, H/8, W/8] — bottleneck deepest
            }
        return det_feature


def build_teacher(config=None):
    arch = str(getattr(config, "TEACHER_ARCH", "convteacher_v2") if config is not None else "convteacher_v2").strip().lower()
    if arch in {"convteacher", "convteacher_v1", "v1", "legacy"}:
        return ConvTeacher()
    if arch in {"convteacher_v2", "v2", "light_yolov8", "yolov8_light"}:
        base_channels = int(getattr(config, "TEACHER_V2_BASE_CHANNELS", 24) if config is not None else 24)
        c2f_blocks = int(getattr(config, "TEACHER_V2_C2F_BLOCKS", 2) if config is not None else 2)
        return ConvTeacherV2(base_channels=base_channels, c2f_blocks=c2f_blocks)
    if arch in {"convteacher_v3", "v3"}:
        base_channels = int(getattr(config, "TEACHER_V3_BASE_CHANNELS", 24) if config is not None else 24)
        c2f_blocks = int(getattr(config, "TEACHER_V3_C2F_BLOCKS", 2) if config is not None else 2)
        residual_scale = float(getattr(config, "TEACHER_V3_RESIDUAL_SCALE", 0.30) if config is not None else 0.30)
        return ConvTeacherV3(base_channels=base_channels, c2f_blocks=c2f_blocks, residual_scale=residual_scale)
    if arch in {"convteacher_unet", "unet"}:
        base_channels = int(getattr(config, "TEACHER_UNET_BASE_CHANNELS", 16) if config is not None else 16)
        residual_scale = float(getattr(config, "TEACHER_UNET_RESIDUAL_SCALE", 0.30) if config is not None else 0.30)
        return ConvTeacherUNet(base_channels=base_channels, residual_scale=residual_scale)
    raise ValueError(f"Unsupported TEACHER_ARCH: {arch}")
