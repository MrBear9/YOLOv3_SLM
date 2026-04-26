import yaml
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.ops import nms
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import matplotlib.patches as patches
from collections import Counter

# 本地实现所有需要的函数，避免依赖optical_teacher.py的配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def load_class_names(yaml_path):
    """从YAML文件加载类别名称"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    if 'names' in cfg:
        class_names = {i: name for i, name in enumerate(cfg['names'])}
        num_classes = len(cfg['names'])
    else:
        class_names = {0: 'object'}
        num_classes = 1
    
    return class_names, num_classes

def load_anchor_groups(anchor_yaml_path):
    """从外部YAML文件加载分组的YOLO锚点"""
    with open(anchor_yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    anchors = cfg.get('anchors')
    if anchors is None:
        raise ValueError(f"'anchors' not found in anchor config: {anchor_yaml_path}")
    if not isinstance(anchors, list) or len(anchors) != 3:
        raise ValueError(f"'anchors' must contain exactly 3 layers: {anchor_yaml_path}")

    normalized = []
    for layer_idx, layer_anchors in enumerate(anchors):
        if not isinstance(layer_anchors, list) or len(layer_anchors) != 3:
            raise ValueError(f"Layer {layer_idx} must contain exactly 3 anchors: {anchor_yaml_path}")

        layer_values = []
        for anchor_idx, anchor in enumerate(layer_anchors):
            if not isinstance(anchor, (list, tuple)) or len(anchor) != 2:
                raise ValueError(f"Anchor {anchor_idx} in layer {layer_idx} must be [w, h]: {anchor_yaml_path}")

            w = int(anchor[0])
            h = int(anchor[1])
            if w <= 0 or h <= 0:
                raise ValueError(f"Anchor {anchor_idx} in layer {layer_idx} must be positive: {anchor_yaml_path}")
            layer_values.append([w, h])
        normalized.append(layer_values)

    return normalized

def init_log_file():
    """初始化日志文件，使用ConfigYOLO路径"""
    os.makedirs(ConfigYOLO.LOG_ROOT_DIR, exist_ok=True)
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ConfigYOLO.TRAIN_START_TIME = start_time
    with open(ConfigYOLO.LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("光学教师网络训练日志\n")
        f.write("=" * 80 + "\n")
        f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

def log_to_file(message, also_print=True):
    """记录消息到文件，使用ConfigYOLO路径"""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_message = f"{timestamp} {message}"
    skip_file_write = ConfigYOLO.should_skip_file_log(message)
    
    if not skip_file_write:
        with open(ConfigYOLO.LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_message + "\n")
    
    if also_print:
        print(log_message)

def append_plain_log(message=""):
    with open(ConfigYOLO.LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + "\n")

def init_epoch_log_table():
    separator = ConfigYOLO.get_epoch_table_separator()
    append_plain_log("")
    append_plain_log(separator)
    append_plain_log(ConfigYOLO.get_epoch_table_header())
    append_plain_log(separator)

def _format_table_value(value, width, decimals=4):
    if value is None:
        return f"{'N/A':<{width}}"
    try:
        if np.isnan(value):
            return f"{'N/A':<{width}}"
    except TypeError:
        pass
    return f"{value:<{width}.{decimals}f}"

def log_epoch_table_row(epoch, phase, train_loss, val_loss, precision, recall, f1_score, map50, lr, best_status):
    phase_text = str(phase)[: ConfigYOLO.EPOCH_TABLE_PHASE_WIDTH - 1]
    append_plain_log(
        f"{epoch + 1:<{ConfigYOLO.EPOCH_TABLE_EPOCH_WIDTH}}"
        f"{phase_text:<{ConfigYOLO.EPOCH_TABLE_PHASE_WIDTH}}"
        f"{_format_table_value(train_loss, ConfigYOLO.EPOCH_TABLE_TRAIN_LOSS_WIDTH)}"
        f"{_format_table_value(val_loss, ConfigYOLO.EPOCH_TABLE_VAL_LOSS_WIDTH)}"
        f"{_format_table_value(precision, ConfigYOLO.EPOCH_TABLE_METRIC_WIDTH, 3)}"
        f"{_format_table_value(recall, ConfigYOLO.EPOCH_TABLE_METRIC_WIDTH, 3)}"
        f"{_format_table_value(f1_score, ConfigYOLO.EPOCH_TABLE_METRIC_WIDTH, 3)}"
        f"{_format_table_value(map50, ConfigYOLO.EPOCH_TABLE_METRIC_WIDTH, 3)}"
        f"{_format_table_value(lr, ConfigYOLO.EPOCH_TABLE_LR_WIDTH, 6)}"
        f"{str(best_status):<{ConfigYOLO.EPOCH_TABLE_BEST_WIDTH}}"
    )

# =========================================================
# 光学层（相位 + 幅度调制）
# =========================================================
class SLMLayer(nn.Module):
    def __init__(self, resolution, mode="phase"):
        super().__init__()
        assert mode in ["phase", "amp_phase"]
        self.mode = mode

        self.phase_raw = nn.Parameter(
            torch.rand(1, 1, *resolution) * 2 * np.pi
        )

        if self.mode == "amp_phase":
            self.amp_raw = nn.Parameter(
                torch.rand(1, 1, *resolution)
            )
        else:
            self.register_parameter("amp_raw", None)

    def forward(self, field):
        phase = torch.remainder(self.phase_raw, 2 * np.pi)
        mod = torch.exp(1j * phase)

        if self.mode == "amp_phase":
            amp = torch.sigmoid(self.amp_raw)
            mod = mod * amp

        return field * mod

class ASMPropagation(nn.Module):
    def __init__(self, distance, wavelength=None, pixel_size=None, resolution=None):
        super().__init__()
        if wavelength is None:
            wavelength = Config.WAVELENGTH
        if pixel_size is None:
            pixel_size = Config.PIXEL_SIZE
        if resolution is None:
            resolution = Config.RESOLUTION
        fx = torch.fft.fftfreq(resolution[0], pixel_size)
        fy = torch.fft.fftfreq(resolution[1], pixel_size)
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')
        k2 = 1 / wavelength**2 - FX**2 - FY**2
        k2 = torch.clamp(k2, min=0)
        H = torch.exp(1j * 2 * np.pi * distance * torch.sqrt(k2))
        self.register_buffer("H", H)

    def forward(self, field):
        return torch.fft.ifft2(torch.fft.fft2(field) * self.H)

# =========================================================
# 学生网络（多层光学传播）
# =========================================================
class OpticalStudent(nn.Module):
    def __init__(self, resolution=None, mode=None):
        super().__init__()
        if resolution is None:
            resolution = Config.RESOLUTION
        if mode is None:
            mode = Config.SLM_MODE
        self.slm1 = SLMLayer(resolution, mode)
        self.prop1 = ASMPropagation(Config.PROP_DISTANCE_1)
        self.slm2 = SLMLayer(resolution, mode)
        self.prop2 = ASMPropagation(Config.PROP_DISTANCE_2)
        self.enable_norm = False

    def forward(self, intensity):
        amp = torch.sqrt(intensity.clamp(min=0)+1e-8)
        field = torch.complex(amp, torch.zeros_like(amp))
        field = self.prop1(self.slm1(field))
        field = self.prop2(self.slm2(field))
        out = torch.abs(field)**2
        if self.enable_norm:
            out = out / (out.mean(dim=[2,3], keepdim=True) + 1e-6)
        return out

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
        scale = self.fc(self.pool(x))
        return x * scale


class TeacherResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcite(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + identity
        return self.act(out)

# =========================================================
# 教师网络（卷积版YOLOV3）
# =========================================================
class ConvTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )

        self.stage1 = nn.Sequential(
            TeacherResidualBlock(16),
            TeacherResidualBlock(16),
        )

        self.stage2 = nn.Sequential(
            TeacherResidualBlock(32),
            TeacherResidualBlock(32),
        )

        self.stage3 = nn.Sequential(
            TeacherResidualBlock(64),
            TeacherResidualBlock(64),
            TeacherResidualBlock(64, dilation=2),
        )

        self.skip1 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )

        self.skip2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )

        self.context = nn.Sequential(
            TeacherResidualBlock(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        self.project = nn.Conv2d(32, 1, kernel_size=1, bias=False)

    def forward(self, x):
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)

        x1 = self.stage1(self.conv1(x))
        x2 = self.stage2(self.conv2(x1))
        x3 = self.stage3(self.conv3(x2))

        skip1 = F.interpolate(self.skip1(x1), size=x3.shape[-2:], mode="bilinear", align_corners=False)
        skip2 = F.interpolate(self.skip2(x2), size=x3.shape[-2:], mode="bilinear", align_corners=False)
        f = self.context(x3 + skip1 + skip2)
        f = self.refine(f)
        f = self.project(f)
        f = torch.abs(f)
        f = F.interpolate(
            f, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.sigmoid(f)

# =========================================================
# 轻量化卷积块
# =========================================================
class LightConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(LightConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

# =========================================================
# 轻量化检测头
# =========================================================
class YOLOLightHead(nn.Module):
    def __init__(self, in_channels=1, out_channels=27):
        super(YOLOLightHead, self).__init__()
        base_ch = 32

        self.init_conv = LightConvBlock(in_channels, base_ch, kernel_size=3, stride=1)

        self.down_to_p5 = nn.Sequential(
            LightConvBlock(base_ch, base_ch * 2, stride=2),
            LightConvBlock(base_ch * 2, base_ch * 4, stride=2),
            LightConvBlock(base_ch * 4, base_ch * 8, stride=2),
            LightConvBlock(base_ch * 8, base_ch * 8, stride=2),
            LightConvBlock(base_ch * 8, base_ch * 8, stride=2)
        )

        self.up_p5_to_p4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse_p4 = LightConvBlock(base_ch * 8 + base_ch * 8, base_ch * 4)

        self.up_p4_to_p3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse_p3 = LightConvBlock(base_ch * 4 + base_ch * 8, base_ch * 2)

        self.head_p5 = nn.Conv2d(base_ch * 8, out_channels, 1)
        self.head_p4 = nn.Conv2d(base_ch * 4, out_channels, 1)
        self.head_p3 = nn.Conv2d(base_ch * 2, out_channels, 1)

    def forward(self, x):
        x_init = self.init_conv(x)

        x320 = self.down_to_p5[0](x_init)
        x160 = self.down_to_p5[1](x320)
        x80 = self.down_to_p5[2](x160)
        x40 = self.down_to_p5[3](x80)
        p5 = self.down_to_p5[4](x40)

        p5_up = self.up_p5_to_p4(p5)
        p4_fuse = torch.cat([p5_up, x40], dim=1)
        p4 = self.fuse_p4(p4_fuse)

        p4_up = self.up_p4_to_p3(p4)
        p3_fuse = torch.cat([p4_up, x80], dim=1)
        p3 = self.fuse_p3(p3_fuse)

        p5_out = self.head_p5(p5)
        p4_out = self.head_p4(p4)
        p3_out = self.head_p3(p3)

        return p3_out, p4_out, p5_out

# =========================================================
# 辅助函数
# =========================================================
def enhance_feature_for_display(feature_map):
    feature_map = np.asarray(feature_map, dtype=np.float32)
    low = np.percentile(feature_map, 2)
    high = np.percentile(feature_map, 98)
    if high - low < 1e-6:
        return np.zeros_like(feature_map)
    
    feature_map = np.clip((feature_map - low) / (high - low), 0.0, 1.0)
    return np.power(feature_map, 0.8)

def wh_iou_scalar(w1, h1, w2, h2, eps=1e-6):
    inter = min(float(w1), float(w2)) * min(float(h1), float(h2))
    union = float(w1) * float(h1) + float(w2) * float(h2) - inter + eps
    return inter / union

def draw_best_matching_anchor_boxes(ax, x_center, y_center, width, height):
    anchor_colors = ["#ffd166", "#00d1ff", "#ff5db1"]
    for scale_idx, scale_anchors in enumerate(Config.ANCHORS):
        best_anchor = scale_anchors[0]
        best_iou = -1.0
        for anchor_w, anchor_h in scale_anchors:
            match_iou = wh_iou_scalar(width, height, anchor_w, anchor_h)
            if match_iou > best_iou:
                best_iou = match_iou
                best_anchor = (anchor_w, anchor_h)

        anchor_w, anchor_h = best_anchor
        x1 = x_center - anchor_w / 2.0
        y1 = y_center - anchor_h / 2.0
        rect = patches.Rectangle(
            (x1, y1),
            anchor_w,
            anchor_h,
            linewidth=1.1,
            edgecolor=anchor_colors[scale_idx % len(anchor_colors)],
            facecolor="none",
            linestyle="--",
            alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(8, y1 - 4),
            f"A@{Config.STRIDES[scale_idx]}",
            color=anchor_colors[scale_idx % len(anchor_colors)],
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.25, edgecolor="none"),
        )


def build_teacher_target_map(targets, img_size, device, dtype):
    batch_size = len(targets)
    target_map = torch.zeros((batch_size, 1, img_size, img_size), device=device, dtype=dtype)
    foreground_mask = torch.zeros((batch_size, 1, img_size, img_size), device=device, dtype=torch.bool)
    sigma = max(Config.FEATURE_HEATMAP_SIGMA, 1e-3)
    box_fill_value = float(np.clip(Config.FEATURE_BOX_FILL_VALUE, 0.0, 1.0))
    core_fill_value = float(np.clip(Config.FEATURE_CORE_FILL_VALUE, box_fill_value, 1.0))
    core_ratio = float(np.clip(Config.FEATURE_CORE_RATIO, 0.1, 1.0))

    for batch_idx, sample_targets in enumerate(targets):
        if len(sample_targets) == 0:
            continue

        for target in sample_targets:
            if target.shape[0] < 5:
                continue
            x_center = float(target[1].item() * img_size)
            y_center = float(target[2].item() * img_size)
            width = max(float(target[3].item() * img_size), 1.0)
            height = max(float(target[4].item() * img_size), 1.0)
            area_ratio = min((width * height) / max(float(Config.FEATURE_LARGE_OBJECT_AREA), 1.0), 1.0)
            adaptive_box_fill = float(np.clip(
                box_fill_value + area_ratio * Config.FEATURE_LARGE_BOX_FILL_BOOST,
                0.0,
                1.0,
            ))
            adaptive_core_fill = float(np.clip(
                core_fill_value + area_ratio * Config.FEATURE_LARGE_CORE_FILL_BOOST,
                adaptive_box_fill,
                1.0,
            ))
            adaptive_core_ratio = float(np.clip(
                core_ratio + area_ratio * Config.FEATURE_LARGE_CORE_RATIO_BOOST,
                0.1,
                0.9,
            ))

            x1 = max(0, int(np.floor(x_center - width / 2.0)))
            y1 = max(0, int(np.floor(y_center - height / 2.0)))
            x2 = min(img_size, int(np.ceil(x_center + width / 2.0)))
            y2 = min(img_size, int(np.ceil(y_center + height / 2.0)))
            if x2 <= x1 or y2 <= y1:
                continue

            box_w = x2 - x1
            box_h = y2 - y1
            xs = torch.linspace(-1.0, 1.0, box_w, device=device, dtype=dtype)
            ys = torch.linspace(-1.0, 1.0, box_h, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            heat = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2.0 * sigma * sigma))
            region = target_map[batch_idx, 0, y1:y2, x1:x2]
            base_fill = torch.full_like(region, adaptive_box_fill)
            region_target = torch.maximum(torch.maximum(region, base_fill), heat)

            core_w = max(1, int(round(box_w * adaptive_core_ratio)))
            core_h = max(1, int(round(box_h * adaptive_core_ratio)))
            core_x1 = x1 + max(0, (box_w - core_w) // 2)
            core_y1 = y1 + max(0, (box_h - core_h) // 2)
            core_x2 = min(x2, core_x1 + core_w)
            core_y2 = min(y2, core_y1 + core_h)
            if core_x2 > core_x1 and core_y2 > core_y1:
                core_region = region_target[(core_y1 - y1):(core_y2 - y1), (core_x1 - x1):(core_x2 - x1)]
                core_fill = torch.full_like(core_region, adaptive_core_fill)
                region_target[(core_y1 - y1):(core_y2 - y1), (core_x1 - x1):(core_x2 - x1)] = torch.maximum(core_region, core_fill)

            target_map[batch_idx, 0, y1:y2, x1:x2] = region_target
            foreground_mask[batch_idx, 0, y1:y2, x1:x2] = True

    return target_map, foreground_mask


def compute_teacher_guidance_loss(teacher_feature, targets, detector_frozen=False):
    target_map, foreground_mask = build_teacher_target_map(
        targets=targets,
        img_size=teacher_feature.shape[-1],
        device=teacher_feature.device,
        dtype=teacher_feature.dtype,
    )
    background_mask = ~foreground_mask

    heatmap_loss = F.binary_cross_entropy(teacher_feature.clamp(1e-4, 1.0 - 1e-4), target_map)

    if foreground_mask.any():
        fg_mean = teacher_feature[foreground_mask].mean()
    else:
        fg_mean = torch.zeros((), device=teacher_feature.device, dtype=teacher_feature.dtype)

    if background_mask.any():
        bg_mean = teacher_feature[background_mask].mean()
    else:
        bg_mean = torch.zeros((), device=teacher_feature.device, dtype=teacher_feature.dtype)

    contrast_loss = (
        F.relu(Config.FEATURE_FOREGROUND_TARGET - fg_mean) +
        F.relu(bg_mean - Config.FEATURE_BACKGROUND_TARGET)
    )
    sparsity_loss = bg_mean
    tv_loss = (
        torch.abs(teacher_feature[:, :, 1:, :] - teacher_feature[:, :, :-1, :]).mean() +
        torch.abs(teacher_feature[:, :, :, 1:] - teacher_feature[:, :, :, :-1]).mean()
    )

    weights = Config.get_teacher_guidance_weights(detector_frozen=detector_frozen)
    total = (
        weights["heatmap"] * heatmap_loss +
        weights["contrast"] * contrast_loss +
        weights["sparsity"] * sparsity_loss +
        weights["tv"] * tv_loss
    )
    stats = {
        "feature_heatmap": float(heatmap_loss.detach().item()),
        "feature_contrast": float(contrast_loss.detach().item()),
        "feature_sparsity": float(sparsity_loss.detach().item()),
        "feature_tv": float(tv_loss.detach().item()),
        "feature_total": float(total.detach().item()),
    }
    return total, stats

def xywh_to_xyxy(boxes):
    half_w = boxes[:, 2] / 2
    half_h = boxes[:, 3] / 2
    return torch.stack([
        boxes[:, 0] - half_w,
        boxes[:, 1] - half_h,
        boxes[:, 0] + half_w,
        boxes[:, 1] + half_h
    ], dim=1)

def bbox_iou_xywh(box1, box2, ciou=False, eps=1e-7):
    box1 = box1.reshape(-1, 4)
    box2 = box2.reshape(-1, 4)
    box1_xyxy = xywh_to_xyxy(box1)
    box2_xyxy = xywh_to_xyxy(box2)

    inter_x1 = torch.max(box1_xyxy[:, 0], box2_xyxy[:, 0])
    inter_y1 = torch.max(box1_xyxy[:, 1], box2_xyxy[:, 1])
    inter_x2 = torch.min(box1_xyxy[:, 2], box2_xyxy[:, 2])
    inter_y2 = torch.min(box1_xyxy[:, 3], box2_xyxy[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    box1_area = (box1_xyxy[:, 2] - box1_xyxy[:, 0]).clamp(min=0) * (box1_xyxy[:, 3] - box1_xyxy[:, 1]).clamp(min=0)
    box2_area = (box2_xyxy[:, 2] - box2_xyxy[:, 0]).clamp(min=0) * (box2_xyxy[:, 3] - box2_xyxy[:, 1]).clamp(min=0)
    union = box1_area + box2_area - inter_area + eps
    iou = inter_area / union

    if not ciou:
        return iou

    center_dist = (box1[:, 0] - box2[:, 0]) ** 2 + (box1[:, 1] - box2[:, 1]) ** 2
    enc_x1 = torch.min(box1_xyxy[:, 0], box2_xyxy[:, 0])
    enc_y1 = torch.min(box1_xyxy[:, 1], box2_xyxy[:, 1])
    enc_x2 = torch.max(box1_xyxy[:, 2], box2_xyxy[:, 2])
    enc_y2 = torch.max(box1_xyxy[:, 3], box2_xyxy[:, 3])
    enc_w = (enc_x2 - enc_x1).clamp(min=0)
    enc_h = (enc_y2 - enc_y1).clamp(min=0)
    c2 = enc_w ** 2 + enc_h ** 2 + eps

    w1 = box1[:, 2].clamp(min=eps)
    h1 = box1[:, 3].clamp(min=eps)
    w2 = box2[:, 2].clamp(min=eps)
    h2 = box2[:, 3].clamp(min=eps)
    v = (4.0 / (np.pi ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    return iou - (center_dist / c2) - alpha * v

def bbox_iou_matrix_xywh(box1, box2, eps=1e-7):
    box1 = box1.reshape(-1, 4)
    box2 = box2.reshape(-1, 4)

    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((box1.shape[0], box2.shape[0]), device=box1.device, dtype=box1.dtype)

    box1_xyxy = xywh_to_xyxy(box1)
    box2_xyxy = xywh_to_xyxy(box2)

    inter_x1 = torch.maximum(box1_xyxy[:, None, 0], box2_xyxy[None, :, 0])
    inter_y1 = torch.maximum(box1_xyxy[:, None, 1], box2_xyxy[None, :, 1])
    inter_x2 = torch.minimum(box1_xyxy[:, None, 2], box2_xyxy[None, :, 2])
    inter_y2 = torch.minimum(box1_xyxy[:, None, 3], box2_xyxy[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (box1_xyxy[:, 2] - box1_xyxy[:, 0]).clamp(min=0) * (box1_xyxy[:, 3] - box1_xyxy[:, 1]).clamp(min=0)
    area2 = (box2_xyxy[:, 2] - box2_xyxy[:, 0]).clamp(min=0) * (box2_xyxy[:, 3] - box2_xyxy[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter_area + eps
    return inter_area / union

def box_intersection_over_smaller_xyxy(box1_xyxy, box2_xyxy, eps=1e-7):
    inter_x1 = torch.maximum(box1_xyxy[:, None, 0], box2_xyxy[None, :, 0])
    inter_y1 = torch.maximum(box1_xyxy[:, None, 1], box2_xyxy[None, :, 1])
    inter_x2 = torch.minimum(box1_xyxy[:, None, 2], box2_xyxy[None, :, 2])
    inter_y2 = torch.minimum(box1_xyxy[:, None, 3], box2_xyxy[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (box1_xyxy[:, 2] - box1_xyxy[:, 0]).clamp(min=0) * (box1_xyxy[:, 3] - box1_xyxy[:, 1]).clamp(min=0)
    area2 = (box2_xyxy[:, 2] - box2_xyxy[:, 0]).clamp(min=0) * (box2_xyxy[:, 3] - box2_xyxy[:, 1]).clamp(min=0)
    smaller_area = torch.minimum(area1[:, None], area2[None, :]).clamp(min=eps)
    return inter_area / smaller_area

def suppress_contained_detections(det_tensor, containment_ratio, area_ratio_limit, class_agnostic=False):
    if det_tensor.numel() == 0 or det_tensor.shape[0] <= 1:
        return det_tensor

    det_tensor = det_tensor[det_tensor[:, 4].argsort(descending=True)]
    boxes_xyxy = xywh_to_xyxy(det_tensor[:, :4])
    class_ids = det_tensor[:, 5]
    areas = det_tensor[:, 2] * det_tensor[:, 3]
    keep_mask = torch.ones(det_tensor.shape[0], dtype=torch.bool, device=det_tensor.device)

    for i in range(det_tensor.shape[0]):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, det_tensor.shape[0]):
            if not keep_mask[j]:
                continue
            if not class_agnostic and class_ids[i] != class_ids[j]:
                continue

            smaller_area = torch.minimum(areas[i], areas[j]).clamp(min=1e-6)
            larger_area = torch.maximum(areas[i], areas[j]).clamp(min=1e-6)
            if (smaller_area / larger_area).item() > area_ratio_limit:
                continue

            overlap_on_smaller = box_intersection_over_smaller_xyxy(
                boxes_xyxy[i:i + 1],
                boxes_xyxy[j:j + 1],
            )[0, 0].item()
            if overlap_on_smaller < containment_ratio:
                continue

            # Keep the higher-score / larger context box, suppress the contained fragment.
            if areas[i] >= areas[j]:
                keep_mask[j] = False
            else:
                keep_mask[i] = False
                break

    return det_tensor[keep_mask]

def weighted_mean(values, weights=None, eps=1e-6):
    if values.numel() == 0:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    if weights is None:
        return values.mean()

    weights = weights.to(device=values.device, dtype=values.dtype)
    if weights.shape != values.shape:
        weights = weights.expand_as(values)
    weighted_sum = (values * weights).sum()
    normalizer = weights.sum().clamp(min=eps)
    return weighted_sum / normalizer

def apply_nms(detections, nms_thresh, max_det, class_agnostic=None):
    if len(detections) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    if class_agnostic is None:
        class_agnostic = Config.AGNOSTIC_NMS

    det_tensor = torch.as_tensor(detections, dtype=torch.float32)
    boxes_xyxy = xywh_to_xyxy(det_tensor[:, :4])
    scores = det_tensor[:, 4]
    class_ids = det_tensor[:, 5]

    if class_agnostic:
        keep_indices = nms(boxes_xyxy, scores, nms_thresh)
        det_tensor = det_tensor[keep_indices]
    else:
        kept = []
        for cls_id in class_ids.unique(sorted=False):
            cls_mask = class_ids == cls_id
            keep_indices = nms(boxes_xyxy[cls_mask], scores[cls_mask], nms_thresh)
            kept.append(det_tensor[cls_mask][keep_indices])

        if len(kept) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        det_tensor = torch.cat(kept, dim=0)

    if Config.ENABLE_CONTAINMENT_SUPPRESSION:
        det_tensor = suppress_contained_detections(
            det_tensor,
            containment_ratio=Config.CONTAINMENT_SUPPRESS_RATIO,
            area_ratio_limit=Config.CONTAINMENT_AREA_RATIO_LIMIT,
            class_agnostic=class_agnostic,
        )

    det_tensor = det_tensor[det_tensor[:, 4].argsort(descending=True)]
    return det_tensor[:max_det].cpu().numpy()

def apply_classwise_nms(detections, nms_thresh, max_det):
    if len(detections) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    det_tensor = torch.as_tensor(detections, dtype=torch.float32)
    boxes_xyxy = xywh_to_xyxy(det_tensor[:, :4])
    scores = det_tensor[:, 4]
    class_ids = det_tensor[:, 5]
    kept = []

    for cls_id in class_ids.unique(sorted=False):
        cls_mask = class_ids == cls_id
        keep_indices = nms(boxes_xyxy[cls_mask], scores[cls_mask], nms_thresh)
        kept.append(det_tensor[cls_mask][keep_indices])

    if len(kept) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    det_tensor = torch.cat(kept, dim=0)
    det_tensor = det_tensor[det_tensor[:, 4].argsort(descending=True)]
    return det_tensor[:max_det].cpu().numpy()

def decode_detections(preds, conf_thresh=None, nms_thresh=None, max_det=None, img_size=None):
    """
    解码YOLO检测结果，返回边界框坐标、置信度和类别ID
    
    参数:
        preds: 预测结果列表 [p3, p4, p5]
        conf_thresh: 置信度阈值
        nms_thresh: NMS阈值
        max_det: 最大检测数量
        img_size: 图像尺寸
    
    返回:
        detections: 每个样本的检测结果列表，每个检测为 [x_center, y_center, w, h, conf, cls_id]
    """
    if conf_thresh is None:
        conf_thresh = Config.CONF_THRESH
    if nms_thresh is None:
        nms_thresh = Config.NMS_THRESH
    if max_det is None:
        max_det = Config.MAX_DET
    if img_size is None:
        img_size = Config.IMG_SIZE
    
    batch_size = preds[0].shape[0]
    detections = [[] for _ in range(batch_size)]
    strides = [8, 16, 32]
    anchors = Config.ANCHORS
    
    for i, pred in enumerate(preds):
        grid_h, grid_w = pred.shape[2], pred.shape[3]
        stride = strides[i]
        anchor_set = anchors[i]
        
        # Convert to (batch_size, grid_h, grid_w, 3, 5+num_classes)
        pred = pred.permute(0, 2, 3, 1).reshape(batch_size, grid_h, grid_w, 3, -1)
        
        # Apply sigmoid activation to the confidence scores
        obj_conf = torch.sigmoid(pred[..., 4])  # Object confidence
        cls_conf = torch.sigmoid(pred[..., 5:])  # 类别 confidence
        bbox_pred = pred[..., :4]  # 坐标预测 (tx, ty, tw, th)
        
        for b in range(batch_size):
            for gh in range(grid_h):
                for gw in range(grid_w):
                    for a in range(3):
                        # 获取目标置信度
                        obj_score = obj_conf[b, gh, gw, a].item()
                        if obj_score < conf_thresh:
                            continue
                        
                        # 获取类别置信度和类别ID
                        cls_scores = cls_conf[b, gh, gw, a]
                        cls_score, cls_id = cls_scores.max(dim=-1)
                        final_conf = obj_score * cls_score.item()
                        
                        if final_conf < conf_thresh:
                            continue
                        
                        # 解码边界框坐标(YOLO格式)
                        tx, ty, tw, th = bbox_pred[b, gh, gw, a]
                        
                        # 转换为绝对坐标
                        x_center = (gw + torch.sigmoid(tx).item()) * stride
                        y_center = (gh + torch.sigmoid(ty).item()) * stride
                        
                        # 解码宽度和高度
                        anchor_w, anchor_h = anchor_set[a]
                        w = anchor_w * torch.exp(torch.clamp(tw, min=-8.0, max=8.0)).item()
                        h = anchor_h * torch.exp(torch.clamp(th, min=-8.0, max=8.0)).item()
                        
                        # 限制边界框在图像范围内
                        x_center = max(0, min(x_center, img_size - 1))
                        y_center = max(0, min(y_center, img_size - 1))
                        w = max(1, min(w, img_size))
                        h = max(1, min(h, img_size))
                        
                        detections[b].append([x_center, y_center, w, h, final_conf, cls_id.item()])
    
    # 按置信度排序并限制最大检测数量
    for b in range(batch_size):
        if len(detections[b]) > 0:
            detections[b] = np.array(detections[b])
            # 按置信度降序排序
            sorted_indices = detections[b][:, 4].argsort()[::-1]
            detections[b] = detections[b][sorted_indices]
            # 限制最大检测数量
            if len(detections[b]) > max_det:
                detections[b] = detections[b][:max_det]
    
    return detections

def decode_detections_fixed(preds, conf_thresh=None, nms_thresh=None, max_det=None, img_size=None):
    if conf_thresh is None:
        conf_thresh = Config.CONF_THRESH
    if nms_thresh is None:
        nms_thresh = Config.NMS_THRESH
    if max_det is None:
        max_det = Config.MAX_DET
    if img_size is None:
        img_size = Config.IMG_SIZE

    batch_size = preds[0].shape[0]
    detections = [[] for _ in range(batch_size)]

    for i, pred in enumerate(preds):
        grid_h, grid_w = pred.shape[2], pred.shape[3]
        stride = Config.STRIDES[i]
        pred = pred.permute(0, 2, 3, 1).reshape(batch_size, grid_h, grid_w, 3, -1)

        obj_conf = torch.sigmoid(pred[..., 4])
        cls_conf = torch.sigmoid(pred[..., 5:])
        cls_score, cls_id = cls_conf.max(dim=-1)
        final_conf = obj_conf * cls_score
        valid_mask = (obj_conf >= conf_thresh) & (final_conf >= conf_thresh)

        if not valid_mask.any():
            continue

        device = pred.device
        dtype = pred.dtype
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h, device=device, dtype=dtype),
            torch.arange(grid_w, device=device, dtype=dtype),
            indexing='ij'
        )
        grid_x = grid_x.view(1, grid_h, grid_w, 1)
        grid_y = grid_y.view(1, grid_h, grid_w, 1)
        anchor_tensor = torch.tensor(Config.ANCHORS[i], device=device, dtype=dtype).view(1, 1, 1, 3, 2)

        tx = pred[..., 0]
        ty = pred[..., 1]
        tw = pred[..., 2]
        th = pred[..., 3]

        x_center = (grid_x + torch.sigmoid(tx)) * stride
        y_center = (grid_y + torch.sigmoid(ty)) * stride
        w = anchor_tensor[..., 0] * torch.exp(torch.clamp(tw, min=-8.0, max=8.0))
        h = anchor_tensor[..., 1] * torch.exp(torch.clamp(th, min=-8.0, max=8.0))

        x_center = x_center.clamp(0, img_size - 1)
        y_center = y_center.clamp(0, img_size - 1)
        w = w.clamp(1, img_size)
        h = h.clamp(1, img_size)

        for b in range(batch_size):
            sample_mask = valid_mask[b]
            if not sample_mask.any():
                continue

            sample_detections = torch.stack([
                x_center[b][sample_mask],
                y_center[b][sample_mask],
                w[b][sample_mask],
                h[b][sample_mask],
                final_conf[b][sample_mask],
                cls_id[b][sample_mask].to(dtype),
            ], dim=1)
            detections[b].append(sample_detections)

    for b in range(batch_size):
        if len(detections[b]) > 0:
            detections[b] = apply_nms(torch.cat(detections[b], dim=0), nms_thresh, max_det)
        else:
            detections[b] = np.zeros((0, 6), dtype=np.float32)

    return detections

decode_detections = decode_detections_fixed

class ConfigYOLO:
    # Data and outputs
    YAML_PATH = r"data\military\data.yaml"
    CLASS_NAMES = None
    NUM_CLASSES = None
    TEACHER_OUTPUT_DIR = r"output\OpticalTeacherYOLO_deep_teacher"
    LOG_ROOT_DIR = None
    LOG_FILE = None
    TIMESTAMP = None
    TRAIN_START_TIME = None

    # Runtime
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 640
    BATCH_SIZE = 8
    EPOCHS = 120

    # 经常调整的训练参数
    BOX_WEIGHT_BASE = 1.32  # 保留大目标回归力度，但略收一点，减少框体过大
    OBJ_WEIGHT_BASE = 0.56  # 从上一版回调，给真实目标更多正样本驱动
    NOOBJ_WEIGHT_BASE = 0.26  # 背景抑制保留，但不再过强
    CLS_WEIGHT_BASE = 0.32  # 保留分类监督，同时回到更接近旧稳态

    POSITION_PHASE_EPOCHS = 12  # 继续让前期定位学稳
    BALANCE_PHASE_EPOCHS = 8  # 缩短过渡段，减少中期被塑形权重拖慢

    IOU_THRESHOLD = 0.5  # 主要正样本匹配阈值；较大的值使正样本分配更严格，可能降低召回率
    POSITIVE_ANCHOR_IOU = 0.25  # 回退到旧版更稳的正样本匹配阈值
    MAX_POSITIVE_ANCHORS = 1  # 收回到单锚点分配，优先减少重复框
    NOOBJ_IGNORE_IOU = 0.6  # 避免把近邻潜在正样本也过度压成背景

    SMALL_OBJ_AREA = 32 * 32  # 将对象分组为小目标的阈值
    LARGE_OBJ_AREA = 128 * 128  # 将对象分组为大目标的阈值
    SMALL_OBJ_WEIGHT = 0.8  # 给小目标一点权重恢复，减少小目标漏检
    MEDIUM_OBJ_WEIGHT = 1.0  # 中等目标的基准损失权重
    LARGE_OBJ_WEIGHT = 1.5  # 保留大目标强调，但降低一个目标多框的倾向

    FOCAL_ALPHA = 0.28  # 向旧稳态回调，避免过度偏向负样本
    FOCAL_GAMMA = 1.6  # 略降，减少后期被困难背景样本牵制

    CONF_THRESH = 0.62  # 再收一点背景召回，优先抑制虚警和碎框
    NMS_THRESH = 0.18  # 对同类近邻框更强抑制，减少一个大目标多个小框
    MAX_DET = 3  # 单图输出继续收紧，减少画面噪声
    AGNOSTIC_NMS = False  # 多类别任务使用按类 NMS，避免不同类别互相压制
    ENABLE_CONTAINMENT_SUPPRESSION = True
    CONTAINMENT_SUPPRESS_RATIO = 0.88  # 小框大部分被同类大框包住时直接压掉
    CONTAINMENT_AREA_RATIO_LIMIT = 0.55  # 只抑制明显更小的碎框，避免压掉相邻独立目标

    # 验证和指标
    VAL_INTERVAL = 5
    METRIC_IOU_THRESHOLD = 0.5  # 较大的值在评估期间使TP匹配更严格，可能降低召回率，但可能增加假阳性

    # Teacher feature shaping and staged training
    DETECTOR_FREEZE_EPOCH = 85
    DETECTOR_FINETUNE_LR = 2e-4
    FEATURE_HEATMAP_WEIGHT_JOINT = 0.06
    FEATURE_HEATMAP_WEIGHT_FROZEN = 0.14
    FEATURE_CONTRAST_WEIGHT_JOINT = 0.024
    FEATURE_CONTRAST_WEIGHT_FROZEN = 0.055
    FEATURE_SPARSITY_WEIGHT_JOINT = 0.012
    FEATURE_SPARSITY_WEIGHT_FROZEN = 0.028
    FEATURE_TV_WEIGHT_JOINT = 0.004
    FEATURE_TV_WEIGHT_FROZEN = 0.008
    FEATURE_FOREGROUND_TARGET = 0.74
    FEATURE_BACKGROUND_TARGET = 0.05
    FEATURE_HEATMAP_SIGMA = 0.35
    FEATURE_BOX_FILL_VALUE = 0.14
    FEATURE_CORE_FILL_VALUE = 0.40
    FEATURE_CORE_RATIO = 0.48
    FEATURE_LARGE_OBJECT_AREA = 160 * 160
    FEATURE_LARGE_BOX_FILL_BOOST = 0.05
    FEATURE_LARGE_CORE_FILL_BOOST = 0.08
    FEATURE_LARGE_CORE_RATIO_BOOST = 0.12

    # Class balance sampler
    USE_CLASS_BALANCED_SAMPLER = True
    CLASS_BALANCE_POWER = 0.6  # 对少数类稍微更积极，进一步缓解 aircraft 过多
    MAX_CLASS_BALANCE_GAIN = 3.0  # 单类最高采样增益上限
    MAJORITY_ONLY_IMAGE_WEIGHT = 0.45  # 对纯多数类图像再多降一点权重
    EMPTY_IMAGE_SAMPLE_WEIGHT = 0.7  # 空标签图保留，但略微降权
    MIN_IMAGE_SAMPLE_WEIGHT = 0.35  # 防止样本权重过低导致几乎永不被采到

    # 模型和锚点
    STRIDES = [8, 16, 32]
    DEFAULT_ANCHORS = [
        [[26, 23], [47, 49], [100, 67]],  # P3: 小目标
        [[103, 169], [203, 107], [351, 177]],  # P4: 中等目标
        [[241, 354], [534, 299], [568, 528]],  # P5: 大目标
    ]
    ANCHOR_CONFIG_PATH = r"output\anchor_clustering\yolo_anchors.yaml"
    USE_EXTERNAL_ANCHORS = True
    ANCHORS = None
    ANCHOR_SOURCE = "default"

    # Optimizer and checkpoints
    LEARNING_RATE = 4e-4
    WEIGHT_DECAY = 3e-5
    OPTIMIZER = "Adam"
    TEACHER_INIT_MODE = "checkpoint"  # "scratch" 或 "checkpoint"
    TEACHER_INIT_CHECKPOINT = r"output\OpticalTeacherYOLO\teacher_final.pth"
    FREEZE_TEACHER = False
    SAVE_TEACHER_WEIGHTS = True

    # Visualization
    VIS_INTERVAL = 5
    VIS_BATCH_SIZE = 4
    VIS_DPI = 130
    VIS_DATASET_SPLIT = "val"  # 可视化时使用的数据集分割
    VIS_SEED = 20260421  # 保持与本次 ft 实验一致，便于前后对比
    VIS_CONF_THRESH = 0.70  # 可视化单独用更干净的显示阈值
    VIS_NMS_THRESH = 0.16
    VIS_MAX_DET = 3
    VIS_SHOW_BEST_MATCHED_ANCHORS = True
    VIS_MAX_GT_ANCHOR_OVERLAYS = 2

    # Logging table
    EPOCH_TABLE_EPOCH_WIDTH = 8
    EPOCH_TABLE_PHASE_WIDTH = 18
    EPOCH_TABLE_TRAIN_LOSS_WIDTH = 13
    EPOCH_TABLE_VAL_LOSS_WIDTH = 13
    EPOCH_TABLE_METRIC_WIDTH = 11
    EPOCH_TABLE_LR_WIDTH = 12
    EPOCH_TABLE_BEST_WIDTH = 8
    EPOCH_TABLE_BEST_MARK = "Yes"
    SKIP_FILE_LOG_MESSAGES = (
        "best checkpoint updated",
        "saved best model",
        "best model saved",
    )
    @classmethod
    def get_dynamic_weights(cls, epoch):
        if epoch < cls.POSITION_PHASE_EPOCHS:
            box_weight = cls.BOX_WEIGHT_BASE * 1.3
            cls_weight = cls.CLS_WEIGHT_BASE * 0.8
            phase = "position_focus"
            size_weights = {
                "small": cls.SMALL_OBJ_WEIGHT,
                "medium": cls.MEDIUM_OBJ_WEIGHT,
                "large": cls.LARGE_OBJ_WEIGHT,
            }
        elif epoch < cls.POSITION_PHASE_EPOCHS + cls.BALANCE_PHASE_EPOCHS:
            progress = (epoch - cls.POSITION_PHASE_EPOCHS) / max(cls.BALANCE_PHASE_EPOCHS, 1)
            box_weight = cls.BOX_WEIGHT_BASE * (1.3 - 0.3 * progress)
            cls_weight = cls.CLS_WEIGHT_BASE * (0.8 + 0.2 * progress)
            phase = "balance_transition"
            size_weights = {
                "small": cls.SMALL_OBJ_WEIGHT + (1.0 - cls.SMALL_OBJ_WEIGHT) * progress,
                "medium": 1.0,
                "large": cls.LARGE_OBJ_WEIGHT - (cls.LARGE_OBJ_WEIGHT - 1.0) * progress,
            }
        else:
            box_weight = cls.BOX_WEIGHT_BASE
            cls_weight = cls.CLS_WEIGHT_BASE
            phase = "balanced"
            size_weights = {
                "small": 1.0,
                "medium": 1.0,
                "large": 1.0,
            }

        return {
            "box_weight": box_weight,
            "obj_weight": cls.OBJ_WEIGHT_BASE,
            "noobj_weight": cls.NOOBJ_WEIGHT_BASE,
            "cls_weight": cls_weight,
            "size_weights": size_weights,
            "phase": phase,
        }

    @classmethod
    def get_teacher_guidance_weights(cls, detector_frozen=False):
        if detector_frozen:
            return {
                "heatmap": cls.FEATURE_HEATMAP_WEIGHT_FROZEN,
                "contrast": cls.FEATURE_CONTRAST_WEIGHT_FROZEN,
                "sparsity": cls.FEATURE_SPARSITY_WEIGHT_FROZEN,
                "tv": cls.FEATURE_TV_WEIGHT_FROZEN,
            }
        return {
            "heatmap": cls.FEATURE_HEATMAP_WEIGHT_JOINT,
            "contrast": cls.FEATURE_CONTRAST_WEIGHT_JOINT,
            "sparsity": cls.FEATURE_SPARSITY_WEIGHT_JOINT,
            "tv": cls.FEATURE_TV_WEIGHT_JOINT,
        }

    @classmethod
    def apply_runtime_overrides(cls):
        yaml_path = os.environ.get("OPTICAL_TEACHER_YAML_PATH")
        if yaml_path:
            cls.YAML_PATH = yaml_path

        output_dir = os.environ.get("OPTICAL_TEACHER_OUTPUT_DIR")
        if output_dir:
            cls.TEACHER_OUTPUT_DIR = output_dir

        teacher_init_mode = os.environ.get("OPTICAL_TEACHER_INIT_MODE")
        if teacher_init_mode:
            cls.TEACHER_INIT_MODE = teacher_init_mode

        teacher_init_checkpoint = os.environ.get("OPTICAL_TEACHER_INIT_CHECKPOINT")
        if teacher_init_checkpoint:
            cls.TEACHER_INIT_CHECKPOINT = teacher_init_checkpoint

        freeze_teacher = os.environ.get("OPTICAL_TEACHER_FREEZE_TEACHER")
        if freeze_teacher:
            cls.FREEZE_TEACHER = freeze_teacher.strip().lower() in {"1", "true", "yes", "on"}

        use_class_balanced_sampler = os.environ.get("OPTICAL_TEACHER_USE_CLASS_BALANCED_SAMPLER")
        if use_class_balanced_sampler:
            cls.USE_CLASS_BALANCED_SAMPLER = use_class_balanced_sampler.strip().lower() in {"1", "true", "yes", "on"}

        class_balance_power = os.environ.get("OPTICAL_TEACHER_CLASS_BALANCE_POWER")
        if class_balance_power:
            cls.CLASS_BALANCE_POWER = float(class_balance_power)

    @classmethod
    def initialize(cls):
        cls.apply_runtime_overrides()
        cls.CLASS_NAMES, cls.NUM_CLASSES = load_class_names(cls.YAML_PATH)
        cls.ANCHORS = [[anchor.copy() for anchor in layer] for layer in cls.DEFAULT_ANCHORS]
        cls.ANCHOR_SOURCE = "default"
        if cls.USE_EXTERNAL_ANCHORS:
            try:
                anchor_config_path = cls.ANCHOR_CONFIG_PATH
                if not os.path.isabs(anchor_config_path):
                    anchor_config_path = os.path.join(PROJECT_ROOT, anchor_config_path)
                cls.ANCHORS = load_anchor_groups(anchor_config_path)
                cls.ANCHOR_SOURCE = anchor_config_path
            except Exception as exc:
                cls.ANCHORS = [[anchor.copy() for anchor in layer] for layer in cls.DEFAULT_ANCHORS]
                cls.ANCHOR_SOURCE = f"default (external load failed: {exc})"
        os.makedirs(cls.TEACHER_OUTPUT_DIR, exist_ok=True)
        cls.LOG_ROOT_DIR = os.path.join(cls.TEACHER_OUTPUT_DIR, "logs")
        os.makedirs(cls.LOG_ROOT_DIR, exist_ok=True)
        cls.TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.LOG_FILE = os.path.join(cls.LOG_ROOT_DIR, f"training_log_{cls.TIMESTAMP}.txt")

    @classmethod
    def get_teacher_init_mode(cls):
        mode = str(cls.TEACHER_INIT_MODE).strip().lower()
        return mode if mode in {"scratch", "checkpoint"} else "scratch"

    @classmethod
    def get_teacher_init_checkpoint(cls):
        if cls.get_teacher_init_mode() != "checkpoint":
            return None
        checkpoint_path = str(cls.TEACHER_INIT_CHECKPOINT).strip()
        return checkpoint_path if checkpoint_path else None
    
    @classmethod
    def get_detector_output_channels(cls):
        return 3 * (4 + 1 + cls.NUM_CLASSES)

    @classmethod
    def should_skip_file_log(cls, message):
        return any(token in message for token in cls.SKIP_FILE_LOG_MESSAGES)

    @classmethod
    def get_epoch_table_separator(cls):
        return "-" * sum(width for _, width in cls.get_epoch_table_columns())

    @classmethod
    def get_epoch_table_columns(cls):
        return [
            ("Epoch", cls.EPOCH_TABLE_EPOCH_WIDTH),
            ("Phase", cls.EPOCH_TABLE_PHASE_WIDTH),
            ("Train Loss", cls.EPOCH_TABLE_TRAIN_LOSS_WIDTH),
            ("Val Loss", cls.EPOCH_TABLE_VAL_LOSS_WIDTH),
            ("Precision", cls.EPOCH_TABLE_METRIC_WIDTH),
            ("Recall", cls.EPOCH_TABLE_METRIC_WIDTH),
            ("F1", cls.EPOCH_TABLE_METRIC_WIDTH),
            ("mAP50", cls.EPOCH_TABLE_METRIC_WIDTH),
            ("LR", cls.EPOCH_TABLE_LR_WIDTH),
            ("Best", cls.EPOCH_TABLE_BEST_WIDTH),
        ]

    @classmethod
    def get_epoch_table_header(cls):
        return "".join(f"{title:<{width}}" for title, width in cls.get_epoch_table_columns())
    
    @classmethod
    def print_config(cls):
        print("="*80)
        print("光学教师YOLO训练配置")
        print("="*80)
        print(f"Device: {cls.DEVICE}")
        print(f"Image Size: {cls.IMG_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Num Classes: {cls.NUM_CLASSES}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Class Names: {cls.CLASS_NAMES}")
        print("="*80)


Config = ConfigYOLO

Config.initialize()
Config.print_config()

init_log_file()
log_to_file(f"Log file path: {Config.LOG_FILE}")
log_to_file(f"Visualization save path: {Config.TEACHER_OUTPUT_DIR}")
log_to_file(f"Class info: {Config.CLASS_NAMES}, Num classes: {Config.NUM_CLASSES}")

# 打印所有配置参数
def log_all_parameters():
    """Log all configuration parameters."""
    log_to_file("="*80)
    log_to_file("Optical teacher YOLO training configuration")
    log_to_file("="*80)
    
    log_to_file("\n[Dataset]")
    log_to_file(f"  YAML path: {Config.YAML_PATH}")
    log_to_file(f"  Num classes: {Config.NUM_CLASSES}")
    log_to_file(f"  Class names: {Config.CLASS_NAMES}")
    
    log_to_file("\n[Training]")
    log_to_file(f"  Device: {Config.DEVICE}")
    log_to_file(f"  Image size: {Config.IMG_SIZE}")
    log_to_file(f"  Batch size: {Config.BATCH_SIZE}")
    log_to_file(f"  Epochs: {Config.EPOCHS}")
    
    log_to_file("\n[Loss Weights]")
    log_to_file(f"  Box weight: {Config.BOX_WEIGHT_BASE}")
    log_to_file(f"  Obj weight: {Config.OBJ_WEIGHT_BASE}")
    log_to_file(f"  Noobj weight: {Config.NOOBJ_WEIGHT_BASE}")
    log_to_file(f"  Class weight: {Config.CLS_WEIGHT_BASE}")
    log_to_file(f"  Position phase epochs: {Config.POSITION_PHASE_EPOCHS}")
    log_to_file(f"  Balance phase epochs: {Config.BALANCE_PHASE_EPOCHS}")
    
    log_to_file("\n[Anchors]")
    log_to_file(f"  Strides: {Config.STRIDES}")
    log_to_file(f"  Use external anchors: {Config.USE_EXTERNAL_ANCHORS}")
    log_to_file(f"  Anchor config path: {Config.ANCHOR_CONFIG_PATH}")
    log_to_file(f"  Anchor source: {Config.ANCHOR_SOURCE}")
    log_to_file(f"  P3 anchors: {Config.ANCHORS[0]}")
    log_to_file(f"  P4 anchors: {Config.ANCHORS[1]}")
    log_to_file(f"  P5 anchors: {Config.ANCHORS[2]}")
    log_to_file(f"  IOU threshold: {Config.IOU_THRESHOLD}")
    log_to_file(f"  Positive anchor IoU: {Config.POSITIVE_ANCHOR_IOU}")
    log_to_file(f"  Max positive anchors: {Config.MAX_POSITIVE_ANCHORS}")
    log_to_file(f"  Ignore IoU for noobj: {Config.NOOBJ_IGNORE_IOU}")
    
    log_to_file("\n[Optimizer]")
    log_to_file(f"  Optimizer: {Config.OPTIMIZER}")
    log_to_file(f"  Learning rate: {Config.LEARNING_RATE}")
    log_to_file(f"  Weight decay: {Config.WEIGHT_DECAY}")
    log_to_file(f"  Teacher init mode: {Config.get_teacher_init_mode()}")
    log_to_file(f"  Teacher init checkpoint: {Config.get_teacher_init_checkpoint() or 'None'}")
    log_to_file(f"  Freeze teacher: {Config.FREEZE_TEACHER}")
    
    log_to_file("\n[Detection]")
    log_to_file(f"  Confidence threshold: {Config.CONF_THRESH}")
    log_to_file(f"  NMS threshold: {Config.NMS_THRESH}")
    log_to_file(f"  Max detections: {Config.MAX_DET}")
    log_to_file(f"  Containment suppression: {Config.ENABLE_CONTAINMENT_SUPPRESSION}")
    log_to_file(f"  Containment ratio / area limit: {Config.CONTAINMENT_SUPPRESS_RATIO} / {Config.CONTAINMENT_AREA_RATIO_LIMIT}")

    log_to_file("\n[Teacher Feature Shaping]")
    log_to_file(f"  Detector freeze epoch: {Config.DETECTOR_FREEZE_EPOCH}")
    log_to_file(f"  Detector-frozen LR: {Config.DETECTOR_FINETUNE_LR}")
    log_to_file(f"  Heatmap weight (joint/frozen): {Config.FEATURE_HEATMAP_WEIGHT_JOINT} / {Config.FEATURE_HEATMAP_WEIGHT_FROZEN}")
    log_to_file(f"  Contrast weight (joint/frozen): {Config.FEATURE_CONTRAST_WEIGHT_JOINT} / {Config.FEATURE_CONTRAST_WEIGHT_FROZEN}")
    log_to_file(f"  Sparsity weight (joint/frozen): {Config.FEATURE_SPARSITY_WEIGHT_JOINT} / {Config.FEATURE_SPARSITY_WEIGHT_FROZEN}")
    log_to_file(f"  TV weight (joint/frozen): {Config.FEATURE_TV_WEIGHT_JOINT} / {Config.FEATURE_TV_WEIGHT_FROZEN}")
    log_to_file(f"  Foreground target: {Config.FEATURE_FOREGROUND_TARGET}")
    log_to_file(f"  Background target: {Config.FEATURE_BACKGROUND_TARGET}")
    log_to_file(f"  Heatmap sigma: {Config.FEATURE_HEATMAP_SIGMA}")
    log_to_file(f"  Box / core fill value: {Config.FEATURE_BOX_FILL_VALUE} / {Config.FEATURE_CORE_FILL_VALUE}")
    log_to_file(f"  Core ratio: {Config.FEATURE_CORE_RATIO}")
    log_to_file(
        f"  Large-object area / fill boosts: {Config.FEATURE_LARGE_OBJECT_AREA} / "
        f"{Config.FEATURE_LARGE_BOX_FILL_BOOST} / {Config.FEATURE_LARGE_CORE_FILL_BOOST} / "
        f"{Config.FEATURE_LARGE_CORE_RATIO_BOOST}"
    )

    log_to_file("\n[Sampling]")
    log_to_file(f"  Use class balanced sampler: {Config.USE_CLASS_BALANCED_SAMPLER}")
    log_to_file(f"  Class balance power: {Config.CLASS_BALANCE_POWER}")
    log_to_file(f"  Max class balance gain: {Config.MAX_CLASS_BALANCE_GAIN}")
    log_to_file(f"  Majority-only image weight: {Config.MAJORITY_ONLY_IMAGE_WEIGHT}")
    log_to_file(f"  Empty-image sample weight: {Config.EMPTY_IMAGE_SAMPLE_WEIGHT}")
    log_to_file(f"  Min image sample weight: {Config.MIN_IMAGE_SAMPLE_WEIGHT}")
    
    log_to_file("\n[Visualization]")
    log_to_file(f"  Visualization interval: {Config.VIS_INTERVAL} epochs")
    log_to_file(f"  Visualization batch size: {Config.VIS_BATCH_SIZE}")
    log_to_file(f"  Visualization DPI: {Config.VIS_DPI}")
    log_to_file(f"  Visualization split: {Config.VIS_DATASET_SPLIT}")
    log_to_file(f"  Visualization seed: {Config.VIS_SEED}")
    log_to_file(f"  Visualization conf / nms / max det: {Config.VIS_CONF_THRESH} / {Config.VIS_NMS_THRESH} / {Config.VIS_MAX_DET}")
    log_to_file(f"  Visualization anchor overlays: {Config.VIS_SHOW_BEST_MATCHED_ANCHORS} (max GT: {Config.VIS_MAX_GT_ANCHOR_OVERLAYS})")
    
    log_to_file("\n[Paths]")
    log_to_file(f"  Teacher output path: {Config.TEACHER_OUTPUT_DIR}")
    log_to_file(f"  Log root path: {Config.LOG_ROOT_DIR}")
    log_to_file(f"  Log file path: {Config.LOG_FILE}")
    log_to_file(f"  Timestamp: {Config.TIMESTAMP}")
    
    # Instantiate the teacher and detector models
    teacher = ConvTeacher()
    detector = YOLOLightHead(in_channels=1, out_channels=Config.get_detector_output_channels())
    
    log_to_file("\n[Model Stats]")
    log_to_file(f"  Teacher parameters: {sum(p.numel() for p in teacher.parameters() if p.requires_grad):,}")
    log_to_file(f"  Detector parameters: {sum(p.numel() for p in detector.parameters() if p.requires_grad):,}")
    log_to_file(f"  Detector output channels: {Config.get_detector_output_channels()}")

    log_to_file(f"Focal alpha: {Config.FOCAL_ALPHA}")
    log_to_file(f"Focal gamma: {Config.FOCAL_GAMMA}")
    log_to_file(f"Agnostic NMS: {Config.AGNOSTIC_NMS}")
    log_to_file(f"Validation interval: {Config.VAL_INTERVAL}")
    
    log_to_file("\n" + "="*80)
    log_to_file("Parameter logging complete")
    log_to_file("="*80 + "\n")

# Log all parameters
log_all_parameters()

class YOLODataset(Dataset):
    def __init__(self, yaml_path=None, split="train"):
        if yaml_path is None:
            yaml_path = Config.YAML_PATH
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        img_dir = os.path.join(cfg["path"], f"{split}/images")
        label_dir = os.path.join(cfg["path"], f"{split}/labels")
        
        self.files = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith(".jpg") or f.endswith(".png")
        ])
        
        self.label_dir = label_dir
        self.img_size = Config.IMG_SIZE
        self.num_classes = Config.NUM_CLASSES
        self._sampling_metadata = None
        
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Grayscale(1),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def get_label_path(self, img_path):
        return os.path.join(self.label_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")

    def get_sampling_metadata(self):
        if self._sampling_metadata is not None:
            return self._sampling_metadata

        image_class_counters = []
        class_box_counts = Counter()
        empty_image_count = 0

        for img_path in self.files:
            label_path = self.get_label_path(img_path)
            image_class_counter = Counter()

            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        try:
                            cls_id = int(parts[0])
                        except ValueError:
                            continue
                        if 0 <= cls_id < self.num_classes:
                            image_class_counter[cls_id] += 1
                            class_box_counts[cls_id] += 1

            if len(image_class_counter) == 0:
                empty_image_count += 1
            image_class_counters.append(image_class_counter)

        self._sampling_metadata = {
            "image_class_counters": image_class_counters,
            "class_box_counts": class_box_counts,
            "empty_image_count": empty_image_count,
        }
        return self._sampling_metadata

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        
        img_tensor = self.transform(img)
        
        label_path = self.get_label_path(img_path)
        
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        targets.append([cls_id, x_center, y_center, width, height])
        
        if len(targets) > 0:
            targets = torch.tensor(targets, dtype=torch.float32)
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)
        
        return img_tensor, targets


def build_class_balanced_train_sampler(dataset):
    metadata = dataset.get_sampling_metadata()
    class_box_counts = metadata["class_box_counts"]

    if len(class_box_counts) == 0:
        return None, {"enabled": False, "reason": "no_valid_labels"}

    majority_class_id, majority_count = class_box_counts.most_common(1)[0]
    class_gains = {}
    for cls_id, cls_count in class_box_counts.items():
        raw_gain = (majority_count / max(cls_count, 1)) ** Config.CLASS_BALANCE_POWER
        class_gains[cls_id] = min(Config.MAX_CLASS_BALANCE_GAIN, max(1.0, raw_gain))

    image_weights = []
    boosted_images = 0
    majority_only_images = 0

    for image_class_counter in metadata["image_class_counters"]:
        if len(image_class_counter) == 0:
            weight = Config.EMPTY_IMAGE_SAMPLE_WEIGHT
        else:
            total_boxes = sum(image_class_counter.values())
            weighted_gain = sum(
                box_count * class_gains.get(cls_id, 1.0)
                for cls_id, box_count in image_class_counter.items()
            ) / max(total_boxes, 1)
            weight = weighted_gain

            if len(image_class_counter) == 1 and majority_class_id in image_class_counter:
                majority_only_images += 1
                weight *= Config.MAJORITY_ONLY_IMAGE_WEIGHT

            if weight > 1.05:
                boosted_images += 1

        image_weights.append(max(Config.MIN_IMAGE_SAMPLE_WEIGHT, weight))

    weights_tensor = torch.tensor(image_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights=weights_tensor, num_samples=len(dataset), replacement=True)

    summary = {
        "enabled": True,
        "majority_class_id": majority_class_id,
        "majority_class_name": Config.CLASS_NAMES.get(majority_class_id, str(majority_class_id)),
        "majority_count": int(majority_count),
        "class_box_counts": {Config.CLASS_NAMES.get(cls_id, str(cls_id)): int(count) for cls_id, count in class_box_counts.items()},
        "class_gains": {Config.CLASS_NAMES.get(cls_id, str(cls_id)): round(gain, 4) for cls_id, gain in class_gains.items()},
        "boosted_images": boosted_images,
        "majority_only_images": majority_only_images,
        "empty_images": metadata["empty_image_count"],
        "min_weight": round(float(weights_tensor.min().item()), 4),
        "max_weight": round(float(weights_tensor.max().item()), 4),
        "mean_weight": round(float(weights_tensor.mean().item()), 4),
    }
    return sampler, summary

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, strides):
        super().__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.num_classes = num_classes
        self.strides = strides
        
        # Loss weights
        self.box_weight = Config.BOX_WEIGHT_BASE
        self.obj_weight = Config.OBJ_WEIGHT_BASE
        self.noobj_weight = Config.NOOBJ_WEIGHT_BASE
        self.cls_weight = Config.CLS_WEIGHT_BASE
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    
    def set_epoch_weights(self, epoch):
        # Update dynamic loss weights according to the current training epoch.
        weights = Config.get_dynamic_weights(epoch)
        self.box_weight = weights['box_weight']
        self.obj_weight = weights['obj_weight']
        self.noobj_weight = weights['noobj_weight']
        self.cls_weight = weights['cls_weight']
        return weights['phase']

    def forward(self, predictions, targets):
        total_loss = 0
        
        for i, pred in enumerate(predictions):
            batch_size, _, grid_h, grid_w = pred.shape
            stride = self.strides[i]
            anchors = self.anchors[i] / stride
            
            pred = pred.permute(0, 2, 3, 1).reshape(batch_size, grid_h, grid_w, 3, -1)
            
            pred_boxes = pred[..., :4]
            pred_obj = pred[..., 4]
            pred_cls = pred[..., 5:]
            
            target_boxes = torch.zeros_like(pred_boxes)
            target_obj = torch.zeros_like(pred_obj)
            target_cls = torch.zeros_like(pred_cls)
            
            for b in range(batch_size):
                if len(targets[b]) == 0:
                    continue
                
                for target_idx in range(len(targets[b])):
                    cls_id, tx, ty, tw, th = targets[b][target_idx]
                    cls_id = int(cls_id.item())
                    
                    gx = int(tx * grid_w)
                    gy = int(ty * grid_h)
                    
                    gx = max(0, min(gx, grid_w - 1))
                    gy = max(0, min(gy, grid_h - 1))
                    
                    best_iou = 0
                    best_anchor = 0
                    
                    for a in range(3):
                        anchor_w, anchor_h = anchors[a]
                        iou = min(tw, anchor_w) * min(th, anchor_h) / (tw * th + anchor_w * anchor_h - min(tw, anchor_w) * min(th, anchor_h) + 1e-6)
                        if iou > best_iou:
                            best_iou = iou
                            best_anchor = a
                    
                    if best_iou > Config.IOU_THRESHOLD:
                        target_boxes[b, gy, gx, best_anchor, 0] = tx * grid_w - gx
                        target_boxes[b, gy, gx, best_anchor, 1] = ty * grid_h - gy
                        target_boxes[b, gy, gx, best_anchor, 2] = torch.log(tw / anchors[best_anchor, 0] + 1e-6)
                        target_boxes[b, gy, gx, best_anchor, 3] = torch.log(th / anchors[best_anchor, 1] + 1e-6)
                        target_obj[b, gy, gx, best_anchor] = 1.0
                        target_cls[b, gy, gx, best_anchor, cls_id] = 1.0
            
            obj_mask = target_obj > 0.5
            noobj_mask = target_obj <= 0.5
            
            # Compute loss for object boxes
            if obj_mask.sum() > 0:
                # Compute box loss
                box_loss = self.mse_loss(pred_boxes[obj_mask], target_boxes[obj_mask])
                
                # Compute object loss
                obj_loss = self.bce_loss(pred_obj[obj_mask], target_obj[obj_mask])
                
                # Compute class loss
                cls_loss = self.bce_loss(pred_cls[obj_mask], target_cls[obj_mask])
            else:
                box_loss = torch.tensor(0.0, device=pred_boxes.device)
                obj_loss = torch.tensor(0.0, device=pred_boxes.device)
                cls_loss = torch.tensor(0.0, device=pred_boxes.device)
            
            # Compute loss for no-object boxes
            if noobj_mask.sum() > 0:
                noobj_loss = self.bce_loss(pred_obj[noobj_mask], target_obj[noobj_mask])
            else:
                noobj_loss = torch.tensor(0.0, device=pred_boxes.device)
            
            # Compute total loss
            scale_loss = (self.box_weight * box_loss + 
                         self.obj_weight * obj_loss + 
                         self.noobj_weight * noobj_loss + 
                         self.cls_weight * cls_loss)
            
            # Handle NaN and infinite values
            if torch.isnan(scale_loss) or torch.isinf(scale_loss):
                scale_loss = torch.tensor(0.0, device=pred_boxes.device)
                print(f"警告: 检测到无效损失值，已重置为0")
            
            total_loss += scale_loss
        
        return total_loss

def yolo_loss_forward_fixed(self, predictions, targets):
    total_loss = 0
    batch_size = predictions[0].shape[0]
    prepared_scales = []

    for i, pred in enumerate(predictions):
        _, _, grid_h, grid_w = pred.shape
        pred = pred.permute(0, 2, 3, 1).reshape(batch_size, grid_h, grid_w, 3, -1)
        prepared_scales.append({
            "pred_boxes": pred[..., :4],
            "pred_obj": pred[..., 4],
            "pred_cls": pred[..., 5:],
            "target_boxes": torch.zeros_like(pred[..., :4]),
            "target_obj": torch.zeros_like(pred[..., 4]),
            "target_cls": torch.zeros_like(pred[..., 5:]),
            "grid_h": grid_h,
            "grid_w": grid_w,
            "anchors": self.anchors[i].to(pred.device)
        })

    for b in range(batch_size):
        if len(targets[b]) == 0:
            continue

        current_targets = targets[b].to(predictions[0].device)
        for target in current_targets:
            cls_id = int(target[0].item())
            tx = target[1]
            ty = target[2]
            tw = target[3] * Config.IMG_SIZE
            th = target[4] * Config.IMG_SIZE

            best_scale_idx = 0
            best_anchor_idx = 0
            best_iou = -1.0

            for scale_idx, scale_data in enumerate(prepared_scales):
                for anchor_idx in range(3):
                    anchor_w, anchor_h = scale_data["anchors"][anchor_idx]
                    inter = torch.minimum(tw, anchor_w) * torch.minimum(th, anchor_h)
                    union = tw * th + anchor_w * anchor_h - inter + 1e-6
                    iou = (inter / union).item()
                    if iou > best_iou:
                        best_iou = iou
                        best_scale_idx = scale_idx
                        best_anchor_idx = anchor_idx

            scale_data = prepared_scales[best_scale_idx]
            gx = tx * scale_data["grid_w"]
            gy = ty * scale_data["grid_h"]
            grid_x = max(0, min(int(gx.item()), scale_data["grid_w"] - 1))
            grid_y = max(0, min(int(gy.item()), scale_data["grid_h"] - 1))
            anchor_w, anchor_h = scale_data["anchors"][best_anchor_idx]

            scale_data["target_boxes"][b, grid_y, grid_x, best_anchor_idx, 0] = gx - grid_x
            scale_data["target_boxes"][b, grid_y, grid_x, best_anchor_idx, 1] = gy - grid_y
            scale_data["target_boxes"][b, grid_y, grid_x, best_anchor_idx, 2] = torch.log(tw / anchor_w + 1e-6)
            scale_data["target_boxes"][b, grid_y, grid_x, best_anchor_idx, 3] = torch.log(th / anchor_h + 1e-6)
            scale_data["target_obj"][b, grid_y, grid_x, best_anchor_idx] = 1.0
            scale_data["target_cls"][b, grid_y, grid_x, best_anchor_idx, cls_id] = 1.0

    for scale_data in prepared_scales:
        pred_boxes = scale_data["pred_boxes"]
        pred_obj = scale_data["pred_obj"]
        pred_cls = scale_data["pred_cls"]
        target_boxes = scale_data["target_boxes"]
        target_obj = scale_data["target_obj"]
        target_cls = scale_data["target_cls"]

        obj_mask = target_obj > 0.5
        noobj_mask = target_obj <= 0.5

        if obj_mask.sum() > 0:
            pred_xy = torch.sigmoid(pred_boxes[..., :2])
            pred_wh = pred_boxes[..., 2:4]
            target_xy = target_boxes[..., :2]
            target_wh = target_boxes[..., 2:4]

            xy_loss = self.mse_loss(pred_xy[obj_mask], target_xy[obj_mask])
            wh_loss = self.mse_loss(pred_wh[obj_mask], target_wh[obj_mask])
            box_loss = xy_loss + wh_loss
            obj_loss = self.bce_loss(pred_obj[obj_mask], target_obj[obj_mask])
            cls_loss = self.bce_loss(pred_cls[obj_mask], target_cls[obj_mask])
        else:
            box_loss = torch.tensor(0.0, device=pred_boxes.device)
            obj_loss = torch.tensor(0.0, device=pred_boxes.device)
            cls_loss = torch.tensor(0.0, device=pred_boxes.device)

        if noobj_mask.sum() > 0:
            noobj_loss = self.bce_loss(pred_obj[noobj_mask], target_obj[noobj_mask])
        else:
            noobj_loss = torch.tensor(0.0, device=pred_boxes.device)

        scale_loss = (self.box_weight * box_loss +
                     self.obj_weight * obj_loss +
                     self.noobj_weight * noobj_loss +
                     self.cls_weight * cls_loss)

        if torch.isnan(scale_loss) or torch.isinf(scale_loss):
            scale_loss = torch.tensor(0.0, device=pred_boxes.device)
            print("Warning: Infinite or NaN loss value. Setting to 0.")

        total_loss += scale_loss

    return total_loss

YOLOLoss.forward = yolo_loss_forward_fixed

class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, sample_weight=None):
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_term = (1.0 - p_t).pow(self.gamma)
        loss = alpha_t * focal_term * loss

        if sample_weight is not None:
            sample_weight = sample_weight.to(device=loss.device, dtype=loss.dtype)
            while sample_weight.dim() < loss.dim():
                sample_weight = sample_weight.unsqueeze(-1)
            loss = loss * sample_weight
            if self.reduction == "sum":
                return loss.sum()
            if self.reduction == "none":
                return loss
            normalizer = sample_weight.expand_as(loss).sum().clamp(min=1e-6)
            return loss.sum() / normalizer

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()

def decode_boxes_to_absolute(pred_boxes, anchors, stride):
    grid_h, grid_w = pred_boxes.shape[1], pred_boxes.shape[2]
    device = pred_boxes.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=pred_boxes.dtype),
        torch.arange(grid_w, device=device, dtype=pred_boxes.dtype),
        indexing='ij'
    )
    grid_x = grid_x.view(1, grid_h, grid_w, 1)
    grid_y = grid_y.view(1, grid_h, grid_w, 1)
    anchor_tensor = anchors.view(1, 1, 1, 3, 2).to(device=device, dtype=pred_boxes.dtype)

    x = (torch.sigmoid(pred_boxes[..., 0]) + grid_x) * stride
    y = (torch.sigmoid(pred_boxes[..., 1]) + grid_y) * stride
    w = torch.exp(torch.clamp(pred_boxes[..., 2], min=-8.0, max=8.0)) * anchor_tensor[..., 0]
    h = torch.exp(torch.clamp(pred_boxes[..., 3], min=-8.0, max=8.0)) * anchor_tensor[..., 1]
    return torch.stack([x, y, w, h], dim=-1)

class EnhancedYOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, strides):
        super().__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.num_classes = num_classes
        self.strides = strides
        self.box_weight = Config.BOX_WEIGHT_BASE
        self.obj_weight = Config.OBJ_WEIGHT_BASE
        self.noobj_weight = Config.NOOBJ_WEIGHT_BASE
        self.cls_weight = Config.CLS_WEIGHT_BASE
        self.focal_loss = SigmoidFocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA, reduction="mean")
        self.size_weights = {
            "small": 1.0,
            "medium": 1.0,
            "large": 1.0,
        }
        self.last_components = {
            "total": 0.0,
            "box": 0.0,
            "obj": 0.0,
            "noobj": 0.0,
            "cls": 0.0,
        }

    def set_epoch_weights(self, epoch):
        weights = Config.get_dynamic_weights(epoch)
        self.box_weight = weights["box_weight"]
        self.obj_weight = weights["obj_weight"]
        self.noobj_weight = weights["noobj_weight"]
        self.cls_weight = weights["cls_weight"]
        self.size_weights = weights["size_weights"]
        return weights["phase"]

    def _get_size_weight(self, width, height):
        area = float(width * height)
        if area >= Config.LARGE_OBJ_AREA:
            return self.size_weights["large"]
        if area >= Config.SMALL_OBJ_AREA:
            return self.size_weights["medium"]
        return self.size_weights["small"]

    def forward(self, predictions, targets):
        device = predictions[0].device
        total_loss = torch.zeros((), device=device)
        component_sums = {
            "box": torch.zeros((), device=device),
            "obj": torch.zeros((), device=device),
            "noobj": torch.zeros((), device=device),
            "cls": torch.zeros((), device=device),
        }

        batch_size = predictions[0].shape[0]
        prepared_scales = []
        for i, pred in enumerate(predictions):
            _, _, grid_h, grid_w = pred.shape
            pred = pred.permute(0, 2, 3, 1).reshape(batch_size, grid_h, grid_w, 3, -1)
            prepared_scales.append({
                "pred_boxes": pred[..., :4],
                "pred_obj": pred[..., 4],
                "pred_cls": pred[..., 5:],
                "target_boxes": torch.zeros_like(pred[..., :4]),
                "target_boxes_abs": torch.zeros_like(pred[..., :4]),
                "target_obj": torch.zeros_like(pred[..., 4]),
                "target_cls": torch.zeros_like(pred[..., 5:]),
                "target_match_iou": torch.full_like(pred[..., 4], -1.0),
                "target_scale_weight": torch.ones_like(pred[..., 4]),
                "grid_h": grid_h,
                "grid_w": grid_w,
                "stride": self.strides[i],
                "anchors": self.anchors[i].to(device)
            })

        gt_boxes_abs_by_batch = [[] for _ in range(batch_size)]
        for b in range(batch_size):
            if len(targets[b]) == 0:
                continue

            current_targets = targets[b].to(device)
            for target in current_targets:
                cls_id = int(target[0].item())
                tx = target[1]
                ty = target[2]
                tw = target[3] * Config.IMG_SIZE
                th = target[4] * Config.IMG_SIZE
                gt_boxes_abs_by_batch[b].append(torch.stack([tx * Config.IMG_SIZE, ty * Config.IMG_SIZE, tw, th]))

                size_weight = self._get_size_weight(tw.item(), th.item())
                candidate_matches = []
                for scale_idx, scale_data in enumerate(prepared_scales):
                    for anchor_idx in range(3):
                        anchor_w, anchor_h = scale_data["anchors"][anchor_idx]
                        inter = torch.minimum(tw, anchor_w) * torch.minimum(th, anchor_h)
                        union = tw * th + anchor_w * anchor_h - inter + 1e-6
                        iou = (inter / union).item()
                        candidate_matches.append((iou, scale_idx, anchor_idx))

                candidate_matches.sort(key=lambda item: item[0], reverse=True)
                selected_matches = []
                for match_iou, scale_idx, anchor_idx in candidate_matches:
                    if not selected_matches or match_iou >= Config.POSITIVE_ANCHOR_IOU:
                        selected_matches.append((match_iou, scale_idx, anchor_idx))
                    if len(selected_matches) >= Config.MAX_POSITIVE_ANCHORS:
                        break

                for match_iou, scale_idx, anchor_idx in selected_matches:
                    scale_data = prepared_scales[scale_idx]
                    gx = tx * scale_data["grid_w"]
                    gy = ty * scale_data["grid_h"]
                    grid_x = max(0, min(int(gx.item()), scale_data["grid_w"] - 1))
                    grid_y = max(0, min(int(gy.item()), scale_data["grid_h"] - 1))
                    if scale_data["target_match_iou"][b, grid_y, grid_x, anchor_idx] >= match_iou:
                        continue

                    anchor_w, anchor_h = scale_data["anchors"][anchor_idx]
                    scale_data["target_boxes"][b, grid_y, grid_x, anchor_idx, 0] = gx - grid_x
                    scale_data["target_boxes"][b, grid_y, grid_x, anchor_idx, 1] = gy - grid_y
                    scale_data["target_boxes"][b, grid_y, grid_x, anchor_idx, 2] = torch.log(tw / anchor_w + 1e-6)
                    scale_data["target_boxes"][b, grid_y, grid_x, anchor_idx, 3] = torch.log(th / anchor_h + 1e-6)
                    scale_data["target_boxes_abs"][b, grid_y, grid_x, anchor_idx, 0] = tx * Config.IMG_SIZE
                    scale_data["target_boxes_abs"][b, grid_y, grid_x, anchor_idx, 1] = ty * Config.IMG_SIZE
                    scale_data["target_boxes_abs"][b, grid_y, grid_x, anchor_idx, 2] = tw
                    scale_data["target_boxes_abs"][b, grid_y, grid_x, anchor_idx, 3] = th
                    scale_data["target_obj"][b, grid_y, grid_x, anchor_idx] = 1.0
                    scale_data["target_cls"][b, grid_y, grid_x, anchor_idx, cls_id] = 1.0
                    scale_data["target_match_iou"][b, grid_y, grid_x, anchor_idx] = match_iou
                    scale_data["target_scale_weight"][b, grid_y, grid_x, anchor_idx] = size_weight

        gt_boxes_abs_by_batch = [
            torch.stack(sample_boxes).to(device=device, dtype=predictions[0].dtype)
            if len(sample_boxes) > 0 else torch.zeros((0, 4), device=device, dtype=predictions[0].dtype)
            for sample_boxes in gt_boxes_abs_by_batch
        ]

        for scale_data in prepared_scales:
            pred_boxes = scale_data["pred_boxes"]
            pred_obj = scale_data["pred_obj"]
            pred_cls = scale_data["pred_cls"]
            target_obj = scale_data["target_obj"]
            target_cls = scale_data["target_cls"]
            target_boxes_abs = scale_data["target_boxes_abs"]
            target_scale_weight = scale_data["target_scale_weight"]
            pred_boxes_abs = decode_boxes_to_absolute(pred_boxes, scale_data["anchors"], scale_data["stride"])

            obj_mask = target_obj > 0.5
            ignore_mask = torch.zeros_like(target_obj, dtype=torch.bool)
            for b in range(batch_size):
                gt_boxes_abs = gt_boxes_abs_by_batch[b]
                if gt_boxes_abs.numel() == 0:
                    continue
                flat_pred_boxes = pred_boxes_abs[b].reshape(-1, 4)
                max_iou = bbox_iou_matrix_xywh(flat_pred_boxes, gt_boxes_abs).max(dim=1).values
                ignore_mask[b] = max_iou.view(scale_data["grid_h"], scale_data["grid_w"], 3) >= Config.NOOBJ_IGNORE_IOU

            noobj_mask = (target_obj <= 0.5) & (~ignore_mask)

            if obj_mask.any():
                positive_weights = target_scale_weight[obj_mask]
                ciou = bbox_iou_xywh(pred_boxes_abs[obj_mask], target_boxes_abs[obj_mask], ciou=True)
                box_loss = weighted_mean(1.0 - ciou, positive_weights)
                obj_loss = self.focal_loss(pred_obj[obj_mask], target_obj[obj_mask], sample_weight=positive_weights)
                cls_loss = self.focal_loss(pred_cls[obj_mask], target_cls[obj_mask], sample_weight=positive_weights)
            else:
                box_loss = torch.zeros((), device=device)
                obj_loss = torch.zeros((), device=device)
                cls_loss = torch.zeros((), device=device)

            if noobj_mask.any():
                noobj_loss = self.focal_loss(pred_obj[noobj_mask], target_obj[noobj_mask])
            else:
                noobj_loss = torch.zeros((), device=device)

            scale_loss = (
                self.box_weight * box_loss +
                self.obj_weight * obj_loss +
                self.noobj_weight * noobj_loss +
                self.cls_weight * cls_loss
            )

            if torch.isnan(scale_loss) or torch.isinf(scale_loss):
                continue

            total_loss = total_loss + scale_loss
            component_sums["box"] = component_sums["box"] + box_loss.detach()
            component_sums["obj"] = component_sums["obj"] + obj_loss.detach()
            component_sums["noobj"] = component_sums["noobj"] + noobj_loss.detach()
            component_sums["cls"] = component_sums["cls"] + cls_loss.detach()

        stats = {name: float(value.item()) for name, value in component_sums.items()}
        stats["total"] = float(total_loss.detach().item())
        self.last_components = stats
        return total_loss, stats

def prepare_batch(batch, device):
    batch_targets = []
    batch_images = []
    for img_tensor, targets in batch:
        batch_images.append(img_tensor)
        batch_targets.append(targets)
    batch_images = torch.stack(batch_images).to(device)
    return batch_images, batch_targets

def compute_average_precision(detections, total_gt):
    if total_gt == 0:
        return None
    if len(detections) == 0:
        return 0.0

    detections = sorted(detections, key=lambda item: item[0], reverse=True)
    tp = np.array([item[1] for item in detections], dtype=np.float32)
    fp = 1.0 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / (total_gt + 1e-6)
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    metric_storage = {cls_id: [] for cls_id in range(Config.NUM_CLASSES)}
    gt_counts = {cls_id: 0 for cls_id in range(Config.NUM_CLASSES)}
    component_totals = {
        "total": 0.0,
        "box": 0.0,
        "obj": 0.0,
        "noobj": 0.0,
        "cls": 0.0,
        "feature_total": 0.0,
        "feature_heatmap": 0.0,
        "feature_contrast": 0.0,
        "feature_sparsity": 0.0,
        "feature_tv": 0.0,
    }
    total_tp = 0
    total_fp = 0
    total_fn = 0
    detector_frozen = is_detector_frozen(model)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            batch_images, batch_targets = prepare_batch(batch, device)
            teacher_features, predictions = model.forward_with_feature(batch_images)
            loss, loss_stats = criterion(predictions, batch_targets)
            feature_loss, feature_stats = compute_teacher_guidance_loss(
                teacher_features,
                batch_targets,
                detector_frozen=detector_frozen,
            )
            loss = loss + feature_loss

            component_totals["total"] += float(loss.detach().item())
            component_totals["box"] += loss_stats["box"]
            component_totals["obj"] += loss_stats["obj"]
            component_totals["noobj"] += loss_stats["noobj"]
            component_totals["cls"] += loss_stats["cls"]
            component_totals["feature_total"] += feature_stats["feature_total"]
            component_totals["feature_heatmap"] += feature_stats["feature_heatmap"]
            component_totals["feature_contrast"] += feature_stats["feature_contrast"]
            component_totals["feature_sparsity"] += feature_stats["feature_sparsity"]
            component_totals["feature_tv"] += feature_stats["feature_tv"]

            detections = decode_detections(predictions)

            for sample_idx, sample_detections in enumerate(detections):
                gt_by_class = {}
                for gt in batch_targets[sample_idx]:
                    if gt.shape[0] < 5 or gt[3] <= 0 or gt[4] <= 0:
                        continue
                    cls_id = int(gt[0].item())
                    gt_box = [
                        float(gt[1].item() * Config.IMG_SIZE),
                        float(gt[2].item() * Config.IMG_SIZE),
                        float(gt[3].item() * Config.IMG_SIZE),
                        float(gt[4].item() * Config.IMG_SIZE),
                    ]
                    gt_by_class.setdefault(cls_id, []).append(gt_box)
                    gt_counts[cls_id] += 1

                matched = {cls_id: set() for cls_id in gt_by_class}
                sample_detections = sorted(sample_detections, key=lambda det: det[4], reverse=True)

                for det in sample_detections:
                    cls_id = int(det[5])
                    gt_boxes = gt_by_class.get(cls_id, [])
                    best_iou = 0.0
                    best_gt_idx = -1

                    for gt_idx, gt_box in enumerate(gt_boxes):
                        if gt_idx in matched.get(cls_id, set()):
                            continue
                        det_tensor = torch.tensor(det[:4], dtype=torch.float32).unsqueeze(0)
                        gt_tensor = torch.tensor(gt_box, dtype=torch.float32).unsqueeze(0)
                        iou = float(bbox_iou_xywh(det_tensor, gt_tensor).item())
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    is_tp = best_iou >= Config.METRIC_IOU_THRESHOLD
                    metric_storage[cls_id].append((float(det[4]), 1.0 if is_tp else 0.0))
                    if is_tp:
                        total_tp += 1
                        matched.setdefault(cls_id, set()).add(best_gt_idx)
                    else:
                        total_fp += 1

                for cls_id, gt_boxes in gt_by_class.items():
                    total_fn += len(gt_boxes) - len(matched.get(cls_id, set()))

    num_batches = max(len(dataloader), 1)
    avg_losses = {key: value / num_batches for key, value in component_totals.items()}
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1_score = 2.0 * precision * recall / (precision + recall + 1e-6)

    ap_values = []
    for cls_id in range(Config.NUM_CLASSES):
        ap = compute_average_precision(metric_storage[cls_id], gt_counts[cls_id])
        if ap is not None:
            ap_values.append(ap)
    map50 = float(np.mean(ap_values)) if len(ap_values) > 0 else 0.0

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score),
        "map50": map50,
    }
    return avg_losses, metrics

def _valid_history_points(values):
    points = [(epoch + 1, value) for epoch, value in enumerate(values) if not np.isnan(value)]
    if not points:
        return [], []
    xs, ys = zip(*points)
    return list(xs), list(ys)

def save_training_curves(history, output_dir):
    if len(history["train_total"]) == 0:
        return

    epochs = np.arange(1, len(history["train_total"]) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    axes[0].plot(epochs, history["train_total"], label="Train Loss", linewidth=2)
    val_x, val_y = _valid_history_points(history["val_total"])
    if len(val_x) > 0:
        axes[0].plot(val_x, val_y, label="Val Loss", linewidth=2, marker="o", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train / Val Loss")
    axes[0].grid(True)
    axes[0].legend()

    precision_x, precision_y = _valid_history_points(history["precision"])
    recall_x, recall_y = _valid_history_points(history["recall"])
    f1_x, f1_y = _valid_history_points(history["f1"])
    map_x, map_y = _valid_history_points(history["map50"])
    if len(precision_x) > 0:
        axes[1].plot(precision_x, precision_y, label="Precision", linewidth=2, marker="o", markersize=4)
    if len(recall_x) > 0:
        axes[1].plot(recall_x, recall_y, label="Recall", linewidth=2, marker="o", markersize=4)
    if len(f1_x) > 0:
        axes[1].plot(f1_x, f1_y, label="F1", linewidth=2, marker="o", markersize=4)
    if len(map_x) > 0:
        axes[1].plot(map_x, map_y, label="mAP@0.5", linewidth=2, marker="o", markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].set_title("Validation Metrics")
    axes[1].grid(True)
    if len(axes[1].lines) > 0:
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=Config.VIS_DPI)
    plt.close()

YOLOLoss = EnhancedYOLOLoss

def save_detection_visualization(epoch, model, dataset, save_dir, prefix="train", device=None):
    if dataset is None or len(dataset) == 0:
        return

    if device is None:
        device = Config.DEVICE

    os.makedirs(save_dir, exist_ok=True)
    was_training = model.training
    model.eval()

    num_samples = min(Config.VIS_BATCH_SIZE, len(dataset))
    generator = torch.Generator()
    generator.manual_seed(Config.VIS_SEED)
    indices = torch.randperm(len(dataset), generator=generator)[:num_samples]

    fig, axes = plt.subplots(num_samples, 4, figsize=(24, 6 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, targets = dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)

            img_np = img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

            teacher_output = model.teacher(img_tensor)
            teacher_output = teacher_output.squeeze(0).cpu().numpy()
            teacher_output = enhance_feature_for_display(teacher_output)
            if teacher_output.ndim == 3:
                teacher_output = teacher_output.squeeze(0)

            preds = list(model(img_tensor))
            detections = decode_detections(
                preds,
                conf_thresh=Config.VIS_CONF_THRESH,
                nms_thresh=Config.VIS_NMS_THRESH,
                max_det=Config.VIS_MAX_DET,
            )

            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"Original Image {i+1}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(teacher_output, cmap="hot")
            axes[i, 1].set_title(f"Teacher Feature {i+1}")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(img_np)
            axes[i, 2].set_title(f"Ground Truth {i+1}")
            axes[i, 2].axis("off")

            target_indices_for_anchor_overlay = []
            if len(targets) > 0 and Config.VIS_SHOW_BEST_MATCHED_ANCHORS:
                target_areas = []
                for target_idx in range(len(targets)):
                    width = float(targets[target_idx][3].item() * Config.IMG_SIZE)
                    height = float(targets[target_idx][4].item() * Config.IMG_SIZE)
                    target_areas.append(width * height)
                anchor_overlay_count = min(Config.VIS_MAX_GT_ANCHOR_OVERLAYS, len(targets))
                target_indices_for_anchor_overlay = sorted(
                    range(len(targets)),
                    key=lambda target_idx: target_areas[target_idx],
                    reverse=True,
                )[:anchor_overlay_count]

            for target_idx in range(len(targets)):
                cls_id, x_center, y_center, width, height = targets[target_idx]
                cls_id = int(cls_id.item())
                x_center_px = float(x_center.item() * Config.IMG_SIZE)
                y_center_px = float(y_center.item() * Config.IMG_SIZE)
                width_px = float(width.item() * Config.IMG_SIZE)
                height_px = float(height.item() * Config.IMG_SIZE)

                x1 = int(x_center_px - width_px / 2)
                y1 = int(y_center_px - height_px / 2)
                rect = patches.Rectangle(
                    (x1, y1),
                    width_px,
                    height_px,
                    linewidth=2,
                    edgecolor="green",
                    facecolor="none",
                )
                axes[i, 2].add_patch(rect)
                axes[i, 2].text(
                    x1,
                    y1 - 5,
                    Config.CLASS_NAMES[cls_id],
                    color="green",
                    fontsize=10,
                    fontweight="bold",
                )

                if target_idx in target_indices_for_anchor_overlay:
                    draw_best_matching_anchor_boxes(
                        axes[i, 2],
                        x_center=x_center_px,
                        y_center=y_center_px,
                        width=width_px,
                        height=height_px,
                    )

            axes[i, 3].imshow(img_np)
            axes[i, 3].set_title(f"Predictions {i+1}")
            axes[i, 3].axis("off")

            if len(detections[0]) > 0:
                for det in detections[0]:
                    x_center, y_center, width, height, conf, cls_id = det
                    cls_id = int(cls_id)
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)

                    color = plt.cm.tab20(cls_id / max(Config.NUM_CLASSES, 1))
                    rect = patches.Rectangle(
                        (x1, y1),
                        int(width),
                        int(height),
                        linewidth=2.1,
                        edgecolor=color,
                        facecolor="none",
                    )
                    axes[i, 3].add_patch(rect)

                    label = f"{Config.CLASS_NAMES[cls_id]}: {conf:.2f}"
                    axes[i, 3].text(
                        x1,
                        y1 - 5,
                        label,
                        color=color,
                        fontsize=9,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.35),
                    )

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_epoch_{epoch:03d}.png")
    plt.savefig(save_path, dpi=Config.VIS_DPI)
    plt.close()
    if was_training:
        model.train()

class TeacherWithDetector(nn.Module):
    def __init__(self, teacher=None, detector=None):
        super().__init__()
        if teacher is None:
            self.teacher = ConvTeacher()
        else:
            self.teacher = teacher
        
        if detector is None:
            self.detector = YOLOLightHead(in_channels=1, 
                                         out_channels=Config.get_detector_output_channels())
        else:
            self.detector = detector

    def forward(self, x):
        features = self.teacher(x)
        detections = self.detector(features)
        return detections

    def forward_with_feature(self, x):
        features = self.teacher(x)
        detections = self.detector(features)
        return features, detections

def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("teacher_state_dict", "model_state_dict", "state_dict", "model"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint

def load_teacher_checkpoint(teacher, checkpoint_path, device):
    if not checkpoint_path:
        return False, "Teacher checkpoint: not configured"
    if not os.path.exists(checkpoint_path):
        return False, f"Teacher checkpoint not found: {checkpoint_path}"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    teacher_state = teacher.state_dict()
    compatible_state = {}

    for key, value in state_dict.items():
        normalized_key = key[8:] if key.startswith("teacher.") else key
        if normalized_key in teacher_state and teacher_state[normalized_key].shape == value.shape:
            compatible_state[normalized_key] = value

    if len(compatible_state) == 0:
        return False, f"No compatible ConvTeacher weights found in: {checkpoint_path}"

    teacher.load_state_dict({**teacher_state, **compatible_state}, strict=False)
    return True, f"Loaded {len(compatible_state)} teacher tensors from: {checkpoint_path}"

def load_teacher_checkpoint_safe(teacher, checkpoint_path, device):
    if not checkpoint_path:
        return False, "Teacher checkpoint: not configured"
    if not os.path.exists(checkpoint_path):
        return False, f"Teacher checkpoint not found: {checkpoint_path}"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    teacher_state = teacher.state_dict()
    compatible_state = {}

    for key, value in state_dict.items():
        normalized_key = key[8:] if key.startswith("teacher.") else key
        if normalized_key in teacher_state and teacher_state[normalized_key].shape == value.shape:
            compatible_state[normalized_key] = value

    if len(compatible_state) == 0:
        return False, f"No compatible ConvTeacher weights found in: {checkpoint_path}"

    teacher.load_state_dict({**teacher_state, **compatible_state}, strict=False)
    return True, f"Loaded {len(compatible_state)} teacher tensors from: {checkpoint_path}"

def initialize_teacher_weights(teacher, device):
    init_mode = Config.get_teacher_init_mode()
    if init_mode == "scratch":
        return False, "Teacher init mode: scratch (training from random initialization)"

    checkpoint_path = Config.get_teacher_init_checkpoint()
    loaded_teacher, teacher_message = load_teacher_checkpoint_safe(teacher, checkpoint_path, device)
    if loaded_teacher:
        return True, f"Teacher init mode: checkpoint ({teacher_message})"

    return False, f"Teacher init mode: checkpoint requested but unavailable, fallback to scratch ({teacher_message})"


def build_optimizer_from_model(model, lr=None):
    if lr is None:
        lr = Config.LEARNING_RATE
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable_params, lr=lr, weight_decay=Config.WEIGHT_DECAY)


def set_detector_trainable(model, trainable):
    for p in model.detector.parameters():
        p.requires_grad = trainable


def is_detector_frozen(model):
    return not any(p.requires_grad for p in model.detector.parameters())

def train():
    device = Config.DEVICE
    log_to_file(f"Using device: {device}")
    
    log_to_file("Loading ConvTeacher...")
    teacher = ConvTeacher()
    loaded_teacher, teacher_message = initialize_teacher_weights(teacher, device)
    log_to_file(teacher_message)

    freeze_teacher = Config.FREEZE_TEACHER and loaded_teacher
    if Config.FREEZE_TEACHER and not loaded_teacher:
        log_to_file("Teacher requested to freeze, but no initialized teacher weights are available; keep teacher trainable.")

    for p in teacher.parameters():
        p.requires_grad = not freeze_teacher
    log_to_file(f"Teacher status: {'frozen' if freeze_teacher else 'trainable'}")
    log_to_file("Teacher parameter setup applied") 
    
    log_to_file("Loading YOLOLightHead...")
    detector = YOLOLightHead(in_channels=1, 
                           out_channels=Config.get_detector_output_channels())
    
    log_to_file("Loading model components...")
    model = TeacherWithDetector(teacher=teacher, detector=detector).to(device)
    
    for p in model.teacher.parameters():
        p.requires_grad = not freeze_teacher
    set_detector_trainable(model, True)
    
    log_to_file("Loading training dataset...")
    train_dataset = YOLODataset(split="train")
    train_sampler = None
    if Config.USE_CLASS_BALANCED_SAMPLER:
        train_sampler, sampler_summary = build_class_balanced_train_sampler(train_dataset)
        if train_sampler is not None and sampler_summary.get("enabled"):
            log_to_file("Class balanced sampler enabled")
            log_to_file(f"  Majority class: {sampler_summary['majority_class_name']} ({sampler_summary['majority_count']})")
            log_to_file(f"  Boosted images: {sampler_summary['boosted_images']}")
            log_to_file(f"  Majority-only images: {sampler_summary['majority_only_images']}")
            log_to_file(f"  Empty images: {sampler_summary['empty_images']}")
            log_to_file(f"  Image weight range: {sampler_summary['min_weight']} ~ {sampler_summary['max_weight']}")
            log_to_file(f"  Image weight mean: {sampler_summary['mean_weight']}")
            log_to_file(f"  Class gains: {sampler_summary['class_gains']}")
        else:
            log_to_file(f"Class balanced sampler disabled at runtime: {sampler_summary.get('reason', 'unknown reason')}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=lambda x: x,
    )
    log_to_file(f"Training dataset size: {len(train_dataset)}")
    
    optimizer = build_optimizer_from_model(model, lr=Config.LEARNING_RATE)
    
    criterion = YOLOLoss(anchors=Config.ANCHORS, 
                        num_classes=Config.NUM_CLASSES, 
                        strides=Config.STRIDES)
    val_loader = None
    val_dataset = None
    try:
        val_dataset = YOLODataset(split="val")
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                                   shuffle=False, collate_fn=lambda x: x)
            log_to_file(f"Validation dataset size: {len(val_dataset)}")
        else:
            log_to_file("Validation dataset is empty; validation will be skipped.")
    except Exception as exc:
        log_to_file(f"Validation dataset unavailable: {exc}")

    if Config.VIS_DATASET_SPLIT == "val" and val_dataset is not None and len(val_dataset) > 0:
        vis_dataset = val_dataset
        vis_prefix = "val"
    else:
        vis_dataset = train_dataset
        vis_prefix = "train"
    log_to_file(f"Visualization dataset: {vis_prefix}")
    
    vis_dir = os.path.join(Config.TEACHER_OUTPUT_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    teacher_best_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "teacher_best.pth")
    teacher_final_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "teacher_final.pth")
    joint_best_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "teacher_detector_best.pth")
    joint_final_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "teacher_detector_final.pth")
    
    history = {
        "train_total": [],
        "val_total": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "map50": [],
    }
    best_loss = float('inf')
    best_map50 = -1.0
    detector_stage_frozen = False
    
    log_to_file("="*60)
    log_to_file("Training model...")
    log_to_file("="*60)
    init_epoch_log_table()
    
    for epoch in range(Config.EPOCHS):
        model.train()
        train_component_sums = {
            "total": 0.0,
            "box": 0.0,
            "obj": 0.0,
            "noobj": 0.0,
            "cls": 0.0,
        }
        
        # Set phase weights
        phase = criterion.set_epoch_weights(epoch)

        if (not detector_stage_frozen) and epoch >= Config.DETECTOR_FREEZE_EPOCH:
            detector_stage_frozen = True
            set_detector_trainable(model, False)
            optimizer = build_optimizer_from_model(model, lr=Config.DETECTOR_FINETUNE_LR)
            log_to_file(
                f"Epoch {epoch}: detector frozen; continue shaping teacher features only "
                f"(lr={Config.DETECTOR_FINETUNE_LR})"
            )
        if detector_stage_frozen:
            phase = f"{phase}+teacher_shape"
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [{phase}]", leave=True):
            batch_images, batch_targets = prepare_batch(batch, device)
            
            optimizer.zero_grad()
            
            teacher_features, predictions = model.forward_with_feature(batch_images)
            loss, loss_stats = criterion(predictions, batch_targets)
            feature_loss, feature_stats = compute_teacher_guidance_loss(
                teacher_features,
                batch_targets,
                detector_frozen=detector_stage_frozen,
            )
            loss = loss + feature_loss
            
            loss.backward()
            optimizer.step()
            
            train_component_sums["total"] += float(loss.detach().item())
            train_component_sums["box"] += loss_stats["box"]
            train_component_sums["obj"] += loss_stats["obj"]
            train_component_sums["noobj"] += loss_stats["noobj"]
            train_component_sums["cls"] += loss_stats["cls"]

        avg_train = {key: value / max(len(train_loader), 1) for key, value in train_component_sums.items()}
        history["train_total"].append(avg_train["total"])
        current_lr = optimizer.param_groups[0]["lr"]
        
        val_losses = None
        val_metrics = None
        if val_loader is not None and ((epoch + 1) % Config.VAL_INTERVAL == 0):
            val_losses, val_metrics = evaluate_model(model, val_loader, criterion, device)
            history["val_total"].append(val_losses["total"])
            history["precision"].append(val_metrics["precision"])
            history["recall"].append(val_metrics["recall"])
            history["f1"].append(val_metrics["f1"])
            history["map50"].append(val_metrics["map50"])
        else:
            history["val_total"].append(np.nan)
            history["precision"].append(np.nan)
            history["recall"].append(np.nan)
            history["f1"].append(np.nan)
            history["map50"].append(np.nan)

        is_best = False
        if val_metrics is not None:
            if val_metrics["map50"] > best_map50:
                best_map50 = val_metrics["map50"]
                is_best = True
        elif avg_train["total"] < best_loss:
            is_best = True

        if is_best:
            best_loss = avg_train["total"]
            best_model_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "detector_best.pth")
            torch.save(model.detector.state_dict(), best_model_path)
            if Config.SAVE_TEACHER_WEIGHTS:
                torch.save(model.teacher.state_dict(), teacher_best_path)
                torch.save({
                    "teacher_state_dict": model.teacher.state_dict(),
                    "detector_state_dict": model.detector.state_dict(),
                    "epoch": epoch,
                    "loss": avg_train["total"],
                    "val_map50": best_map50 if val_metrics is not None else None,
                }, joint_best_path)
            if val_metrics is not None:
                print(
                    f"Epoch {epoch}: best checkpoint updated "
                    f"(train_loss={avg_train['total']:.6f}, val_map50={val_metrics['map50']:.4f})"
                )
            # log_to_file(f"Epoch {epoch}: 保存最佳模型(Loss: {best_loss:.6f})", also_print=False)
        
        if epoch % Config.VIS_INTERVAL == 0:
            save_detection_visualization(epoch, model, vis_dataset, vis_dir, prefix=vis_prefix, device=device)

        log_epoch_table_row(
            epoch=epoch,
            phase=phase,
            train_loss=avg_train["total"],
            val_loss=val_losses["total"] if val_losses is not None else None,
            precision=val_metrics["precision"] if val_metrics is not None else None,
            recall=val_metrics["recall"] if val_metrics is not None else None,
            f1_score=val_metrics["f1"] if val_metrics is not None else None,
            map50=val_metrics["map50"] if val_metrics is not None else None,
            lr=current_lr,
            best_status=Config.EPOCH_TABLE_BEST_MARK if is_best else "",
        )

        save_training_curves(history, Config.TEACHER_OUTPUT_DIR)

    model_save_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "detector_final.pth")
    torch.save(model.detector.state_dict(), model_save_path)
    if Config.SAVE_TEACHER_WEIGHTS:
        torch.save(model.teacher.state_dict(), teacher_final_path)
        torch.save({
            "teacher_state_dict": model.teacher.state_dict(),
            "detector_state_dict": model.detector.state_dict(),
            "epoch": Config.EPOCHS - 1,
            "loss": history["train_total"][-1] if len(history["train_total"]) > 0 else None,
            "val_map50": best_map50 if best_map50 >= 0 else None,
        }, joint_final_path)
        log_to_file(f"Teacher best weights saved to: {teacher_best_path}")
        log_to_file(f"Teacher final weights saved to: {teacher_final_path}")
    
    append_plain_log(Config.get_epoch_table_separator())
    log_to_file("="*60)
    log_to_file("Training complete")
    log_to_file(f"Best detector model saved to: {os.path.join(Config.TEACHER_OUTPUT_DIR, 'detector_best.pth')}")
    log_to_file(f"Final detector model saved to: {model_save_path}")
    log_to_file(f"Teacher output directory: {Config.TEACHER_OUTPUT_DIR}")
    log_to_file("="*60)

if __name__ == "__main__":
    train()
