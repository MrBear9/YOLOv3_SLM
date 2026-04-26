import yaml
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import copy
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

def resolve_project_path(path):
    if path is None:
        return None
    path = str(path).strip()
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)

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

def _format_table_cell(value, width):
    text = str(value)
    if len(text) >= width:
        text = text[: max(width - 1, 1)]
    return f"{text:<{width}}"


def init_epoch_log_table(stage_name=None):
    separator = ConfigYOLO.get_epoch_table_separator(stage_name)
    append_plain_log("")
    append_plain_log(separator)
    append_plain_log(ConfigYOLO.get_epoch_table_header(stage_name))
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

def _format_epoch_table_row(columns, values):
    return "".join(_format_table_cell(values.get(title, ""), width) for title, width in columns)


def log_epoch_table_row(
    epoch,
    phase,
    train_loss,
    val_loss,
    precision,
    recall,
    f1_score,
    map50,
    map5095,
    lr,
    best_status,
    stage_name=None,
    feature_stats=None,
):
    feature_stats = feature_stats or {}
    values = {
        "Epoch": epoch + 1,
        "Phase": phase,
        "Train": _format_table_value(train_loss, ConfigYOLO.EPOCH_TABLE_TRAIN_LOSS_WIDTH).strip(),
        "Val": _format_table_value(val_loss, ConfigYOLO.EPOCH_TABLE_VAL_LOSS_WIDTH).strip(),
        "Prec": _format_table_value(precision, ConfigYOLO.EPOCH_TABLE_METRIC_WIDTH, 3).strip(),
        "Recall": _format_table_value(recall, ConfigYOLO.EPOCH_TABLE_METRIC_WIDTH, 3).strip(),
        "F1": _format_table_value(f1_score, ConfigYOLO.EPOCH_TABLE_METRIC_WIDTH, 3).strip(),
        "mAP50": _format_table_value(map50, ConfigYOLO.EPOCH_TABLE_METRIC_WIDTH, 3).strip(),
        "mAP5095": _format_table_value(map5095, ConfigYOLO.EPOCH_TABLE_METRIC_WIDTH, 3).strip(),
        "LR": _format_table_value(lr, ConfigYOLO.EPOCH_TABLE_LR_WIDTH, 6).strip(),
        "Best": best_status,
        "Feat": f"{feature_stats.get('feature_total', 0.0):.4f}",
        "Heat": f"{feature_stats.get('feature_heatmap', 0.0):.4f}",
        "Fg": f"{feature_stats.get('feature_fg_mean', 0.0):.4f}",
        "Bg": f"{feature_stats.get('feature_bg_mean', 0.0):.4f}",
        "Edge": f"{feature_stats.get('feature_edge_mean', 0.0):.4f}",
        "Std": f"{feature_stats.get('feature_std', 0.0):.4f}",
        "DetAux": f"{feature_stats.get('detector_aux', 0.0):.4f}",
        "DetFeat": f"{feature_stats.get('detector_feat_aux', 0.0):.4f}",
        "DetObj": f"{feature_stats.get('detector_obj_aux', 0.0):.4f}",
    }
    append_plain_log(_format_epoch_table_row(ConfigYOLO.get_epoch_table_columns(stage_name), values))

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

        self.project = nn.Conv2d(32, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.project.bias)

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
        self._init_detection_biases()

    def _init_detection_biases(self):
        attrs = 5 + Config.NUM_CLASSES if Config.NUM_CLASSES is not None else None
        if attrs is None or attrs <= 5:
            return
        for head in (self.head_p3, self.head_p4, self.head_p5):
            if head.bias is None or head.bias.numel() % attrs != 0:
                continue
            with torch.no_grad():
                bias = head.bias.view(-1, attrs)
                bias[:, 4].fill_(Config.DETECTOR_INIT_OBJ_BIAS)
                bias[:, 5:].fill_(Config.DETECTOR_INIT_CLS_BIAS)

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


def minmax_normalize_map(value, eps=1e-6):
    min_value = value.amin(dim=(-2, -1), keepdim=True)
    max_value = value.amax(dim=(-2, -1), keepdim=True)
    return (value - min_value) / (max_value - min_value + eps)


def build_image_texture_map(images, img_size, device, dtype):
    if images is None:
        return None

    gray = images.to(device=device, dtype=dtype)
    if gray.shape[1] > 1:
        gray = gray.mean(dim=1, keepdim=True)
    if gray.shape[-1] != img_size or gray.shape[-2] != img_size:
        gray = F.interpolate(gray, size=(img_size, img_size), mode="bilinear", align_corners=False)

    gray = minmax_normalize_map(gray.float())
    smooth = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
    local_contrast = (gray - smooth).abs()

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    gradient = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)

    texture = (
        Config.FEATURE_TEXTURE_CONTRAST_WEIGHT * minmax_normalize_map(local_contrast) +
        Config.FEATURE_TEXTURE_EDGE_WEIGHT * minmax_normalize_map(gradient)
    )
    return minmax_normalize_map(texture).to(dtype=dtype)


def sharpen_local_texture(texture_region):
    if texture_region.numel() == 0:
        return texture_region
    local_min = texture_region.amin()
    local_max = texture_region.amax()
    local = (texture_region - local_min) / (local_max - local_min + 1e-6)
    keep_quantile = float(np.clip(Config.FEATURE_TEXTURE_KEEP_QUANTILE, 0.0, 0.95))
    threshold = torch.quantile(local.reshape(-1), keep_quantile)
    local = ((local - threshold) / (1.0 - threshold + 1e-6)).clamp(0.0, 1.0)
    return local.pow(Config.FEATURE_TEXTURE_POWER)


def get_small_object_gain(box_w, box_h):
    area = max(float(box_w * box_h), 1.0)
    gain = np.sqrt(Config.FEATURE_SMALL_OBJECT_AREA_REF / area)
    return float(np.clip(gain, 1.0, Config.FEATURE_SMALL_OBJECT_MAX_GAIN))


def build_teacher_target_map(targets, img_size, device, dtype, images=None):
    batch_size = len(targets)
    target_map = torch.zeros((batch_size, 1, img_size, img_size), device=device, dtype=dtype)
    foreground_mask = torch.zeros((batch_size, 1, img_size, img_size), device=device, dtype=torch.bool)
    sigma = max(Config.FEATURE_HEATMAP_SIGMA, 1e-3)
    heatmap_power = max(Config.FEATURE_HEATMAP_POWER, 1.0)
    box_fill_value = float(np.clip(Config.FEATURE_BOX_FILL_VALUE, 0.0, 1.0))
    core_fill_value = float(np.clip(Config.FEATURE_CORE_FILL_VALUE, box_fill_value, 1.0))
    core_ratio = float(np.clip(Config.FEATURE_CORE_RATIO, 0.1, 1.0))
    texture_map = build_image_texture_map(images, img_size, device, dtype)

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
            heat = heat.pow(heatmap_power)
            small_object_gain = get_small_object_gain(box_w, box_h)
            region = target_map[batch_idx, 0, y1:y2, x1:x2]
            base_fill = torch.full_like(region, box_fill_value)
            gaussian_target = (Config.FEATURE_GAUSSIAN_TARGET_GAIN * small_object_gain * heat).clamp(0.0, 1.0)
            region_target = torch.maximum(torch.maximum(region, base_fill), gaussian_target)

            if texture_map is not None and Config.FEATURE_TEXTURE_TARGET_GAIN > 0:
                texture_region = sharpen_local_texture(texture_map[batch_idx, 0, y1:y2, x1:x2])
                center_gate = (
                    Config.FEATURE_TEXTURE_CENTER_BIAS +
                    (1.0 - Config.FEATURE_TEXTURE_CENTER_BIAS) * heat.sqrt()
                ).clamp(0.0, 1.0)
                texture_target = Config.FEATURE_TEXTURE_TARGET_BASE + (
                    Config.FEATURE_TEXTURE_TARGET_GAIN * small_object_gain * texture_region
                )
                texture_target = (
                    texture_target * center_gate +
                    0.35 * gaussian_target +
                    0.45 * base_fill
                ).clamp(0.0, 1.0)
                region_target = torch.maximum(region_target, texture_target)

            core_w = max(1, int(round(box_w * core_ratio)))
            core_h = max(1, int(round(box_h * core_ratio)))
            core_x1 = x1 + max(0, (box_w - core_w) // 2)
            core_y1 = y1 + max(0, (box_h - core_h) // 2)
            core_x2 = min(x2, core_x1 + core_w)
            core_y2 = min(y2, core_y1 + core_h)
            if core_x2 > core_x1 and core_y2 > core_y1:
                core_region = region_target[(core_y1 - y1):(core_y2 - y1), (core_x1 - x1):(core_x2 - x1)]
                core_fill = torch.full_like(core_region, core_fill_value)
                region_target[(core_y1 - y1):(core_y2 - y1), (core_x1 - x1):(core_x2 - x1)] = torch.maximum(core_region, core_fill)

            target_map[batch_idx, 0, y1:y2, x1:x2] = region_target
            foreground_mask[batch_idx, 0, y1:y2, x1:x2] = True

    return target_map, foreground_mask


def build_feature_border_mask(batch_size, img_size, device, margin_ratio):
    margin = max(1, int(round(img_size * margin_ratio)))
    border_mask = torch.zeros((batch_size, 1, img_size, img_size), device=device, dtype=torch.bool)
    border_mask[:, :, :margin, :] = True
    border_mask[:, :, -margin:, :] = True
    border_mask[:, :, :, :margin] = True
    border_mask[:, :, :, -margin:] = True
    return border_mask


def compute_teacher_guidance_loss(teacher_feature, targets, stage="joint", images=None):
    target_map, foreground_mask = build_teacher_target_map(
        targets=targets,
        img_size=teacher_feature.shape[-1],
        device=teacher_feature.device,
        dtype=teacher_feature.dtype,
        images=images,
    )
    background_mask = ~foreground_mask
    active_foreground_mask = foreground_mask & (target_map >= Config.FEATURE_ACTIVE_TARGET_THRESH)
    if not active_foreground_mask.any():
        active_foreground_mask = foreground_mask
    border_mask = build_feature_border_mask(
        batch_size=teacher_feature.shape[0],
        img_size=teacher_feature.shape[-1],
        device=teacher_feature.device,
        margin_ratio=Config.FEATURE_EDGE_MARGIN_RATIO,
    )
    edge_background_mask = background_mask & border_mask
    inner_background_mask = background_mask & ~border_mask

    autocast_device = teacher_feature.device.type if teacher_feature.device.type in {"cuda", "cpu"} else "cpu"
    with torch.autocast(device_type=autocast_device, enabled=False):
        teacher_feature_fp32 = teacher_feature.float().clamp(1e-4, 1.0 - 1e-4)
        target_map_fp32 = target_map.float()

        def masked_bce(mask):
            if mask.any():
                return F.binary_cross_entropy(
                    teacher_feature_fp32[mask],
                    target_map_fp32[mask],
                )
            return torch.zeros((), device=teacher_feature.device, dtype=torch.float32)

        fg_pixel_count = foreground_mask.float().sum()
        bg_pixel_count = background_mask.float().sum()
        fg_weight = torch.sqrt(bg_pixel_count / fg_pixel_count.clamp(min=1.0))
        fg_weight = fg_weight.clamp(min=1.0, max=Config.FEATURE_FOREGROUND_BCE_MAX_GAIN)

        fg_heatmap_loss = masked_bce(foreground_mask)
        bg_heatmap_loss = masked_bce(inner_background_mask)
        edge_heatmap_loss = masked_bce(edge_background_mask)
        heatmap_loss = (
            fg_weight * fg_heatmap_loss +
            bg_heatmap_loss +
            Config.FEATURE_EDGE_BCE_GAIN * edge_heatmap_loss
        )

        if active_foreground_mask.any():
            fg_mean = teacher_feature_fp32[active_foreground_mask].mean()
        else:
            fg_mean = torch.zeros((), device=teacher_feature.device, dtype=torch.float32)

        if background_mask.any():
            bg_mean = teacher_feature_fp32[background_mask].mean()
        else:
            bg_mean = torch.zeros((), device=teacher_feature.device, dtype=torch.float32)

        if edge_background_mask.any():
            edge_bg_mean = teacher_feature_fp32[edge_background_mask].mean()
        else:
            edge_bg_mean = torch.zeros((), device=teacher_feature.device, dtype=torch.float32)

        contrast_loss = (
            F.relu(torch.tensor(Config.FEATURE_FOREGROUND_TARGET, device=teacher_feature.device, dtype=torch.float32) - fg_mean) +
            F.relu(bg_mean - torch.tensor(Config.FEATURE_BACKGROUND_TARGET, device=teacher_feature.device, dtype=torch.float32))
        )
        sparsity_loss = bg_mean
        tv_loss = (
            torch.abs(teacher_feature_fp32[:, :, 1:, :] - teacher_feature_fp32[:, :, :-1, :]).mean() +
            torch.abs(teacher_feature_fp32[:, :, :, 1:] - teacher_feature_fp32[:, :, :, :-1]).mean()
        )

    weights = Config.get_teacher_guidance_weights(stage=stage)
    total = (
        weights["heatmap"] * heatmap_loss +
        weights["contrast"] * contrast_loss +
        weights["sparsity"] * sparsity_loss +
        weights["tv"] * tv_loss
    )
    stats = {
        "feature_heatmap": float(heatmap_loss.detach().item()),
        "feature_fg_bce": float(fg_heatmap_loss.detach().item()),
        "feature_bg_bce": float(bg_heatmap_loss.detach().item()),
        "feature_edge_bce": float(edge_heatmap_loss.detach().item()),
        "feature_contrast": float(contrast_loss.detach().item()),
        "feature_sparsity": float(sparsity_loss.detach().item()),
        "feature_tv": float(tv_loss.detach().item()),
        "feature_fg_mean": float(fg_mean.detach().item()),
        "feature_bg_mean": float(bg_mean.detach().item()),
        "feature_edge_mean": float(edge_bg_mean.detach().item()),
        "feature_mean": float(teacher_feature_fp32.mean().detach().item()),
        "feature_std": float(teacher_feature_fp32.std(unbiased=False).detach().item()),
        "feature_fg_weight": float(fg_weight.detach().item()),
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
        conf_thresh = Config.EVAL_CONF_THRESH
    if nms_thresh is None:
        nms_thresh = Config.EVAL_NMS_THRESH
    if max_det is None:
        max_det = Config.EVAL_MAX_DET
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
        conf_thresh = Config.EVAL_CONF_THRESH
    if nms_thresh is None:
        nms_thresh = Config.EVAL_NMS_THRESH
    if max_det is None:
        max_det = Config.EVAL_MAX_DET
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
    TEACHER_OUTPUT_DIR = r"output\OpticalTeacherYOLO_enhance_teacher"
    LOG_ROOT_DIR = None
    LOG_FILE = None
    TIMESTAMP = None
    TRAIN_START_TIME = None

    # Runtime
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 640
    BATCH_SIZE = 8
    EPOCHS = 120*2
    TEACHER_STAGE_EPOCHS = 20*2  # 教师阶段训练轮数
    DETECTOR_STAGE_EPOCHS = 70*2  # 检测器训练轮数
    JOINT_FINETUNE_EPOCHS = 30*2  # 联合微调训练轮数

    # 经常调整的训练参数
    BOX_WEIGHT_BASE = 1.32  # 保留大目标回归力度，但略收一点，减少框体过大
    OBJ_WEIGHT_BASE = 0.56  # 从上一版回调，给真实目标更多正样本驱动
    NOOBJ_WEIGHT_BASE = 0.40  # 提高背景抑制，避免 detector 长期“多报框”
    CLS_WEIGHT_BASE = 0.40  # 提高分类约束，减少同一目标多类别重复框

    POSITION_PHASE_EPOCHS = 12  # 继续让前期定位学稳
    BALANCE_PHASE_EPOCHS = 8  # 缩短过渡段，减少中期被塑形权重拖慢

    IOU_THRESHOLD = 0.5  # 主要正样本匹配阈值；较大的值使正样本分配更严格，可能降低召回率
    POSITIVE_ANCHOR_IOU = 0.30  # 进一步过滤掉形状偏差太大的锚框
    MAX_POSITIVE_ANCHORS = 2  # 最多保留两个代表性尺度，避免大目标同时激活过多小尺度框
    NOOBJ_IGNORE_IOU = 0.72  # 让更多近邻伪框接受背景抑制
    ANCHOR_MATCH_RATIO_THRESH = 4.0
    ASSIGN_NEIGHBOR_CELLS = True
    NEIGHBOR_ASSIGN_MARGIN = 0.25
    POSITIVE_SCALE_IOU_RATIO = 0.70

    SMALL_OBJ_AREA = 32 * 32  # 将对象分组为小目标的阈值
    LARGE_OBJ_AREA = 128 * 128  # 将对象分组为大目标的阈值
    SMALL_OBJ_WEIGHT = 0.8  # 给小目标一点权重恢复，减少小目标漏检
    MEDIUM_OBJ_WEIGHT = 1.0  # 中等目标的基准损失权重
    LARGE_OBJ_WEIGHT = 1.5  # 保留大目标强调，但降低一个目标多框的倾向

    FOCAL_ALPHA = 0.28  # 向旧稳态回调，避免过度偏向负样本
    FOCAL_GAMMA = 1.6  # 略降，减少后期被困难背景样本牵制

    EVAL_CONF_THRESH = 0.001
    VIS_CONF_THRESH = 0.25
    DEPLOY_CONF_THRESH = 0.58
    EVAL_NMS_THRESH = 0.45
    VIS_NMS_THRESH = 0.40
    DEPLOY_NMS_THRESH = 0.22
    EVAL_MAX_DET = 300
    VIS_MAX_DET = 100
    MAX_DET = 100
    AGNOSTIC_NMS = False  # 多类别任务使用按类 NMS，避免不同类别互相压制

    # 验证和指标
    VAL_INTERVAL = 5
    METRIC_IOU_THRESHOLD = 0.5  # 较大的值在评估期间使TP匹配更严格，可能降低召回率，但可能增加假阳性
    MAP_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # Teacher feature shaping and staged training
    TEACHER_STAGE_LR = 1.5e-4
    DETECTOR_STAGE_LR = 3e-4
    JOINT_TEACHER_LR = 3e-5
    JOINT_DETECTOR_LR = 1e-4
    WARMUP_EPOCHS = 3
    WARMUP_START_FACTOR = 0.1
    USE_AMP = True
    USE_EMA = True
    EMA_DECAY = 0.9998
    GRAD_CLIP_NORM = 1.0
    MIXUP_PROB = 0.15
    MIXUP_ALPHA = 8.0
    FEATURE_HEATMAP_WEIGHT_TEACHER = 0.08
    FEATURE_CONTRAST_WEIGHT_TEACHER = 0.015
    FEATURE_SPARSITY_WEIGHT_TEACHER = 0.004
    FEATURE_TV_WEIGHT_TEACHER = 0.001
    FEATURE_HEATMAP_WEIGHT_JOINT = 0.03
    FEATURE_CONTRAST_WEIGHT_JOINT = 0.008
    FEATURE_SPARSITY_WEIGHT_JOINT = 0.002
    FEATURE_TV_WEIGHT_JOINT = 0.0005
    FEATURE_FOREGROUND_TARGET = 0.70
    FEATURE_BACKGROUND_TARGET = 0.04
    FEATURE_HEATMAP_SIGMA = 0.24
    FEATURE_HEATMAP_POWER = 1.10
    FEATURE_BOX_FILL_VALUE = 0.04
    FEATURE_CORE_FILL_VALUE = 0.12
    FEATURE_CORE_RATIO = 0.38
    FEATURE_FOREGROUND_BCE_MAX_GAIN = 6.0
    FEATURE_EDGE_BCE_GAIN = 2.5
    FEATURE_EDGE_MARGIN_RATIO = 0.06
    FEATURE_GAUSSIAN_TARGET_GAIN = 0.22
    FEATURE_TEXTURE_TARGET_BASE = 0.03
    FEATURE_TEXTURE_TARGET_GAIN = 0.74
    FEATURE_TEXTURE_CONTRAST_WEIGHT = 0.55
    FEATURE_TEXTURE_EDGE_WEIGHT = 0.45
    FEATURE_TEXTURE_KEEP_QUANTILE = 0.52
    FEATURE_TEXTURE_POWER = 1.55
    FEATURE_TEXTURE_CENTER_BIAS = 0.62
    FEATURE_SMALL_OBJECT_AREA_REF = 32 * 32
    FEATURE_SMALL_OBJECT_MAX_GAIN = 2.4
    FEATURE_ACTIVE_TARGET_THRESH = 0.12
    DETECTOR_HEAT_AUX_WEIGHT = 0.08
    DETECTOR_FEATURE_AUX_WEIGHT = 0.04
    DETECTOR_OBJ_HEAT_AUX_WEIGHT = 0.01
    DETECTOR_AUX_TARGET_POWER = 1.8
    DETECTOR_AUX_TARGET_THRESH = 0.12
    DETECTOR_INIT_OBJ_BIAS = -4.5
    DETECTOR_INIT_CLS_BIAS = -1.5

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
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 3e-5
    OPTIMIZER = "Adam"
    TEACHER_INIT_MODE = "scratch"  # "scratch" 或 "checkpoint"
    TEACHER_INIT_CHECKPOINT = r"output\oty_m1\teacher_final.pth"
    DETECTOR_INIT_MODE = "scratch"
    DETECTOR_INIT_CHECKPOINT = r"output\oty_m1\detector_final.pth"
    FREEZE_TEACHER = False
    SAVE_TEACHER_WEIGHTS = True
    SKIP_TEACHER_STAGE_IF_INIT = False # 初始化时跳过教师阶段，直接开始检测训练
    TEACHER_CONSISTENCY_WEIGHT = 0.05

    # Visualization
    VIS_INTERVAL = 1*2
    VIS_BATCH_SIZE = 4
    VIS_DPI = 130
    VIS_DATASET_SPLIT = "val"  # 可视化时使用的数据集分割
    VIS_SEED = 20260425  # 保持与本次 ft 实验一致，便于前后对比
    VIS_DETECTOR_CONF_THRESH = 0.30
    VIS_EDGE_CONF_THRESH = 0.45
    VIS_EDGE_MARGIN_RATIO = 0.04
    VIS_MATCHED_ONLY = True
    VIS_MATCHED_IOU_THRESH = 0.30
    VIS_DRAW_GT_ANCHORS = True

    # Logging table
    EPOCH_TABLE_EPOCH_WIDTH = 8
    EPOCH_TABLE_PHASE_WIDTH = 18
    EPOCH_TABLE_TRAIN_LOSS_WIDTH = 13
    EPOCH_TABLE_VAL_LOSS_WIDTH = 13
    EPOCH_TABLE_METRIC_WIDTH = 11
    EPOCH_TABLE_LR_WIDTH = 12
    EPOCH_TABLE_BEST_WIDTH = 8
    EPOCH_TABLE_STAT_WIDTH = 10
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
    def get_teacher_guidance_weights(cls, stage="joint"):
        if stage == "teacher":
            return {
                "heatmap": cls.FEATURE_HEATMAP_WEIGHT_TEACHER,
                "contrast": cls.FEATURE_CONTRAST_WEIGHT_TEACHER,
                "sparsity": cls.FEATURE_SPARSITY_WEIGHT_TEACHER,
                "tv": cls.FEATURE_TV_WEIGHT_TEACHER,
            }
        if stage == "joint":
            return {
                "heatmap": cls.FEATURE_HEATMAP_WEIGHT_JOINT,
                "contrast": cls.FEATURE_CONTRAST_WEIGHT_JOINT,
                "sparsity": cls.FEATURE_SPARSITY_WEIGHT_JOINT,
                "tv": cls.FEATURE_TV_WEIGHT_JOINT,
            }
        return {
            "heatmap": 0.0,
            "contrast": 0.0,
            "sparsity": 0.0,
            "tv": 0.0,
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

        detector_init_mode = os.environ.get("OPTICAL_DETECTOR_INIT_MODE")
        if detector_init_mode:
            cls.DETECTOR_INIT_MODE = detector_init_mode

        detector_init_checkpoint = os.environ.get("OPTICAL_DETECTOR_INIT_CHECKPOINT")
        if detector_init_checkpoint:
            cls.DETECTOR_INIT_CHECKPOINT = detector_init_checkpoint

        anchor_config_path = os.environ.get("OPTICAL_TEACHER_ANCHOR_CONFIG_PATH")
        if anchor_config_path:
            cls.ANCHOR_CONFIG_PATH = anchor_config_path

        use_external_anchors = os.environ.get("OPTICAL_TEACHER_USE_EXTERNAL_ANCHORS")
        if use_external_anchors:
            cls.USE_EXTERNAL_ANCHORS = use_external_anchors.strip().lower() in {"1", "true", "yes", "on"}

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
        cls.YAML_PATH = resolve_project_path(cls.YAML_PATH)
        cls.TEACHER_OUTPUT_DIR = resolve_project_path(cls.TEACHER_OUTPUT_DIR)
        cls.TEACHER_INIT_CHECKPOINT = resolve_project_path(cls.TEACHER_INIT_CHECKPOINT)
        cls.DETECTOR_INIT_CHECKPOINT = resolve_project_path(cls.DETECTOR_INIT_CHECKPOINT)
        cls.ANCHOR_CONFIG_PATH = resolve_project_path(cls.ANCHOR_CONFIG_PATH)
        cls.CLASS_NAMES, cls.NUM_CLASSES = load_class_names(cls.YAML_PATH)
        cls.ANCHORS = [[anchor.copy() for anchor in layer] for layer in cls.DEFAULT_ANCHORS]
        cls.ANCHOR_SOURCE = "default"
        if cls.USE_EXTERNAL_ANCHORS:
            try:
                anchor_config_path = cls.ANCHOR_CONFIG_PATH
                cls.ANCHORS = load_anchor_groups(anchor_config_path)
                cls.ANCHOR_SOURCE = anchor_config_path
            except Exception as exc:
                cls.ANCHORS = [[anchor.copy() for anchor in layer] for layer in cls.DEFAULT_ANCHORS]
                cls.ANCHOR_SOURCE = f"default (external load failed: {exc})"
        cls.EPOCHS = cls.get_total_epochs()
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
        checkpoint_path = str(resolve_project_path(cls.TEACHER_INIT_CHECKPOINT)).strip()
        return checkpoint_path if checkpoint_path else None

    @classmethod
    def get_detector_init_mode(cls):
        mode = str(cls.DETECTOR_INIT_MODE).strip().lower()
        return mode if mode in {"scratch", "checkpoint"} else "scratch"

    @classmethod
    def get_detector_init_checkpoint(cls):
        if cls.get_detector_init_mode() != "checkpoint":
            return None
        checkpoint_path = str(resolve_project_path(cls.DETECTOR_INIT_CHECKPOINT)).strip()
        return checkpoint_path if checkpoint_path else None

    @classmethod
    def get_total_epochs(cls):
        return max(0, cls.TEACHER_STAGE_EPOCHS) + max(0, cls.DETECTOR_STAGE_EPOCHS) + max(0, cls.JOINT_FINETUNE_EPOCHS)

    @classmethod
    def get_training_stage(cls, epoch):
        if epoch < cls.TEACHER_STAGE_EPOCHS:
            return "teacher"
        if epoch < cls.TEACHER_STAGE_EPOCHS + cls.DETECTOR_STAGE_EPOCHS:
            return "detector"
        return "joint"
    
    @classmethod
    def get_detector_output_channels(cls):
        return 3 * (4 + 1 + cls.NUM_CLASSES)

    @classmethod
    def should_skip_file_log(cls, message):
        return any(token in message for token in cls.SKIP_FILE_LOG_MESSAGES)

    @classmethod
    def get_epoch_table_separator(cls, stage_name=None):
        return "-" * sum(width for _, width in cls.get_epoch_table_columns(stage_name))

    @classmethod
    def get_epoch_table_columns(cls, stage_name=None):
        base = [
            ("Epoch", cls.EPOCH_TABLE_EPOCH_WIDTH),
            ("Phase", cls.EPOCH_TABLE_PHASE_WIDTH),
            ("Train", cls.EPOCH_TABLE_TRAIN_LOSS_WIDTH),
            ("Val", cls.EPOCH_TABLE_VAL_LOSS_WIDTH),
        ]
        if stage_name == "teacher":
            return base + [
                ("Feat", cls.EPOCH_TABLE_STAT_WIDTH),
                ("Heat", cls.EPOCH_TABLE_STAT_WIDTH),
                ("Fg", cls.EPOCH_TABLE_STAT_WIDTH),
                ("Bg", cls.EPOCH_TABLE_STAT_WIDTH),
                ("Edge", cls.EPOCH_TABLE_STAT_WIDTH),
                ("Std", cls.EPOCH_TABLE_STAT_WIDTH),
                ("LR", cls.EPOCH_TABLE_LR_WIDTH),
            ]
        return base + [
            ("Prec", cls.EPOCH_TABLE_METRIC_WIDTH),
            ("Recall", cls.EPOCH_TABLE_METRIC_WIDTH),
            ("F1", cls.EPOCH_TABLE_METRIC_WIDTH),
            ("mAP50", cls.EPOCH_TABLE_METRIC_WIDTH),
            ("mAP5095", cls.EPOCH_TABLE_METRIC_WIDTH),
            ("DetAux", cls.EPOCH_TABLE_STAT_WIDTH),
            ("DetFeat", cls.EPOCH_TABLE_STAT_WIDTH),
            ("DetObj", cls.EPOCH_TABLE_STAT_WIDTH),
            ("LR", cls.EPOCH_TABLE_LR_WIDTH),
            ("Best", cls.EPOCH_TABLE_BEST_WIDTH),
        ]

    @classmethod
    def get_epoch_table_header(cls, stage_name=None):
        return "".join(f"{title:<{width}}" for title, width in cls.get_epoch_table_columns(stage_name))
    
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
    log_to_file(f"  Stage epochs (teacher/detector/joint): {Config.TEACHER_STAGE_EPOCHS} / {Config.DETECTOR_STAGE_EPOCHS} / {Config.JOINT_FINETUNE_EPOCHS}")
    
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
    log_to_file(f"  Anchor match ratio threshold: {Config.ANCHOR_MATCH_RATIO_THRESH}")
    log_to_file(f"  Assign neighbor cells: {Config.ASSIGN_NEIGHBOR_CELLS}")
    log_to_file(f"  Neighbor assign margin: {Config.NEIGHBOR_ASSIGN_MARGIN}")
    log_to_file(f"  Positive scale IoU ratio: {Config.POSITIVE_SCALE_IOU_RATIO}")
    
    log_to_file("\n[Optimizer]")
    log_to_file(f"  Optimizer: {Config.OPTIMIZER}")
    log_to_file(f"  Learning rate: {Config.LEARNING_RATE}")
    log_to_file(f"  Weight decay: {Config.WEIGHT_DECAY}")
    log_to_file(f"  Stage LR (teacher/detector/joint_t/joint_d): {Config.TEACHER_STAGE_LR} / {Config.DETECTOR_STAGE_LR} / {Config.JOINT_TEACHER_LR} / {Config.JOINT_DETECTOR_LR}")
    log_to_file(f"  Teacher init mode: {Config.get_teacher_init_mode()}")
    log_to_file(f"  Teacher init checkpoint: {Config.get_teacher_init_checkpoint() or 'None'}")
    log_to_file(f"  Detector init mode: {Config.get_detector_init_mode()}")
    log_to_file(f"  Detector init checkpoint: {Config.get_detector_init_checkpoint() or 'None'}")
    log_to_file(f"  Freeze teacher: {Config.FREEZE_TEACHER}")
    log_to_file(f"  Skip teacher stage if init teacher exists: {Config.SKIP_TEACHER_STAGE_IF_INIT}")
    log_to_file(f"  Teacher consistency weight: {Config.TEACHER_CONSISTENCY_WEIGHT}")
    
    log_to_file("\n[Detection]")
    log_to_file(f"  Eval conf / vis conf / deploy conf: {Config.EVAL_CONF_THRESH} / {Config.VIS_CONF_THRESH} / {Config.DEPLOY_CONF_THRESH}")
    log_to_file(f"  Eval nms / vis nms / deploy nms: {Config.EVAL_NMS_THRESH} / {Config.VIS_NMS_THRESH} / {Config.DEPLOY_NMS_THRESH}")
    log_to_file(f"  Eval max det / vis max det: {Config.EVAL_MAX_DET} / {Config.VIS_MAX_DET}")

    log_to_file("\n[Teacher Feature Shaping]")
    log_to_file(f"  Warmup epochs / start factor: {Config.WARMUP_EPOCHS} / {Config.WARMUP_START_FACTOR}")
    log_to_file(f"  AMP / EMA / grad clip: {Config.USE_AMP} / {Config.USE_EMA} / {Config.GRAD_CLIP_NORM}")
    log_to_file(f"  Mixup prob / alpha: {Config.MIXUP_PROB} / {Config.MIXUP_ALPHA}")
    log_to_file(f"  Heatmap weight (teacher/joint): {Config.FEATURE_HEATMAP_WEIGHT_TEACHER} / {Config.FEATURE_HEATMAP_WEIGHT_JOINT}")
    log_to_file(f"  Contrast weight (teacher/joint): {Config.FEATURE_CONTRAST_WEIGHT_TEACHER} / {Config.FEATURE_CONTRAST_WEIGHT_JOINT}")
    log_to_file(f"  Sparsity weight (teacher/joint): {Config.FEATURE_SPARSITY_WEIGHT_TEACHER} / {Config.FEATURE_SPARSITY_WEIGHT_JOINT}")
    log_to_file(f"  TV weight (teacher/joint): {Config.FEATURE_TV_WEIGHT_TEACHER} / {Config.FEATURE_TV_WEIGHT_JOINT}")
    log_to_file(f"  Foreground target: {Config.FEATURE_FOREGROUND_TARGET}")
    log_to_file(f"  Background target: {Config.FEATURE_BACKGROUND_TARGET}")
    log_to_file(f"  Heatmap sigma: {Config.FEATURE_HEATMAP_SIGMA}")
    log_to_file(f"  Heatmap power: {Config.FEATURE_HEATMAP_POWER}")
    log_to_file(f"  Box / core fill value: {Config.FEATURE_BOX_FILL_VALUE} / {Config.FEATURE_CORE_FILL_VALUE}")
    log_to_file(f"  Core ratio: {Config.FEATURE_CORE_RATIO}")
    log_to_file(f"  Foreground BCE max gain: {Config.FEATURE_FOREGROUND_BCE_MAX_GAIN}")
    log_to_file(f"  Edge BCE gain / margin ratio: {Config.FEATURE_EDGE_BCE_GAIN} / {Config.FEATURE_EDGE_MARGIN_RATIO}")
    log_to_file(f"  Gaussian target gain: {Config.FEATURE_GAUSSIAN_TARGET_GAIN}")
    log_to_file(
        f"  Texture target base/gain: {Config.FEATURE_TEXTURE_TARGET_BASE} / "
        f"{Config.FEATURE_TEXTURE_TARGET_GAIN}"
    )
    log_to_file(
        f"  Texture contrast/edge weight: {Config.FEATURE_TEXTURE_CONTRAST_WEIGHT} / "
        f"{Config.FEATURE_TEXTURE_EDGE_WEIGHT}"
    )
    log_to_file(
        f"  Texture keep quantile / power: {Config.FEATURE_TEXTURE_KEEP_QUANTILE} / "
        f"{Config.FEATURE_TEXTURE_POWER}"
    )
    log_to_file(f"  Texture center bias: {Config.FEATURE_TEXTURE_CENTER_BIAS}")
    log_to_file(
        f"  Small object area ref / max gain: {Config.FEATURE_SMALL_OBJECT_AREA_REF} / "
        f"{Config.FEATURE_SMALL_OBJECT_MAX_GAIN}"
    )
    log_to_file(f"  Active foreground target threshold: {Config.FEATURE_ACTIVE_TARGET_THRESH}")
    log_to_file(
        f"  Detector heat aux weights (heat/feat/obj): {Config.DETECTOR_HEAT_AUX_WEIGHT} / "
        f"{Config.DETECTOR_FEATURE_AUX_WEIGHT} / {Config.DETECTOR_OBJ_HEAT_AUX_WEIGHT}"
    )
    log_to_file(
        f"  Detector aux target power / threshold: {Config.DETECTOR_AUX_TARGET_POWER} / "
        f"{Config.DETECTOR_AUX_TARGET_THRESH}"
    )
    log_to_file(f"  Detector init obj/cls bias: {Config.DETECTOR_INIT_OBJ_BIAS} / {Config.DETECTOR_INIT_CLS_BIAS}")

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
    log_to_file(f"  Detector visualization conf / edge conf: {Config.VIS_DETECTOR_CONF_THRESH} / {Config.VIS_EDGE_CONF_THRESH}")
    log_to_file(f"  Detector visualization matched only / IoU: {Config.VIS_MATCHED_ONLY} / {Config.VIS_MATCHED_IOU_THRESH}")
    log_to_file(f"  Draw GT best anchors in visualization: {Config.VIS_DRAW_GT_ANCHORS}")
    
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
                scale_best_matches = {}
                for scale_idx, scale_data in enumerate(prepared_scales):
                    for anchor_idx in range(3):
                        anchor_w, anchor_h = scale_data["anchors"][anchor_idx]
                        inter = torch.minimum(tw, anchor_w) * torch.minimum(th, anchor_h)
                        union = tw * th + anchor_w * anchor_h - inter + 1e-6
                        iou = (inter / union).item()
                        rw = max(float(tw.item() / (anchor_w.item() + 1e-6)), float(anchor_w.item() / (tw.item() + 1e-6)))
                        rh = max(float(th.item() / (anchor_h.item() + 1e-6)), float(anchor_h.item() / (th.item() + 1e-6)))
                        ratio = max(rw, rh)
                        match_item = (iou, ratio, scale_idx, anchor_idx)
                        candidate_matches.append(match_item)
                        if ratio < Config.ANCHOR_MATCH_RATIO_THRESH and iou >= Config.POSITIVE_ANCHOR_IOU:
                            best_item = scale_best_matches.get(scale_idx)
                            if best_item is None or (iou > best_item[0]) or (iou == best_item[0] and ratio < best_item[1]):
                                scale_best_matches[scale_idx] = match_item

                if len(scale_best_matches) == 0:
                    relaxed_scale_matches = {}
                    for match_item in candidate_matches:
                        iou, ratio, scale_idx, _ = match_item
                        if ratio >= Config.ANCHOR_MATCH_RATIO_THRESH:
                            continue
                        best_item = relaxed_scale_matches.get(scale_idx)
                        if best_item is None or (iou > best_item[0]) or (iou == best_item[0] and ratio < best_item[1]):
                            relaxed_scale_matches[scale_idx] = match_item
                    scale_best_matches = relaxed_scale_matches

                ratio_matches = sorted(scale_best_matches.values(), key=lambda item: (-item[0], item[1]))
                if len(ratio_matches) == 0:
                    ratio_matches = [max(candidate_matches, key=lambda item: (item[0], -item[1]))]

                best_scale_iou = ratio_matches[0][0]
                ratio_matches = [
                    item for item in ratio_matches
                    if item[0] >= best_scale_iou * Config.POSITIVE_SCALE_IOU_RATIO
                ]
                max_positive = Config.MAX_POSITIVE_ANCHORS if Config.MAX_POSITIVE_ANCHORS > 0 else len(prepared_scales)
                ratio_matches = ratio_matches[:max_positive]

                for match_iou, _, scale_idx, anchor_idx in ratio_matches:
                    scale_data = prepared_scales[scale_idx]
                    gx = tx * scale_data["grid_w"]
                    gy = ty * scale_data["grid_h"]
                    grid_x = max(0, min(int(gx.item()), scale_data["grid_w"] - 1))
                    grid_y = max(0, min(int(gy.item()), scale_data["grid_h"] - 1))

                    anchor_w, anchor_h = scale_data["anchors"][anchor_idx]
                    offsets = [(0, 0)]
                    if Config.ASSIGN_NEIGHBOR_CELLS:
                        dx_l = gx.item() - grid_x
                        dx_r = (grid_x + 1) - gx.item()
                        dy_t = gy.item() - grid_y
                        dy_b = (grid_y + 1) - gy.item()
                        if dx_l < Config.NEIGHBOR_ASSIGN_MARGIN:
                            offsets.append((-1, 0))
                        if dx_r < Config.NEIGHBOR_ASSIGN_MARGIN:
                            offsets.append((1, 0))
                        if dy_t < Config.NEIGHBOR_ASSIGN_MARGIN:
                            offsets.append((0, -1))
                        if dy_b < Config.NEIGHBOR_ASSIGN_MARGIN:
                            offsets.append((0, 1))

                    for offset_x, offset_y in offsets:
                        assign_x = grid_x + offset_x
                        assign_y = grid_y + offset_y
                        if assign_x < 0 or assign_x >= scale_data["grid_w"] or assign_y < 0 or assign_y >= scale_data["grid_h"]:
                            continue
                        if scale_data["target_match_iou"][b, assign_y, assign_x, anchor_idx] >= match_iou:
                            continue

                        scale_data["target_boxes"][b, assign_y, assign_x, anchor_idx, 0] = gx - assign_x
                        scale_data["target_boxes"][b, assign_y, assign_x, anchor_idx, 1] = gy - assign_y
                        scale_data["target_boxes"][b, assign_y, assign_x, anchor_idx, 2] = torch.log(tw / anchor_w + 1e-6)
                        scale_data["target_boxes"][b, assign_y, assign_x, anchor_idx, 3] = torch.log(th / anchor_h + 1e-6)
                        scale_data["target_boxes_abs"][b, assign_y, assign_x, anchor_idx, 0] = tx * Config.IMG_SIZE
                        scale_data["target_boxes_abs"][b, assign_y, assign_x, anchor_idx, 1] = ty * Config.IMG_SIZE
                        scale_data["target_boxes_abs"][b, assign_y, assign_x, anchor_idx, 2] = tw
                        scale_data["target_boxes_abs"][b, assign_y, assign_x, anchor_idx, 3] = th
                        scale_data["target_obj"][b, assign_y, assign_x, anchor_idx] = 1.0
                        scale_data["target_cls"][b, assign_y, assign_x, anchor_idx, cls_id] = 1.0
                        scale_data["target_match_iou"][b, assign_y, assign_x, anchor_idx] = match_iou
                        scale_data["target_scale_weight"][b, assign_y, assign_x, anchor_idx] = size_weight

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


def compute_detector_heatmap_aux_loss(predictions, teacher_features, targets):
    device = teacher_features.device
    if (
        Config.DETECTOR_HEAT_AUX_WEIGHT <= 0 and
        Config.DETECTOR_FEATURE_AUX_WEIGHT <= 0 and
        Config.DETECTOR_OBJ_HEAT_AUX_WEIGHT <= 0
    ):
        zero = torch.zeros((), device=device)
        return zero, {"detector_aux": 0.0, "detector_heat_aux": 0.0, "detector_feat_aux": 0.0, "detector_obj_aux": 0.0}

    target_heat = teacher_features.detach().float().clamp(0.0, 1.0)
    target_heat = target_heat.pow(Config.DETECTOR_AUX_TARGET_POWER)
    target_heat = torch.where(
        target_heat >= Config.DETECTOR_AUX_TARGET_THRESH,
        target_heat,
        torch.zeros_like(target_heat),
    )
    heat_aux = torch.zeros((), device=device)
    feat_aux = torch.zeros((), device=device)
    obj_aux = torch.zeros((), device=device)
    bce_logits = nn.BCEWithLogitsLoss()

    if teacher_features.requires_grad and Config.DETECTOR_HEAT_AUX_WEIGHT > 0:
        count = 0
        _, _, heat_h, heat_w = teacher_features.shape
        for sample_idx, sample_targets in enumerate(targets):
            if len(sample_targets) == 0:
                continue
            sample_targets = sample_targets.to(device)
            cx = (sample_targets[:, 1] * heat_w).long().clamp(0, heat_w - 1)
            cy = (sample_targets[:, 2] * heat_h).long().clamp(0, heat_h - 1)
            heat_aux = heat_aux + (1.0 - teacher_features[sample_idx, 0, cy, cx]).mean()
            count += 1
        if count > 0:
            heat_aux = heat_aux / count

    for pred in predictions:
        batch_size, _, grid_h, grid_w = pred.shape
        pred_view = pred.view(batch_size, 3, 5 + Config.NUM_CLASSES, grid_h, grid_w)
        pred_obj = pred_view[:, :, 4:5, :, :]
        target_scale = F.interpolate(
            target_heat,
            size=(grid_h, grid_w),
            mode="bilinear",
            align_corners=False,
        ).unsqueeze(1)

        obj_aux = obj_aux + bce_logits(pred_obj, target_scale.expand_as(pred_obj))
        obj_prob = torch.sigmoid(pred_obj).mean(dim=1)
        feat_aux = feat_aux + F.mse_loss(obj_prob, target_scale.squeeze(1))

    num_scales = max(len(predictions), 1)
    feat_aux = feat_aux / num_scales
    obj_aux = obj_aux / num_scales
    total = (
        Config.DETECTOR_HEAT_AUX_WEIGHT * heat_aux +
        Config.DETECTOR_FEATURE_AUX_WEIGHT * feat_aux +
        Config.DETECTOR_OBJ_HEAT_AUX_WEIGHT * obj_aux
    )
    stats = {
        "detector_aux": float(total.detach().item()),
        "detector_heat_aux": float(heat_aux.detach().item()),
        "detector_feat_aux": float(feat_aux.detach().item()),
        "detector_obj_aux": float(obj_aux.detach().item()),
    }
    return total, stats

def prepare_batch(batch, device):
    batch_targets = []
    batch_images = []
    for img_tensor, targets in batch:
        batch_images.append(img_tensor)
        batch_targets.append(targets)
    batch_images = torch.stack(batch_images).to(device)
    return batch_images, batch_targets


def apply_mixup_batch(batch_images, batch_targets):
    if (
        batch_images.shape[0] < 2 or
        Config.MIXUP_PROB <= 0.0 or
        np.random.rand() >= Config.MIXUP_PROB
    ):
        return batch_images, batch_targets

    lam = float(np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA))
    indices = torch.randperm(batch_images.shape[0], device=batch_images.device)
    mixed_images = lam * batch_images + (1.0 - lam) * batch_images[indices]
    mixed_targets = []
    for sample_idx, targets in enumerate(batch_targets):
        paired_targets = batch_targets[int(indices[sample_idx].item())]
        if len(targets) == 0:
            mixed_targets.append(paired_targets.clone())
        elif len(paired_targets) == 0:
            mixed_targets.append(targets.clone())
        else:
            mixed_targets.append(torch.cat([targets, paired_targets], dim=0))
    return mixed_images, mixed_targets


def apply_warmup_lr(optimizer, base_lrs, global_step, warmup_steps):
    if warmup_steps <= 0:
        return
    ratio = min((global_step + 1) / warmup_steps, 1.0)
    factor = Config.WARMUP_START_FACTOR + (1.0 - Config.WARMUP_START_FACTOR) * ratio
    for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
        param_group["lr"] = base_lr * factor


class ModelEMA:
    def __init__(self, model, decay=0.9998):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad = False

    def update(self, model):
        self.updates += 1
        with torch.no_grad():
            msd = model.state_dict()
            for key, value in self.ema.state_dict().items():
                if key not in msd:
                    continue
                model_value = msd[key].detach()
                if value.dtype.is_floating_point:
                    value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
                else:
                    value.copy_(model_value)

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

def evaluate_model(model, dataloader, criterion, device, stage="joint", detector_module=None):
    model.eval()
    eval_detector = detector_module if detector_module is not None else model.detector
    metric_storage = {
        iou_thr: {cls_id: [] for cls_id in range(Config.NUM_CLASSES)}
        for iou_thr in Config.MAP_IOU_THRESHOLDS
    }
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

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            batch_images, batch_targets = prepare_batch(batch, device)
            teacher_features = model.teacher(batch_images)
            feature_loss, feature_stats = compute_teacher_guidance_loss(
                teacher_features,
                batch_targets,
                stage=stage,
                images=batch_images,
            )
            if stage == "teacher":
                loss = feature_loss
                loss_stats = {"box": 0.0, "obj": 0.0, "noobj": 0.0, "cls": 0.0}
                predictions = None
            else:
                predictions = eval_detector(teacher_features)
                loss, loss_stats = criterion(predictions, batch_targets)
                detector_aux_loss, detector_aux_stats = compute_detector_heatmap_aux_loss(
                    predictions,
                    teacher_features,
                    batch_targets,
                )
                loss = loss + feature_loss + detector_aux_loss
                feature_stats.update(detector_aux_stats)

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

            if predictions is None:
                continue

            detections = decode_detections(
                predictions,
                conf_thresh=Config.EVAL_CONF_THRESH,
                nms_thresh=Config.EVAL_NMS_THRESH,
                max_det=Config.EVAL_MAX_DET,
            )

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

                sample_detections = sorted(sample_detections, key=lambda det: det[4], reverse=True)
                matched_by_threshold = {
                    iou_thr: {cls_id: set() for cls_id in gt_by_class}
                    for iou_thr in Config.MAP_IOU_THRESHOLDS
                }

                for det in sample_detections:
                    cls_id = int(det[5])
                    gt_boxes = gt_by_class.get(cls_id, [])
                    det_tensor = torch.tensor(det[:4], dtype=torch.float32).unsqueeze(0)

                    for iou_thr in Config.MAP_IOU_THRESHOLDS:
                        best_iou = 0.0
                        best_gt_idx = -1
                        matched = matched_by_threshold[iou_thr].setdefault(cls_id, set())
                        for gt_idx, gt_box in enumerate(gt_boxes):
                            if gt_idx in matched:
                                continue
                            gt_tensor = torch.tensor(gt_box, dtype=torch.float32).unsqueeze(0)
                            iou = float(bbox_iou_xywh(det_tensor, gt_tensor).item())
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx

                        is_tp = best_iou >= iou_thr
                        metric_storage[iou_thr][cls_id].append((float(det[4]), 1.0 if is_tp else 0.0))
                        if is_tp:
                            matched.add(best_gt_idx)
                        if iou_thr == Config.METRIC_IOU_THRESHOLD:
                            if is_tp:
                                total_tp += 1
                            else:
                                total_fp += 1

                metric_matched = matched_by_threshold[Config.METRIC_IOU_THRESHOLD]
                for cls_id, gt_boxes in gt_by_class.items():
                    total_fn += len(gt_boxes) - len(metric_matched.get(cls_id, set()))

    num_batches = max(len(dataloader), 1)
    avg_losses = {key: value / num_batches for key, value in component_totals.items()}
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1_score = 2.0 * precision * recall / (precision + recall + 1e-6)

    ap_by_threshold = {}
    for iou_thr in Config.MAP_IOU_THRESHOLDS:
        ap_values = []
        for cls_id in range(Config.NUM_CLASSES):
            ap = compute_average_precision(metric_storage[iou_thr][cls_id], gt_counts[cls_id])
            if ap is not None:
                ap_values.append(ap)
        ap_by_threshold[iou_thr] = float(np.mean(ap_values)) if len(ap_values) > 0 else 0.0

    map50 = ap_by_threshold.get(0.5, 0.0)
    map5095 = float(np.mean(list(ap_by_threshold.values()))) if len(ap_by_threshold) > 0 else 0.0

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score),
        "map50": map50,
        "map5095": map5095,
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
    map5095_x, map5095_y = _valid_history_points(history["map5095"])
    if len(precision_x) > 0:
        axes[1].plot(precision_x, precision_y, label="Precision", linewidth=2, marker="o", markersize=4)
    if len(recall_x) > 0:
        axes[1].plot(recall_x, recall_y, label="Recall", linewidth=2, marker="o", markersize=4)
    if len(f1_x) > 0:
        axes[1].plot(f1_x, f1_y, label="F1", linewidth=2, marker="o", markersize=4)
    if len(map_x) > 0:
        axes[1].plot(map_x, map_y, label="mAP@0.5", linewidth=2, marker="o", markersize=4)
    if len(map5095_x) > 0:
        axes[1].plot(map5095_x, map5095_y, label="mAP@0.5:0.95", linewidth=2, marker="o", markersize=4)
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


def filter_visual_detections(detections, img_size):
    if detections is None or len(detections) == 0:
        return detections
    det = np.asarray(detections, dtype=np.float32)
    if det.size == 0:
        return det.reshape(0, 6)

    margin = float(img_size) * Config.VIS_EDGE_MARGIN_RATIO
    x_center = det[:, 0]
    y_center = det[:, 1]
    width = det[:, 2]
    height = det[:, 3]
    conf = det[:, 4]
    x1 = x_center - width / 2.0
    y1 = y_center - height / 2.0
    x2 = x_center + width / 2.0
    y2 = y_center + height / 2.0
    touches_edge = (x1 <= margin) | (y1 <= margin) | (x2 >= img_size - margin) | (y2 >= img_size - margin)
    keep = (~touches_edge) | (conf >= Config.VIS_EDGE_CONF_THRESH)
    det = det[keep]
    if len(det) > Config.VIS_MAX_DET:
        det = det[np.argsort(det[:, 4])[::-1][:Config.VIS_MAX_DET]]
    return det


def _wh_iou_np(w1, h1, w2, h2, eps=1e-6):
    inter = min(w1, w2) * min(h1, h2)
    union = (w1 * h1) + (w2 * h2) - inter + eps
    return float(inter / union)


def draw_best_matching_anchor_boxes(ax, x_center, y_center, width, height, img_size):
    anchor_colors = ["yellow", "cyan", "magenta"]
    center_x = float(x_center) * img_size
    center_y = float(y_center) * img_size
    target_w = float(width) * img_size
    target_h = float(height) * img_size

    for scale_idx, scale_anchors in enumerate(Config.ANCHORS):
        best_anchor = scale_anchors[0]
        best_iou = -1.0
        for aw, ah in scale_anchors:
            iou = _wh_iou_np(target_w, target_h, float(aw), float(ah))
            if iou > best_iou:
                best_iou = iou
                best_anchor = (aw, ah)

        aw, ah = float(best_anchor[0]), float(best_anchor[1])
        x1 = center_x - aw / 2.0
        y1 = center_y - ah / 2.0
        rect = patches.Rectangle(
            (x1, y1),
            aw,
            ah,
            linewidth=1.0,
            edgecolor=anchor_colors[scale_idx % len(anchor_colors)],
            facecolor="none",
            linestyle="--",
            alpha=0.9,
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(2.0, y1 - 4.0),
            f"A@{Config.STRIDES[scale_idx]}",
            color=anchor_colors[scale_idx % len(anchor_colors)],
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35),
        )


def _xywh_iou_np(box1, box2, eps=1e-6):
    x1_1 = box1[0] - box1[2] / 2.0
    y1_1 = box1[1] - box1[3] / 2.0
    x2_1 = box1[0] + box1[2] / 2.0
    y2_1 = box1[1] + box1[3] / 2.0
    x1_2 = box2[0] - box2[2] / 2.0
    y1_2 = box2[1] - box2[3] / 2.0
    x2_2 = box2[0] + box2[2] / 2.0
    y2_2 = box2[1] + box2[3] / 2.0

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area1 = max(0.0, x2_1 - x1_1) * max(0.0, y2_1 - y1_1)
    area2 = max(0.0, x2_2 - x1_2) * max(0.0, y2_2 - y1_2)
    union = area1 + area2 - inter + eps
    return inter / union


def select_matched_visual_detections(detections, targets, img_size, iou_thresh=None):
    det = np.asarray(detections, dtype=np.float32)
    if det.size == 0:
        det = det.reshape(0, 6)
    if iou_thresh is None:
        iou_thresh = Config.VIS_MATCHED_IOU_THRESH

    gt_boxes = []
    for target in targets:
        if target.shape[0] < 5:
            continue
        gt_boxes.append([
            float(target[1].item() * img_size),
            float(target[2].item() * img_size),
            float(target[3].item() * img_size),
            float(target[4].item() * img_size),
            int(target[0].item()),
        ])

    if len(det) == 0 or len(gt_boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32), {
            "matched": 0,
            "total": int(len(det)),
            "gt": int(len(gt_boxes)),
        }

    matched_det_indices = []
    used_det_indices = set()
    for gt_box in gt_boxes:
        gt_cls = gt_box[4]
        best_idx = -1
        best_iou = iou_thresh
        for det_idx, det_item in enumerate(det):
            if det_idx in used_det_indices or int(det_item[5]) != gt_cls:
                continue
            iou = _xywh_iou_np(det_item[:4], gt_box[:4])
            if iou > best_iou:
                best_iou = iou
                best_idx = det_idx
        if best_idx >= 0:
            used_det_indices.add(best_idx)
            matched_det_indices.append(best_idx)

    if len(matched_det_indices) == 0:
        matched = np.zeros((0, 6), dtype=np.float32)
    else:
        matched = det[matched_det_indices]
        matched = matched[np.argsort(matched[:, 4])[::-1]]

    return matched, {
        "matched": int(len(matched_det_indices)),
        "total": int(len(det)),
        "gt": int(len(gt_boxes)),
    }


def save_detection_visualization(epoch, model, dataset, save_dir, prefix="train", device=None, detector_module=None, stage_name=None):
    # Save a detection visualization image for the current epoch.
    if dataset is None or len(dataset) == 0:
        return

    if device is None:
        device = Config.DEVICE
    eval_detector = detector_module if detector_module is not None else model.detector
    
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set the model to evaluation mode
    was_training = model.training
    model.eval()
    
    # Use a fixed seeded subset so visualizations stay comparable across runs
    num_samples = min(Config.VIS_BATCH_SIZE, len(dataset))
    generator = torch.Generator()
    generator.manual_seed(Config.VIS_SEED)
    indices = torch.randperm(len(dataset), generator=generator)[:num_samples]
    
    # Create the figure and axes for the visualization image
    fig, axes = plt.subplots(num_samples, 4, figsize=(24, 6 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Process each sample in the dataset
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Load the image and targets
            img_tensor, targets = dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device
            
            # Convert the image to numpy array
            img_np = img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)  # Convert to RGB if grayscale
            
            # Get the teacher feature
            teacher_feature_tensor = model.teacher(img_tensor)
            teacher_output = teacher_feature_tensor.squeeze(0).cpu().numpy()
            teacher_output = enhance_feature_for_display(teacher_output)  # Enhance the feature for display purposes
            target_map_tensor, foreground_mask = build_teacher_target_map(
                targets=[targets],
                img_size=teacher_feature_tensor.shape[-1],
                device=teacher_feature_tensor.device,
                dtype=teacher_feature_tensor.dtype,
                images=img_tensor,
            )
            target_map_np = target_map_tensor.squeeze(0).squeeze(0).cpu().numpy()
            feature_fp32 = teacher_feature_tensor[0].float()
            fg_mask_sample = foreground_mask[0]
            bg_mask_sample = ~fg_mask_sample
            feature_mean = float(feature_fp32.mean().item())
            feature_std = float(feature_fp32.std(unbiased=False).item())
            fg_mean = float(feature_fp32[fg_mask_sample].mean().item()) if fg_mask_sample.any() else 0.0
            bg_mean = float(feature_fp32[bg_mask_sample].mean().item()) if bg_mask_sample.any() else 0.0
            
            # Process the teacher feature
            if teacher_output.ndim == 3:
                teacher_output = teacher_output.squeeze(0)
            
            detections = [np.zeros((0, 6), dtype=np.float32)]
            vis_match_stats = {"matched": 0, "total": 0, "gt": int(len(targets))}
            if stage_name != "teacher":
                vis_conf_thresh = Config.VIS_DETECTOR_CONF_THRESH if stage_name in {"detector", "joint"} else Config.VIS_CONF_THRESH
                preds = eval_detector(teacher_feature_tensor)
                detections = decode_detections(
                    preds,
                    conf_thresh=vis_conf_thresh,
                    nms_thresh=Config.VIS_NMS_THRESH,
                    max_det=Config.VIS_MAX_DET,
                )
                detections[0] = filter_visual_detections(detections[0], Config.IMG_SIZE)
                vis_match_stats["total"] = int(len(detections[0]))
                if Config.VIS_MATCHED_ONLY and stage_name in {"detector", "joint"}:
                    detections[0], vis_match_stats = select_matched_visual_detections(
                        detections[0],
                        targets,
                        Config.IMG_SIZE,
                    )
            
            # Display the original image
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"Original Image {i+1}")
            axes[i, 0].axis("off")
            
            # Display the teacher feature
            axes[i, 1].imshow(teacher_output, cmap="hot") # 使用热映射显示教师特征
            axes[i, 1].set_title(
                f"Teacher Feature {i+1}\nmean={feature_mean:.3f} std={feature_std:.3f} fg={fg_mean:.3f} bg={bg_mean:.3f}"
            )
            axes[i, 1].axis("off")
            
            # Display the ground truth
            axes[i, 2].imshow(img_np)
            axes[i, 2].set_title(f"Ground Truth {i+1}")
            axes[i, 2].axis("off")
            
            # Display the model predictions
            for target_idx in range(len(targets)):
                cls_id, x_center, y_center, width, height = targets[target_idx]
                cls_id = int(cls_id.item())
                
                # Calculate the coordinates of the bounding box
                x1 = int((x_center - width / 2) * Config.IMG_SIZE)
                y1 = int((y_center - height / 2) * Config.IMG_SIZE)
                w = int(width * Config.IMG_SIZE)
                h = int(height * Config.IMG_SIZE)
                
                # Draw the bounding box
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                        edgecolor='green', facecolor='none')
                axes[i, 2].add_patch(rect)
                # Add the class name
                axes[i, 2].text(x1, y1 - 5, Config.CLASS_NAMES[cls_id], 
                              color='green', fontsize=10, fontweight='bold')
                if Config.VIS_DRAW_GT_ANCHORS:
                    draw_best_matching_anchor_boxes(
                        axes[i, 2],
                        float(x_center.item()),
                        float(y_center.item()),
                        float(width.item()),
                        float(height.item()),
                        Config.IMG_SIZE,
                    )

            if Config.VIS_DRAW_GT_ANCHORS:
                axes[i, 2].set_title(f"Ground Truth + Best Anchors {i+1}")
            
            # Display the model predictions
            if stage_name == "teacher":
                axes[i, 3].imshow(target_map_np, cmap="magma", vmin=0.0, vmax=1.0)
                axes[i, 3].set_title(f"Teacher Target {i+1}")
                axes[i, 3].axis("off")
            else:
                axes[i, 3].imshow(img_np)
                axes[i, 3].set_title(
                    f"Predictions {i+1}\nmatched={vis_match_stats['matched']}/{vis_match_stats['gt']} shown={len(detections[0])} raw={vis_match_stats['total']}"
                )
                axes[i, 3].axis("off")

                if len(detections[0]) > 0:
                    for det in detections[0]:
                        x_center, y_center, w, h, conf, cls_id = det
                        cls_id = int(cls_id)

                        x1 = int(x_center - w / 2)
                        y1 = int(y_center - h / 2)
                        w = int(w)
                        h = int(h)

                        color = plt.cm.tab20(cls_id / max(Config.NUM_CLASSES, 1))
                        rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                                edgecolor=color, facecolor='none')
                        axes[i, 3].add_patch(rect)

                        label = f"{Config.CLASS_NAMES[cls_id]}: {conf:.2f}"
                        axes[i, 3].text(x1, y1 - 5, label,
                                      color=color, fontsize=10, fontweight='bold',
                                      bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor=color, alpha=0.7))
    
    # Save the figure
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_epoch_{epoch:03d}.png")
    plt.savefig(save_path, dpi=Config.VIS_DPI)
    plt.close()  # Close the figure to release memory.
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

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
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

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
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


def load_detector_checkpoint(detector, checkpoint_path, device):
    if not checkpoint_path:
        return False, "Detector checkpoint: not configured"
    if not os.path.exists(checkpoint_path):
        return False, f"Detector checkpoint not found: {checkpoint_path}"

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("detector_state_dict", "model_state_dict", "state_dict", "model"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break

    detector_state = detector.state_dict()
    compatible_state = {}
    for key, value in state_dict.items():
        normalized_key = key[9:] if key.startswith("detector.") else key
        if normalized_key in detector_state and detector_state[normalized_key].shape == value.shape:
            compatible_state[normalized_key] = value

    if len(compatible_state) == 0:
        return False, f"No compatible YOLOLightHead weights found in: {checkpoint_path}"

    detector.load_state_dict({**detector_state, **compatible_state}, strict=False)
    return True, f"Loaded {len(compatible_state)} detector tensors from: {checkpoint_path}"


def initialize_detector_weights(detector, device):
    init_mode = Config.get_detector_init_mode()
    if init_mode == "scratch":
        return False, "Detector init mode: scratch (training from random initialization)"

    checkpoint_path = Config.get_detector_init_checkpoint()
    loaded_detector, detector_message = load_detector_checkpoint(detector, checkpoint_path, device)
    if loaded_detector:
        return True, f"Detector init mode: checkpoint ({detector_message})"

    return False, f"Detector init mode: checkpoint requested but unavailable, fallback to scratch ({detector_message})"


def build_optimizer_from_model(model, stage):
    if stage == "teacher":
        trainable_params = [p for p in model.teacher.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable_params, lr=Config.TEACHER_STAGE_LR, weight_decay=Config.WEIGHT_DECAY)
    if stage == "detector":
        trainable_params = [p for p in model.detector.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable_params, lr=Config.DETECTOR_STAGE_LR, weight_decay=Config.WEIGHT_DECAY)

    param_groups = []
    teacher_params = [p for p in model.teacher.parameters() if p.requires_grad]
    detector_params = [p for p in model.detector.parameters() if p.requires_grad]
    if teacher_params:
        param_groups.append({"params": teacher_params, "lr": Config.JOINT_TEACHER_LR})
    if detector_params:
        param_groups.append({"params": detector_params, "lr": Config.JOINT_DETECTOR_LR})
    return torch.optim.Adam(param_groups, weight_decay=Config.WEIGHT_DECAY)


def set_teacher_trainable(model, trainable):
    for p in model.teacher.parameters():
        p.requires_grad = trainable


def set_detector_trainable(model, trainable):
    for p in model.detector.parameters():
        p.requires_grad = trainable


def is_detector_frozen(model):
    return not any(p.requires_grad for p in model.detector.parameters())


def compute_teacher_consistency_loss(teacher_reference, batch_images, teacher_features):
    if teacher_reference is None or Config.TEACHER_CONSISTENCY_WEIGHT <= 0:
        zero = torch.zeros((), device=teacher_features.device, dtype=teacher_features.dtype)
        return zero, 0.0

    with torch.no_grad():
        reference_features = teacher_reference(batch_images)
    consistency = F.mse_loss(teacher_features, reference_features)
    return consistency * Config.TEACHER_CONSISTENCY_WEIGHT, float(consistency.detach().item())

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
    loaded_detector, detector_message = initialize_detector_weights(detector, device)
    log_to_file(detector_message)
    
    log_to_file("Loading model components...")
    model = TeacherWithDetector(teacher=teacher, detector=detector).to(device)
    teacher_reference = None
    if loaded_teacher:
        teacher_reference = copy.deepcopy(model.teacher).eval()
        for p in teacher_reference.parameters():
            p.requires_grad = False
    
    for p in model.teacher.parameters():
        p.requires_grad = not freeze_teacher
    set_detector_trainable(model, False)
    
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
        "map5095": [],
    }
    best_loss = float('inf')
    best_map5095 = -1.0
    best_map50 = -1.0
    use_amp = Config.USE_AMP and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ModelEMA(model.detector, decay=Config.EMA_DECAY) if Config.USE_EMA else None

    stage_plan = []
    skip_teacher_stage = (
        loaded_teacher and
        Config.SKIP_TEACHER_STAGE_IF_INIT and
        Config.get_teacher_init_mode() == "checkpoint"
    )
    if skip_teacher_stage:
        log_to_file("Teacher checkpoint is available; skip teacher-only bootstrap stage to avoid collapsing teacher features.")
    elif not freeze_teacher and Config.TEACHER_STAGE_EPOCHS > 0:
        stage_plan.append(("teacher", Config.TEACHER_STAGE_EPOCHS))
    if Config.DETECTOR_STAGE_EPOCHS > 0:
        stage_plan.append(("detector", Config.DETECTOR_STAGE_EPOCHS))
    if Config.JOINT_FINETUNE_EPOCHS > 0:
        stage_plan.append(("joint", Config.JOINT_FINETUNE_EPOCHS))
    
    log_to_file("="*60)
    log_to_file("Training model...")
    log_to_file("="*60)
    global_epoch = 0
    for stage_name, stage_epochs in stage_plan:
        if stage_name == "teacher":
            set_teacher_trainable(model, True)
            set_detector_trainable(model, False)
        elif stage_name == "detector":
            set_teacher_trainable(model, False)
            set_detector_trainable(model, True)
        else:
            set_teacher_trainable(model, True)
            set_detector_trainable(model, True)

        optimizer = build_optimizer_from_model(model, stage_name)
        base_lrs = [param_group["lr"] for param_group in optimizer.param_groups]
        stage_warmup_steps = Config.WARMUP_EPOCHS * max(len(train_loader), 1)
        stage_global_step = 0
        log_to_file(f"Entering stage: {stage_name} ({stage_epochs} epochs)")
        init_epoch_log_table(stage_name)

        for stage_epoch in range(stage_epochs):
            epoch = global_epoch
            model.train()
            train_component_sums = {
                "total": 0.0,
                "box": 0.0,
                "obj": 0.0,
                "noobj": 0.0,
                "cls": 0.0,
            }
            feature_stat_sums = {}

            if stage_name == "teacher":
                phase = "teacher_bootstrap"
            else:
                detector_epoch = stage_epoch if stage_name == "detector" else Config.DETECTOR_STAGE_EPOCHS + stage_epoch
                phase = f"{stage_name}_{criterion.set_epoch_weights(detector_epoch)}"

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [{phase}]", leave=True):
                batch_images, batch_targets = prepare_batch(batch, device)
                if stage_name != "teacher":
                    batch_images, batch_targets = apply_mixup_batch(batch_images, batch_targets)

                if stage_global_step < stage_warmup_steps:
                    apply_warmup_lr(optimizer, base_lrs, stage_global_step, stage_warmup_steps)

                optimizer.zero_grad(set_to_none=True)
                autocast_device = "cuda" if use_amp else "cpu"
                with torch.autocast(device_type=autocast_device, enabled=use_amp):
                    teacher_features = model.teacher(batch_images)
                    feature_loss, feature_stats = compute_teacher_guidance_loss(
                        teacher_features,
                        batch_targets,
                        stage=stage_name,
                        images=batch_images,
                    )
                    consistency_loss, _ = compute_teacher_consistency_loss(
                        teacher_reference if stage_name == "joint" else None,
                        batch_images,
                        teacher_features,
                    )
                    if stage_name == "teacher":
                        loss = feature_loss + consistency_loss
                        loss_stats = {"box": 0.0, "obj": 0.0, "noobj": 0.0, "cls": 0.0}
                    else:
                        predictions = model.detector(teacher_features)
                        det_loss, loss_stats = criterion(predictions, batch_targets)
                        detector_aux_loss, detector_aux_stats = compute_detector_heatmap_aux_loss(
                            predictions,
                            teacher_features,
                            batch_targets,
                        )
                        loss = det_loss + feature_loss + consistency_loss + detector_aux_loss
                        feature_stats.update(detector_aux_stats)

                scaler.scale(loss).backward()
                if Config.GRAD_CLIP_NORM > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        Config.GRAD_CLIP_NORM,
                    )
                scaler.step(optimizer)
                scaler.update()
                if stage_name != "teacher" and ema is not None:
                    ema.update(model.detector)

                train_component_sums["total"] += float(loss.detach().item())
                train_component_sums["box"] += loss_stats["box"]
                train_component_sums["obj"] += loss_stats["obj"]
                train_component_sums["noobj"] += loss_stats["noobj"]
                train_component_sums["cls"] += loss_stats["cls"]
                for key, value in feature_stats.items():
                    feature_stat_sums[key] = feature_stat_sums.get(key, 0.0) + float(value)
                stage_global_step += 1

            avg_train = {key: value / max(len(train_loader), 1) for key, value in train_component_sums.items()}
            avg_feature_stats = {
                key: value / max(len(train_loader), 1)
                for key, value in feature_stat_sums.items()
            }
            history["train_total"].append(avg_train["total"])
            current_lr = max(param_group["lr"] for param_group in optimizer.param_groups)

            val_losses = None
            val_metrics = None
            if val_loader is not None and ((epoch + 1) % Config.VAL_INTERVAL == 0):
                eval_detector = ema.ema if (ema is not None and stage_name != "teacher") else model.detector
                val_losses, val_metrics = evaluate_model(
                    model,
                    val_loader,
                    criterion,
                    device,
                    stage=stage_name,
                    detector_module=eval_detector,
                )
                history["val_total"].append(val_losses["total"])
                history["precision"].append(val_metrics["precision"])
                history["recall"].append(val_metrics["recall"])
                history["f1"].append(val_metrics["f1"])
                history["map50"].append(val_metrics["map50"])
                history["map5095"].append(val_metrics["map5095"])
            else:
                history["val_total"].append(np.nan)
                history["precision"].append(np.nan)
                history["recall"].append(np.nan)
                history["f1"].append(np.nan)
                history["map50"].append(np.nan)
                history["map5095"].append(np.nan)

            is_best = False
            if val_metrics is not None and stage_name != "teacher":
                if val_metrics["map5095"] > best_map5095:
                    best_map5095 = val_metrics["map5095"]
                    best_map50 = val_metrics["map50"]
                    is_best = True
            elif stage_name == "teacher" and avg_train["total"] < best_loss:
                best_loss = avg_train["total"]

            if is_best:
                best_loss = avg_train["total"]
                best_model_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "detector_best.pth")
                detector_to_save = ema.ema if ema is not None else model.detector
                torch.save(detector_to_save.state_dict(), best_model_path)
                if Config.SAVE_TEACHER_WEIGHTS:
                    torch.save(model.teacher.state_dict(), teacher_best_path)
                    torch.save({
                        "teacher_state_dict": model.teacher.state_dict(),
                        "detector_state_dict": detector_to_save.state_dict(),
                        "epoch": epoch,
                        "loss": avg_train["total"],
                        "val_map50": best_map50 if val_metrics is not None else None,
                        "val_map5095": best_map5095 if val_metrics is not None else None,
                    }, joint_best_path)
                if val_metrics is not None:
                    print(
                        f"Epoch {epoch + 1}: best checkpoint updated "
                        f"(train_loss={avg_train['total']:.6f}, val_map50={val_metrics['map50']:.4f}, "
                        f"val_map5095={val_metrics['map5095']:.4f})"
                    )

            if (epoch + 1) % Config.VIS_INTERVAL == 0:
                vis_detector = ema.ema if (ema is not None and stage_name != "teacher") else model.detector
                save_detection_visualization(
                    epoch,
                    model,
                    vis_dataset,
                    vis_dir,
                    prefix=f"{vis_prefix}_{stage_name}",
                    device=device,
                    detector_module=vis_detector,
                    stage_name=stage_name,
                )

            log_epoch_table_row(
                epoch=epoch,
                phase=phase,
                train_loss=avg_train["total"],
                val_loss=val_losses["total"] if val_losses is not None else None,
                precision=val_metrics["precision"] if val_metrics is not None else None,
                recall=val_metrics["recall"] if val_metrics is not None else None,
                f1_score=val_metrics["f1"] if val_metrics is not None else None,
                map50=val_metrics["map50"] if val_metrics is not None else None,
                map5095=val_metrics["map5095"] if val_metrics is not None else None,
                lr=current_lr,
                best_status=Config.EPOCH_TABLE_BEST_MARK if is_best else "",
                stage_name=stage_name,
                feature_stats=avg_feature_stats,
            )

            save_training_curves(history, Config.TEACHER_OUTPUT_DIR)
            global_epoch += 1

    model_save_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "detector_final.pth")
    final_detector_to_save = ema.ema if ema is not None else model.detector
    torch.save(final_detector_to_save.state_dict(), model_save_path)
    if Config.SAVE_TEACHER_WEIGHTS:
        torch.save(model.teacher.state_dict(), teacher_final_path)
        torch.save({
            "teacher_state_dict": model.teacher.state_dict(),
            "detector_state_dict": final_detector_to_save.state_dict(),
            "epoch": global_epoch - 1,
            "loss": history["train_total"][-1] if len(history["train_total"]) > 0 else None,
            "val_map50": best_map50 if best_map50 >= 0 else None,
            "val_map5095": best_map5095 if best_map5095 >= 0 else None,
        }, joint_final_path)
        log_to_file(f"Teacher best weights saved to: {teacher_best_path}")
        log_to_file(f"Teacher final weights saved to: {teacher_final_path}")
    
    append_plain_log(Config.get_epoch_table_separator(stage_plan[-1][0] if len(stage_plan) > 0 else None))
    log_to_file("="*60)
    log_to_file("Training complete")
    log_to_file(f"Best detector model saved to: {os.path.join(Config.TEACHER_OUTPUT_DIR, 'detector_best.pth')}")
    log_to_file(f"Final detector model saved to: {model_save_path}")
    log_to_file(f"Teacher output directory: {Config.TEACHER_OUTPUT_DIR}")
    log_to_file("="*60)

if __name__ == "__main__":
    train()
