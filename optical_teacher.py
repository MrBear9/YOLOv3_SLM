import yaml
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.ops import nms
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from PIL import Image

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
    checkpoint_path = resolve_project_path(checkpoint_path)
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


def load_detector_checkpoint(detector, checkpoint_path, device):
    if not checkpoint_path:
        return False, "Detector checkpoint: not configured"
    checkpoint_path = resolve_project_path(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        return False, f"Detector checkpoint not found: {checkpoint_path}"

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = None
    for key in ("detector_state_dict", "model_state_dict", "state_dict", "model"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state_dict = checkpoint[key]
            break
    if state_dict is None:
        state_dict = checkpoint

    detector_state = detector.state_dict()
    compatible_state = {}
    for key, value in state_dict.items():
        normalized_key = key[9:] if key.startswith("detector.") else key
        if normalized_key in detector_state and detector_state[normalized_key].shape == value.shape:
            compatible_state[normalized_key] = value

    if len(compatible_state) == 0:
        return False, f"No compatible detector weights found in: {checkpoint_path}"

    detector.load_state_dict({**detector_state, **compatible_state}, strict=False)
    return True, f"Loaded {len(compatible_state)} detector tensors from: {checkpoint_path}"

# =========================================================
# 读取data.yaml获取类别信息
# =========================================================
def load_class_names(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    class_names = config.get('names', [])

    if isinstance(class_names, list):
        CLASS_NAMES = {i: name for i, name in enumerate(class_names)}
    else:
        raise ValueError(f"Unsupported 'names' format in {yaml_path}, must be list or dict")

    num_classes = len(CLASS_NAMES)
    return CLASS_NAMES, num_classes

def load_anchor_groups(anchor_yaml_path):
    """Load grouped YOLO anchors from an external yaml file."""
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

# =========================================================
# 配置类 - 集中管理所有参数
# =========================================================
class Config:
    """光学教师网络训练配置类"""
    
    # 数据集参数
    YAML_PATH = r"data\military\data.yaml"
    CLASS_NAMES = None
    NUM_CLASSES = None
    
    # 输出路径配置
    TEACHER_OUTPUT_DIR = r"output\OpticalTeacher_deep_teacher"
    TEACHER_CHECKPOINT = r"output\oty_m1\teacher_best.pth" # path to teacher checkpoint
    DETECTOR_CHECKPOINT = r"output\oty_m1\detector_best.pth"
    LOG_ROOT_DIR = None
    LOG_FILE = None
    TIMESTAMP = None
    TRAIN_START_TIME = None
    VISUALIZATION_DIR = None
    
    # 训练参数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 640
    BATCH_SIZE = 8  # 批次大小，用于训练时的内存占用
    EPOCHS = 200
    
    # 损失权重
    BOX_WEIGHT = 0.05  # 目标框损失权重，用于计算总损失
    OBJ_WEIGHT = 1.5  # 目标损失权重，用于计算总损失
    NOOBJ_WEIGHT = 0.2
    CLS_WEIGHT = 0.15  # 类别损失权重，用于计算总损失
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # 锚框设置
    STRIDES = [8, 16, 32]
    DEFAULT_ANCHORS = [
        [[26,23], [47,49], [100,67]],   # P3: 小目标 / 较小目标
        [[103,169], [203,107], [351,177]],  # P4: 中目标 / 长条目标
        [[241,354], [534,299], [568,528]]  # P5: 大目标 / 超大目标（军舰等）
    ]
    ANCHOR_CONFIG_PATH = r"output\anchor_clustering\yolo_anchors.yaml"
    USE_EXTERNAL_ANCHORS = True
    ANCHORS = None
    ANCHOR_SOURCE = "default"
    
    # 优化器参数
    LEARNING_RATE = 3e-4  # 学习率，用于优化模型参数
    WEIGHT_DECAY = 3e-5  # 权重衰减，用于防止过拟合
    OPTIMIZER = "Adam"
    
    # 光学传播参数
    WAVELENGTH = 532e-9  # 波长，单位：米
    PIXEL_SIZE = 6.4e-6  # 像素大小，单位：米
    PROP_DISTANCE_1 = 0.01  # 传播距离1，单位：米
    PROP_DISTANCE_2 = 0.02  # 传播距离2，单位：米
    SLM_MODE = "phase"  # SLM模式，"phase"或"amplitude"
    RESOLUTION = (640, 640)
    OPTICAL_FIELD_EPS = 1e-8
    OPTICAL_NORM_EPS = 1e-6
    DISPLAY_EPS = 1e-6
    IOU_EPS = 1e-6
    YOLO_HEAD_IN_CHANNELS = 1
    YOLO_HEAD_BASE_CHANNELS = 32
    
    # 训练控制参数
    NORM_ENABLE_EPOCH = 50  # 启用归一化的轮数
    VIS_INTERVAL = 2  # 可视化间隔，单位：轮数
    
    # 损失函数权重
    LOSS_FULL_WEIGHT = 0.2
    LOSS_LOW1_WEIGHT = 0.8
    LOSS_LOW2_WEIGHT = 0.4
    LOSS_SSIM_WEIGHT = 0.3
    LOSS_GRAD_WEIGHT = 0.25
    LOSS_FREQ_WEIGHT = 0.15
    LOSS_PHASE_SMOOTH_WEIGHT = 0.02
    RESPONSE_MAP_WEIGHT = 0.3
    USE_DETECTION_RESPONSE_LOSS = True
    
    # 检测参数
    CONF_THRESH = 0.5  # 置信度阈值，用于筛选检测框
    NMS_THRESH = 0.4  # NMS阈值，用于合并重叠检测框
    MAX_DET = 8  # 最大检测框数量，用于限制检测框数量
    AGNOSTIC_NMS = True
    METRIC_IOU_THRESHOLD = 0.5
    
    # 可视化参数
    VIS_BATCH_SIZE = 4  # 可视化批次大小，用于可视化检测结果
    VIS_DPI = 120  # 可视化DPI，用于调整可视化结果的清晰度
    
    @classmethod
    def initialize(cls):
        """初始化配置，加载类别信息和创建输出目录"""
        teacher_checkpoint = os.environ.get("OPTICAL_STUDENT_TEACHER_CHECKPOINT")
        if teacher_checkpoint:
            cls.TEACHER_CHECKPOINT = teacher_checkpoint
        detector_checkpoint = os.environ.get("OPTICAL_STUDENT_DETECTOR_CHECKPOINT")
        if detector_checkpoint:
            cls.DETECTOR_CHECKPOINT = detector_checkpoint
        output_dir = os.environ.get("OPTICAL_STUDENT_OUTPUT_DIR")
        if output_dir:
            cls.TEACHER_OUTPUT_DIR = output_dir
        anchor_config_path = os.environ.get("OPTICAL_STUDENT_ANCHOR_CONFIG_PATH")
        if anchor_config_path:
            cls.ANCHOR_CONFIG_PATH = anchor_config_path

        cls.YAML_PATH = resolve_project_path(cls.YAML_PATH)
        cls.TEACHER_OUTPUT_DIR = resolve_project_path(cls.TEACHER_OUTPUT_DIR)
        cls.TEACHER_CHECKPOINT = resolve_project_path(cls.TEACHER_CHECKPOINT)
        cls.DETECTOR_CHECKPOINT = resolve_project_path(cls.DETECTOR_CHECKPOINT)
        cls.ANCHOR_CONFIG_PATH = resolve_project_path(cls.ANCHOR_CONFIG_PATH)
        cls.CLASS_NAMES, cls.NUM_CLASSES = load_class_names(cls.YAML_PATH)
        cls.ANCHORS = [[anchor.copy() for anchor in layer] for layer in cls.DEFAULT_ANCHORS]
        cls.ANCHOR_SOURCE = "default"
        if cls.USE_EXTERNAL_ANCHORS:
            try:
                cls.ANCHORS = load_anchor_groups(cls.ANCHOR_CONFIG_PATH)
                cls.ANCHOR_SOURCE = cls.ANCHOR_CONFIG_PATH
            except Exception as exc:
                cls.ANCHORS = [[anchor.copy() for anchor in layer] for layer in cls.DEFAULT_ANCHORS]
                cls.ANCHOR_SOURCE = f"default (external load failed: {exc})"
        os.makedirs(cls.TEACHER_OUTPUT_DIR, exist_ok=True)
        cls.LOG_ROOT_DIR = os.path.join(cls.TEACHER_OUTPUT_DIR, "logs")
        cls.VISUALIZATION_DIR = os.path.join(cls.TEACHER_OUTPUT_DIR, "visualizations")
        os.makedirs(cls.LOG_ROOT_DIR, exist_ok=True)
        os.makedirs(cls.VISUALIZATION_DIR, exist_ok=True)
        cls.TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.LOG_FILE = os.path.join(cls.LOG_ROOT_DIR, f"training_log_{cls.TIMESTAMP}.txt")
    
    @classmethod
    def get_detector_output_channels(cls):
        """获取检测头输出通道数"""
        return 3 * (4 + 1 + cls.NUM_CLASSES)

    @classmethod
    def get_best_model_path(cls):
        return os.path.join(cls.TEACHER_OUTPUT_DIR, "optical_student_best.pth")

    @classmethod
    def get_final_model_path(cls):
        return os.path.join(cls.TEACHER_OUTPUT_DIR, "optical_student_final.pth")

    @classmethod
    def get_loss_curve_path(cls):
        return os.path.join(cls.TEACHER_OUTPUT_DIR, "loss_curve.png")
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("="*80)
        print("光学教师网络训练配置")
        print("="*80)
        print(f"设备: {cls.DEVICE}")
        print(f"图像尺寸: {cls.IMG_SIZE}")
        print(f"批次大小: {cls.BATCH_SIZE}")
        print(f"训练轮数: {cls.EPOCHS}")
        print(f"类别数量: {cls.NUM_CLASSES}")
        print(f"学习率: {cls.LEARNING_RATE}")
        print(f"权重衰减: {cls.WEIGHT_DECAY}")
        print("="*80)

# 初始化配置
Config.initialize()
Config.print_config()

def log_to_file(message, log_file=None, also_print=True):
    if log_file is None:
        log_file = Config.LOG_FILE
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")
    if also_print:
        print(message)

def init_log_file(log_file=None):
    if log_file is None:
        log_file = Config.LOG_FILE
    Config.TRAIN_START_TIME = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("光学教师网络训练日志\n")
        f.write("="*80 + "\n")
        f.write(f"训练开始时间: {Config.TRAIN_START_TIME}\n")
        f.write("="*80 + "\n\n")

init_log_file()
log_to_file(f"日志文件路径: {Config.LOG_FILE}")
log_to_file(f"可视化结果保存路径: {Config.TEACHER_OUTPUT_DIR}")
log_to_file(f"加载类别信息: {Config.CLASS_NAMES}, 类别数: {Config.NUM_CLASSES}")

# =========================================================
# 光学层（相位 + 振幅调制）
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
        amp = torch.sqrt(intensity.clamp(min=0) + Config.OPTICAL_FIELD_EPS)
        field = torch.complex(amp, torch.zeros_like(amp))
        field = self.prop1(self.slm1(field))
        field = self.prop2(self.slm2(field))
        out = torch.abs(field)**2
        if self.enable_norm:
            out = out / (out.mean(dim=[2,3], keepdim=True) + Config.OPTICAL_NORM_EPS)
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
        return x * self.fc(self.pool(x))


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
# 教师网络（卷积-仿YOLOV3）
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
        base_ch = Config.YOLO_HEAD_BASE_CHANNELS

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
# 数据集
# =========================================================
class MilitaryFeatureDataset(Dataset):
    def __init__(self, yaml_path=None, teacher=None, device=None):
        if yaml_path is None:
            yaml_path = Config.YAML_PATH
        if device is None:
            device = Config.DEVICE
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        img_dir = os.path.join(cfg["path"], cfg["train"])
        self.files = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith(".jpg")
        ])
        self.teacher = teacher.to(device).eval()
        self.device = device
        self.gray = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.Grayscale(1),
            transforms.ToTensor()
        ])
        self.rgb = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        x = self.gray(img)
        with torch.no_grad():
            t = self.teacher(self.rgb(img).unsqueeze(0).to(self.device))
            t = F.interpolate(t, size=(Config.IMG_SIZE, Config.IMG_SIZE), mode="bilinear", align_corners=False)
            t = t.squeeze(0).cpu()
        return x, t

# =========================================================
# 损失函数
# =========================================================
class CompositeOpticalFeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = nn.AvgPool2d(8)
        self.pool2 = nn.AvgPool2d(32)
        self.avg_pool = nn.AvgPool2d(3, 1, 1)

    def ssim_loss(self, s, t):
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu_s = self.avg_pool(s)
        mu_t = self.avg_pool(t)
        sigma_s = self.avg_pool(s * s) - mu_s * mu_s
        sigma_t = self.avg_pool(t * t) - mu_t * mu_t
        sigma_st = self.avg_pool(s * t) - mu_s * mu_t
        ssim_map = ((2 * mu_s * mu_t + c1) * (2 * sigma_st + c2)) / (
            (mu_s * mu_s + mu_t * mu_t + c1) * (sigma_s + sigma_t + c2) + Config.OPTICAL_FIELD_EPS
        )
        return torch.clamp((1.0 - ssim_map.mean()) * 0.5, min=0.0)

    def gradient_loss(self, s, t):
        grad_s_x = s[:, :, :, 1:] - s[:, :, :, :-1]
        grad_t_x = t[:, :, :, 1:] - t[:, :, :, :-1]
        grad_s_y = s[:, :, 1:, :] - s[:, :, :-1, :]
        grad_t_y = t[:, :, 1:, :] - t[:, :, :-1, :]
        return F.l1_loss(grad_s_x, grad_t_x) + F.l1_loss(grad_s_y, grad_t_y)

    def frequency_loss(self, s, t):
        freq_s = torch.fft.fft2(s.squeeze(1), norm="ortho")
        freq_t = torch.fft.fft2(t.squeeze(1), norm="ortho")
        mag_s = torch.log1p(torch.abs(freq_s))
        mag_t = torch.log1p(torch.abs(freq_t))
        return F.l1_loss(mag_s, mag_t)

    def phase_smoothness_loss(self, student):
        phase_terms = []
        for slm_layer in (student.slm1, student.slm2):
            phase = torch.remainder(slm_layer.phase_raw, 2 * np.pi)
            cos_phase = torch.cos(phase)
            sin_phase = torch.sin(phase)
            phase_terms.append(torch.abs(cos_phase[:, :, :, 1:] - cos_phase[:, :, :, :-1]).mean())
            phase_terms.append(torch.abs(cos_phase[:, :, 1:, :] - cos_phase[:, :, :-1, :]).mean())
            phase_terms.append(torch.abs(sin_phase[:, :, :, 1:] - sin_phase[:, :, :, :-1]).mean())
            phase_terms.append(torch.abs(sin_phase[:, :, 1:, :] - sin_phase[:, :, :-1, :]).mean())
        return sum(phase_terms) / max(len(phase_terms), 1)

    def forward(self, s, t, student):
        loss_full = F.mse_loss(s, t)
        loss_low1 = F.mse_loss(self.pool1(s), self.pool1(t))
        loss_low2 = F.mse_loss(self.pool2(s), self.pool2(t))
        loss_ssim = self.ssim_loss(s, t)
        loss_grad = self.gradient_loss(s, t)
        loss_freq = self.frequency_loss(s, t)
        loss_phase = self.phase_smoothness_loss(student)
        total = (
            loss_full * Config.LOSS_FULL_WEIGHT +
            loss_low1 * Config.LOSS_LOW1_WEIGHT +
            loss_low2 * Config.LOSS_LOW2_WEIGHT +
            loss_ssim * Config.LOSS_SSIM_WEIGHT +
            loss_grad * Config.LOSS_GRAD_WEIGHT +
            loss_freq * Config.LOSS_FREQ_WEIGHT +
            loss_phase * Config.LOSS_PHASE_SMOOTH_WEIGHT
        )
        stats = {
            "full": float(loss_full.detach().item()),
            "low1": float(loss_low1.detach().item()),
            "low2": float(loss_low2.detach().item()),
            "ssim": float(loss_ssim.detach().item()),
            "grad": float(loss_grad.detach().item()),
            "freq": float(loss_freq.detach().item()),
            "phase": float(loss_phase.detach().item()),
            "total": float(total.detach().item()),
        }
        return total, stats

# =========================================================
# 辅助函数
# =========================================================
def prediction_response_map(pred):
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    pred = pred.permute(1, 2, 0).reshape(grid_h, grid_w, 3, -1)
    obj_conf = torch.sigmoid(pred[..., 4])
    cls_conf, _ = torch.sigmoid(pred[..., 5:]).max(dim=-1)
    response = (obj_conf * cls_conf).max(dim=-1).values
    return response.detach().cpu().numpy()


def prediction_response_tensor(preds):
    response_maps = []
    for scale_idx, pred in enumerate(preds):
        grid_h, grid_w = pred.shape[2], pred.shape[3]
        pred = pred.permute(0, 2, 3, 1).reshape(pred.shape[0], grid_h, grid_w, 3, -1)
        obj_conf = torch.sigmoid(pred[..., 4])
        cls_conf = torch.sigmoid(pred[..., 5:]).max(dim=-1).values
        response = (obj_conf * cls_conf).max(dim=-1).values.unsqueeze(1)
        response = F.interpolate(response, size=(Config.IMG_SIZE, Config.IMG_SIZE), mode="bilinear", align_corners=False)
        response_maps.append(response)
    return torch.stack(response_maps, dim=0).max(dim=0).values


def normalize_response_map(response):
    peak = response.amax(dim=(2, 3), keepdim=True)
    return response / (peak + Config.OPTICAL_NORM_EPS)


def compute_detection_response_loss(detector, student_feature, teacher_feature):
    if detector is None:
        zero = torch.zeros((), device=student_feature.device, dtype=student_feature.dtype)
        return zero, {"response": 0.0}

    student_preds = detector(student_feature)
    student_response = normalize_response_map(prediction_response_tensor(student_preds))

    with torch.no_grad():
        teacher_preds = detector(teacher_feature.detach())
        teacher_response = normalize_response_map(prediction_response_tensor(teacher_preds))

    response_loss = F.mse_loss(student_response, teacher_response)
    weighted_loss = response_loss * Config.RESPONSE_MAP_WEIGHT
    return weighted_loss, {"response": float(response_loss.detach().item())}

def enhance_feature_for_display(feature_map):
    feature_map = np.asarray(feature_map, dtype=np.float32)
    low = np.percentile(feature_map, 2)
    high = np.percentile(feature_map, 98)
    if high - low < Config.DISPLAY_EPS:
        return np.zeros_like(feature_map)
    
    feature_map = np.clip((feature_map - low) / (high - low), 0.0, 1.0)
    return np.power(feature_map, 0.8)

def xywh_to_xyxy(boxes):
    half_w = boxes[:, 2] / 2
    half_h = boxes[:, 3] / 2
    return torch.stack([
        boxes[:, 0] - half_w,
        boxes[:, 1] - half_h,
        boxes[:, 0] + half_w,
        boxes[:, 1] + half_h,
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
    strides = Config.STRIDES
    
    for i, pred in enumerate(preds):
        grid_h, grid_w = pred.shape[2], pred.shape[3]
        stride = strides[i]
        
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

        x_center = ((torch.sigmoid(tx) + grid_x) * stride).clamp(0, img_size - 1)
        y_center = ((torch.sigmoid(ty) + grid_y) * stride).clamp(0, img_size - 1)
        w = (torch.exp(torch.clamp(tw, min=-8.0, max=8.0)) * anchor_tensor[..., 0]).clamp(1, img_size)
        h = (torch.exp(torch.clamp(th, min=-8.0, max=8.0)) * anchor_tensor[..., 1]).clamp(1, img_size)

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
    
    # 按置信度排序并限制最大检测数量
    for b in range(batch_size):
        if len(detections[b]) > 0:
            detections[b] = apply_nms(torch.cat(detections[b], dim=0), nms_thresh, max_det)
        else:
            detections[b] = np.zeros((0, 6), dtype=np.float32)

    return detections

# =========================================================
# 可视化工具
# =========================================================
def save_feature_comparison(epoch, teacher, student, save_dir, prefix="train", idx=None, input_images=None):
    os.makedirs(save_dir, exist_ok=True)
    
    if idx is None:
        batch_size = min(Config.VIS_BATCH_SIZE, teacher.shape[0])
        indices = torch.randperm(teacher.shape[0])[:batch_size]
    else:
        indices = [idx] if isinstance(idx, int) else idx
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    
    def norm(x): return (x - x.min()) / (x.max() - x.min() + Config.OPTICAL_FIELD_EPS)
    
    for i, idx in enumerate(indices):
        if input_images is not None:
            input_img = input_images[idx, 0].numpy()
            axes[i, 0].imshow(norm(input_img), cmap="gray")
            axes[i, 0].set_title(f"Input {i+1}")
            axes[i, 0].axis("off")
        else:
            axes[i, 0].axis("off")
        
        t = teacher[idx, 0].numpy()
        s = student[idx, 0].numpy()
        r = np.abs(s - t)
        
        axes[i, 1].imshow(norm(t), cmap="hot")
        axes[i, 1].set_title(f"Teacher {i+1}")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(norm(s), cmap="hot")
        axes[i, 2].set_title(f"Student {i+1}")
        axes[i, 2].axis("off")
        
        im = axes[i, 3].imshow(r, cmap="viridis")
        axes[i, 3].set_title(f"Error {i+1}")
        axes[i, 3].axis("off")
        
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046)
    
    for i in range(len(indices), 4):
        for j in range(4):
            axes[i, j].axis("off")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_epoch_{epoch:03d}.png")
    plt.savefig(save_path, dpi=Config.VIS_DPI)
    plt.close()

def save_detection_visualization(images, detections, save_dir, epoch, class_names):
    os.makedirs(save_dir, exist_ok=True)
    
    num_images = min(4, len(images))
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    colors = ['red', 'green', 'blue', 'yellow']
    
    for i in range(num_images):
        img = images[i].permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        
        axes[i].imshow(img)
        
        if i < len(detections) and len(detections[i]) > 0:
            for det in detections[i]:
                x, y, w, h, conf, cls_id = det
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
                
                color = colors[int(cls_id) % len(colors)]
                cls_name = class_names.get(int(cls_id), f"cls_{int(cls_id)}")
                
                axes[i].add_patch(plt.Rectangle((x1, y1), w, h, 
                                               fill=False, edgecolor=color, linewidth=2))
                axes[i].text(x1, y1-5, f"{cls_name}: {conf:.2f}", 
                            color=color, fontsize=10, 
                            bbox=dict(facecolor='white', alpha=0.7))
        
        axes[i].set_title(f"Image {i+1}")
        axes[i].axis("off")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"detections_epoch_{epoch:03d}.png")
    plt.savefig(save_path, dpi=Config.VIS_DPI)
    plt.close()

# =========================================================
# 日志输出所有参数
# =========================================================
def log_all_parameters():
    log_to_file("="*80)
    log_to_file("训练参数设置")
    log_to_file("="*80)
    
    log_to_file("\n【数据集参数】")
    log_to_file(f"  YAML路径: {Config.YAML_PATH}")
    log_to_file(f"  类别数量: {Config.NUM_CLASSES}")
    log_to_file(f"  类别名称: {Config.CLASS_NAMES}")
    
    log_to_file("\n【训练参数】")
    log_to_file(f"  设备: {Config.DEVICE}")
    log_to_file(f"  图像尺寸: {Config.IMG_SIZE}")
    log_to_file(f"  批次大小: {Config.BATCH_SIZE}")
    log_to_file(f"  训练轮数: {Config.EPOCHS}")
    
    log_to_file("\n【损失权重】")
    log_to_file(f"  边界框损失权重: {Config.BOX_WEIGHT}")
    log_to_file(f"  目标性损失权重: {Config.OBJ_WEIGHT}")
    log_to_file(f"  非目标损失权重: {Config.NOOBJ_WEIGHT}")
    log_to_file(f"  类别损失权重: {Config.CLS_WEIGHT}")
    log_to_file(f"  Focal Alpha: {Config.FOCAL_ALPHA}")
    log_to_file(f"  Focal Gamma: {Config.FOCAL_GAMMA}")
    
    log_to_file("\n【锚框设置】")
    log_to_file(f"  步长: {Config.STRIDES}")
    log_to_file(f"  使用外部锚框: {Config.USE_EXTERNAL_ANCHORS}")
    log_to_file(f"  外部锚框配置: {Config.ANCHOR_CONFIG_PATH}")
    log_to_file(f"  当前锚框来源: {Config.ANCHOR_SOURCE}")
    log_to_file(f"  P3锚框: {Config.ANCHORS[0]}")
    log_to_file(f"  P4锚框: {Config.ANCHORS[1]}")
    log_to_file(f"  P5锚框: {Config.ANCHORS[2]}")
    log_to_file(f"  Agnostic NMS: {Config.AGNOSTIC_NMS}")
    log_to_file(f"  Metric IoU Threshold: {Config.METRIC_IOU_THRESHOLD}")
    
    log_to_file("\n【模型参数】")
    teacher = ConvTeacher()
    student = OpticalStudent()
    detector = YOLOLightHead(
        in_channels=Config.YOLO_HEAD_IN_CHANNELS,
        out_channels=Config.get_detector_output_channels(),
    )
    
    log_to_file(f"  ConvTeacher可训练参数: {sum(p.numel() for p in teacher.parameters() if p.requires_grad):,}")
    log_to_file(f"  OpticalStudent可训练参数: {sum(p.numel() for p in student.parameters() if p.requires_grad):,}")
    log_to_file(f"  检测头输出通道数: {Config.get_detector_output_channels()}")
    log_to_file(f"  检测头可训练参数: {sum(p.numel() for p in detector.parameters() if p.requires_grad):,}")
    
    log_to_file("\n【优化器参数】")
    log_to_file(f"  优化器: {Config.OPTIMIZER}")
    log_to_file(f"  学习率: {Config.LEARNING_RATE}")
    log_to_file(f"  权重衰减: {Config.WEIGHT_DECAY}")
    
    log_to_file("\n【光学参数】")
    log_to_file(f"  波长: {Config.WAVELENGTH}")
    log_to_file(f"  像素尺寸: {Config.PIXEL_SIZE}")
    log_to_file(f"  传播距离1: {Config.PROP_DISTANCE_1}")
    log_to_file(f"  传播距离2: {Config.PROP_DISTANCE_2}")
    log_to_file(f"  SLM模式: {Config.SLM_MODE}")
    
    log_to_file("\n【路径设置】")
    log_to_file(f"  日志文件: {Config.LOG_FILE}")
    log_to_file(f"  可视化结果: {Config.VISUALIZATION_DIR}")
    log_to_file(f"  模型保存: {Config.get_final_model_path()}")
    log_to_file(f"  教师权重: {Config.TEACHER_CHECKPOINT}")
    log_to_file(f"  检测头权重: {Config.DETECTOR_CHECKPOINT}")

    log_to_file("\n【学生损失】")
    log_to_file(f"  Full / Low1 / Low2: {Config.LOSS_FULL_WEIGHT} / {Config.LOSS_LOW1_WEIGHT} / {Config.LOSS_LOW2_WEIGHT}")
    log_to_file(f"  SSIM / Grad / Freq / Phase: {Config.LOSS_SSIM_WEIGHT} / {Config.LOSS_GRAD_WEIGHT} / {Config.LOSS_FREQ_WEIGHT} / {Config.LOSS_PHASE_SMOOTH_WEIGHT}")
    log_to_file(f"  Use detection response loss: {Config.USE_DETECTION_RESPONSE_LOSS}")
    log_to_file(f"  Response map weight: {Config.RESPONSE_MAP_WEIGHT}")
    
    log_to_file("\n" + "="*80)
    log_to_file("参数设置输出完成")
    log_to_file("="*80 + "\n")

# =========================================================
# 训练函数
# =========================================================
def train():
    log_all_parameters()
    device = Config.DEVICE
    log_to_file(f"使用设备: {device}")
    
    log_to_file("初始化教师网络（ConvTeacher）...")
    teacher = ConvTeacher().to(device)
    teacher_loaded, teacher_message = load_teacher_checkpoint(teacher, Config.TEACHER_CHECKPOINT, device)
    log_to_file(teacher_message)
    if not teacher_loaded:
        raise FileNotFoundError(
            f"Unable to start optical training without a trained teacher checkpoint: {Config.TEACHER_CHECKPOINT}"
        )

    for p in teacher.parameters():
        p.requires_grad = False
    log_to_file("教师网络参数已冻结")

    detector = None
    if Config.USE_DETECTION_RESPONSE_LOSS:
        log_to_file("初始化冻结检测头（YOLOLightHead）用于 response map 约束...")
        detector = YOLOLightHead(
            in_channels=Config.YOLO_HEAD_IN_CHANNELS,
            out_channels=Config.get_detector_output_channels(),
        ).to(device)
        detector_loaded, detector_message = load_detector_checkpoint(detector, Config.DETECTOR_CHECKPOINT, device)
        log_to_file(detector_message)
        if detector_loaded:
            detector.eval()
            for p in detector.parameters():
                p.requires_grad = False
            log_to_file("已启用检测响应图约束")
        else:
            detector = None
            log_to_file("检测头权重不可用，跳过 response map 约束")

    log_to_file("初始化学生网络（OpticalStudent）...")
    student = OpticalStudent().to(device)
    
    log_to_file("加载数据集...")
    dataset = MilitaryFeatureDataset(teacher=teacher, device=device)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    log_to_file(f"数据集大小: {len(dataset)}")
    
    optimizer = torch.optim.Adam(student.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = CompositeOpticalFeatureLoss()
    
    vis_dir = Config.VISUALIZATION_DIR
    
    loss_curve = []
    best_loss = float('inf')
    
    log_to_file("="*60)
    log_to_file("开始训练光学相位层...")
    log_to_file("="*60)
    
    for epoch in range(Config.EPOCHS):
        student.train()
        loss_sum = 0
        feature_loss_sum = 0
        response_loss_sum = 0
        
        for x, t in tqdm(loader, desc=f"Epoch {epoch}/{Config.EPOCHS}", leave=True):
            x = x.to(device)
            t = t.to(device)
            y = student(x)
            feature_loss, _ = criterion(y, t, student)
            response_loss, _ = compute_detection_response_loss(detector, y, t)
            loss = feature_loss + response_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            feature_loss_sum += feature_loss.item()
            response_loss_sum += response_loss.item()

        avg_loss = loss_sum / len(loader)
        avg_feature_loss = feature_loss_sum / len(loader)
        avg_response_loss = response_loss_sum / len(loader)
        loss_curve.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = Config.get_best_model_path()
            torch.save(student.state_dict(), best_model_path)
            log_to_file(f"Epoch {epoch}: 保存最佳模型 (Loss: {best_loss:.6f})", also_print=False)
        
        if epoch == Config.NORM_ENABLE_EPOCH:
            student.enable_norm = True
            log_to_file(f"Epoch {epoch}: 启用归一化")
        
        if epoch % Config.VIS_INTERVAL == 0:
            student.eval()
            with torch.no_grad():
                vis_loader = torch.utils.data.DataLoader(dataset, batch_size=Config.VIS_BATCH_SIZE, shuffle=True)
                vis_batch = next(iter(vis_loader))
                x_vis, t_vis = vis_batch
                x_vis = x_vis.to(device)
                t_vis = t_vis
                y_vis = student(x_vis).cpu()
            save_feature_comparison(epoch, t_vis, y_vis, vis_dir, input_images=x_vis.cpu())

        log_to_file(
            f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | "
            f"Feature: {avg_feature_loss:.6f} | Response: {avg_response_loss:.6f}"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(loss_curve, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(Config.get_loss_curve_path(), dpi=Config.VIS_DPI)
    plt.close()

    model_save_path = Config.get_final_model_path()
    torch.save(student.state_dict(), model_save_path)
    
    log_to_file("="*60)
    log_to_file("训练完成！")
    log_to_file(f"最佳模型已保存到: {Config.get_best_model_path()}")
    log_to_file(f"最终模型已保存到: {model_save_path}")
    log_to_file(f"所有结果已保存到: {Config.TEACHER_OUTPUT_DIR}")
    log_to_file("="*60)

if __name__ == "__main__":
    train()
