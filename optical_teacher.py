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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import copy

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_project_path(path):
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def should_use_channels_last():
    return torch.cuda.is_available() and getattr(Config, "ENABLE_CHANNELS_LAST", False)


def prepare_tensor_for_device(tensor, device, *, channels_last=False):
    tensor = tensor.to(device, non_blocking=Config.PIN_MEMORY)
    if not tensor.is_floating_point():
        return tensor
    if channels_last and tensor.dim() == 4 and should_use_channels_last():
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor.contiguous()


def prepare_conv_tensor(tensor):
    if tensor.is_floating_point() and tensor.dim() == 4 and should_use_channels_last():
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor.contiguous() if tensor.is_floating_point() else tensor

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
    TEACHER_OUTPUT_DIR = r"output\OpticalTeacher_StrengTea_teacher"
    TEACHER_CHECKPOINT = r"output\OpticalTeacherYOLO_StrengTea_teacher\teacher_best.pth" # path to teacher checkpoint
    DETECTOR_CHECKPOINT = r"output\OpticalTeacherYOLO_StrengTea_teacher\detector_best.pth"
    LOG_ROOT_DIR = None
    LOG_FILE = None
    TIMESTAMP = None
    TRAIN_START_TIME = None
    VISUALIZATION_DIR = None
    
    # 训练参数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_IDS = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    IMG_SIZE = 640
    BATCH_SIZE = 8  # 批次大小，用于训练时的内存占用
    PHASE1_STUDENT_EPOCHS = 70
    PHASE2_DETECTOR_EPOCHS = 70
    PHASE3_JOINT_EPOCHS = 30
    EPOCHS = PHASE1_STUDENT_EPOCHS + PHASE2_DETECTOR_EPOCHS + PHASE3_JOINT_EPOCHS
    
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
    USE_EXTERNAL_ANCHORS = False
    ANCHORS = None
    ANCHOR_SOURCE = "default"
    
    # 优化器参数
    LEARNING_RATE = 3e-4  # 学习率，用于优化模型参数
    WEIGHT_DECAY = 3e-5  # 权重衰减，用于防止过拟合
    OPTIMIZER = "Adam"
    STUDENT_ONLY_LR = 3e-4
    DETECTOR_ONLY_LR = 2e-4
    JOINT_STUDENT_LR = 5e-5
    JOINT_DETECTOR_LR = 1.5e-4
    
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
    NORM_STRATEGY = "disabled"  # disabled / always / after_epoch
    NORM_ENABLE_EPOCH = 155  # 仅在 NORM_STRATEGY=after_epoch 时生效
    VIS_INTERVAL = 2  # 可视化间隔，单位：轮数
    GRAD_CLIP_NORM = 1.0
    USE_LR_SCHEDULER = True
    LR_SCHEDULER_FACTOR = 0.5
    LR_SCHEDULER_PATIENCE = 12
    MIN_LEARNING_RATE = 5e-5
    DIVERGENCE_RATIO = 1.8
    DIVERGENCE_PATIENCE = 3
    FINAL_MODEL_SOURCE = "best_joint"  # best_joint / last
    RESTORE_PHASE1_BEST_FOR_PHASE2 = True
    RESTORE_PHASE2_BEST_FOR_PHASE3 = True
    
    # 损失函数权重
    LOSS_FULL_WEIGHT = 0.2
    LOSS_LOW1_WEIGHT = 0.8
    LOSS_LOW2_WEIGHT = 0.4
    LOSS_SSIM_WEIGHT = 0.3
    LOSS_GRAD_WEIGHT = 0.25
    LOSS_FREQ_WEIGHT = 0.15
    LOSS_PHASE_SMOOTH_WEIGHT = 0.005
    LOSS_PHASE_DIVERSITY_WEIGHT = 0.05
    PHASE_STD_TARGET = 0.45
    RESPONSE_MAP_WEIGHT = 0.15
    USE_DETECTION_RESPONSE_LOSS = True
    RESPONSE_WARMUP_EPOCHS = 12
    RESPONSE_WARMUP_START_FACTOR = 0.25
    DETECTION_LOSS_WEIGHT_PHASE2 = 1.0
    DETECTION_LOSS_WEIGHT_PHASE3 = 0.8
    FEATURE_LOSS_WEIGHT_PHASE1 = 1.0
    FEATURE_LOSS_WEIGHT_PHASE3 = 0.6
    RESPONSE_LOSS_WEIGHT_PHASE1 = 1.0
    RESPONSE_LOSS_WEIGHT_PHASE3 = 0.5

    POSITIVE_ANCHOR_IOU = 0.25
    NOOBJ_IGNORE_IOU = 0.5
    
    # 检测参数
    CONF_THRESH = 0.5  # 置信度阈值，用于筛选检测框
    NMS_THRESH = 0.4  # NMS阈值，用于合并重叠检测框
    MAX_DET = 8  # 最大检测框数量，用于限制检测框数量
    AGNOSTIC_NMS = True
    METRIC_IOU_THRESHOLD = 0.5
    
    # 可视化参数
    VIS_BATCH_SIZE = 4  # 可视化批次大小，用于可视化检测结果
    VIS_DPI = 120  # 可视化DPI，用于调整可视化结果的清晰度
    NUM_WORKERS = min(8, os.cpu_count() or 0)
    PIN_MEMORY = torch.cuda.is_available()
    PERSISTENT_WORKERS = True
    ENABLE_CUDNN_BENCHMARK = True  # 是否启用 cudnn_benchmark，提高推理效率
    ENABLE_CHANNELS_LAST = False  # 是否使用 channels_last 格式，提高内存效率
    
    @classmethod
    def initialize(cls):
        """初始化配置，加载类别信息和创建输出目录"""
        cls.YAML_PATH = resolve_project_path(cls.YAML_PATH)
        cls.TEACHER_OUTPUT_DIR = resolve_project_path(cls.TEACHER_OUTPUT_DIR)
        cls.TEACHER_CHECKPOINT = resolve_project_path(cls.TEACHER_CHECKPOINT)
        cls.DETECTOR_CHECKPOINT = resolve_project_path(cls.DETECTOR_CHECKPOINT)
        cls.CLASS_NAMES, cls.NUM_CLASSES = load_class_names(cls.YAML_PATH)
        cls.ANCHORS = [[anchor.copy() for anchor in layer] for layer in cls.DEFAULT_ANCHORS]
        cls.ANCHOR_SOURCE = "default"
        if cls.USE_EXTERNAL_ANCHORS:
            try:
                anchor_config_path = resolve_project_path(cls.ANCHOR_CONFIG_PATH)
                cls.ANCHORS = load_anchor_groups(anchor_config_path)
                cls.ANCHOR_SOURCE = anchor_config_path
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
    def get_last_model_path(cls):
        return os.path.join(cls.TEACHER_OUTPUT_DIR, "optical_student_last.pth")

    @classmethod
    def get_detector_best_model_path(cls):
        return os.path.join(cls.TEACHER_OUTPUT_DIR, "detector_best.pth")

    @classmethod
    def get_detector_final_model_path(cls):
        return os.path.join(cls.TEACHER_OUTPUT_DIR, "detector_final.pth")

    @classmethod
    def get_detector_last_model_path(cls):
        return os.path.join(cls.TEACHER_OUTPUT_DIR, "detector_last.pth")

    @classmethod
    def get_phase1_student_best_model_path(cls):
        return os.path.join(cls.TEACHER_OUTPUT_DIR, "optical_student_phase1_best.pth")

    @classmethod
    def get_phase2_student_best_model_path(cls):
        return os.path.join(cls.TEACHER_OUTPUT_DIR, "optical_student_phase2_best.pth")

    @classmethod
    def get_phase2_detector_best_model_path(cls):
        return os.path.join(cls.TEACHER_OUTPUT_DIR, "detector_phase2_best.pth")

    @classmethod
    def get_loss_curve_path(cls):
        return os.path.join(cls.TEACHER_OUTPUT_DIR, "loss_curve.png")

    @classmethod
    def should_enable_student_norm(cls, epoch):
        strategy = str(cls.NORM_STRATEGY).strip().lower()
        if strategy == "disabled":
            return False
        if strategy == "always":
            return True
        if strategy == "after_epoch":
            return epoch >= cls.NORM_ENABLE_EPOCH
        raise ValueError(f"Unsupported NORM_STRATEGY: {cls.NORM_STRATEGY}")

    @classmethod
    def get_response_loss_weight(cls, epoch):
        if not cls.USE_DETECTION_RESPONSE_LOSS:
            return 0.0

        base_weight = float(cls.RESPONSE_MAP_WEIGHT)
        warmup_epochs = max(int(cls.RESPONSE_WARMUP_EPOCHS), 0)
        start_factor = float(cls.RESPONSE_WARMUP_START_FACTOR)
        start_factor = min(max(start_factor, 0.0), 1.0)

        if warmup_epochs <= 0:
            return base_weight

        progress = min(max(epoch / max(warmup_epochs - 1, 1), 0.0), 1.0)
        factor = start_factor + (1.0 - start_factor) * progress
        return base_weight * factor

    @classmethod
    def get_training_phase(cls, epoch):
        if epoch < cls.PHASE1_STUDENT_EPOCHS:
            return "student_only"
        if epoch < cls.PHASE1_STUDENT_EPOCHS + cls.PHASE2_DETECTOR_EPOCHS:
            return "detector_only"
        return "joint_balance"
    
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


def unwrap_module(module):
    return module.module if isinstance(module, nn.DataParallel) else module


def get_runtime_device():
    return torch.device(f"cuda:{Config.GPU_IDS[0]}") if Config.GPU_IDS else torch.device("cpu")


def wrap_data_parallel(module, module_name="module", use_channels_last=False):
    device = get_runtime_device()
    module = module.to(device)
    if use_channels_last and should_use_channels_last():
        module = module.to(memory_format=torch.channels_last)
    if len(Config.GPU_IDS) > 1:
        module = nn.DataParallel(module, device_ids=Config.GPU_IDS, output_device=Config.GPU_IDS[0])
        log_to_file(f"{module_name} wrapped with DataParallel on GPUs: {Config.GPU_IDS}")
    return module


def get_dataloader_kwargs(shuffle=False):
    kwargs = {
        "shuffle": shuffle,
        "num_workers": Config.NUM_WORKERS,
        "pin_memory": Config.PIN_MEMORY,
    }
    if Config.NUM_WORKERS > 0:
        kwargs["persistent_workers"] = Config.PERSISTENT_WORKERS
    return kwargs


def get_phase_statistics(student):
    student_core = unwrap_module(student)
    stats = {}
    for slm_name in ("slm1", "slm2"):
        slm_layer = getattr(student_core, slm_name)
        phase_raw = slm_layer.phase_raw.detach()
        phase_eff = torch.remainder(phase_raw, 2 * np.pi)
        stats[f"{slm_name}_raw_min"] = float(phase_raw.min().item())
        stats[f"{slm_name}_raw_max"] = float(phase_raw.max().item())
        stats[f"{slm_name}_raw_span"] = float((phase_raw.max() - phase_raw.min()).item())
        stats[f"{slm_name}_eff_std"] = float(phase_eff.std().item())
    return stats


def save_phase_visualization(student, save_dir, epoch):
    student_core = unwrap_module(student)
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        ("slm1", "Raw Phase", student_core.slm1.phase_raw.detach().cpu().squeeze().numpy()),
        ("slm1", "Effective Phase", torch.remainder(student_core.slm1.phase_raw.detach(), 2 * np.pi).cpu().squeeze().numpy()),
        ("slm2", "Raw Phase", student_core.slm2.phase_raw.detach().cpu().squeeze().numpy()),
        ("slm2", "Effective Phase", torch.remainder(student_core.slm2.phase_raw.detach(), 2 * np.pi).cpu().squeeze().numpy()),
    ]
    for ax, (slm_name, title, phase_map) in zip(axes.flatten(), panels):
        im = ax.imshow(phase_map, cmap="twilight")
        ax.set_title(f"{slm_name} {title}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"phase_epoch_{epoch:03d}.png"), dpi=Config.VIS_DPI)
    plt.close()


def bootstrap_runtime():
    Config.initialize()
    Config.print_config()
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
    def __init__(self, yaml_path=None):
        if yaml_path is None:
            yaml_path = Config.YAML_PATH
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        img_dir = os.path.join(cfg["path"], cfg["train"])
        labels_dir = os.path.join(os.path.dirname(img_dir), "labels")
        self.files = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])
        self.labels_dir = labels_dir
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

    def _load_targets(self, img_path):
        label_path = os.path.join(self.labels_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}.txt")
        targets = []
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id, x_center, y_center, width, height = map(float, parts[:5])
                    targets.append([cls_id, x_center, y_center, width, height])
        if targets:
            return torch.tensor(targets, dtype=torch.float32)
        return torch.zeros((0, 5), dtype=torch.float32)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        gray_image = self.gray(img)
        rgb_image = self.rgb(img)
        targets = self._load_targets(img_path)
        return gray_image, rgb_image, targets

def feature_dataset_collate_fn(batch):
    gray_images = torch.stack([item[0] for item in batch], dim=0)
    rgb_images = torch.stack([item[1] for item in batch], dim=0)
    targets = [item[2] for item in batch]
    return gray_images, rgb_images, targets

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
        student_core = unwrap_module(student)
        for slm_layer in (student_core.slm1, student_core.slm2):
            phase = torch.remainder(slm_layer.phase_raw, 2 * np.pi)
            cos_phase = torch.cos(phase)
            sin_phase = torch.sin(phase)
            phase_terms.append(torch.abs(cos_phase[:, :, :, 1:] - cos_phase[:, :, :, :-1]).mean())
            phase_terms.append(torch.abs(cos_phase[:, :, 1:, :] - cos_phase[:, :, :-1, :]).mean())
            phase_terms.append(torch.abs(sin_phase[:, :, :, 1:] - sin_phase[:, :, :, :-1]).mean())
            phase_terms.append(torch.abs(sin_phase[:, :, 1:, :] - sin_phase[:, :, :-1, :]).mean())
        return sum(phase_terms) / max(len(phase_terms), 1)

    def phase_diversity_loss(self, student):
        student_core = unwrap_module(student)
        diversity_terms = []
        for slm_layer in (student_core.slm1, student_core.slm2):
            phase = torch.remainder(slm_layer.phase_raw, 2 * np.pi)
            phase_std = phase.std()
            diversity_terms.append(F.relu(Config.PHASE_STD_TARGET - phase_std))
        return sum(diversity_terms) / max(len(diversity_terms), 1)

    def forward(self, s, t, student):
        loss_full = F.mse_loss(s, t)
        loss_low1 = F.mse_loss(self.pool1(s), self.pool1(t))
        loss_low2 = F.mse_loss(self.pool2(s), self.pool2(t))
        loss_ssim = self.ssim_loss(s, t)
        loss_grad = self.gradient_loss(s, t)
        loss_freq = self.frequency_loss(s, t)
        loss_phase = self.phase_smoothness_loss(student)
        loss_phase_diversity = self.phase_diversity_loss(student)
        total = (
            loss_full * Config.LOSS_FULL_WEIGHT +
            loss_low1 * Config.LOSS_LOW1_WEIGHT +
            loss_low2 * Config.LOSS_LOW2_WEIGHT +
            loss_ssim * Config.LOSS_SSIM_WEIGHT +
            loss_grad * Config.LOSS_GRAD_WEIGHT +
            loss_freq * Config.LOSS_FREQ_WEIGHT +
            loss_phase * Config.LOSS_PHASE_SMOOTH_WEIGHT +
            loss_phase_diversity * Config.LOSS_PHASE_DIVERSITY_WEIGHT
        )
        stats = {
            "full": float(loss_full.detach().item()),
            "low1": float(loss_low1.detach().item()),
            "low2": float(loss_low2.detach().item()),
            "ssim": float(loss_ssim.detach().item()),
            "grad": float(loss_grad.detach().item()),
            "freq": float(loss_freq.detach().item()),
            "phase": float(loss_phase.detach().item()),
            "phase_diversity": float(loss_phase_diversity.detach().item()),
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


def compute_detection_response_loss(detector, student_feature, teacher_feature, response_weight=None):
    if detector is None:
        zero = torch.zeros((), device=student_feature.device, dtype=student_feature.dtype)
        return zero, {"response": 0.0, "weight": 0.0}

    if response_weight is None:
        response_weight = Config.RESPONSE_MAP_WEIGHT

    student_preds = detector(prepare_conv_tensor(student_feature))
    student_response = normalize_response_map(prediction_response_tensor(student_preds))

    with torch.no_grad():
        teacher_preds = detector(prepare_conv_tensor(teacher_feature.detach()))
        teacher_response = normalize_response_map(prediction_response_tensor(teacher_preds))

    response_loss = F.mse_loss(student_response, teacher_response)
    weighted_loss = response_loss * response_weight
    return weighted_loss, {
        "response": float(response_loss.detach().item()),
        "weight": float(response_weight),
    }

class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * ((1.0 - p_t).clamp(min=0.0) ** self.gamma)
        loss = loss * focal_weight
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()

def decode_boxes_to_absolute(pred_boxes, anchors, stride):
    device = pred_boxes.device
    grid_h, grid_w = pred_boxes.shape[1], pred_boxes.shape[2]
    dtype = pred_boxes.dtype

    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=dtype),
        torch.arange(grid_w, device=device, dtype=dtype),
        indexing="ij",
    )
    grid_x = grid_x.view(1, grid_h, grid_w, 1)
    grid_y = grid_y.view(1, grid_h, grid_w, 1)
    anchor_tensor = anchors.view(1, 1, 1, 3, 2).to(device=device, dtype=dtype)

    x = (torch.sigmoid(pred_boxes[..., 0]) + grid_x) * stride
    y = (torch.sigmoid(pred_boxes[..., 1]) + grid_y) * stride
    w = torch.exp(torch.clamp(pred_boxes[..., 2], min=-8.0, max=8.0)) * anchor_tensor[..., 0]
    h = torch.exp(torch.clamp(pred_boxes[..., 3], min=-8.0, max=8.0)) * anchor_tensor[..., 1]
    return torch.stack([x, y, w, h], dim=-1)

class DetectionAdaptationLoss(nn.Module):
    def __init__(self, anchors, num_classes, strides):
        super().__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.num_classes = num_classes
        self.strides = strides
        self.focal_loss = SigmoidFocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA, reduction="mean")
        self.last_components = {
            "total": 0.0,
            "box": 0.0,
            "obj": 0.0,
            "noobj": 0.0,
            "cls": 0.0,
        }

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
                "grid_h": grid_h,
                "grid_w": grid_w,
                "stride": self.strides[i],
                "anchors": self.anchors[i].to(device),
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

                best_match = None
                for scale_idx, scale_data in enumerate(prepared_scales):
                    for anchor_idx in range(3):
                        anchor_w, anchor_h = scale_data["anchors"][anchor_idx]
                        inter = torch.minimum(tw, anchor_w) * torch.minimum(th, anchor_h)
                        union = tw * th + anchor_w * anchor_h - inter + 1e-6
                        iou = (inter / union).item()
                        if best_match is None or iou > best_match[0]:
                            best_match = (iou, scale_idx, anchor_idx)

                if best_match is None or best_match[0] < Config.POSITIVE_ANCHOR_IOU:
                    continue

                match_iou, scale_idx, anchor_idx = best_match
                scale_data = prepared_scales[scale_idx]
                gx = tx * scale_data["grid_w"]
                gy = ty * scale_data["grid_h"]
                grid_x = max(0, min(int(gx.item()), scale_data["grid_w"] - 1))
                grid_y = max(0, min(int(gy.item()), scale_data["grid_h"] - 1))

                anchor_w, anchor_h = scale_data["anchors"][anchor_idx]
                scale_data["target_boxes"][b, grid_y, grid_x, anchor_idx, 0] = gx - grid_x
                scale_data["target_boxes"][b, grid_y, grid_x, anchor_idx, 1] = gy - grid_y
                scale_data["target_boxes"][b, grid_y, grid_x, anchor_idx, 2] = torch.log(tw / anchor_w + 1e-6)
                scale_data["target_boxes"][b, grid_y, grid_x, anchor_idx, 3] = torch.log(th / anchor_h + 1e-6)
                scale_data["target_boxes_abs"][b, grid_y, grid_x, anchor_idx] = torch.stack([tx * Config.IMG_SIZE, ty * Config.IMG_SIZE, tw, th])
                scale_data["target_obj"][b, grid_y, grid_x, anchor_idx] = 1.0
                scale_data["target_cls"][b, grid_y, grid_x, anchor_idx, cls_id] = 1.0
                scale_data["target_match_iou"][b, grid_y, grid_x, anchor_idx] = match_iou

        gt_boxes_abs_by_batch = [
            torch.stack(sample_boxes).to(device=device, dtype=predictions[0].dtype)
            if sample_boxes else torch.zeros((0, 4), device=device, dtype=predictions[0].dtype)
            for sample_boxes in gt_boxes_abs_by_batch
        ]

        for scale_data in prepared_scales:
            pred_boxes = scale_data["pred_boxes"]
            pred_obj = scale_data["pred_obj"]
            pred_cls = scale_data["pred_cls"]
            target_obj = scale_data["target_obj"]
            target_cls = scale_data["target_cls"]
            target_boxes_abs = scale_data["target_boxes_abs"]
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
                ciou = bbox_iou_xywh(pred_boxes_abs[obj_mask], target_boxes_abs[obj_mask], ciou=True)
                box_loss = (1.0 - ciou).mean()
                obj_loss = self.focal_loss(pred_obj[obj_mask], target_obj[obj_mask])
                cls_loss = self.focal_loss(pred_cls[obj_mask], target_cls[obj_mask])
            else:
                box_loss = torch.zeros((), device=device)
                obj_loss = torch.zeros((), device=device)
                cls_loss = torch.zeros((), device=device)

            if noobj_mask.any():
                noobj_loss = self.focal_loss(pred_obj[noobj_mask], target_obj[noobj_mask])
            else:
                noobj_loss = torch.zeros((), device=device)

            scale_loss = (
                Config.BOX_WEIGHT * box_loss +
                Config.OBJ_WEIGHT * obj_loss +
                Config.NOOBJ_WEIGHT * noobj_loss +
                Config.CLS_WEIGHT * cls_loss
            )
            total_loss = total_loss + scale_loss
            component_sums["box"] = component_sums["box"] + box_loss.detach()
            component_sums["obj"] = component_sums["obj"] + obj_loss.detach()
            component_sums["noobj"] = component_sums["noobj"] + noobj_loss.detach()
            component_sums["cls"] = component_sums["cls"] + cls_loss.detach()

        self.last_components = {
            "total": float(total_loss.detach().item()),
            "box": float(component_sums["box"].item()),
            "obj": float(component_sums["obj"].item()),
            "noobj": float(component_sums["noobj"].item()),
            "cls": float(component_sums["cls"].item()),
        }
        return total_loss, self.last_components

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

def bbox_iou_matrix_xywh(boxes1, boxes2, eps=1e-7):
    boxes1 = boxes1.reshape(-1, 4)
    boxes2 = boxes2.reshape(-1, 4)

    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0], boxes2.shape[0]),
            device=boxes1.device,
            dtype=boxes1.dtype,
        )

    boxes1_xyxy = xywh_to_xyxy(boxes1)
    boxes2_xyxy = xywh_to_xyxy(boxes2)

    inter_x1 = torch.maximum(boxes1_xyxy[:, None, 0], boxes2_xyxy[None, :, 0])
    inter_y1 = torch.maximum(boxes1_xyxy[:, None, 1], boxes2_xyxy[None, :, 1])
    inter_x2 = torch.minimum(boxes1_xyxy[:, None, 2], boxes2_xyxy[None, :, 2])
    inter_y2 = torch.minimum(boxes1_xyxy[:, None, 3], boxes2_xyxy[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (
        (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]).clamp(min=0) *
        (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1]).clamp(min=0)
    )
    area2 = (
        (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]).clamp(min=0) *
        (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1]).clamp(min=0)
    )
    union = area1[:, None] + area2[None, :] - inter_area + eps
    return inter_area / union

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

    if torch.is_tensor(teacher):
        teacher = teacher.detach().cpu()
    if torch.is_tensor(student):
        student = student.detach().cpu()
    if input_images is not None and torch.is_tensor(input_images):
        input_images = input_images.detach().cpu()
    
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
    log_to_file(f"  GPU IDs: {Config.GPU_IDS if Config.GPU_IDS else 'CPU only'}")
    log_to_file(f"  图像尺寸: {Config.IMG_SIZE}")
    log_to_file(f"  批次大小: {Config.BATCH_SIZE}")
    log_to_file(f"  训练轮数: {Config.EPOCHS}")
    log_to_file(f"  Phase1/Phase2/Phase3: {Config.PHASE1_STUDENT_EPOCHS} / {Config.PHASE2_DETECTOR_EPOCHS} / {Config.PHASE3_JOINT_EPOCHS}")
    log_to_file(f"  Num workers / pin memory: {Config.NUM_WORKERS} / {Config.PIN_MEMORY}")
    log_to_file(f"  CuDNN benchmark / channels last: {Config.ENABLE_CUDNN_BENCHMARK} / {Config.ENABLE_CHANNELS_LAST and torch.cuda.is_available()}")
    log_to_file("  Winograd: 由 cuDNN 在 benchmark 开启时自动选择可用卷积算法")
    
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
    log_to_file(f"  Student-only LR: {Config.STUDENT_ONLY_LR}")
    log_to_file(f"  Detector-only LR: {Config.DETECTOR_ONLY_LR}")
    log_to_file(f"  Joint student/detector LR: {Config.JOINT_STUDENT_LR} / {Config.JOINT_DETECTOR_LR}")
    
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
    log_to_file(f"  检测头最佳/最终保存: {Config.get_detector_best_model_path()} / {Config.get_detector_final_model_path()}")
    log_to_file(f"  Phase2 学生/检测头最佳保存: {Config.get_phase2_student_best_model_path()} / {Config.get_phase2_detector_best_model_path()}")
    log_to_file(f"  教师权重: {Config.TEACHER_CHECKPOINT}")
    log_to_file(f"  检测头权重: {Config.DETECTOR_CHECKPOINT}")

    log_to_file("\n【学生损失】")
    log_to_file(f"  Full / Low1 / Low2: {Config.LOSS_FULL_WEIGHT} / {Config.LOSS_LOW1_WEIGHT} / {Config.LOSS_LOW2_WEIGHT}")
    log_to_file(
        f"  SSIM / Grad / Freq / PhaseSmooth / PhaseDiversity: "
        f"{Config.LOSS_SSIM_WEIGHT} / {Config.LOSS_GRAD_WEIGHT} / {Config.LOSS_FREQ_WEIGHT} / "
        f"{Config.LOSS_PHASE_SMOOTH_WEIGHT} / {Config.LOSS_PHASE_DIVERSITY_WEIGHT}"
    )
    log_to_file(f"  Phase std target: {Config.PHASE_STD_TARGET}")
    log_to_file(f"  Use detection response loss: {Config.USE_DETECTION_RESPONSE_LOSS}")
    log_to_file(f"  Response map weight: {Config.RESPONSE_MAP_WEIGHT}")
    log_to_file(f"  Response warmup epochs / start factor: {Config.RESPONSE_WARMUP_EPOCHS} / {Config.RESPONSE_WARMUP_START_FACTOR}")
    log_to_file(f"  Detection loss weight phase2/phase3: {Config.DETECTION_LOSS_WEIGHT_PHASE2} / {Config.DETECTION_LOSS_WEIGHT_PHASE3}")
    log_to_file(f"  Feature loss weight phase1/phase3: {Config.FEATURE_LOSS_WEIGHT_PHASE1} / {Config.FEATURE_LOSS_WEIGHT_PHASE3}")
    log_to_file(f"  Response loss weight phase1/phase3: {Config.RESPONSE_LOSS_WEIGHT_PHASE1} / {Config.RESPONSE_LOSS_WEIGHT_PHASE3}")
    log_to_file("\n【稳定性控制】")
    log_to_file(f"  Norm strategy: {Config.NORM_STRATEGY}")
    log_to_file(f"  Norm enable epoch: {Config.NORM_ENABLE_EPOCH}")
    log_to_file(f"  Gradient clip norm: {Config.GRAD_CLIP_NORM}")
    log_to_file(f"  Use LR scheduler: {Config.USE_LR_SCHEDULER}")
    log_to_file(f"  Scheduler factor / patience / min lr: {Config.LR_SCHEDULER_FACTOR} / {Config.LR_SCHEDULER_PATIENCE} / {Config.MIN_LEARNING_RATE}")
    log_to_file(f"  Divergence ratio / patience: {Config.DIVERGENCE_RATIO} / {Config.DIVERGENCE_PATIENCE}")
    log_to_file(f"  Final model source: {Config.FINAL_MODEL_SOURCE}")
    log_to_file(f"  Restore phase1 best for phase2: {Config.RESTORE_PHASE1_BEST_FOR_PHASE2}")
    log_to_file(f"  Restore phase2 best for phase3: {Config.RESTORE_PHASE2_BEST_FOR_PHASE3}")
    
    log_to_file("\n" + "="*80)
    log_to_file("参数设置输出完成")
    log_to_file("="*80 + "\n")

# =========================================================
# 训练函数
# =========================================================
def train():
    bootstrap_runtime()
    log_all_parameters()
    device = get_runtime_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = Config.ENABLE_CUDNN_BENCHMARK
    log_to_file(f"使用设备: {device}")
    log_to_file(f"检测到 GPU 数量: {torch.cuda.device_count()}")
    
    log_to_file("初始化教师网络（ConvTeacher）...")
    log_to_file(
        f"Acceleration active: cudnn_benchmark={device.type == 'cuda' and Config.ENABLE_CUDNN_BENCHMARK}, "
        f"channels_last={device.type == 'cuda' and Config.ENABLE_CHANNELS_LAST}"
    )
    teacher = ConvTeacher().to(device)
    teacher_loaded, teacher_message = load_teacher_checkpoint(teacher, Config.TEACHER_CHECKPOINT, device)
    log_to_file(teacher_message)
    if not teacher_loaded:
        raise FileNotFoundError(
            f"Unable to start optical training without a trained teacher checkpoint: {Config.TEACHER_CHECKPOINT}"
        )

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    teacher = wrap_data_parallel(teacher, module_name="ConvTeacher", use_channels_last=True)
    log_to_file("教师网络参数已冻结")

    response_detector = None
    if Config.USE_DETECTION_RESPONSE_LOSS:
        log_to_file("初始化冻结检测头（YOLOLightHead）用于 response map 约束...")
        response_detector = YOLOLightHead(
            in_channels=Config.YOLO_HEAD_IN_CHANNELS,
            out_channels=Config.get_detector_output_channels(),
        ).to(device)
        detector_loaded, detector_message = load_detector_checkpoint(response_detector, Config.DETECTOR_CHECKPOINT, device)
        log_to_file(detector_message)
        if detector_loaded:
            response_detector.eval()
            for p in response_detector.parameters():
                p.requires_grad = False
            response_detector = wrap_data_parallel(response_detector, module_name="ResponseDetector", use_channels_last=True)
            log_to_file("已启用检测响应图约束")
        else:
            response_detector = None
            log_to_file("检测头权重不可用，跳过 response map 约束")

    log_to_file("初始化学生网络（OpticalStudent）...")
    student = OpticalStudent().to(device)
    unwrap_module(student).enable_norm = Config.should_enable_student_norm(epoch=0)
    student = wrap_data_parallel(student, module_name="OpticalStudent", use_channels_last=False)

    log_to_file("初始化可训练检测头（YOLOLightHead）...")
    detector = YOLOLightHead(
        in_channels=Config.YOLO_HEAD_IN_CHANNELS,
        out_channels=Config.get_detector_output_channels(),
    ).to(device)
    trainable_detector_loaded, trainable_detector_message = load_detector_checkpoint(detector, Config.DETECTOR_CHECKPOINT, device)
    log_to_file(trainable_detector_message if trainable_detector_loaded else f"Trainable detector init fallback: {trainable_detector_message}")
    detector = wrap_data_parallel(detector, module_name="TrainableDetector", use_channels_last=True)
    
    log_to_file("加载数据集...")
    dataset = MilitaryFeatureDataset()
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        collate_fn=feature_dataset_collate_fn,
        **get_dataloader_kwargs(shuffle=True),
    )
    log_to_file(f"数据集大小: {len(dataset)}")

    feature_criterion = CompositeOpticalFeatureLoss()
    detection_criterion = DetectionAdaptationLoss(
        anchors=Config.ANCHORS,
        num_classes=Config.NUM_CLASSES,
        strides=Config.STRIDES,
    )
    
    vis_dir = Config.VISUALIZATION_DIR
    
    loss_curve = []
    best_phase1_feature_loss = float('inf')
    best_phase1_epoch = -1
    best_phase1_student_state = None
    best_phase2_detection_loss = float('inf')
    best_phase2_epoch = -1
    best_phase2_student_state = None
    best_phase2_detector_state = None
    best_joint_loss = float('inf')
    best_joint_epoch = -1
    best_student_state = None
    best_detector_state = None
    current_phase = None
    optimizer = None
    scheduler = None
    
    log_to_file("="*60)
    log_to_file("开始三阶段训练：学生 -> 检测头 -> 联合平衡 ...")
    log_to_file("="*60)

    def set_requires_grad(module, flag):
        for param in module.parameters():
            param.requires_grad = flag

    def configure_phase(phase_name):
        nonlocal optimizer, scheduler
        if phase_name == "student_only":
            set_requires_grad(student, True)
            set_requires_grad(detector, False)
            optimizer = torch.optim.Adam(student.parameters(), lr=Config.STUDENT_ONLY_LR, weight_decay=Config.WEIGHT_DECAY)
        elif phase_name == "detector_only":
            if Config.RESTORE_PHASE1_BEST_FOR_PHASE2 and best_phase1_student_state is not None:
                unwrap_module(student).load_state_dict(best_phase1_student_state, strict=False)
                log_to_file(
                    f"Phase switch -> detector_only: restored phase1 best student "
                    f"(epoch={best_phase1_epoch}, feature={best_phase1_feature_loss:.6f})"
                )
            set_requires_grad(student, False)
            set_requires_grad(detector, True)
            optimizer = torch.optim.Adam(detector.parameters(), lr=Config.DETECTOR_ONLY_LR, weight_decay=Config.WEIGHT_DECAY)
        else:
            if Config.RESTORE_PHASE2_BEST_FOR_PHASE3 and best_phase2_student_state is not None and best_phase2_detector_state is not None:
                unwrap_module(student).load_state_dict(best_phase2_student_state, strict=False)
                unwrap_module(detector).load_state_dict(best_phase2_detector_state, strict=False)
                log_to_file(
                    f"Phase switch -> joint_balance: restored phase2 best pair "
                    f"(epoch={best_phase2_epoch}, detection={best_phase2_detection_loss:.6f})"
                )
            set_requires_grad(student, True)
            set_requires_grad(detector, True)
            optimizer = torch.optim.Adam([
                {"params": student.parameters(), "lr": Config.JOINT_STUDENT_LR},
                {"params": detector.parameters(), "lr": Config.JOINT_DETECTOR_LR},
            ], weight_decay=Config.WEIGHT_DECAY)

        scheduler = None
        if Config.USE_LR_SCHEDULER:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=Config.LR_SCHEDULER_FACTOR,
                patience=Config.LR_SCHEDULER_PATIENCE,
                min_lr=Config.MIN_LEARNING_RATE,
            )
        log_to_file(f"切换训练阶段: {phase_name}")

    for epoch in range(Config.EPOCHS):
        phase_name = Config.get_training_phase(epoch)
        if phase_name != current_phase:
            current_phase = phase_name
            configure_phase(current_phase)

        desired_norm = Config.should_enable_student_norm(epoch)
        student_core = unwrap_module(student)
        detector_core = unwrap_module(detector)
        if student_core.enable_norm != desired_norm:
            student_core.enable_norm = desired_norm
            log_to_file(f"Epoch {epoch}: 学生输出归一化切换为 {student_core.enable_norm}")

        current_response_weight = Config.get_response_loss_weight(epoch)
        student.train(mode=current_phase != "detector_only")
        detector.train(mode=current_phase != "student_only")
        loss_sum = 0
        feature_loss_sum = 0
        response_loss_sum = 0
        response_raw_sum = 0
        detection_loss_sum = 0
        detection_box_sum = 0
        detection_obj_sum = 0
        detection_noobj_sum = 0
        detection_cls_sum = 0

        for x, rgb, targets in tqdm(loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [{current_phase}]", leave=True):
            x = prepare_tensor_for_device(x, device, channels_last=False)
            rgb = prepare_tensor_for_device(rgb, device, channels_last=True)
            targets = [target.to(device, non_blocking=Config.PIN_MEMORY) for target in targets]
            with torch.no_grad():
                t = teacher(rgb)
                t = F.interpolate(t, size=(Config.IMG_SIZE, Config.IMG_SIZE), mode="bilinear", align_corners=False)

            if current_phase == "student_only":
                y = student(x)
                feature_loss, _ = feature_criterion(y, t, student)
                response_loss, response_stats = compute_detection_response_loss(
                    response_detector,
                    y,
                    t,
                    response_weight=current_response_weight,
                )
                detection_loss = torch.zeros((), device=device)
                detection_stats = {"box": 0.0, "obj": 0.0, "noobj": 0.0, "cls": 0.0}
                loss = (
                    Config.FEATURE_LOSS_WEIGHT_PHASE1 * feature_loss +
                    Config.RESPONSE_LOSS_WEIGHT_PHASE1 * response_loss
                )
            elif current_phase == "detector_only":
                with torch.no_grad():
                    y = student(x)
                    feature_loss, _ = feature_criterion(y, t, student)
                    response_loss, response_stats = compute_detection_response_loss(
                        response_detector,
                        y,
                        t,
                        response_weight=current_response_weight,
                    )
                predictions = detector(prepare_conv_tensor(y.detach()))
                detection_loss, detection_stats = detection_criterion(predictions, targets)
                loss = Config.DETECTION_LOSS_WEIGHT_PHASE2 * detection_loss
            else:
                y = student(x)
                feature_loss, _ = feature_criterion(y, t, student)
                response_loss, response_stats = compute_detection_response_loss(
                    response_detector,
                    y,
                    t,
                    response_weight=current_response_weight,
                )
                predictions = detector(prepare_conv_tensor(y))
                detection_loss, detection_stats = detection_criterion(predictions, targets)
                loss = (
                    Config.FEATURE_LOSS_WEIGHT_PHASE3 * feature_loss +
                    Config.RESPONSE_LOSS_WEIGHT_PHASE3 * response_loss +
                    Config.DETECTION_LOSS_WEIGHT_PHASE3 * detection_loss
                )

            optimizer.zero_grad()
            loss.backward()
            if Config.GRAD_CLIP_NORM and Config.GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in list(student.parameters()) + list(detector.parameters()) if param.requires_grad],
                    max_norm=Config.GRAD_CLIP_NORM,
                )
            optimizer.step()
            loss_sum += loss.item()
            feature_loss_sum += feature_loss.item()
            response_loss_sum += response_loss.item()
            response_raw_sum += response_stats["response"]
            detection_loss_sum += detection_loss.item()
            detection_box_sum += detection_stats["box"]
            detection_obj_sum += detection_stats["obj"]
            detection_noobj_sum += detection_stats["noobj"]
            detection_cls_sum += detection_stats["cls"]

        avg_loss = loss_sum / len(loader)
        avg_feature_loss = feature_loss_sum / len(loader)
        avg_response_loss = response_loss_sum / len(loader)
        avg_response_raw = response_raw_sum / len(loader)
        avg_detection_loss = detection_loss_sum / len(loader)
        avg_detection_box = detection_box_sum / len(loader)
        avg_detection_obj = detection_obj_sum / len(loader)
        avg_detection_noobj = detection_noobj_sum / len(loader)
        avg_detection_cls = detection_cls_sum / len(loader)
        loss_curve.append(avg_loss)

        if current_phase == "student_only":
            monitor_value = avg_feature_loss
        elif current_phase == "detector_only":
            monitor_value = avg_detection_loss
        else:
            monitor_value = avg_loss

        if scheduler is not None:
            scheduler.step(monitor_value)

        if current_phase == "student_only" and avg_feature_loss < best_phase1_feature_loss:
            best_phase1_feature_loss = avg_feature_loss
            best_phase1_epoch = epoch
            best_phase1_student_state = copy.deepcopy(student_core.state_dict())
            torch.save(best_phase1_student_state, Config.get_phase1_student_best_model_path())
            log_to_file(
                f"Epoch {epoch}: 保存 phase1 最佳学生模型 "
                f"(Feature: {best_phase1_feature_loss:.6f})",
                also_print=False,
            )

        if current_phase == "detector_only" and avg_detection_loss < best_phase2_detection_loss:
            best_phase2_detection_loss = avg_detection_loss
            best_phase2_epoch = epoch
            best_phase2_student_state = copy.deepcopy(student_core.state_dict())
            best_phase2_detector_state = copy.deepcopy(detector_core.state_dict())
            torch.save(best_phase2_student_state, Config.get_phase2_student_best_model_path())
            torch.save(best_phase2_detector_state, Config.get_phase2_detector_best_model_path())
            log_to_file(
                f"Epoch {epoch}: 保存 phase2 最佳学生+检测头配对模型 "
                f"(Detection: {best_phase2_detection_loss:.6f})",
                also_print=False,
            )

        if current_phase in {"detector_only", "joint_balance"} and monitor_value < best_joint_loss:
            best_joint_loss = monitor_value
            best_joint_epoch = epoch
            best_student_state = copy.deepcopy(student_core.state_dict())
            best_detector_state = copy.deepcopy(detector_core.state_dict())
            torch.save(best_student_state, Config.get_best_model_path())
            torch.save(best_detector_state, Config.get_detector_best_model_path())
            log_to_file(
                f"Epoch {epoch}: 保存最佳学生+检测头配对模型 "
                f"(phase={current_phase}, monitor={best_joint_loss:.6f})",
                also_print=False,
            )

        if epoch % Config.VIS_INTERVAL == 0:
            student.eval()
            detector.eval()
            with torch.no_grad():
                vis_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=Config.VIS_BATCH_SIZE,
                    collate_fn=feature_dataset_collate_fn,
                    **get_dataloader_kwargs(shuffle=True),
                )
                vis_batch = next(iter(vis_loader))
                x_vis, rgb_vis, _ = vis_batch
                x_vis = prepare_tensor_for_device(x_vis, device, channels_last=False)
                rgb_vis = prepare_tensor_for_device(rgb_vis, device, channels_last=True)
                t_vis = teacher(rgb_vis)
                t_vis = F.interpolate(t_vis, size=(Config.IMG_SIZE, Config.IMG_SIZE), mode="bilinear", align_corners=False)
                y_vis_gpu = student(x_vis)
                y_vis = y_vis_gpu.cpu()
                detections_vis = decode_detections(detector(prepare_conv_tensor(y_vis_gpu)))
            save_feature_comparison(epoch, t_vis, y_vis, vis_dir, input_images=x_vis)
            save_detection_visualization(x_vis.cpu(), detections_vis, vis_dir, epoch, Config.CLASS_NAMES)
            save_phase_visualization(student, vis_dir, epoch)

        phase_stats = get_phase_statistics(student)

        lr_str = ','.join(f"{group['lr']:.6g}" for group in optimizer.param_groups)
        log_to_file(
            f"Epoch {epoch:3d} [{current_phase}] | Loss: {avg_loss:.6f} | "
            f"Feature: {avg_feature_loss:.6f} | "
            f"ResponseRaw: {avg_response_raw:.6f} | "
            f"ResponseWeighted: {avg_response_loss:.6f} | "
            f"Detection: {avg_detection_loss:.6f} | "
            f"DetBox/Obj/NoObj/Cls: {avg_detection_box:.4f}/{avg_detection_obj:.4f}/{avg_detection_noobj:.4f}/{avg_detection_cls:.4f} | "
            f"PhaseSpan(slm1/slm2): {phase_stats['slm1_raw_span']:.4f}/{phase_stats['slm2_raw_span']:.4f} | "
            f"PhaseStd(slm1/slm2): {phase_stats['slm1_eff_std']:.4f}/{phase_stats['slm2_eff_std']:.4f} | "
            f"RespW: {current_response_weight:.6f} | "
            f"LR: {lr_str}"
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

    last_student_path = Config.get_last_model_path()
    last_detector_path = Config.get_detector_last_model_path()
    torch.save(unwrap_module(student).state_dict(), last_student_path)
    torch.save(unwrap_module(detector).state_dict(), last_detector_path)

    if Config.FINAL_MODEL_SOURCE == "best_joint" and best_student_state is not None and best_detector_state is not None:
        final_student_state = best_student_state
        final_detector_state = best_detector_state
    else:
        final_student_state = unwrap_module(student).state_dict()
        final_detector_state = unwrap_module(detector).state_dict()

    final_student_path = Config.get_final_model_path()
    final_detector_path = Config.get_detector_final_model_path()
    torch.save(final_student_state, final_student_path)
    torch.save(final_detector_state, final_detector_path)
    
    log_to_file("="*60)
    log_to_file("训练完成！")
    log_to_file(f"最佳模型已保存到: {Config.get_best_model_path()}")
    log_to_file(f"最佳检测头已保存到: {Config.get_detector_best_model_path()}")
    log_to_file(f"Phase1 最佳学生模型: {Config.get_phase1_student_best_model_path()} (epoch={best_phase1_epoch})")
    log_to_file(
        f"Phase2 最佳学生/检测头模型: "
        f"{Config.get_phase2_student_best_model_path()} / {Config.get_phase2_detector_best_model_path()} "
        f"(epoch={best_phase2_epoch})"
    )
    log_to_file(f"最后一轮学生/检测头模型: {last_student_path} / {last_detector_path}")
    log_to_file(f"最终学生/检测头模型: {final_student_path} / {final_detector_path}")
    log_to_file(f"最佳联合轮数: {best_joint_epoch}")
    log_to_file(f"最佳联合监控损失: {best_joint_loss:.6f}")
    log_to_file(f"所有结果已保存到: {Config.TEACHER_OUTPUT_DIR}")
    log_to_file("="*60)

if __name__ == "__main__":
    train()
