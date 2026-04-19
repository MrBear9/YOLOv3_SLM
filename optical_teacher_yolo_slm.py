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
import matplotlib.patches as patches

# =========================================================
# 配置类 - 集中管理所有参数
# =========================================================
class ConfigSLMYOLO:
    """光学教师YOLO SLM训练配置类"""
    
    # 数据集参数
    YAML_PATH = r"data\military\data.yaml"
    CLASS_NAMES = None
    NUM_CLASSES = None
    
    # 输出路径配置
    OUTPUT_DIR = r"output\OpticalTeacherYOLO_SLM"
    TEACHER_CHECKPOINT = r"output\OpticalTeacherYOLO\teacher_best.pth"
    LOG_ROOT_DIR = None
    LOG_FILE = None
    TIMESTAMP = None
    TRAIN_START_TIME = None
    VISUALIZATION_DIR = None
    LOSS_CURVE_DIR = None
    IS_INITIALIZED = False
    LOG_INITIALIZED = False
    
    # 训练参数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 640
    BATCH_SIZE = 8
    EPOCHS = 100  # 总轮数：30+20+50
    
    # 阶段划分参数
    PHASE1_EPOCHS = 30  # 阶段1：教师网络对光学层的学习训练
    PHASE2_EPOCHS = 20  # 阶段2：教师网络对光学层的约束 + 检测头训练
    PHASE3_EPOCHS = 50  # 阶段3：光学层 + 检测头的训练
    
    # 损失权重
    BOX_WEIGHT = 0.8
    OBJ_WEIGHT = 0.8
    NOOBJ_WEIGHT = 0.2
    CLS_WEIGHT = 0.4
    LOSS_FULL_WEIGHT = 0.02
    LOSS_LOW1_WEIGHT = 1.0
    LOSS_LOW2_WEIGHT = 0.5
    TEACHER_STUDENT_WEIGHT = 0.5
    YOLO_LOSS_WEIGHT = 0.5
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
    OPTICAL_LEARNING_RATE = 1e-3      # 光学部分学习率（较高，因为光学层需要快速收敛）
    DETECTOR_LEARNING_RATE = 5e-4    # 检测头学习率（适中）
    OPTICAL_WEIGHT_DECAY = 1e-5      # 光学部分权重衰减
    DETECTOR_WEIGHT_DECAY = 1e-5     # 检测头权重衰减
    OPTIMIZER = "Adam"
    
    # 光学传播参数
    WAVELENGTH = 532e-9
    PIXEL_SIZE = 6.4e-6
    PROP_DISTANCE_1 = 0.01
    PROP_DISTANCE_2 = 0.02
    SLM_MODE = "phase"
    RESOLUTION = (640, 640)
    OPTICAL_FIELD_EPS = 1e-8
    OPTICAL_NORM_EPS = 1e-6
    DISPLAY_EPS = 1e-6
    IOU_EPS = 1e-6
    YOLO_HEAD_IN_CHANNELS = 1
    YOLO_HEAD_BASE_CHANNELS = 32
    AGNOSTIC_NMS = True
    
    # 训练控制参数
    VIS_INTERVAL = 5
    SAVE_INTERVAL = 10
    VAL_INTERVAL = 5  # 验证间隔，每5轮验证一次
    METRIC_IOU_THRESHOLD = 0.5
    
    # 检测参数
    CONF_THRESH = 0.5
    NMS_THRESH = 0.4
    MAX_DET = 8
    
    # 可视化参数
    VIS_BATCH_SIZE = 4
    VIS_DPI = 120
    EPOCH_TABLE_EPOCH_WIDTH = 8
    EPOCH_TABLE_PHASE_WIDTH = 18
    EPOCH_TABLE_TRAIN_LOSS_WIDTH = 13
    EPOCH_TABLE_VAL_LOSS_WIDTH = 13
    EPOCH_TABLE_METRIC_WIDTH = 11
    EPOCH_TABLE_LR_WIDTH = 12
    EPOCH_TABLE_BEST_WIDTH = 8
    EPOCH_TABLE_BEST_MARK = "Yes"
    
    @classmethod
    def initialize(cls):
        """初始化配置"""
        if cls.IS_INITIALIZED and cls.LOG_FILE is not None:
            return
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
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        cls.LOG_ROOT_DIR = os.path.join(cls.OUTPUT_DIR, "logs")
        cls.VISUALIZATION_DIR = os.path.join(cls.OUTPUT_DIR, "visualizations")
        cls.LOSS_CURVE_DIR = os.path.join(cls.OUTPUT_DIR, "loss_curves")
        os.makedirs(cls.LOG_ROOT_DIR, exist_ok=True)
        os.makedirs(cls.VISUALIZATION_DIR, exist_ok=True)
        os.makedirs(cls.LOSS_CURVE_DIR, exist_ok=True)
        cls.TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.LOG_FILE = os.path.join(cls.LOG_ROOT_DIR, f"training_log_{cls.TIMESTAMP}.txt")
        cls.IS_INITIALIZED = True
        cls.LOG_INITIALIZED = False
    
    @classmethod
    def get_detector_output_channels(cls):
        """获取检测头输出通道数"""
        return 3 * (4 + 1 + cls.NUM_CLASSES)

    @classmethod
    def get_optical_yolo_checkpoint_path(cls, epoch):
        return os.path.join(cls.OUTPUT_DIR, f"optical_yolo_epoch_{epoch}.pth")

    @classmethod
    def get_teacher_optical_checkpoint_path(cls, epoch):
        return os.path.join(cls.OUTPUT_DIR, f"teacher_optical_epoch_{epoch}.pth")

    @classmethod
    def get_best_optical_yolo_path(cls):
        return os.path.join(cls.OUTPUT_DIR, "optical_yolo_best.pth")

    @classmethod
    def get_best_teacher_optical_path(cls):
        return os.path.join(cls.OUTPUT_DIR, "teacher_optical_best.pth")

    @classmethod
    def get_training_curve_path(cls):
        return os.path.join(cls.OUTPUT_DIR, "loss_curve.png")

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
    def get_epoch_table_separator(cls):
        return "-" * sum(width for _, width in cls.get_epoch_table_columns())

    @classmethod
    def get_epoch_table_header(cls):
        return "".join(f"{title:<{width}}" for title, width in cls.get_epoch_table_columns())
    
    @classmethod
    def get_current_phase(cls, epoch):
        """获取当前训练阶段"""
        if epoch < cls.PHASE1_EPOCHS:
            return "phase1", "教师网络对光学层学习"
        elif epoch < cls.PHASE1_EPOCHS + cls.PHASE2_EPOCHS:
            return "phase2", "教师约束+检测头训练"
        else:
            return "phase3", "光学层+检测头训练"
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("="*80)
        print("光学教师YOLO SLM训练配置")
        print("="*80)
        print(f"设备: {cls.DEVICE}")
        print(f"图像尺寸: {cls.IMG_SIZE}")
        print(f"批次大小: {cls.BATCH_SIZE}")
        print(f"总轮数: {cls.EPOCHS}")
        print(f"阶段1轮数: {cls.PHASE1_EPOCHS}")
        print(f"阶段2轮数: {cls.PHASE2_EPOCHS}")
        print(f"阶段3轮数: {cls.PHASE3_EPOCHS}")
        print(f"类别数量: {cls.NUM_CLASSES}")
        print(f"使用外部锚框: {cls.USE_EXTERNAL_ANCHORS}")
        print(f"锚框来源: {cls.ANCHOR_SOURCE}")
        print(f"光学部分学习率: {cls.OPTICAL_LEARNING_RATE}")
        print(f"检测头学习率: {cls.DETECTOR_LEARNING_RATE}")
        print(f"光学部分权重衰减: {cls.OPTICAL_WEIGHT_DECAY}")
        print(f"检测头权重衰减: {cls.DETECTOR_WEIGHT_DECAY}")
        print("="*80)

# =========================================================
# 辅助函数
# =========================================================
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

def init_log_file():
    """初始化日志文件"""
    if ConfigSLMYOLO.LOG_INITIALIZED and ConfigSLMYOLO.LOG_FILE and os.path.exists(ConfigSLMYOLO.LOG_FILE):
        return
    os.makedirs(ConfigSLMYOLO.LOG_ROOT_DIR, exist_ok=True)
    ConfigSLMYOLO.TRAIN_START_TIME = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(ConfigSLMYOLO.LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("光学教师YOLO SLM训练日志\n")
        f.write("=" * 80 + "\n")
        f.write(f"训练开始时间: {ConfigSLMYOLO.TRAIN_START_TIME}\n")
        f.write("=" * 80 + "\n\n")
    ConfigSLMYOLO.LOG_INITIALIZED = True

def log_to_file(message, also_print=True):
    """记录消息到文件"""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_message = f"{timestamp} {message}"
    
    with open(ConfigSLMYOLO.LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_message + "\n")
    
    if also_print:
        print(log_message)

def append_plain_log(message=""):
    with open(ConfigSLMYOLO.LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + "\n")

def init_epoch_log_table():
    separator = ConfigSLMYOLO.get_epoch_table_separator()
    append_plain_log("")
    append_plain_log(separator)
    append_plain_log(ConfigSLMYOLO.get_epoch_table_header())
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
    phase_text = str(phase)[: ConfigSLMYOLO.EPOCH_TABLE_PHASE_WIDTH - 1]
    append_plain_log(
        f"{epoch + 1:<{ConfigSLMYOLO.EPOCH_TABLE_EPOCH_WIDTH}}"
        f"{phase_text:<{ConfigSLMYOLO.EPOCH_TABLE_PHASE_WIDTH}}"
        f"{_format_table_value(train_loss, ConfigSLMYOLO.EPOCH_TABLE_TRAIN_LOSS_WIDTH)}"
        f"{_format_table_value(val_loss, ConfigSLMYOLO.EPOCH_TABLE_VAL_LOSS_WIDTH)}"
        f"{_format_table_value(precision, ConfigSLMYOLO.EPOCH_TABLE_METRIC_WIDTH, 3)}"
        f"{_format_table_value(recall, ConfigSLMYOLO.EPOCH_TABLE_METRIC_WIDTH, 3)}"
        f"{_format_table_value(f1_score, ConfigSLMYOLO.EPOCH_TABLE_METRIC_WIDTH, 3)}"
        f"{_format_table_value(map50, ConfigSLMYOLO.EPOCH_TABLE_METRIC_WIDTH, 3)}"
        f"{_format_table_value(lr, ConfigSLMYOLO.EPOCH_TABLE_LR_WIDTH, 6)}"
        f"{str(best_status):<{ConfigSLMYOLO.EPOCH_TABLE_BEST_WIDTH}}"
    )

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

def apply_classwise_nms(detections, nms_thresh, max_det):
    return apply_nms(detections, nms_thresh, max_det, class_agnostic=False)
    
def apply_nms(detections, nms_thresh, max_det, class_agnostic=None):
    if len(detections) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    det_tensor = torch.as_tensor(detections, dtype=torch.float32)
    boxes_xyxy = xywh_to_xyxy(det_tensor[:, :4])
    scores = det_tensor[:, 4]
    class_ids = det_tensor[:, 5]
    if class_agnostic is None:
        class_agnostic = ConfigSLMYOLO.AGNOSTIC_NMS

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

def enhance_feature_for_display(feature_map):
    """增强特征图显示效果"""
    feature_map = np.asarray(feature_map, dtype=np.float32)
    low = np.percentile(feature_map, 2)
    high = np.percentile(feature_map, 98)
    if high - low < ConfigSLMYOLO.DISPLAY_EPS:
        return np.zeros_like(feature_map)
    
    feature_map = np.clip((feature_map - low) / (high - low), 0.0, 1.0)
    return np.power(feature_map, 0.8)

# =========================================================
# 光学层组件（从optical_teacher.py导入）
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
            wavelength = ConfigSLMYOLO.WAVELENGTH
        if pixel_size is None:
            pixel_size = ConfigSLMYOLO.PIXEL_SIZE
        if resolution is None:
            resolution = ConfigSLMYOLO.RESOLUTION
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
# 教师网络（从optical_teacher.py导入）
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

        f = self.conv1(x)
        f = self.conv2(f)
        f = self.conv3(f)
        f = self.refine(f)
        f = self.project(f)
        f = torch.abs(f)
        f = F.interpolate(
            f, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.sigmoid(f)

# =========================================================
# 光学学生网络（多层光学传播）
# =========================================================
class OpticalStudent(nn.Module):
    def __init__(self, resolution=None, mode=None):
        super().__init__()
        if resolution is None:
            resolution = ConfigSLMYOLO.RESOLUTION
        if mode is None:
            mode = ConfigSLMYOLO.SLM_MODE
        self.slm1 = SLMLayer(resolution, mode)
        self.prop1 = ASMPropagation(ConfigSLMYOLO.PROP_DISTANCE_1)
        self.slm2 = SLMLayer(resolution, mode)
        self.prop2 = ASMPropagation(ConfigSLMYOLO.PROP_DISTANCE_2)
        self.enable_norm = False

    def forward(self, intensity):
        amp = torch.sqrt(intensity.clamp(min=0) + ConfigSLMYOLO.OPTICAL_FIELD_EPS)
        field = torch.complex(amp, torch.zeros_like(amp))
        field = self.prop1(self.slm1(field))
        field = self.prop2(self.slm2(field))
        out = torch.abs(field)**2
        if self.enable_norm:
            out = out / (out.mean(dim=[2,3], keepdim=True) + ConfigSLMYOLO.OPTICAL_NORM_EPS)
        return out

# =========================================================
# 轻量化卷积块和检测头（从optical_teacher.py导入）
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

class YOLOLightHead(nn.Module):
    def __init__(self, in_channels=1, out_channels=27):
        super(YOLOLightHead, self).__init__()
        base_ch = ConfigSLMYOLO.YOLO_HEAD_BASE_CHANNELS

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
# 数据集类
# =========================================================
class YOLODataset(Dataset):
    def __init__(self, yaml_path=None, split="train"):
        if yaml_path is None:
            yaml_path = ConfigSLMYOLO.YAML_PATH
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        # 获取数据集根目录
        data_root = cfg.get("path", "")
        if not os.path.exists(data_root):
            # 如果路径不存在，尝试使用YAML文件所在目录
            data_root = os.path.dirname(yaml_path)
        
        img_dir = os.path.join(data_root, f"{split}/images")
        label_dir = os.path.join(data_root, f"{split}/labels")
        
        self.files = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith(".jpg") or f.endswith(".png")
        ])
        
        self.label_dir = label_dir
        self.img_size = ConfigSLMYOLO.IMG_SIZE
        self.num_classes = ConfigSLMYOLO.NUM_CLASSES
        
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Grayscale(1),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        
        img_tensor = self.transform(img)
        
        label_path = os.path.join(self.label_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        
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

# =========================================================
# 损失函数
# =========================================================
class MultiScaleMSELoss(nn.Module):
    """多尺度MSE损失函数（阶段1使用）"""
    def __init__(self):
        super().__init__()
        self.pool1 = nn.AvgPool2d(8)
        self.pool2 = nn.AvgPool2d(32)
        self.loss_full_weight = ConfigSLMYOLO.LOSS_FULL_WEIGHT
        self.loss_low1_weight = ConfigSLMYOLO.LOSS_LOW1_WEIGHT
        self.loss_low2_weight = ConfigSLMYOLO.LOSS_LOW2_WEIGHT

    def forward(self, student_output, teacher_output):
        loss_full = F.mse_loss(student_output, teacher_output)
        loss_low1 = F.mse_loss(self.pool1(student_output), self.pool1(teacher_output))
        loss_low2 = F.mse_loss(self.pool2(student_output), self.pool2(teacher_output))
        return (loss_full * self.loss_full_weight + 
                loss_low1 * self.loss_low1_weight + 
                loss_low2 * self.loss_low2_weight)

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
        focal_term = (1.0 - p_t).pow(self.gamma)
        loss = alpha_t * focal_term * loss

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
        self.box_weight = ConfigSLMYOLO.BOX_WEIGHT
        self.obj_weight = ConfigSLMYOLO.OBJ_WEIGHT
        self.noobj_weight = ConfigSLMYOLO.NOOBJ_WEIGHT
        self.cls_weight = ConfigSLMYOLO.CLS_WEIGHT
        self.focal_loss = SigmoidFocalLoss(
            alpha=ConfigSLMYOLO.FOCAL_ALPHA,
            gamma=ConfigSLMYOLO.FOCAL_GAMMA,
            reduction="mean",
        )
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
                "target_boxes_abs": torch.zeros_like(pred[..., :4]),
                "target_obj": torch.zeros_like(pred[..., 4]),
                "target_cls": torch.zeros_like(pred[..., 5:]),
                "grid_h": grid_h,
                "grid_w": grid_w,
                "stride": self.strides[i],
                "anchors": self.anchors[i].to(device),
            })

        for b in range(batch_size):
            if len(targets[b]) == 0:
                continue

            current_targets = targets[b].to(device)
            if current_targets.dim() == 1:
                current_targets = current_targets.unsqueeze(0)

            for target in current_targets:
                cls_id = int(target[0].item())
                tx = target[1]
                ty = target[2]
                tw = target[3] * ConfigSLMYOLO.IMG_SIZE
                th = target[4] * ConfigSLMYOLO.IMG_SIZE

                best_scale_idx = 0
                best_anchor_idx = 0
                best_iou = -1.0

                for scale_idx, scale_data in enumerate(prepared_scales):
                    for anchor_idx in range(3):
                        anchor_w, anchor_h = scale_data["anchors"][anchor_idx]
                        inter = torch.minimum(tw, anchor_w) * torch.minimum(th, anchor_h)
                        union = tw * th + anchor_w * anchor_h - inter + ConfigSLMYOLO.IOU_EPS
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
                scale_data["target_boxes_abs"][b, grid_y, grid_x, best_anchor_idx, 0] = tx * ConfigSLMYOLO.IMG_SIZE
                scale_data["target_boxes_abs"][b, grid_y, grid_x, best_anchor_idx, 1] = ty * ConfigSLMYOLO.IMG_SIZE
                scale_data["target_boxes_abs"][b, grid_y, grid_x, best_anchor_idx, 2] = tw
                scale_data["target_boxes_abs"][b, grid_y, grid_x, best_anchor_idx, 3] = th
                scale_data["target_obj"][b, grid_y, grid_x, best_anchor_idx] = 1.0
                scale_data["target_cls"][b, grid_y, grid_x, best_anchor_idx, cls_id] = 1.0

        for scale_data in prepared_scales:
            pred_boxes = scale_data["pred_boxes"]
            pred_obj = scale_data["pred_obj"]
            pred_cls = scale_data["pred_cls"]
            target_obj = scale_data["target_obj"]
            target_cls = scale_data["target_cls"]
            target_boxes_abs = scale_data["target_boxes_abs"]
            pred_boxes_abs = decode_boxes_to_absolute(pred_boxes, scale_data["anchors"], scale_data["stride"])

            obj_mask = target_obj > 0.5
            noobj_mask = target_obj <= 0.5

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

class CombinedLoss(nn.Module):
    """组合损失函数（阶段2使用）"""
    def __init__(self, anchors, num_classes, strides):
        super().__init__()
        self.teacher_student_loss = MultiScaleMSELoss()
        self.yolo_loss = EnhancedYOLOLoss(anchors, num_classes, strides)
        self.teacher_student_weight = ConfigSLMYOLO.TEACHER_STUDENT_WEIGHT
        self.yolo_weight = ConfigSLMYOLO.YOLO_LOSS_WEIGHT

    def forward(self, teacher_output, optical_output, predictions, targets):
        teacher_student_loss = self.teacher_student_loss(optical_output, teacher_output)
        yolo_loss, yolo_stats = self.yolo_loss(predictions, targets)
        total_loss = (
            self.teacher_student_weight * teacher_student_loss +
            self.yolo_weight * yolo_loss
        )
        stats = dict(yolo_stats)
        stats["total"] = float(total_loss.detach().item())
        stats["teacher"] = float(teacher_student_loss.detach().item())
        return total_loss, stats

YOLOLoss = EnhancedYOLOLoss

# =========================================================
# 模型组合类
# =========================================================
class OpticalYOLOModel(nn.Module):
    """光学YOLO检测模型（光学层 + 检测头）"""
    def __init__(self, optical_student=None, detector=None):
        super().__init__()
        if optical_student is None:
            self.optical_student = OpticalStudent()
        else:
            self.optical_student = optical_student
        
        if detector is None:
            self.detector = YOLOLightHead(in_channels=ConfigSLMYOLO.YOLO_HEAD_IN_CHANNELS, 
                                         out_channels=ConfigSLMYOLO.get_detector_output_channels())
        else:
            self.detector = detector

    def forward(self, x):
        optical_features = self.optical_student(x)
        detections = self.detector(optical_features)
        return detections

class TeacherOpticalModel(nn.Module):
    """教师-光学模型（教师网络 + 光学层）"""
    def __init__(self, teacher=None, optical_student=None):
        super().__init__()
        if teacher is None:
            self.teacher = ConvTeacher()
        else:
            self.teacher = teacher
        
        if optical_student is None:
            self.optical_student = OpticalStudent()
        else:
            self.optical_student = optical_student

    def forward(self, x):
        with torch.no_grad():
            teacher_features = self.teacher(x)
        optical_features = self.optical_student(x)
        return teacher_features, optical_features

# =========================================================
# 解码检测结果函数
# =========================================================
def decode_detections(preds, conf_thresh=None, nms_thresh=None, max_det=None, img_size=None):
    """解码YOLO检测结果"""
    if conf_thresh is None:
        conf_thresh = ConfigSLMYOLO.CONF_THRESH
    if nms_thresh is None:
        nms_thresh = ConfigSLMYOLO.NMS_THRESH
    if max_det is None:
        max_det = ConfigSLMYOLO.MAX_DET
    if img_size is None:
        img_size = ConfigSLMYOLO.IMG_SIZE
    
    batch_size = preds[0].shape[0]
    detections = [[] for _ in range(batch_size)]
    strides = ConfigSLMYOLO.STRIDES
    
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
        anchor_tensor = torch.tensor(ConfigSLMYOLO.ANCHORS[i], device=device, dtype=dtype).view(1, 1, 1, 3, 2)

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
    
    for b in range(batch_size):
        if len(detections[b]) > 0:
            detections[b] = apply_nms(torch.cat(detections[b], dim=0), nms_thresh, max_det)
        else:
            detections[b] = np.zeros((0, 6), dtype=np.float32)

    return detections

# =========================================================
# 可视化函数
# =========================================================
def save_phase1_visualization(epoch, input_images, teacher_features, optical_features, save_dir, prefix="train"):
    """阶段1可视化：类似optical_teacher.py的可视化"""
    os.makedirs(save_dir, exist_ok=True)
    
    num_samples = min(ConfigSLMYOLO.VIS_BATCH_SIZE, len(input_images))
    indices = torch.randperm(len(input_images))[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # 第一列：输入图像
            img_tensor = input_images[idx].cpu()
            if img_tensor.dim() == 4:
                img_tensor = img_tensor.squeeze(0)
            
            if img_tensor.shape[0] == 1:
                img_np = img_tensor.squeeze(0).numpy()
                img_np = np.stack([img_np] * 3, axis=-1)
            else:
                img_np = img_tensor.numpy().transpose(1, 2, 0)
            
            img_np = (img_np * 255).astype(np.uint8)
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"Input {i+1}")
            axes[i, 0].axis("off")
            
            # 第二列：教师特征
            teacher_np = teacher_features[idx].squeeze(0).cpu().numpy()
            teacher_np = enhance_feature_for_display(teacher_np)
            axes[i, 1].imshow(teacher_np, cmap="hot")
            axes[i, 1].set_title(f"Teacher Feature {i+1}")
            axes[i, 1].axis("off")
            
            # 第三列：光学特征
            optical_np = optical_features[idx].squeeze(0).cpu().numpy()
            optical_np = enhance_feature_for_display(optical_np)
            axes[i, 2].imshow(optical_np, cmap="hot")
            axes[i, 2].set_title(f"Optical Feature {i+1}")
            axes[i, 2].axis("off")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"phase1_{prefix}_epoch_{epoch:03d}.png")
    plt.savefig(save_path, dpi=ConfigSLMYOLO.VIS_DPI)
    plt.close()

def save_phase2_visualization(epoch, input_images, teacher_features, optical_features, 
                             ground_truth, predictions, save_dir, prefix="train"):
    """阶段2可视化：输入图像+真实框，光学特征，教师约束，预测结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    num_samples = min(ConfigSLMYOLO.VIS_BATCH_SIZE, len(input_images))
    indices = torch.randperm(len(input_images))[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(24, 6 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # 第一列：输入图像 + 真实框
            img_tensor = input_images[idx].cpu()
            if img_tensor.dim() == 4:
                img_tensor = img_tensor.squeeze(0)
            
            if img_tensor.shape[0] == 1:
                img_np = img_tensor.squeeze(0).numpy()
                img_np = np.stack([img_np] * 3, axis=-1)
            else:
                img_np = img_tensor.numpy().transpose(1, 2, 0)
            
            img_np = (img_np * 255).astype(np.uint8)
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"Input + GT {i+1}")
            axes[i, 0].axis("off")
            
            # 绘制真实边界框
            for target_idx in range(len(ground_truth[idx])):
                cls_id, x_center, y_center, width, height = ground_truth[idx][target_idx]
                cls_id = int(cls_id.item())
                
                x1 = int((x_center - width / 2) * ConfigSLMYOLO.IMG_SIZE)
                y1 = int((y_center - height / 2) * ConfigSLMYOLO.IMG_SIZE)
                w = int(width * ConfigSLMYOLO.IMG_SIZE)
                h = int(height * ConfigSLMYOLO.IMG_SIZE)
                
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                        edgecolor='green', facecolor='none')
                axes[i, 0].add_patch(rect)
                axes[i, 0].text(x1, y1 - 5, ConfigSLMYOLO.CLASS_NAMES[cls_id], 
                              color='green', fontsize=10, fontweight='bold')
            
            # 第二列：光学特征
            optical_np = optical_features[idx].squeeze(0).cpu().numpy()
            optical_np = enhance_feature_for_display(optical_np)
            axes[i, 1].imshow(optical_np, cmap="hot")
            axes[i, 1].set_title(f"Optical Feature {i+1}")
            axes[i, 1].axis("off")
            
            # 第三列：约束图（教师特征）
            teacher_np = teacher_features[idx].squeeze(0).cpu().numpy()
            teacher_np = enhance_feature_for_display(teacher_np)
            axes[i, 2].imshow(teacher_np, cmap="hot")
            axes[i, 2].set_title(f"Teacher Constraint {i+1}")
            axes[i, 2].axis("off")
            
            # 第四列：预测结果（置信度分类框）
            axes[i, 3].imshow(img_np)
            axes[i, 3].set_title(f"Predictions {i+1}")
            axes[i, 3].axis("off")
            
            # 绘制预测边界框
            if len(predictions[idx]) > 0:
                for det in predictions[idx]:
                    x_center, y_center, w, h, conf, cls_id = det
                    cls_id = int(cls_id)
                    
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    w = int(w)
                    h = int(h)
                    
                    color = plt.cm.tab20(cls_id / max(ConfigSLMYOLO.NUM_CLASSES, 1))
                    rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                            edgecolor=color, facecolor='none')
                    axes[i, 3].add_patch(rect)
                    
                    label = f"{ConfigSLMYOLO.CLASS_NAMES[cls_id]}: {conf:.2f}"
                    axes[i, 3].text(x1, y1 - 5, label, 
                                  color=color, fontsize=10, fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.3', 
                                          facecolor=color, alpha=0.7))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"phase2_{prefix}_epoch_{epoch:03d}.png")
    plt.savefig(save_path, dpi=ConfigSLMYOLO.VIS_DPI)
    plt.close()

def save_phase3_visualization(epoch, input_images, optical_features, ground_truth, 
                             predictions, save_dir, prefix="train"):
    """阶段3可视化：类似optical_teacher_yolo.py的可视化"""
    os.makedirs(save_dir, exist_ok=True)
    
    num_samples = min(ConfigSLMYOLO.VIS_BATCH_SIZE, len(input_images))
    indices = torch.randperm(len(input_images))[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # 第一列：输入图像 + 真实框
            img_tensor = input_images[idx].cpu()
            if img_tensor.dim() == 4:
                img_tensor = img_tensor.squeeze(0)
            
            if img_tensor.shape[0] == 1:
                img_np = img_tensor.squeeze(0).numpy()
                img_np = np.stack([img_np] * 3, axis=-1)
            else:
                img_np = img_tensor.numpy().transpose(1, 2, 0)
            
            img_np = (img_np * 255).astype(np.uint8)
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"Input + GT {i+1}")
            axes[i, 0].axis("off")
            
            # 绘制真实边界框
            for target_idx in range(len(ground_truth[idx])):
                cls_id, x_center, y_center, width, height = ground_truth[idx][target_idx]
                cls_id = int(cls_id.item())
                
                x1 = int((x_center - width / 2) * ConfigSLMYOLO.IMG_SIZE)
                y1 = int((y_center - height / 2) * ConfigSLMYOLO.IMG_SIZE)
                w = int(width * ConfigSLMYOLO.IMG_SIZE)
                h = int(height * ConfigSLMYOLO.IMG_SIZE)
                
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                        edgecolor='green', facecolor='none')
                axes[i, 0].add_patch(rect)
                axes[i, 0].text(x1, y1 - 5, ConfigSLMYOLO.CLASS_NAMES[cls_id], 
                              color='green', fontsize=10, fontweight='bold')
            
            # 第二列：光学特征
            optical_np = optical_features[idx].squeeze(0).cpu().numpy()
            optical_np = enhance_feature_for_display(optical_np)
            axes[i, 1].imshow(optical_np, cmap="hot")
            axes[i, 1].set_title(f"Optical Feature {i+1}")
            axes[i, 1].axis("off")
            
            # 第三列：预测结果（置信度分类框）
            axes[i, 2].imshow(img_np)
            axes[i, 2].set_title(f"Predictions {i+1}")
            axes[i, 2].axis("off")
            
            # 绘制预测边界框
            if len(predictions[idx]) > 0:
                for det in predictions[idx]:
                    x_center, y_center, w, h, conf, cls_id = det
                    cls_id = int(cls_id)
                    
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    w = int(w)
                    h = int(h)
                    
                    color = plt.cm.tab20(cls_id / max(ConfigSLMYOLO.NUM_CLASSES, 1))
                    rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                            edgecolor=color, facecolor='none')
                    axes[i, 2].add_patch(rect)
                    
                    label = f"{ConfigSLMYOLO.CLASS_NAMES[cls_id]}: {conf:.2f}"
                    axes[i, 2].text(x1, y1 - 5, label, 
                                  color=color, fontsize=10, fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.3', 
                                          facecolor=color, alpha=0.7))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"phase3_{prefix}_epoch_{epoch:03d}.png")
    plt.savefig(save_path, dpi=ConfigSLMYOLO.VIS_DPI)
    plt.close()

# =========================================================
# 验证函数
# =========================================================
def prepare_batch(batch, device):
    images, targets = batch
    return images.to(device), [target.to(device) for target in targets]

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

def validate(phase, teacher_optical_model, optical_yolo_model, val_loader,
            phase1_loss, phase2_loss, phase3_loss, device):
    """验证函数，计算验证集损失与检测指标"""
    teacher_optical_model.eval()
    optical_yolo_model.eval()

    if phase == "phase1":
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, _ = prepare_batch(batch, device)
                teacher_features, optical_features = teacher_optical_model(images)
                loss = phase1_loss(optical_features, teacher_features)
                total_loss += float(loss.item())

        avg_total = total_loss / max(len(val_loader), 1)
        return {
            "total": avg_total,
            "box": 0.0,
            "obj": 0.0,
            "noobj": 0.0,
            "cls": 0.0,
        }, None

    metric_storage = {cls_id: [] for cls_id in range(ConfigSLMYOLO.NUM_CLASSES)}
    gt_counts = {cls_id: 0 for cls_id in range(ConfigSLMYOLO.NUM_CLASSES)}
    component_totals = {"total": 0.0, "box": 0.0, "obj": 0.0, "noobj": 0.0, "cls": 0.0}
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for batch in val_loader:
            images, targets = prepare_batch(batch, device)

            if phase == "phase2":
                teacher_features, optical_features = teacher_optical_model(images)
                predictions = optical_yolo_model.detector(optical_features)
                _, loss_stats = phase2_loss(teacher_features, optical_features, predictions, targets)
            else:
                predictions = optical_yolo_model(images)
                _, loss_stats = phase3_loss(predictions, targets)

            for key in component_totals:
                component_totals[key] += float(loss_stats.get(key, 0.0))

            detections = decode_detections(predictions)
            for sample_idx, sample_detections in enumerate(detections):
                gt_by_class = {}
                for gt in targets[sample_idx]:
                    if gt.shape[0] < 5 or gt[3] <= 0 or gt[4] <= 0:
                        continue

                    cls_id = int(gt[0].item())
                    gt_box = [
                        float(gt[1].item() * ConfigSLMYOLO.IMG_SIZE),
                        float(gt[2].item() * ConfigSLMYOLO.IMG_SIZE),
                        float(gt[3].item() * ConfigSLMYOLO.IMG_SIZE),
                        float(gt[4].item() * ConfigSLMYOLO.IMG_SIZE),
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

                    is_tp = best_iou >= ConfigSLMYOLO.METRIC_IOU_THRESHOLD
                    metric_storage[cls_id].append((float(det[4]), 1.0 if is_tp else 0.0))
                    if is_tp:
                        total_tp += 1
                        matched.setdefault(cls_id, set()).add(best_gt_idx)
                    else:
                        total_fp += 1

                for cls_id, gt_boxes in gt_by_class.items():
                    total_fn += len(gt_boxes) - len(matched.get(cls_id, set()))

    num_batches = max(len(val_loader), 1)
    avg_losses = {key: value / num_batches for key, value in component_totals.items()}
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1_score = 2.0 * precision * recall / (precision + recall + 1e-6)

    ap_values = []
    for cls_id in range(ConfigSLMYOLO.NUM_CLASSES):
        ap = compute_average_precision(metric_storage[cls_id], gt_counts[cls_id])
        if ap is not None:
            ap_values.append(ap)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score),
        "map50": float(np.mean(ap_values)) if len(ap_values) > 0 else 0.0,
    }
    return avg_losses, metrics

def save_phase_loss_curves(phase_histories, output_dir):
    loss_curve_dir = ConfigSLMYOLO.LOSS_CURVE_DIR or os.path.join(output_dir, "loss_curves")
    os.makedirs(loss_curve_dir, exist_ok=True)

    phase_titles = {
        "phase1": "Phase 1 Loss Curve",
        "phase2": "Phase 2 Loss Curve",
        "phase3": "Phase 3 Loss Curve",
    }

    for phase_name, history in phase_histories.items():
        if len(history["train_epochs"]) == 0:
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(history["train_epochs"], history["train_loss"], label="Train Loss", linewidth=2)

        if len(history["val_epochs"]) > 0:
            plt.plot(history["val_epochs"], history["val_loss"], label="Val Loss", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(phase_titles.get(phase_name, f"{phase_name} Loss Curve"))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(loss_curve_dir, f"{phase_name}_loss_curve.png"),
            dpi=ConfigSLMYOLO.VIS_DPI
        )
        plt.close()

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
    plt.savefig(ConfigSLMYOLO.get_training_curve_path(), dpi=ConfigSLMYOLO.VIS_DPI)
    plt.close()

# =========================================================
# 训练函数
# =========================================================
def train():
    """主训练函数"""
    ConfigSLMYOLO.initialize()
    init_log_file()
    ConfigSLMYOLO.print_config()

    device = ConfigSLMYOLO.DEVICE
    log_to_file(f"Log file path: {ConfigSLMYOLO.LOG_FILE}")
    log_to_file(f"Visualization save path: {ConfigSLMYOLO.OUTPUT_DIR}")
    log_to_file(f"使用设备: {device}")
    log_to_file("初始化模型组件...")

    teacher = ConvTeacher().to(device)
    teacher_loaded, teacher_message = load_teacher_checkpoint(teacher, ConfigSLMYOLO.TEACHER_CHECKPOINT, device)
    log_to_file(teacher_message)
    if not teacher_loaded:
        raise FileNotFoundError(
            f"Unable to start SLM training without a trained teacher checkpoint: {ConfigSLMYOLO.TEACHER_CHECKPOINT}"
        )
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    optical_student = OpticalStudent().to(device)
    detector = YOLOLightHead(
        in_channels=ConfigSLMYOLO.YOLO_HEAD_IN_CHANNELS,
        out_channels=ConfigSLMYOLO.get_detector_output_channels(),
    ).to(device)

    teacher_optical_model = TeacherOpticalModel(teacher, optical_student).to(device)
    optical_yolo_model = OpticalYOLOModel(optical_student, detector).to(device)

    phase1_loss = MultiScaleMSELoss().to(device)
    phase2_loss = CombinedLoss(ConfigSLMYOLO.ANCHORS, ConfigSLMYOLO.NUM_CLASSES, ConfigSLMYOLO.STRIDES).to(device)
    phase3_loss = YOLOLoss(ConfigSLMYOLO.ANCHORS, ConfigSLMYOLO.NUM_CLASSES, ConfigSLMYOLO.STRIDES).to(device)

    optimizer_optical = optim.Adam(
        optical_student.parameters(),
        lr=ConfigSLMYOLO.OPTICAL_LEARNING_RATE,
        weight_decay=ConfigSLMYOLO.OPTICAL_WEIGHT_DECAY,
    )
    optimizer_detector = optim.Adam(
        detector.parameters(),
        lr=ConfigSLMYOLO.DETECTOR_LEARNING_RATE,
        weight_decay=ConfigSLMYOLO.DETECTOR_WEIGHT_DECAY,
    )

    def yolo_collate_fn(batch):
        images, targets = zip(*batch)
        return torch.stack(images, 0), list(targets)

    log_to_file("加载数据集...")
    train_dataset = YOLODataset(split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=ConfigSLMYOLO.BATCH_SIZE,
        shuffle=True,
        collate_fn=yolo_collate_fn,
    )

    val_loader = None
    has_validation = False
    try:
        val_dataset = YOLODataset(split="val")
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=ConfigSLMYOLO.BATCH_SIZE,
                shuffle=False,
                collate_fn=yolo_collate_fn,
            )
            has_validation = True
            log_to_file(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
        else:
            log_to_file(f"训练集大小: {len(train_dataset)}, 验证集为空")
    except Exception as exc:
        log_to_file(f"警告: 验证集加载失败: {exc}")
        log_to_file(f"训练集大小: {len(train_dataset)}，将仅使用训练集进行训练")

    history = {
        "train_total": [],
        "val_total": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "map50": [],
    }
    phase_histories = {
        "phase1": {"train_epochs": [], "train_loss": [], "val_epochs": [], "val_loss": []},
        "phase2": {"train_epochs": [], "train_loss": [], "val_epochs": [], "val_loss": []},
        "phase3": {"train_epochs": [], "train_loss": [], "val_epochs": [], "val_loss": []},
    }
    best_train_loss = float("inf")
    best_map50 = -1.0

    log_to_file("开始训练...")
    init_epoch_log_table()

    for epoch in range(ConfigSLMYOLO.EPOCHS):
        phase, _ = ConfigSLMYOLO.get_current_phase(epoch)
        teacher_optical_model.train()
        optical_yolo_model.train()
        teacher_optical_model.teacher.eval()

        train_component_sums = {
            "total": 0.0,
            "box": 0.0,
            "obj": 0.0,
            "noobj": 0.0,
            "cls": 0.0,
        }

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{ConfigSLMYOLO.EPOCHS} [{phase}]", leave=True):
            images, targets = prepare_batch(batch, device)

            if phase == "phase1":
                optimizer_optical.zero_grad()
                teacher_features, optical_features = teacher_optical_model(images)
                loss = phase1_loss(optical_features, teacher_features)
                loss.backward()
                optimizer_optical.step()
                loss_stats = {"total": float(loss.item()), "box": 0.0, "obj": 0.0, "noobj": 0.0, "cls": 0.0}
            elif phase == "phase2":
                optimizer_optical.zero_grad()
                optimizer_detector.zero_grad()
                teacher_features, optical_features = teacher_optical_model(images)
                predictions = optical_yolo_model.detector(optical_features)
                loss, loss_stats = phase2_loss(teacher_features, optical_features, predictions, targets)
                loss.backward()
                optimizer_optical.step()
                optimizer_detector.step()
            else:
                optimizer_optical.zero_grad()
                optimizer_detector.zero_grad()
                predictions = optical_yolo_model(images)
                loss, loss_stats = phase3_loss(predictions, targets)
                loss.backward()
                optimizer_optical.step()
                optimizer_detector.step()

            for key in train_component_sums:
                train_component_sums[key] += float(loss_stats.get(key, 0.0))

        avg_train = {key: value / max(len(train_loader), 1) for key, value in train_component_sums.items()}
        phase_histories[phase]["train_epochs"].append(epoch + 1)
        phase_histories[phase]["train_loss"].append(avg_train["total"])
        history["train_total"].append(avg_train["total"])

        val_losses = None
        val_metrics = None
        if has_validation and val_loader is not None and ((epoch + 1) % ConfigSLMYOLO.VAL_INTERVAL == 0):
            val_losses, val_metrics = validate(
                phase,
                teacher_optical_model,
                optical_yolo_model,
                val_loader,
                phase1_loss,
                phase2_loss,
                phase3_loss,
                device,
            )
            phase_histories[phase]["val_epochs"].append(epoch + 1)
            phase_histories[phase]["val_loss"].append(val_losses["total"])
            history["val_total"].append(val_losses["total"])
            history["precision"].append(val_metrics["precision"] if val_metrics is not None else np.nan)
            history["recall"].append(val_metrics["recall"] if val_metrics is not None else np.nan)
            history["f1"].append(val_metrics["f1"] if val_metrics is not None else np.nan)
            history["map50"].append(val_metrics["map50"] if val_metrics is not None else np.nan)
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
        elif avg_train["total"] < best_train_loss:
            best_train_loss = avg_train["total"]
            is_best = True

        if is_best:
            torch.save(
                {
                    "epoch": epoch,
                    "phase": phase,
                    "optical_student_state_dict": optical_student.state_dict(),
                    "detector_state_dict": detector.state_dict(),
                    "optimizer_optical_state_dict": optimizer_optical.state_dict(),
                    "optimizer_detector_state_dict": optimizer_detector.state_dict(),
                    "loss": avg_train["total"],
                    "val_map50": val_metrics["map50"] if val_metrics is not None else None,
                },
                ConfigSLMYOLO.get_best_optical_yolo_path(),
            )
            torch.save(
                {
                    "epoch": epoch,
                    "phase": phase,
                    "teacher_state_dict": teacher.state_dict(),
                    "optical_student_state_dict": optical_student.state_dict(),
                    "optimizer_optical_state_dict": optimizer_optical.state_dict(),
                    "loss": avg_train["total"],
                },
                ConfigSLMYOLO.get_best_teacher_optical_path(),
            )

        if (epoch + 1) % ConfigSLMYOLO.SAVE_INTERVAL == 0 or epoch + 1 == ConfigSLMYOLO.EPOCHS:
            torch.save(
                {
                    "epoch": epoch,
                    "phase": phase,
                    "optical_student_state_dict": optical_student.state_dict(),
                    "detector_state_dict": detector.state_dict(),
                    "optimizer_optical_state_dict": optimizer_optical.state_dict(),
                    "optimizer_detector_state_dict": optimizer_detector.state_dict(),
                    "loss": avg_train["total"],
                    "val_map50": val_metrics["map50"] if val_metrics is not None else None,
                },
                ConfigSLMYOLO.get_optical_yolo_checkpoint_path(epoch + 1),
            )
            torch.save(
                {
                    "epoch": epoch,
                    "phase": phase,
                    "teacher_state_dict": teacher.state_dict(),
                    "optical_student_state_dict": optical_student.state_dict(),
                    "optimizer_optical_state_dict": optimizer_optical.state_dict(),
                    "loss": avg_train["total"],
                },
                ConfigSLMYOLO.get_teacher_optical_checkpoint_path(epoch + 1),
            )

        if (epoch + 1) % ConfigSLMYOLO.VIS_INTERVAL == 0:
            teacher_optical_model.eval()
            optical_yolo_model.eval()
            with torch.no_grad():
                sample_images, sample_targets = next(iter(train_loader))
                sample_images = sample_images.to(device)
                teacher_features, optical_features = teacher_optical_model(sample_images)
                detections = (
                    optical_yolo_model.detector(optical_features)
                    if phase == "phase2"
                    else optical_yolo_model(sample_images)
                )
                pred_results = decode_detections(detections)

                if phase == "phase1":
                    save_phase1_visualization(
                        epoch + 1,
                        sample_images,
                        teacher_features,
                        optical_features,
                        ConfigSLMYOLO.VISUALIZATION_DIR,
                    )
                elif phase == "phase2":
                    save_phase2_visualization(
                        epoch + 1,
                        sample_images,
                        teacher_features,
                        optical_features,
                        sample_targets,
                        pred_results,
                        ConfigSLMYOLO.VISUALIZATION_DIR,
                    )
                else:
                    save_phase3_visualization(
                        epoch + 1,
                        sample_images,
                        optical_features,
                        sample_targets,
                        pred_results,
                        ConfigSLMYOLO.VISUALIZATION_DIR,
                    )

        current_lr = (
            optimizer_optical.param_groups[0]["lr"]
            if phase == "phase1"
            else optimizer_detector.param_groups[0]["lr"]
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
            lr=current_lr,
            best_status=ConfigSLMYOLO.EPOCH_TABLE_BEST_MARK if is_best else "",
        )

        save_phase_loss_curves(phase_histories, ConfigSLMYOLO.OUTPUT_DIR)
        save_training_curves(history, ConfigSLMYOLO.OUTPUT_DIR)

    append_plain_log(ConfigSLMYOLO.get_epoch_table_separator())
    log_to_file(f"Phase loss curves saved to: {ConfigSLMYOLO.LOSS_CURVE_DIR}")
    log_to_file(f"Training curve saved to: {ConfigSLMYOLO.get_training_curve_path()}")
    log_to_file(f"Best optical YOLO model saved to: {ConfigSLMYOLO.get_best_optical_yolo_path()}")
    log_to_file(f"Best teacher-optical model saved to: {ConfigSLMYOLO.get_best_teacher_optical_path()}")
    log_to_file("训练完成！")

if __name__ == "__main__":
    train()
