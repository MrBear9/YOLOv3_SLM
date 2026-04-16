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
    VISUALIZATION_DIR = None
    
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
    OBJ_WEIGHT = 1.0
    CLS_WEIGHT = 0.15
    
    # 锚框设置
    STRIDES = [8, 16, 32]
    ANCHORS = [
        # P3: 小目标（士兵、小型装备）
        [[16, 20], [24, 32], [32, 48]],
        # P4: 中目标（坦克、战机主体）
        [[48, 64], [64, 96], [96, 128]],
        # P5: 大目标（军舰、大型战机）
        [[128, 160], [192, 240], [256, 320]]
    ]
    
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
    
    # 训练控制参数
    VIS_INTERVAL = 5
    SAVE_INTERVAL = 10
    VAL_INTERVAL = 5  # 验证间隔，每5轮验证一次
    
    # 检测参数
    CONF_THRESH = 0.5
    NMS_THRESH = 0.4
    MAX_DET = 8
    
    # 可视化参数
    VIS_BATCH_SIZE = 4
    VIS_DPI = 120
    
    @classmethod
    def initialize(cls):
        """初始化配置"""
        cls.CLASS_NAMES, cls.NUM_CLASSES = load_class_names(cls.YAML_PATH)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        cls.LOG_ROOT_DIR = os.path.join(cls.OUTPUT_DIR, "logs")
        cls.VISUALIZATION_DIR = os.path.join(cls.OUTPUT_DIR, "visualizations")
        os.makedirs(cls.LOG_ROOT_DIR, exist_ok=True)
        os.makedirs(cls.VISUALIZATION_DIR, exist_ok=True)
        cls.TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.LOG_FILE = os.path.join(cls.LOG_ROOT_DIR, f"training_log_{cls.TIMESTAMP}.txt")
    
    @classmethod
    def get_detector_output_channels(cls):
        """获取检测头输出通道数"""
        return 3 * (4 + 1 + cls.NUM_CLASSES)
    
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

def init_log_file():
    """初始化日志文件"""
    os.makedirs(ConfigSLMYOLO.LOG_ROOT_DIR, exist_ok=True)
    with open(ConfigSLMYOLO.LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("光学教师YOLO SLM训练日志\n")
        f.write("=" * 80 + "\n")
        f.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

def log_to_file(message, also_print=True):
    """记录消息到文件"""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_message = f"{timestamp} {message}"
    
    with open(ConfigSLMYOLO.LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_message + "\n")
    
    if also_print:
        print(log_message)

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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True) # 仅加载模型权重 兼容性：如果您的模型包含自定义对象，weights_only=True 可能无法加载 原本是没有这个参数
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

def enhance_feature_for_display(feature_map):
    """增强特征图显示效果"""
    feature_map = np.asarray(feature_map, dtype=np.float32)
    low = np.percentile(feature_map, 2)
    high = np.percentile(feature_map, 98)
    if high - low < 1e-6:
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
        amp = torch.sqrt(intensity.clamp(min=0)+1e-8)
        field = torch.complex(amp, torch.zeros_like(amp))
        field = self.prop1(self.slm1(field))
        field = self.prop2(self.slm2(field))
        out = torch.abs(field)**2
        if self.enable_norm:
            out = out / (out.mean(dim=[2,3], keepdim=True) + 1e-6)
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
        self.loss_full_weight = 0.02
        self.loss_low1_weight = 1.0
        self.loss_low2_weight = 0.5

    def forward(self, student_output, teacher_output):
        loss_full = F.mse_loss(student_output, teacher_output)
        loss_low1 = F.mse_loss(self.pool1(student_output), self.pool1(teacher_output))
        loss_low2 = F.mse_loss(self.pool2(student_output), self.pool2(teacher_output))
        return (loss_full * self.loss_full_weight + 
                loss_low1 * self.loss_low1_weight + 
                loss_low2 * self.loss_low2_weight)

class YOLOLoss(nn.Module):
    """YOLO检测损失函数（阶段2和阶段3使用）"""
    def __init__(self, anchors, num_classes, strides):
        super().__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.num_classes = num_classes
        self.strides = strides
        
        self.box_weight = ConfigSLMYOLO.BOX_WEIGHT
        self.obj_weight = ConfigSLMYOLO.OBJ_WEIGHT
        self.cls_weight = ConfigSLMYOLO.CLS_WEIGHT
        
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    
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
                    # 确保目标张量有正确的形状
                    if targets[b].dim() == 1:
                        # 单个目标的情况
                        cls_id, tx, ty, tw, th = targets[b]
                        cls_id = int(cls_id.item())
                    else:
                        # 多个目标的情况
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
                    
                    if best_iou > 0.5:
                        target_boxes[b, gy, gx, best_anchor, 0] = tx * grid_w - gx
                        target_boxes[b, gy, gx, best_anchor, 1] = ty * grid_h - gy
                        target_boxes[b, gy, gx, best_anchor, 2] = torch.log(tw / anchors[best_anchor, 0] + 1e-6)
                        target_boxes[b, gy, gx, best_anchor, 3] = torch.log(th / anchors[best_anchor, 1] + 1e-6)
                        target_obj[b, gy, gx, best_anchor] = 1.0
                        target_cls[b, gy, gx, best_anchor, cls_id] = 1.0
            
            obj_mask = target_obj > 0.5
            noobj_mask = target_obj <= 0.5
            
            if obj_mask.sum() > 0:
                box_loss = self.mse_loss(pred_boxes[obj_mask], target_boxes[obj_mask]).mean()
                obj_loss = self.bce_loss(pred_obj[obj_mask], target_obj[obj_mask]).mean()
                cls_loss = self.bce_loss(pred_cls[obj_mask], target_cls[obj_mask]).mean()
            else:
                box_loss = torch.tensor(0.0, device=pred_boxes.device)
                obj_loss = torch.tensor(0.0, device=pred_boxes.device)
                cls_loss = torch.tensor(0.0, device=pred_boxes.device)
            
            if noobj_mask.sum() > 0:
                noobj_loss = self.bce_loss(pred_obj[noobj_mask], target_obj[noobj_mask]).mean()
            else:
                noobj_loss = torch.tensor(0.0, device=pred_boxes.device)
            
            scale_loss = (self.box_weight * box_loss + 
                         self.obj_weight * obj_loss + 
                         self.obj_weight * 0.5 * noobj_loss + 
                         self.cls_weight * cls_loss)
            
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
                     self.obj_weight * 0.5 * noobj_loss +
                     self.cls_weight * cls_loss)

        if torch.isnan(scale_loss) or torch.isinf(scale_loss):
            scale_loss = torch.tensor(0.0, device=pred_boxes.device)
            print("Warning: invalid detection loss encountered, reset to 0")

        total_loss += scale_loss

    return total_loss

YOLOLoss.forward = yolo_loss_forward_fixed

class CombinedLoss(nn.Module):
    """组合损失函数（阶段2使用）"""
    def __init__(self, anchors, num_classes, strides):
        super().__init__()
        self.teacher_student_loss = MultiScaleMSELoss()
        self.yolo_loss = YOLOLoss(anchors, num_classes, strides)
        self.teacher_student_weight = 0.5
        self.yolo_weight = 0.5
    
    def forward(self, teacher_output, optical_output, predictions, targets):
        teacher_student_loss = self.teacher_student_loss(optical_output, teacher_output)
        yolo_loss = self.yolo_loss(predictions, targets)
        return (self.teacher_student_weight * teacher_student_loss + 
                self.yolo_weight * yolo_loss)

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
            self.detector = YOLOLightHead(in_channels=1, 
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
    strides = [8, 16, 32]
    anchors = ConfigSLMYOLO.ANCHORS
    
    for i, pred in enumerate(preds):
        grid_h, grid_w = pred.shape[2], pred.shape[3]
        stride = strides[i]
        anchor_set = anchors[i]
        
        pred = pred.permute(0, 2, 3, 1).reshape(batch_size, grid_h, grid_w, 3, -1)
        
        obj_conf = torch.sigmoid(pred[..., 4])
        cls_conf = torch.sigmoid(pred[..., 5:])
        bbox_pred = pred[..., :4]
        
        for b in range(batch_size):
            for gh in range(grid_h):
                for gw in range(grid_w):
                    for a in range(3):
                        obj_score = obj_conf[b, gh, gw, a].item()
                        if obj_score < conf_thresh:
                            continue
                        
                        cls_scores = cls_conf[b, gh, gw, a]
                        cls_score, cls_id = cls_scores.max(dim=-1)
                        final_conf = obj_score * cls_score.item()
                        
                        if final_conf < conf_thresh:
                            continue
                        
                        tx, ty, tw, th = bbox_pred[b, gh, gw, a]
                        
                        x_center = (gw + torch.sigmoid(tx).item()) * stride
                        y_center = (gh + torch.sigmoid(ty).item()) * stride
                        
                        anchor_w, anchor_h = anchor_set[a]
                        w = anchor_w * torch.exp(torch.clamp(tw, min=-8.0, max=8.0)).item()
                        h = anchor_h * torch.exp(torch.clamp(th, min=-8.0, max=8.0)).item()
                        
                        x_center = max(0, min(x_center, img_size - 1))
                        y_center = max(0, min(y_center, img_size - 1))
                        w = max(1, min(w, img_size))
                        h = max(1, min(h, img_size))
                        
                        detections[b].append([x_center, y_center, w, h, final_conf, cls_id.item()])
    
    for b in range(batch_size):
        if len(detections[b]) > 0:
            detections[b] = apply_classwise_nms(detections[b], nms_thresh, max_det)
    
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
def validate(phase, teacher_optical_model, optical_yolo_model, val_loader, 
            phase1_loss, phase2_loss, phase3_loss, device):
    """验证函数，计算验证集损失"""
    teacher_optical_model.eval()
    optical_yolo_model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            
            if phase == "phase1":
                # 阶段1验证：只计算教师-光学损失
                teacher_features, optical_features = teacher_optical_model(images)
                loss = phase1_loss(optical_features, teacher_features)
            elif phase == "phase2":
                # 阶段2验证：计算组合损失
                teacher_features, optical_features = teacher_optical_model(images)
                detections = optical_yolo_model(images)
                loss = phase2_loss(teacher_features, optical_features, detections, targets)
            else:
                # 阶段3验证：计算YOLO检测损失
                detections = optical_yolo_model(images)
                loss = phase3_loss(detections, targets)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss

# =========================================================
# 训练函数
# =========================================================
def train():
    """主训练函数"""
    # 先初始化配置
    ConfigSLMYOLO.initialize()
    init_log_file()
    ConfigSLMYOLO.print_config()
    
    device = ConfigSLMYOLO.DEVICE
    log_to_file(f"使用设备: {device}")
    
    log_to_file("初始化模型组件...")
    
    # 初始化教师网络
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
    
    # 初始化光学学生网络
    optical_student = OpticalStudent().to(device)
    
    # 初始化检测头
    detector = YOLOLightHead(in_channels=1, 
                           out_channels=ConfigSLMYOLO.get_detector_output_channels()).to(device)
    
    # 初始化组合模型
    teacher_optical_model = TeacherOpticalModel(teacher, optical_student).to(device)
    optical_yolo_model = OpticalYOLOModel(optical_student, detector).to(device)
    
    # 初始化损失函数
    phase1_loss = MultiScaleMSELoss().to(device)  # 阶段1：教师-光学层损失
    phase2_loss = CombinedLoss(ConfigSLMYOLO.ANCHORS, ConfigSLMYOLO.NUM_CLASSES, ConfigSLMYOLO.STRIDES).to(device)  # 阶段2：组合损失
    phase3_loss = YOLOLoss(ConfigSLMYOLO.ANCHORS, ConfigSLMYOLO.NUM_CLASSES, ConfigSLMYOLO.STRIDES).to(device)  # 阶段3：YOLO检测损失
    
    # 初始化优化器（使用不同的学习率）
    optimizer_optical = optim.Adam(
        optical_student.parameters(), 
        lr=ConfigSLMYOLO.OPTICAL_LEARNING_RATE,
        weight_decay=ConfigSLMYOLO.OPTICAL_WEIGHT_DECAY
    )
    optimizer_detector = optim.Adam(
        detector.parameters(), 
        lr=ConfigSLMYOLO.DETECTOR_LEARNING_RATE,
        weight_decay=ConfigSLMYOLO.DETECTOR_WEIGHT_DECAY
    )
    
    # 自定义collate函数处理不同数量的边界框
    def yolo_collate_fn(batch):
        """处理YOLO数据集中不同数量边界框的问题"""
        images = []
        targets = []
        
        for img, target in batch:
            images.append(img)
            targets.append(target)
        
        # 堆叠图像
        images = torch.stack(images, 0)
        
        return images, targets
    
    # 加载数据集
    log_to_file("加载数据集...")
    train_dataset = YOLODataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=ConfigSLMYOLO.BATCH_SIZE, 
                             shuffle=True, collate_fn=yolo_collate_fn)
    
    # 加载验证集（如果存在）
    try:
        val_dataset = YOLODataset(split="val")
        val_loader = DataLoader(val_dataset, batch_size=ConfigSLMYOLO.BATCH_SIZE, 
                               shuffle=False, collate_fn=yolo_collate_fn)
        log_to_file(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
        has_validation = True
    except Exception as e:
        log_to_file(f"警告: 验证集加载失败: {e}")
        log_to_file("将仅使用训练集进行训练")
        has_validation = False
        val_loader = None
    
    # 训练循环
    log_to_file("开始训练...")
    
    for epoch in range(ConfigSLMYOLO.EPOCHS):
        phase, phase_desc = ConfigSLMYOLO.get_current_phase(epoch)
        log_to_file(f"Epoch {epoch+1}/{ConfigSLMYOLO.EPOCHS} - {phase_desc}")
        
        total_loss = 0
        teacher_optical_model.train()
        optical_yolo_model.train()
        teacher_optical_model.teacher.eval()
        
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            
            if phase == "phase1":
                # 阶段1：只训练光学层（教师网络约束）
                optimizer_optical.zero_grad()
                
                teacher_features, optical_features = teacher_optical_model(images)
                loss = phase1_loss(optical_features, teacher_features)
                
                loss.backward()
                optimizer_optical.step()
                total_loss += loss.item()
                
            elif phase == "phase2":
                # 阶段2：教师约束 + 检测头训练
                optimizer_optical.zero_grad()
                optimizer_detector.zero_grad()
                
                # 获取教师特征和光学特征
                teacher_features, optical_features = teacher_optical_model(images)
                # 获取检测结果
                detections = optical_yolo_model(images)
                # 组合损失
                loss = phase2_loss(teacher_features, optical_features, detections, targets)
                
                loss.backward()
                optimizer_optical.step()
                optimizer_detector.step()
                total_loss += loss.item()
                
            else:
                # 阶段3：只训练光学层 + 检测头
                optimizer_optical.zero_grad()
                optimizer_detector.zero_grad()
                
                detections = optical_yolo_model(images)
                loss = phase3_loss(detections, targets)
                
                loss.backward()
                optimizer_optical.step()
                optimizer_detector.step()
                total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        log_to_file(f"Epoch {epoch+1} 平均损失: {avg_loss:.6f}")
        
        # 验证集验证（如果存在验证集）
        if has_validation and (epoch + 1) % ConfigSLMYOLO.VAL_INTERVAL == 0:
            log_to_file("进行验证集验证...")
            val_loss = validate(phase, teacher_optical_model, optical_yolo_model, 
                              val_loader, phase1_loss, phase2_loss, phase3_loss, device)
            log_to_file(f"Epoch {epoch+1} 验证损失: {val_loss:.6f}")
        
        # 保存模型
        if (epoch + 1) % ConfigSLMYOLO.SAVE_INTERVAL == 0 or epoch + 1 == ConfigSLMYOLO.EPOCHS:
            # 保存光学YOLO检测模型
            optical_yolo_path = os.path.join(ConfigSLMYOLO.OUTPUT_DIR, f"optical_yolo_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'optical_student_state_dict': optical_student.state_dict(),
                'detector_state_dict': detector.state_dict(),
                'optimizer_optical_state_dict': optimizer_optical.state_dict(),
                'optimizer_detector_state_dict': optimizer_detector.state_dict(),
                'loss': avg_loss,
            }, optical_yolo_path)
            log_to_file(f"保存光学YOLO模型: {optical_yolo_path}")
            
            # 保存教师-光学模型
            teacher_optical_path = os.path.join(ConfigSLMYOLO.OUTPUT_DIR, f"teacher_optical_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'teacher_state_dict': teacher.state_dict(),
                'optical_student_state_dict': optical_student.state_dict(),
                'optimizer_optical_state_dict': optimizer_optical.state_dict(),
                'loss': avg_loss,
            }, teacher_optical_path)
            log_to_file(f"保存教师-光学模型: {teacher_optical_path}")
        
        # 可视化
        if (epoch + 1) % ConfigSLMYOLO.VIS_INTERVAL == 0:
            log_to_file("生成可视化结果...")
            teacher_optical_model.eval()
            optical_yolo_model.eval()
            
            with torch.no_grad():
                # 获取一批样本用于可视化
                sample_images, sample_targets = next(iter(train_loader))
                sample_images = sample_images.to(device)
                
                # 获取特征和预测
                teacher_features, optical_features = teacher_optical_model(sample_images)
                detections = optical_yolo_model(sample_images)
                pred_results = decode_detections(detections)
                
                # 根据阶段选择不同的可视化函数
                if phase == "phase1":
                    # 阶段1：类似optical_teacher.py的可视化
                    save_phase1_visualization(
                        epoch + 1, sample_images, teacher_features, optical_features,
                        ConfigSLMYOLO.VISUALIZATION_DIR
                    )
                elif phase == "phase2":
                    # 阶段2：输入图+真实框，光学特征，约束，预测结果
                    save_phase2_visualization(
                        epoch + 1, sample_images, teacher_features, optical_features,
                        sample_targets, pred_results, ConfigSLMYOLO.VISUALIZATION_DIR
                    )
                else:
                    # 阶段3：类似optical_teacher_yolo.py的可视化
                    save_phase3_visualization(
                        epoch + 1, sample_images, optical_features, sample_targets,
                        pred_results, ConfigSLMYOLO.VISUALIZATION_DIR
                    )
            
            log_to_file(f"{phase_desc}可视化结果已保存到: {ConfigSLMYOLO.VISUALIZATION_DIR}")
    
    log_to_file("训练完成！")

if __name__ == "__main__":
    train()
