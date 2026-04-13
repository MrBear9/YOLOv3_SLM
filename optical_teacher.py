import yaml
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from PIL import Image

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
    TEACHER_OUTPUT_DIR = r"output\OpticalTeacher"
    LOG_ROOT_DIR = None
    LOG_FILE = None
    TIMESTAMP = None
    
    # 训练参数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 640
    BATCH_SIZE = 8  # 批次大小，用于训练时的内存占用
    EPOCHS = 200
    
    # 损失权重
    BOX_WEIGHT = 0.05  # 目标框损失权重，用于计算总损失
    OBJ_WEIGHT = 1.5  # 目标损失权重，用于计算总损失
    CLS_WEIGHT = 0.15  # 类别损失权重，用于计算总损失
    
    # 锚框设置
    STRIDES = [8, 16, 32]
    ANCHORS = [
        [[10,13], [16,30], [33,23]],
        [[30,61], [62,45], [59,119]],
        [[116,90], [156,198], [373,326]]
    ]
    
    # 优化器参数
    LEARNING_RATE = 5e-4  # 学习率，用于优化模型参数
    WEIGHT_DECAY = 1e-5  # 权重衰减，用于防止过拟合
    OPTIMIZER = "Adam"
    
    # 光学传播参数
    WAVELENGTH = 532e-9  # 波长，单位：米
    PIXEL_SIZE = 6.4e-6  # 像素大小，单位：米
    PROP_DISTANCE_1 = 0.01  # 传播距离1，单位：米
    PROP_DISTANCE_2 = 0.02  # 传播距离2，单位：米
    SLM_MODE = "phase"  # SLM模式，"phase"或"amplitude"
    RESOLUTION = (640, 640)
    
    # 训练控制参数
    NORM_ENABLE_EPOCH = 50  # 启用归一化的轮数
    VIS_INTERVAL = 2  # 可视化间隔，单位：轮数
    
    # 损失函数权重
    LOSS_FULL_WEIGHT = 0.02  # 完整损失权重，用于计算总损失
    LOSS_LOW1_WEIGHT = 1.0  # 低1损失权重，用于计算总损失
    LOSS_LOW2_WEIGHT = 0.5  # 低2损失权重，用于计算总损失
    
    # 检测参数
    CONF_THRESH = 0.5  # 置信度阈值，用于筛选检测框
    NMS_THRESH = 0.4  # NMS阈值，用于合并重叠检测框
    MAX_DET = 8  # 最大检测框数量，用于限制检测框数量
    
    # 可视化参数
    VIS_BATCH_SIZE = 4  # 可视化批次大小，用于可视化检测结果
    VIS_DPI = 120  # 可视化DPI，用于调整可视化结果的清晰度
    
    @classmethod
    def initialize(cls):
        """初始化配置，加载类别信息和创建输出目录"""
        cls.CLASS_NAMES, cls.NUM_CLASSES = load_class_names(cls.YAML_PATH)
        os.makedirs(cls.TEACHER_OUTPUT_DIR, exist_ok=True)
        cls.LOG_ROOT_DIR = os.path.join(cls.TEACHER_OUTPUT_DIR, "logs")
        os.makedirs(cls.LOG_ROOT_DIR, exist_ok=True)
        cls.TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.LOG_FILE = os.path.join(cls.LOG_ROOT_DIR, f"training_log_{cls.TIMESTAMP}.txt")
    
    @classmethod
    def get_detector_output_channels(cls):
        """获取检测头输出通道数"""
        return 3 * (4 + 1 + cls.NUM_CLASSES)
    
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
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("光学教师网络训练日志\n")
        f.write("="*80 + "\n")
        f.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

init_log_file()
log_to_file(f"日志文件路径: {Config.LOG_FILE}")
log_to_file(f"可视化结果保存路径: {Config.LOG_ROOT_DIR.replace('\\logs', '')}")
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
        amp = torch.sqrt(intensity.clamp(min=0)+1e-8)
        field = torch.complex(amp, torch.zeros_like(amp))
        field = self.prop1(self.slm1(field))
        field = self.prop2(self.slm2(field))
        out = torch.abs(field)**2
        if self.enable_norm:
            out = out / (out.mean(dim=[2,3], keepdim=True) + 1e-6)
        return out

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
class MultiScaleMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = nn.AvgPool2d(8)
        self.pool2 = nn.AvgPool2d(32)

    def forward(self, s, t):
        loss_full = F.mse_loss(s, t)
        loss_low1 = F.mse_loss(self.pool1(s), self.pool1(t))
        loss_low2 = F.mse_loss(self.pool2(s), self.pool2(t))
        return (loss_full * Config.LOSS_FULL_WEIGHT + 
                loss_low1 * Config.LOSS_LOW1_WEIGHT + 
                loss_low2 * Config.LOSS_LOW2_WEIGHT)

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

def enhance_feature_for_display(feature_map):
    feature_map = np.asarray(feature_map, dtype=np.float32)
    low = np.percentile(feature_map, 2)
    high = np.percentile(feature_map, 98)
    if high - low < 1e-6:
        return np.zeros_like(feature_map)
    
    feature_map = np.clip((feature_map - low) / (high - low), 0.0, 1.0)
    return np.power(feature_map, 0.8)

def decode_detections(preds, conf_thresh=None, nms_thresh=None, max_det=None, img_size=None):
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
    
    for i, pred in enumerate(preds):
        grid_h, grid_w = pred.shape[2], pred.shape[3]
        stride = strides[i]
        
        pred = pred.permute(0, 2, 3, 1).reshape(batch_size, grid_h, grid_w, 3, -1)
        
        obj_conf = torch.sigmoid(pred[..., 4])
        cls_conf = torch.sigmoid(pred[..., 5:])
        bbox_pred = pred[..., :4]
        
        for b in range(batch_size):
            for gh in range(grid_h):
                for gw in range(grid_w):
                    for a in range(3):
                        conf = obj_conf[b, gh, gw, a].item()
                        if conf < conf_thresh:
                            continue
                        
                        cls_score, cls_id = cls_conf[b, gh, gw, a].max(dim=-1)
                        final_conf = conf * cls_score.item()
                        
                        if final_conf < conf_thresh:
                            continue
                        
                        x_center = (gw + 0.5) * stride
                        y_center = (gh + 0.5) * stride
                        w = stride
                        h = stride
                        
                        detections[b].append([x_center, y_center, w, h, final_conf, cls_id.item()])
    
    for b in range(batch_size):
        if len(detections[b]) > 0:
            detections[b] = np.array(detections[b])
            detections[b] = detections[b][detections[b][:, 4].argsort()[::-1]]
            
            keep = []
            while len(detections[b]) > 0 and len(keep) < max_det:
                keep.append(0)
                if len(detections[b]) == 1:
                    break
                
                ious = []
                for i in range(1, len(detections[b])):
                    box1 = detections[b][0]
                    box2 = detections[b][i]
                    x1 = max(box1[0] - box1[2]/2, box2[0] - box2[2]/2)
                    y1 = max(box1[1] - box1[3]/2, box2[1] - box2[3]/2)
                    x2 = min(box1[0] + box1[2]/2, box2[0] + box2[2]/2)
                    y2 = min(box1[1] + box1[3]/2, box2[1] + box2[3]/2)
                    
                    if x2 < x1 or y2 < y1:
                        ious.append(0)
                    else:
                        inter = (x2 - x1) * (y2 - y1)
                        union = box1[2] * box1[3] + box2[2] * box2[3] - inter
                        ious.append(inter / union)
                
                detections[b] = detections[b][1:][np.array(ious) < nms_thresh]
            
            detections[b] = detections[b][keep]
    
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
    
    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
    
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
    log_to_file(f"[Vis] saved: {save_path}", also_print=False)

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
    log_to_file(f"[Detection Vis] saved: {save_path}", also_print=False)

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
    log_to_file(f"  类别损失权重: {Config.CLS_WEIGHT}")
    
    log_to_file("\n【锚框设置】")
    log_to_file(f"  步长: {Config.STRIDES}")
    log_to_file(f"  P3锚框: {Config.ANCHORS[0]}")
    log_to_file(f"  P4锚框: {Config.ANCHORS[1]}")
    log_to_file(f"  P5锚框: {Config.ANCHORS[2]}")
    
    log_to_file("\n【模型参数】")
    teacher = ConvTeacher()
    student = OpticalStudent()
    detector = YOLOLightHead(in_channels=1, out_channels=Config.get_detector_output_channels())
    
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
    log_to_file(f"  可视化结果: {Config.TEACHER_OUTPUT_DIR}")
    log_to_file(f"  模型保存: {os.path.join(Config.TEACHER_OUTPUT_DIR, 'optical_student_final.pth')}")
    
    log_to_file("\n" + "="*80)
    log_to_file("参数设置输出完成")
    log_to_file("="*80 + "\n")

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
    log_to_file(f"  类别损失权重: {Config.CLS_WEIGHT}")
    
    log_to_file("\n【锚框设置】")
    log_to_file(f"  步长: {Config.STRIDES}")
    log_to_file(f"  P3锚框: {Config.ANCHORS[0]}")
    log_to_file(f"  P4锚框: {Config.ANCHORS[1]}")
    log_to_file(f"  P5锚框: {Config.ANCHORS[2]}")
    
    log_to_file("\n【模型参数】")
    teacher = ConvTeacher()
    student = OpticalStudent((640, 640), mode="phase")
    detector = YOLOLightHead(in_channels=1, out_channels=3*(4 + 1 + Config.NUM_CLASSES))
    
    log_to_file(f"  ConvTeacher可训练参数: {sum(p.numel() for p in teacher.parameters() if p.requires_grad):,}")
    log_to_file(f"  OpticalStudent可训练参数: {sum(p.numel() for p in student.parameters() if p.requires_grad):,}")
    log_to_file(f"  检测头输出通道数: {3*(4 + 1 + Config.NUM_CLASSES)}")
    log_to_file(f"  检测头可训练参数: {sum(p.numel() for p in detector.parameters() if p.requires_grad):,}")
    
    log_to_file("\n【优化器参数】")
    log_to_file(f"  优化器: {Config.OPTIMIZER}")
    log_to_file(f"  学习率: {Config.LEARNING_RATE}")
    log_to_file(f"  权重衰减: {Config.WEIGHT_DECAY}")
    
    log_to_file("\n【路径设置】")
    log_to_file(f"  日志文件: {Config.LOG_FILE}")
    log_to_file(f"  可视化结果: {Config.TEACHER_OUTPUT_DIR}")
    log_to_file(f"  模型保存: {os.path.join(Config.TEACHER_OUTPUT_DIR, 'optical_student_final.pth')}")
    
    log_to_file("\n" + "="*80)
    log_to_file("参数设置输出完成")
    log_to_file("="*80 + "\n")

log_all_parameters()

# =========================================================
# 训练函数
# =========================================================
def train():
    device = Config.DEVICE
    log_to_file(f"使用设备: {device}")
    
    log_to_file("初始化教师网络（ConvTeacher）...")
    teacher = ConvTeacher()
    
    for p in teacher.parameters():
        p.requires_grad = False
    log_to_file("教师网络参数已冻结")
    
    log_to_file("初始化学生网络（OpticalStudent）...")
    student = OpticalStudent().to(device)
    
    log_to_file("加载数据集...")
    dataset = MilitaryFeatureDataset(teacher=teacher, device=device)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    log_to_file(f"数据集大小: {len(dataset)}")
    
    optimizer = torch.optim.Adam(student.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = MultiScaleMSELoss()
    
    vis_dir = os.path.join(Config.TEACHER_OUTPUT_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    loss_curve = []
    best_loss = float('inf')
    
    log_to_file("="*60)
    log_to_file("开始训练光学相位层...")
    log_to_file("="*60)
    
    for epoch in range(Config.EPOCHS):
        student.train()
        loss_sum = 0
        
        for x, t in tqdm(loader, desc=f"Epoch {epoch}/{Config.EPOCHS}", leave=False):
            x = x.to(device)
            t = t.to(device)
            y = student(x)
            loss = criterion(y, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        avg_loss = loss_sum / len(loader)
        loss_curve.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "optical_student_best.pth")
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

        log_to_file(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_curve, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(Config.TEACHER_OUTPUT_DIR, "loss_curve.png"), dpi=Config.VIS_DPI)
    plt.close()

    model_save_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "optical_student_final.pth")
    torch.save(student.state_dict(), model_save_path)
    
    log_to_file("="*60)
    log_to_file("训练完成！")
    log_to_file(f"最佳模型已保存到: {os.path.join(Config.TEACHER_OUTPUT_DIR, 'optical_student_best.pth')}")
    log_to_file(f"最终模型已保存到: {model_save_path}")
    log_to_file(f"所有结果已保存到: {Config.TEACHER_OUTPUT_DIR}")
    log_to_file("="*60)

if __name__ == "__main__":
    train()