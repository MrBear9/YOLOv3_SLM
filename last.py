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
from datetime import datetime



# 读取data.yaml获取类别信息

def load_class_names(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    class_names = config.get('names', [])

    if isinstance(class_names, list):
        # 列表格式：按索引生成字典
        CLASS_NAMES = {i: name for i, name in enumerate(class_names)}
    else:
        raise ValueError(f"Unsupported 'names' format in {yaml_path}, must be list or dict")

    num_classes = len(CLASS_NAMES)
    return CLASS_NAMES, num_classes

# 类别名称映射（从data.yaml读取）
YAML_PATH = r"data\military\data.yaml"
CLASS_NAMES, NUM_CLASSES = load_class_names(YAML_PATH)
print(f"加载类别信息: {CLASS_NAMES}, 类别数: {NUM_CLASSES}")

# ===================== 输出路径配置 =====================
# 统一输出路径配置，便于后期修改
TEACHER_OUTPUT_DIR = r"output\Teacher"
os.makedirs(TEACHER_OUTPUT_DIR, exist_ok=True)
print(f"教师网络输出目录: {TEACHER_OUTPUT_DIR}")

# ===================== 日志输出设置 =====================
# 日志输出根目录（与可视化结果同一路径）
LOG_ROOT_DIR = os.path.join(TEACHER_OUTPUT_DIR, "logs")
os.makedirs(LOG_ROOT_DIR, exist_ok=True)

# 生成带时间戳的日志文件名
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_ROOT_DIR, f"training_log_{TIMESTAMP}.txt")

# 日志输出函数
def log_to_file(message, log_file=LOG_FILE, also_print=True):
    """
    将消息输出到日志文件，并可选择是否同时打印到控制台
    Args:
        message: 要输出的消息
        log_file: 日志文件路径
        also_print: 是否同时打印到控制台
    """
    with open(log_file, 'a', encoding='utf-8') as f:
        # 添加时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")
    if also_print:
        print(message)

# 初始化日志文件（清空旧内容）
def init_log_file(log_file=LOG_FILE):
    """
    初始化日志文件，写入训练开始信息
    """
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("YOLOv3 教师网络-检测头训练日志\n")
        f.write("="*80 + "\n")
        f.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

# 初始化日志文件
init_log_file()
log_to_file(f"日志文件路径: {LOG_FILE}")
log_to_file(f"可视化结果保存路径: {LOG_ROOT_DIR.replace('\\logs', '')}")
log_to_file("")

# ===================== 日志输出设置 =====================
# 日志输出根目录（与可视化结果同一路径）
LOG_ROOT_DIR = os.path.join(TEACHER_OUTPUT_DIR, "logs")
os.makedirs(LOG_ROOT_DIR, exist_ok=True)

# 生成带时间戳的日志文件名
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_ROOT_DIR, f"training_log_{TIMESTAMP}.txt")

# 日志输出函数
def log_to_file(message, log_file=LOG_FILE, also_print=True):
    """
    将消息输出到日志文件，并可选择是否同时打印到控制台
    Args:
        message: 要输出的消息
        log_file: 日志文件路径
        also_print: 是否同时打印到控制台
    """
    with open(log_file, 'a', encoding='utf-8') as f:
        # 添加时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")
    if also_print:
        print(message)

# 初始化日志文件（清空旧内容）
def init_log_file(log_file=LOG_FILE):
    """
    初始化日志文件，写入训练开始信息
    """
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("YOLOv3 教师网络-检测头训练日志\n")
        f.write("="*80 + "\n")
        f.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

# 初始化日志文件
init_log_file()
log_to_file(f"日志文件路径: {LOG_FILE}")
log_to_file(f"可视化结果保存路径: {LOG_ROOT_DIR.replace('\\logs', '')}")
log_to_file("")


#教师网络
class MultiScaleTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        #高斯低通
        self.gauss = nn.Conv2d(1, 1, kernel_size=15, padding=7, bias=False)
        grid = torch.arange(15) - 7
        X, Y = torch.meshgrid(grid, grid, indexing='ij')
        sigma = 3.0
        g = torch.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        self.gauss.weight.data = g.unsqueeze(0).unsqueeze(0)

        # Sobel 边缘
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        self.sobel = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel.weight.data = sobel_x.unsqueeze(0).unsqueeze(0)


    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        low = self.gauss(gray)
        edge = torch.abs(self.sobel(gray))  # 去符号
        edge = F.avg_pool2d(edge, 4)  # 去高频
        edge = F.interpolate(edge, gray.shape[-2:])

        return torch.sigmoid(low + edge)

# =========================================================
# 教师网络2（卷积-仿YOLOV3）
# =========================================================
class ConvTeacher(nn.Module):
        """
        Conv-based teacher that mimics YOLOv5s P3 feature
        Output:
            - single-channel
            - same spatial resolution as input
            - P3-equivalent receptive field (~8x downsample)
        """

        def __init__(self):
            super().__init__()

            # Stem: downsample x2
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.SiLU()
            )

            # Downsample x4
            self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU()
            )

            # Downsample x8  → P3 scale
            self.conv3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU()
            )

            # P3 refinement (YOLOv5 C3-like, but lightweight)
            self.refine = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(),

                nn.Conv2d(64, 32, kernel_size=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU()
            )

            # Projection to single channel
            self.project = nn.Conv2d(32, 1, kernel_size=1, bias=False)


        def forward(self, x):
            """
            x: RGB or Gray image tensor [B, C, H, W]
            """
            if x.shape[1] > 1:
                x = x.mean(dim=1, keepdim=True)  # to gray

            f = self.conv1(x)
            f = self.conv2(f)
            f = self.conv3(f)  # P3 scale (H/8, W/8)

            f = self.refine(f)
            f = self.project(f)

            # Remove sign & high-frequency sensitivity
            f = torch.abs(f)

            # Upsample back to full resolution for pixelwise loss
            f = F.interpolate(
                f, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

            # Bounded intensity-like output
            return torch.sigmoid(f)

# 测试
teacher1 = MultiScaleTeacher()
teacher2 = ConvTeacher()
# 查看是否可训练
log_to_file(f"MultiScaleTeacher 可训练参数：{sum(p.numel() for p in teacher1.parameters() if p.requires_grad):,}")
log_to_file(f"ConvTeacher 可训练参数：{sum(p.numel() for p in teacher2.parameters() if p.requires_grad):,}")

#检测头
# 轻量化卷积块：深度可分离卷积 + BN + SiLU（轻量化激活函数）
class LightConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(LightConvBlock, self).__init__()
        # 深度卷积（逐通道卷积，轻量化核心）
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   bias=False)
        # 点卷积（1×1卷积，调整通道数）
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # 比ReLU更高效的轻量化激活

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# 轻量化检测头（基于FPN的多尺度特征生成）
class YOLOLightHead(nn.Module):
    def __init__(self, in_channels=1, out_channels=27):
        super(YOLOLightHead, self).__init__()
        # 基础通道数（轻量化设计，避免过多通道）
        base_ch = 32

        # 1. 初始特征提取（将1通道特征图扩展为基础通道）
        self.init_conv = LightConvBlock(in_channels, base_ch, kernel_size=3, stride=1)

        # 2. 下采样生成P5（20×20）：640→320→160→80→40→20（5次下采样，stride=2）
        self.down_to_p5 = nn.Sequential(
            LightConvBlock(base_ch, base_ch * 2, stride=2),  # 640→320, 32→64
            LightConvBlock(base_ch * 2, base_ch * 4, stride=2),  # 320→160, 64→128
            LightConvBlock(base_ch * 4, base_ch * 8, stride=2),  # 160→80, 128→256
            LightConvBlock(base_ch * 8, base_ch * 8, stride=2),  # 80→40, 256→256
            LightConvBlock(base_ch * 8, base_ch * 8, stride=2)  # 40→20, 256→256
        )

        # 3. P5上采样 + 融合生成P4（40×40）
        self.up_p5_to_p4 = nn.Upsample(scale_factor=2, mode='nearest')  # 轻量化上采样
        self.fuse_p4 = LightConvBlock(base_ch * 8 + base_ch * 8, base_ch * 4)  # 融合P5上采样特征和40×40中间特征

        # 4. P4上采样 + 融合生成P3（80×80）
        self.up_p4_to_p3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse_p3 = LightConvBlock(base_ch * 4 + base_ch * 8, base_ch * 2)  # 融合P4上采样特征和80×80中间特征

        # 5. 各尺度特征调整为27通道输出
        self.head_p5 = nn.Conv2d(base_ch * 8, out_channels, 1)  # P5:20×20×27
        self.head_p4 = nn.Conv2d(base_ch * 4, out_channels, 1)  # P4:40×40×27
        self.head_p3 = nn.Conv2d(base_ch * 2, out_channels, 1)  # P3:80×80×27

    def forward(self, x):
        # x: [B, 1, 640, 640]（Backbone输出）
        B = x.shape[0]

        # 初始特征提取
        x_init = self.init_conv(x)  # [B, 32, 640, 640]

        # 分步下采样，保存中间特征（用于融合）
        x320 = self.down_to_p5[0](x_init)  # [B, 64, 320, 320]
        x160 = self.down_to_p5[1](x320)  # [B, 128, 160, 160]
        x80 = self.down_to_p5[2](x160)  # [B, 256, 80, 80]
        x40 = self.down_to_p5[3](x80)  # [B, 256, 40, 40]
        p5 = self.down_to_p5[4](x40)  # [B, 256, 20, 20] → P5基础特征

        # 生成P4
        p5_up = self.up_p5_to_p4(p5)  # [B, 256, 40, 40]
        p4_fuse = torch.cat([p5_up, x40], dim=1)  # 特征融合：[B, 512, 40, 40]
        p4 = self.fuse_p4(p4_fuse)  # [B, 256, 40, 40] → P4基础特征

        # 生成P3
        p4_up = self.up_p4_to_p3(p4)  # [B, 256, 80, 80]
        p3_fuse = torch.cat([p4_up, x80], dim=1)  # 特征融合：[B, 512, 80, 80]
        p3 = self.fuse_p3(p3_fuse)  # [B, 128, 80, 80] → P3基础特征

        # 调整通道数为27
        p5_out = self.head_p5(p5)  # [B, 27, 20, 20]
        p4_out = self.head_p4(p4)  # [B, 27, 40, 40]
        p3_out = self.head_p3(p3)  # [B, 27, 80, 80]

        return p3_out, p4_out, p5_out


# 测试代码
if __name__ == "__main__":
    # 模拟Backbone输出：batch_size=1, 1×640×640
    x = torch.randn(1, 1, 640, 640)
    # 初始化检测头
    head = YOLOLightHead(in_channels=1, out_channels=27)
    # 前向传播
    p3, p4, p5 = head(x)
    # 输出各尺度特征图形状
    log_to_file(f"P3 shape: {p3.shape}")  # 预期：torch.Size([1, 27, 80, 80])
    log_to_file(f"P4 shape: {p4.shape}")  # 预期：torch.Size([1, 27, 40, 40])
    log_to_file(f"P5 shape: {p5.shape}")  # 预期：torch.Size([1, 27, 20, 20])


#超参数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 640
BATCH_SIZE = 4
EPOCHS = 100

#损失权重
BOX_WEIGHT   = 0.05
OBJ_WEIGHT   = 1.5
CLS_WEIGHT   = 0.15
#三个感受野尺寸
STRIDES = [8, 16, 32]
ANCHORS = [
    [[10,13], [16,30], [33,23]],   # P3 小目标（军人）
    [[30,61], [62,45], [59,119]],  # P4 中目标（坦克、战机）
    [[116,90], [156,198], [373,326]] # P5 大目标（军舰）
]

# ===================== 输出所有参数设置 =====================
def log_all_parameters():
    """
    将所有参数设置输出到日志文件
    """
    log_to_file("="*80)
    log_to_file("训练参数设置")
    log_to_file("="*80)
    
    # 数据集参数
    log_to_file("\n【数据集参数】")
    log_to_file(f"  YAML路径: {YAML_PATH}")
    log_to_file(f"  类别数量: {NUM_CLASSES}")
    log_to_file(f"  类别名称: {CLASS_NAMES}")
    
    # 训练参数
    log_to_file("\n【训练参数】")
    log_to_file(f"  设备: {DEVICE}")
    log_to_file(f"  图像尺寸: {IMG_SIZE}")
    log_to_file(f"  批次大小: {BATCH_SIZE}")
    log_to_file(f"  训练轮数: {EPOCHS}")
    
    # 损失权重
    log_to_file("\n【损失权重】")
    log_to_file(f"  边界框损失权重: {BOX_WEIGHT}")
    log_to_file(f"  目标性损失权重: {OBJ_WEIGHT}")
    log_to_file(f"  类别损失权重: {CLS_WEIGHT}")
    
    # 锚框设置
    log_to_file("\n【锚框设置】")
    log_to_file(f"  步长: {STRIDES}")
    log_to_file(f"  P3锚框 (小目标): {ANCHORS[0]}")
    log_to_file(f"  P4锚框 (中目标): {ANCHORS[1]}")
    log_to_file(f"  P5锚框 (大目标): {ANCHORS[2]}")
    
    # 模型参数
    log_to_file("\n【模型参数】")
    teacher1 = MultiScaleTeacher()
    teacher2 = ConvTeacher()
    detector = YOLOLightHead(in_channels=1, out_channels=3*(4 + 1 + NUM_CLASSES))
    
    log_to_file(f"  MultiScaleTeacher可训练参数: {sum(p.numel() for p in teacher1.parameters() if p.requires_grad):,}")
    log_to_file(f"  ConvTeacher可训练参数: {sum(p.numel() for p in teacher2.parameters() if p.requires_grad):,}")
    log_to_file(f"  检测头输出通道数: {3*(4 + 1 + NUM_CLASSES)}")
    log_to_file(f"  检测头可训练参数: {sum(p.numel() for p in detector.parameters() if p.requires_grad):,}")
    
    # 优化器参数
    log_to_file("\n【优化器参数】")
    log_to_file(f"  优化器: Adam")
    log_to_file(f"  学习率: 1e-3")
    log_to_file(f"  权重衰减: 1e-4")
    
    # 路径设置
    log_to_file("\n【路径设置】")
    log_to_file(f"  日志文件: {LOG_FILE}")
    log_to_file(f"  可视化结果: {LOG_ROOT_DIR.replace('\\logs', '')}")
    log_to_file(f"  模型保存: {os.path.join(TEACHER_OUTPUT_DIR, 'military_teacher_detector.pth')}")
    
    log_to_file("\n" + "="*80)
    log_to_file("参数设置输出完成")
    log_to_file("="*80 + "\n")

# 输出所有参数设置
log_all_parameters()

# ===================== 输出所有参数设置 =====================
def log_all_parameters():
    """
    将所有参数设置输出到日志文件
    """
    log_to_file("="*80)
    log_to_file("训练参数设置")
    log_to_file("="*80)
    
    # 数据集参数
    log_to_file("\n【数据集参数】")
    log_to_file(f"  YAML路径: {YAML_PATH}")
    log_to_file(f"  类别数量: {NUM_CLASSES}")
    log_to_file(f"  类别名称: {CLASS_NAMES}")
    
    # 训练参数
    log_to_file("\n【训练参数】")
    log_to_file(f"  设备: {DEVICE}")
    log_to_file(f"  图像尺寸: {IMG_SIZE}")
    log_to_file(f"  批次大小: {BATCH_SIZE}")
    log_to_file(f"  训练轮数: {EPOCHS}")
    
    # 损失权重
    log_to_file("\n【损失权重】")
    log_to_file(f"  边界框损失权重: {BOX_WEIGHT}")
    log_to_file(f"  目标性损失权重: {OBJ_WEIGHT}")
    log_to_file(f"  类别损失权重: {CLS_WEIGHT}")
    
    # 锚框设置
    log_to_file("\n【锚框设置】")
    log_to_file(f"  步长: {STRIDES}")
    log_to_file(f"  P3锚框 (小目标): {ANCHORS[0]}")
    log_to_file(f"  P4锚框 (中目标): {ANCHORS[1]}")
    log_to_file(f"  P5锚框 (大目标): {ANCHORS[2]}")
    
    # 模型参数
    log_to_file("\n【模型参数】")
    teacher1 = MultiScaleTeacher()
    teacher2 = ConvTeacher()
    detector = YOLOLightHead(in_channels=1, out_channels=3*(4 + 1 + NUM_CLASSES))
    
    log_to_file(f"  MultiScaleTeacher可训练参数: {sum(p.numel() for p in teacher1.parameters() if p.requires_grad):,}")
    log_to_file(f"  ConvTeacher可训练参数: {sum(p.numel() for p in teacher2.parameters() if p.requires_grad):,}")
    log_to_file(f"  检测头输出通道数: {3*(4 + 1 + NUM_CLASSES)}")
    log_to_file(f"  检测头可训练参数: {sum(p.numel() for p in detector.parameters() if p.requires_grad):,}")
    
    # 优化器参数
    log_to_file("\n【优化器参数】")
    log_to_file(f"  优化器: Adam")
    log_to_file(f"  学习率: 1e-3")
    log_to_file(f"  权重衰减: 1e-4")
    
    # 路径设置
    log_to_file("\n【路径设置】")
    log_to_file(f"  日志文件: {LOG_FILE}")
    log_to_file(f"  可视化结果: {LOG_ROOT_DIR.replace('\\logs', '')}")
    log_to_file(f"  模型保存: {os.path.join(TEACHER_OUTPUT_DIR, 'military_teacher_detector.pth')}")
    
    log_to_file("\n" + "="*80)
    log_to_file("参数设置输出完成")
    log_to_file("="*80 + "\n")

# 输出所有参数设置
log_all_parameters()

#辅助函数
def prediction_response_map(pred):
    """
    将检测头输出转成更可解释的响应热图，而不是直接平均27个logit通道。
    使用obj_conf * cls_conf的组合作为响应值。
    Args:
        pred: 预测张量 [channels, H, W]
    Returns:
        response: 响应热图 [H, W]
    """
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    pred = pred.permute(1, 2, 0).reshape(grid_h, grid_w, 3, -1)
    obj_conf = torch.sigmoid(pred[..., 4])
    cls_conf, _ = torch.sigmoid(pred[..., 5:]).max(dim=-1)
    response = (obj_conf * cls_conf).max(dim=-1).values
    return response.detach().cpu().numpy()


def enhance_feature_for_display(feature_map):
    """
    用稳健分位数拉伸弱响应，避免特征图看起来一片发黑。
    Args:
        feature_map: 特征图numpy数组
    Returns:
        enhanced: 增强后的特征图
    """
    feature_map = np.asarray(feature_map, dtype=np.float32)
    low = np.percentile(feature_map, 2)
    high = np.percentile(feature_map, 98)
    if high - low < 1e-6:
        return np.zeros_like(feature_map)
    
    feature_map = np.clip((feature_map - low) / (high - low), 0.0, 1.0)
    return np.power(feature_map, 0.8)


#检测解码函数
def decode_detections(preds, conf_thresh=0.5, nms_thresh=0.4, max_det=100, img_size=640):
    """
    解码检测结果，返回边界框、置信度和类别
    Args:
        preds: 预测结果列表 [p3, p4, p5]
        conf_thresh: 置信度阈值
        nms_thresh: 非极大值抑制阈值
        max_det: 最大检测数量
        img_size: 图像尺寸
    Returns:
        detections: 每个样本的检测结果列表，每个检测为 [x, y, w, h, conf, cls_id]
    """
    batch_size = preds[0].shape[0]
    detections = [[] for _ in range(batch_size)]
    strides = [8, 16, 32]
    
    for i, pred in enumerate(preds):
        grid_h, grid_w = pred.shape[2], pred.shape[3]
        stride = strides[i]
        
        # 重塑预测为 (batch_size, grid_h, grid_w, 3, 5+num_classes)
        pred = pred.permute(0, 2, 3, 1).reshape(batch_size, grid_h, grid_w, 3, -1)
        
        obj_conf = torch.sigmoid(pred[..., 4])
        cls_probs = torch.sigmoid(pred[..., 5:])
        cls_conf, cls_id = cls_probs.max(dim=-1)
        final_conf = obj_conf * cls_conf
        obj_mask = final_conf > conf_thresh
        
        for b in range(batch_size):
            for h in range(grid_h):
                for w in range(grid_w):
                    for anchor in range(3):
                        if obj_mask[b, h, w, anchor]:
                            bx, by, bw, bh = pred[b, h, w, anchor, :4]
                            conf = final_conf[b, h, w, anchor].item()
                            current_cls_id = cls_id[b, h, w, anchor].item()
                            
                            # 解码边界框坐标
                            x = float(torch.clamp(bx * stride, 0, img_size - 1).item())
                            y = float(torch.clamp(by * stride, 0, img_size - 1).item())
                            width = float(torch.clamp(torch.abs(bw) * stride, 1, img_size).item())
                            height = float(torch.clamp(torch.abs(bh) * stride, 1, img_size).item())
                            
                            detections[b].append([x, y, width, height, conf, current_cls_id])
    
    # 非极大值抑制
    final_detections = []
    for batch_detections in detections:
        if batch_detections:
            batch_detections = torch.tensor(batch_detections, dtype=torch.float32)
            keep = non_max_suppression(batch_detections, nms_thresh)
            kept = batch_detections[keep]
            if kept.shape[0] > max_det:
                kept = kept[torch.argsort(kept[:, 4], descending=True)[:max_det]]
            final_detections.append(kept.tolist())
        else:
            final_detections.append([])
    
    return final_detections


def non_max_suppression(detections, nms_thresh):
    """非极大值抑制"""
    if len(detections) == 0:
        return []
    
    # 按置信度排序
    confidences = detections[:, 4]
    sorted_indices = torch.argsort(confidences, descending=True)
    
    keep = []
    while len(sorted_indices) > 0:
        # 取置信度最高的检测
        current_idx = sorted_indices[0]
        keep.append(current_idx.item())
        
        if len(sorted_indices) == 1:
            break
        
        # 计算与剩余检测的IoU
        current_box = detections[current_idx, :4]
        other_boxes = detections[sorted_indices[1:], :4]
        
        ious = calculate_iou(current_box.unsqueeze(0), other_boxes)
        
        # 移除重叠度高的检测
        keep_indices = torch.where(ious < nms_thresh)[0]
        sorted_indices = sorted_indices[keep_indices + 1]
    
    return keep


def calculate_iou(box1, box2):
    """计算IoU
    box格式: [x, y, w, h] (中心点坐标格式)
    """
    # 转换为左上角和右下角坐标
    box1_x1 = box1[..., 0] - box1[..., 2] / 2
    box1_y1 = box1[..., 1] - box1[..., 3] / 2
    box1_x2 = box1[..., 0] + box1[..., 2] / 2
    box1_y2 = box1[..., 1] + box1[..., 3] / 2
    
    box2_x1 = box2[..., 0] - box2[..., 2] / 2
    box2_y1 = box2[..., 1] - box2[..., 3] / 2
    box2_x2 = box2[..., 0] + box2[..., 2] / 2
    box2_y2 = box2[..., 1] + box2[..., 3] / 2
    
    # 计算交集
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # 计算并集
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    return iou


#辅助函数
def prediction_response_map(pred):
    """
    将检测头输出转成更可解释的响应热图，而不是直接平均27个logit通道。
    使用obj_conf * cls_conf的组合作为响应值。
    Args:
        pred: 预测张量 [channels, H, W]
    Returns:
        response: 响应热图 [H, W]
    """
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    pred = pred.permute(1, 2, 0).reshape(grid_h, grid_w, 3, -1)
    obj_conf = torch.sigmoid(pred[..., 4])
    cls_conf, _ = torch.sigmoid(pred[..., 5:]).max(dim=-1)
    response = (obj_conf * cls_conf).max(dim=-1).values
    return response.detach().cpu().numpy()


def enhance_feature_for_display(feature_map):
    """
    用稳健分位数拉伸弱响应，避免特征图看起来一片发黑。
    Args:
        feature_map: 特征图numpy数组
    Returns:
        enhanced: 增强后的特征图
    """
    feature_map = np.asarray(feature_map, dtype=np.float32)
    low = np.percentile(feature_map, 2)
    high = np.percentile(feature_map, 98)
    if high - low < 1e-6:
        return np.zeros_like(feature_map)
    
    feature_map = np.clip((feature_map - low) / (high - low), 0.0, 1.0)
    return np.power(feature_map, 0.8)


#可视化
def visualize_teacher_features_and_scales(
        teacher_model, detector_model, dataloader, device,
        save_path=None, num_samples=4,
        conf_thresh=0.5, nms_thresh=0.4
):
    # 如果没有指定保存路径，使用默认路径
    if save_path is None:
        save_path = os.path.join(TEACHER_OUTPUT_DIR, "teacher_feature_visualization.png")
    """
    可视化教师网络特征图 + 检测头输出的P3/P4/P5多尺度特征
    对应你提供的4行输入图结构：Input → Teacher Feature → P3 → P4 → P5
    Args:
        teacher_model: 教师网络
        detector_model: 检测头模型
        dataloader: 数据加载器
        device: 设备
        save_path: 保存路径
        num_samples: 可视化样本数量
        conf_thresh: 置信度阈值
        nms_thresh: 非极大值抑制阈值
    """
    teacher_model.eval()
    detector_model.eval()

    # 取num_samples个样本进行可视化
    samples = []
    for batch in dataloader:
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs).to(device)
        targets = [t.to(device) for t in targets]
        samples = list(zip(imgs, targets))[:num_samples]
        break  # 只取第一个batch的前num_samples个样本

    # 创建画布：4行 × 5列（对应4个样本，每个样本5张子图）
    fig, axes = plt.subplots(nrows=num_samples, ncols=5, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    # 颜色映射：使用热力图cmap='hot'，和你提供的图一致
    cmap = plt.cm.get_cmap('hot')

    for idx, (img, target) in enumerate(samples):
        # ========== 第1列：原始输入图 + 标注框 ==========
        ax_input = axes[idx, 0]
        # 还原图像：Tensor → numpy → 0-255 → RGB
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        # 转灰度（和你提供的输入图一致）
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        ax_input.imshow(img_gray, cmap='gray')

        # 绘制真实标注框（红色）
        h, w = img_gray.shape
        for t in target:
            cls, cx, cy, bw, bh = t.cpu().numpy()
            # YOLO格式(x_center,y_center,w,h) → 像素坐标(x1,y1,x2,y2)
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            # 绘制红色框 + 类别标签
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
            ax_input.add_patch(rect)
            ax_input.text(x1, y1 - 5, CLASS_NAMES[int(cls)], color='white',
                          bbox=dict(facecolor='red', alpha=0.8), fontsize=8)

        # 解码检测结果并绘制（绿色框 + 置信度）
        with torch.no_grad():
            feat = teacher_model(img.unsqueeze(0).to(device))
            p3, p4, p5 = detector_model(feat)
            detections = decode_detections([p3, p4, p5], conf_thresh, nms_thresh, img_size=IMG_SIZE)
        
        # 绘制检测框（绿色）
        for det in detections[0]:
            x, y, width, height, conf, cls_id = det
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)
            
            # 绘制绿色检测框
            rect_det = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, 
                                    edgecolor='green', linewidth=2, linestyle='--')
            ax_input.add_patch(rect_det)
            
            # 添加置信度标签
            class_name = CLASS_NAMES.get(int(cls_id), f"Class {int(cls_id)}")
            text_y = min(h - 12, max(2, y1 + 2))
            ax_input.text(x1, text_y, f"{class_name} {conf:.2f}", color='white',
                         bbox=dict(facecolor='green', alpha=0.85, pad=1.5), fontsize=7)

        ax_input.set_title(f"Input {idx + 1} (真实: {len(target)}, 检测: {len(detections[0])})", fontsize=10)
        ax_input.axis('off')

        # ========== 第2列：教师网络输出特征图 ==========
        ax_teacher = axes[idx, 1]
        with torch.no_grad():
            feat = teacher_model(img.unsqueeze(0).to(device))  # [1,1,H,W]
        feat_np = feat.squeeze().cpu().numpy()
        im_teacher = ax_teacher.imshow(feat_np, cmap=cmap, vmin=0, vmax=1)
        ax_teacher.set_title(f"Teacher Feature {idx + 1}", fontsize=10)
        ax_teacher.axis('off')
        # 添加颜色条
        fig.colorbar(im_teacher, ax=ax_teacher, fraction=0.046, pad=0.04)

        # ========== 第3-5列：检测头输出的P3/P4/P5特征图 ==========
        # P3 (80×80) - 使用obj*cls响应图
        ax_p3 = axes[idx, 2]
        p3_response = prediction_response_map(p3.squeeze())
        p3_response = enhance_feature_for_display(p3_response)
        im_p3 = ax_p3.imshow(p3_response, cmap=cmap, vmin=0, vmax=1)
        ax_p3.set_title(f"P3 (80×80)", fontsize=10)
        ax_p3.axis('off')
        fig.colorbar(im_p3, ax=ax_p3, fraction=0.046, pad=0.04)

        # P4 (40×40) - 使用obj*cls响应图
        ax_p4 = axes[idx, 3]
        p4_response = prediction_response_map(p4.squeeze())
        p4_response = enhance_feature_for_display(p4_response)
        im_p4 = ax_p4.imshow(p4_response, cmap=cmap, vmin=0, vmax=1)
        ax_p4.set_title(f"P4 (40×40)", fontsize=10)
        ax_p4.axis('off')
        fig.colorbar(im_p4, ax=ax_p4, fraction=0.046, pad=0.04)

        # P5 (20×20) - 使用obj*cls响应图
        ax_p5 = axes[idx, 4]
        p5_response = prediction_response_map(p5.squeeze())
        p5_response = enhance_feature_for_display(p5_response)
        im_p5 = ax_p5.imshow(p5_response, cmap=cmap, vmin=0, vmax=1)
        ax_p5.set_title(f"P5 (20×20)", fontsize=10)
        ax_p5.axis('off')
        fig.colorbar(im_p5, ax=ax_p5, fraction=0.046, pad=0.04)

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_to_file(f"可视化结果已保存至: {save_path}")


#数据集
class MilitaryDataset(Dataset):
    def __init__(self, root_dir, mode='train', img_size=640):
        self.img_size = img_size

        # 精准路径：train/val/test 下的 images 和 labels
        self.img_dir = os.path.join(root_dir, mode, 'images')
        self.label_dir = os.path.join(root_dir, mode, 'labels')

        self.images = [f for f in os.listdir(self.img_dir) if f.endswith(('jpg', 'png', 'jpeg'))]
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")
        targets = torch.zeros((0, 5))
        if os.path.exists(label_path):
            try:
                labels = np.loadtxt(label_path).reshape(-1, 5)
                targets = torch.tensor(labels, dtype=torch.float32)
            except:
                pass
        return img, targets

#损失函数
class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target, batch_size):
        pred = pred.permute(0, 2, 3, 1).reshape(batch_size, pred.shape[2], pred.shape[3], 3, -1)

        obj_mask = target[..., 4] == 1
        noobj_mask = target[..., 4] == 0

        # Box loss
        box_loss = self.mse(pred[obj_mask][:, :4], target[obj_mask][:, :4])

        # Obj loss（均衡正负样本，防止过早收敛）
        obj_loss = self.bce(pred[obj_mask][:, 4], target[obj_mask][:, 4])
        noobj_loss = self.bce(pred[noobj_mask][:, 4], target[noobj_mask][:, 4])
        obj_loss = obj_loss + 0.5 * noobj_loss

        # Cls loss
        cls_loss = self.bce(pred[obj_mask][:, 5:], target[obj_mask][:, 5:])

        # 加权总和
        total = BOX_WEIGHT * box_loss + OBJ_WEIGHT * obj_loss + CLS_WEIGHT * cls_loss
        return total, box_loss, obj_loss, cls_loss

# ===================== 辅助函数 =====================
def calculate_detection_metrics(detections, targets):
    """
    计算检测指标（精确率、召回率、F1分数）
    Args:
        detections: 检测结果列表，每个元素为 [x, y, w, h, conf, cls_id]
        targets: 真实目标张量 [N, 5]，格式为 [cls, cx, cy, bw, bh]
    Returns:
        precision: 精确率
        recall: 召回率
        f1_score: F1分数
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for i in range(len(detections)):
        # 检测到的目标数
        det_count = len(detections[i])
        
        # 真实目标数（过滤空目标）
        gt_count = 0
        for t in targets[i]:
            if t[4] != 0:  # 非空目标
                gt_count += 1
        
        true_positives += min(det_count, gt_count)
        false_positives += max(0, det_count - gt_count)
        false_negatives += max(0, gt_count - det_count)
    
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    
    return precision, recall, f1_score

# ===================== 辅助函数 =====================
def calculate_detection_metrics(detections, targets):
    """
    计算检测指标（精确率、召回率、F1分数）
    Args:
        detections: 检测结果列表，每个元素为 [x, y, w, h, conf, cls_id]
        targets: 真实目标张量 [N, 5]，格式为 [cls, cx, cy, bw, bh]
    Returns:
        precision: 精确率
        recall: 召回率
        f1_score: F1分数
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for i in range(len(detections)):
        # 检测到的目标数
        det_count = len(detections[i])
        
        # 真实目标数（过滤空目标）
        gt_count = 0
        for t in targets[i]:
            if t[4] != 0:  # 非空目标
                gt_count += 1
        
        true_positives += min(det_count, gt_count)
        false_positives += max(0, det_count - gt_count)
        false_negatives += max(0, gt_count - det_count)
    
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    
    return precision, recall, f1_score

# ===================== 构建模型 + 教师模型 =====================
def build_target(targets, anchors, stride, num_classes, img_size):
    batch_size = targets.shape[0]
    h, w = img_size // stride, img_size // stride
    num_anchors = len(anchors)

    target_tensor = torch.zeros((batch_size, h, w, num_anchors, 5 + num_classes), device=DEVICE)

    for b in range(batch_size):
        for t in targets[b]:
            cls, cx, cy, bw, bh = t
            cx_s = cx * w
            cy_s = cy * h
            bw_s = bw * w
            bh_s = bh * h

            i = int(cx_s)
            j = int(cy_s)

            best_idx = 0
            best_iou = 0
            for a_idx, (aw, ah) in enumerate(anchors):
                inter = min(bw_s, aw) * min(bh_s, ah)
                union = bw_s * bh_s + aw * ah - inter
                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
                    best_idx = a_idx

            target_tensor[b, j, i, best_idx, 0:4] = torch.tensor([cx_s, cy_s, bw_s, bh_s], device=DEVICE)
            target_tensor[b, j, i, best_idx, 4] = 1
            target_tensor[b, j, i, best_idx, 5 + int(cls)] = 1

    return target_tensor

# ===================== 模型 & 优化器 =====================
teacher = MultiScaleTeacher().to(DEVICE)
teacher.eval()  # 教师网络固定，不训练

detector = YOLOLightHead(in_channels=1, out_channels=3*(4 + 1 + NUM_CLASSES)).to(DEVICE)
optimizer = optim.Adam(detector.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = YOLOLoss()

#数据集加载
root_path = r"data\military"

# 加载训练集
train_dataset = MilitaryDataset(root_path, mode='train', img_size=IMG_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)

#训练循环
from tqdm import tqdm

log_to_file("="*80)
log_to_file("开始训练模型...")
log_to_file("="*80)

for epoch in range(EPOCHS):
    detector.train()
    total_loss = 0.0
    # ===================== 进度条 =====================
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}]", leave=True)

    for batch in pbar:
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs).to(DEVICE)
        batch_size = imgs.shape[0]

        # 对齐target
        max_len = max([t.shape[0] for t in targets])
        target_tensor = torch.zeros(batch_size, max_len, 5, device=DEVICE)
        for i, t in enumerate(targets):
            target_tensor[i, :t.shape[0]] = t.to(DEVICE)

        # 教师网络生成特征
        with torch.no_grad():
            feat = teacher(imgs)

        # 检测头推理
        p3, p4, p5 = detector(feat)
        preds = [p3, p4, p5]

        # 多尺度损失
        loss = 0
        for i, pred in enumerate(preds):
            stride = STRIDES[i]
            anchors = ANCHORS[i]
            gt = build_target(target_tensor, anchors, stride, NUM_CLASSES, IMG_SIZE)
            total_l, box_l, obj_l, cls_l = criterion(pred, gt, batch_size)
            loss += total_l

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # ===================== 实时更新进度条显示损失 =====================
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "total_loss": f"{total_loss:.4f}"})

    log_to_file(f"Epoch [{epoch + 1}/{EPOCHS}] 完成 | 总损失: {total_loss:.4f}")
    
    # ===================== 计算检测指标（从第5轮开始） =====================
    if epoch >= 5:
        # 使用验证集计算检测指标
        val_dataset = MilitaryDataset(root_path, mode='val', img_size=IMG_SIZE)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)
        
        # 获取一个批次的检测结果
        with torch.no_grad():
            for batch in val_loader:
                imgs_val, targets_val = zip(*batch)
                imgs_val = torch.stack(imgs_val).to(DEVICE)
                batch_size_val = imgs_val.shape[0]
                
                # 对齐target
                max_len_val = max([t.shape[0] for t in targets_val])
                target_tensor_val = torch.zeros(batch_size_val, max_len_val, 5, device=DEVICE)
                for i, t in enumerate(targets_val):
                    target_tensor_val[i, :t.shape[0]] = t.to(DEVICE)
                
                # 教师网络生成特征
                feat_val = teacher(imgs_val)
                
                # 检测头推理
                p3_val, p4_val, p5_val = detector(feat_val)
                
                # 解码检测结果
                detections_val = decode_detections([p3_val, p4_val, p5_val], conf_thresh=0.5, nms_thresh=0.4, img_size=IMG_SIZE)
                
                # 计算检测指标
                precision, recall, f1_score = calculate_detection_metrics(detections_val, target_tensor_val)
                
                log_to_file(f"  检测指标 - 精确率: {precision:.3f}, 召回率: {recall:.3f}, F1: {f1_score:.3f}")
                break  # 只计算一个批次
    
    # ===================== 每5个epoch生成可视化图 =====================
    # 可视化保存根目录（与日志同一路径）
    vis_root_dir = os.path.join(os.path.dirname(LOG_ROOT_DIR), "visualizations")
    os.makedirs(vis_root_dir, exist_ok=True)  # 自动创建文件夹，不存在则新建

    # 每5个epoch生成一张可视化图（epoch从1开始计数）
    if (epoch + 1) % 5 == 0:
        # 加载验证集（仅在需要可视化时加载，不影响训练速度）
        val_dataset = MilitaryDataset(root_path, mode='val', img_size=IMG_SIZE)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)

        # 生成带epoch编号的可视化文件名（补零对齐，方便排序）
        vis_save_path = os.path.join(TEACHER_OUTPUT_DIR, f"teacher_feature_epoch_{epoch + 1:03d}.png")

        # 调用可视化函数生成图片
        visualize_teacher_features_and_scales(
            teacher_model=teacher,
            detector_model=detector,
            dataloader=val_loader,
            device=DEVICE,
            save_path=vis_save_path,
            num_samples=4,  # 对应4行输入图，和你提供的示例完全一致
            conf_thresh=0.5, nms_thresh=0.4
        )
        log_to_file(f"✅ Epoch {epoch + 1} 可视化图已保存至: {vis_save_path}")

# 保存模型
# 使用统一的Teacher输出目录
save_dir = TEACHER_OUTPUT_DIR
os.makedirs(save_dir, exist_ok=True)  # 不存在就自动创建

# 保存模型（保存到 Teacher 文件夹）
model_save_path = os.path.join(TEACHER_OUTPUT_DIR, "military_teacher_detector.pth")
torch.save(detector.state_dict(), model_save_path)
log_to_file("="*80)
log_to_file(f"训练完成，模型已保存至：{model_save_path}")
log_to_file(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_to_file("="*80)

# 调用可视化
# 加载验证集用于可视化（避免用训练集，保证可视化样本独立）
val_dataset = MilitaryDataset(root_path, mode='val', img_size=IMG_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)

# 执行可视化（图片也保存到 Teacher 文件夹）
vis_save_path = os.path.join(TEACHER_OUTPUT_DIR, "teacher_feature_visualization.png")
visualize_teacher_features_and_scales(
    teacher_model=teacher,
    detector_model=detector,
    dataloader=val_loader,
    device=DEVICE,
    save_path=vis_save_path,  # 路径已修改
    num_samples=4,
    conf_thresh=0.5, nms_thresh=0.4
)
log_to_file(f"最终可视化图已保存至：{vis_save_path}")
log_to_file(f"日志文件：{LOG_FILE}")