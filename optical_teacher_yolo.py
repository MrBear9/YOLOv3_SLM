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

# 本地实现所有需要的函数，避免依赖optical_teacher.py的配置

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
    """初始化日志文件，使用ConfigYOLO路径"""
    os.makedirs(ConfigYOLO.LOG_ROOT_DIR, exist_ok=True)
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ConfigYOLO.TRAIN_START_TIME = start_time
    with open(ConfigYOLO.LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("光学教师网络训练日志\n")
        f.write("=" * 80 + "\n")
        f.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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
        
        # 重塑预测为 (batch_size, grid_h, grid_w, 3, 5+num_classes)
        pred = pred.permute(0, 2, 3, 1).reshape(batch_size, grid_h, grid_w, 3, -1)
        
        # 应用sigmoid激活函数
        obj_conf = torch.sigmoid(pred[..., 4])  # 目标置信度
        cls_conf = torch.sigmoid(pred[..., 5:])  # 类别置信度
        bbox_pred = pred[..., :4]  # 边界框预测 (tx, ty, tw, th)
        
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
                        
                        # 解码边界框坐标 (YOLO格式)
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
        anchor_set = Config.ANCHORS[i]
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
            detections[b] = apply_nms(detections[b], nms_thresh, max_det)
        else:
            detections[b] = np.zeros((0, 6), dtype=np.float32)

    return detections

decode_detections = decode_detections_fixed

class ConfigYOLO:
    FOCAL_ALPHA = 0.25  # 背景类权重 增大会增加背景类的损失
    FOCAL_GAMMA = 2.0  # 前向传播时的gamma参数 增大会增加分类损失的权重
    AGNOSTIC_NMS = True  # 是否使用agnostic NMS 增加检测框的多样性
    VAL_INTERVAL = 5  # 验证间隔 每5个epoch验证一次模型
    METRIC_IOU_THRESHOLD = 0.5  # 验证指标的IOU阈值
    EPOCH_TABLE_EPOCH_WIDTH = 8  # 训练表格中epoch列的宽度
    EPOCH_TABLE_PHASE_WIDTH = 18  # 训练表格中phase列的宽度
    EPOCH_TABLE_TRAIN_LOSS_WIDTH = 18 
    EPOCH_TABLE_TRAIN_LOSS_WIDTH = 13
    EPOCH_TABLE_VAL_LOSS_WIDTH = 13
    EPOCH_TABLE_METRIC_WIDTH = 11
    EPOCH_TABLE_LR_WIDTH = 12
    EPOCH_TABLE_BEST_WIDTH = 8
    EPOCH_TABLE_BEST_MARK = "Yes"  # 训练表格中best列的标记
    SKIP_FILE_LOG_MESSAGES = (
        "best checkpoint updated",
        "保存最佳模型",
        "saved best model",
    )
    YAML_PATH = r"data\military\data.yaml"
    CLASS_NAMES = None
    NUM_CLASSES = None
    
    TEACHER_OUTPUT_DIR = r"output\OpticalTeacherYOLO"
    LOG_ROOT_DIR = None
    LOG_FILE = None
    TIMESTAMP = None
    TRAIN_START_TIME = None
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 640
    BATCH_SIZE = 8
    EPOCHS = 100
    
    # 动态损失权重配置（先位置后分类策略）
    BOX_WEIGHT_BASE = 0.8     # 基础边界框权重
    OBJ_WEIGHT_BASE = 0.8     # 基础目标性权重
    NOOBJ_WEIGHT_BASE = 0.2   # 基础非目标权重
    CLS_WEIGHT_BASE = 0.4    # 基础分类权重
    
    # 阶段划分参数
    POSITION_PHASE_EPOCHS = 20  # 位置优先阶段轮数
    BALANCE_PHASE_EPOCHS = 10   # 平衡过渡阶段轮数
    
    IOU_THRESHOLD = 0.5    # 目标匹配的IOU阈值（标准YOLOv3设置）
    
    STRIDES = [8, 16, 32] # strides for each feature map
    ANCHORS = [
        [[26,23], [47,49], [100,67]],   # P3: 小目标 / 较小目标
        [[103,169], [203,107], [351,177]],  # P4: 中目标 / 长条目标
        [[241,354], [534,299], [568,528]]  # P5: 大目标 / 超大目标（军舰等）
    ]
    
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-5
    OPTIMIZER = "Adam"
    TEACHER_CHECKPOINT = None
    FREEZE_TEACHER = False
    SAVE_TEACHER_WEIGHTS = True
    
    VIS_INTERVAL = 5 # visualization interval in epochs
    
    CONF_THRESH = 0.5 # confidence threshold
    NMS_THRESH = 0.4 # NMS threshold for post-processing
    MAX_DET = 8 # maximum number of detections to keep
    
    VIS_BATCH_SIZE = 4 # visualization batch size
    VIS_DPI = 120 # visualization DPI

    @classmethod
    def get_dynamic_weights(cls, epoch):
        if epoch < cls.POSITION_PHASE_EPOCHS:
            box_weight = cls.BOX_WEIGHT_BASE * 1.3
            cls_weight = cls.CLS_WEIGHT_BASE * 0.8
            phase = "position_focus"
        elif epoch < cls.POSITION_PHASE_EPOCHS + cls.BALANCE_PHASE_EPOCHS:
            progress = (epoch - cls.POSITION_PHASE_EPOCHS) / max(cls.BALANCE_PHASE_EPOCHS, 1)
            box_weight = cls.BOX_WEIGHT_BASE * (1.3 - 0.3 * progress)
            cls_weight = cls.CLS_WEIGHT_BASE * (0.8 + 0.2 * progress)
            phase = "balance_transition"
        else:
            box_weight = cls.BOX_WEIGHT_BASE
            cls_weight = cls.CLS_WEIGHT_BASE
            phase = "balanced"

        return {
            "box_weight": box_weight,
            "obj_weight": cls.OBJ_WEIGHT_BASE,
            "noobj_weight": cls.NOOBJ_WEIGHT_BASE,
            "cls_weight": cls_weight,
            "phase": phase,
        }

    @classmethod
    def initialize(cls):
        cls.CLASS_NAMES, cls.NUM_CLASSES = load_class_names(cls.YAML_PATH)
        os.makedirs(cls.TEACHER_OUTPUT_DIR, exist_ok=True)
        cls.LOG_ROOT_DIR = os.path.join(cls.TEACHER_OUTPUT_DIR, "logs")
        os.makedirs(cls.LOG_ROOT_DIR, exist_ok=True)
        cls.TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.LOG_FILE = os.path.join(cls.LOG_ROOT_DIR, f"training_log_{cls.TIMESTAMP}.txt")
    
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
        print("Optical Teacher YOLO Training Config")
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

# 记录完整的参数配置到日志文件
def log_all_parameters():
    """记录所有训练参数到日志文件"""
    log_to_file("="*80)
    log_to_file("光学教师YOLO训练参数配置")
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
    
    log_to_file("\n【损失权重配置】")
    log_to_file(f"  基础边界框权重: {Config.BOX_WEIGHT_BASE}")
    log_to_file(f"  基础目标性权重: {Config.OBJ_WEIGHT_BASE}")
    log_to_file(f"  基础非目标权重: {Config.NOOBJ_WEIGHT_BASE}")
    log_to_file(f"  基础分类权重: {Config.CLS_WEIGHT_BASE}")
    log_to_file(f"  位置优先阶段轮数: {Config.POSITION_PHASE_EPOCHS}")
    log_to_file(f"  平衡过渡阶段轮数: {Config.BALANCE_PHASE_EPOCHS}")
    
    log_to_file("\n【锚框设置】")
    log_to_file(f"  步长: {Config.STRIDES}")
    log_to_file(f"  P3锚框(小目标): {Config.ANCHORS[0]}")
    log_to_file(f"  P4锚框(中目标): {Config.ANCHORS[1]}")
    log_to_file(f"  P5锚框(大目标): {Config.ANCHORS[2]}")
    log_to_file(f"  IOU阈值: {Config.IOU_THRESHOLD}")
    
    log_to_file("\n【优化器参数】")
    log_to_file(f"  优化器: {Config.OPTIMIZER}")
    log_to_file(f"  学习率: {Config.LEARNING_RATE}")
    log_to_file(f"  权重衰减: {Config.WEIGHT_DECAY}")
    
    log_to_file("\n【检测参数】")
    log_to_file(f"  置信度阈值: {Config.CONF_THRESH}")
    log_to_file(f"  NMS阈值: {Config.NMS_THRESH}")
    log_to_file(f"  最大检测数量: {Config.MAX_DET}")
    
    log_to_file("\n【可视化参数】")
    log_to_file(f"  可视化间隔: {Config.VIS_INTERVAL} 轮")
    log_to_file(f"  可视化批次大小: {Config.VIS_BATCH_SIZE}")
    log_to_file(f"  可视化DPI: {Config.VIS_DPI}")
    
    log_to_file("\n【路径设置】")
    log_to_file(f"  教师输出目录: {Config.TEACHER_OUTPUT_DIR}")
    log_to_file(f"  日志根目录: {Config.LOG_ROOT_DIR}")
    log_to_file(f"  日志文件: {Config.LOG_FILE}")
    log_to_file(f"  时间戳: {Config.TIMESTAMP}")
    
    # 计算模型参数数量
    teacher = ConvTeacher()
    detector = YOLOLightHead(in_channels=1, out_channels=Config.get_detector_output_channels())
    
    log_to_file("\n【模型参数统计】")
    log_to_file(f"  教师网络可训练参数: {sum(p.numel() for p in teacher.parameters() if p.requires_grad):,}")
    log_to_file(f"  检测头可训练参数: {sum(p.numel() for p in detector.parameters() if p.requires_grad):,}")
    log_to_file(f"  检测头输出通道数: {Config.get_detector_output_channels()}")

    log_to_file(f"Focal alpha: {Config.FOCAL_ALPHA}")
    log_to_file(f"Focal gamma: {Config.FOCAL_GAMMA}")
    log_to_file(f"Agnostic NMS: {Config.AGNOSTIC_NMS}")
    log_to_file(f"Validation interval: {Config.VAL_INTERVAL}")
    
    log_to_file("\n" + "="*80)
    log_to_file("参数配置记录完成")
    log_to_file("="*80 + "\n")

# 在训练开始前记录所有参数
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

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, strides):
        super().__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.num_classes = num_classes
        self.strides = strides
        
        # 使用动态损失权重
        self.box_weight = Config.BOX_WEIGHT_BASE
        self.obj_weight = Config.OBJ_WEIGHT_BASE
        self.noobj_weight = Config.NOOBJ_WEIGHT_BASE
        self.cls_weight = Config.CLS_WEIGHT_BASE
        
        # 使用带reduction的损失函数，避免数值爆炸
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    
    def set_epoch_weights(self, epoch):
        """根据训练轮数设置动态权重"""
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
            
            # 计算各项损失（修复损失值过小问题）
            if obj_mask.sum() > 0:
                # 边界框损失 - 使用正确的损失计算
                box_loss = self.mse_loss(pred_boxes[obj_mask], target_boxes[obj_mask])
                
                # 目标存在性损失
                obj_loss = self.bce_loss(pred_obj[obj_mask], target_obj[obj_mask])
                
                # 类别损失
                cls_loss = self.bce_loss(pred_cls[obj_mask], target_cls[obj_mask])
            else:
                box_loss = torch.tensor(0.0, device=pred_boxes.device)
                obj_loss = torch.tensor(0.0, device=pred_boxes.device)
                cls_loss = torch.tensor(0.0, device=pred_boxes.device)
            
            # 非目标损失
            if noobj_mask.sum() > 0:
                noobj_loss = self.bce_loss(pred_obj[noobj_mask], target_obj[noobj_mask])
            else:
                noobj_loss = torch.tensor(0.0, device=pred_boxes.device)
            
            # 总损失计算（修复权重应用）
            scale_loss = (self.box_weight * box_loss + 
                         self.obj_weight * obj_loss + 
                         self.noobj_weight * noobj_loss + 
                         self.cls_weight * cls_loss)
            
            # 检查损失值是否合理，避免数值爆炸
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
        self.box_weight = Config.BOX_WEIGHT_BASE
        self.obj_weight = Config.OBJ_WEIGHT_BASE
        self.noobj_weight = Config.NOOBJ_WEIGHT_BASE
        self.cls_weight = Config.CLS_WEIGHT_BASE
        self.focal_loss = SigmoidFocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA, reduction="mean")
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
        return weights["phase"]

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
                "grid_h": grid_h,
                "grid_w": grid_w,
                "stride": self.strides[i],
                "anchors": self.anchors[i].to(device)
            })

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
                scale_data["target_boxes_abs"][b, grid_y, grid_x, best_anchor_idx, 0] = tx * Config.IMG_SIZE
                scale_data["target_boxes_abs"][b, grid_y, grid_x, best_anchor_idx, 1] = ty * Config.IMG_SIZE
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
    component_totals = {"total": 0.0, "box": 0.0, "obj": 0.0, "noobj": 0.0, "cls": 0.0}
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            batch_images, batch_targets = prepare_batch(batch, device)
            predictions = model(batch_images)
            loss, loss_stats = criterion(predictions, batch_targets)

            for key in component_totals:
                component_totals[key] += loss_stats[key]

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

def save_training_curves(history, output_dir):
    if len(history["train_total"]) == 0:
        return

    epochs = range(len(history["train_total"]))
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    axes[0].plot(epochs, history["train_total"], label="Train Loss", linewidth=2)
    if len(history["val_total"]) > 0:
        axes[0].plot(epochs, history["val_total"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train / Val Loss")
    axes[0].grid(True)
    axes[0].legend()

    if len(history["precision"]) > 0:
        axes[1].plot(epochs, history["precision"], label="Precision", linewidth=2)
        axes[1].plot(epochs, history["recall"], label="Recall", linewidth=2)
        axes[1].plot(epochs, history["f1"], label="F1", linewidth=2)
        axes[1].plot(epochs, history["map50"], label="mAP@0.5", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].set_title("Validation Metrics")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=Config.VIS_DPI)
    plt.close()

YOLOLoss = EnhancedYOLOLoss

def save_detection_visualization(epoch, model, dataset, save_dir, prefix="train", device=None):
    """
    保存检测结果可视化图像
    
    参数:
        epoch: 当前训练轮数
        model: 训练好的模型
        dataset: 数据集对象
        save_dir: 保存目录
        prefix: 文件名前缀，默认为"train"
        device: 计算设备，默认为Config.DEVICE
    """
    if device is None:
        device = Config.DEVICE
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置模型为评估模式
    model.eval()
    
    # 随机选择样本进行可视化
    num_samples = min(Config.VIS_BATCH_SIZE, len(dataset))
    indices = torch.randperm(len(dataset))[:num_samples]
    
    # 创建子图布局：4列（原图、教师特征、真实标签、预测结果）
    fig, axes = plt.subplots(num_samples, 4, figsize=(24, 6 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # 禁用梯度计算以节省内存
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # 获取图像和标签
            img_tensor, targets = dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)  # 添加批次维度
            
            # 将张量转换为numpy数组用于显示
            img_np = img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)  # 灰度转RGB
            
            # 获取教师网络的特征图
            teacher_output = model.teacher(img_tensor)
            teacher_output = teacher_output.squeeze(0).cpu().numpy()
            teacher_output = enhance_feature_for_display(teacher_output)  # 增强特征显示
            
            # 确保教师特征输出为2D以便imshow显示
            if teacher_output.ndim == 3:
                teacher_output = teacher_output.squeeze(0)
            
            # 获取模型预测结果
            pred_p3, pred_p4, pred_p5 = model(img_tensor)
            preds = [pred_p3, pred_p4, pred_p5]
            detections = decode_detections(preds)  # 解码检测结果
            
            # 第一列：显示原始图像
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"Original Image {i+1}")
            axes[i, 0].axis("off")
            
            # 第二列：显示教师网络特征图
            axes[i, 1].imshow(teacher_output, cmap="hot")
            axes[i, 1].set_title(f"Teacher Feature {i+1}")
            axes[i, 1].axis("off")
            
            # 第三列：显示真实标签（绿色边界框）
            axes[i, 2].imshow(img_np)
            axes[i, 2].set_title(f"Ground Truth {i+1}")
            axes[i, 2].axis("off")
            
            # 绘制真实边界框
            for target_idx in range(len(targets)):
                cls_id, x_center, y_center, width, height = targets[target_idx]
                cls_id = int(cls_id.item())
                
                # 计算边界框坐标（YOLO格式转像素坐标）
                x1 = int((x_center - width / 2) * Config.IMG_SIZE)
                y1 = int((y_center - height / 2) * Config.IMG_SIZE)
                w = int(width * Config.IMG_SIZE)
                h = int(height * Config.IMG_SIZE)
                
                # 绘制绿色边界框
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                        edgecolor='green', facecolor='none')
                axes[i, 2].add_patch(rect)
                # 添加类别标签
                axes[i, 2].text(x1, y1 - 5, Config.CLASS_NAMES[cls_id], 
                              color='green', fontsize=10, fontweight='bold')
            
            # 第四列：显示预测结果（彩色边界框）
            axes[i, 3].imshow(img_np)
            axes[i, 3].set_title(f"Predictions {i+1}")
            axes[i, 3].axis("off")
            
            # 绘制预测边界框
            if len(detections[0]) > 0:
                for det in detections[0]:
                    x_center, y_center, w, h, conf, cls_id = det
                    cls_id = int(cls_id)
                    
                    # 计算边界框坐标
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    w = int(w)
                    h = int(h)
                    
                    # 根据类别ID选择颜色
                    color = plt.cm.tab20(cls_id / max(Config.NUM_CLASSES, 1))
                    # 绘制彩色边界框
                    rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                            edgecolor=color, facecolor='none')
                    axes[i, 3].add_patch(rect)
                    
                    # 添加类别标签和置信度
                    label = f"{Config.CLASS_NAMES[cls_id]}: {conf:.2f}"
                    axes[i, 3].text(x1, y1 - 5, label, 
                                  color=color, fontsize=10, fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.3', 
                                          facecolor=color, alpha=0.7))
    
    # 调整布局并保存图像
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_epoch_{epoch:03d}.png")
    plt.savefig(save_path, dpi=Config.VIS_DPI)
    plt.close()  # 关闭图像以释放内存

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

def train():
    device = Config.DEVICE
    log_to_file(f"使用设备: {device}")
    
    log_to_file("初始化教师网络（ConvTeacher）...")
    teacher = ConvTeacher()
    loaded_teacher, teacher_message = load_teacher_checkpoint_safe(teacher, Config.TEACHER_CHECKPOINT, device)
    log_to_file(teacher_message)

    freeze_teacher = Config.FREEZE_TEACHER and loaded_teacher
    if Config.FREEZE_TEACHER and not loaded_teacher:
        log_to_file("Teacher checkpoint not found, not freezing teacher.")

    for p in teacher.parameters():
        p.requires_grad = not freeze_teacher
    log_to_file(f"Teacher status: {'frozen' if freeze_teacher else 'trainable'}")
    log_to_file("Teacher parameter setup applied") # 应用教师参数设置
    
    log_to_file("初始化检测头（YOLOLightHead）...")
    detector = YOLOLightHead(in_channels=1, 
                           out_channels=Config.get_detector_output_channels())
    
    log_to_file("构建教师网络+检测头模型...")
    model = TeacherWithDetector(teacher=teacher, detector=detector).to(device)
    
    for p in model.teacher.parameters():
        p.requires_grad = not freeze_teacher
    for p in model.detector.parameters():
        p.requires_grad = True
    
    log_to_file("加载数据集...")
    train_dataset = YOLODataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                             shuffle=True, collate_fn=lambda x: x)
    log_to_file(f"训练集大小: {len(train_dataset)}")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params,
                                lr=Config.LEARNING_RATE,
                                weight_decay=Config.WEIGHT_DECAY)
    
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
    
    log_to_file("="*60)
    log_to_file("开始训练检测头...")
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
        
        # 根据训练阶段动态调整损失权重
        phase = criterion.set_epoch_weights(epoch)
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [{phase}]", leave=True):
            batch_images, batch_targets = prepare_batch(batch, device)
            
            optimizer.zero_grad()
            
            predictions = model(batch_images)
            loss, loss_stats = criterion(predictions, batch_targets)
            
            loss.backward()
            optimizer.step()
            
            for key in train_component_sums:
                train_component_sums[key] += loss_stats[key]

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
            log_to_file(f"Epoch {epoch}: 保存最佳模型 (Loss: {best_loss:.6f})", also_print=False)
        
        if epoch % Config.VIS_INTERVAL == 0:
            save_detection_visualization(epoch, model, train_dataset, vis_dir, prefix="train", device=device)

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
    log_to_file("训练完成！")
    log_to_file(f"最佳模型已保存到: {os.path.join(Config.TEACHER_OUTPUT_DIR, 'detector_best.pth')}")
    log_to_file(f"最终模型已保存到: {model_save_path}")
    log_to_file(f"所有结果已保存到: {Config.TEACHER_OUTPUT_DIR}")
    log_to_file("="*60)

if __name__ == "__main__":
    train()
