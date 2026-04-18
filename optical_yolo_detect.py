
"""
光学YOLO目标检测系统 - Optical YOLO Detection System
==========================================================

功能概述:
本系统实现了一个基于光学图像处理和YOLO目标检测的完整检测流程，主要功能包括：

1. 图像预处理阶段:
   - 使用光学图像处理技术提取亮光区域
   - 自动调整图像尺寸以适应检测模型输入要求
   - 灰度化处理以适配单通道检测模型

2. 目标检测阶段:
   - 加载预训练的光学YOLO检测器权重
   - 实现多尺度特征金字塔检测架构
   - 支持军事目标检测（坦克、战机、军舰等）

3. 结果后处理:
   - 非极大值抑制(NMS)去除重叠检测框
   - 按类别绘制彩色边界框和置信度标签
   - 自动保存检测结果到指定目录

4. 运行模式:
   - 单张图像检测模式：处理指定单张图像
   - 批量检测模式：处理指定目录下所有图像

文件结构说明:
- image_Origin/: 原始输入图像（手动放置的测试图像）
- image_input/: 探测器采集的图像（经过光学传播后的图像）
- imageProcess/: 处理后的图像（亮光区域提取+尺寸调整）
- imageDetect/: 最终检测结果图像（带边界框和标签）

作者: 光学检测系统开发团队  Mr. Bear
版本: 1.0
创建日期: 2026-04-17
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from torchvision.ops import nms
import yaml
import sys

# 添加父目录到路径，以便导入相关模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入图像处理模块
from Optical_yolo_detect.imageProcess import extract_and_resize_light_area

# =========================================================
# 路径配置类 - 输入输出路径管理
# =========================================================
class PathConfig:
    """路径配置类，管理所有输入输出路径"""
    
    # 基础目录
    BASE_DIR = "Optical_yolo_detect"
    
    # 输入路径
    IMAGE_ORIGIN_DIR = os.path.join(BASE_DIR, "image_Origin")      # 原始输入图像
    IMAGE_INPUT_DIR = os.path.join(BASE_DIR, "image_input")        # 探测器采集图像
    
    # 输出路径
    IMAGE_PROCESS_DIR = os.path.join(BASE_DIR, "imageProcess")     # 处理后的图像
    IMAGE_DETECT_DIR = os.path.join(BASE_DIR, "imageDetect")       # 检测结果图像
    
    # 模型路径
    MODEL_PATH = r"output\OpticalTeacherYOLO\detector_best.pth"
    
    # 测试图像路径（单张图像测试时使用）
    TEST_IMAGE_PATH = os.path.join(IMAGE_INPUT_DIR, "2026-02-06_16-18-00_454.jpg")
    
    @classmethod
    def create_directories(cls):
        """创建所有必要的目录"""
        directories = [
            cls.IMAGE_ORIGIN_DIR,
            cls.IMAGE_INPUT_DIR, 
            cls.IMAGE_PROCESS_DIR,
            cls.IMAGE_DETECT_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"目录已创建/确认存在: {directory}")

# =========================================================
# 检测配置类 - 与训练时保持一致
# =========================================================
class ConfigYOLO:
    """YOLO检测器配置类，包含模型参数和检测参数"""
    
    YAML_PATH = r"data\military\data.yaml"
    CLASS_NAMES = None
    NUM_CLASSES = None
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 640
    
    # 检测参数
    CONF_THRESH = 0.5
    NMS_THRESH = 0.4
    MAX_DET = 8
    AGNOSTIC_NMS = True
    
    # 预测框偏移参数（补偿探测器采集图像时的位置偏移）
    BOX_OFFSET_X = 0.0  # 横向偏移量（像素）
    BOX_OFFSET_Y = 0.0  # 纵向偏移量（像素）
    
    # 锚框设置
    STRIDES = [8, 16, 32]
    ANCHORS = [
        # P3: 小目标（士兵、小型装备）
        [[26, 23], [47, 49], [100, 67]],
        # P4: 中目标（坦克、战机主体）
        [[103, 169], [203, 107], [351, 177]],
        # P5: 大目标（军舰、大型战机）
        [[241, 354], [534, 299], [568, 528]]
    ]
    
    @classmethod
    def initialize(cls):
        cls.CLASS_NAMES, cls.NUM_CLASSES = load_class_names(cls.YAML_PATH)
    
    @classmethod
    def get_detector_output_channels(cls):
        return 3 * (4 + 1 + cls.NUM_CLASSES)

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

def xywh_to_xyxy(boxes):
    """将YOLO格式的边界框(x_center, y_center, w, h)转换为(x1, y1, x2, y2)格式"""
    half_w = boxes[:, 2] / 2
    half_h = boxes[:, 3] / 2
    return torch.stack([
        boxes[:, 0] - half_w,
        boxes[:, 1] - half_h,
        boxes[:, 0] + half_w,
        boxes[:, 1] + half_h
    ], dim=1)

def apply_nms(detections, nms_thresh, max_det, class_agnostic=None):
    if len(detections) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    if isinstance(detections, list):
        detections = np.array(detections, dtype=np.float32)

    det_tensor = torch.as_tensor(detections, dtype=torch.float32)
    if det_tensor.numel() == 0:
        return np.zeros((0, 6), dtype=np.float32)

    if class_agnostic is None:
        class_agnostic = ConfigYOLO.AGNOSTIC_NMS

    boxes_xyxy = xywh_to_xyxy(det_tensor[:, :4])
    scores = det_tensor[:, 4]
    class_ids = det_tensor[:, 5].long()

    if class_agnostic:
        keep_indices = nms(boxes_xyxy, scores, nms_thresh)
        det_tensor = det_tensor[keep_indices]
    else:
        kept = []
        for cls_id in class_ids.unique(sorted=False):
            cls_mask = class_ids == cls_id
            if cls_mask.sum() == 0:
                continue
            keep_indices = nms(boxes_xyxy[cls_mask], scores[cls_mask], nms_thresh)
            kept.append(det_tensor[cls_mask][keep_indices])

        if len(kept) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        det_tensor = torch.cat(kept, dim=0)

    det_tensor = det_tensor[det_tensor[:, 4].argsort(descending=True)]
    return det_tensor[:max_det].cpu().numpy()

def apply_classwise_nms(detections, nms_thresh, max_det):
    return apply_nms(detections, nms_thresh, max_det, class_agnostic=False)
    """按类别应用NMS"""
    if len(detections) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # 确保detections是numpy数组
    if isinstance(detections, list):
        detections = np.array(detections, dtype=np.float32)
    
    det_tensor = torch.as_tensor(detections, dtype=torch.float32)
    
    # 检查是否有有效的检测结果
    if det_tensor.numel() == 0:
        return np.zeros((0, 6), dtype=np.float32)
    
    boxes_xyxy = xywh_to_xyxy(det_tensor[:, :4])
    scores = det_tensor[:, 4]
    class_ids = det_tensor[:, 5].long()  # 确保类别ID为整数类型
    kept = []

    for cls_id in class_ids.unique(sorted=False):
        cls_mask = class_ids == cls_id
        if cls_mask.sum() == 0:
            continue
        
        # 确保有足够的检测框进行NMS
        if cls_mask.sum() > 1:
            keep_indices = torch.ops.torchvision.nms(boxes_xyxy[cls_mask], scores[cls_mask], nms_thresh)
        else:
            keep_indices = torch.tensor([0], dtype=torch.long)
        
        kept.append(det_tensor[cls_mask][keep_indices])

    if len(kept) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    det_tensor = torch.cat(kept, dim=0)
    det_tensor = det_tensor[det_tensor[:, 4].argsort(descending=True)]
    
    # 确保返回正确的numpy数组格式
    if len(det_tensor) > 0:
        return det_tensor[:max_det].cpu().numpy()
    else:
        return np.zeros((0, 6), dtype=np.float32)

def decode_detections(preds, conf_thresh=None, nms_thresh=None, max_det=None, img_size=None):
    """解码YOLO检测结果"""
    if conf_thresh is None:
        conf_thresh = ConfigYOLO.CONF_THRESH
    if nms_thresh is None:
        nms_thresh = ConfigYOLO.NMS_THRESH
    if max_det is None:
        max_det = ConfigYOLO.MAX_DET
    if img_size is None:
        img_size = ConfigYOLO.IMG_SIZE

    batch_size = preds[0].shape[0]
    detections = [[] for _ in range(batch_size)]

    for i, pred in enumerate(preds):
        grid_h, grid_w = pred.shape[2], pred.shape[3]
        stride = ConfigYOLO.STRIDES[i]
        anchor_set = ConfigYOLO.ANCHORS[i]
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

def draw_detections(image, detections, class_names, conf_thresh=0.5, offset_x=None, offset_y=None):
    """在图像上绘制检测结果"""
    img_with_boxes = image.copy()
    
    # 使用配置中的偏移量，如果未指定参数
    if offset_x is None:
        offset_x = ConfigYOLO.BOX_OFFSET_X
    if offset_y is None:
        offset_y = ConfigYOLO.BOX_OFFSET_Y
    
    # 定义颜色映射
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色  
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 品红
        (0, 255, 255),  # 黄色
        (128, 0, 0),    # 深红
        (0, 128, 0),    # 深绿
        (0, 0, 128),    # 深蓝
    ]
    
    for det in detections:
        if len(det) < 6:
            continue
            
        x_center, y_center, w, h, conf, cls_id = det
        if conf < conf_thresh:
            continue
            
        # 确保cls_id为整数类型
        cls_id = int(cls_id)
        
        # 应用偏移量
        x_center_offset = x_center + offset_x
        y_center_offset = y_center + offset_y
            
        # 转换为边界框坐标
        x1 = int(x_center_offset - w/2)
        y1 = int(y_center_offset - h/2)
        x2 = int(x_center_offset + w/2)
        y2 = int(y_center_offset + h/2)
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, image.shape[1]-1))
        y1 = max(0, min(y1, image.shape[0]-1))
        x2 = max(0, min(x2, image.shape[1]-1))
        y2 = max(0, min(y2, image.shape[0]-1))
        
        # 选择颜色
        color = colors[cls_id % len(colors)]
        
        # 绘制边界框
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label = f"{class_names.get(int(cls_id), 'unknown')}: {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # 标签背景
        cv2.rectangle(img_with_boxes, 
                     (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1),
                     color, -1)
        
        # 标签文字
        cv2.putText(img_with_boxes, label,
                   (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_with_boxes

# =========================================================
# 主检测类
# =========================================================
class OpticalYOLODetector:
    def __init__(self, model_path, yaml_path=None, offset_x=None, offset_y=None):
        """初始化检测器"""
        if yaml_path is None:
            yaml_path = ConfigYOLO.YAML_PATH
            
        # 初始化配置
        ConfigYOLO.initialize()
        
        # 设置偏移参数
        self.offset_x = offset_x if offset_x is not None else ConfigYOLO.BOX_OFFSET_X
        self.offset_y = offset_y if offset_y is not None else ConfigYOLO.BOX_OFFSET_Y
        
        # 创建检测器模型
        out_channels = ConfigYOLO.get_detector_output_channels()
        self.detector = YOLOLightHead(in_channels=1, out_channels=out_channels)
        
        # 加载权重
        self.device = ConfigYOLO.DEVICE
        self.load_model(model_path)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((ConfigYOLO.IMG_SIZE, ConfigYOLO.IMG_SIZE)),
            transforms.Grayscale(1),
            transforms.ToTensor()
        ])
        
        print(f"检测器初始化完成，设备: {self.device}")
        print(f"类别数量: {ConfigYOLO.NUM_CLASSES}")
        print(f"类别名称: {ConfigYOLO.CLASS_NAMES}")
        print(f"预测框偏移量: X={self.offset_x}, Y={self.offset_y}")
    
    def load_model(self, model_path):
        """加载训练好的模型权重"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 尝试不同的键名
        state_dict = None
        for key in ['detector_state_dict', 'model_state_dict', 'state_dict']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        
        if state_dict is None:
            # 如果没有找到标准键名，尝试直接使用整个checkpoint
            state_dict = checkpoint
        
        # 加载权重
        self.detector.load_state_dict(state_dict, strict=False)
        self.detector.to(self.device)
        self.detector.eval()
        
        print(f"模型权重加载成功: {model_path}")
        print(f"加载的参数数量: {len(state_dict)}")
    
    def preprocess_image(self, image_path):
        """预处理图像：先进行光学图像处理，然后转换为模型输入格式"""
        # 使用imageProcess.py中的函数处理图像
        processed_img, output_path = extract_and_resize_light_area(
            image_path, 
            output_dir="Optical_yolo_detect/imageProcess",
            target_size=(ConfigYOLO.IMG_SIZE, ConfigYOLO.IMG_SIZE)
        )
        
        # 转换为PIL图像进行后续处理
        pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        
        # 转换为模型输入格式
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        return input_tensor, processed_img, output_path
    
    def detect(self, image_path, conf_thresh=None, save_detection=True, offset_x=None, offset_y=None):
        """对单张图像进行检测"""
        if conf_thresh is None:
            conf_thresh = ConfigYOLO.CONF_THRESH
            
        # 使用自定义偏移参数或默认参数
        offset_x = offset_x if offset_x is not None else self.offset_x
        offset_y = offset_y if offset_y is not None else self.offset_y
            
        # 预处理图像：使用探测器采集的图像（image_input）进行检测
        input_tensor, processed_img, processed_path = self.preprocess_image(image_path)
        
        # 进行检测
        with torch.no_grad():
            preds = self.detector(input_tensor)
            detections = decode_detections(preds, conf_thresh=conf_thresh)
        
        # 处理检测结果
        if len(detections) > 0:
            detections = detections[0]  # 单张图像，取第一个batch
        else:
            detections = np.array([])
        
        # 在处理后的图像（640x640）上绘制检测结果，应用偏移量
        result_img = draw_detections(processed_img, detections, ConfigYOLO.CLASS_NAMES, conf_thresh, offset_x, offset_y)
        
        # 保存检测结果
        if save_detection:
            # 确保输出目录存在
            os.makedirs(PathConfig.IMAGE_DETECT_DIR, exist_ok=True)
            
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(PathConfig.IMAGE_DETECT_DIR, f"{base_name}_detected.png")
            
            # 保存图像
            cv2.imwrite(output_path, result_img)
            print(f"检测结果已保存: {output_path}")
            print(f"应用偏移量: X={offset_x}, Y={offset_y}")
        
        return detections, result_img, processed_path
    
    def _get_original_image_path(self, detector_image_path):
        """根据探测器采集图像路径获取对应的原始图像路径"""
        # 获取图像文件名
        image_name = os.path.basename(detector_image_path)
        
        # 构建原始图像路径
        original_path = os.path.join(PathConfig.IMAGE_ORIGIN_DIR, image_name)
        
        # 检查原始图像是否存在
        if not os.path.exists(original_path):
            # 如果原始图像不存在，使用探测器图像作为备选
            print(f"警告: 未找到原始图像 {original_path}，将使用探测器图像进行检测")
            return detector_image_path
        
        return original_path
    
    def _resize_detections_to_original(self, detections, original_shape):
        """将检测框坐标从模型尺寸(640x640)调整到原始图像尺寸"""
        if len(detections) == 0:
            return detections
        
        # 原始图像尺寸
        orig_h, orig_w = original_shape
        
        # 模型输入尺寸
        model_h, model_w = ConfigYOLO.IMG_SIZE, ConfigYOLO.IMG_SIZE
        
        # 计算缩放比例
        scale_x = orig_w / model_w
        scale_y = orig_h / model_h
        
        # 调整检测框坐标
        detections_resized = detections.copy()
        for i, det in enumerate(detections_resized):
            if len(det) >= 4:
                # 调整中心点坐标和宽高
                det[0] = det[0] * scale_x  # x_center
                det[1] = det[1] * scale_y  # y_center
                det[2] = det[2] * scale_x  # width
                det[3] = det[3] * scale_y  # height
        
        return detections_resized
    
    def detect_batch(self, image_dir, conf_thresh=None, offset_x=None, offset_y=None):
        """对目录中的多张图像进行批量检测"""
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"图像目录不存在: {image_dir}")
            
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) == 0:
            print(f"在目录 {image_dir} 中未找到图像文件")
            return []
        
        # 使用自定义偏移参数或默认参数
        offset_x = offset_x if offset_x is not None else self.offset_x
        offset_y = offset_y if offset_y is not None else self.offset_y
        
        print(f"批量检测应用偏移量: X={offset_x}, Y={offset_y}")
        
        results = []
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            print(f"处理图像: {image_file}")
            
            try:
                detections, result_img, processed_path = self.detect(image_path, conf_thresh, offset_x=offset_x, offset_y=offset_y)
                results.append({
                    'image_path': image_path,
                    'processed_path': processed_path,
                    'detections': detections,
                    'result_image': result_img
                })
                
                # 打印检测结果
                if len(detections) > 0:
                    print(f"  检测到 {len(detections)} 个目标:")
                    for i, det in enumerate(detections):
                        cls_name = ConfigYOLO.CLASS_NAMES.get(int(det[5]), 'unknown')
                        print(f"    {i+1}. {cls_name}: 置信度 {det[4]:.3f}")
                else:
                    print("  未检测到目标")
                    
            except Exception as e:
                print(f"处理图像 {image_file} 时出错: {e}")
                continue
        
        return results

# =========================================================
# 检测模式选择类
# =========================================================
class DetectionMode:
    """检测模式选择类，支持单张图像和批量检测模式"""
    
    SINGLE_IMAGE = "single"      # 单张图像检测模式
    BATCH_IMAGES = "batch"       # 批量图像检测模式
    
    @classmethod
    def get_user_choice(cls):
        """获取用户选择的检测模式"""
        print("\n" + "="*60)
        print("光学YOLO目标检测系统")
        print("="*60)
        print("请选择检测模式:")
        print("1. 单张图像检测 (处理指定单张图像)")
        print("2. 批量图像检测 (处理指定目录下所有图像)")
        print("="*60)
        
        while True:
            choice = input("请输入选择 (1 或 2): ").strip()
            if choice == "1":
                return cls.SINGLE_IMAGE
            elif choice == "2":
                return cls.BATCH_IMAGES
            else:
                print("无效选择，请输入 1 或 2")
    
    @classmethod
    def get_single_image_path(cls):
        """获取单张图像检测的路径"""
        print(f"\n默认测试图像路径: {PathConfig.TEST_IMAGE_PATH}")
        
        if os.path.exists(PathConfig.TEST_IMAGE_PATH):
            use_default = input("是否使用默认测试图像? (y/n): ").strip().lower()
            if use_default == 'y' or use_default == '':
                return PathConfig.TEST_IMAGE_PATH
        
        while True:
            custom_path = input("请输入图像文件路径: ").strip()
            if os.path.exists(custom_path):
                return custom_path
            else:
                print(f"文件不存在: {custom_path}")
                print("请检查路径是否正确")
    
    @classmethod
    def get_batch_image_dir(cls):
        """获取批量检测的目录路径"""
        print(f"\n默认图像目录: {PathConfig.IMAGE_INPUT_DIR}")
        
        if os.path.exists(PathConfig.IMAGE_INPUT_DIR):
            use_default = input("是否使用默认图像目录? (y/n): ").strip().lower()
            if use_default == 'y' or use_default == '':
                return PathConfig.IMAGE_INPUT_DIR
        
        while True:
            custom_dir = input("请输入图像目录路径: ").strip()
            if os.path.exists(custom_dir):
                return custom_dir
            else:
                print(f"目录不存在: {custom_dir}")
                print("请检查路径是否正确")

# =========================================================
# 主函数
# =========================================================
def main():
    """主函数 - 光学YOLO目标检测系统入口"""
    
    # 创建必要的目录
    PathConfig.create_directories()
    
    # 初始化配置
    ConfigYOLO.initialize()
    
    # 创建检测器
    try:
        detector = OpticalYOLODetector(PathConfig.MODEL_PATH)
        print(f"检测器初始化成功，设备: {ConfigYOLO.DEVICE}")
    except Exception as e:
        print(f"检测器初始化失败: {e}")
        return
    
    # 选择检测模式
    mode = DetectionMode.get_user_choice()
    
    if mode == DetectionMode.SINGLE_IMAGE:
        # 单张图像检测模式
        image_path = DetectionMode.get_single_image_path()
        
        print(f"\n开始单张图像检测:")
        print(f"图像路径: {image_path}")
        
        try:
            detections, result_img, processed_path = detector.detect(image_path)
            
            # 显示检测结果
            print("\n" + "="*40)
            print("检测结果:")
            print("="*40)
            
            if len(detections) > 0:
                print(f"检测完成！共检测到 {len(detections)} 个目标:")
                for i, det in enumerate(detections):
                    cls_name = ConfigYOLO.CLASS_NAMES.get(int(det[5]), 'unknown')
                    print(f"  {i+1}. {cls_name}: 置信度 {det[4]:.3f}")
                print(f"处理后的图像已保存到: {processed_path}")
            else:
                print("未检测到目标")
                
        except Exception as e:
            print(f"检测过程中出错: {e}")
    
    else:
        # 批量图像检测模式
        image_dir = DetectionMode.get_batch_image_dir()
        
        print(f"\n开始批量图像检测:")
        print(f"图像目录: {image_dir}")
        
        try:
            results = detector.detect_batch(image_dir)
            
            # 显示批量检测结果统计
            print("\n" + "="*50)
            print("批量检测结果统计:")
            print("="*50)
            
            total_images = len(results)
            total_detections = sum(len(result['detections']) for result in results)
            images_with_detections = sum(1 for result in results if len(result['detections']) > 0)
            
            print(f"处理图像总数: {total_images}")
            print(f"检测到目标的图像数: {images_with_detections}")
            print(f"总检测目标数: {total_detections}")
            print(f"检测结果已保存到: {PathConfig.IMAGE_DETECT_DIR}")
            
            # 显示每张图像的检测结果
            if total_detections > 0:
                print("\n详细检测结果:")
                for i, result in enumerate(results):
                    if len(result['detections']) > 0:
                        image_name = os.path.basename(result['image_path'])
                        print(f"  {image_name}: {len(result['detections'])} 个目标")
                        
        except Exception as e:
            print(f"批量检测过程中出错: {e}")
    
    print("\n检测任务完成！")

if __name__ == "__main__":
    main()
