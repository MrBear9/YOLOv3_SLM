﻿import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import time
from datetime import datetime

# 导入光学YOLOv3模型
from Optical_class import OpticalYOLOv3, YOLOLoss, build_target

# =========================================================
# 配置参数
# =========================================================
class Config:
    # 设备设置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 训练参数
    EPOCHS = 100
    BATCH_SIZE = 8
    IMG_SIZE = 640
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    
    # 损失权重
    BOX_WEIGHT = 0.05
    OBJ_WEIGHT = 1.5
    CLS_WEIGHT = 0.15
    OPTICAL_CONSTRAINT_WEIGHT = 0.1  # 光学约束损失权重
    USE_VORTEX_INIT = True  # 是否使用涡旋初始化
    SLM1_VORTEX_CHARGE = 1  # SLM1涡旋电荷
    SLM2_VORTEX_CHARGE = -1  # SLM2涡旋电荷
    VORTEX_PERTURBATION = 0.12  # 涡旋相位上的小随机扰动，避免先验过硬 %
    
    # 锚框设置（针对军事目标优化）
    STRIDES = [8, 16, 32]
    ANCHORS = [
        [[10,13], [16,30], [33,23]],   # P3: 小目标（军人）
        [[30,61], [62,45], [59,119]],  # P4: 中目标（坦克、战机）
        [[116,90], [156,198], [373,326]] # P5: 大目标（军舰）
    ]
    
    # 路径设置
    DATA_YAML_PATH = r"data\military\data.yaml"
    ROOT_PATH = r"data\military"
    # 统一输出路径配置
    OPTICAL_YOLO_OUTPUT_DIR = r"output\OpticalYOLO"
    SAVE_DIR = os.path.join(OPTICAL_YOLO_OUTPUT_DIR, "models")
    LOG_DIR = os.path.join(OPTICAL_YOLO_OUTPUT_DIR, "logs")
    VISUALIZATION_DIR = os.path.join(OPTICAL_YOLO_OUTPUT_DIR, "visualizations")
    
    # 训练策略
    ENABLE_NORM_AFTER_EPOCH = 0  # 从训练开始就保持光学输出尺度稳定
    ENABLE_CONSTRAINT_AFTER_EPOCH = 10  # 检测头先学基础表征，再逐步加入光学约束
    CONSTRAINT_WARMUP_EPOCHS = 10  # 光学约束线性预热，避免验证损失突然抬升
    VISUALIZE_EVERY = 5  # 每5轮可视化一次
    SAVE_EVERY = 10  # 每10轮保存一次模型
    EARLY_STOPPING_PATIENCE = 12  # 早停耐心轮数，12轮没有改进则早停
    EARLY_STOPPING_MIN_DELTA = 1e-3 # 最小损失变化阈值，用于早停
    
    # 检测阈值
    CONF_THRESH = 0.5  # 置信度阈值 # 平衡精度和召回率
    NMS_THRESH = 0.4   # 非极大值抑制阈值 # 适中的去重强度
    VIS_CONF_THRESH = 0.6  # 可视化时使用更严格的阈值，避免框和标签堆叠，只显示高质量的检测结果
    VIS_MAX_DETECTIONS = 8  # 可视化时最多显示8个检测框
    
    # 批次选择设置
    VISUALIZE_BATCH_INDEX = -1  # 可视化批次索引，-1表示随机选择，0-N表示指定批次

# =========================================================
# 数据集类
# =========================================================
class MilitaryDataset(Dataset):
    def __init__(self, root_dir, mode='train', img_size=640):
        self.img_size = img_size
        self.mode = mode
        
        # 构建路径
        self.img_dir = os.path.join(root_dir, mode, 'images')
        self.label_dir = os.path.join(root_dir, mode, 'labels')
        
        # 检查路径是否存在
        if not os.path.exists(self.img_dir):
            raise ValueError(f"图像目录不存在: {self.img_dir}")
        
        # 获取图像列表
        self.images = [f for f in os.listdir(self.img_dir) 
                      if f.lower().endswith(('jpg', 'png', 'jpeg'))]
        
        if len(self.images) == 0:
            raise ValueError(f"在 {self.img_dir} 中没有找到图像文件")
        
        print(f"加载 {mode} 数据集: {len(self.images)} 张图像")
        
        # 数据增强（仅训练时使用）
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # 可以添加更多数据增强
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 加载图像
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法加载图像: {img_path}")
        
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        
        # 加载标签
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)
        
        targets = torch.zeros((0, 5))
        if os.path.exists(label_path):
            try:
                labels = np.loadtxt(label_path).reshape(-1, 5)
                targets = torch.tensor(labels, dtype=torch.float32)
            except Exception as e:
                # 空标签文件或格式错误
                pass
        
        return img, targets

# =========================================================
# 训练器类
# =========================================================
class OpticalYOLOv3Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        # 创建输出目录
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)
        
        # 加载类别信息
        self.class_names, self.num_classes = self.load_class_names()
        print(f"类别数量: {self.num_classes}, 类别名称: {self.class_names}")
        
        # 初始化模型
        self.model = self.init_model()
        self.criterion = YOLOLoss(config.BOX_WEIGHT, config.OBJ_WEIGHT, config.CLS_WEIGHT)
        
        # 光学约束损失函数
        self.optical_constraint_criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.no_improve_epochs = 0
        
        # 检测指标
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        
        print(f"训练器初始化完成，使用设备: {self.device}")

    def load_class_names(self):
        """从YAML文件加载类别信息"""
        with open(self.config.DATA_YAML_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        names = config.get('names', [])
        if isinstance(names, list):
            class_names = {i: name for i, name in enumerate(names)}
        else:
            raise ValueError(f"不支持的names格式: {type(names)}")
        
        num_classes = len(class_names)
        return class_names, num_classes

    def init_model(self):
        """初始化光学YOLOv3模型"""
        model = OpticalYOLOv3(
            num_classes=self.num_classes,
            img_size=self.config.IMG_SIZE,
            optical_mode="phase",  # 使用纯相位调制
            enable_constraint=False,  # 约束损失按训练策略延后启用
            slm1_vortex_charge=self.config.SLM1_VORTEX_CHARGE if self.config.USE_VORTEX_INIT else 0,
            slm2_vortex_charge=self.config.SLM2_VORTEX_CHARGE if self.config.USE_VORTEX_INIT else 0,
            vortex_perturbation=self.config.VORTEX_PERTURBATION
        ).to(self.device)
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params:,}")
        if self.config.USE_VORTEX_INIT:
            print(
                f"已启用涡旋相位初始化: "
                f"SLM1 charge={self.config.SLM1_VORTEX_CHARGE}, "
                f"SLM2 charge={self.config.SLM2_VORTEX_CHARGE}, "
                f"perturbation={self.config.VORTEX_PERTURBATION}"
            )
        
        return model

    def create_dataloaders(self):
        """创建训练和验证数据加载器"""
        try:
            train_dataset = MilitaryDataset(self.config.ROOT_PATH, 'train', self.config.IMG_SIZE)
            val_dataset = MilitaryDataset(self.config.ROOT_PATH, 'val', self.config.IMG_SIZE)
        except Exception as e:
            print(f"数据集创建失败: {e}")
            # 如果没有验证集，使用训练集的一部分作为验证
            train_dataset = MilitaryDataset(self.config.ROOT_PATH, 'train', self.config.IMG_SIZE)
            val_dataset = train_dataset
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True, 
            collate_fn=self.collate_fn,
            num_workers=0,  # 设置为0避免多进程CUDA冲突
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False, 
            collate_fn=self.collate_fn,
            num_workers=0,  # 设置为0避免多进程CUDA冲突
            pin_memory=True
        )
        
        return train_loader, val_loader

    def collate_fn(self, batch):
        """自定义批次处理函数 - 优化内存管理"""
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs)
        
        # 对齐targets的长度 - 使用CPU张量避免CUDA冲突
        max_len = max([t.shape[0] for t in targets])
        target_tensor = torch.zeros(len(imgs), max_len, 5)  # 先创建CPU张量
        
        for i, t in enumerate(targets):
            if t.shape[0] > 0:
                target_tensor[i, :t.shape[0]] = t  # 保持CPU张量
        
        return imgs, target_tensor

    def get_constraint_weight(self, epoch):
        """线性预热光学约束，避免在某一轮突然改变优化目标。"""
        if epoch < self.config.ENABLE_CONSTRAINT_AFTER_EPOCH:
            return 0.0

        warmup_epochs = max(1, self.config.CONSTRAINT_WARMUP_EPOCHS)
        progress = min(1.0, (epoch - self.config.ENABLE_CONSTRAINT_AFTER_EPOCH + 1) / warmup_epochs)
        return self.config.OPTICAL_CONSTRAINT_WEIGHT * progress

    def normalize_feature_map(self, feature_map):
        """按样本做min-max归一化，使光学约束与可视化都落在稳定范围。"""
        feat_min = feature_map.amin(dim=(2, 3), keepdim=True)
        feat_max = feature_map.amax(dim=(2, 3), keepdim=True)
        return (feature_map - feat_min) / (feat_max - feat_min + 1e-6)

    def enhance_feature_for_display(self, feature_map):
        """用稳健分位数拉伸弱响应，避免特征图看起来一片发黑。"""
        feature_map = np.asarray(feature_map, dtype=np.float32)
        low = np.percentile(feature_map, 2)
        high = np.percentile(feature_map, 98)
        if high - low < 1e-6:
            return np.zeros_like(feature_map)

        feature_map = np.clip((feature_map - low) / (high - low), 0.0, 1.0)
        return np.power(feature_map, 0.8)

    def prediction_response_map(self, pred):
        """将检测头输出转成更可解释的响应热图，而不是直接平均27个logit通道。"""
        grid_h, grid_w = pred.shape[1], pred.shape[2]
        pred = pred.permute(1, 2, 0).reshape(grid_h, grid_w, 3, -1)
        obj_conf = torch.sigmoid(pred[..., 4])
        cls_conf, _ = torch.sigmoid(pred[..., 5:]).max(dim=-1)
        response = (obj_conf * cls_conf).max(dim=-1).values
        return response.detach().cpu().numpy()

    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}")
        
        for batch_idx, (imgs, targets) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device)  # 将targets移动到GPU
            batch_size = imgs.shape[0]
            
            # 前向传播
            p3, p4, p5, optical_feature, constraint_target = self.model(imgs)
            preds = [p3, p4, p5]
            
            # 多尺度损失计算
            loss = 0
            for i, pred in enumerate(preds):
                stride = self.config.STRIDES[i]
                anchors = self.config.ANCHORS[i]
                gt = build_target(targets, anchors, stride, self.num_classes, 
                                 self.config.IMG_SIZE, self.device)
                total_l, box_l, obj_l, cls_l = self.criterion(pred, gt, batch_size)
                loss += total_l
            
            # 光学约束损失（如果启用）
            constraint_weight = self.get_constraint_weight(epoch)
            if constraint_target is not None and constraint_weight > 0:
                normalized_optical = self.normalize_feature_map(optical_feature)
                optical_constraint_loss = self.optical_constraint_criterion(normalized_optical, constraint_target)
                loss += constraint_weight * optical_constraint_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            postfix = {
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss/(batch_idx+1):.4f}"
            }
            
            # 显示光学约束损失（如果存在）
            if constraint_target is not None and constraint_weight > 0:
                postfix["optical_constraint"] = f"{optical_constraint_loss.item():.4f}"
                postfix["constraint_w"] = f"{constraint_weight:.3f}"
            
            pbar.set_postfix(postfix)
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss

    def validate(self, val_loader, epoch):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device)  # 将targets移动到GPU
                batch_size = imgs.shape[0]
                
                # 前向传播
                p3, p4, p5, optical_feature, constraint_target = self.model(imgs)
                preds = [p3, p4, p5]
                
                # 多尺度损失计算
                loss = 0
                for i, pred in enumerate(preds):
                    stride = self.config.STRIDES[i]
                    anchors = self.config.ANCHORS[i]
                    gt = build_target(targets, anchors, stride, self.num_classes, 
                                     self.config.IMG_SIZE, self.device)
                    total_l, _, _, _ = self.criterion(pred, gt, batch_size)
                    loss += total_l
                
                # 光学约束损失（如果启用）
                constraint_weight = self.get_constraint_weight(epoch)
                if constraint_target is not None and constraint_weight > 0:
                    normalized_optical = self.normalize_feature_map(optical_feature)
                    optical_constraint_loss = self.optical_constraint_criterion(normalized_optical, constraint_target)
                    loss += constraint_weight * optical_constraint_loss
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss

    def save_model(self, epoch, val_loss, is_best=False):
        """保存模型检查点"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config.__dict__,
            'class_names': self.class_names,
            'num_classes': self.num_classes
        }
        
        if is_best:
            filename = f"optical_yolo_best_{timestamp}.pth"
        else:
            filename = f"optical_yolo_epoch_{epoch+1}_{timestamp}.pth"
        
        filepath = os.path.join(self.config.SAVE_DIR, filename)
        torch.save(checkpoint, filepath)
        print(f"模型已保存: {filepath}")

    def decode_detections(self, preds, conf_thresh=0.5, nms_thresh=0.4, max_det=100):
        """解码检测结果，返回边界框、置信度和类别"""
        batch_size = preds[0].shape[0]
        detections = [[] for _ in range(batch_size)]
        
        for i, pred in enumerate(preds):
            grid_h, grid_w = pred.shape[2], pred.shape[3]
            stride = self.config.STRIDES[i]
            
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

                                # 训练目标使用整张特征图坐标，因此解码时不应重复叠加网格索引或anchor尺度。
                                x = float(torch.clamp(bx * stride, 0, self.config.IMG_SIZE - 1).item())
                                y = float(torch.clamp(by * stride, 0, self.config.IMG_SIZE - 1).item())
                                width = float(torch.clamp(torch.abs(bw) * stride, 1, self.config.IMG_SIZE).item())
                                height = float(torch.clamp(torch.abs(bh) * stride, 1, self.config.IMG_SIZE).item())

                                detections[b].append([x, y, width, height, conf, current_cls_id])

        final_detections = []
        for batch_detections in detections:
            if batch_detections:
                batch_detections = torch.tensor(batch_detections, dtype=torch.float32)
                keep = self.non_max_suppression(batch_detections, nms_thresh)
                kept = batch_detections[keep]
                if kept.shape[0] > max_det:
                    kept = kept[torch.argsort(kept[:, 4], descending=True)[:max_det]]
                final_detections.append(kept.tolist())
            else:
                final_detections.append([])
        
        return final_detections
    
    def non_max_suppression(self, detections, nms_thresh):
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
            
            ious = self.calculate_iou(current_box.unsqueeze(0), other_boxes)
            
            # 移除重叠度高的检测
            keep_indices = torch.where(ious < nms_thresh)[0]
            sorted_indices = sorted_indices[keep_indices + 1]
        
        return keep
    
    def calculate_iou(self, box1, box2):
        """计算IoU"""
        # box格式: [x, y, w, h]
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

    def calculate_detection_metrics(self, detections, targets):
        """计算检测指标"""
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

    def visualize_results(self, dataloader, epoch):
        """可视化训练结果 - 包含边界框、类别标签和置信度"""
        self.model.eval()
        
        # 获取指定批次的数据
        if self.config.VISUALIZE_BATCH_INDEX >= 0:
            # 手动选择指定批次
            batch_iterator = iter(dataloader)
            for i in range(self.config.VISUALIZE_BATCH_INDEX + 1):
                imgs, targets = next(batch_iterator)
            print(f"使用手动选择的批次索引: {self.config.VISUALIZE_BATCH_INDEX}")
        else:
            # 随机选择一个批次
            imgs, targets = next(iter(dataloader))
            print("使用随机选择的批次")
        
        imgs = imgs[:4]  # 取前4个样本
        targets = targets[:4]  # 对应的目标
        
        with torch.no_grad():
            p3, p4, p5, optical_feat, constraint_target = self.model(imgs.to(self.device))
            
            # 解码检测结果（添加置信度显示）
            detections = self.decode_detections(
                [p3, p4, p5],
                self.config.VIS_CONF_THRESH,
                self.config.NMS_THRESH,
                max_det=self.config.VIS_MAX_DETECTIONS
            )
        
        # 创建可视化（根据是否启用约束调整布局）
        num_rows = min(4, imgs.shape[0])
        if constraint_target is not None:
            fig, axes = plt.subplots(
                num_rows,
                6,
                figsize=(18, 3.6 * num_rows),
                squeeze=False,
                gridspec_kw={'width_ratios': [1.45, 1, 1, 1, 1, 1]}
            )
            optical_col, constraint_col, p3_col, p4_col, p5_col = 1, 2, 3, 4, 5
        else:
            fig, axes = plt.subplots(
                num_rows,
                5,
                figsize=(15.5, 3.6 * num_rows),
                squeeze=False,
                gridspec_kw={'width_ratios': [1.45, 1, 1, 1, 1]}
            )
            optical_col, constraint_col, p3_col, p4_col, p5_col = 1, None, 2, 3, 4
        
        for idx in range(num_rows):
            # 输入图像（带边界框和标签）
            img_np = imgs[idx].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)  # 转换为0-255范围
            
            # 绘制图像
            axes[idx, 0].imshow(img_np)
            
            # 绘制真实边界框和类别标签
            h, w = img_np.shape[:2]
            target_data = targets[idx]
            
            # 统计检测到的目标数量
            obj_count = 0
            for t_idx in range(target_data.shape[0]):
                target = target_data[t_idx]
                if target[4] == 0:  # 跳过空目标
                    continue
                    
                cls_id, cx, cy, bw, bh = target.cpu().numpy()
                
                # 转换为像素坐标
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                # 绘制边界框
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, edgecolor='red', linewidth=2)
                axes[idx, 0].add_patch(rect)
                
                # 添加类别标签
                class_name = self.class_names.get(int(cls_id), f"Class {int(cls_id)}")
                text_y = max(2, y1 - 10)
                axes[idx, 0].text(
                    x1,
                    text_y,
                    class_name,
                    color='white',
                    bbox=dict(facecolor='red', alpha=0.85, pad=1.5),
                    fontsize=7
                )
                
                obj_count += 1
            
            # 绘制模型检测结果（带置信度）
            img_detections = sorted(
                detections[idx] if idx < len(detections) else [],
                key=lambda det: det[4],
                reverse=True
            )[:self.config.VIS_MAX_DETECTIONS]
            for det in img_detections:
                x, y, w, h, conf, cls_id = det[:6]
                
                # 转换为像素坐标（x/y为中心点）
                x1 = int(max(0, x - w / 2))
                y1 = int(max(0, y - h / 2))
                x2 = int(min(self.config.IMG_SIZE, x + w / 2))
                y2 = int(min(self.config.IMG_SIZE, y + h / 2))
                
                # 绘制检测框（绿色，带置信度）
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                   fill=False, edgecolor='green', linewidth=2, linestyle='--')
                axes[idx, 0].add_patch(rect)
                
                # 添加置信度标签
                class_name_det = self.class_names.get(int(cls_id), f"Class {int(cls_id)}")
                text_y = min(self.config.IMG_SIZE - 12, max(2, y1 + 2))
                axes[idx, 0].text(
                    x1,
                    text_y,
                    f"{class_name_det} {conf:.2f}",
                    color='white',
                    bbox=dict(facecolor='green', alpha=0.85, pad=1.5),
                    fontsize=7
                )
            
            axes[idx, 0].set_title(f"Input {idx+1} (真实: {obj_count}, 检测: {len(img_detections)})")
            axes[idx, 0].axis('off')
            
            # 光学特征
            optical_np = self.enhance_feature_for_display(
                self.normalize_feature_map(optical_feat[idx:idx + 1]).squeeze().cpu().numpy()
            )
            im = axes[idx, optical_col].imshow(optical_np, cmap='inferno')
            axes[idx, optical_col].set_title("Optical Feature")
            axes[idx, optical_col].axis('off')
            plt.colorbar(im, ax=axes[idx, optical_col], fraction=0.035, pad=0.01)
            
            # 光学约束目标（如果存在）
            if constraint_target is not None:
                constraint_np = self.enhance_feature_for_display(constraint_target[idx].squeeze().cpu().numpy())
                im = axes[idx, constraint_col].imshow(constraint_np, cmap='inferno')
                axes[idx, constraint_col].set_title("Constraint Target")
                axes[idx, constraint_col].axis('off')
                plt.colorbar(im, ax=axes[idx, constraint_col], fraction=0.035, pad=0.01)
            
            # 多尺度响应图：显示 obj * cls 的响应，而不是直接均值化原始logit
            p3_np = self.enhance_feature_for_display(self.prediction_response_map(p3[idx]))
            im = axes[idx, p3_col].imshow(p3_np, cmap='inferno', interpolation='nearest')
            axes[idx, p3_col].set_title("P3 (80×80)")
            axes[idx, p3_col].axis('off')
            plt.colorbar(im, ax=axes[idx, p3_col], fraction=0.035, pad=0.01)
            
            p4_np = self.enhance_feature_for_display(self.prediction_response_map(p4[idx]))
            im = axes[idx, p4_col].imshow(p4_np, cmap='inferno', interpolation='nearest')
            axes[idx, p4_col].set_title("P4 (40×40)")
            axes[idx, p4_col].axis('off')
            plt.colorbar(im, ax=axes[idx, p4_col], fraction=0.035, pad=0.01)
            
            p5_np = self.enhance_feature_for_display(self.prediction_response_map(p5[idx]))
            im = axes[idx, p5_col].imshow(p5_np, cmap='inferno', interpolation='nearest')
            axes[idx, p5_col].set_title("P5 (20×20)")
            axes[idx, p5_col].axis('off')
            plt.colorbar(im, ax=axes[idx, p5_col], fraction=0.035, pad=0.01)
        
        fig.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.02, wspace=0.08, hspace=0.18)
        vis_path = os.path.join(self.config.VISUALIZATION_DIR, f"visualization_epoch_{epoch+1}.png")
        plt.savefig(vis_path, dpi=180, bbox_inches='tight', pad_inches=0.08)
        plt.close()
        print(f"可视化结果已保存: {vis_path}")

    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        history_path = os.path.join(self.config.VISUALIZATION_DIR, "training_history.png")
        plt.savefig(history_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"训练历史图已保存: {history_path}")

    def train(self):
        """主训练循环"""
        print("开始训练光学YOLOv3模型...")
        
        # 添加CUDA内存优化设置
        torch.backends.cudnn.benchmark = True  # 加速卷积运算
        
        # 创建数据加载器
        train_loader, val_loader = self.create_dataloaders()
        
        # 训练开始时间
        start_time = time.time()
        
        # 添加定期内存清理
        torch.cuda.empty_cache()
        
        for epoch in range(self.config.EPOCHS):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.config.EPOCHS}")
            print(f"{'='*50}")
            
            # 训练策略调整
            if epoch >= self.config.ENABLE_NORM_AFTER_EPOCH:
                self.model.enable_normalization(True)
                print("已启用光学输出归一化")
            else:
                self.model.enable_normalization(False)
            
            # 启用光学约束（在训练稳定后）
            if epoch >= self.config.ENABLE_CONSTRAINT_AFTER_EPOCH:
                self.model.enable_constraint_loss(True)
                print(f"已启用光学约束损失，当前权重: {self.get_constraint_weight(epoch):.4f}")
            else:
                self.model.enable_constraint_loss(False)
            
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss = self.validate(val_loader, epoch)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 计算检测指标（从第5轮开始）
            if epoch >= 5:
                # 获取指定批次的检测结果
                with torch.no_grad():
                    if self.config.VISUALIZE_BATCH_INDEX >= 0:
                        # 使用与可视化相同的批次
                        batch_iterator = iter(val_loader)
                        for i in range(self.config.VISUALIZE_BATCH_INDEX + 1):
                            imgs_sample, targets_sample = next(batch_iterator)
                    else:
                        # 随机选择一个批次
                        imgs_sample, targets_sample = next(iter(val_loader))
                    
                    imgs_sample = imgs_sample.to(self.device)
                    p3, p4, p5, _, _ = self.model(imgs_sample)
                    detections = self.decode_detections([p3, p4, p5], self.config.CONF_THRESH, self.config.NMS_THRESH)
                    precision, recall, f1_score = self.calculate_detection_metrics(detections, targets_sample)
                    
                    self.precisions.append(precision)
                    self.recalls.append(recall)
                    self.f1_scores.append(f1_score)
            else:
                precision = recall = f1_score = 0
            
            if epoch < 5:
                print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 学习率: {current_lr:.6f}")
            else:
                print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, "
                      f"精确率: {precision:.3f}, 召回率: {recall:.3f}, F1: {f1_score:.3f}, "
                      f"学习率: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss - self.config.EARLY_STOPPING_MIN_DELTA:
                self.best_val_loss = val_loss
                self.no_improve_epochs = 0
                self.save_model(epoch, val_loss, is_best=True)
                print("新的最佳模型已保存!")
            else:
                self.no_improve_epochs += 1
            
            # 定期保存和可视化
            if (epoch + 1) % self.config.SAVE_EVERY == 0:
                self.save_model(epoch, val_loss)
            
            if (epoch + 1) % self.config.VISUALIZE_EVERY == 0:
                self.visualize_results(val_loader, epoch)
            
            # 定期清理GPU内存
            if (epoch + 1) % 5 == 0:
                torch.cuda.empty_cache()

            if self.no_improve_epochs >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"验证损失连续 {self.no_improve_epochs} 轮未改善，提前停止训练。")
                break
        
        # 训练结束
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n{'='*50}")
        print("训练完成!")
        print(f"总训练时间: {training_time/60:.2f} 分钟")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"模型保存在: {self.config.SAVE_DIR}")
        print(f"日志保存在: {self.config.LOG_DIR}")
        
        # 绘制训练历史
        self.plot_training_history()

# =========================================================
# 主函数
# =========================================================
def main():
    # 初始化配置
    config = Config()
    
    # 创建训练器
    trainer = OpticalYOLOv3Trainer(config)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()