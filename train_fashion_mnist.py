import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import time
from datetime import datetime
from PIL import Image
import requests
import zipfile
import tarfile

# 导入光学YOLOv3模型
from Optical_class import OpticalYOLOv3, YOLOLoss, build_target

# =========================================================
# Fashion-MNIST数据集配置
# =========================================================
class FashionMNISTConfig:
    # 设备设置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 训练参数（针对Fashion-MNIST优化）
    EPOCHS = 100
    BATCH_SIZE = 32  # Fashion-MNIST图像较小，可以增大批次
    ORIGINAL_SIZE = 28  # Fashion-MNIST原始尺寸
    TARGET_SIZE = 416   # 训练目标尺寸
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # 损失权重（针对时尚物品检测优化）
    BOX_WEIGHT = 0.06   # 边界框权重
    OBJ_WEIGHT = 1.0    # 目标权重
    CLS_WEIGHT = 0.25   # 分类权重（Fashion-MNIST有10个类别）
    OPTICAL_CONSTRAINT_WEIGHT = 0.1  # 光学约束损失权重
    
    # 锚框设置（针对Fashion-MNIST物品优化）
    STRIDES = [8, 16, 32]
    ANCHORS = [
        [[12, 16], [20, 24], [28, 32]],     # P3: 小物品
        [[36, 48], [48, 64], [64, 80]],      # P4: 中等物品
        [[80, 96], [96, 128], [128, 160]]    # P5: 较大物品
    ]
    
    # 路径设置
    DATA_ROOT = r"data\fashion_mnist"
    SAVE_DIR = r"output\fashion_mnist_models"
    LOG_DIR = r"output\fashion_mnist_logs"
    
    # 训练策略
    ENABLE_NORM_AFTER_EPOCH = 20  # 20轮后启用归一化
    ENABLE_CONSTRAINT_AFTER_EPOCH = 10  # 10轮后启用光学约束
    VISUALIZE_EVERY = 5  # 每5轮可视化一次
    SAVE_EVERY = 10  # 每10轮保存一次
    
    # 检测阈值
    CONF_THRESH = 0.5  # 置信度阈值
    NMS_THRESH = 0.4   # 非极大值抑制阈值
    
    # Fashion-MNIST类别名称
    CLASS_NAMES = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

# =========================================================
# Fashion-MNIST数据集处理类
# =========================================================
class FashionMNISTDataset(Dataset):
    def __init__(self, images, labels, img_size=416, augment=True, mode='train'):
        self.images = images
        self.labels = labels
        self.img_size = img_size
        self.mode = mode
        self.augment = augment and mode == 'train'
        
        print(f"加载Fashion-MNIST {mode}数据集: {len(images)} 张图像")
        
        # 数据增强（使用torchvision替代albumentations）
        if self.augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.2860, 0.2860, 0.2860], std=[0.3530, 0.3530, 0.3530]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.2860, 0.2860, 0.2860], std=[0.3530, 0.3530, 0.3530]),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 获取图像和标签
        image = self.images[idx]
        label = self.labels[idx]
        
        # 将单通道图像转换为PIL图像，然后转换为三通道
        if len(image.shape) == 2:
            # 转换为PIL图像
            image_pil = Image.fromarray(image, mode='L')
            # 转换为RGB三通道
            image_pil = image_pil.convert('RGB')
        else:
            image_pil = Image.fromarray(image)
        
        # 由于Fashion-MNIST是单目标数据集，我们为每个图像创建一个边界框
        # 边界框覆盖整个图像（因为每个图像只有一个时尚物品）
        bbox = [0.5, 0.5, 0.9, 0.9]  # 中心点(0.5,0.5)，宽高0.9（几乎覆盖整个图像）
        
        # 应用数据增强
        image_tensor = self.transform(image_pil)
        
        # 创建目标张量
        targets = torch.zeros((1, 5))
        targets[0, 0] = label  # 类别
        targets[0, 1:5] = torch.tensor(bbox)  # 边界框
        
        return image_tensor, targets

# =========================================================
# Fashion-MNIST数据下载和处理工具
# =========================================================
class FashionMNISTDataLoader:
    def __init__(self, config):
        self.config = config
        self.data_root = config.DATA_ROOT
        os.makedirs(self.data_root, exist_ok=True)
        
    def download_fashion_mnist(self):
        """下载Fashion-MNIST数据集"""
        print("正在下载Fashion-MNIST数据集...")
        
        try:
            # 使用torchvision内置的Fashion-MNIST数据集
            train_dataset = datasets.FashionMNIST(
                root=self.data_root, 
                train=True, 
                download=True,
                transform=transforms.ToTensor()
            )
            
            test_dataset = datasets.FashionMNIST(
                root=self.data_root, 
                train=False, 
                download=True,
                transform=transforms.ToTensor()
            )
            
            print("Fashion-MNIST数据集下载完成!")
            return train_dataset, test_dataset
            
        except Exception as e:
            print(f"下载Fashion-MNIST失败: {e}")
            # 尝试备用下载方式
            return self.download_fashion_mnist_fallback()
    
    def download_fashion_mnist_fallback(self):
        """备用下载方式"""
        print("尝试备用下载方式...")
        
        # 这里可以添加其他下载逻辑
        # 暂时返回空数据集，让用户手动处理
        raise Exception("Fashion-MNIST下载失败，请检查网络连接或手动下载数据集")
    
    def create_yaml_config(self):
        """创建Fashion-MNIST的YAML配置文件"""
        yaml_content = {
            'path': os.path.abspath(self.data_root),
            'train': 'train',
            'val': 'test',  # 使用测试集作为验证集
            'test': 'test',
            'nc': 10,  # 10个类别
            'names': self.config.CLASS_NAMES
        }
        
        yaml_path = os.path.join(self.data_root, 'data.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
        print(f"创建配置文件: {yaml_path}")
        return yaml_path
    
    def prepare_datasets(self):
        """准备训练和验证数据集"""
        print("准备Fashion-MNIST数据集...")
        
        # 下载数据集
        train_dataset, test_dataset = self.download_fashion_mnist()
        
        # 提取图像和标签
        train_images = train_dataset.data.numpy()
        train_labels = train_dataset.targets.numpy()
        
        test_images = test_dataset.data.numpy()
        test_labels = test_dataset.targets.numpy()
        
        print(f"训练集: {len(train_images)} 张图像")
        print(f"测试集: {len(test_images)} 张图像")
        
        # 创建YAML配置文件
        self.create_yaml_config()
        
        return (train_images, train_labels), (test_images, test_labels)

# =========================================================
# Fashion-MNIST训练器
# =========================================================
class FashionMNISTTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        # 创建输出目录
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        
        # 加载数据
        self.data_loader = FashionMNISTDataLoader(config)
        (train_images, train_labels), (test_images, test_labels) = self.data_loader.prepare_datasets()
        
        # 初始化模型
        self.model = self.init_model()
        self.criterion = YOLOLoss(config.BOX_WEIGHT, config.OBJ_WEIGHT, config.CLS_WEIGHT)
        
        # 光学约束损失函数
        self.optical_constraint_criterion = nn.MSELoss()
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.EPOCHS,
            eta_min=1e-6
        )
        
        # 创建数据加载器
        self.train_loader, self.val_loader = self.create_dataloaders(
            train_images, train_labels, test_images, test_labels
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # 检测指标
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        
        print(f"Fashion-MNIST训练器初始化完成，使用设备: {self.device}")

    def init_model(self):
        """初始化针对Fashion-MNIST优化的光学YOLOv3模型（带约束）"""
        model = OpticalYOLOv3(
            num_classes=10,  # Fashion-MNIST有10个类别
            img_size=self.config.TARGET_SIZE,
            optical_mode="phase",
            enable_constraint=True  # 启用光学约束
        ).to(self.device)
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Fashion-MNIST检测模型总参数量: {total_params:,}")
        
        return model

    def create_dataloaders(self, train_images, train_labels, test_images, test_labels):
        """创建训练和验证数据加载器"""
        train_dataset = FashionMNISTDataset(
            train_images, train_labels, 
            img_size=self.config.TARGET_SIZE, 
            augment=True, 
            mode='train'
        )
        
        val_dataset = FashionMNISTDataset(
            test_images, test_labels, 
            img_size=self.config.TARGET_SIZE, 
            augment=False, 
            mode='val'
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True, 
            collate_fn=self.collate_fn,
            num_workers=0,  # 设置为0避免CUDA多进程冲突
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False, 
            collate_fn=self.collate_fn,
            num_workers=0,  # 设置为0避免CUDA多进程冲突
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

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # 进度条
        pbar = tqdm(self.train_loader, desc=f"Fashion-MNIST训练 Epoch {epoch+1}/{self.config.EPOCHS}")
        
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
                gt = build_target(targets, anchors, stride, 10, self.config.TARGET_SIZE, self.device)
                total_l, box_l, obj_l, cls_l = self.criterion(pred, gt, batch_size)
                loss += total_l
            
            # 光学约束损失（如果启用）
            if constraint_target is not None:
                optical_constraint_loss = self.optical_constraint_criterion(optical_feature, constraint_target)
                loss += self.config.OPTICAL_CONSTRAINT_WEIGHT * optical_constraint_loss  # 光学约束损失权重
            
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
            if constraint_target is not None:
                postfix["optical_constraint"] = f"{optical_constraint_loss.item():.4f}"
            
            pbar.set_postfix(postfix)
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss

    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for imgs, targets in self.val_loader:
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
                    gt = build_target(targets, anchors, stride, 10, self.config.TARGET_SIZE, self.device)
                    total_l, _, _, _ = self.criterion(pred, gt, batch_size)
                    loss += total_l
                
                # 光学约束损失（如果启用）
                if constraint_target is not None:
                    optical_constraint_loss = self.optical_constraint_criterion(optical_feature, constraint_target)
                    loss += self.config.OPTICAL_CONSTRAINT_WEIGHT * optical_constraint_loss
                
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
            'class_names': self.config.CLASS_NAMES,
            'description': 'Fashion-MNIST时尚物品检测光学YOLOv3模型'
        }
        
        if is_best:
            filename = f"fashion_mnist_best_{timestamp}.pth"
        else:
            filename = f"fashion_mnist_epoch_{epoch+1}_{timestamp}.pth"
        
        filepath = os.path.join(self.config.SAVE_DIR, filename)
        torch.save(checkpoint, filepath)
        print(f"Fashion-MNIST模型已保存: {filepath}")

    def decode_detections(self, preds, conf_thresh=0.5, nms_thresh=0.4):
        """解码检测结果，返回边界框、置信度和类别"""
        detections = []
        
        for i, pred in enumerate(preds):
            batch_size = pred.shape[0]
            grid_h, grid_w = pred.shape[2], pred.shape[3]
            
            # 重塑预测为 (batch_size, grid_h, grid_w, 3, 5+num_classes)
            pred = pred.permute(0, 2, 3, 1).reshape(batch_size, grid_h, grid_w, 3, -1)
            
            # 应用置信度阈值
            obj_conf = torch.sigmoid(pred[..., 4])
            obj_mask = obj_conf > conf_thresh
            
            for b in range(batch_size):
                batch_detections = []
                for h in range(grid_h):
                    for w in range(grid_w):
                        for anchor in range(3):
                            if obj_mask[b, h, w, anchor]:
                                # 提取边界框和类别
                                bx, by, bw, bh = pred[b, h, w, anchor, :4]
                                cls_probs = torch.softmax(pred[b, h, w, anchor, 5:], dim=0)
                                cls_id = torch.argmax(cls_probs).item()
                                cls_conf = cls_probs[cls_id].item()
                                
                                # 计算最终置信度
                                conf = obj_conf[b, h, w, anchor].item() * cls_conf
                                
                                if conf > conf_thresh:
                                    # 转换为图像坐标
                                    x = (w + bx) * self.config.STRIDES[i]
                                    y = (h + by) * self.config.STRIDES[i]
                                    width = bw * self.config.ANCHORS[i][anchor][0]
                                    height = bh * self.config.ANCHORS[i][anchor][1]
                                    
                                    detection = [x, y, width, height, conf, cls_id]
                                    batch_detections.append(detection)
                
                # 应用非极大值抑制
                if len(batch_detections) > 0:
                    batch_detections = torch.tensor(batch_detections)
                    keep = self.non_max_suppression(batch_detections, nms_thresh)
                    detections.append(batch_detections[keep].tolist())
                else:
                    detections.append([])
        
        return detections
    
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
        
        # 计算交集区域
        inter_x1 = torch.max(box1_x1, box2_x1)
        inter_y1 = torch.max(box1_y1, box2_y1)
        inter_x2 = torch.min(box1_x2, box2_x2)
        inter_y2 = torch.min(box1_y2, box2_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 计算并集区域
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - inter_area
        
        # 计算IoU
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

    def visualize_fashion_results(self, epoch):
        """可视化Fashion-MNIST训练结果 - 包含边界框、类别标签和置信度"""
        self.model.eval()
        
        # 获取一个批次的数据
        try:
            imgs, targets = next(iter(self.val_loader))
            imgs = imgs[:4]  # 取前4个样本
            targets = targets[:4]  # 对应的目标
        except StopIteration:
            print("数据加载器为空，跳过可视化")
            return
        
        with torch.no_grad():
            p3, p4, p5, optical_feat, constraint_target = self.model(imgs.to(self.device))
            
            # 解码检测结果（添加置信度显示）
            detections = self.decode_detections([p3, p4, p5], self.config.CONF_THRESH, self.config.NMS_THRESH)
        
        # 创建可视化（根据是否启用约束调整布局）
        if constraint_target is not None:
            fig, axes = plt.subplots(4, 6, figsize=(24, 16))  # 6列布局
        else:
            fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        
        for idx in range(min(4, imgs.shape[0])):
            # 输入图像（带边界框和标签）
            img_np = imgs[idx].permute(1, 2, 0).cpu().numpy()
            # 反归一化
            img_np = img_np * np.array([0.3530, 0.3530, 0.3530]) + np.array([0.2860, 0.2860, 0.2860])
            img_np = np.clip(img_np, 0, 1)
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
                class_name = self.config.CLASS_NAMES[int(cls_id)]
                axes[idx, 0].text(x1, y1-5, class_name, color='white',
                                bbox=dict(facecolor='red', alpha=0.8), fontsize=8)
                
                obj_count += 1
            
            # 绘制模型检测结果（带置信度）
            img_detections = detections[idx] if idx < len(detections) else []
            for det in img_detections:
                x, y, w, h, conf, cls_id = det[:6]
                
                # 转换为像素坐标
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
                
                # 绘制检测框（绿色，带置信度）
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, edgecolor='green', linewidth=2, linestyle='--')
                axes[idx, 0].add_patch(rect)
                
                # 添加置信度标签
                class_name_det = self.config.CLASS_NAMES[int(cls_id)]
                axes[idx, 0].text(x1, y1-25, f"{class_name_det}: {conf:.2f}", 
                                color='white', bbox=dict(facecolor='green', alpha=0.8), fontsize=8)
            
            axes[idx, 0].set_title(f"{class_name}\n输入图像 (真实: {obj_count}, 检测: {len(img_detections)})")
            axes[idx, 0].axis('off')
            
            # 光学特征
            optical_np = optical_feat[idx].squeeze().cpu().numpy()
            im = axes[idx, 1].imshow(optical_np, cmap='hot')
            axes[idx, 1].set_title("光学调制特征")
            axes[idx, 1].axis('off')
            plt.colorbar(im, ax=axes[idx, 1])
            
            # 光学约束目标（如果存在）
            if constraint_target is not None:
                constraint_np = constraint_target[idx].squeeze().cpu().numpy()
                im = axes[idx, 2].imshow(constraint_np, cmap='hot')
                axes[idx, 2].set_title("约束目标")
                axes[idx, 2].axis('off')
                plt.colorbar(im, ax=axes[idx, 2])
            else:
                axes[idx, 2].axis('off')
            
            # 多尺度特征
            p3_np = p3[idx].abs().mean(dim=0).cpu().numpy()
            axes[idx, 3].imshow(p3_np, cmap='hot')
            axes[idx, 3].set_title("P3检测特征")
            axes[idx, 3].axis('off')
            
            p4_np = p4[idx].abs().mean(dim=0).cpu().numpy()
            axes[idx, 4].imshow(p4_np, cmap='hot')
            axes[idx, 4].set_title("P4检测特征")
            axes[idx, 4].axis('off')
            
            # 如果启用了约束，调整子图布局
            if constraint_target is not None:
                # 添加第6列显示P5特征
                p5_np = p5[idx].abs().mean(dim=0).cpu().numpy()
                im = axes[idx, 5].imshow(p5_np, cmap='hot')
                axes[idx, 5].set_title("P5检测特征")
                axes[idx, 5].axis('off')
                plt.colorbar(im, ax=axes[idx, 5])
            else:
                p5_np = p5[idx].abs().mean(dim=0).cpu().numpy()
                axes[idx, 4].imshow(p5_np, cmap='hot')
                axes[idx, 4].set_title("P5检测特征")
                axes[idx, 4].axis('off')
        
        plt.tight_layout()
        vis_path = os.path.join(self.config.LOG_DIR, f"fashion_mnist_epoch_{epoch+1}.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Fashion-MNIST可视化结果已保存: {vis_path}")

    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label='训练损失', color='blue', linewidth=2)
        if self.val_losses:
            plt.plot(self.val_losses, label='验证损失', color='red', linewidth=2)
        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
        plt.title('Fashion-MNIST时尚物品检测训练历史')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        # 显示类别分布示例
        class_counts = [self.train_losses.count(i) for i in range(10)]
        plt.bar(range(10), [len(self.train_losses)//10] * 10, alpha=0.5)
        plt.xlabel('类别ID')
        plt.ylabel('样本数量')
        plt.title('Fashion-MNIST类别分布')
        plt.xticks(range(10), self.config.CLASS_NAMES, rotation=45, ha='right')
        
        plt.tight_layout()
        history_path = os.path.join(self.config.LOG_DIR, "fashion_mnist_training_history.png")
        plt.savefig(history_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Fashion-MNIST训练历史图已保存: {history_path}")

    def train(self):
        """主训练循环"""
        print("开始训练Fashion-MNIST时尚物品检测光学YOLOv3模型...")
        print("=" * 60)
        print("数据集信息:")
        print(f"  训练样本: {len(self.train_loader.dataset)}")
        print(f"  验证样本: {len(self.val_loader.dataset)}")
        print(f"  目标尺寸: {self.config.TARGET_SIZE}×{self.config.TARGET_SIZE}")
        print(f"  类别数量: 10")
        print("=" * 60)
        
        # 训练开始时间
        start_time = time.time()
        
        # 添加CUDA内存优化设置
        torch.backends.cudnn.benchmark = True  # 加速卷积运算
        torch.cuda.empty_cache()  # 清理GPU内存
        
        for epoch in range(self.config.EPOCHS):
            print(f"\n{'='*50}")
            print(f"Fashion-MNIST训练 Epoch {epoch+1}/{self.config.EPOCHS}")
            print(f"{'='*50}")
            
            # 训练策略调整
            if epoch >= self.config.ENABLE_NORM_AFTER_EPOCH:
                self.model.enable_normalization(True)
                print("已启用光学输出归一化")
            
            # 训练一个epoch
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate()
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 训练策略调整
            if epoch >= self.config.ENABLE_NORM_AFTER_EPOCH:
                self.model.enable_normalization(True)
                print("已启用光学输出归一化")
            
            # 启用光学约束（在训练稳定后）
            if epoch >= self.config.ENABLE_CONSTRAINT_AFTER_EPOCH:
                self.model.enable_constraint_loss(True)
                print("已启用光学约束损失")
            
            print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 学习率: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(epoch, val_loss, is_best=True)
                print("💾 新的最佳模型已保存!")
            
            # 定期保存和可视化
            if (epoch + 1) % self.config.SAVE_EVERY == 0:
                self.save_model(epoch, val_loss)
            
            if (epoch + 1) % self.config.VISUALIZE_EVERY == 0:
                self.visualize_fashion_results(epoch)
            
            # 定期清理GPU内存
            if (epoch + 1) % 5 == 0:
                torch.cuda.empty_cache()
        
        # 训练结束
        end_time = time.time()
        training_time = (end_time - start_time) / 60
        
        print(f"\n{'='*60}")
        print("🎉 Fashion-MNIST时尚物品检测训练完成!")
        print(f"⏱️  总训练时间: {training_time:.2f} 分钟")
        print(f"🏆 最佳验证损失: {self.best_val_loss:.4f}")
        print(f"💾 模型保存在: {self.config.SAVE_DIR}")
        print(f"📊 日志保存在: {self.config.LOG_DIR}")
        print(f"👗 检测类别: {self.config.CLASS_NAMES}")
        print(f"{'='*60}")
        
        # 绘制训练历史
        self.plot_training_history()

# =========================================================
# 主函数
# =========================================================
def main():
    # 初始化配置
    config = FashionMNISTConfig()
    
    # 创建训练器
    trainer = FashionMNISTTrainer(config)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()