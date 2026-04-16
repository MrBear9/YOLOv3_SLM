
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
    BOX_WEIGHT = 0.05 # 目标框损失权重
    OBJ_WEIGHT = 1.0 # 目标检测损失权重（降低，避免过度关注目标存在性）
    NOOBJ_WEIGHT = 0.5 # 非目标损失权重（降低，减少背景误检的影响）
    CLS_WEIGHT = 0.15 # 分类损失权重
    PHASE1_TEACHER_WEIGHT = 1.0  # 阶段1只做教师约束时的损失权重
    TEACHER_CONSTRAINT_WEIGHT = 0.05  # 阶段2联合训练时的教师约束权重
    
    # 差异化学习率设置
    OPTICAL_LR_RATIO = 0.1  # 相位层学习率比例 10%（相对于主学习率），相位层对学习率敏感度较低
    USE_VORTEX_INIT = True  # 是否使用涡旋初始化
    SLM1_VORTEX_CHARGE = 1  # SLM1涡旋电荷
    SLM2_VORTEX_CHARGE = -1  # SLM2涡旋电荷
    VORTEX_PERTURBATION = 0.12  # 涡旋相位上的小随机扰动，避免先验过硬 %
    
    # 锚框设置（针对军事目标优化）
    STRIDES = [8, 16, 32]
    ANCHORS = [
        [[26,23], [47,49], [100,67]],   # P3: 小目标 / 较小目标
        [[103,169], [203,107], [351,177]],  # P4: 中目标 / 长条目标
        [[241,354], [534,299], [568,528]]  # P5: 大目标 / 超大目标（军舰等）
    ]
    
    # 路径设置
    DATA_YAML_PATH = r"data\military\data.yaml"
    ROOT_PATH = r"data\military"
    # 统一输出路径配置
    OPTICAL_YOLO_OUTPUT_DIR = r"output\OpticalYOLO"
    SAVE_DIR = os.path.join(OPTICAL_YOLO_OUTPUT_DIR, "models")
    LOG_DIR = os.path.join(OPTICAL_YOLO_OUTPUT_DIR, "logs")
    VISUALIZATION_DIR = os.path.join(OPTICAL_YOLO_OUTPUT_DIR, "visualizations")
    TEACHER_CHECKPOINT = r"output\OpticalTeacherYOLO\teacher_best.pth"
    
    # 训练策略
    ENABLE_NORM_AFTER_EPOCH = 0  # 从训练开始就保持光学输出尺度稳定
    PHASE1_EPOCHS = 10  # 阶段1：教师约束光学层
    PHASE2_EPOCHS = 90  # 阶段2：教师约束 + 光学层 + 检测头
    TEACHER_WARMUP_EPOCHS = 10  # 阶段2中教师约束的线性预热
    TEACHER_INIT_MODE = "checkpoint_or_random"  # checkpoint | checkpoint_or_random | random
    FREEZE_TEACHER = True
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

    @classmethod
    def get_current_phase(cls, epoch):
        if epoch < cls.PHASE1_EPOCHS:
            return "phase1", "教师约束光学层"
        if epoch < cls.PHASE1_EPOCHS + cls.PHASE2_EPOCHS:
            return "phase2", "教师约束 + 光学层 + 检测头"
        return "phase3", "光学层 + 检测头"

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
        self.criterion = YOLOLoss(config.BOX_WEIGHT, config.OBJ_WEIGHT, config.NOOBJ_WEIGHT, config.CLS_WEIGHT)
        
        # 光学约束损失函数
        self.optical_constraint_criterion = nn.MSELoss()
        
        # 差异化学习率：相位层使用更小的学习率
        optical_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'phase_raw' in name or 'amp_raw' in name:
                optical_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {'params': other_params, 'lr': config.LEARNING_RATE, 'weight_decay': config.WEIGHT_DECAY},
            {'params': optical_params, 'lr': config.LEARNING_RATE * config.OPTICAL_LR_RATIO, 'weight_decay': config.WEIGHT_DECAY}
        ]
        self.optimizer = optim.Adam(param_groups)
        
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
        
        # 初始化日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config_log_path = os.path.join(config.LOG_DIR, f"config_{timestamp}.txt")
        self.training_log_path = os.path.join(config.LOG_DIR, f"training_log_{timestamp}.txt")
        self.summary_log_path = os.path.join(config.LOG_DIR, f"training_summary_{timestamp}.txt")
        
        # 保存配置参数
        self.save_config_to_txt()
        
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
            enable_constraint=True,
            teacher_checkpoint=self.config.TEACHER_CHECKPOINT,
            teacher_init_mode=self.config.TEACHER_INIT_MODE,
            freeze_teacher=self.config.FREEZE_TEACHER,
            teacher_device=self.device,
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
        print(f"Teacher 配置: {model.teacher_status_message}")
        
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

    def set_phase_mode(self, phase):
        """按阶段切换 teacher 约束和检测头训练状态。"""
        enable_teacher_constraint = phase in {"phase1", "phase2"}
        detector_trainable = phase != "phase1"

        self.model.enable_constraint_loss(enable_teacher_constraint)
        for param in self.model.detector.parameters():
            param.requires_grad = detector_trainable
        if detector_trainable:
            self.model.detector.train()
        else:
            self.model.detector.eval()

        for param in self.model.optical_frontend.parameters():
            param.requires_grad = True

        if hasattr(self.model, "teacher") and self.config.FREEZE_TEACHER:
            self.model.teacher.eval()

    def get_constraint_weight(self, epoch, phase):
        """阶段1使用纯教师约束，阶段2使用带预热的联合约束。"""
        if phase == "phase1":
            return self.config.PHASE1_TEACHER_WEIGHT

        if phase == "phase2":
            warmup_epochs = max(1, self.config.TEACHER_WARMUP_EPOCHS)
            phase_epoch = epoch - self.config.PHASE1_EPOCHS
            progress = min(1.0, (phase_epoch + 1) / warmup_epochs)
            return self.config.TEACHER_CONSTRAINT_WEIGHT * progress

        return 0.0

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
        phase, _ = self.config.get_current_phase(epoch)
        self.model.train()
        self.set_phase_mode(phase)
        total_loss = 0.0
        num_batches = len(train_loader)
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS} [{phase}]")
        
        for batch_idx, (imgs, targets) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device)  # 将targets移动到GPU
            batch_size = imgs.shape[0]
            
            # 前向传播
            p3, p4, p5, optical_feature, constraint_target = self.model(imgs)
            preds = [p3, p4, p5]
            
            detection_loss = torch.zeros((), device=self.device)
            if phase != "phase1":
                for i, pred in enumerate(preds):
                    stride = self.config.STRIDES[i]
                    anchors = self.config.ANCHORS[i]
                    gt = build_target(targets, anchors, stride, self.num_classes, self.config.IMG_SIZE, self.device)
                    total_l, _, _, _ = self.criterion(pred, gt, batch_size)
                    detection_loss = detection_loss + total_l
            
            teacher_constraint_loss = torch.zeros((), device=self.device)
            constraint_weight = self.get_constraint_weight(epoch, phase)
            if constraint_target is not None and constraint_weight > 0:
                normalized_optical = self.normalize_feature_map(optical_feature)
                teacher_constraint_loss = self.optical_constraint_criterion(normalized_optical, constraint_target)

            if phase == "phase1":
                loss = constraint_weight * teacher_constraint_loss
            else:
                loss = detection_loss + constraint_weight * teacher_constraint_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            postfix = {
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss/(batch_idx+1):.4f}",
                "phase": phase,
            }
            
            if phase != "phase1":
                postfix["det"] = f"{detection_loss.item():.4f}"
            if constraint_weight > 0:
                postfix["teacher"] = f"{teacher_constraint_loss.item():.4f}"
                postfix["teacher_w"] = f"{constraint_weight:.3f}"
            
            pbar.set_postfix(postfix)
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss

    def validate(self, val_loader, epoch):
        """验证模型"""
        phase, _ = self.config.get_current_phase(epoch)
        self.model.eval()
        self.set_phase_mode(phase)
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
                
                detection_loss = torch.zeros((), device=self.device)
                if phase != "phase1":
                    for i, pred in enumerate(preds):
                        stride = self.config.STRIDES[i]
                        anchors = self.config.ANCHORS[i]
                        gt = build_target(targets, anchors, stride, self.num_classes, self.config.IMG_SIZE, self.device)
                        total_l, _, _, _ = self.criterion(pred, gt, batch_size)
                        detection_loss = detection_loss + total_l
                
                teacher_constraint_loss = torch.zeros((), device=self.device)
                constraint_weight = self.get_constraint_weight(epoch, phase)
                if constraint_target is not None and constraint_weight > 0:
                    normalized_optical = self.normalize_feature_map(optical_feature)
                    teacher_constraint_loss = self.optical_constraint_criterion(normalized_optical, constraint_target)

                if phase == "phase1":
                    loss = constraint_weight * teacher_constraint_loss
                else:
                    loss = detection_loss + constraint_weight * teacher_constraint_loss
                
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
            'teacher_status_message': getattr(self.model, 'teacher_status_message', ''),
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
            anchor_tensor = torch.tensor(
                self.config.ANCHORS[i], device=pred.device, dtype=pred.dtype
            )
            
            # 重塑预测为 (batch_size, grid_h, grid_w, 3, 5+num_classes)
            pred = pred.permute(0, 2, 3, 1).reshape(batch_size, grid_h, grid_w, 3, -1)
            
            obj_conf = torch.sigmoid(pred[..., 4])
            cls_probs = torch.sigmoid(pred[..., 5:])
            cls_conf, cls_id = cls_probs.max(dim=-1)
            final_conf = obj_conf * cls_conf
            obj_mask = final_conf > conf_thresh
            
            for b in range(batch_size):
                for h_idx in range(grid_h):
                    for w_idx in range(grid_w):
                        for anchor in range(3):
                            if obj_mask[b, h_idx, w_idx, anchor]:
                                bx, by, bw, bh = pred[b, h_idx, w_idx, anchor, :4]
                                conf = final_conf[b, h_idx, w_idx, anchor].item()
                                current_cls_id = cls_id[b, h_idx, w_idx, anchor].item()
                                anchor_w, anchor_h = anchor_tensor[anchor]

                                # 与 build_target/YOLOLoss 保持一致：xy 为网格内偏移，wh 为相对 anchor 的 log 编码。
                                x = float(
                                    torch.clamp(
                                        (torch.sigmoid(bx) + w_idx) * stride,
                                        0,
                                        self.config.IMG_SIZE - 1
                                    ).item()
                                )
                                y = float(
                                    torch.clamp(
                                        (torch.sigmoid(by) + h_idx) * stride,
                                        0,
                                        self.config.IMG_SIZE - 1
                                    ).item()
                                )
                                width = float(
                                    torch.clamp(
                                        torch.exp(torch.clamp(bw, min=-8.0, max=8.0)) * anchor_w,
                                        1,
                                        self.config.IMG_SIZE
                                    ).item()
                                )
                                height = float(
                                    torch.clamp(
                                        torch.exp(torch.clamp(bh, min=-8.0, max=8.0)) * anchor_h,
                                        1,
                                        self.config.IMG_SIZE
                                    ).item()
                                )

                                detections[b].append([x, y, width, height, conf, current_cls_id])

        final_detections = []
        for batch_detections in detections:
            if batch_detections:
                batch_detections = torch.tensor(batch_detections, dtype=torch.float32)
                keep = self.non_max_suppression(batch_detections, nms_thresh, max_det=max_det)
                kept = batch_detections[keep]
                final_detections.append(kept.tolist())
            else:
                final_detections.append([])
        
        return final_detections
    
    def xywh_to_xyxy(self, boxes):
        half_w = boxes[:, 2] / 2
        half_h = boxes[:, 3] / 2
        return torch.stack([
            boxes[:, 0] - half_w,
            boxes[:, 1] - half_h,
            boxes[:, 0] + half_w,
            boxes[:, 1] + half_h
        ], dim=1)

    def non_max_suppression(self, detections, nms_thresh, max_det=None):
        """按类别执行 NMS，避免不同类别之间互相抑制。"""
        if len(detections) == 0:
            return []

        boxes_xyxy = self.xywh_to_xyxy(detections[:, :4])
        scores = detections[:, 4]
        class_ids = detections[:, 5]
        keep = []

        for cls_id in class_ids.unique(sorted=False):
            cls_mask = class_ids == cls_id
            cls_indices = torch.where(cls_mask)[0]
            cls_keep = nms(boxes_xyxy[cls_mask], scores[cls_mask], nms_thresh)
            keep.append(cls_indices[cls_keep])

        if not keep:
            return []

        keep = torch.cat(keep)
        keep = keep[scores[keep].argsort(descending=True)]
        if max_det is not None:
            keep = keep[:max_det]
        return keep.tolist()
    
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
        """计算检测指标（单个批次）"""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for i in range(len(detections)):
            gt_boxes = []
            for t in targets[i]:
                if t[3] <= 0 or t[4] <= 0:
                    continue

                cls_id, cx, cy, bw, bh = t.tolist()
                gt_boxes.append([
                    cx * self.config.IMG_SIZE,
                    cy * self.config.IMG_SIZE,
                    bw * self.config.IMG_SIZE,
                    bh * self.config.IMG_SIZE,
                    int(cls_id)
                ])

            matched_gt = set()
            det_boxes = sorted(detections[i], key=lambda det: det[4], reverse=True)

            for det in det_boxes:
                det_tensor = torch.tensor(det[:4], dtype=torch.float32).unsqueeze(0)
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(gt_boxes):
                    if gt_idx in matched_gt or int(det[5]) != gt[4]:
                        continue

                    gt_tensor = torch.tensor(gt[:4], dtype=torch.float32).unsqueeze(0)
                    iou = float(self.calculate_iou(det_tensor, gt_tensor).item())
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= 0.5:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                else:
                    false_positives += 1

            false_negatives += len(gt_boxes) - len(matched_gt)
        
        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        
        return precision, recall, f1_score
    
    def calculate_full_detection_metrics(self, val_loader):
        """计算完整验证集的检测指标"""
        self.model.eval()
        all_true_positives = 0
        all_false_positives = 0
        all_false_negatives = 0
        
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(self.device, non_blocking=True)
                
                p3, p4, p5, _, _ = self.model(imgs)
                detections = self.decode_detections(
                    [p3, p4, p5], 
                    self.config.CONF_THRESH, 
                    self.config.NMS_THRESH
                )
                
                for i in range(len(detections)):
                    gt_boxes = []
                    for t in targets[i]:
                        if t[3] <= 0 or t[4] <= 0:
                            continue

                        cls_id, cx, cy, bw, bh = t.tolist()
                        gt_boxes.append([
                            cx * self.config.IMG_SIZE,
                            cy * self.config.IMG_SIZE,
                            bw * self.config.IMG_SIZE,
                            bh * self.config.IMG_SIZE,
                            int(cls_id)
                        ])

                    matched_gt = set()
                    det_boxes = sorted(detections[i], key=lambda det: det[4], reverse=True)

                    for det in det_boxes:
                        det_tensor = torch.tensor(det[:4], dtype=torch.float32).unsqueeze(0)
                        best_iou = 0.0
                        best_gt_idx = -1

                        for gt_idx, gt in enumerate(gt_boxes):
                            if gt_idx in matched_gt or int(det[5]) != gt[4]:
                                continue

                            gt_tensor = torch.tensor(gt[:4], dtype=torch.float32).unsqueeze(0)
                            iou = float(self.calculate_iou(det_tensor, gt_tensor).item())
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx

                        if best_iou >= 0.5:
                            all_true_positives += 1
                            matched_gt.add(best_gt_idx)
                        else:
                            all_false_positives += 1

                    all_false_negatives += len(gt_boxes) - len(matched_gt)
        
        precision = all_true_positives / (all_true_positives + all_false_positives + 1e-6)
        recall = all_true_positives / (all_true_positives + all_false_negatives + 1e-6)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        
        return precision, recall, f1_score

    def visualize_results(self, dataloader, epoch):
        """四列可视化：输入+GT、光学特征、教师约束、输出预测。"""
        self.model.eval()
        
        # 获取指定批次的数据
        if self.config.VISUALIZE_BATCH_INDEX >= 0:
            # 手动选择指定批次
            batch_iterator = iter(dataloader)
            for i in range(self.config.VISUALIZE_BATCH_INDEX + 1):
                imgs, targets = next(batch_iterator)
            print(f"使用手动选择的批次索引: {self.config.VISUALIZE_BATCH_INDEX}")
        else:
            # 真正的随机选择一个批次
            num_batches = len(dataloader)
            random_batch_idx = np.random.randint(0, num_batches)
            batch_iterator = iter(dataloader)
            for i in range(random_batch_idx + 1):
                imgs, targets = next(batch_iterator)
            print(f"使用随机选择的批次索引: {random_batch_idx}/{num_batches-1}")
        
        imgs = imgs[:4]  # 取前4个样本
        targets = targets[:4]  # 对应的目标
        
        with torch.no_grad():
            p3, p4, p5, optical_feat, constraint_target = self.model(imgs.to(self.device))
            
            detections = self.decode_detections(
                [p3, p4, p5],
                self.config.VIS_CONF_THRESH,
                self.config.NMS_THRESH,
                max_det=self.config.VIS_MAX_DETECTIONS
            )
        
        num_rows = min(4, imgs.shape[0])
        fig, axes = plt.subplots(
            num_rows,
            4,
            figsize=(17.5, 4.2 * num_rows),
            squeeze=False,
            gridspec_kw={'width_ratios': [1.45, 1, 1, 1.45]}
        )
        
        for idx in range(num_rows):
            img_np = imgs[idx].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            axes[idx, 0].imshow(img_np)
            h, w = img_np.shape[:2]
            target_data = targets[idx]

            gt_count = 0
            for t_idx in range(target_data.shape[0]):
                target = target_data[t_idx]
                if target[4] <= 0:
                    continue

                cls_id, cx, cy, bw, bh = target.cpu().numpy()
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='lime', linewidth=2)
                axes[idx, 0].add_patch(rect)
                class_name = self.class_names.get(int(cls_id), f"Class {int(cls_id)}")
                text_y = max(2, y1 - 10)
                axes[idx, 0].text(
                    x1,
                    text_y,
                    class_name,
                    color='white',
                    bbox=dict(facecolor='green', alpha=0.85, pad=1.5),
                    fontsize=7
                )
                gt_count += 1

            axes[idx, 0].set_title(f"Input + GT {idx + 1} ({gt_count})")
            axes[idx, 0].axis('off')

            optical_np = self.enhance_feature_for_display(
                self.normalize_feature_map(optical_feat[idx:idx + 1]).squeeze().cpu().numpy()
            )
            axes[idx, 1].imshow(optical_np, cmap='inferno')
            axes[idx, 1].set_title("Optical Feature")
            axes[idx, 1].axis('off')

            if constraint_target is not None:
                constraint_np = self.enhance_feature_for_display(constraint_target[idx].squeeze().cpu().numpy())
            else:
                constraint_np = np.zeros_like(optical_np)
            axes[idx, 2].imshow(constraint_np, cmap='inferno')
            axes[idx, 2].set_title("Teacher Constraint")
            axes[idx, 2].axis('off')

            axes[idx, 3].imshow(img_np)
            raw_detections = detections[idx] if idx < len(detections) else []
            img_detections = sorted(
                raw_detections,
                key=lambda det: det[4],
                reverse=True
            )[:self.config.VIS_MAX_DETECTIONS]
            for det in img_detections:
                x, y, w, h, conf, cls_id = det[:6]
                x1 = int(max(0, x - w / 2))
                y1 = int(max(0, y - h / 2))
                x2 = int(min(self.config.IMG_SIZE, x + w / 2))
                y2 = int(min(self.config.IMG_SIZE, y + h / 2))

                color = plt.cm.tab20(int(cls_id) / max(self.num_classes, 1))
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
                axes[idx, 3].add_patch(rect)

                class_name_det = self.class_names.get(int(cls_id), f"Class {int(cls_id)}")
                text_y = max(2, y1 - 10)
                axes[idx, 3].text(
                    x1,
                    text_y,
                    f"{class_name_det} {conf:.2f}",
                    color='white',
                    bbox=dict(facecolor=color, alpha=0.85, pad=1.5),
                    fontsize=7
                )
            axes[idx, 3].set_title(f"Output + Pred {idx + 1} ({len(img_detections)})")
            axes[idx, 3].axis('off')
        
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

    def save_config_to_txt(self):
        """将训练配置参数保存为txt文件"""
        with open(self.config_log_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("光学YOLOv3训练配置参数\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("设备设置\n")
            f.write("-"*80 + "\n")
            f.write(f"DEVICE: {self.config.DEVICE}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("训练参数\n")
            f.write("-"*80 + "\n")
            f.write(f"EPOCHS: {self.config.EPOCHS}\n")
            f.write(f"BATCH_SIZE: {self.config.BATCH_SIZE}\n")
            f.write(f"IMG_SIZE: {self.config.IMG_SIZE}\n")
            f.write(f"LEARNING_RATE: {self.config.LEARNING_RATE}\n")
            f.write(f"WEIGHT_DECAY: {self.config.WEIGHT_DECAY}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("损失权重\n")
            f.write("-"*80 + "\n")
            f.write(f"BOX_WEIGHT: {self.config.BOX_WEIGHT}\n")
            f.write(f"OBJ_WEIGHT: {self.config.OBJ_WEIGHT}\n")
            f.write(f"CLS_WEIGHT: {self.config.CLS_WEIGHT}\n")
            f.write(f"PHASE1_TEACHER_WEIGHT: {self.config.PHASE1_TEACHER_WEIGHT}\n")
            f.write(f"TEACHER_CONSTRAINT_WEIGHT: {self.config.TEACHER_CONSTRAINT_WEIGHT}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("光学设置\n")
            f.write("-"*80 + "\n")
            f.write(f"USE_VORTEX_INIT: {self.config.USE_VORTEX_INIT}\n")
            f.write(f"SLM1_VORTEX_CHARGE: {self.config.SLM1_VORTEX_CHARGE}\n")
            f.write(f"SLM2_VORTEX_CHARGE: {self.config.SLM2_VORTEX_CHARGE}\n")
            f.write(f"VORTEX_PERTURBATION: {self.config.VORTEX_PERTURBATION}\n\n")

            f.write("-"*80 + "\n")
            f.write("教师设置\n")
            f.write("-"*80 + "\n")
            f.write(f"TEACHER_CHECKPOINT: {self.config.TEACHER_CHECKPOINT}\n")
            f.write(f"TEACHER_INIT_MODE: {self.config.TEACHER_INIT_MODE}\n")
            f.write(f"FREEZE_TEACHER: {self.config.FREEZE_TEACHER}\n")
            f.write(f"TEACHER_STATUS: {getattr(self.model, 'teacher_status_message', '')}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("锚框设置\n")
            f.write("-"*80 + "\n")
            f.write(f"STRIDES: {self.config.STRIDES}\n")
            f.write("ANCHORS:\n")
            for i, anchors in enumerate(self.config.ANCHORS):
                f.write(f"  P{i+3} (stride={self.config.STRIDES[i]}): {anchors}\n")
            f.write("\n")
            
            f.write("-"*80 + "\n")
            f.write("路径设置\n")
            f.write("-"*80 + "\n")
            f.write(f"DATA_YAML_PATH: {self.config.DATA_YAML_PATH}\n")
            f.write(f"ROOT_PATH: {self.config.ROOT_PATH}\n")
            f.write(f"OPTICAL_YOLO_OUTPUT_DIR: {self.config.OPTICAL_YOLO_OUTPUT_DIR}\n")
            f.write(f"SAVE_DIR: {self.config.SAVE_DIR}\n")
            f.write(f"LOG_DIR: {self.config.LOG_DIR}\n")
            f.write(f"VISUALIZATION_DIR: {self.config.VISUALIZATION_DIR}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("训练策略\n")
            f.write("-"*80 + "\n")
            f.write(f"ENABLE_NORM_AFTER_EPOCH: {self.config.ENABLE_NORM_AFTER_EPOCH}\n")
            f.write(f"PHASE1_EPOCHS: {self.config.PHASE1_EPOCHS}\n")
            f.write(f"PHASE2_EPOCHS: {self.config.PHASE2_EPOCHS}\n")
            f.write(f"TEACHER_WARMUP_EPOCHS: {self.config.TEACHER_WARMUP_EPOCHS}\n")
            f.write(f"VISUALIZE_EVERY: {self.config.VISUALIZE_EVERY}\n")
            f.write(f"SAVE_EVERY: {self.config.SAVE_EVERY}\n")
            f.write(f"EARLY_STOPPING_PATIENCE: {self.config.EARLY_STOPPING_PATIENCE}\n")
            f.write(f"EARLY_STOPPING_MIN_DELTA: {self.config.EARLY_STOPPING_MIN_DELTA}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("检测阈值\n")
            f.write("-"*80 + "\n")
            f.write(f"CONF_THRESH: {self.config.CONF_THRESH}\n")
            f.write(f"NMS_THRESH: {self.config.NMS_THRESH}\n")
            f.write(f"VIS_CONF_THRESH: {self.config.VIS_CONF_THRESH}\n")
            f.write(f"VIS_MAX_DETECTIONS: {self.config.VIS_MAX_DETECTIONS}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("类别信息\n")
            f.write("-"*80 + "\n")
            f.write(f"类别数量: {self.num_classes}\n")
            f.write("类别名称:\n")
            for idx, name in self.class_names.items():
                f.write(f"  {idx}: {name}\n")
            f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("配置参数保存完成\n")
            f.write("="*80 + "\n")
        
        print(f"配置参数已保存: {self.config_log_path}")

    def log_training_epoch(self, epoch, phase, train_loss, val_loss, precision, recall, f1_score, lr, constraint_weight=0.0):
        """记录每个epoch的训练日志到txt文件"""
        with open(self.training_log_path, 'a', encoding='utf-8') as f:
            if epoch == 0:
                f.write("="*80 + "\n")
                f.write("光学YOLOv3训练日志\n")
                f.write("="*80 + "\n\n")
                f.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Epoch':<8} {'Phase':<10} {'Train Loss':<12} {'Val Loss':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'LR':<12} {'Teacher W':<12}\n")
                f.write("-"*80 + "\n")
            
            f.write(f"{epoch+1:<8} {phase:<10} {train_loss:<12.4f} {val_loss:<12.4f} {precision:<10.3f} {recall:<10.3f} {f1_score:<10.3f} {lr:<12.6f} {constraint_weight:<12.4f}\n")
    
    def save_training_summary(self, total_time, best_epoch, best_val_loss):
        """保存训练总结到txt文件"""
        with open(self.summary_log_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("光学YOLOv3训练总结\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总训练时间: {total_time/60:.2f} 分钟 ({total_time:.2f} 秒)\n\n")
            
            f.write("-"*80 + "\n")
            f.write("训练统计\n")
            f.write("-"*80 + "\n")
            f.write(f"总Epoch数: {len(self.train_losses)}\n")
            f.write(f"最佳Epoch: {best_epoch + 1}\n")
            f.write(f"最佳验证损失: {best_val_loss:.4f}\n")
            f.write(f"最终训练损失: {self.train_losses[-1]:.4f}\n")
            f.write(f"最终验证损失: {self.val_losses[-1]:.4f}\n\n")
            
            if self.precisions:
                f.write("-"*80 + "\n")
                f.write("检测指标统计\n")
                f.write("-"*80 + "\n")
                f.write(f"最佳精确率: {max(self.precisions):.3f}\n")
                f.write(f"最佳召回率: {max(self.recalls):.3f}\n")
                f.write(f"最佳F1分数: {max(self.f1_scores):.3f}\n")
                f.write(f"最终精确率: {self.precisions[-1]:.3f}\n")
                f.write(f"最终召回率: {self.recalls[-1]:.3f}\n")
                f.write(f"最终F1分数: {self.f1_scores[-1]:.3f}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("损失曲线统计\n")
            f.write("-"*80 + "\n")
            f.write(f"训练损失范围: [{min(self.train_losses):.4f}, {max(self.train_losses):.4f}]\n")
            f.write(f"训练损失均值: {np.mean(self.train_losses):.4f}\n")
            f.write(f"训练损失标准差: {np.std(self.train_losses):.4f}\n")
            if self.val_losses:
                f.write(f"验证损失范围: [{min(self.val_losses):.4f}, {max(self.val_losses):.4f}]\n")
                f.write(f"验证损失均值: {np.mean(self.val_losses):.4f}\n")
                f.write(f"验证损失标准差: {np.std(self.val_losses):.4f}\n")
            f.write("\n")
            
            f.write("-"*80 + "\n")
            f.write("文件路径\n")
            f.write("-"*80 + "\n")
            f.write(f"模型保存目录: {self.config.SAVE_DIR}\n")
            f.write(f"日志保存目录: {self.config.LOG_DIR}\n")
            f.write(f"可视化保存目录: {self.config.VISUALIZATION_DIR}\n")
            f.write(f"配置文件: {self.config_log_path}\n")
            f.write(f"训练日志: {self.training_log_path}\n")
            f.write(f"总结文件: {self.summary_log_path}\n\n")
            
            f.write("="*80 + "\n")
            f.write("训练总结保存完成\n")
            f.write("="*80 + "\n")
        
        print(f"训练总结已保存: {self.summary_log_path}")

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
            phase, phase_desc = self.config.get_current_phase(epoch)
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.config.EPOCHS}")
            print(f"阶段: {phase_desc}")
            print(f"{'='*50}")
            
            if epoch >= self.config.ENABLE_NORM_AFTER_EPOCH:
                self.model.enable_normalization(True)
                print("已启用光学输出归一化")
            else:
                self.model.enable_normalization(False)

            self.set_phase_mode(phase)
            teacher_weight = self.get_constraint_weight(epoch, phase)
            if phase in {"phase1", "phase2"}:
                print(f"已启用教师约束，当前权重: {teacher_weight:.4f}")
            else:
                print("当前阶段不使用教师约束")
            
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss = self.validate(val_loader, epoch)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            if phase != "phase1":
                precision, recall, f1_score = self.calculate_full_detection_metrics(val_loader)
                self.precisions.append(precision)
                self.recalls.append(recall)
                self.f1_scores.append(f1_score)
            else:
                precision = recall = f1_score = 0
            
            self.log_training_epoch(epoch, phase, train_loss, val_loss, precision, recall, f1_score, current_lr, teacher_weight)
            
            if phase == "phase1":
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
        best_epoch = len(self.train_losses) - self.no_improve_epochs - 1
        
        print(f"\n{'='*50}")
        print("训练完成!")
        print(f"总训练时间: {training_time/60:.2f} 分钟")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"模型保存在: {self.config.SAVE_DIR}")
        print(f"日志保存在: {self.config.LOG_DIR}")
        
        # 保存训练总结
        self.save_training_summary(training_time, best_epoch, self.best_val_loss)
        
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
