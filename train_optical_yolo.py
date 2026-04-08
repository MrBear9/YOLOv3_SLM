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
    WEIGHT_DECAY = 1e-4
    
    # 损失权重
    BOX_WEIGHT = 0.05
    OBJ_WEIGHT = 1.5
    CLS_WEIGHT = 0.15
    
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
    SAVE_DIR = r"output\optical_yolo_models"
    LOG_DIR = r"output\training_logs"
    
    # 训练策略
    ENABLE_NORM_AFTER_EPOCH = 50  # 50轮后启用归一化
    VISUALIZE_EVERY = 5  # 每5轮可视化一次
    SAVE_EVERY = 10  # 每10轮保存一次模型

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
        
        # 加载类别信息
        self.class_names, self.num_classes = self.load_class_names()
        print(f"类别数量: {self.num_classes}, 类别名称: {self.class_names}")
        
        # 初始化模型
        self.model = self.init_model()
        self.criterion = YOLOLoss(config.BOX_WEIGHT, config.OBJ_WEIGHT, config.CLS_WEIGHT)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
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
            optical_mode="phase"  # 使用纯相位调制
        ).to(self.device)
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params:,}")
        
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
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False, 
            collate_fn=self.collate_fn,
            num_workers=2
        )
        
        return train_loader, val_loader

    def collate_fn(self, batch):
        """自定义批次处理函数"""
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs)
        
        # 对齐targets的长度
        max_len = max([t.shape[0] for t in targets])
        target_tensor = torch.zeros(len(imgs), max_len, 5, device=self.device)
        
        for i, t in enumerate(targets):
            if t.shape[0] > 0:
                target_tensor[i, :t.shape[0]] = t.to(self.device)
        
        return imgs, target_tensor

    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}")
        
        for batch_idx, (imgs, targets) in enumerate(pbar):
            imgs = imgs.to(self.device)
            batch_size = imgs.shape[0]
            
            # 前向传播
            p3, p4, p5, _ = self.model(imgs)
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
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss/(batch_idx+1):.4f}"
            })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss

    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(self.device)
                batch_size = imgs.shape[0]
                
                # 前向传播
                p3, p4, p5, _ = self.model(imgs)
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

    def visualize_results(self, dataloader, epoch):
        """可视化训练结果"""
        self.model.eval()
        
        # 获取一个批次的数据
        imgs, targets = next(iter(dataloader))
        imgs = imgs[:4]  # 取前4个样本
        
        with torch.no_grad():
            p3, p4, p5, optical_feat = self.model(imgs.to(self.device))
        
        # 创建可视化
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        
        for idx in range(min(4, imgs.shape[0])):
            # 输入图像
            img_np = imgs[idx].permute(1, 2, 0).cpu().numpy()
            axes[idx, 0].imshow(img_np)
            axes[idx, 0].set_title(f"Input {idx+1}")
            axes[idx, 0].axis('off')
            
            # 光学特征
            optical_np = optical_feat[idx].squeeze().cpu().numpy()
            im = axes[idx, 1].imshow(optical_np, cmap='hot')
            axes[idx, 1].set_title("Optical Feature")
            axes[idx, 1].axis('off')
            plt.colorbar(im, ax=axes[idx, 1])
            
            # 多尺度特征
            p3_np = p3[idx].abs().mean(dim=0).cpu().numpy()
            axes[idx, 2].imshow(p3_np, cmap='hot')
            axes[idx, 2].set_title("P3 (80×80)")
            axes[idx, 2].axis('off')
            
            p4_np = p4[idx].abs().mean(dim=0).cpu().numpy()
            axes[idx, 3].imshow(p4_np, cmap='hot')
            axes[idx, 3].set_title("P4 (40×40)")
            axes[idx, 3].axis('off')
            
            p5_np = p5[idx].abs().mean(dim=0).cpu().numpy()
            axes[idx, 4].imshow(p5_np, cmap='hot')
            axes[idx, 4].set_title("P5 (20×20)")
            axes[idx, 4].axis('off')
        
        plt.tight_layout()
        vis_path = os.path.join(self.config.LOG_DIR, f"visualization_epoch_{epoch+1}.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
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
        
        history_path = os.path.join(self.config.LOG_DIR, "training_history.png")
        plt.savefig(history_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"训练历史图已保存: {history_path}")

    def train(self):
        """主训练循环"""
        print("开始训练光学YOLOv3模型...")
        
        # 创建数据加载器
        train_loader, val_loader = self.create_dataloaders()
        
        # 训练开始时间
        start_time = time.time()
        
        for epoch in range(self.config.EPOCHS):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.config.EPOCHS}")
            print(f"{'='*50}")
            
            # 训练策略调整
            if epoch >= self.config.ENABLE_NORM_AFTER_EPOCH:
                self.model.enable_normalization(True)
                print("已启用光学输出归一化")
            
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 学习率: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(epoch, val_loss, is_best=True)
                print("新的最佳模型已保存!")
            
            # 定期保存和可视化
            if (epoch + 1) % self.config.SAVE_EVERY == 0:
                self.save_model(epoch, val_loss)
            
            if (epoch + 1) % self.config.VISUALIZE_EVERY == 0:
                self.visualize_results(val_loader, epoch)
        
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
