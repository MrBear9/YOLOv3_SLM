import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datetime import datetime


# =========================================================
# 教师网络（多尺度滤波器）
# =========================================================
class MultiScaleTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        # 高斯低通
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

        for p in self.parameters():
            p.requires_grad = False

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

        # Freeze parameters
        for p in self.parameters():
            p.requires_grad = False

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

        # Bounded intensity-like output (归一化到0-1)
        return torch.sigmoid(f)


# =========================================================
# 数据集（适配YOLO格式的自定义数据集）
# =========================================================
class YOLOStyleDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        适配YOLO格式的数据集类
        Args:
            root_dir: 数据集根目录 (data/military)
            split: 数据集划分 ('train', 'test', 'val')
            transform: 图片变换
        """

        self.root_dir = root_dir
        self.split = split
        self.images_dir = os.path.join(root_dir, split, 'images')

        # 获取所有图片文件
        self.image_files = sorted([
            os.path.join(self.images_dir, f)
            for f in os.listdir(self.images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif"))
        ])

        # 检查数据集是否为空
        if len(self.image_files) == 0:
            raise ValueError(f"未在 {self.images_dir} 目录下找到图片文件！")

        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读取图片
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")

        # 应用变换
        x = self.transform(img)

        # 返回输入图片（教师网络的输入）
        return x, x


# =========================================================
# 可视化工具（2×2特征对比图）
# =========================================================
def save_feature_visualization(epoch, input_images, teacher_outputs, save_dir, prefix="teacher"):
    os.makedirs(save_dir, exist_ok=True)

    # 选择最多4张图进行可视化
    batch_size = min(4, input_images.shape[0])
    indices = torch.randperm(input_images.shape[0])[:batch_size]

    # 创建2×2子图（输入图片，教师特征图）
    fig, axes = plt.subplots(2, batch_size, figsize=(4 * batch_size, 8))

    # 归一化函数
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    for i, idx in enumerate(indices):
        # 显示输入图片
        input_img = input_images[idx].permute(1, 2, 0).numpy()  # CHW → HWC
        input_img = norm(input_img)
        axes[0, i].imshow(input_img)
        axes[0, i].set_title(f"Input {i + 1}")
        axes[0, i].axis("off")

        # 显示教师网络输出的特征图
        teacher_feat = teacher_outputs[idx, 0].numpy()
        teacher_feat = norm(teacher_feat)
        im = axes[1, i].imshow(teacher_feat, cmap="hot")
        axes[1, i].set_title(f"Teacher Feature {i + 1}")
        axes[1, i].axis("off")

        # 添加颜色条
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_epoch_{epoch:03d}.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[Vis] saved: {save_path}")


# =========================================================
# 主特征提取流程
# =========================================================
def extract_teacher_features():
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 初始化教师网络（可以选择使用MultiScaleTeacher或ConvTeacher）
    # teacher = MultiScaleTeacher().to(device)
    teacher = ConvTeacher().to(device)
    teacher.eval()  # 教师网络参数冻结，仅用于特征提取
    print("=" * 60)
    print("Using ConvTeacher (YOLOv3-like) for feature extraction")
    print("=" * 60)

    # ==================== 数据集配置 ====================
    # 数据集根目录（已设置为你的路径）
    DATASET_ROOT = r"data/military"
    # 选择使用的数据集划分 (train/test/val)
    DATA_SPLIT = "train"
    # ====================================================

    # 验证数据集路径是否存在
    if not os.path.exists(DATASET_ROOT):
        raise FileNotFoundError(f"数据集根目录不存在: {DATASET_ROOT}")

    # 创建数据集和数据加载器
    try:
        dataset = YOLOStyleDataset(
            root_dir=DATASET_ROOT,
            split=DATA_SPLIT,
            transform=None  # 使用默认变换
        )
        print(f"成功加载 {DATA_SPLIT} 数据集，共 {len(dataset)} 张图片")
    except ValueError as e:
        print(f"数据集加载失败: {e}")
        return

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)  # Windows下num_workers设为0

    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = f"output/teacher_vis_{timestamp}"
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Results will be saved to: {vis_dir}")

    # 记录特征图均值曲线（替代损失曲线）
    feat_mean_curve = []
    # 归一化开关
    enable_norm = False

    # 特征提取轮次
    num_epochs = 400
    for epoch in range(num_epochs):
        mean_sum = 0

        # 使用tqdm显示进度
        with tqdm(loader, desc=f"Epoch {epoch}") as pbar:
            for x, _ in pbar:
                x = x.to(device)

                # 教师网络前向传播提取特征
                with torch.no_grad():
                    teacher_feat = teacher(x)

                    # 第50轮开启归一化
                    if enable_norm:
                        teacher_feat = teacher_feat / (teacher_feat.mean(dim=[2, 3], keepdim=True) + 1e-6)

                # 计算当前批次特征图的均值
                batch_mean = teacher_feat.mean().item()
                mean_sum += batch_mean

                # 更新进度条
                pbar.set_postfix({"feat_mean": batch_mean})

        # 计算本轮平均特征图均值
        avg_mean = mean_sum / len(loader)
        feat_mean_curve.append(avg_mean)

        # 第50轮开启归一化
        if epoch == 50:
            enable_norm = True
            print("Enabled output normalization for stability")

        # 每2轮可视化一次结果
        if epoch % 2 == 0:
            # 获取一批可视化数据
            vis_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
            vis_batch = next(iter(vis_loader))
            x_vis, _ = vis_batch
            x_vis = x_vis.to(device)

            # 教师网络提取特征
            with torch.no_grad():
                teacher_vis = teacher(x_vis)
                if enable_norm:
                    teacher_vis = teacher_vis / (teacher_vis.mean(dim=[2, 3], keepdim=True) + 1e-6)

            # 保存可视化结果
            save_feature_visualization(epoch, x_vis.cpu(), teacher_vis.cpu(), vis_dir)

        print(f"Epoch {epoch} | Avg Feature Mean {avg_mean:.6f}")

    # 保存特征图均值曲线（替代损失曲线）
    plt.figure(figsize=(10, 6))
    plt.plot(feat_mean_curve, label="Feature Map Mean Value", color='green')
    plt.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='Normalization Enabled')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Value of Feature Map")
    plt.title("Teacher Network Feature Map Mean Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "feature_mean_curve.png"), dpi=120)
    plt.close()

    # 保存教师网络状态
    torch.save(teacher.state_dict(), os.path.join(vis_dir, "teacher_network.pth"))

    print("=" * 60)
    print(f"Feature extraction completed!")
    print(f"Teacher model saved to: {os.path.join(vis_dir, 'teacher_network.pth')}")
    print(f"Feature mean curve saved to: {os.path.join(vis_dir, 'feature_mean_curve.png')}")
    print(f"All visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    # 运行教师网络特征提取
    extract_teacher_features()