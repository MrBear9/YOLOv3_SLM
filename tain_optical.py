import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import yaml
import matplotlib.pyplot as plt
from datetime import datetime

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
        # Phase modulation (always enabled)
        phase = torch.remainder(self.phase_raw, 2 * np.pi)
        mod = torch.exp(1j * phase)

        # Optional amplitude modulation
        if self.mode == "amp_phase":
            amp = torch.sigmoid(self.amp_raw)
            mod = mod * amp

        return field * mod

class ASMPropagation(nn.Module):
    def __init__(self, distance, wavelength, pixel_size, resolution):
        super().__init__()
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
    def __init__(self, resolution, mode="amp_phase"):
        super().__init__()
        self.slm1 = SLMLayer(resolution, mode)
        self.prop1 = ASMPropagation(0.01, 532e-9, 6.4e-6, resolution)
        self.slm2 = SLMLayer(resolution, mode)
        self.prop2 = ASMPropagation(0.02, 532e-9, 6.4e-6, resolution)
        # self.slm3 = SLMLayer(resolution)
        # self.prop3 = ASMPropagation(0.03, 532e-9, 6.4e-6, resolution)
        self.enable_norm = False

    def forward(self, intensity):
        amp = torch.sqrt(intensity.clamp(min=0)+1e-8)
        field = torch.complex(amp, torch.zeros_like(amp))
        field = self.prop1(self.slm1(field))
        field = self.prop2(self.slm2(field))
        # field = self.prop3(self.slm3(field))
        out = torch.abs(field)**2
        if self.enable_norm:
            out = out / (out.mean(dim=[2,3], keepdim=True) + 1e-6)
        return out

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
        g = torch.exp(-(X**2 + Y**2)/(2*sigma**2))
        g = g/g.sum()
        self.gauss.weight.data = g.unsqueeze(0).unsqueeze(0)

        # Sobel 边缘
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        self.sobel = nn.Conv2d(1,1,kernel_size=3,padding=1,bias=False)
        self.sobel.weight.data = sobel_x.unsqueeze(0).unsqueeze(0)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        low = self.gauss(gray)
        edge = torch.abs(self.sobel(gray))   # 去符号
        edge = F.avg_pool2d(edge, 4)         # 去高频
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
        f = self.conv3(f)      # P3 scale (H/8, W/8)

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

# =========================================================
# Dataset
# =========================================================
class COCOFeatureDataset(Dataset):
    def __init__(self, yaml_path, teacher, device):
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
            transforms.Resize((640, 640)),
            transforms.Grayscale(1),
            transforms.ToTensor()
        ])
        self.rgb = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        x = self.gray(img)
        with torch.no_grad():
            t = self.teacher(self.rgb(img).unsqueeze(0).to(self.device))
            t = F.interpolate(t, size=(640, 640), mode="bilinear", align_corners=False)
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
        return loss_full*0.02 + 1 * loss_low1 + 0.5 * loss_low2

# =========================================================
# 可视化工具
# =========================================================
def save_feature_comparison(epoch, teacher, student, save_dir, prefix="train", idx=None, input_images=None):
    os.makedirs(save_dir, exist_ok=True)
    
    # 如果没有指定idx，随机选择4张图
    if idx is None:
        batch_size = min(4, teacher.shape[0])  # 确保不超过批次大小
        indices = torch.randperm(teacher.shape[0])[:batch_size]
    else:
        indices = [idx] if isinstance(idx, int) else idx
    
    # 创建4个子图，每行4个，共4行（输入、教师、学生、差异）
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    
    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    for i, idx in enumerate(indices):
        # 获取输入图像（如果提供）
        if input_images is not None:
            input_img = input_images[idx, 0].numpy()
            axes[i, 0].imshow(norm(input_img), cmap="gray")
            axes[i, 0].set_title(f"Input {i+1}")
            axes[i, 0].axis("off")
        else:
            # 如果没有输入图像，隐藏该位置
            axes[i, 0].axis("off")
        
        t = teacher[idx, 0].numpy()
        s = student[idx, 0].numpy()
        r = np.abs(s - t)
        
        # 显示教师特征
        axes[i, 1].imshow(norm(t), cmap="hot")
        axes[i, 1].set_title(f"Teacher {i+1}")
        axes[i, 1].axis("off")
        
        # 显示学生特征
        axes[i, 2].imshow(norm(s), cmap="hot")
        axes[i, 2].set_title(f"Student {i+1}")
        axes[i, 2].axis("off")
        
        # 显示差异图
        im = axes[i, 3].imshow(r, cmap="viridis")
        axes[i, 3].set_title(f"Error {i+1}")
        axes[i, 3].axis("off")
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046)
    
    # 隐藏多余的子图
    for i in range(len(indices), 4):
        for j in range(4):
            axes[i, j].axis("off")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_epoch_{epoch:03d}.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[Vis] saved: {save_path}")

# =========================================================
# 训练
# =========================================================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # teacher = MultiScaleTeacher()
    teacher = ConvTeacher()
    print("="*60)
    print("OpticalStudent with phase mode")
    student = OpticalStudent((640, 640), mode="phase").to(device)
    print("="*60)
    dataset = COCOFeatureDataset("data/data.yaml", teacher, device)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(student.parameters(), lr=5e-4, weight_decay=1e-5)
    criterion = MultiScaleMSELoss()
    
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = f"output/vis_{timestamp}"
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Results will be saved to: {vis_dir}")
    
    loss_curve = []

    for epoch in range(400):
        student.train()
        loss_sum = 0
        for x, t in tqdm(loader, desc=f"Epoch {epoch}"):
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
        if epoch == 50:
            student.enable_norm = True
        if epoch % 2 == 0:
            student.eval()
            with torch.no_grad():
                # 获取一批数据进行可视化
                vis_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
                vis_batch = next(iter(vis_loader))
                x_vis, t_vis = vis_batch
                x_vis = x_vis.to(device)
                t_vis = t_vis
                y_vis = student(x_vis).cpu()
            save_feature_comparison(epoch, t_vis, y_vis, vis_dir, input_images=x_vis.cpu())

        print(f"Epoch {epoch} | Loss {avg_loss:.6f}")

    # 保存损失曲线
    plt.figure()
    plt.plot(loss_curve, label="Train Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig(os.path.join(vis_dir, "loss_curve.png"), dpi=120)
    plt.close()

    torch.save(student.state_dict(), os.path.join(vis_dir, "optical_student_final.pth"))
    print(f"Training completed! Model saved to: {os.path.join(vis_dir, 'optical_student_final.pth')}")
    print(f"All results saved to: {vis_dir}")


if __name__ == "__main__":
    train()
