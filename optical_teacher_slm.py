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
        phase = torch.remainder(self.phase_raw, 2 * np.pi)
        mod = torch.exp(1j * phase)

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
# 教师网络（从last.py复制的ConvTeacher）
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
# 数据集（使用military数据集）
# =========================================================
class MilitaryFeatureDataset(Dataset):
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
    
    if idx is None:
        batch_size = min(4, teacher.shape[0])
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
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[Vis] saved: {save_path}")

# =========================================================
# 训练函数
# =========================================================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 初始化教师网络（从last.py复制的ConvTeacher）
    print("初始化教师网络（ConvTeacher）...")
    teacher = ConvTeacher()
    
    # 冻结教师网络的参数
    for p in teacher.parameters():
        p.requires_grad = False
    print("教师网络参数已冻结")
    
    # 初始化学生网络（光学相位层）
    print("初始化学生网络（OpticalStudent）...")
    student = OpticalStudent((640, 640), mode="phase").to(device)
    
    # 加载military数据集
    print("加载数据集...")
    dataset = MilitaryFeatureDataset("data/military/data.yaml", teacher, device)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"数据集大小: {len(dataset)}")
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(student.parameters(), lr=5e-4, weight_decay=1e-5)
    criterion = MultiScaleMSELoss()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/OpticalTeacher_{timestamp}"
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"结果将保存到: {output_dir}")
    
    # 训练参数
    EPOCHS = 400
    loss_curve = []
    
    print("="*60)
    print("开始训练光学相位层...")
    print("="*60)
    
    for epoch in range(EPOCHS):
        student.train()
        loss_sum = 0
        
        for x, t in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
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
        
        # 在第50轮后启用归一化
        if epoch == 50:
            student.enable_norm = True
            print(f"Epoch {epoch}: 启用归一化")
        
        # 每2轮进行可视化
        if epoch % 2 == 0:
            student.eval()
            with torch.no_grad():
                vis_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
                vis_batch = next(iter(vis_loader))
                x_vis, t_vis = vis_batch
                x_vis = x_vis.to(device)
                t_vis = t_vis
                y_vis = student(x_vis).cpu()
            save_feature_comparison(epoch, t_vis, y_vis, vis_dir, input_images=x_vis.cpu())

        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f}")

    # 保存损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(loss_curve, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=120)
    plt.close()

    # 保存训练好的模型
    model_save_path = os.path.join(output_dir, "optical_student_final.pth")
    torch.save(student.state_dict(), model_save_path)
    
    print("="*60)
    print("训练完成！")
    print(f"模型已保存到: {model_save_path}")
    print(f"所有结果已保存到: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    train()
