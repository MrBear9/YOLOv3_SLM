import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# =========================================================
# 光学层（相位调制 + 角谱传播）
# =========================================================
class SLMLayer(nn.Module):
    """空间光调制器层 - 相位调制"""
    def __init__(self, resolution, mode="phase"):
        super().__init__()
        assert mode in ["phase", "amp_phase"]
        self.mode = mode

        # 相位参数初始化 [0, 2π]
        self.phase_raw = nn.Parameter(
            torch.rand(1, 1, *resolution) * 2 * np.pi
        )

        # 可选振幅调制
        if self.mode == "amp_phase":
            self.amp_raw = nn.Parameter(
                torch.rand(1, 1, *resolution)
            )
        else:
            self.register_parameter("amp_raw", None)

    def forward(self, field):
        # 相位调制
        phase = torch.remainder(self.phase_raw, 2 * np.pi)
        mod = torch.exp(1j * phase)

        # 可选振幅调制
        if self.mode == "amp_phase":
            amp = torch.sigmoid(self.amp_raw)
            mod = mod * amp

        return field * mod

class ASMPropagation(nn.Module):
    """角谱传播模型"""
    def __init__(self, distance, wavelength, pixel_size, resolution):
        super().__init__()
        # 计算频域传递函数
        fx = torch.fft.fftfreq(resolution[0], pixel_size)
        fy = torch.fft.fftfreq(resolution[1], pixel_size)
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')
        k2 = 1 / wavelength**2 - FX**2 - FY**2
        k2 = torch.clamp(k2, min=0)
        H = torch.exp(1j * 2 * np.pi * distance * torch.sqrt(k2))
        self.register_buffer("H", H)

    def forward(self, field):
        # 傅里叶光学传播：FFT → 频域相乘 → IFFT
        return torch.fft.ifft2(torch.fft.fft2(field) * self.H)

# =========================================================
# 光学前端（两层相位调制）
# =========================================================
class OpticalFrontend(nn.Module):
    """光学前端 - 两层相位调制系统"""
    def __init__(self, resolution=(640, 640), mode="phase"):
        super().__init__()
        self.slm1 = SLMLayer(resolution, mode)
        self.prop1 = ASMPropagation(0.01, 532e-9, 6.4e-6, resolution)  # 10mm传播
        self.slm2 = SLMLayer(resolution, mode)
        self.prop2 = ASMPropagation(0.02, 532e-9, 6.4e-6, resolution)  # 20mm传播
        self.enable_norm = False

    def forward(self, intensity):
        """输入强度图，输出调制后的强度图"""
        # 强度转振幅场
        amp = torch.sqrt(intensity.clamp(min=0) + 1e-8)
        field = torch.complex(amp, torch.zeros_like(amp))
        
        # 两层光学调制
        field = self.prop1(self.slm1(field))
        field = self.prop2(self.slm2(field))
        
        # 输出强度图
        out = torch.abs(field)**2
        
        # 可选归一化
        if self.enable_norm:
            out = out / (out.mean(dim=[2, 3], keepdim=True) + 1e-6)
        
        return out

# =========================================================
# 轻量化卷积块（YOLOv3检测头组件）
# =========================================================
class LightConvBlock(nn.Module):
    """轻量化卷积块：深度可分离卷积 + BN + SiLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(LightConvBlock, self).__init__()
        # 深度卷积（逐通道卷积）
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                 stride, padding, groups=in_channels, bias=False)
        # 点卷积（1×1卷积）
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # 轻量化激活函数

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

# =========================================================
# YOLOv3检测头（多尺度特征生成）
# =========================================================
class YOLOLightHead(nn.Module):
    """轻量化YOLOv3检测头 - 基于FPN的多尺度特征生成"""
    def __init__(self, in_channels=1, out_channels=27):
        super(YOLOLightHead, self).__init__()
        # 基础通道数（轻量化设计）
        base_ch = 32

        # 1. 初始特征提取
        self.init_conv = LightConvBlock(in_channels, base_ch, kernel_size=3, stride=1)

        # 2. 下采样生成P5（20×20）：640→320→160→80→40→20
        self.down_to_p5 = nn.Sequential(
            LightConvBlock(base_ch, base_ch * 2, stride=2),  # 640→320, 32→64
            LightConvBlock(base_ch * 2, base_ch * 4, stride=2),  # 320→160, 64→128
            LightConvBlock(base_ch * 4, base_ch * 8, stride=2),  # 160→80, 128→256
            LightConvBlock(base_ch * 8, base_ch * 8, stride=2),  # 80→40, 256→256
            LightConvBlock(base_ch * 8, base_ch * 8, stride=2)   # 40→20, 256→256
        )

        # 3. P5上采样 + 融合生成P4（40×40）
        self.up_p5_to_p4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse_p4 = LightConvBlock(base_ch * 8 + base_ch * 8, base_ch * 4)

        # 4. P4上采样 + 融合生成P3（80×80）
        self.up_p4_to_p3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse_p3 = LightConvBlock(base_ch * 4 + base_ch * 8, base_ch * 2)

        # 5. 各尺度检测头输出
        self.head_p5 = nn.Conv2d(base_ch * 8, out_channels, 1)  # P5: 20×20×27
        self.head_p4 = nn.Conv2d(base_ch * 4, out_channels, 1)  # P4: 40×40×27
        self.head_p3 = nn.Conv2d(base_ch * 2, out_channels, 1)  # P3: 80×80×27

    def forward(self, x):
        """前向传播：输入光学特征图，输出三尺度检测结果"""
        # 初始特征提取
        x_init = self.init_conv(x)  # [B, 32, 640, 640]

        # 分步下采样，保存中间特征
        x320 = self.down_to_p5[0](x_init)  # [B, 64, 320, 320]
        x160 = self.down_to_p5[1](x320)    # [B, 128, 160, 160]
        x80 = self.down_to_p5[2](x160)     # [B, 256, 80, 80]
        x40 = self.down_to_p5[3](x80)      # [B, 256, 40, 40]
        p5 = self.down_to_p5[4](x40)       # [B, 256, 20, 20] → P5

        # 生成P4（上采样 + 特征融合）
        p5_up = self.up_p5_to_p4(p5)
        p4_fuse = torch.cat([p5_up, x40], dim=1)
        p4 = self.fuse_p4(p4_fuse)         # [B, 256, 40, 40] → P4

        # 生成P3（上采样 + 特征融合）
        p4_up = self.up_p4_to_p3(p4)
        p3_fuse = torch.cat([p4_up, x80], dim=1)
        p3 = self.fuse_p3(p3_fuse)         # [B, 128, 80, 80] → P3

        # 检测头输出
        p5_out = self.head_p5(p5)  # [B, 27, 20, 20]
        p4_out = self.head_p4(p4)  # [B, 27, 40, 40]
        p3_out = self.head_p3(p3)  # [B, 27, 80, 80]

        return p3_out, p4_out, p5_out

# =========================================================
# 完整光学YOLOv3系统
# =========================================================
class OpticalYOLOv3(nn.Module):
    """完整的光学YOLOv3系统：光学前端 + YOLOv3检测头"""
    def __init__(self, num_classes=4, img_size=640, optical_mode="phase"):
        super(OpticalYOLOv3, self).__init__()
        self.img_size = img_size
        
        # 光学前端（两层相位调制）
        self.optical_frontend = OpticalFrontend(
            resolution=(img_size, img_size), 
            mode=optical_mode
        )
        
        # YOLOv3检测头（输出通道数 = 3×(4+1+num_classes)）
        out_channels = 3 * (4 + 1 + num_classes)
        self.detector = YOLOLightHead(
            in_channels=1, 
            out_channels=out_channels
        )
        
        # 类别信息
        self.num_classes = num_classes

    def forward(self, x):
        """端到端前向传播"""
        # 输入：RGB图像 [B, 3, H, W]
        # 转换为灰度强度图
        if x.shape[1] == 3:
            intensity = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        else:
            intensity = x
        
        # 光学调制
        optical_feature = self.optical_frontend(intensity)  # [B, 1, H, W]
        
        # 目标检测
        p3, p4, p5 = self.detector(optical_feature)  # 三尺度检测结果
        
        return p3, p4, p5, optical_feature

    def enable_normalization(self, enable=True):
        """启用/禁用光学输出归一化"""
        self.optical_frontend.enable_norm = enable

# =========================================================
# 损失函数（YOLOv3多尺度损失）
# =========================================================
class YOLOLoss(nn.Module):
    """YOLOv3多尺度损失函数"""
    def __init__(self, box_weight=0.05, obj_weight=1.5, cls_weight=0.15):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight

    def forward(self, pred, target, batch_size):
        """计算多尺度损失"""
        # 调整预测张量形状
        pred = pred.permute(0, 2, 3, 1).reshape(batch_size, pred.shape[2], pred.shape[3], 3, -1)

        # 正负样本掩码
        obj_mask = target[..., 4] == 1
        noobj_mask = target[..., 4] == 0

        # 边界框损失
        box_loss = self.mse(pred[obj_mask][:, :4], target[obj_mask][:, :4])

        # 目标置信度损失（均衡正负样本）
        obj_loss = self.bce(pred[obj_mask][:, 4], target[obj_mask][:, 4])
        noobj_loss = self.bce(pred[noobj_mask][:, 4], target[noobj_mask][:, 4])
        obj_loss = obj_loss + 0.5 * noobj_loss

        # 分类损失
        cls_loss = self.bce(pred[obj_mask][:, 5:], target[obj_mask][:, 5:])

        # 加权总和
        total_loss = (self.box_weight * box_loss + 
                     self.obj_weight * obj_loss + 
                     self.cls_weight * cls_loss)
        
        return total_loss, box_loss, obj_loss, cls_loss

# =========================================================
# 目标构建工具函数
# =========================================================
def build_target(targets, anchors, stride, num_classes, img_size, device):
    """构建YOLO格式的训练目标"""
    batch_size = targets.shape[0]
    h, w = img_size // stride, img_size // stride
    num_anchors = len(anchors)

    target_tensor = torch.zeros((batch_size, h, w, num_anchors, 5 + num_classes), device=device)

    for b in range(batch_size):
        for t in targets[b]:
            cls, cx, cy, bw, bh = t
            cx_s = cx * w
            cy_s = cy * h
            bw_s = bw * w
            bh_s = bh * h

            i = int(cx_s)
            j = int(cy_s)

            # 选择最佳锚框
            best_idx = 0
            best_iou = 0
            for a_idx, (aw, ah) in enumerate(anchors):
                inter = min(bw_s, aw) * min(bh_s, ah)
                union = bw_s * bh_s + aw * ah - inter
                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
                    best_idx = a_idx

            # 填充目标张量
            target_tensor[b, j, i, best_idx, 0:4] = torch.tensor([cx_s, cy_s, bw_s, bh_s], device=device)
            target_tensor[b, j, i, best_idx, 4] = 1
            target_tensor[b, j, i, best_idx, 5 + int(cls)] = 1

    return target_tensor

# =========================================================
# 可视化工具
# =========================================================
def visualize_optical_detection(model, dataloader, device, save_path="optical_detection.png", num_samples=4):
    """可视化光学调制结果和检测特征"""
    model.eval()
    
    # 获取样本数据
    samples = []
    for batch in dataloader:
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs).to(device)
        samples = list(zip(imgs, targets))[:num_samples]
        break

    # 创建可视化画布
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, (img, target) in enumerate(samples):
        with torch.no_grad():
            # 模型推理
            p3, p4, p5, optical_feat = model(img.unsqueeze(0))
            
            # 转换为numpy用于可视化
            img_np = img.permute(1, 2, 0).cpu().numpy()
            optical_np = optical_feat.squeeze().cpu().numpy()
            p3_np = p3.squeeze().abs().mean(dim=0).cpu().numpy()
            p4_np = p4.squeeze().abs().mean(dim=0).cpu().numpy()
            p5_np = p5.squeeze().abs().mean(dim=0).cpu().numpy()

        # 第1列：输入图像
        axes[idx, 0].imshow(img_np)
        axes[idx, 0].set_title(f"Input {idx+1}")
        axes[idx, 0].axis('off')

        # 第2列：光学调制特征
        axes[idx, 1].imshow(optical_np, cmap='hot')
        axes[idx, 1].set_title("Optical Feature")
        axes[idx, 1].axis('off')

        # 第3-5列：多尺度检测特征
        axes[idx, 2].imshow(p3_np, cmap='hot')
        axes[idx, 2].set_title("P3 (80×80)")
        axes[idx, 2].axis('off')

        axes[idx, 3].imshow(p4_np, cmap='hot')
        axes[idx, 3].set_title("P4 (40×40)")
        axes[idx, 3].axis('off')

        axes[idx, 4].imshow(p5_np, cmap='hot')
        axes[idx, 4].set_title("P5 (20×20)")
        axes[idx, 4].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存至: {save_path}")

# =========================================================
# 测试代码
# =========================================================
if __name__ == "__main__":
    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建模型实例
    model = OpticalYOLOv3(num_classes=4, img_size=640, optical_mode="phase").to(device)
    
    # 测试前向传播
    x = torch.randn(2, 3, 640, 640).to(device)
    p3, p4, p5, optical_feat = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"光学特征形状: {optical_feat.shape}")
    print(f"P3检测输出形状: {p3.shape}")
    print(f"P4检测输出形状: {p4.shape}")
    print(f"P5检测输出形状: {p5.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    
    print("光学YOLOv3系统创建成功！")
