import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')


# =========================================================
# 配置文件读取
# =========================================================
def load_yaml_config(yaml_path):
    """加载YAML配置文件"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    nc = config.get('nc', 1)
    names = config.get('names', ['object'])

    if isinstance(names, dict):
        names = [names[i] for i in range(nc)]

    print(f"Loaded dataset config: {nc} classes")
    return nc, names


# =========================================================
# 锚框聚类 (K-means)
# =========================================================
def kmeans_anchors(dataset, n_anchors=9, img_size=640):
    """使用K-means聚类锚框"""
    print("Starting anchor clustering...")

    all_wh = []
    for i in tqdm(range(min(len(dataset), 1000)), desc="Collecting boxes"):
        _, targets, _ = dataset[i]
        if targets is not None and len(targets) > 0:
            wh = targets[:, 3:5]
            wh = wh * img_size
            all_wh.append(wh)

    if len(all_wh) == 0:
        print("Warning: No boxes found, using default anchors")
        return torch.tensor([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                             [59, 119], [116, 90], [156, 198], [373, 326]], dtype=torch.float32)

    all_wh = torch.cat(all_wh, dim=0).numpy()
    all_wh = all_wh[(all_wh[:, 0] > 0) & (all_wh[:, 1] > 0)]

    if len(all_wh) < n_anchors:
        print(f"Warning: Only {len(all_wh)} boxes found, using default anchors")
        return torch.tensor([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                             [59, 119], [116, 90], [156, 198], [373, 326]], dtype=torch.float32)

    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_anchors, random_state=42, n_init=10)
        kmeans.fit(all_wh)
        anchors = kmeans.cluster_centers_
        anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]
        print(f"Clustered anchors: {anchors}")
        return torch.tensor(anchors, dtype=torch.float32)
    except ImportError:
        print("sklearn not installed, using default anchors")
        return torch.tensor([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                             [59, 119], [116, 90], [156, 198], [373, 326]], dtype=torch.float32)


# =========================================================
# 改进的损失函数（修复版）
# =========================================================
class YOLOLoss(nn.Module):
    """完整的YOLO损失函数 - 修复版"""

    def __init__(self, anchors, num_classes, img_size=640):
        super().__init__()
        self.anchors = anchors  # [9, 2]
        self.num_classes = num_classes
        self.num_anchors = 3  # 每个尺度3个锚框
        self.img_size = img_size

    def forward(self, p3_out, p4_out, p5_out, targets):
        """
        p3_out: [B, 27, 80, 80]
        p4_out: [B, 27, 40, 40]
        p5_out: [B, 27, 20, 20]
        targets: list of [N, 5] (class, x, y, w, h) 归一化坐标
        """
        batch_size = p3_out.shape[0]
        device = p3_out.device

        # 初始化损失
        total_box_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        total_conf_loss = torch.tensor(0.0, device=device)
        num_targets = 0

        # 为每个尺度创建网格
        scales = [
            (80, 80, p3_out, 0),  # P3尺度，使用前3个锚框
            (40, 40, p4_out, 3),  # P4尺度，使用中间3个锚框
            (20, 20, p5_out, 6)  # P5尺度，使用后3个锚框
        ]

        for h, w, pred, anchor_start in scales:
            # 重塑预测值 [B, 27, H, W] -> [B, 3, 9, H, W] -> [B, 3, H, W, 9]
            pred = pred.view(batch_size, self.num_anchors, -1, h, w)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()  # [B, 3, H, W, 9]

            # 提取预测分量
            pred_tx = pred[..., 0]  # x偏移
            pred_ty = pred[..., 1]  # y偏移
            pred_tw = pred[..., 2]  # 宽度缩放
            pred_th = pred[..., 3]  # 高度缩放
            pred_conf = torch.sigmoid(pred[..., 4])  # 置信度

            # 类别概率 - 确保只取到实际类别数
            if pred.shape[-1] >= 5 + self.num_classes:
                pred_cls = torch.sigmoid(pred[..., 5:5 + self.num_classes])
            else:
                # 如果通道数不够，使用零张量
                pred_cls = torch.zeros(*pred.shape[:-1], self.num_classes, device=device)

            # 生成网格坐标（归一化）
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_x = grid_x.to(device).float() / w
            grid_y = grid_y.to(device).float() / h

            # 扩展网格到 [B, 3, H, W]
            grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_anchors, -1, -1)
            grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_anchors, -1, -1)

            # 获取当前尺度的锚框
            scale_anchors = self.anchors[anchor_start:anchor_start + self.num_anchors].to(device)  # [3, 2]

            # 解码预测框（归一化坐标）
            pred_x = (torch.sigmoid(pred_tx) + grid_x)  # x中心
            pred_y = (torch.sigmoid(pred_ty) + grid_y)  # y中心
            pred_w = torch.exp(pred_tw) * scale_anchors[:, 0].view(1, -1, 1, 1) / self.img_size
            pred_h = torch.exp(pred_th) * scale_anchors[:, 1].view(1, -1, 1, 1) / self.img_size

            # 对每个batch计算损失
            for b in range(batch_size):
                if len(targets[b]) == 0:
                    continue

                # 获取当前图片的目标并验证坐标
                t = targets[b].to(device)

                # 过滤无效的目标
                valid_mask = (t[:, 1] >= 0) & (t[:, 1] <= 1) & \
                             (t[:, 2] >= 0) & (t[:, 2] <= 1) & \
                             (t[:, 3] > 0) & (t[:, 3] <= 1) & \
                             (t[:, 4] > 0) & (t[:, 4] <= 1)

                t = t[valid_mask]
                if len(t) == 0:
                    continue

                num_targets += len(t)

                # 为每个真实框分配最佳锚框和位置
                for t_idx in range(len(t)):
                    cls_id = int(t[t_idx, 0].item())
                    # 确保类别ID在有效范围内
                    if cls_id >= self.num_classes:
                        continue

                    tx, ty, tw, th = t[t_idx, 1:5]

                    # 计算与当前尺度锚框的IoU
                    best_iou = -1
                    best_anchor_idx = 0
                    for a_idx in range(self.num_anchors):
                        anchor_w = scale_anchors[a_idx, 0] / self.img_size
                        anchor_h = scale_anchors[a_idx, 1] / self.img_size

                        # 计算IoU
                        inter_w = min(tw.item(), anchor_w)
                        inter_h = min(th.item(), anchor_h)
                        inter_area = inter_w * inter_h
                        union_area = tw.item() * th.item() + anchor_w * anchor_h - inter_area
                        iou = inter_area / (union_area + 1e-7)

                        if iou > best_iou:
                            best_iou = iou
                            best_anchor_idx = a_idx

                    # 找到最佳预测位置
                    grid_x_idx = int(tx.item() * w)
                    grid_y_idx = int(ty.item() * h)
                    grid_x_idx = min(max(0, grid_x_idx), w - 1)
                    grid_y_idx = min(max(0, grid_y_idx), h - 1)

                    # 获取预测值
                    pred_box = torch.stack([
                        pred_x[b, best_anchor_idx, grid_y_idx, grid_x_idx],
                        pred_y[b, best_anchor_idx, grid_y_idx, grid_x_idx],
                        pred_w[b, best_anchor_idx, grid_y_idx, grid_x_idx],
                        pred_h[b, best_anchor_idx, grid_y_idx, grid_x_idx]
                    ])

                    pred_cls_val = pred_cls[b, best_anchor_idx, grid_y_idx, grid_x_idx]
                    pred_conf_val = pred_conf[b, best_anchor_idx, grid_y_idx, grid_x_idx]

                    # 计算定位损失 (GIoU Loss)
                    target_box = torch.tensor([tx, ty, tw, th], device=device)
                    giou = self.compute_giou(pred_box, target_box)
                    box_loss = 1 - giou
                    total_box_loss += box_loss

                    # 计算分类损失
                    target_cls = torch.zeros(self.num_classes, device=device)
                    target_cls[cls_id] = 1
                    cls_loss = F.binary_cross_entropy(pred_cls_val, target_cls)
                    total_cls_loss += cls_loss

                    # 置信度损失（正样本）
                    conf_target = torch.clamp(giou.detach(), 0, 1)  # 限制在[0,1]范围内
                    conf_loss = F.binary_cross_entropy(
                        pred_conf_val.view(1),
                        conf_target.view(1)
                    )
                    total_conf_loss += conf_loss

        # 归一化损失（除以目标数量）
        if num_targets > 0:
            total_box_loss = total_box_loss / num_targets
            total_cls_loss = total_cls_loss / num_targets
            total_conf_loss = total_conf_loss / num_targets

        # 添加负样本置信度损失（简化版本）
        for h, w, pred, anchor_start in scales:
            pred = pred.view(batch_size, self.num_anchors, -1, h, w)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            pred_conf = torch.sigmoid(pred[..., 4])

            # 随机采样负样本
            neg_conf_loss = torch.tensor(0.0, device=device)
            neg_count = 0
            for b in range(batch_size):
                # 随机选择一部分网格作为负样本
                conf_vals = pred_conf[b].view(-1)
                num_neg = min(len(conf_vals), 500)  # 限制负样本数量
                indices = torch.randperm(len(conf_vals))[:num_neg]
                if len(indices) > 0:
                    neg_conf_loss += F.binary_cross_entropy(
                        conf_vals[indices],
                        torch.zeros_like(conf_vals[indices])
                    )
                    neg_count += len(indices)

            if neg_count > 0:
                total_conf_loss += 0.5 * (neg_conf_loss / neg_count)

        # 总损失
        total_loss = total_box_loss + total_cls_loss + total_conf_loss

        losses = {
            'box': total_box_loss,
            'cls': total_cls_loss,
            'conf': total_conf_loss
        }

        return total_loss, losses

    def compute_giou(self, box1, box2):
        """计算GIoU (Generalized IoU)"""
        # box: [x, y, w, h] 归一化坐标
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2

        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2

        # 计算IoU
        inter_x_min = torch.max(x1_min, x2_min)
        inter_y_min = torch.max(y1_min, y2_min)
        inter_x_max = torch.min(x1_max, x2_max)
        inter_y_max = torch.min(y1_max, y2_max)

        inter_area = torch.clamp(inter_x_max - inter_x_min, min=0) * torch.clamp(inter_y_max - inter_y_min, min=0)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union_area = area1 + area2 - inter_area + 1e-7

        iou = inter_area / union_area

        # 计算最小外接矩形
        enclose_x_min = torch.min(x1_min, x2_min)
        enclose_y_min = torch.min(y1_min, y2_min)
        enclose_x_max = torch.max(x1_max, x2_max)
        enclose_y_max = torch.max(y1_max, y2_max)
        enclose_area = (enclose_x_max - enclose_x_min) * (enclose_y_max - enclose_y_min) + 1e-7

        # GIoU = IoU - (外接矩形面积 - 并集面积) / 外接矩形面积
        giou = iou - (enclose_area - union_area) / enclose_area

        return giou


# =========================================================
# 教师网络（可训练版本）
# =========================================================
class ConvTeacher(nn.Module):
    """Conv-based teacher that mimics YOLOv5s P3 feature (可训练)"""

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

        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x):
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)

        f = self.conv1(x)
        f = self.conv2(f)
        f = self.conv3(f)
        f = self.refine(f)
        f = self.project(f)
        f = torch.abs(f)
        f = F.interpolate(f, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(f)


# =========================================================
# 轻量化检测头（保持原结构，输出27通道）
# =========================================================
class LightConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(LightConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


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
# 增强的数据集（支持读取标注）
# =========================================================
class YOLOStyleDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=640, transform=None, class_names=None):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.labels_dir = os.path.join(root_dir, split, 'labels')

        self.image_files = sorted([
            os.path.join(self.images_dir, f)
            for f in os.listdir(self.images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif"))
        ])

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

        self.class_names = class_names

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")

        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # 验证并修正坐标范围
                        x_center = max(0.0, min(1.0, x_center))
                        y_center = max(0.0, min(1.0, y_center))
                        width = max(0.01, min(1.0, width))  # 确保宽度不为0
                        height = max(0.01, min(1.0, height))

                        targets.append([cls, x_center, y_center, width, height])

        x = self.transform(img)

        if len(targets) > 0:
            targets = torch.tensor(targets, dtype=torch.float32)
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)

        return x, targets, img_path

    @staticmethod
    def collate_fn(batch):
        images = []
        targets = []
        paths = []
        for img, tgt, path in batch:
            images.append(img)
            targets.append(tgt)
            paths.append(path)
        return torch.stack(images), targets, paths


# =========================================================
# 可视化函数
# =========================================================
def save_feature_visualization(epoch, input_images, teacher_outputs, save_dir, prefix="teacher"):
    os.makedirs(save_dir, exist_ok=True)

    batch_size = min(4, input_images.shape[0])

    fig, axes = plt.subplots(2, batch_size, figsize=(4 * batch_size, 8))
    if batch_size == 1:
        axes = axes.reshape(2, 1)

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    for i in range(batch_size):
        input_img = input_images[i].permute(1, 2, 0).numpy()
        input_img = norm(input_img)
        axes[0, i].imshow(input_img)
        axes[0, i].set_title(f"Input {i + 1}")
        axes[0, i].axis("off")

        teacher_feat = teacher_outputs[i, 0].numpy()
        teacher_feat = norm(teacher_feat)
        im = axes[1, i].imshow(teacher_feat, cmap="hot")
        axes[1, i].set_title(f"Teacher Feature {i + 1}")
        axes[1, i].axis("off")
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_epoch_{epoch:03d}.png")
    plt.savefig(save_path, dpi=120)
    plt.close()


def save_head_visualization(epoch, p3, p4, p5, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    p3_vis = p3[0].cpu().detach()
    p4_vis = p4[0].cpu().detach()
    p5_vis = p5[0].cpu().detach()

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    p3_norm = norm(p3_vis[0])
    axes[0].imshow(p3_norm, cmap="hot")
    axes[0].set_title(f"P3 Output (80×80) - Epoch {epoch}")
    axes[0].axis("off")

    p4_norm = norm(p4_vis[0])
    axes[1].imshow(p4_norm, cmap="hot")
    axes[1].set_title(f"P4 Output (40×40) - Epoch {epoch}")
    axes[1].axis("off")

    p5_norm = norm(p5_vis[0])
    axes[2].imshow(p5_norm, cmap="hot")
    axes[2].set_title(f"P5 Output (20×20) - Epoch {epoch}")
    axes[2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"head_output_epoch_{epoch:03d}.png")
    plt.savefig(save_path, dpi=120)
    plt.close()


# =========================================================
# 主训练函数
# =========================================================
def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    yaml_path = r"data/military/data.yaml"

    if not os.path.exists(yaml_path):
        print(f"Warning: YAML file not found at {yaml_path}, using default values")
        nc = 4
        class_names = ['military_tank', 'soldier', 'military_aircraft', 'military_warship']
    else:
        nc, class_names = load_yaml_config(yaml_path)

    print(f"Number of classes: {nc}")

    dataset_root = r"data/military"
    img_size = 640

    if not os.path.exists(dataset_root):
        print(f"Error: Dataset root not found at {dataset_root}")
        return None, None

    train_dataset = YOLOStyleDataset(
        root_dir=dataset_root,
        split='train',
        img_size=img_size,
        class_names=class_names
    )

    val_split = 'val' if os.path.exists(os.path.join(dataset_root, 'val')) else 'test'
    val_dataset = YOLOStyleDataset(
        root_dir=dataset_root,
        split=val_split,
        img_size=img_size,
        class_names=class_names
    )

    # 锚框聚类
    anchors = kmeans_anchors(train_dataset, n_anchors=9, img_size=img_size)
    print(f"Anchors shape: {anchors.shape}")

    # 创建模型
    teacher = ConvTeacher().to(device)
    detection_head = YOLOLightHead(in_channels=1, out_channels=27).to(device)

    print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters())}")
    print(f"Detection head parameters: {sum(p.numel() for p in detection_head.parameters())}")

    # 损失函数
    yolo_loss = YOLOLoss(anchors, num_classes=nc, img_size=img_size)

    # 优化器
    optimizer = optim.AdamW([
        {'params': teacher.parameters(), 'lr': 1e-4},
        {'params': detection_head.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    # 学习率调度器
    num_epochs = 400
    warmup_epochs = 50

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=YOLOStyleDataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=YOLOStyleDataset.collate_fn
    )

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/training_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # 训练记录
    train_losses = []
    val_losses = []

    # 训练循环
    for epoch in range(num_epochs):
        teacher.train()
        detection_head.train()

        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (images, targets, paths) in enumerate(progress_bar):
            images = images.to(device)

            # 前向传播
            teacher_feat = teacher(images)
            p3_out, p4_out, p5_out = detection_head(teacher_feat)

            # 打印形状（仅第一次）
            if batch_idx == 0 and epoch == 0:
                print(f"P3 shape: {p3_out.shape}")
                print(f"P4 shape: {p4_out.shape}")
                print(f"P5 shape: {p5_out.shape}")

            # 计算损失
            loss, loss_details = yolo_loss(p3_out, p4_out, p5_out, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=10.0)
            torch.nn.utils.clip_grad_norm_(detection_head.parameters(), max_norm=10.0)

            optimizer.step()

            epoch_loss += loss.item()

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'box': f'{loss_details["box"].item():.4f}',
                'cls': f'{loss_details["cls"].item():.4f}',
                'conf': f'{loss_details["conf"].item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        # 更新学习率
        scheduler.step()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证
        teacher.eval()
        detection_head.eval()
        val_loss = 0

        with torch.no_grad():
            for images, targets, paths in val_loader:
                images = images.to(device)
                teacher_feat = teacher(images)
                p3_out, p4_out, p5_out = detection_head(teacher_feat)

                loss, _ = yolo_loss(p3_out, p4_out, p5_out, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_losses.append(avg_val_loss)

        # 每10轮保存可视化
        if epoch % 10 == 0 and len(val_loader) > 0:
            val_iter = iter(val_loader)
            vis_images, _, _ = next(val_iter)
            vis_images = vis_images.to(device)

            with torch.no_grad():
                teacher_feat = teacher(vis_images[:4])
                p3_out, p4_out, p5_out = detection_head(teacher_feat)

                save_feature_visualization(epoch, vis_images[:4].cpu(), teacher_feat[:4].cpu(), vis_dir)
                save_head_visualization(epoch, p3_out[:4], p4_out[:4], p5_out[:4], vis_dir)

        print(
            f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")

        # 保存检查点
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'teacher_state_dict': teacher.state_dict(),
                'detection_head_state_dict': detection_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'anchors': anchors,
                'nc': nc,
                'class_names': class_names,
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

    # 保存最终模型
    torch.save({
        'teacher_state_dict': teacher.state_dict(),
        'detection_head_state_dict': detection_head.state_dict(),
        'anchors': anchors,
        'nc': nc,
        'class_names': class_names,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, os.path.join(output_dir, 'final_model.pth'))

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    if len(val_losses) > 0 and val_losses[0] > 0:
        plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=120)
    plt.close()

    print(f"Training completed! Results saved to: {output_dir}")
    print(f"Final train loss: {train_losses[-1]:.4f}")

    return teacher, detection_head


# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    try:
        teacher, detection_head = train_model()

        if teacher is not None:
            print("=" * 60)
            print("Training completed successfully!")
            print("Model is now learning proper object detection features!")
            print("=" * 60)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()