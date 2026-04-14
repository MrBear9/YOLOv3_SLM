import yaml
import torch.nn.functional as F
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
from datetime import datetime
from PIL import Image
import matplotlib.patches as patches

from optical_teacher import (
    load_class_names,
    log_to_file,
    init_log_file,
    SLMLayer,
    ASMPropagation,
    OpticalStudent,
    ConvTeacher,
    LightConvBlock,
    YOLOLightHead,
    enhance_feature_for_display,
    decode_detections
)

class ConfigYOLO:
    YAML_PATH = r"data\military\data.yaml"
    CLASS_NAMES = None
    NUM_CLASSES = None
    
    TEACHER_OUTPUT_DIR = r"output\OpticalTeacherYOLO"
    LOG_ROOT_DIR = None
    LOG_FILE = None
    TIMESTAMP = None
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 640
    BATCH_SIZE = 8
    EPOCHS = 100
    
    BOX_WEIGHT = 0.05 # box loss weight
    OBJ_WEIGHT = 1.5 # object loss weight
    CLS_WEIGHT = 0.15 # class loss weight
    
    STRIDES = [8, 16, 32]
    ANCHORS = [
        [[10,13], [16,30], [33,23]],
        [[30,61], [62,45], [59,119]],
        [[116,90], [156,198], [373,326]]
    ]
    
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-5
    OPTIMIZER = "Adam"
    
    VIS_INTERVAL = 5 # visualization interval in epochs
    
    CONF_THRESH = 0.5 # confidence threshold
    NMS_THRESH = 0.4 # NMS threshold for post-processing
    MAX_DET = 8 # maximum number of detections to keep
    
    VIS_BATCH_SIZE = 4 # visualization batch size
    VIS_DPI = 120 # visualization DPI
    
    @classmethod
    def initialize(cls):
        cls.CLASS_NAMES, cls.NUM_CLASSES = load_class_names(cls.YAML_PATH)
        os.makedirs(cls.TEACHER_OUTPUT_DIR, exist_ok=True)
        cls.LOG_ROOT_DIR = os.path.join(cls.TEACHER_OUTPUT_DIR, "logs")
        os.makedirs(cls.LOG_ROOT_DIR, exist_ok=True)
        cls.TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.LOG_FILE = os.path.join(cls.LOG_ROOT_DIR, f"training_log_{cls.TIMESTAMP}.txt")
    
    @classmethod
    def get_detector_output_channels(cls):
        return 3 * (4 + 1 + cls.NUM_CLASSES)
    
    @classmethod
    def print_config(cls):
        print("="*80)
        print("Optical Teacher YOLO Training Config")
        print("="*80)
        print(f"Device: {cls.DEVICE}")
        print(f"Image Size: {cls.IMG_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Num Classes: {cls.NUM_CLASSES}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Class Names: {cls.CLASS_NAMES}")
        print("="*80)

Config = ConfigYOLO

Config.initialize()
Config.print_config()

init_log_file()
log_to_file(f"Log file path: {Config.LOG_FILE}")
log_to_file(f"Visualization save path: {Config.LOG_ROOT_DIR.replace('\\logs', '')}")
log_to_file(f"Class info: {Config.CLASS_NAMES}, Num classes: {Config.NUM_CLASSES}")

class YOLODataset(Dataset):
    def __init__(self, yaml_path=None, split="train"):
        if yaml_path is None:
            yaml_path = Config.YAML_PATH
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        img_dir = os.path.join(cfg["path"], f"{split}/images")
        label_dir = os.path.join(cfg["path"], f"{split}/labels")
        
        self.files = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith(".jpg") or f.endswith(".png")
        ])
        
        self.label_dir = label_dir
        self.img_size = Config.IMG_SIZE
        self.num_classes = Config.NUM_CLASSES
        
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Grayscale(1),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        
        img_tensor = self.transform(img)
        
        label_path = os.path.join(self.label_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        targets.append([cls_id, x_center, y_center, width, height])
        
        if len(targets) > 0:
            targets = torch.tensor(targets, dtype=torch.float32)
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)
        
        return img_tensor, targets

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, strides):
        super().__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.num_classes = num_classes
        self.strides = strides
        self.box_weight = Config.BOX_WEIGHT
        self.obj_weight = Config.OBJ_WEIGHT
        self.cls_weight = Config.CLS_WEIGHT
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions, targets):
        total_loss = 0
        
        for i, pred in enumerate(predictions):
            batch_size, _, grid_h, grid_w = pred.shape
            stride = self.strides[i]
            anchors = self.anchors[i] / stride
            
            pred = pred.permute(0, 2, 3, 1).reshape(batch_size, grid_h, grid_w, 3, -1)
            
            pred_boxes = pred[..., :4]
            pred_obj = pred[..., 4]
            pred_cls = pred[..., 5:]
            
            target_boxes = torch.zeros_like(pred_boxes)
            target_obj = torch.zeros_like(pred_obj)
            target_cls = torch.zeros_like(pred_cls)
            
            for b in range(batch_size):
                if len(targets[b]) == 0:
                    continue
                
                for target_idx in range(len(targets[b])):
                    cls_id, tx, ty, tw, th = targets[b][target_idx]
                    cls_id = int(cls_id.item())
                    
                    gx = int(tx * grid_w)
                    gy = int(ty * grid_h)
                    
                    gx = max(0, min(gx, grid_w - 1))
                    gy = max(0, min(gy, grid_h - 1))
                    
                    best_iou = 0
                    best_anchor = 0
                    
                    for a in range(3):
                        anchor_w, anchor_h = anchors[a]
                        iou = min(tw, anchor_w) * min(th, anchor_h) / (tw * th + anchor_w * anchor_h - min(tw, anchor_w) * min(th, anchor_h) + 1e-6)
                        if iou > best_iou:
                            best_iou = iou
                            best_anchor = a
                    
                    if best_iou > 0.3:
                        target_boxes[b, gy, gx, best_anchor, 0] = tx * grid_w - gx
                        target_boxes[b, gy, gx, best_anchor, 1] = ty * grid_h - gy
                        target_boxes[b, gy, gx, best_anchor, 2] = torch.log(tw / anchors[best_anchor, 0] + 1e-6)
                        target_boxes[b, gy, gx, best_anchor, 3] = torch.log(th / anchors[best_anchor, 1] + 1e-6)
                        target_obj[b, gy, gx, best_anchor] = 1.0
                        target_cls[b, gy, gx, best_anchor, cls_id] = 1.0
            
            obj_mask = target_obj > 0.5
            noobj_mask = target_obj <= 0.5
            
            box_loss = self.mse_loss(pred_boxes, target_boxes).sum() / (obj_mask.sum() + 1e-6)
            obj_loss = (self.bce_loss(pred_obj, target_obj) * obj_mask).sum() / (obj_mask.sum() + 1e-6)
            noobj_loss = (self.bce_loss(pred_obj, target_obj) * noobj_mask).sum() / (noobj_mask.sum() + 1e-6)
            cls_loss = (self.bce_loss(pred_cls, target_cls) * obj_mask).sum() / (obj_mask.sum() + 1e-6)
            
            total_loss += (self.box_weight * box_loss + 
                          self.obj_weight * (obj_loss + 0.5 * noobj_loss) + 
                          self.cls_weight * cls_loss)
        
        return total_loss

def save_detection_visualization(epoch, model, dataset, save_dir, prefix="train", device=None):
    if device is None:
        device = Config.DEVICE
    
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    num_samples = min(Config.VIS_BATCH_SIZE, len(dataset))
    indices = torch.randperm(len(dataset))[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(24, 6 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, targets = dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            img_np = img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            
            teacher_output = model.teacher(img_tensor)
            teacher_output = teacher_output.squeeze(0).cpu().numpy()
            teacher_output = enhance_feature_for_display(teacher_output)
            
            pred_p3, pred_p4, pred_p5 = model(img_tensor)
            preds = [pred_p3, pred_p4, pred_p5]
            detections = decode_detections(preds)
            
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"Original Image {i+1}")
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(teacher_output, cmap="hot")
            axes[i, 1].set_title(f"Teacher Feature {i+1}")
            axes[i, 1].axis("off")
            
            axes[i, 2].imshow(img_np)
            axes[i, 2].set_title(f"Ground Truth {i+1}")
            axes[i, 2].axis("off")
            
            for target_idx in range(len(targets)):
                cls_id, x_center, y_center, width, height = targets[target_idx]
                cls_id = int(cls_id.item())
                
                x1 = int((x_center - width / 2) * Config.IMG_SIZE)
                y1 = int((y_center - height / 2) * Config.IMG_SIZE)
                w = int(width * Config.IMG_SIZE)
                h = int(height * Config.IMG_SIZE)
                
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                        edgecolor='green', facecolor='none')
                axes[i, 2].add_patch(rect)
                axes[i, 2].text(x1, y1 - 5, Config.CLASS_NAMES[cls_id], 
                              color='green', fontsize=10, fontweight='bold')
            
            axes[i, 3].imshow(img_np)
            axes[i, 3].set_title(f"Predictions {i+1}")
            axes[i, 3].axis("off")
            
            if len(detections[0]) > 0:
                for det in detections[0]:
                    x_center, y_center, w, h, conf, cls_id = det
                    cls_id = int(cls_id)
                    
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    w = int(w)
                    h = int(h)
                    
                    color = plt.cm.tab20(cls_id / max(Config.NUM_CLASSES, 1))
                    rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                            edgecolor=color, facecolor='none')
                    axes[i, 3].add_patch(rect)
                    
                    label = f"{Config.CLASS_NAMES[cls_id]}: {conf:.2f}"
                    axes[i, 3].text(x1, y1 - 5, label, 
                                  color=color, fontsize=10, fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.3', 
                                          facecolor=color, alpha=0.7))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_epoch_{epoch:03d}.png")
    plt.savefig(save_path, dpi=Config.VIS_DPI)
    plt.close()

class TeacherWithDetector(nn.Module):
    def __init__(self, teacher=None, detector=None):
        super().__init__()
        if teacher is None:
            self.teacher = ConvTeacher()
        else:
            self.teacher = teacher
        
        if detector is None:
            self.detector = YOLOLightHead(in_channels=1, 
                                         out_channels=Config.get_detector_output_channels())
        else:
            self.detector = detector

    def forward(self, x):
        features = self.teacher(x)
        detections = self.detector(features)
        return detections

def train():
    device = Config.DEVICE
    log_to_file(f"使用设备: {device}")
    
    log_to_file("初始化教师网络（ConvTeacher）...")
    teacher = ConvTeacher()
    
    for p in teacher.parameters():
        p.requires_grad = False
    log_to_file("教师网络参数已冻结")
    
    log_to_file("初始化检测头（YOLOLightHead）...")
    detector = YOLOLightHead(in_channels=1, 
                           out_channels=Config.get_detector_output_channels())
    
    log_to_file("构建教师网络+检测头模型...")
    model = TeacherWithDetector(teacher=teacher, detector=detector).to(device)
    
    for p in model.teacher.parameters():
        p.requires_grad = False
    for p in model.detector.parameters():
        p.requires_grad = True
    
    log_to_file("加载数据集...")
    train_dataset = YOLODataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                             shuffle=True, collate_fn=lambda x: x)
    log_to_file(f"训练集大小: {len(train_dataset)}")
    
    optimizer = torch.optim.Adam(model.detector.parameters(), 
                                lr=Config.LEARNING_RATE, 
                                weight_decay=Config.WEIGHT_DECAY)
    
    criterion = YOLOLoss(anchors=Config.ANCHORS, 
                        num_classes=Config.NUM_CLASSES, 
                        strides=Config.STRIDES)
    
    vis_dir = os.path.join(Config.TEACHER_OUTPUT_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    loss_curve = []
    best_loss = float('inf')
    
    log_to_file("="*60)
    log_to_file("开始训练检测头...")
    log_to_file("="*60)
    
    for epoch in range(Config.EPOCHS):
        model.train()
        loss_sum = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS}", leave=True):
            batch_targets = []
            batch_images = []
            
            for item in batch:
                img_tensor, targets = item
                batch_images.append(img_tensor)
                batch_targets.append(targets)
            
            batch_images = torch.stack(batch_images).to(device)
            
            optimizer.zero_grad()
            
            predictions = model(batch_images)
            loss = criterion(predictions, batch_targets)
            
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()

        avg_loss = loss_sum / len(train_loader)
        loss_curve.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "detector_best.pth")
            torch.save(model.detector.state_dict(), best_model_path)
            log_to_file(f"Epoch {epoch}: 保存最佳模型 (Loss: {best_loss:.6f})", also_print=False)
        
        if epoch % Config.VIS_INTERVAL == 0:
            save_detection_visualization(epoch, model, train_dataset, vis_dir, prefix="train", device=device)

        log_to_file(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_curve, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(Config.TEACHER_OUTPUT_DIR, "loss_curve.png"), dpi=Config.VIS_DPI)
    plt.close()

    model_save_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "detector_final.pth")
    torch.save(model.detector.state_dict(), model_save_path)
    
    log_to_file("="*60)
    log_to_file("训练完成！")
    log_to_file(f"最佳模型已保存到: {os.path.join(Config.TEACHER_OUTPUT_DIR, 'detector_best.pth')}")
    log_to_file(f"最终模型已保存到: {model_save_path}")
    log_to_file(f"所有结果已保存到: {Config.TEACHER_OUTPUT_DIR}")
    log_to_file("="*60)

if __name__ == "__main__":
    train()