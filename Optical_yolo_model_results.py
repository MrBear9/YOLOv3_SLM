import argparse
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.ops import nms


ROOT = Path(__file__).resolve().parent
DEFAULT_YAML = ROOT / "data" / "military" / "data.yaml"
DEFAULT_CHECKPOINT = ROOT / "output" / "OpticalTeacherYOLO" / "teacher_detector_final.pth"
DEFAULT_OUTPUT_DIR = ROOT / "Optical_yolo_detect" / "Optical_yolo_model_results"
DEFAULT_ANCHOR_YAML = ROOT / "output" / "anchor_clustering" / "yolo_anchors.yaml"
DEFAULT_STRIDES = [8, 16, 32]
DEFAULT_ANCHORS = [
    [[26, 23], [47, 49], [100, 67]],
    [[103, 169], [203, 107], [351, 177]],
    [[241, 354], [534, 299], [568, 528]],
]


def windows_safe_path(path: Path | str) -> str:
    text = str(path)
    if os.name != "nt":
        return text
    if text.startswith("\\\\?\\"):
        return text
    resolved = str(Path(text).resolve(strict=False))
    if resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved.lstrip("\\")
    return "\\\\?\\" + resolved


def file_exists(path: Path | str) -> bool:
    return os.path.exists(windows_safe_path(path))


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_yaml(path: Path):
    with open(windows_safe_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_text(path: Path, content: str):
    ensure_dir(path.parent)
    with open(windows_safe_path(path), "w", encoding="utf-8", newline="") as f:
        f.write(content)


def load_class_names(yaml_path: Path):
    cfg = read_yaml(yaml_path)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        ordered = {int(k): v for k, v in names.items()}
    else:
        ordered = {idx: name for idx, name in enumerate(names)}
    return ordered, len(ordered)


def load_dataset_info(yaml_path: Path):
    cfg = read_yaml(yaml_path)
    class_names, num_classes = load_class_names(yaml_path)

    raw_dataset_root = Path(cfg.get("path", "."))
    candidates = []
    if raw_dataset_root.is_absolute():
        candidates.append(raw_dataset_root)
    else:
        candidates.append((yaml_path.parent / raw_dataset_root).resolve())
        candidates.append((ROOT / raw_dataset_root).resolve())
        candidates.append(yaml_path.parent.resolve())
    dataset_root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])

    splits = {}
    for split in ("train", "val", "test"):
        split_rel = cfg.get(split)
        if split_rel is None:
            continue
        split_path = Path(split_rel)
        images_dir = split_path if split_path.is_absolute() else (dataset_root / split_path).resolve()
        labels_dir = images_dir.parent / "labels"
        splits[split] = {
            "images_dir": images_dir,
            "labels_dir": labels_dir,
        }

    return {
        "cfg": cfg,
        "dataset_root": dataset_root,
        "class_names": class_names,
        "num_classes": num_classes,
        "splits": splits,
    }


def load_anchor_groups(anchor_yaml_path: Path | None):
    if anchor_yaml_path is not None and file_exists(anchor_yaml_path):
        cfg = read_yaml(Path(anchor_yaml_path))
        anchors = cfg.get("anchors")
        if isinstance(anchors, list) and len(anchors) == 3:
            return anchors
    return DEFAULT_ANCHORS


def enhance_feature_for_display(feature_map):
    feature_map = np.asarray(feature_map, dtype=np.float32)
    low = np.percentile(feature_map, 2)
    high = np.percentile(feature_map, 98)
    if high - low < 1e-6:
        return np.zeros_like(feature_map)
    feature_map = np.clip((feature_map - low) / (high - low), 0.0, 1.0)
    return np.power(feature_map, 0.8)


def xywh_to_xyxy(boxes: torch.Tensor):
    half_w = boxes[:, 2] / 2
    half_h = boxes[:, 3] / 2
    return torch.stack([
        boxes[:, 0] - half_w,
        boxes[:, 1] - half_h,
        boxes[:, 0] + half_w,
        boxes[:, 1] + half_h,
    ], dim=1)


def bbox_iou_xywh(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7):
    box1 = box1.reshape(-1, 4)
    box2 = box2.reshape(-1, 4)
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((box1.shape[0],), dtype=box1.dtype, device=box1.device)

    box1_xyxy = xywh_to_xyxy(box1)
    box2_xyxy = xywh_to_xyxy(box2)

    inter_x1 = torch.max(box1_xyxy[:, 0], box2_xyxy[:, 0])
    inter_y1 = torch.max(box1_xyxy[:, 1], box2_xyxy[:, 1])
    inter_x2 = torch.min(box1_xyxy[:, 2], box2_xyxy[:, 2])
    inter_y2 = torch.min(box1_xyxy[:, 3], box2_xyxy[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    box1_area = (box1_xyxy[:, 2] - box1_xyxy[:, 0]).clamp(min=0) * (box1_xyxy[:, 3] - box1_xyxy[:, 1]).clamp(min=0)
    box2_area = (box2_xyxy[:, 2] - box2_xyxy[:, 0]).clamp(min=0) * (box2_xyxy[:, 3] - box2_xyxy[:, 1]).clamp(min=0)
    union = box1_area + box2_area - inter_area + eps
    return inter_area / union


def apply_nms(detections, nms_thresh, max_det, class_agnostic=False):
    if len(detections) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    det_tensor = torch.as_tensor(detections, dtype=torch.float32)
    boxes_xyxy = xywh_to_xyxy(det_tensor[:, :4])
    scores = det_tensor[:, 4]
    class_ids = det_tensor[:, 5]

    if class_agnostic:
        keep_indices = nms(boxes_xyxy, scores, nms_thresh)
        det_tensor = det_tensor[keep_indices]
    else:
        kept = []
        for cls_id in class_ids.unique(sorted=False):
            cls_mask = class_ids == cls_id
            keep_indices = nms(boxes_xyxy[cls_mask], scores[cls_mask], nms_thresh)
            kept.append(det_tensor[cls_mask][keep_indices])
        if len(kept) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        det_tensor = torch.cat(kept, dim=0)

    det_tensor = det_tensor[det_tensor[:, 4].argsort(descending=True)]
    return det_tensor[:max_det].cpu().numpy()


def decode_detections(preds, anchors, strides, num_classes, conf_thresh, nms_thresh, max_det, img_size, class_agnostic=False):
    batch_size = preds[0].shape[0]
    detections = [[] for _ in range(batch_size)]

    for i, pred in enumerate(preds):
        grid_h, grid_w = pred.shape[2], pred.shape[3]
        stride = strides[i]
        pred = pred.permute(0, 2, 3, 1).reshape(batch_size, grid_h, grid_w, 3, 5 + num_classes)

        obj_conf = torch.sigmoid(pred[..., 4])
        cls_conf = torch.sigmoid(pred[..., 5:])
        cls_score, cls_id = cls_conf.max(dim=-1)
        final_conf = obj_conf * cls_score
        valid_mask = (obj_conf >= conf_thresh) & (final_conf >= conf_thresh)

        if not valid_mask.any():
            continue

        device = pred.device
        dtype = pred.dtype
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h, device=device, dtype=dtype),
            torch.arange(grid_w, device=device, dtype=dtype),
            indexing="ij",
        )
        grid_x = grid_x.view(1, grid_h, grid_w, 1)
        grid_y = grid_y.view(1, grid_h, grid_w, 1)
        anchor_tensor = torch.tensor(anchors[i], device=device, dtype=dtype).view(1, 1, 1, 3, 2)

        tx = pred[..., 0]
        ty = pred[..., 1]
        tw = pred[..., 2]
        th = pred[..., 3]

        x_center = (grid_x + torch.sigmoid(tx)) * stride
        y_center = (grid_y + torch.sigmoid(ty)) * stride
        w = anchor_tensor[..., 0] * torch.exp(torch.clamp(tw, min=-8.0, max=8.0))
        h = anchor_tensor[..., 1] * torch.exp(torch.clamp(th, min=-8.0, max=8.0))

        x_center = x_center.clamp(0, img_size - 1)
        y_center = y_center.clamp(0, img_size - 1)
        w = w.clamp(1, img_size)
        h = h.clamp(1, img_size)

        for b in range(batch_size):
            sample_mask = valid_mask[b]
            if not sample_mask.any():
                continue

            sample_detections = torch.stack([
                x_center[b][sample_mask],
                y_center[b][sample_mask],
                w[b][sample_mask],
                h[b][sample_mask],
                final_conf[b][sample_mask],
                cls_id[b][sample_mask].to(dtype),
            ], dim=1)
            detections[b].append(sample_detections)

    for b in range(batch_size):
        if len(detections[b]) > 0:
            detections[b] = apply_nms(
                torch.cat(detections[b], dim=0),
                nms_thresh=nms_thresh,
                max_det=max_det,
                class_agnostic=class_agnostic,
            )
        else:
            detections[b] = np.zeros((0, 6), dtype=np.float32)
    return detections


class ConvTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
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
        f = F.interpolate(f, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(f)


class LightConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
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
        super().__init__()
        base_ch = 32
        self.init_conv = LightConvBlock(in_channels, base_ch, kernel_size=3, stride=1)
        self.down_to_p5 = nn.Sequential(
            LightConvBlock(base_ch, base_ch * 2, stride=2),
            LightConvBlock(base_ch * 2, base_ch * 4, stride=2),
            LightConvBlock(base_ch * 4, base_ch * 8, stride=2),
            LightConvBlock(base_ch * 8, base_ch * 8, stride=2),
            LightConvBlock(base_ch * 8, base_ch * 8, stride=2),
        )
        self.up_p5_to_p4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse_p4 = LightConvBlock(base_ch * 8 + base_ch * 8, base_ch * 4)
        self.up_p4_to_p3 = nn.Upsample(scale_factor=2, mode="nearest")
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


class TeacherWithDetector(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.teacher = ConvTeacher()
        self.detector = YOLOLightHead(in_channels=1, out_channels=3 * (5 + num_classes))

    def forward(self, x):
        feature = self.teacher(x)
        return self.detector(feature)


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("teacher_state_dict", "detector_state_dict", "model_state_dict", "state_dict", "model"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def _normalize_ckpt_key(key: str):
    if key.startswith("module."):
        key = key[7:]
    return key


def load_joint_checkpoint(model: TeacherWithDetector, checkpoint_path: Path, device: torch.device):
    if not file_exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(windows_safe_path(checkpoint_path), map_location=device)
    teacher_loaded = 0
    detector_loaded = 0

    if isinstance(checkpoint, dict) and "teacher_state_dict" in checkpoint and "detector_state_dict" in checkpoint:
        teacher_state = extract_state_dict(checkpoint["teacher_state_dict"])
        detector_state = extract_state_dict(checkpoint["detector_state_dict"])
        model.teacher.load_state_dict(teacher_state, strict=False)
        model.detector.load_state_dict(detector_state, strict=False)
        teacher_loaded = len(teacher_state)
        detector_loaded = len(detector_state)
        return teacher_loaded, detector_loaded

    state_dict = extract_state_dict(checkpoint)
    teacher_target = model.teacher.state_dict()
    detector_target = model.detector.state_dict()
    teacher_compatible = {}
    detector_compatible = {}

    for raw_key, value in state_dict.items():
        key = _normalize_ckpt_key(raw_key)

        if key.startswith("teacher."):
            key = key[len("teacher."):]
            if key in teacher_target and teacher_target[key].shape == value.shape:
                teacher_compatible[key] = value
            continue

        if key.startswith("detector."):
            key = key[len("detector."):]
            if key in detector_target and detector_target[key].shape == value.shape:
                detector_compatible[key] = value
            continue

        if key in teacher_target and teacher_target[key].shape == value.shape:
            teacher_compatible[key] = value
        elif key in detector_target and detector_target[key].shape == value.shape:
            detector_compatible[key] = value

    if len(teacher_compatible) > 0:
        model.teacher.load_state_dict({**teacher_target, **teacher_compatible}, strict=False)
    if len(detector_compatible) > 0:
        model.detector.load_state_dict({**detector_target, **detector_compatible}, strict=False)

    if len(teacher_compatible) == 0 and len(detector_compatible) == 0:
        raise RuntimeError(f"No compatible teacher/detector weights found in checkpoint: {checkpoint_path}")

    return len(teacher_compatible), len(detector_compatible)


class InferenceDataset(Dataset):
    def __init__(self, entries, img_size):
        self.entries = entries
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image_path = Path(entry["image_path"])
        image = Image.open(windows_safe_path(image_path)).convert("RGB")
        image_tensor = self.transform(image)

        targets = []
        label_path = Path(entry["label_path"]) if entry.get("label_path") else None
        if label_path is not None and file_exists(label_path):
            with open(windows_safe_path(label_path), "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                    except ValueError:
                        continue
                    if width <= 0 or height <= 0:
                        continue
                    targets.append([cls_id, x_center, y_center, width, height])

        if len(targets) > 0:
            targets_tensor = torch.tensor(targets, dtype=torch.float32)
        else:
            targets_tensor = torch.zeros((0, 5), dtype=torch.float32)

        return {
            "image_tensor": image_tensor,
            "targets": targets_tensor,
            "image_path": str(image_path),
            "label_path": str(label_path) if label_path is not None else "",
            "stem": image_path.stem,
        }


def collate_inference(batch):
    images = torch.stack([item["image_tensor"] for item in batch], dim=0)
    targets = [item["targets"] for item in batch]
    infos = [
        {
            "image_path": item["image_path"],
            "label_path": item["label_path"],
            "stem": item["stem"],
        }
        for item in batch
    ]
    return images, targets, infos


def build_entries_from_split(dataset_info, split: str, max_eval_images: int | None):
    if split not in dataset_info["splits"]:
        raise ValueError(f"Split '{split}' not found in dataset yaml.")

    images_dir = dataset_info["splits"][split]["images_dir"]
    labels_dir = dataset_info["splits"][split]["labels_dir"]
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_files = sorted([
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    ])
    if max_eval_images is not None and max_eval_images > 0:
        image_files = image_files[:max_eval_images]

    entries = []
    for image_path in image_files:
        entries.append({
            "image_path": str(image_path),
            "label_path": str(labels_dir / f"{image_path.stem}.txt"),
        })
    return entries


def infer_label_path_for_image(image_path: Path, dataset_info):
    for split, split_cfg in dataset_info["splits"].items():
        images_dir = split_cfg["images_dir"].resolve()
        try:
            image_path.resolve().relative_to(images_dir)
            return split_cfg["labels_dir"] / f"{image_path.stem}.txt", split
        except ValueError:
            continue
    return None, None


def build_entries_from_single_image(image_path: Path, dataset_info, label_path: Path | None):
    if not file_exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    inferred_split = None
    if label_path is None:
        inferred_label_path, inferred_split = infer_label_path_for_image(image_path, dataset_info)
        label_path = inferred_label_path

    return [{
        "image_path": str(image_path.resolve()),
        "label_path": str(label_path.resolve()) if label_path is not None else "",
        "inferred_split": inferred_split or "single",
    }]


def compute_average_precision(detections, total_gt):
    if total_gt == 0:
        return None
    if len(detections) == 0:
        return 0.0

    detections = sorted(detections, key=lambda item: item[0], reverse=True)
    tp = np.array([item[1] for item in detections], dtype=np.float32)
    fp = 1.0 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / (total_gt + 1e-6)
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def init_metric_state(num_classes):
    return {
        "metric_storage": {cls_id: [] for cls_id in range(num_classes)},
        "gt_counts": {cls_id: 0 for cls_id in range(num_classes)},
        "det_counts": {cls_id: 0 for cls_id in range(num_classes)},
        "total_tp": 0,
        "total_fp": 0,
        "total_fn": 0,
    }


def update_metric_state(metric_state, sample_detections, sample_targets, img_size, iou_thresh):
    gt_by_class = defaultdict(list)
    for gt in sample_targets:
        if gt.shape[0] < 5 or gt[3] <= 0 or gt[4] <= 0:
            continue
        cls_id = int(gt[0].item())
        gt_box = [
            float(gt[1].item() * img_size),
            float(gt[2].item() * img_size),
            float(gt[3].item() * img_size),
            float(gt[4].item() * img_size),
        ]
        gt_by_class[cls_id].append(gt_box)
        metric_state["gt_counts"][cls_id] += 1

    matched = {cls_id: set() for cls_id in gt_by_class}
    sample_detections = sorted(sample_detections, key=lambda det: det[4], reverse=True)

    for det in sample_detections:
        cls_id = int(det[5])
        metric_state["det_counts"][cls_id] += 1
        gt_boxes = gt_by_class.get(cls_id, [])
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched.get(cls_id, set()):
                continue
            det_tensor = torch.tensor(det[:4], dtype=torch.float32).unsqueeze(0)
            gt_tensor = torch.tensor(gt_box, dtype=torch.float32).unsqueeze(0)
            iou = float(bbox_iou_xywh(det_tensor, gt_tensor).item())
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        is_tp = best_iou >= iou_thresh
        metric_state["metric_storage"][cls_id].append((float(det[4]), 1.0 if is_tp else 0.0))
        if is_tp:
            metric_state["total_tp"] += 1
            matched.setdefault(cls_id, set()).add(best_gt_idx)
        else:
            metric_state["total_fp"] += 1

    for cls_id, gt_boxes in gt_by_class.items():
        metric_state["total_fn"] += len(gt_boxes) - len(matched.get(cls_id, set()))


def finalize_metrics(metric_state, class_names):
    total_tp = metric_state["total_tp"]
    total_fp = metric_state["total_fp"]
    total_fn = metric_state["total_fn"]

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1_score = 2.0 * precision * recall / (precision + recall + 1e-6)

    per_class_rows = []
    ap_values = []
    for cls_id in sorted(class_names):
        ap = compute_average_precision(metric_state["metric_storage"][cls_id], metric_state["gt_counts"][cls_id])
        if ap is not None:
            ap_values.append(ap)
        per_class_rows.append({
            "class_id": cls_id,
            "class_name": class_names[cls_id],
            "gt_count": metric_state["gt_counts"][cls_id],
            "det_count": metric_state["det_counts"][cls_id],
            "ap50": ap,
        })

    map50 = float(np.mean(ap_values)) if len(ap_values) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score),
        "map50": map50,
        "per_class": per_class_rows,
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_fn": int(total_fn),
    }


def tensor_to_rgb_image(image_tensor):
    img_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    if img_np.shape[2] == 1:
        img_np = np.repeat(img_np, 3, axis=2)
    return img_np


def draw_ground_truth(ax, img_np, targets, class_names, img_size):
    ax.imshow(img_np)
    ax.set_title("Ground Truth")
    ax.axis("off")

    if len(targets) == 0:
        ax.text(10, 20, "No GT labels", color="yellow", fontsize=11, bbox=dict(facecolor="black", alpha=0.6))
        return

    for target in targets:
        cls_id, x_center, y_center, width, height = target.tolist()
        cls_id = int(cls_id)
        x1 = (x_center - width / 2) * img_size
        y1 = (y_center - height / 2) * img_size
        w = width * img_size
        h = height * img_size
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="green", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 5), class_names[cls_id], color="green", fontsize=10, fontweight="bold")


def draw_predictions(ax, img_np, detections, class_names):
    ax.imshow(img_np)
    ax.set_title("Predictions")
    ax.axis("off")

    if len(detections) == 0:
        ax.text(10, 20, "No predictions", color="yellow", fontsize=11, bbox=dict(facecolor="black", alpha=0.6))
        return

    for det in detections:
        x_center, y_center, w, h, conf, cls_id = det
        cls_id = int(cls_id)
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        color = plt.cm.tab20(cls_id / max(len(class_names), 1))
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        label = f"{class_names[cls_id]}: {conf:.2f}"
        ax.text(
            x1,
            max(0, y1 - 5),
            label,
            color=color,
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.35),
        )


def save_visualization(save_path: Path, image_tensor, teacher_feature, targets, detections, class_names, img_size, figure_title):
    ensure_dir(save_path.parent)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    img_np = tensor_to_rgb_image(image_tensor)
    teacher_np = teacher_feature.detach().cpu().numpy().squeeze(0)
    teacher_np = enhance_feature_for_display(teacher_np)

    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(teacher_np, cmap="hot")
    axes[1].set_title("Optical Feature")
    axes[1].axis("off")

    draw_ground_truth(axes[2], img_np, targets, class_names, img_size)
    draw_predictions(axes[3], img_np, detections, class_names)

    fig.suptitle(figure_title, fontsize=14)
    plt.tight_layout()
    plt.savefig(windows_safe_path(save_path), dpi=130, bbox_inches="tight")
    plt.close(fig)


def evaluate_and_visualize(model, dataloader, args, class_names, output_dir, mode_name):
    device = args.device
    vis_dir = output_dir / "visualizations"
    ensure_dir(vis_dir)

    metric_state = init_metric_state(len(class_names))
    saved_visualizations = []
    evaluated_images = 0

    model.eval()
    with torch.no_grad():
        for images, targets_list, infos in dataloader:
            images = images.to(device)
            teacher_features = model.teacher(images)
            predictions = model.detector(teacher_features)
            detections = decode_detections(
                predictions,
                anchors=args.anchors,
                strides=args.strides,
                num_classes=len(class_names),
                conf_thresh=args.conf_thresh,
                nms_thresh=args.nms_thresh,
                max_det=args.max_det,
                img_size=args.img_size,
                class_agnostic=args.agnostic_nms,
            )

            for idx in range(images.shape[0]):
                evaluated_images += 1
                sample_targets = targets_list[idx]
                sample_detections = detections[idx]
                update_metric_state(metric_state, sample_detections, sample_targets, args.img_size, args.metric_iou_thresh)

                if args.max_vis_images < 0 or len(saved_visualizations) < args.max_vis_images:
                    stem = infos[idx]["stem"]
                    vis_path = vis_dir / f"{mode_name}_{evaluated_images:04d}_{stem}.png"
                    figure_title = Path(infos[idx]["image_path"]).name
                    save_visualization(
                        vis_path,
                        images[idx].cpu(),
                        teacher_features[idx].cpu(),
                        sample_targets,
                        sample_detections,
                        class_names,
                        args.img_size,
                        figure_title,
                    )
                    saved_visualizations.append(vis_path)

    metrics = finalize_metrics(metric_state, class_names)
    return metrics, saved_visualizations, evaluated_images


def build_summary_text(args, metrics, evaluated_images, saved_visualizations, checkpoint_info, mode_description):
    lines = [
        "Optical YOLO Model Results Summary",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Mode: {mode_description}",
        f"Checkpoint: {args.checkpoint}",
        f"YAML: {args.yaml}",
        f"Output directory: {args.output_dir}",
        f"Device: {args.device}",
        f"Image size: {args.img_size}",
        f"Confidence threshold: {args.conf_thresh}",
        f"NMS threshold: {args.nms_thresh}",
        f"Max detections per image: {args.max_det}",
        f"Metric IoU threshold: {args.metric_iou_thresh}",
        f"Class-agnostic NMS: {args.agnostic_nms}",
        f"Evaluated images: {evaluated_images}",
        f"Saved visualizations: {len(saved_visualizations)}",
        f"Checkpoint load info: teacher={checkpoint_info[0]}, detector={checkpoint_info[1]}",
        "",
        "[Overall Metrics]",
        f"Precision: {metrics['precision']:.6f}",
        f"Recall: {metrics['recall']:.6f}",
        f"F1: {metrics['f1']:.6f}",
        f"mAP50: {metrics['map50']:.6f}",
        f"TP: {metrics['total_tp']}",
        f"FP: {metrics['total_fp']}",
        f"FN: {metrics['total_fn']}",
        "",
        "[Per-class AP50]",
    ]

    for row in metrics["per_class"]:
        ap_text = "N/A" if row["ap50"] is None else f"{row['ap50']:.6f}"
        lines.append(
            f"- class_id={row['class_id']}, class_name={row['class_name']}, "
            f"gt_count={row['gt_count']}, det_count={row['det_count']}, ap50={ap_text}"
        )

    lines.extend([
        "",
        "[Visualization Directory]",
        str(Path(args.output_dir) / "visualizations"),
    ])
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone evaluation and visualization for optical_teacher_yolo checkpoints.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Path to teacher_detector_final.pth or compatible checkpoint.")
    parser.add_argument("--yaml", type=Path, default=DEFAULT_YAML, help="Dataset yaml path.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test", help="Dataset split for evaluation.")
    parser.add_argument("--image-path", type=Path, default=None, help="Optional single image path. If provided, single-image mode is used.")
    parser.add_argument("--label-path", type=Path, default=None, help="Optional label path for single-image mode.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for all outputs.")
    parser.add_argument("--anchor-yaml", type=Path, default=DEFAULT_ANCHOR_YAML, help="Optional external anchor yaml.")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for split evaluation.")
    parser.add_argument("--conf-thresh", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--nms-thresh", type=float, default=0.35, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=5, help="Maximum detections per image.")
    parser.add_argument("--metric-iou-thresh", type=float, default=0.5, help="IoU threshold used for TP/FP and AP50.")
    parser.add_argument("--max-eval-images", type=int, default=-1, help="Limit number of evaluated images for split mode. -1 means all.")
    parser.add_argument("--max-vis-images", type=int, default=20, help="How many 4-panel visualization figures to save. -1 means all.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device string, e.g. cuda or cpu.")
    parser.add_argument("--agnostic-nms", action="store_true", help="Use class-agnostic NMS.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.yaml = args.yaml.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.anchor_yaml is not None:
        args.anchor_yaml = args.anchor_yaml.resolve()
    if args.image_path is not None:
        args.image_path = args.image_path.resolve()
    if args.label_path is not None:
        args.label_path = args.label_path.resolve()

    ensure_dir(args.output_dir)

    dataset_info = load_dataset_info(args.yaml)
    class_names = dataset_info["class_names"]
    num_classes = dataset_info["num_classes"]
    args.anchors = load_anchor_groups(args.anchor_yaml)
    args.strides = DEFAULT_STRIDES
    args.device = torch.device(args.device)

    if args.image_path is not None:
        entries = build_entries_from_single_image(args.image_path, dataset_info, args.label_path)
        mode_description = f"single_image ({args.image_path.name})"
        loader_batch_size = 1
        mode_name = "single"
    else:
        max_eval_images = None if args.max_eval_images is None or args.max_eval_images < 0 else args.max_eval_images
        entries = build_entries_from_split(dataset_info, args.split, max_eval_images)
        mode_description = f"split={args.split}"
        loader_batch_size = max(1, args.batch_size)
        mode_name = args.split

    if len(entries) == 0:
        raise RuntimeError("No images were found for evaluation.")

    dataset = InferenceDataset(entries, img_size=args.img_size)
    dataloader = DataLoader(dataset, batch_size=loader_batch_size, shuffle=False, collate_fn=collate_inference)

    model = TeacherWithDetector(num_classes=num_classes).to(args.device)
    checkpoint_info = load_joint_checkpoint(model, args.checkpoint, args.device)

    metrics, saved_visualizations, evaluated_images = evaluate_and_visualize(
        model=model,
        dataloader=dataloader,
        args=args,
        class_names=class_names,
        output_dir=args.output_dir,
        mode_name=mode_name,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = args.output_dir / f"model_results_summary_{timestamp}.txt"
    summary_text = build_summary_text(
        args=args,
        metrics=metrics,
        evaluated_images=evaluated_images,
        saved_visualizations=saved_visualizations,
        checkpoint_info=checkpoint_info,
        mode_description=mode_description,
    )
    write_text(summary_path, summary_text)

    print(summary_text)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
