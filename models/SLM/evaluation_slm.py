import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from models.geometry import bbox_iou_xywh
from models.runtime import unwrap_module
from models.teacher_guidance import enhance_feature_for_display
from models.yolov8.decode_anchor_v8 import decode_detections_anchor_v8
from models.yolov8.metrics_anchor_v8 import compute_average_precision


def _move_batch_to_device(config, batch, device):
    gray = batch["gray_tensor"].to(device, non_blocking=config.PIN_MEMORY)
    teacher_input = batch["rgb_tensor"].to(device, non_blocking=config.PIN_MEMORY)
    if config.ENABLE_CHANNELS_LAST and torch.cuda.is_available():
        gray = gray.contiguous(memory_format=torch.channels_last)
        teacher_input = teacher_input.contiguous(memory_format=torch.channels_last)
    return gray, teacher_input, batch["targets"]


def evaluate_slm_detector(config, teacher, student, detector, dataloader, detection_criterion, feature_criterion, device, stage_name):
    teacher_core = unwrap_module(teacher)
    student_core = unwrap_module(student)
    detector_core = unwrap_module(detector)
    was_student_training = student_core.training
    was_detector_training = detector_core.training
    teacher_core.eval()
    student_core.eval()
    detector_core.eval()

    metric_storage = {cls_id: [] for cls_id in range(config.NUM_CLASSES)}
    gt_counts = {cls_id: 0 for cls_id in range(config.NUM_CLASSES)}
    totals = {key: 0.0 for key in ("total", "feature", "detection", "box", "obj", "noobj", "cls")}
    total_tp = total_fp = total_fn = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="SLM validation", leave=False):
            gray, teacher_input, targets = _move_batch_to_device(config, batch, device)
            teacher_feature = teacher_core(teacher_input)
            student_feature = student_core(gray)
            feature_loss, _ = feature_criterion(student_feature, teacher_feature, student_core)

            evaluate_detector = stage_name not in {"student_only", "student_adapt_max"}
            if evaluate_detector:
                predictions = detector_core(student_feature)
                detection_loss, loss_stats = detection_criterion(predictions, targets)
            else:
                predictions = None
                detection_loss = torch.zeros((), device=device)
                loss_stats = {"box": 0.0, "obj": 0.0, "noobj": 0.0, "cls": 0.0}

            if stage_name == "student_only":
                total_loss = feature_loss * config.FEATURE_LOSS_WEIGHT_STUDENT + detection_loss * config.DETECTION_LOSS_WEIGHT_STUDENT
            elif stage_name == "student_adapt_max":
                total_loss = feature_loss * config.FEATURE_LOSS_WEIGHT_ADAPT
            elif stage_name == "detector_only":
                total_loss = detection_loss * config.DETECTION_LOSS_WEIGHT_DETECTOR
            else:
                total_loss = feature_loss * config.FEATURE_LOSS_WEIGHT_JOINT + detection_loss * config.DETECTION_LOSS_WEIGHT_JOINT

            totals["total"] += float(total_loss.detach().item())
            totals["feature"] += float(feature_loss.detach().item())
            totals["detection"] += float(detection_loss.detach().item())
            for key in ("box", "obj", "noobj", "cls"):
                totals[key] += loss_stats.get(key, 0.0)

            if not evaluate_detector:
                continue

            detections = decode_detections_anchor_v8(config, predictions)
            for sample_idx, sample_detections in enumerate(detections):
                gt_by_class = {}
                for gt in targets[sample_idx]:
                    if gt.shape[0] < 5 or gt[3] <= 0 or gt[4] <= 0:
                        continue
                    cls_id = int(gt[0].item())
                    gt_box = [
                        float(gt[1].item() * config.IMG_SIZE),
                        float(gt[2].item() * config.IMG_SIZE),
                        float(gt[3].item() * config.IMG_SIZE),
                        float(gt[4].item() * config.IMG_SIZE),
                    ]
                    gt_by_class.setdefault(cls_id, []).append(gt_box)
                    gt_counts[cls_id] += 1
                matched = {cls_id: set() for cls_id in gt_by_class}
                for det in sorted(sample_detections, key=lambda item: item[4], reverse=True):
                    cls_id = int(det[5])
                    best_iou, best_gt_idx = 0.0, -1
                    for gt_idx, gt_box in enumerate(gt_by_class.get(cls_id, [])):
                        if gt_idx in matched.get(cls_id, set()):
                            continue
                        iou = float(
                            bbox_iou_xywh(
                                torch.tensor(det[:4]).float().unsqueeze(0),
                                torch.tensor(gt_box).float().unsqueeze(0),
                            ).item()
                        )
                        if iou > best_iou:
                            best_iou, best_gt_idx = iou, gt_idx
                    is_tp = best_iou >= config.METRIC_IOU_THRESHOLD
                    metric_storage[cls_id].append((float(det[4]), 1.0 if is_tp else 0.0))
                    if is_tp:
                        total_tp += 1
                        matched.setdefault(cls_id, set()).add(best_gt_idx)
                    else:
                        total_fp += 1
                for cls_id, gt_boxes in gt_by_class.items():
                    total_fn += len(gt_boxes) - len(matched.get(cls_id, set()))

    num_batches = max(len(dataloader), 1)
    avg_losses = {key: value / num_batches for key, value in totals.items()}
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1_score = 2.0 * precision * recall / (precision + recall + 1e-6)
    ap_values = []
    for cls_id in range(config.NUM_CLASSES):
        ap = compute_average_precision(metric_storage[cls_id], gt_counts[cls_id])
        if ap is not None:
            ap_values.append(ap)
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score),
        "map50": float(np.mean(ap_values)) if ap_values else 0.0,
        "stage": stage_name,
    }

    if was_student_training:
        student_core.train()
    if was_detector_training:
        detector_core.train()
    return avg_losses, metrics


def save_slm_detection_visualization(config, epoch, teacher, student, detector, dataset, save_dir, prefix="val", device=None):
    os.makedirs(save_dir, exist_ok=True)
    teacher_core = unwrap_module(teacher)
    student_core = unwrap_module(student)
    detector_core = unwrap_module(detector)
    was_student_training = student_core.training
    was_detector_training = detector_core.training
    teacher_core.eval()
    student_core.eval()
    detector_core.eval()

    rng = np.random.default_rng(config.VIS_SEED + epoch)
    sample_count = min(config.VIS_BATCH_SIZE, len(dataset))
    if sample_count <= 0:
        return
    sample_indices = rng.choice(len(dataset), size=sample_count, replace=False)
    fig, axes = plt.subplots(sample_count, 4, figsize=(24, 6 * sample_count))
    axes = np.asarray(axes).reshape(sample_count, 4)

    with torch.no_grad():
        for row, sample_idx in enumerate(sample_indices):
            sample = dataset[int(sample_idx)]
            gray = sample["gray_tensor"]
            teacher_input = sample["rgb_tensor"]
            targets = sample["targets"]
            gray_batch = gray.unsqueeze(0).to(device)
            teacher_batch = teacher_input.unsqueeze(0).to(device)
            if config.ENABLE_CHANNELS_LAST and torch.cuda.is_available():
                gray_batch = gray_batch.contiguous(memory_format=torch.channels_last)
                teacher_batch = teacher_batch.contiguous(memory_format=torch.channels_last)
            teacher_feature = teacher_core(teacher_batch)
            student_feature = student_core(gray_batch)
            predictions = detector_core(student_feature)
            detections = decode_detections_anchor_v8(
                config,
                predictions,
                conf_thresh=config.VIS_CONF_THRESH,
                nms_thresh=config.VIS_NMS_THRESH,
                max_det=config.VIS_MAX_DET,
            )[0]

            img_np = gray.squeeze(0).cpu().numpy()
            teacher_np = enhance_feature_for_display(teacher_feature.squeeze().detach().cpu().numpy())
            student_np = enhance_feature_for_display(student_feature.squeeze().detach().cpu().numpy())
            axes[row, 0].imshow(img_np, cmap="gray")
            axes[row, 0].set_title("Input")
            axes[row, 1].imshow(teacher_np, cmap="magma")
            axes[row, 1].set_title("Teacher feature")
            axes[row, 2].imshow(student_np, cmap="magma")
            axes[row, 2].set_title("SLM student feature")
            axes[row, 3].imshow(img_np, cmap="gray")
            axes[row, 3].set_title("GT + Predictions")

            for target_idx in range(len(targets)):
                cls_id, cx, cy, w, h = targets[target_idx].tolist()
                cx_px = cx * config.IMG_SIZE
                cy_px = cy * config.IMG_SIZE
                w_px = w * config.IMG_SIZE
                h_px = h * config.IMG_SIZE
                x1 = cx_px - w_px / 2
                y1 = cy_px - h_px / 2
                axes[row, 3].add_patch(
                    plt.Rectangle((x1, y1), w_px, h_px, fill=False, edgecolor="lime", linewidth=1.8, linestyle="--")
                )
                axes[row, 3].text(
                    x1,
                    y1 + 10,
                    str(config.CLASS_NAMES[int(cls_id)]),
                    color="lime",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.20", facecolor="black", alpha=0.35, edgecolor="none"),
                )

            for det in detections:
                cx, cy, w, h, conf, cls_id = det
                x1 = cx - w / 2
                y1 = cy - h / 2
                color = plt.cm.tab20(int(cls_id) / max(config.NUM_CLASSES, 1))
                axes[row, 3].add_patch(plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=1.6))
                axes[row, 3].text(
                    x1,
                    y1 + 10,
                    f"{config.CLASS_NAMES[int(cls_id)]} {conf:.2f}",
                    color=color,
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.20", facecolor="black", alpha=0.35, edgecolor="none"),
                )

            for col in range(4):
                axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_epoch_{epoch:03d}.png"), dpi=config.VIS_DPI)
    plt.close(fig)
    if was_student_training:
        student_core.train()
    if was_detector_training:
        detector_core.train()
