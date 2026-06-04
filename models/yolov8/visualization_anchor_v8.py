import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.runtime import unwrap_module
from models.teacher_guidance import enhance_feature_for_display
from .decode_anchor_v8 import decode_detections_anchor_v8


def wh_iou_scalar(w1, h1, w2, h2, eps=1e-6):
    inter = min(float(w1), float(w2)) * min(float(h1), float(h2))
    union = float(w1) * float(h1) + float(w2) * float(h2) - inter + eps
    return inter / union


def draw_best_matching_anchor_boxes(config, ax, x_center, y_center, width, height):
    anchor_colors = ["#ffd166", "#00d1ff", "#ff5db1"]
    for scale_idx, scale_anchors in enumerate(config.ANCHORS):
        best_anchor = scale_anchors[0]
        best_iou = -1.0
        for anchor_w, anchor_h in scale_anchors:
            match_iou = wh_iou_scalar(width, height, anchor_w, anchor_h)
            if match_iou > best_iou:
                best_iou = match_iou
                best_anchor = (anchor_w, anchor_h)
        anchor_w, anchor_h = best_anchor
        x1 = x_center - anchor_w / 2.0
        y1 = y_center - anchor_h / 2.0
        ax.add_patch(
            plt.Rectangle(
                (x1, y1),
                anchor_w,
                anchor_h,
                linewidth=1.1,
                edgecolor=anchor_colors[scale_idx % len(anchor_colors)],
                facecolor="none",
                linestyle="--",
                alpha=0.85,
            )
        )
        ax.text(
            x1,
            max(8, y1 - 4),
            f"A@{config.STRIDES[scale_idx]}",
            color=anchor_colors[scale_idx % len(anchor_colors)],
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.25, edgecolor="none"),
        )


def save_detection_visualization_anchor_v8(config, epoch, model, dataset, save_dir, prefix="train", device=None):
    os.makedirs(save_dir, exist_ok=True)
    model_core = unwrap_module(model)
    was_training = model_core.training
    model_core.eval()
    rng = np.random.default_rng(config.VIS_SEED + epoch)
    sample_count = min(config.VIS_BATCH_SIZE, len(dataset))
    if sample_count <= 0:
        return
    sample_indices = rng.choice(len(dataset), size=sample_count, replace=False)
    fig, axes = plt.subplots(sample_count, 4, figsize=(24, 6 * sample_count))
    axes = np.asarray(axes).reshape(sample_count, 4)

    with torch.no_grad():
        for row, sample_idx in enumerate(sample_indices):
            image, targets = dataset[int(sample_idx)]
            input_tensor = image.unsqueeze(0).to(device)
            teacher_feature, predictions = model_core(input_tensor, return_feature=True)
            detections = decode_detections_anchor_v8(
                config,
                predictions,
                conf_thresh=config.VIS_CONF_THRESH,
                nms_thresh=config.VIS_NMS_THRESH,
                max_det=config.VIS_MAX_DET,
            )[0]
            img_np = image.squeeze(0).cpu().numpy()
            feat_np = enhance_feature_for_display(teacher_feature.squeeze().detach().cpu().numpy())

            axes[row, 0].imshow(img_np, cmap="gray")
            axes[row, 0].set_title("Input")
            axes[row, 1].imshow(feat_np, cmap="magma")
            axes[row, 1].set_title("Teacher feature")
            axes[row, 2].imshow(img_np, cmap="gray")
            axes[row, 2].set_title("Ground Truth + anchors")
            axes[row, 3].imshow(img_np, cmap="gray")
            axes[row, 3].set_title("Predictions")

            target_indices_for_anchor_overlay = []
            if len(targets) > 0 and config.VIS_SHOW_BEST_MATCHED_ANCHORS:
                target_areas = [
                    float(targets[target_idx][3].item() * config.IMG_SIZE)
                    * float(targets[target_idx][4].item() * config.IMG_SIZE)
                    for target_idx in range(len(targets))
                ]
                overlay_count = min(config.VIS_MAX_GT_ANCHOR_OVERLAYS, len(targets))
                target_indices_for_anchor_overlay = sorted(range(len(targets)), key=lambda idx: target_areas[idx], reverse=True)[:overlay_count]

            for target_idx in range(len(targets)):
                cls_id, cx, cy, w, h = targets[target_idx].tolist()
                cx_px = cx * config.IMG_SIZE
                cy_px = cy * config.IMG_SIZE
                w_px = w * config.IMG_SIZE
                h_px = h * config.IMG_SIZE
                x1 = cx_px - w_px / 2
                y1 = cy_px - h_px / 2
                axes[row, 2].add_patch(plt.Rectangle((x1, y1), w_px, h_px, fill=False, edgecolor="lime", linewidth=1.8))
                axes[row, 2].text(x1, y1 - 4, config.CLASS_NAMES[int(cls_id)], color="lime", fontsize=8)
                if target_idx in target_indices_for_anchor_overlay:
                    draw_best_matching_anchor_boxes(config, axes[row, 2], cx_px, cy_px, w_px, h_px)

            for det in detections:
                cx, cy, w, h, conf, cls_id = det
                x1 = cx - w / 2
                y1 = cy - h / 2
                color = plt.cm.tab20(int(cls_id) / max(config.NUM_CLASSES, 1))
                axes[row, 3].add_patch(plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=1.6))
                axes[row, 3].text(
                    x1,
                    y1 - 5,
                    f"{config.CLASS_NAMES[int(cls_id)]}: {conf:.2f}",
                    color=color,
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.35),
                )

            for col in range(4):
                axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_epoch_{epoch:03d}.png"), dpi=config.VIS_DPI)
    plt.close()
    if was_training:
        model.train()
