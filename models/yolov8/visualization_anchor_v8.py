import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.class_display import class_name_for_id
from models.runtime import unwrap_module
from models.teacher_guidance import enhance_feature_for_display
from .detection_protocol import decode_detections


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
            detections = decode_detections(
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
            axes[row, 2].set_title("Ground Truth")
            axes[row, 3].imshow(img_np, cmap="gray")
            axes[row, 3].set_title("Predictions")

            for target_idx in range(len(targets)):
                cls_id, cx, cy, w, h = targets[target_idx].tolist()
                image_h, image_w = config.RESOLUTION
                cx_px = cx * image_w
                cy_px = cy * image_h
                w_px = w * image_w
                h_px = h * image_h
                x1 = cx_px - w_px / 2
                y1 = cy_px - h_px / 2
                axes[row, 2].add_patch(plt.Rectangle((x1, y1), w_px, h_px, fill=False, edgecolor="lime", linewidth=1.8))
                axes[row, 2].text(
                    x1, y1 - 4,
                    class_name_for_id(config.CLASS_NAMES, int(cls_id)),
                    color="lime", fontsize=8,
                )

            for det in detections:
                cx, cy, w, h, conf, cls_id = det
                x1 = cx - w / 2
                y1 = cy - h / 2
                color = "red"
                axes[row, 3].add_patch(plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=2.6))
                axes[row, 3].text(
                    x1,
                    y1 - 5,
                    f"{class_name_for_id(config.CLASS_NAMES, int(cls_id))}: {conf:.2f}",
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.35),
                )

            for col in range(4):
                axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_epoch_{epoch:03d}.png"), dpi=config.VIS_DPI)
    plt.close()
    if was_training:
        model.train()
