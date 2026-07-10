"""Compact detection visualization helpers.

Contains TensorBoard image logging and PNG file visualization
functions split out from train_loop.py.
"""

import os
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps

from models.runtime import log_to_file, unwrap_module
from models.teacher_guidance import enhance_feature_for_display

from .config import ConfigCompactDetect as Config
from .decode import decode_center_detections


def draw_box(draw, box_xywh, color, label):
    cx, cy, bw, bh = [float(v) for v in box_xywh]
    x1 = max(0.0, cx - bw / 2.0)
    y1 = max(0.0, cy - bh / 2.0)
    x2 = min(float(Config.IMG_SIZE - 1), cx + bw / 2.0)
    y2 = min(float(Config.IMG_SIZE - 1), cy + bh / 2.0)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    if label:
        draw.text((x1 + 2, max(0.0, y1 - 12)), label, fill=color)


def normalize_feature_map(feature):
    feature = feature.detach().float().cpu()
    flat_min = feature.amin(dim=(-2, -1), keepdim=True)
    flat_max = feature.amax(dim=(-2, -1), keepdim=True)
    return ((feature - flat_min) / (flat_max - flat_min + 1e-6)).clamp(0, 1)


def phase_to_tensorboard_image(phase):
    phase = phase.detach().float().cpu()
    phase = torch.remainder(phase, 2 * np.pi)
    return (phase / (2 * np.pi)).clamp(0, 1)


def feature_to_magma_image(feature):
    feature_np = feature.squeeze(0).detach().float().cpu().numpy()
    feature_np = enhance_feature_for_display(feature_np)
    rgb = colormaps["magma"](feature_np)[..., :3]
    return torch.from_numpy(np.asarray(rgb, dtype=np.float32).copy()).permute(2, 0, 1)


def select_visualization_indices(dataset_size, max_images):
    sample_count = min(int(max_images), int(dataset_size))
    if sample_count <= 0:
        return []
    rng = np.random.default_rng(int(getattr(Config, "VIS_SEED", 20260504)))
    return rng.choice(dataset_size, size=sample_count, replace=False).tolist()


def forward_student_with_norm(student, images, enable_norm):
    student_module = unwrap_module(student)
    if not hasattr(student_module, "enable_norm"):
        return student(images)
    previous = student_module.enable_norm
    student_module.enable_norm = bool(enable_norm)
    try:
        return student(images)
    finally:
        student_module.enable_norm = previous


@torch.no_grad()
def add_tensorboard_visualization(writer, step, dataset, student, detector, device, max_images=None):
    if writer is None or dataset is None or len(dataset) == 0:
        return
    if max_images is None:
        max_images = int(getattr(Config, "VIS_MAX_IMAGES", 4))
    student.eval()
    detector.eval()
    sample_indices = select_visualization_indices(len(dataset), max_images)
    items = [dataset[idx] for idx in sample_indices]
    images = torch.stack([item["gray_tensor"] for item in items], dim=0).to(device, non_blocking=Config.PIN_MEMORY)
    targets = [item["targets"] for item in items]
    fp32_ctx = torch.amp.autocast(device_type="cuda", enabled=False) if device.type == "cuda" else nullcontext()
    with fp32_ctx:
        optical_features = student(images.float())
        raw_optical_features = forward_student_with_norm(student, images.float(), enable_norm=False)
        pred = detector(optical_features)
    detections = decode_center_detections(
        Config,
        pred,
        conf_thresh=Config.CONF_THRESH,
        nms_thresh=Config.NMS_THRESH,
        max_det=min(Config.MAX_DET, 30),
    )

    rendered = []
    for idx, item in enumerate(items):
        gray = item["gray_tensor"].squeeze(0).detach().cpu().clamp(0, 1).numpy()
        canvas = Image.fromarray((gray * 255).astype(np.uint8), mode="L").convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for gt in targets[idx]:
            if gt.numel() < 5:
                continue
            cls_id = int(gt[0].item())
            box = [
                gt[1].item() * Config.IMG_SIZE,
                gt[2].item() * Config.IMG_SIZE,
                gt[3].item() * Config.IMG_SIZE,
                gt[4].item() * Config.IMG_SIZE,
            ]
            draw_box(draw, box, (60, 220, 80), f"GT {Config.CLASS_NAMES.get(cls_id, cls_id)}")
        for det in detections[idx]:
            cls_id = int(det[5])
            label = f"{Config.CLASS_NAMES.get(cls_id, cls_id)} {float(det[4]):.2f}"
            draw_box(draw, det[:4], (255, 70, 70), label)
        rendered.append(torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 255.0)
    grid = torch.cat(rendered, dim=2)
    writer.add_image("Visualization/val_gt_green_pred_red", grid, step)

    input_grid = torch.cat([item["gray_tensor"].detach().cpu().clamp(0, 1) for item in items], dim=2)
    writer.add_image("Visualization/input_gray", input_grid, step)

    optical_gray_grid = torch.cat([f for f in normalize_feature_map(raw_optical_features)], dim=2)
    writer.add_image("Visualization/optical_modulated_output_gray", optical_gray_grid, step)

    student_module = unwrap_module(student)
    phase_images = []
    for layer_name in ("slm1", "slm2"):
        slm = getattr(student_module, layer_name, None)
        if slm is None:
            continue
        phase = slm.wrapped_phase()
        writer.add_scalar(f"OpticalPhase/{layer_name}_min_rad", float(phase.min().detach().cpu()), step)
        writer.add_scalar(f"OpticalPhase/{layer_name}_max_rad", float(phase.max().detach().cpu()), step)
        phase_images.append(phase_to_tensorboard_image(phase).squeeze(0))
    if phase_images:
        phase_grid = torch.cat(phase_images, dim=2)
        writer.add_image("Visualization/slm_wrapped_phase", phase_grid, step)
    writer.flush()


@torch.no_grad()
def save_compact_visualization_png(epoch, dataset, student, detector, device, save_dir=None, max_images=None, prefix="val"):
    if dataset is None or len(dataset) == 0:
        return
    if save_dir is None:
        save_dir = Config.VISUALIZATION_DIR
    if max_images is None:
        max_images = int(getattr(Config, "VIS_FILE_MAX_IMAGES", 3))
    os.makedirs(save_dir, exist_ok=True)

    student_module = unwrap_module(student)
    detector_module = unwrap_module(detector)
    was_student_wrapper_training = student.training
    was_detector_wrapper_training = detector.training
    was_student_training = student_module.training
    was_detector_training = detector_module.training
    student.eval()
    detector.eval()

    sample_indices = select_visualization_indices(len(dataset), max_images)
    sample_count = len(sample_indices)
    if sample_count <= 0:
        student.train(was_student_wrapper_training)
        detector.train(was_detector_wrapper_training)
        student_module.train(was_student_training)
        detector_module.train(was_detector_training)
        return

    fig, axes = plt.subplots(sample_count, 3, figsize=(18, 6 * sample_count))
    axes = np.asarray(axes).reshape(sample_count, 3)
    fp32_ctx = torch.amp.autocast(device_type="cuda", enabled=False) if device.type == "cuda" else nullcontext()

    with fp32_ctx:
        for row, sample_idx in enumerate(sample_indices):
            sample = dataset[int(sample_idx)]
            gray = sample["gray_tensor"]
            targets = sample["targets"]
            gray_batch = gray.unsqueeze(0).to(device, non_blocking=Config.PIN_MEMORY).float()

            detector_feature = student(gray_batch)
            raw_feature = forward_student_with_norm(student, gray_batch, enable_norm=False)
            predictions = detector(detector_feature)
            detections = decode_center_detections(
                Config,
                predictions,
                conf_thresh=Config.CONF_THRESH,
                nms_thresh=Config.NMS_THRESH,
                max_det=min(Config.MAX_DET, int(getattr(Config, "VIS_MAX_DET", 5))),
            )[0]

            img_np = gray.squeeze(0).detach().cpu().numpy()
            feature_np = enhance_feature_for_display(raw_feature.squeeze().detach().cpu().numpy())
            axes[row, 0].imshow(img_np, cmap="gray")
            axes[row, 0].set_title("Input")
            axes[row, 1].imshow(feature_np, cmap="magma")
            axes[row, 1].set_title("SLM optical feature")
            axes[row, 2].imshow(img_np, cmap="gray")
            axes[row, 2].set_title("GT + Predictions")

            for target in targets:
                if target.numel() < 5:
                    continue
                cls_id, cx, cy, w, h = target.tolist()
                x1 = (cx - w / 2) * Config.IMG_SIZE
                y1 = (cy - h / 2) * Config.IMG_SIZE
                axes[row, 2].add_patch(
                    plt.Rectangle(
                        (x1, y1),
                        w * Config.IMG_SIZE,
                        h * Config.IMG_SIZE,
                        fill=False,
                        edgecolor="lime",
                        linewidth=1.8,
                        linestyle="--",
                    )
                )
                axes[row, 2].text(
                    x1,
                    y1 + 10,
                    str(Config.CLASS_NAMES.get(int(cls_id), int(cls_id))),
                    color="lime",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.20", facecolor="black", alpha=0.35, edgecolor="none"),
                )

            for det in detections:
                cx, cy, w, h, conf, cls_id = [float(v) for v in det]
                x1 = cx - w / 2
                y1 = cy - h / 2
                axes[row, 2].add_patch(
                    plt.Rectangle((x1, y1), w, h, fill=False, edgecolor="red", linewidth=1.6)
                )
                axes[row, 2].text(
                    x1,
                    y1 + 10,
                    f"{Config.CLASS_NAMES.get(int(cls_id), int(cls_id))} {conf:.2f}",
                    color="red",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.20", facecolor="black", alpha=0.35, edgecolor="none"),
                )

            for col in range(3):
                axes[row, col].axis("off")

    plt.tight_layout()
    output_path = os.path.join(save_dir, f"{prefix}_epoch_{int(epoch):03d}.png")
    plt.savefig(output_path, dpi=int(getattr(Config, "VIS_DPI", 130)))
    plt.close(fig)
    student.train(was_student_wrapper_training)
    detector.train(was_detector_wrapper_training)
    student_module.train(was_student_training)
    detector_module.train(was_detector_training)
    log_to_file(Config, f"Saved compact visualization: {output_path}")
