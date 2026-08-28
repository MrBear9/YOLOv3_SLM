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

from models.class_display import class_name_for_id
from models.compact_detect.compact_utils import _slm_layer_names
from models.runtime import log_to_file, unwrap_module
from models.teacher_guidance import enhance_feature_for_display

from .config import ConfigCompactDetect as Config
from .factory import build_compact_decode_fn


def draw_box(draw, box_xywh, color, label):
    cx, cy, bw, bh = [float(v) for v in box_xywh]
    x1 = max(0.0, cx - bw / 2.0)
    y1 = max(0.0, cy - bh / 2.0)
    image_h, image_w = Config.RESOLUTION
    x2 = min(float(image_w - 1), cx + bw / 2.0)
    y2 = min(float(image_h - 1), cy + bh / 2.0)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    if label:
        draw.text((x1 + 2, max(0.0, y1 - 12)), label, fill=color)


def normalize_feature_map(feature):
    feature = feature.detach().float().cpu()
    flat_min = feature.amin(dim=(-2, -1), keepdim=True)
    flat_max = feature.amax(dim=(-2, -1), keepdim=True)
    return ((feature - flat_min) / (flat_max - flat_min + 1e-6)).clamp(0, 1)


def feature_to_gray_batch(feature):
    if feature.dim() == 3:
        feature = feature.unsqueeze(1)
    if feature.size(1) > 1:
        feature = feature.mean(dim=1, keepdim=True)
    return normalize_feature_map(feature)


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
def add_tensorboard_teacher_feature(writer, step, dataset, teacher, device, max_images=None, prefix="val"):
    if writer is None or dataset is None or teacher is None or len(dataset) == 0:
        return
    if max_images is None:
        max_images = int(getattr(Config, "VIS_MAX_IMAGES", 4))
    was_training = teacher.training
    teacher.eval()
    sample_indices = select_visualization_indices(len(dataset), max_images)
    items = [dataset[idx] for idx in sample_indices]
    teacher_inputs = torch.stack([item["rgb_tensor"] for item in items], dim=0).to(device, non_blocking=Config.PIN_MEMORY)
    fp32_ctx = torch.amp.autocast(device_type="cuda", enabled=False) if device.type == "cuda" else nullcontext()
    with fp32_ctx:
        teacher_feature = teacher(teacher_inputs.float())
    teacher_grid = torch.cat([feature for feature in feature_to_gray_batch(teacher_feature)], dim=2)
    writer.add_image(f"Visualization/{prefix}_teacher_feature_gray", teacher_grid, step)
    writer.flush()
    teacher.train(was_training)


@torch.no_grad()
def add_tensorboard_visualization(writer, step, dataset, student, detector, device, max_images=None, prefix="val"):
    if writer is None or dataset is None or len(dataset) == 0:
        return
    if max_images is None:
        max_images = int(getattr(Config, "VIS_MAX_IMAGES", 4))
    student.eval()
    detector.eval()
    decode_fn = build_compact_decode_fn(Config)
    sample_indices = select_visualization_indices(len(dataset), max_images)
    items = [dataset[idx] for idx in sample_indices]
    images = torch.stack([item["gray_tensor"] for item in items], dim=0).to(device, non_blocking=Config.PIN_MEMORY)
    targets = [item["targets"] for item in items]
    fp32_ctx = torch.amp.autocast(device_type="cuda", enabled=False) if device.type == "cuda" else nullcontext()
    with fp32_ctx:
        optical_features = student(images.float())
        raw_optical_features = forward_student_with_norm(student, images.float(), enable_norm=False)
        pred = detector(optical_features)
    detections = decode_fn(
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
            image_h, image_w = Config.RESOLUTION
            box = [
                gt[1].item() * image_w,
                gt[2].item() * image_h,
                gt[3].item() * image_w,
                gt[4].item() * image_h,
            ]
            draw_box(
                draw, box, (60, 220, 80),
                f"GT {class_name_for_id(Config.CLASS_NAMES, cls_id, str(cls_id))}",
            )
        for det in detections[idx]:
            cls_id = int(det[5])
            label = (
                f"{class_name_for_id(Config.CLASS_NAMES, cls_id, str(cls_id))} "
                f"{float(det[4]):.2f}"
            )
            draw_box(draw, det[:4], (255, 70, 70), label)
        rendered.append(torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 255.0)
    grid = torch.cat(rendered, dim=2)
    writer.add_image(f"Visualization/{prefix}_gt_green_pred_red", grid, step)

    optical_gray_grid = torch.cat([f for f in normalize_feature_map(raw_optical_features)], dim=2)
    writer.add_image(f"Visualization/{prefix}_optical_modulated_output_gray", optical_gray_grid, step)

    student_module = unwrap_module(student)
    phase_images = []
    for layer_name in _slm_layer_names(student_module):
        slm = getattr(student_module, layer_name, None)
        if slm is None:
            continue
        phase = slm.effective_phase()
        writer.add_scalar(f"OpticalPhase/{layer_name}_effective_min_rad", float(phase.min().detach().cpu()), step)
        writer.add_scalar(f"OpticalPhase/{layer_name}_effective_max_rad", float(phase.max().detach().cpu()), step)
        writer.add_image(
            f"Visualization/{layer_name}_gray_drive",
            slm.phase_to_gray_uint8().float().squeeze(0) / (slm.phase_levels - 1),
            step,
        )
        phase_images.append(phase_to_tensorboard_image(phase).squeeze(0))
    if phase_images:
        phase_grid = torch.cat(phase_images, dim=2)
        writer.add_image("Visualization/slm_effective_phase", phase_grid, step)
    writer.flush()


@torch.no_grad()
def save_compact_visualization_png(epoch, dataset, student, detector, device, teacher=None, save_dir=None, max_images=None, prefix="val"):
    if dataset is None or len(dataset) == 0:
        return
    if save_dir is None:
        save_dir = Config.VISUALIZATION_DIR
    if max_images is None:
        max_images = int(getattr(Config, "VIS_FILE_MAX_IMAGES", 3))
    os.makedirs(save_dir, exist_ok=True)

    decode_fn = build_compact_decode_fn(Config)
    student_module = unwrap_module(student)
    detector_module = unwrap_module(detector)
    was_student_wrapper_training = student.training
    was_detector_wrapper_training = detector.training
    was_student_training = student_module.training
    was_detector_training = detector_module.training
    student.eval()
    detector.eval()
    if teacher is not None:
        was_teacher_training = teacher.training
        teacher.eval()

    sample_indices = select_visualization_indices(len(dataset), max_images)
    sample_count = len(sample_indices)
    if sample_count <= 0:
        student.train(was_student_wrapper_training)
        detector.train(was_detector_wrapper_training)
        student_module.train(was_student_training)
        detector_module.train(was_detector_training)
        if teacher is not None:
            teacher.train(was_teacher_training)
        return

    ncols = 4 if teacher is not None else 3
    fig, axes = plt.subplots(sample_count, ncols, figsize=(6 * ncols, 6 * sample_count))
    axes = np.asarray(axes).reshape(sample_count, ncols)
    fp32_ctx = torch.amp.autocast(device_type="cuda", enabled=False) if device.type == "cuda" else nullcontext()

    col_names = ["Input", "Teacher feature", "SLM optical feature", "GT + Predictions"] if teacher is not None else ["Input", "SLM optical feature", "GT + Predictions"]

    with fp32_ctx:
        for row, sample_idx in enumerate(sample_indices):
            sample = dataset[int(sample_idx)]
            gray = sample["gray_tensor"]
            targets = sample["targets"]
            gray_batch = gray.unsqueeze(0).to(device, non_blocking=Config.PIN_MEMORY).float()

            detector_feature = student(gray_batch)
            raw_feature = forward_student_with_norm(student, gray_batch, enable_norm=False)
            predictions = detector(detector_feature)
            detections = decode_fn(
                Config,
                predictions,
                conf_thresh=Config.CONF_THRESH,
                nms_thresh=Config.NMS_THRESH,
                max_det=min(Config.MAX_DET, int(getattr(Config, "VIS_MAX_DET", 5))),
            )[0]

            img_np = gray.squeeze(0).detach().cpu().numpy()

            col = 0
            # Column 0: Input
            axes[row, col].imshow(img_np, cmap="gray")
            axes[row, col].set_title(col_names[col])
            col += 1

            # Column 1 (teacher present): Teacher 1ch feature
            if teacher is not None:
                rgb = sample["rgb_tensor"].unsqueeze(0).to(device, non_blocking=Config.PIN_MEMORY).float()
                teacher_feat = teacher(rgb)
                teacher_np = enhance_feature_for_display(teacher_feat.squeeze().detach().cpu().numpy())
                axes[row, col].imshow(teacher_np, cmap="magma")
                axes[row, col].set_title(col_names[col])
                col += 1

            # Next column: SLM optical feature
            feature_np = enhance_feature_for_display(raw_feature.squeeze().detach().cpu().numpy())
            axes[row, col].imshow(feature_np, cmap="magma")
            axes[row, col].set_title(col_names[col])
            col += 1

            # Last column: GT + Predictions
            axes[row, col].imshow(img_np, cmap="gray")
            axes[row, col].set_title(col_names[col])

            for target in targets:
                if target.numel() < 5:
                    continue
                cls_id, cx, cy, w, h = target.tolist()
                image_h, image_w = Config.RESOLUTION
                x1 = (cx - w / 2) * image_w
                y1 = (cy - h / 2) * image_h
                axes[row, col].add_patch(
                    plt.Rectangle(
                        (x1, y1),
                        w * image_w,
                        h * image_h,
                        fill=False,
                        edgecolor="lime",
                        linewidth=1.8,
                        linestyle="--",
                    )
                )
                axes[row, col].text(
                    x1,
                    y1 + 10,
                    class_name_for_id(Config.CLASS_NAMES, int(cls_id), str(int(cls_id))),
                    color="lime",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.20", facecolor="black", alpha=0.35, edgecolor="none"),
                )

            for det in detections:
                cx, cy, w, h, conf, cls_id = [float(v) for v in det]
                x1 = cx - w / 2
                y1 = cy - h / 2
                axes[row, col].add_patch(
                    plt.Rectangle((x1, y1), w, h, fill=False, edgecolor="red", linewidth=1.6)
                )
                axes[row, col].text(
                    x1,
                    y1 + 10,
                    f"{class_name_for_id(Config.CLASS_NAMES, int(cls_id), str(int(cls_id)))} {conf:.2f}",
                    color="red",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.20", facecolor="black", alpha=0.35, edgecolor="none"),
                )

            for c in range(ncols):
                axes[row, c].axis("off")

    plt.tight_layout()
    output_path = os.path.join(save_dir, f"{prefix}_epoch_{int(epoch):03d}.png")
    plt.savefig(output_path, dpi=int(getattr(Config, "VIS_DPI", 130)))
    plt.close(fig)
    student.train(was_student_wrapper_training)
    detector.train(was_detector_wrapper_training)
    student_module.train(was_student_training)
    detector_module.train(was_detector_training)
    if teacher is not None:
        teacher.train(was_teacher_training)
    log_to_file(Config, f"Saved compact visualization: {output_path}")
