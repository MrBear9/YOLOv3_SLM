"""Training monitoring utilities: gradient/parameter tracking and confusion matrix.

Provides three monitoring capabilities for TensorBoard:
1. Gradient health monitoring — detect dead/dying parameters
2. Parameter change tracking — detect stale/frozen parameters
3. Detection confusion matrix — TP/FP/FN per class visualization
"""

import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch

from models.class_display import class_name_for_id
from models.training_utils import add_tensorboard_scalar


# ═══════════════════════════════════════════════════════════════════════════
# Gradient Monitoring
# ═══════════════════════════════════════════════════════════════════════════


def collect_gradient_stats(model):
    """Collect per-layer gradient statistics.

    Returns a dict mapping layer_name -> {
        "grad_mean": float, "grad_std": float,
        "grad_max": float, "grad_min": float,
        "grad_norm": float, "grad_numel": int,
        "grad_zero_ratio": float,  # ratio of near-zero gradients
    }
    """
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            stats[name] = {
                "grad_mean": 0.0,
                "grad_std": 0.0,
                "grad_max": 0.0,
                "grad_min": 0.0,
                "grad_norm": 0.0,
                "grad_numel": param.numel(),
                "grad_zero_ratio": 1.0,
            }
            continue
        grad = param.grad.detach().float()
        grad_abs = grad.abs()
        near_zero = (grad_abs < 1e-8).float().mean().item()
        stats[name] = {
            "grad_mean": float(grad.mean().item()),
            "grad_std": float(grad.std(correction=0).item()),
            "grad_max": float(grad.max().item()),
            "grad_min": float(grad.min().item()),
            "grad_norm": float(torch.norm(grad).item()),
            "grad_numel": param.numel(),
            "grad_zero_ratio": near_zero,
        }
    return stats


def write_gradient_monitoring(writer, model, step, prefix="Grad"):
    """Write gradient statistics to TensorBoard.

    Logs:
    - Per-layer gradient mean/std/norm as scalars
    - Per-layer gradient zero-ratio (dead gradient detector)
    - Histogram of gradient norms across layers
    - Global summary: total dead layers, avg gradient norm
    """
    if writer is None:
        return
    stats = collect_gradient_stats(model)
    if not stats:
        return

    grad_norms = []
    dead_layers = []
    zero_ratios = []

    for name, s in stats.items():
        short = name.replace("module.", "")
        add_tensorboard_scalar(writer, f"{prefix}/mean/{short}", s["grad_mean"], step)
        add_tensorboard_scalar(writer, f"{prefix}/std/{short}", s["grad_std"], step)
        add_tensorboard_scalar(writer, f"{prefix}/norm/{short}", s["grad_norm"], step)
        add_tensorboard_scalar(writer, f"{prefix}/zero_ratio/{short}", s["grad_zero_ratio"], step)

        grad_norms.append(s["grad_norm"])
        zero_ratios.append(s["grad_zero_ratio"])
        # A constant phase offset is unobservable after intensity detection.
        is_global_phase_bias = short.endswith("phase_field.mlp_field.net.2.bias")
        if s["grad_zero_ratio"] > 0.99 and not is_global_phase_bias:
            dead_layers.append(short)

    # Global summaries
    valid_norms = [v for v in grad_norms if np.isfinite(v) and v > 0]
    if valid_norms:
        add_tensorboard_scalar(writer, f"{prefix}/global/avg_norm", float(np.mean(valid_norms)), step)
        add_tensorboard_scalar(writer, f"{prefix}/global/max_norm", float(np.max(valid_norms)), step)
        add_tensorboard_scalar(writer, f"{prefix}/global/min_norm", float(np.min(valid_norms)), step)
        add_tensorboard_scalar(writer, f"{prefix}/global/dead_layer_count", len(dead_layers), step)
        add_tensorboard_scalar(writer, f"{prefix}/global/avg_zero_ratio", float(np.mean(zero_ratios)), step)

        # Log histogram of gradient norms
        writer.add_histogram(f"{prefix}/norm_distribution", np.array(valid_norms, dtype=np.float64), step)

    # Warn about dead layers
    if dead_layers and len(dead_layers) <= 20:
        writer.add_text(
            f"{prefix}/dead_layers_warning",
            f"Epoch {step}: {len(dead_layers)} layers with >99% zero gradients:\n"
            + "\n".join(f"  - {n}" for n in dead_layers[:20]),
            step,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Parameter Monitoring
# ═══════════════════════════════════════════════════════════════════════════


# Global storage for parameter snapshots between epochs
_param_snapshots = {}


def _snapshot_key(model_id):
    return id(model_id)


def snapshot_parameters(model):
    """Save a snapshot of current parameters for change detection."""
    global _param_snapshots
    key = _snapshot_key(model)
    _param_snapshots[key] = {
        name: param.detach().float().cpu().clone()
        for name, param in model.named_parameters()
    }


def collect_parameter_stats(model, track_changes=True):
    """Collect per-layer parameter statistics.

    Returns dict mapping layer_name -> {
        "param_mean": float, "param_std": float,
        "param_max": float, "param_min": float,
        "param_norm": float, "param_numel": int,
        "param_near_zero_ratio": float,
        "change_norm": float (if track_changes),
        "change_relative": float (if track_changes),
    }
    """
    global _param_snapshots
    key = _snapshot_key(model)
    prev = _param_snapshots.get(key, {})

    stats = {}
    for name, param in model.named_parameters():
        data = param.detach().float()
        near_zero = (data.abs() < 1e-8).float().mean().item()
        s = {
            "param_mean": float(data.mean().item()),
            "param_std": float(data.std(correction=0).item()),
            "param_max": float(data.max().item()),
            "param_min": float(data.min().item()),
            "param_norm": float(torch.norm(data).item()),
            "param_numel": param.numel(),
            "param_near_zero_ratio": near_zero,
        }
        if track_changes and name in prev:
            prev_data = prev[name].to(device=data.device)
            diff = data - prev_data
            s["change_norm"] = float(torch.norm(diff).item())
            param_norm = float(torch.norm(prev_data).item())
            s["change_relative"] = s["change_norm"] / (param_norm + 1e-12)
        stats[name] = s

    return stats


def write_parameter_monitoring(writer, model, step, prefix="Param"):
    """Write parameter statistics to TensorBoard.

    Logs:
    - Per-layer parameter mean/std/norm
    - Per-layer parameter change norm and relative change
    - Stale parameter detector (layers with near-zero change)
    - Near-zero parameter ratio (potential dead parameters)
    """
    if writer is None:
        return
    stats = collect_parameter_stats(model, track_changes=True)
    if not stats:
        return

    change_norms = []
    stale_layers = []
    near_zero_ratios = []

    for name, s in stats.items():
        short = name.replace("module.", "")
        add_tensorboard_scalar(writer, f"{prefix}/mean/{short}", s["param_mean"], step)
        add_tensorboard_scalar(writer, f"{prefix}/std/{short}", s["param_std"], step)
        add_tensorboard_scalar(writer, f"{prefix}/norm/{short}", s["param_norm"], step)
        add_tensorboard_scalar(writer, f"{prefix}/near_zero_ratio/{short}", s["param_near_zero_ratio"], step)

        if "change_norm" in s:
            add_tensorboard_scalar(writer, f"{prefix}/change_norm/{short}", s["change_norm"], step)
            add_tensorboard_scalar(writer, f"{prefix}/change_relative/{short}", s["change_relative"], step)
            change_norms.append(s["change_norm"])
            if s["change_relative"] < 1e-7:
                stale_layers.append(short)

        near_zero_ratios.append(s["param_near_zero_ratio"])

    # Global summaries
    if change_norms:
        add_tensorboard_scalar(writer, f"{prefix}/global/avg_change_norm", float(np.mean(change_norms)), step)
        add_tensorboard_scalar(writer, f"{prefix}/global/max_change_norm", float(np.max(change_norms)), step)
        add_tensorboard_scalar(writer, f"{prefix}/global/stale_layer_count", len(stale_layers), step)

    if near_zero_ratios:
        add_tensorboard_scalar(writer, f"{prefix}/global/avg_near_zero_ratio", float(np.mean(near_zero_ratios)), step)

    # Warn about stale layers
    if stale_layers and len(stale_layers) <= 20:
        writer.add_text(
            f"{prefix}/stale_layers_warning",
            f"Epoch {step}: {len(stale_layers)} layers with relative change < 1e-7 (stale/坏死):\n"
            + "\n".join(f"  - {n}" for n in stale_layers[:20]),
            step,
        )

    # Update snapshot for next comparison
    snapshot_parameters(model)


# ═══════════════════════════════════════════════════════════════════════════
# Detection Confusion Matrix
# ═══════════════════════════════════════════════════════════════════════════


def compute_detection_confusion_matrix(
    detections_list,
    targets_list,
    num_classes,
    iou_threshold=0.5,
    conf_threshold=0.25,
    image_size=None,
):
    """Compute a confusion matrix for object detection.

    Returns a (num_classes+1, num_classes+1) numpy array where:
    - Rows = Ground Truth (class 0..N-1, last=row is "background/no-gt")
    - Cols = Predicted (class 0..N-1, last=col is "background/no-detection")
    - Entry [i,j] = number of GT objects of class i predicted as class j

    Also returns per-class TP/FP/FN counts.
    """
    n = num_classes + 1  # +1 for background
    confusion = np.zeros((n, n), dtype=np.float64)
    class_tp = np.zeros(num_classes, dtype=np.float64)
    class_fp = np.zeros(num_classes, dtype=np.float64)
    class_fn = np.zeros(num_classes, dtype=np.float64)

    for detections, targets in zip(detections_list, targets_list):
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        gt_boxes = []
        gt_classes = []
        for gt in targets:
            if len(gt) < 5 or gt[3] <= 0 or gt[4] <= 0:
                continue
            # ``targets_list`` is reused afterwards by the review exporter.
            # Keep its normalized coordinates intact while this matrix uses
            # a pixel-coordinate working copy.
            box = np.array(gt[1:5], dtype=np.float32, copy=True)
            if image_size is not None:
                if isinstance(image_size, (tuple, list)):
                    image_h, image_w = image_size
                    box *= np.asarray([image_w, image_h, image_w, image_h], dtype=np.float32)
                else:
                    box *= float(image_size)
            gt_boxes.append(box)
            gt_classes.append(int(gt[0]))

        detections = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        detections = detections[detections[:, 4] >= conf_threshold]
        matched_dets = set()
        matched_gts = set()

        if len(detections) and gt_boxes:
            from models.geometry import bbox_iou_matrix_xywh

            det_tensor = torch.from_numpy(detections[:, :4])
            gt_tensor = torch.from_numpy(np.asarray(gt_boxes, dtype=np.float32))
            iou_matrix = bbox_iou_matrix_xywh(det_tensor, gt_tensor)
            candidates = torch.nonzero(iou_matrix >= iou_threshold, as_tuple=False)
            matches = sorted(
                ((float(iou_matrix[d, g]), int(d), int(g)) for d, g in candidates.tolist()),
                reverse=True,
            )
            for _, det_idx, gt_idx in matches:
                if det_idx in matched_dets or gt_idx in matched_gts:
                    continue
                matched_dets.add(det_idx)
                matched_gts.add(gt_idx)
                gt_cls = gt_classes[gt_idx]
                det_cls = int(detections[det_idx, 5])
                confusion[gt_cls, det_cls] += 1
                if gt_cls == det_cls:
                    class_tp[gt_cls] += 1
                else:
                    class_fn[gt_cls] += 1
                    class_fp[det_cls] += 1

        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets:
                det_cls = int(det[5])
                confusion[num_classes, det_cls] += 1
                class_fp[det_cls] += 1
        for gt_idx, gt_cls in enumerate(gt_classes):
            if gt_idx not in matched_gts:
                confusion[gt_cls, num_classes] += 1
                class_fn[gt_cls] += 1

    return confusion, class_tp, class_fp, class_fn


def _resolve_confusion_matrix_font():
    """Prefer Arial while remaining portable on headless Linux servers."""
    for family in ("Arial", "Liberation Sans", "DejaVu Sans"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Sans"


_CONFUSION_MATRIX_FONT = _resolve_confusion_matrix_font()
_CONFUSION_MATRIX_RC = {
    "font.family": _CONFUSION_MATRIX_FONT,
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.titlesize": 25,
    "axes.labelsize": 25,
    "xtick.labelsize": 23,
    "ytick.labelsize": 23,
}


def _publication_class_label(name):
    """Format one display label without changing dataset/checkpoint metadata."""
    label = str(name).replace("_", " ").strip()
    return label[:1].upper() + label[1:] if label else label


def _matrix_labels(class_names, num_classes, include_background):
    labels = [
        _publication_class_label(class_name_for_id(class_names, index))
        for index in range(num_classes)
    ]
    if include_background:
        labels.append("Background")
    return labels


def _matrix_figure(labels, include_background=False):
    """Create a canvas whose margins and width follow the displayed labels."""
    n = len(labels)
    longest_label = max((len(label) for label in labels), default=1)
    if n <= 6:
        if include_background:
            # The extra Background label is much wider than a class name. Use
            # that otherwise-empty horizontal space for wider matrix columns.
            figure_size = (10.4, 7.2)
            margins = dict(left=0.24, right=0.98, bottom=0.20, top=0.88)
            tick_size = 20
        else:
            figure_size = (7.6, 6.8)
            margins = dict(left=0.22, right=0.97, bottom=0.18, top=0.88)
            tick_size = 23
    else:
        height = max(8.5, n * 0.82)
        width = max(
            height,
            n * (1.05 if include_background else 0.95) + longest_label * 0.14,
        )
        figure_size = (width, height)
        margins = dict(
            left=0.15 if include_background else 0.17,
            right=0.98,
            bottom=0.18,
            top=0.90,
        )
        tick_size = 16 if n <= 10 else 13
    with plt.rc_context(_CONFUSION_MATRIX_RC):
        fig, ax = plt.subplots(figsize=figure_size, dpi=200)
    fig.subplots_adjust(**margins)
    ax.tick_params(axis="both", which="major", length=0, pad=8, labelsize=tick_size)
    return fig, ax


def _style_matrix_axes(ax, labels, title, square_cells=True):
    n = len(labels)
    title_size = 21 if n <= 6 else 26
    axis_size = 25 if n <= 6 else 24
    tick_size = (
        (23 if square_cells else 20)
        if n <= 6
        else (16 if n <= 10 else 13)
    )
    ticks = np.arange(n)
    ax.set_xticks(ticks, labels, rotation=0, ha="center")
    ax.set_yticks(ticks, labels, rotation=0, va="center")
    ax.set_xlabel(
        "Predicted class", labelpad=11, fontsize=axis_size,
        fontweight="medium", fontfamily=_CONFUSION_MATRIX_FONT,
    )
    ax.set_ylabel(
        "True class", labelpad=11, fontsize=axis_size,
        fontweight="medium", fontfamily=_CONFUSION_MATRIX_FONT,
    )
    ax.set_title(
        title, pad=14, fontsize=title_size, fontweight="semibold",
        fontfamily=_CONFUSION_MATRIX_FONT,
    )
    ax.set_aspect("equal" if square_cells else "auto")
    ax.set_xticks(np.arange(-0.5, n, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for tick_label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick_label.set_fontfamily(_CONFUSION_MATRIX_FONT)
        tick_label.set_fontsize(tick_size)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(1.2)


def _figure_to_rgb(fig):
    """Convert a Matplotlib figure to an RGB array for PNG/TensorBoard output."""
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
        height, width, 4
    )[:, :, :3]
    plt.close(fig)
    return image


def _render_confusion_matrix_image(confusion, class_names, title="Detection confusion matrix"):
    """Render a publication-style count matrix including background errors."""
    n = confusion.shape[0]
    num_classes = n - 1
    labels = _matrix_labels(class_names, num_classes, include_background=True)

    fig, ax = _matrix_figure(labels, include_background=True)
    maximum = float(np.max(confusion)) if confusion.size else 0.0
    ax.imshow(
        confusion,
        interpolation="nearest",
        cmap="Blues",
        vmin=0.0,
        vmax=maximum if maximum > 0.0 else 1.0,
    )
    _style_matrix_axes(ax, labels, title, square_cells=False)

    threshold = maximum * 0.50
    for i in range(n):
        for j in range(n):
            val = confusion[i, j]
            if val > 0:
                ax.text(
                    j, i, f"{int(val)}",
                    ha="center", va="center",
                    color="white" if val > threshold else "#202020",
                    fontsize=29 if n <= 6 else (16 if n <= 10 else 13),
                    fontfamily=_CONFUSION_MATRIX_FONT,
                )

    return _figure_to_rgb(fig)


def _render_normalized_confusion_matrix_image(
    confusion, class_names, title="Foreground class consistency (%)"
):
    """Render a publication-style foreground GT-row percentage matrix."""
    num_classes = confusion.shape[0] - 1
    # Exclude the background row/column from both the display and normalization.
    foreground_confusion = confusion[:num_classes, :num_classes]
    labels = _matrix_labels(class_names, num_classes, include_background=False)
    row_totals = foreground_confusion.sum(axis=1, keepdims=True)
    percentages = np.divide(
        foreground_confusion * 100.0,
        row_totals,
        out=np.zeros_like(foreground_confusion),
        where=row_totals > 0,
    )
    n = num_classes

    fig, ax = _matrix_figure(labels, include_background=False)
    ax.imshow(
        percentages, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=100.0
    )
    _style_matrix_axes(ax, labels, title)

    for i in range(n):
        for j in range(n):
            value = percentages[i, j]
            ax.text(
                j, i, f"{value:.0f}",
                ha="center", va="center",
                color="white" if value >= 50.0 else "#202020",
                fontsize=29 if n <= 6 else (16 if n <= 10 else 13),
                fontfamily=_CONFUSION_MATRIX_FONT,
            )

    return _figure_to_rgb(fig)


def write_confusion_matrix(
    writer,
    detections_list,
    targets_list,
    num_classes,
    class_names,
    step,
    iou_threshold=0.5,
    conf_threshold=0.25,
    prefix="ConfusionMatrix",
    image_size=None,
):
    """Compute and write detection confusion matrix to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        detections_list: list of detection arrays per image
        targets_list: list of GT tensors per image
        num_classes: number of classes
        class_names: dict mapping class_id -> name
        step: global step/epoch
        iou_threshold: IoU threshold for matching
        conf_threshold: confidence threshold for filtering detections
        prefix: TensorBoard tag prefix
    """
    if writer is None:
        return

    confusion, class_tp, class_fp, class_fn = compute_detection_confusion_matrix(
        detections_list, targets_list, num_classes, iou_threshold, conf_threshold, image_size
    )

    # Write confusion matrix image
    img = _render_confusion_matrix_image(
        confusion, class_names,
        title="Detection confusion matrix"
    )
    # TensorBoard add_image expects (C, H, W) or (H, W, C) with dataformats
    writer.add_image(prefix, img, step, dataformats="HWC")
    normalized_img = _render_normalized_confusion_matrix_image(
        confusion, class_names,
        title="Foreground class consistency (%)"
    )
    writer.add_image(f"{prefix}/NormalizedPercent", normalized_img, step, dataformats="HWC")

    # Write per-class TP/FP/FN as scalars
    for cls_id in range(num_classes):
        cls_name = class_name_for_id(class_names, cls_id)
        add_tensorboard_scalar(writer, f"{prefix}/TP/{cls_name}", float(class_tp[cls_id]), step)
        add_tensorboard_scalar(writer, f"{prefix}/FP/{cls_name}", float(class_fp[cls_id]), step)
        add_tensorboard_scalar(writer, f"{prefix}/FN/{cls_name}", float(class_fn[cls_id]), step)
        # Per-class precision/recall
        tp = class_tp[cls_id]
        fp = class_fp[cls_id]
        fn = class_fn[cls_id]
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        add_tensorboard_scalar(writer, f"{prefix}/Precision/{cls_name}", float(prec), step)
        add_tensorboard_scalar(writer, f"{prefix}/Recall/{cls_name}", float(rec), step)

    # Global totals
    total_tp = float(class_tp.sum())
    total_fp = float(class_fp.sum())
    total_fn = float(class_fn.sum())
    add_tensorboard_scalar(writer, f"{prefix}/Total/TP", total_tp, step)
    add_tensorboard_scalar(writer, f"{prefix}/Total/FP", total_fp, step)
    add_tensorboard_scalar(writer, f"{prefix}/Total/FN", total_fn, step)
    add_tensorboard_scalar(writer, f"{prefix}/Total/Precision", total_tp / (total_tp + total_fp + 1e-6), step)
    add_tensorboard_scalar(writer, f"{prefix}/Total/Recall", total_tp / (total_tp + total_fn + 1e-6), step)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: combined monitoring call
# ═══════════════════════════════════════════════════════════════════════════


def write_full_monitoring(
    writer,
    model,
    step,
    detections_list=None,
    targets_list=None,
    num_classes=None,
    class_names=None,
    grad_prefix="Grad",
    param_prefix="Param",
    cm_prefix="ConfusionMatrix",
    iou_threshold=0.5,
    conf_threshold=0.25,
):
    """Write all monitoring data to TensorBoard in one call.

    Combines gradient monitoring, parameter monitoring, and confusion matrix.
    """
    write_gradient_monitoring(writer, model, step, prefix=grad_prefix)
    write_parameter_monitoring(writer, model, step, prefix=param_prefix)
    if detections_list is not None and targets_list is not None and num_classes is not None:
        write_confusion_matrix(
            writer, detections_list, targets_list, num_classes,
            class_names or {}, step,
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold,
            prefix=cm_prefix,
        )
