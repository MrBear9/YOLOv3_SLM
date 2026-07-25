"""SLM student training helper utilities.

Contains configuration, scheduling, visualization, phase gradient,
checkpoint management, and config logging helpers split out from the
original train_helpers_slm.py so each module stays focused and small.
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.SLM.config_slm import ConfigSLM as Config
from models.SLM.utils_slm import save_student_best, _layer_names_from_student
from models.runtime import log_to_file
from models.training_utils import add_tensorboard_scalar


# ═══════════════════════════════════════════════════════════════════════════
# Configuration & scheduling
# ═══════════════════════════════════════════════════════════════════════════


def configure_backends():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = Config.ENABLE_CUDNN_BENCHMARK
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = Config.ENABLE_TF32
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = Config.ENABLE_TF32


def stage_schedule():
    return [
        ("phase_focus", Config.PHASE_FOCUS_EPOCHS),
        ("detector_focus", Config.DETECTOR_FOCUS_EPOCHS),
        ("joint_fit", Config.JOINT_FIT_EPOCHS),
        ("norm_joint", Config.NORM_JOINT_EPOCHS),
    ]


def prepare_batch(batch, device):
    gray = batch["gray_tensor"].to(device, non_blocking=Config.PIN_MEMORY)
    rgb = batch["rgb_tensor"].to(device, non_blocking=Config.PIN_MEMORY)
    if Config.ENABLE_CHANNELS_LAST and torch.cuda.is_available():
        gray = gray.contiguous(memory_format=torch.channels_last)
        rgb = rgb.contiguous(memory_format=torch.channels_last)
    return gray, rgb, batch["targets"]


def configure_student_norm_for_stage(stage_name, deployment_norm_mode):
    schedule = str(getattr(Config, "STUDENT_NORM_SCHEDULE", "norm_joint_only")).lower()
    if not getattr(Config, "ENABLE_STUDENT_NORM", True) or schedule in {"none", "off", "false"}:
        stage_norm_mode = "none"
    elif schedule in {"joint_and_norm", "joint_norm"}:
        stage_norm_mode = deployment_norm_mode if stage_name in {"joint_fit", "norm_joint"} else "none"
    elif schedule in {"norm_joint_only", "norm_joint"}:
        stage_norm_mode = deployment_norm_mode if stage_name == "norm_joint" else "none"
    else:
        stage_norm_mode = deployment_norm_mode
    Config.STUDENT_NORM_MODE = stage_norm_mode
    return stage_norm_mode


def build_stage_scheduler(optimizer, stage_epochs):
    if str(getattr(Config, "LR_SCHEDULER", "")).lower() != "cosineannealinglr":
        return None
    return CosineAnnealingLR(optimizer, T_max=max(int(stage_epochs), 1), eta_min=Config.ETA_MIN)


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════


def valid_history_points(values):
    xs, ys = [], []
    for idx, value in enumerate(values):
        if value is None:
            continue
        try:
            if np.isnan(value):
                continue
        except TypeError:
            pass
        xs.append(idx + 1)
        ys.append(value)
    return xs, ys


def save_slm_component_curves(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    axes = axes.ravel()

    for key in ("train_feature", "val_feature"):
        xs, ys = valid_history_points(history.get(key, []))
        axes[0].plot(xs, ys, label=key)
    axes[0].set_title("Feature loss")
    axes[0].legend()

    for key in ("train_detection", "val_detection"):
        xs, ys = valid_history_points(history.get(key, []))
        axes[1].plot(xs, ys, label=key)
    axes[1].set_title("Detection loss")
    axes[1].legend()

    for key in ("train_response", "val_response"):
        xs, ys = valid_history_points(history.get(key, []))
        axes[2].plot(xs, ys, label=key)
    axes[2].set_title("Response distillation loss")
    axes[2].legend()

    for key in ("train_privacy", "val_privacy"):
        xs, ys = valid_history_points(history.get(key, []))
        axes[3].plot(xs, ys, label=key)
    axes[3].set_title("Privacy obfuscation loss")
    axes[3].legend()

    for key in ("train_total", "val_total"):
        xs, ys = valid_history_points(history.get(key, []))
        axes[4].plot(xs, ys, label=key)
    axes[4].set_title("Total loss")
    axes[4].legend()

    axes[5].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves_components.png"), dpi=130)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# TensorBoard
# ═══════════════════════════════════════════════════════════════════════════


def write_slm_tensorboard_scalars(
    writer,
    step,
    stage_name,
    train_losses,
    val_losses=None,
    val_metrics=None,
    slm_stats=None,
    phase_grad_norm=None,
    phase_update_norm=None,
    phase_update_rel=None,
    phase_layer_stats=None,
    lr=None,
):
    if writer is None:
        return
    for key, value in train_losses.items():
        add_tensorboard_scalar(writer, f"Loss/{stage_name}/train_{key}", value, step)
        add_tensorboard_scalar(writer, f"Loss/All_Stages/train_{key}", value, step)
    if val_losses is not None:
        for key, value in val_losses.items():
            add_tensorboard_scalar(writer, f"Loss/{stage_name}/val_{key}", value, step)
            add_tensorboard_scalar(writer, f"Loss/All_Stages/val_{key}", value, step)
    if val_metrics is not None:
        for key in ("precision", "recall", "f1", "map50", "precision_op", "recall_op", "f1_op"):
            add_tensorboard_scalar(writer, f"Metrics/{stage_name}/{key}", val_metrics.get(key), step)
            add_tensorboard_scalar(writer, f"Metrics/All_Stages/{key}", val_metrics.get(key), step)
        op_conf = float(getattr(Config, "CONF_THRESH", 0.35))
        threshold_tag = f"conf_{op_conf:g}"
        for key in ("precision_op", "recall_op", "f1_op"):
            add_tensorboard_scalar(
                writer, f"MetricsOperating/{threshold_tag}/{stage_name}/{key}", val_metrics.get(key), step
            )
            add_tensorboard_scalar(
                writer, f"MetricsOperating/{threshold_tag}/All_Stages/{key}", val_metrics.get(key), step
            )
    if slm_stats is not None:
        for key, value in slm_stats.items():
            if any(key.endswith(suffix) for suffix in (
                "_wrapped_std", "_wrapped_span", "_circular_std", "_near_boundary_ratio",
            )):
                add_tensorboard_scalar(writer, f"SLM/{stage_name}/{key}", value, step)
    add_tensorboard_scalar(writer, f"Grad/{stage_name}/phase_grad_norm", phase_grad_norm, step)
    add_tensorboard_scalar(writer, f"Grad/{stage_name}/phase_update_norm", phase_update_norm, step)
    add_tensorboard_scalar(writer, f"Grad/{stage_name}/phase_update_rel", phase_update_rel, step)
    if phase_layer_stats is not None:
        for layer_name, stats in phase_layer_stats.items():
            add_tensorboard_scalar(writer, f"SLM/{stage_name}/{layer_name}_phase_grad_norm", stats["grad_norm"], step)
            add_tensorboard_scalar(writer, f"SLM/{stage_name}/{layer_name}_phase_update_norm", stats["update_norm"], step)
            add_tensorboard_scalar(writer, f"SLM/{stage_name}/{layer_name}_phase_update_rel", stats["update_rel"], step)
    add_tensorboard_scalar(writer, f"LR/{stage_name}", lr, step)


# ═══════════════════════════════════════════════════════════════════════════
# Phase gradient utilities
# ═══════════════════════════════════════════════════════════════════════════


def is_phase_parameter(name):
    return any(token in name for token in ("phase_raw", "amp_raw", "phase_field", "scale_params", "mlp_field"))


def _init_layer_dict(student, value=0.0):
    """Create a dict {layer_name: value} for all SLM layers."""
    return {k: value for k in _layer_names_from_student(student)}


def collect_phase_snapshot(student):
    return {
        name: param.detach().float().clone()
        for name, param in student.named_parameters()
        if is_phase_parameter(name)
    }


def collect_phase_grad_norms(student):
    # Dynamically detect layer names from student or parameter names
    totals = _init_layer_dict(student, 0.0)
    for name, param in student.named_parameters():
        if not is_phase_parameter(name) or param.grad is None:
            continue
        for layer_name in totals:
            if name.startswith(f"{layer_name}.") or name.startswith(f"{layer_name}_"):
                totals[layer_name] += float(torch.sum(param.grad.detach().float().square()).item())
                break
    return {k: v ** 0.5 for k, v in totals.items()}


def collect_phase_grad_norm(student):
    return sum(value * value for value in collect_phase_grad_norms(student).values()) ** 0.5


def clip_phase_grad_norm(student, max_norm):
    max_norm = float(max_norm)
    if max_norm <= 0:
        return None
    params = [
        param
        for name, param in student.named_parameters()
        if is_phase_parameter(name) and param.requires_grad and param.grad is not None
    ]
    if not params:
        return None
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm).detach().item())


def collect_phase_update_norms(student, snapshot):
    layer_names = _layer_names_from_student(student)
    update_sq = {k: 0.0 for k in layer_names}
    param_sq = {k: 0.0 for k in layer_names}
    for name, param in student.named_parameters():
        if name not in snapshot:
            continue
        layer_name = next((layer for layer in layer_names if name.startswith(f"{layer}.") or name.startswith(f"{layer}_")), None)
        if layer_name is None:
            continue
        current = param.detach().float()
        before = snapshot[name].to(device=current.device)
        diff = current - before
        update_sq[layer_name] += float(torch.sum(diff * diff).item())
        param_sq[layer_name] += float(torch.sum(before * before).item())
    return {
        k: {
            "update_norm": update_sq[k] ** 0.5,
            "update_rel": (update_sq[k] ** 0.5) / ((param_sq[k] ** 0.5) + 1e-12),
        }
        for k in update_sq
    }


def collect_phase_update_norm(student, snapshot):
    layer_stats = collect_phase_update_norms(student, snapshot)
    update_norm = sum(stats["update_norm"] ** 2 for stats in layer_stats.values()) ** 0.5
    param_relative = sum(stats["update_rel"] ** 2 for stats in layer_stats.values()) ** 0.5
    return update_norm, param_relative


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint management
# ═══════════════════════════════════════════════════════════════════════════


def save_current_student_checkpoint(
    student,
    path,
    display_epoch,
    avg_total,
    stage_name,
    slm_stats,
    slm_ok,
    Config,
    val_losses=None,
    val_metrics=None,
    phase_grad_norm=None,
    phase_update_norm=None,
    phase_update_rel=None,
    mirror_path=None,
):
    extra = {
        "train_loss": avg_total,
        "val_loss": val_losses["total"] if val_losses is not None else None,
        "val_map50": val_metrics["map50"] if val_metrics is not None else None,
        "slm_stats": slm_stats,
        "paired_with_detector_best": False,
        "paired_detector_checkpoint": None,
        "paired_student_epoch": display_epoch,
        "paired_student_stage": stage_name,
        "selection_metric": "current_optical_student",
        "recommended_inference_checkpoint": False,
        "student_norm_mode": Config.STUDENT_NORM_MODE,
        "student_norm_schedule": Config.STUDENT_NORM_SCHEDULE,
        "slm_quality_passed": bool(slm_ok),
        "phase_grad_norm": phase_grad_norm,
        "phase_update_norm": phase_update_norm,
        "phase_update_rel": phase_update_rel,
    }
    save_student_best(Config, student, path, display_epoch, avg_total, extra=extra)
    if mirror_path is not None:
        mirror_extra = dict(extra)
        mirror_extra["selection_metric"] = "phase_focus_current_optical_student"
        mirror_extra["phase_focus_mirror_checkpoint"] = path
        save_student_best(Config, student, mirror_path, display_epoch, avg_total, extra=mirror_extra)


# ═══════════════════════════════════════════════════════════════════════════
# Config logging
# ═══════════════════════════════════════════════════════════════════════════


def log_config():
    """Compact config summary — details go to TensorBoard hparams."""
    log_to_file(Config, "=" * 80)
    log_to_file(Config, "Optical SLM student training — key config")
    log_to_file(Config, "=" * 80)
    log_to_file(Config, f"Dataset: {Config.YAML_PATH}  |  Output: {Config.OUTPUT_DIR}")
    log_to_file(Config, f"Teacher: {Config.TEACHER_ARCH}  |  Detector: {Config.DETECTOR_HEAD_TYPE} (anchor_free_tal)")
    num_layers = int(getattr(Config, "NUM_LAYERS", 2))
    multi = bool(getattr(Config, "SLM_MULTI_HEAD_ENABLED", False))
    log_to_file(Config, f"Optical: {num_layers}-layer, multi_head={multi}, phase_mode={Config.SLM_PHASE_PARAM_MODE}, init={Config.SLM_INIT_MODE}")
    log_to_file(
        Config,
        f"Physics: input={Config.INPUT_INTENSITY_MODE}, phase_levels={Config.SLM_PHASE_LEVELS}, "
        f"quantization={Config.SIMULATE_PHASE_QUANTIZATION}, inverted_gray={Config.SLM_GRAY_INVERTED}, "
        f"lut={Config.SLM_GRAY_TO_PHASE_LUT or 'ideal_linear'}",
    )
    log_to_file(
        Config,
        f"Stages: phase_focus={Config.PHASE_FOCUS_EPOCHS}  detector_focus={Config.DETECTOR_FOCUS_EPOCHS}  "
        f"joint={Config.JOINT_FIT_EPOCHS}  norm_joint={Config.NORM_JOINT_EPOCHS}",
    )
    log_to_file(
        Config,
        f"LR: phase={Config.PHASE_FOCUS_PHASE_PARAM_LR}  detector={Config.DETECTOR_LR}  "
        f"joint_ph={Config.JOINT_PHASE_PARAM_LR}/{Config.JOINT_DETECTOR_LR}  "
        f"norm_ph={Config.NORM_JOINT_PHASE_PARAM_LR}/{Config.NORM_JOINT_DETECTOR_LR}",
    )
    log_to_file(
        Config,
        f"Norm: {Config.STUDENT_NORM_MODE} p={Config.STUDENT_NORM_PERCENTILE} clamp={Config.STUDENT_OUTPUT_CLAMP_MAX}  "
        f"Align: {Config.FEATURE_DOMAIN_ALIGN_MODE}  "
        f"Loss: f={Config.FEATURE_LOSS_WEIGHT_PHASE_FOCUS}/{Config.DETECTION_LOSS_WEIGHT_PHASE_FOCUS}  "
        f"d={Config.FEATURE_LOSS_WEIGHT_DETECTOR_FOCUS}/{Config.DETECTION_LOSS_WEIGHT_DETECTOR_FOCUS}  "
        f"j={Config.FEATURE_LOSS_WEIGHT_JOINT}/{Config.DETECTION_LOSS_WEIGHT_JOINT}  "
        f"nj={Config.FEATURE_LOSS_WEIGHT_NORM_JOINT}/{Config.DETECTION_LOSS_WEIGHT_NORM_JOINT}",
    )
    log_to_file(Config, f"Phase reg: smooth/diversity by stage — see config for details")
