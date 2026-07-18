"""SLM student training helper utilities.

Contains configuration, scheduling, visualization, phase gradient,
checkpoint management, and config logging helpers split out from the
original train_helpers_slm.py so each module stays focused and small.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.SLM.config_slm import ConfigSLM as Config
from models.SLM.utils_slm import save_student_best
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
        for key in (
            "slm1_wrapped_std",
            "slm2_wrapped_std",
            "slm1_circular_std",
            "slm2_circular_std",
            "slm1_near_boundary_ratio",
            "slm2_near_boundary_ratio",
            "slm1_wrapped_span",
            "slm2_wrapped_span",
        ):
            add_tensorboard_scalar(writer, f"SLM/{stage_name}/{key}", slm_stats.get(key), step)
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


def collect_phase_snapshot(student):
    return {
        name: param.detach().float().clone()
        for name, param in student.named_parameters()
        if is_phase_parameter(name)
    }


def collect_phase_grad_norms(student):
    totals = {"slm1": 0.0, "slm2": 0.0}
    for name, param in student.named_parameters():
        if not is_phase_parameter(name) or param.grad is None:
            continue
        for layer_name in totals:
            if name.startswith(f"{layer_name}."):
                totals[layer_name] += float(torch.sum(param.grad.detach().float().square()).item())
                break
    return {layer_name: total_sq ** 0.5 for layer_name, total_sq in totals.items()}


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
    update_sq = {"slm1": 0.0, "slm2": 0.0}
    param_sq = {"slm1": 0.0, "slm2": 0.0}
    for name, param in student.named_parameters():
        if name not in snapshot:
            continue
        layer_name = next((layer for layer in update_sq if name.startswith(f"{layer}.")), None)
        if layer_name is None:
            continue
        current = param.detach().float()
        before = snapshot[name].to(device=current.device)
        diff = current - before
        update_sq[layer_name] += float(torch.sum(diff * diff).item())
        param_sq[layer_name] += float(torch.sum(before * before).item())
    return {
        layer_name: {
            "update_norm": update_sq[layer_name] ** 0.5,
            "update_rel": (update_sq[layer_name] ** 0.5) / ((param_sq[layer_name] ** 0.5) + 1e-12),
        }
        for layer_name in update_sq
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
    log_to_file(Config, "=" * 80)
    log_to_file(Config, "Optical SLM student training with YOLOv8-head teacher")
    log_to_file(Config, "=" * 80)
    log_to_file(Config, f"Dataset: {Config.YAML_PATH}")
    log_to_file(Config, f"Output: {Config.OUTPUT_DIR}")
    log_to_file(Config, f"Teacher detector checkpoint: {Config.TEACHER_DETECTOR_CHECKPOINT}")
    log_to_file(Config, f"Teacher arch: {Config.TEACHER_ARCH}")
    log_to_file(Config, f"Detector head type: {Config.DETECTOR_HEAD_TYPE}")
    log_to_file(
        Config,
        f"Detection protocol: {Config.DETECTION_PROTOCOL}; anchor matching: mode={Config.ANCHOR_MATCH_MODE}, ratio_thresh={Config.ANCHOR_MATCH_RATIO_THRESH}, "
        f"neighbor_cells={Config.ASSIGN_NEIGHBOR_CELLS}, simota_iou={Config.ANCHOR_MATCH_IOU_THRESH}, "
        f"center_radius={Config.CENTER_PRIOR_RADIUS}, top_n={Config.SIMOTA_TOP_N}, max_assign={Config.SIMOTA_MAX_ASSIGN}",
    )
    if Config.DETECTION_PROTOCOL == "anchor_free_tal":
        log_to_file(
            Config,
            f"Anchor-free classification: loss={Config.ANCHOR_FREE_CLS_LOSS}, "
            f"varifocal_alpha={Config.VARIFOCAL_ALPHA}, varifocal_gamma={Config.VARIFOCAL_GAMMA}",
        )
    log_to_file(
        Config,
        f"Metric conf/nms/max_det: {Config.METRIC_CONF_THRESH}/{Config.METRIC_NMS_THRESH}/{Config.METRIC_MAX_DET}",
    )
    log_to_file(Config, f"Classes: {Config.CLASS_NAMES}")
    log_to_file(
        Config,
        f"Epochs phase_focus/detector_focus/joint_fit/norm_joint: "
        f"{Config.PHASE_FOCUS_EPOCHS}/{Config.DETECTOR_FOCUS_EPOCHS}/"
        f"{Config.JOINT_FIT_EPOCHS}/{Config.NORM_JOINT_EPOCHS}",
    )
    log_to_file(Config, f"Save paired detector best: {Config.get_detector_best_path()}")
    log_to_file(Config, f"Save paired student mirror: {Config.get_student_best_path()}")
    log_to_file(Config, f"Save current optical student snapshot: {Config.get_student_current_path()}")
    log_to_file(
        Config,
        "Recommended inference checkpoint: detector_best.pth, because it carries the detector and "
        "the paired student_state_dict from the same best-mAP epoch.",
    )
    log_to_file(
        Config,
        f"LR phase_focus_phase/detector/joint_phase/joint_detector/norm_joint_phase/norm_joint_detector: "
        f"{Config.PHASE_FOCUS_PHASE_PARAM_LR}/{Config.DETECTOR_LR}/"
        f"{Config.JOINT_PHASE_PARAM_LR}/{Config.JOINT_DETECTOR_LR}/"
        f"{Config.NORM_JOINT_PHASE_PARAM_LR}/{Config.NORM_JOINT_DETECTOR_LR}",
    )
    log_to_file(Config, f"Phase gradient clip norm: {Config.PHASE_GRAD_CLIP_NORM}")
    log_to_file(Config, f"LR scheduler: {Config.LR_SCHEDULER}, eta_min={Config.ETA_MIN}")
    log_to_file(Config, f"Validation interval: {Config.VAL_INTERVAL}")
    log_to_file(Config, f"Visualization: split={Config.VIS_DATASET_SPLIT}, interval={Config.VIS_INTERVAL}")
    log_to_file(Config, f"SLM init mode: {Config.SLM_INIT_MODE}")
    log_to_file(Config, f"SLM init checkpoint: {Config.SLM_INIT_CHECKPOINT}")
    log_to_file(
        Config,
        f"Student normalization: enabled={Config.ENABLE_STUDENT_NORM}, schedule={Config.STUDENT_NORM_SCHEDULE}, "
        f"deployment_mode={Config.STUDENT_NORM_MODE}, "
        f"percentile={Config.STUDENT_NORM_PERCENTILE}, clamp_max={Config.STUDENT_OUTPUT_CLAMP_MAX}, "
        f"blur_kernel={Config.STUDENT_OUTPUT_BLUR_KERNEL}",
    )
    log_to_file(
        Config,
        "Feature loss weights full/low1/low2/ssim/grad/freq/pearson: "
        f"{Config.LOSS_FULL_WEIGHT}/{Config.LOSS_LOW1_WEIGHT}/{Config.LOSS_LOW2_WEIGHT}/"
        f"{Config.LOSS_SSIM_WEIGHT}/{Config.LOSS_GRAD_WEIGHT}/{Config.LOSS_FREQ_WEIGHT}/{Config.LOSS_PEARSON_WEIGHT}",
    )
    log_to_file(Config, f"Feature loss prefilter kernel: {Config.FEATURE_LOSS_PREFILTER_KERNEL}")
    log_to_file(
        Config,
        f"Phase regularization by stage: phase_focus={Config.get_phase_regularization_weights('phase_focus')}, "
        f"detector_focus={Config.get_phase_regularization_weights('detector_focus')}, "
        f"joint_fit={Config.get_phase_regularization_weights('joint_fit')}, "
        f"norm_joint={Config.get_phase_regularization_weights('norm_joint')}",
    )
    log_to_file(
        Config,
        f"Feature domain alignment: enabled={Config.ENABLE_FEATURE_DOMAIN_ALIGNMENT}, "
        f"mode={Config.FEATURE_DOMAIN_ALIGN_MODE}",
    )
    log_to_file(
        Config,
        f"Stage loss weights: phase_focus={Config.get_stage_loss_weights('phase_focus')}, "
        f"detector_focus={Config.get_stage_loss_weights('detector_focus')}, "
        f"joint_fit={Config.get_stage_loss_weights('joint_fit')}, "
        f"norm_joint={Config.get_stage_loss_weights('norm_joint')}",
    )
    log_to_file(
        Config,
        f"Privacy targets corr/ssim: {Config.PRIVACY_CORR_TARGET}/{Config.PRIVACY_SSIM_TARGET}",
    )
    log_to_file(
        Config,
        f"Detector-focus early stop: enabled={Config.ENABLE_DETECTOR_FOCUS_EARLY_STOP}, "
        f"patience={Config.DETECTOR_FOCUS_EARLY_STOP_PATIENCE}, "
        f"min_delta={Config.DETECTOR_FOCUS_EARLY_STOP_MIN_DELTA}",
    )
    log_to_file(
        Config,
        "Phase regularization weight/targets diversity/std/span/circular: "
        f"{Config.LOSS_PHASE_DIVERSITY_WEIGHT}/{Config.PHASE_STD_TARGET}/"
        f"{Config.PHASE_SPAN_TARGET}/{Config.PHASE_CIRCULAR_STD_TARGET}",
    )
    log_to_file(Config, "Checkpoint payload intentionally omits a 'phase' key for SLM extraction compatibility.")
