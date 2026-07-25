"""Compact detection training utilities.

Contains init_compact_log_file(), build_optimizer(),
load_teacher_feature_checkpoint(), save_checkpoint().
"""

import os

import torch
import torch.distributed as dist

from models.SLM.utils_slm import extract_state_dict, load_matching_state, split_student_param_groups
from models.runtime import unwrap_module

from .config import ConfigCompactDetect as Config


def init_compact_log_file():
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    if not is_main:
        return
    with open(Config.LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Compact optical detection training log\n")
        f.write("=" * 80 + "\n")
        f.write(f"Training time: {Config.TRAIN_START_TIME}\n")
        f.write("=" * 80 + "\n\n")


def build_optimizer(student, detector):
    groups = []
    if Config.COMPACT_TRAIN_STUDENT:
        slm_params, other_params = split_student_param_groups(student)
        if slm_params:
            groups.append({"params": slm_params, "lr": Config.COMPACT_PHASE_LR, "weight_decay": 0.0})
        if other_params:
            groups.append({"params": other_params, "lr": Config.COMPACT_PHASE_LR, "weight_decay": Config.COMPACT_WEIGHT_DECAY})
    detector_params = [p for p in detector.parameters() if p.requires_grad]
    if detector_params:
        groups.append({"params": detector_params, "lr": Config.COMPACT_DETECTOR_LR, "weight_decay": Config.COMPACT_WEIGHT_DECAY})
    if not groups:
        raise RuntimeError("No trainable parameters found for compact detector route.")
    return torch.optim.AdamW(groups, weight_decay=0.0)


def load_teacher_feature_checkpoint(teacher, checkpoint_path, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "teacher_state_dict" in checkpoint:
        state_dict = extract_state_dict(checkpoint["teacher_state_dict"])
    else:
        state_dict = extract_state_dict(checkpoint)
    loaded, total = load_matching_state(teacher, state_dict, prefixes=("teacher.",))
    if loaded == 0:
        raise RuntimeError(f"No compatible teacher weights found in: {checkpoint_path}")
    return {
        "teacher_loaded": loaded,
        "teacher_total": total,
        "path": checkpoint_path,
        "teacher_arch": checkpoint.get("teacher_arch") if isinstance(checkpoint, dict) else None,
    }


def save_checkpoint(path, student, detector, epoch, loss_value, metrics=None):
    student = unwrap_module(student)
    detector = unwrap_module(detector)
    payload = {
        "student_state_dict": student.state_dict(),
        "detector_state_dict": detector.state_dict(),
        "epoch": int(epoch),
        "loss": float(loss_value),
        "metrics": metrics or {},
        "model_type": "compact_center_detector",
        "student_enable_norm": bool(getattr(student, "enable_norm", False)),
    }
    for layer_name in _slm_layer_names(student):
        slm = getattr(student, layer_name)
        payload[f"{layer_name}_wrapped_phase"] = slm.wrapped_phase().detach().cpu()
        payload[f"{layer_name}_effective_phase"] = slm.effective_phase().detach().cpu()
        payload[f"{layer_name}_gray_drive"] = slm.phase_to_gray_uint8().cpu()
    torch.save(payload, path)


def module_param_count(module):
    return sum(p.numel() for p in module.parameters())


def _slm_layer_names(student):
    """Return list of SLM layer names, preferring all_slm_layers()."""
    if hasattr(student, "all_slm_layers"):
        return sorted(set(name for name, _ in student.all_slm_layers()),
                      key=lambda n: int(n.replace("slm", "").split("_")[0]) if n.startswith("slm") else 0)
    # Fallback: detect from attributes
    names = []
    for attr in dir(student):
        if attr.startswith("slm") and attr[3:].isdigit():
            names.append(attr)
    if names:
        return sorted(names, key=lambda n: int(n[3:]))
    return ["slm1", "slm2"]
