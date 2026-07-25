import os
import re
from collections import defaultdict

import numpy as np
import torch


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("student_state_dict", "teacher_state_dict", "detector_state_dict", "model_state_dict", "state_dict", "model"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def load_matching_state(module, state_dict, prefixes=()):
    target_state = module.state_dict()

    # Collect non-persistent buffer keys across all submodules — these are
    # recomputed in __init__ and must never be loaded from a checkpoint
    # (they may have wrong shapes if the config changed since the checkpoint
    # was saved).
    non_persistent = set()
    for prefix, submodule in module.named_modules():
        if hasattr(submodule, '_non_persistent_buffers_set'):
            for buf_name in submodule._non_persistent_buffers_set:
                full_name = f"{prefix}.{buf_name}" if prefix else buf_name
                non_persistent.add(full_name)

    compatible = {}
    for raw_key, value in state_dict.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        if key in non_persistent:
            continue
        if key in target_state and target_state[key].shape == value.shape:
            compatible[key] = value
    if compatible:
        module.load_state_dict({**target_state, **compatible}, strict=False)
    return len(compatible), len(target_state)


def load_teacher_detector_checkpoint(teacher, detector, checkpoint_path, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Teacher-detector checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    info = {}
    if isinstance(checkpoint, dict) and "teacher_state_dict" in checkpoint and "detector_state_dict" in checkpoint:
        if "teacher_arch" in checkpoint:
            info["teacher_arch"] = checkpoint["teacher_arch"]
        info["teacher_loaded"], info["teacher_total"] = load_matching_state(teacher, extract_state_dict(checkpoint["teacher_state_dict"]), prefixes=("teacher.",))
        info["detector_loaded"], info["detector_total"] = load_matching_state(detector, extract_state_dict(checkpoint["detector_state_dict"]), prefixes=("detector.",))
        return info
    state_dict = extract_state_dict(checkpoint)
    info["teacher_loaded"], info["teacher_total"] = load_matching_state(teacher, state_dict, prefixes=("teacher.",))
    info["detector_loaded"], info["detector_total"] = load_matching_state(detector, state_dict, prefixes=("detector.",))
    if info["teacher_loaded"] == 0 and info["detector_loaded"] == 0:
        raise RuntimeError(f"No compatible teacher/detector weights found in: {checkpoint_path}")
    return info


def load_student_checkpoint(student, checkpoint_path, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {"loaded": 0, "total": len(student.state_dict()), "path": checkpoint_path, "reason": "not_found"}
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    loaded, total = load_matching_state(student, state_dict, prefixes=("student.",))
    if isinstance(checkpoint, dict) and "student_enable_norm" in checkpoint:
        student.enable_norm = bool(checkpoint["student_enable_norm"])
    if isinstance(checkpoint, dict) and "student_norm_mode" in checkpoint:
        student.config.STUDENT_NORM_MODE = checkpoint["student_norm_mode"]
    if isinstance(checkpoint, dict) and "student_norm_schedule" in checkpoint:
        student.config.STUDENT_NORM_SCHEDULE = checkpoint["student_norm_schedule"]
    return {
        "loaded": loaded,
        "total": total,
        "path": checkpoint_path,
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "loss": checkpoint.get("loss") if isinstance(checkpoint, dict) else None,
        "val_map50": checkpoint.get("val_map50") if isinstance(checkpoint, dict) else None,
    }


def _is_slm_phase_param(name):
    """Return True for any SLM phase-related parameter (all parametrisations)."""
    slm_keywords = ("phase_raw", "amp_raw", "phase_field", "scale_params", "mlp_field")
    return any(kw in name for kw in slm_keywords)


def split_student_param_groups(student):
    slm_params = []
    other_params = []
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        if _is_slm_phase_param(name):
            slm_params.append(param)
        else:
            other_params.append(param)
    return slm_params, other_params


def _param_layer_name(name):
    """Return ``"slm{N}"`` or ``None`` from a parameter name.

    Handles both single-head (``slm1.phase_field…``) and multi-head
    (``slm1_heads.0.phase_field…``) naming for arbitrary layer counts.
    """
    parts = name.split(".")
    for part in parts:
        m = re.match(r"slm(\d+)(?:_|$)", part)
        if m:
            return f"slm{m.group(1)}"
    # Fallback: substring check for nested names
    m = re.search(r"\.(slm\d+)\.", name)
    if m:
        return m.group(1)
    return None


def _layer_names_from_student(student):
    """Return sorted list of layer names (``slm1``, ``slm2``, ...) from a student."""
    if hasattr(student, "num_layers"):
        return [f"slm{i}" for i in range(1, student.num_layers + 1)]
    # Fallback: scan attributes for slmN pattern
    names = set()
    for attr in dir(student):
        m = re.match(r"^(slm\d+)$", attr)
        if m:
            names.add(m.group(1))
    if names:
        return sorted(names, key=lambda n: int(n[3:]))
    return ["slm1", "slm2"]


def split_phase_param_groups(student):
    groups = defaultdict(list)
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        if _is_slm_phase_param(name):
            layer = _param_layer_name(name)
            groups[layer or "other"].append(param)
        else:
            groups["other"].append(param)
    return dict(groups)


def build_stage_optimizer(config, student, detector, stage_name):
    if stage_name == "detector_focus":
        return torch.optim.Adam(
            [p for p in detector.parameters() if p.requires_grad],
            lr=config.DETECTOR_LR, weight_decay=config.WEIGHT_DECAY,
        )

    # Resolve per-stage LRs
    if stage_name == "phase_focus":
        base_lr = config.PHASE_FOCUS_PHASE_PARAM_LR
        lr_mult_stage = "phase_focus"
    elif stage_name == "norm_joint":
        base_lr = config.NORM_JOINT_PHASE_PARAM_LR
        lr_mult_stage = "norm_joint"
    else:
        base_lr = config.JOINT_PHASE_PARAM_LR
        lr_mult_stage = "joint"

    phase_groups = split_phase_param_groups(student)
    groups = []

    for key, params in sorted(phase_groups.items()):
        if not params:
            continue
        if key == "other":
            groups.append({"params": params, "lr": base_lr,
                          "weight_decay": config.WEIGHT_DECAY})
        else:
            # Extract layer index from e.g. "slm2" → 2
            layer_idx = int(re.match(r"slm(\d+)", key).group(1))
            mult = _resolve_lr_mult(config, layer_idx, lr_mult_stage)
            groups.append({"params": params, "lr": base_lr * mult,
                          "weight_decay": config.PHASE_WEIGHT_DECAY})

    if stage_name != "phase_focus":
        detector_params = [p for p in detector.parameters() if p.requires_grad]
        detector_lr = (config.NORM_JOINT_DETECTOR_LR if stage_name == "norm_joint"
                       else config.JOINT_DETECTOR_LR)
        if detector_params:
            groups.append({"params": detector_params, "lr": detector_lr,
                          "weight_decay": config.WEIGHT_DECAY})

    return torch.optim.Adam(groups, weight_decay=0.0)


def _resolve_lr_mult(config, layer_idx, stage):
    """Resolve per-layer LR multiplier, preferring OpticalConfig.accessor."""
    if hasattr(config, "layer_lr_mult"):
        return config.layer_lr_mult(layer_idx, stage)
    # Backward compat: old flat attributes for layer 2
    if layer_idx == 2:
        mapping = {
            "phase_focus": "SLM2_PHASE_FOCUS_LR_MULT",
            "joint": "SLM2_JOINT_LR_MULT",
            "norm_joint": "SLM2_NORM_JOINT_LR_MULT",
        }
        return float(getattr(config, mapping.get(stage, "SLM2_JOINT_LR_MULT"), 1.0))
    return 1.0


def set_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable


def collect_slm_statistics(student):
    stats = {}
    if hasattr(student, "all_slm_layers"):
        for layer_name, slm in student.all_slm_layers():
            _collect_one_layer_stats(stats, slm, layer_name, student.config)
    else:
        # Fallback: try slm1, slm2, ...
        for layer_name in _layer_names_from_student(student):
            slm = getattr(student, layer_name, None)
            if slm is not None:
                _collect_one_layer_stats(stats, slm, layer_name, student.config)
    return stats


def _collect_one_layer_stats(stats, slm, layer_name, config):
    wrapped = slm.wrapped_phase().detach().float()
    centered = torch.atan2(torch.sin(wrapped), torch.cos(wrapped))
    wrapped_flat = wrapped.flatten(1)
    cos_mean = torch.mean(torch.cos(wrapped_flat), dim=1)
    sin_mean = torch.mean(torch.sin(wrapped_flat), dim=1)
    resultant = torch.sqrt(cos_mean ** 2 + sin_mean ** 2)
    circular_std = torch.sqrt(torch.clamp(-2.0 * torch.log(torch.clamp(resultant, min=1e-8)), min=0.0)).mean()
    near_boundary = (
        (wrapped_flat < config.PHASE_NEAR_BOUNDARY_EPS)
        | (wrapped_flat > 2 * np.pi - config.PHASE_NEAR_BOUNDARY_EPS)
    ).float().mean(dim=1).mean()
    stats[f"{layer_name}_wrapped_std"] = float(centered.std(unbiased=False).item())
    stats[f"{layer_name}_wrapped_span"] = float((centered.amax() - centered.amin()).item())
    stats[f"{layer_name}_circular_std"] = float(circular_std.item())
    stats[f"{layer_name}_near_boundary_ratio"] = float(near_boundary.item())


def save_student_best(config, student, path, epoch, loss_value, extra=None):
    student_state = student.state_dict()
    payload = {
        "student_state_dict": student_state,
        "epoch": int(epoch),
        "loss": float(loss_value),
        "student_enable_norm": bool(getattr(student, "enable_norm", False)),
        "num_layers": int(getattr(student, "num_layers", 2)),
        "num_heads": int(getattr(student, "num_heads", 1)),
    }
    # Pre-computed wrapped phases for all layers
    _save_wrapped_phases(payload, student)
    # Legacy keys for backward-compatible extraction
    for key, value in student_state.items():
        if "phase_raw" in key:
            payload[key] = value.detach().cpu()
            payload[key.replace("phase_raw", "wrapped_slm_0_2pi")] = \
                torch.remainder(value.detach().cpu(), 2 * np.pi)
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def save_detector_best(detector, path, epoch, loss_value, extra=None,
                       student=None, config=None):
    payload = {
        "detector_state_dict": detector.state_dict(),
        "epoch": int(epoch),
        "loss": float(loss_value),
    }
    if student is not None:
        student_state = student.state_dict()
        payload["student_state_dict"] = student_state
        payload["student_enable_norm"] = bool(getattr(student, "enable_norm", False))
        payload["num_layers"] = int(getattr(student, "num_layers", 2))
        payload["num_heads"] = int(getattr(student, "num_heads", 1))
        _save_wrapped_phases(payload, student)
        for key, value in student_state.items():
            if "phase_raw" in key:
                payload[key] = value.detach().cpu()
                payload[key.replace("phase_raw", "wrapped_slm_0_2pi")] = \
                    torch.remainder(value.detach().cpu(), 2 * np.pi)
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def _save_wrapped_phases(payload, student):
    """Save continuous phase plus the calibrated, finite-level SLM drive."""
    if hasattr(student, "all_slm_layers"):
        for layer_name, slm in student.all_slm_layers():
            payload[f"{layer_name}_wrapped_phase"] = slm.wrapped_phase().detach().cpu()
            payload[f"{layer_name}_effective_phase"] = slm.effective_phase().detach().cpu()
            payload[f"{layer_name}_gray_drive"] = slm.phase_to_gray_uint8().cpu()
    else:
        for layer_name in _layer_names_from_student(student):
            slm = getattr(student, layer_name, None)
            if slm is not None:
                payload[f"{layer_name}_wrapped_phase"] = slm.wrapped_phase().detach().cpu()
                payload[f"{layer_name}_effective_phase"] = slm.effective_phase().detach().cpu()
                payload[f"{layer_name}_gray_drive"] = slm.phase_to_gray_uint8().cpu()
