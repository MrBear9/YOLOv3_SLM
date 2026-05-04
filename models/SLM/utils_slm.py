import os

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
    compatible = {}
    for raw_key, value in state_dict.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
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
        info["teacher_loaded"], info["teacher_total"] = load_matching_state(teacher, extract_state_dict(checkpoint["teacher_state_dict"]), prefixes=("teacher.",))
        info["detector_loaded"], info["detector_total"] = load_matching_state(detector, extract_state_dict(checkpoint["detector_state_dict"]), prefixes=("detector.",))
        return info
    state_dict = extract_state_dict(checkpoint)
    info["teacher_loaded"], info["teacher_total"] = load_matching_state(teacher, state_dict, prefixes=("teacher.",))
    info["detector_loaded"], info["detector_total"] = load_matching_state(detector, state_dict, prefixes=("detector.",))
    if info["teacher_loaded"] == 0 and info["detector_loaded"] == 0:
        raise RuntimeError(f"No compatible teacher/detector weights found in: {checkpoint_path}")
    return info


def split_student_param_groups(student):
    slm_params = []
    other_params = []
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        if "phase_raw" in name or "amp_raw" in name:
            slm_params.append(param)
        else:
            other_params.append(param)
    return slm_params, other_params


def build_stage_optimizer(config, student, detector, stage_name):
    if stage_name == "student_only":
        slm_params, other_params = split_student_param_groups(student)
        groups = []
        if slm_params:
            groups.append({"params": slm_params, "lr": config.PHASE_PARAM_LR})
        if other_params:
            groups.append({"params": other_params, "lr": config.STUDENT_LR})
        return torch.optim.Adam(groups, weight_decay=config.WEIGHT_DECAY)
    if stage_name == "detector_only":
        return torch.optim.Adam([p for p in detector.parameters() if p.requires_grad], lr=config.DETECTOR_LR, weight_decay=config.WEIGHT_DECAY)
    slm_params, other_params = split_student_param_groups(student)
    groups = []
    if slm_params:
        groups.append({"params": slm_params, "lr": config.JOINT_PHASE_PARAM_LR})
    if other_params:
        groups.append({"params": other_params, "lr": config.JOINT_STUDENT_LR})
    groups.append({"params": [p for p in detector.parameters() if p.requires_grad], "lr": config.JOINT_DETECTOR_LR})
    return torch.optim.Adam(groups, weight_decay=config.WEIGHT_DECAY)


def set_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable


def collect_slm_statistics(student):
    stats = {}
    for layer_name in ("slm1", "slm2"):
        slm = getattr(student, layer_name)
        centered = slm.centered_phase().detach().float()
        stats[f"{layer_name}_wrapped_std"] = float(centered.std(unbiased=False).item())
        stats[f"{layer_name}_wrapped_span"] = float((centered.amax() - centered.amin()).item())
    return stats


def save_student_best(config, student, path, epoch, loss_value, extra=None):
    student_state = student.state_dict()
    payload = {
        "student_state_dict": student_state,
        "epoch": int(epoch),
        "loss": float(loss_value),
        "student_enable_norm": bool(getattr(student, "enable_norm", False)),
    }
    for key, value in student_state.items():
        if "phase_raw" in key:
            payload[key] = value.detach().cpu()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def save_detector_best(detector, path, epoch, loss_value, extra=None):
    payload = {
        "detector_state_dict": detector.state_dict(),
        "epoch": int(epoch),
        "loss": float(loss_value),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
