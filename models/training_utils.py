import os
import matplotlib
matplotlib.use('Agg')   # 无 GUI 的后端
import matplotlib.pyplot as plt
import numpy as np
import torch


def _unwrap(model):
    """Unwrap model from DataParallel or DistributedDataParallel."""
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("teacher_state_dict", "model_state_dict", "state_dict", "model"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def load_teacher_checkpoint(config, teacher, checkpoint_path, device):
    if not checkpoint_path:
        return False, "Teacher checkpoint: not configured"
    checkpoint_path = checkpoint_path if os.path.isabs(checkpoint_path) else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), checkpoint_path)
    if not os.path.exists(checkpoint_path):
        return False, f"Teacher checkpoint not found: {checkpoint_path}"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    teacher_state = teacher.state_dict()
    compatible_state = {}
    for key, value in state_dict.items():
        normalized_key = key[8:] if key.startswith("teacher.") else key
        if normalized_key in teacher_state and teacher_state[normalized_key].shape == value.shape:
            compatible_state[normalized_key] = value
    if not compatible_state:
        return False, f"No compatible ConvTeacher weights found in: {checkpoint_path}"
    teacher.load_state_dict({**teacher_state, **compatible_state}, strict=False)
    return True, f"Loaded {len(compatible_state)} teacher tensors from: {checkpoint_path}"


def initialize_teacher_weights(config, teacher, device):
    if config.get_teacher_init_mode() == "scratch":
        return False, "Teacher init mode: scratch (training from random initialization)"
    loaded, message = load_teacher_checkpoint(config, teacher, config.get_teacher_init_checkpoint(), device)
    if loaded:
        return True, f"Teacher init mode: checkpoint ({message})"
    return False, f"Teacher init mode: checkpoint requested but unavailable, fallback to scratch ({message})"


def build_optimizer_from_model(config, model, teacher_lr=None, detector_lr=None):
    model_core = _unwrap(model)
    teacher_lr = config.LEARNING_RATE if teacher_lr is None else teacher_lr
    detector_lr = config.LEARNING_RATE if detector_lr is None else detector_lr
    param_groups = []
    teacher_params = [p for p in model_core.teacher.parameters() if p.requires_grad]
    detector_params = [p for p in model_core.detector.parameters() if p.requires_grad]
    if teacher_params:
        param_groups.append({"params": teacher_params, "lr": teacher_lr})
    if detector_params:
        param_groups.append({"params": detector_params, "lr": detector_lr})
    if not param_groups:
        raise ValueError("No trainable parameters found when building optimizer.")
    return torch.optim.Adam(param_groups, weight_decay=config.WEIGHT_DECAY)


def set_detector_trainable(model, trainable):
    model_core = _unwrap(model)
    for p in model_core.detector.parameters():
        p.requires_grad = trainable


def _valid_history_points(values):
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


def save_training_curves(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    train_x, train_y = _valid_history_points(history.get("train_total", []))
    val_x, val_y = _valid_history_points(history.get("val_total", []))
    axes[0].plot(train_x, train_y, label="train_total")
    axes[0].plot(val_x, val_y, label="val_total")
    axes[0].set_title("Loss")
    axes[0].legend()
    for axis_idx, metric in enumerate(("precision", "recall", "map50"), start=1):
        xs, ys = _valid_history_points(history.get(metric, []))
        axes[axis_idx].plot(xs, ys, label=metric)
        axes[axis_idx].set_title(metric)
        axes[axis_idx].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=130)
    plt.close()
