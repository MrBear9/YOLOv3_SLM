import os
from datetime import datetime
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


def load_joint_teacher_detector_checkpoint(config, teacher, detector, checkpoint_path, device):
    if not checkpoint_path:
        return None, "Joint checkpoint: not configured"
    checkpoint_path = checkpoint_path if os.path.isabs(checkpoint_path) else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), checkpoint_path
    )
    if not os.path.exists(checkpoint_path):
        return None, f"Joint checkpoint not found: {checkpoint_path}"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "teacher_state_dict" not in checkpoint or "detector_state_dict" not in checkpoint:
        return None, f"Joint checkpoint is missing teacher/detector state dictionaries: {checkpoint_path}"
    teacher.load_state_dict(checkpoint["teacher_state_dict"], strict=True)
    detector.load_state_dict(checkpoint["detector_state_dict"], strict=True)
    return checkpoint, (
        f"Loaded joint teacher/detector checkpoint: {checkpoint_path} "
        f"(epoch={checkpoint.get('epoch')}, mAP50={checkpoint.get('val_map50')})"
    )


def build_optimizer_from_model(config, model, teacher_lr=None, detector_lr=None):
    model_core = _unwrap(model)
    teacher_lr = config.LEARNING_RATE if teacher_lr is None else teacher_lr
    detector_lr = config.LEARNING_RATE if detector_lr is None else detector_lr
    param_groups = []
    teacher_params = [p for p in model_core.teacher.parameters() if p.requires_grad]
    detector_params = [p for p in model_core.detector.parameters() if p.requires_grad]
    if teacher_params:
        param_groups.append({"params": teacher_params, "lr": teacher_lr, "role": "teacher"})
    if detector_params:
        param_groups.append({"params": detector_params, "lr": detector_lr, "role": "detector"})
    if not param_groups:
        raise ValueError("No trainable parameters found when building optimizer.")
    optimizer_name = str(getattr(config, "OPTIMIZER", "Adam")).strip()
    if optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
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


def create_tensorboard_writer(config, output_dir, log_fn=None):
    """Create one isolated TensorBoard run for each training launch."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        if log_fn is not None:
            log_fn(config, f"TensorBoard logging disabled: {exc}")
        return None

    timestamp = getattr(config, "TIMESTAMP", None) or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(output_dir, "tensorboard", f"run_{timestamp}")
    writer = SummaryWriter(log_dir)
    if log_fn is not None:
        log_fn(config, f"TensorBoard log directory: {log_dir}")
    return writer


def add_tensorboard_scalar(writer, tag, value, step):
    if writer is None or value is None:
        return
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return
    if np.isnan(numeric_value) or np.isinf(numeric_value):
        return
    writer.add_scalar(tag, numeric_value, step)


def save_training_curves(history, output_dir, op_conf_threshold=None):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    train_x, train_y = _valid_history_points(history.get("train_total", []))
    val_x, val_y = _valid_history_points(history.get("val_total", []))
    axes[0].plot(train_x, train_y, label="train_total")
    axes[0].plot(val_x, val_y, label="val_total")
    axes[0].set_title("Loss")
    axes[0].legend()

    # Precision: prefer operating-point if available, otherwise fallback to metric threshold
    prec_op_x, prec_op_y = _valid_history_points(history.get("precision_op", []))
    if prec_op_y:
        op_label = "Precision (op)"
        if op_conf_threshold is not None:
            op_label = f"Precision (op, conf={float(op_conf_threshold):g})"
        axes[1].plot(prec_op_x, prec_op_y, label=op_label)
    else:
        prec_x, prec_y = _valid_history_points(history.get("precision", []))
        axes[1].plot(prec_x, prec_y, label="precision")
    axes[1].set_title("Precision")
    axes[1].legend()

    for axis_idx, metric in enumerate(("recall", "map50"), start=2):
        xs, ys = _valid_history_points(history.get(metric, []))
        axes[axis_idx].plot(xs, ys, label=metric)
        axes[axis_idx].set_title(metric)
        axes[axis_idx].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=130)
    plt.close()
