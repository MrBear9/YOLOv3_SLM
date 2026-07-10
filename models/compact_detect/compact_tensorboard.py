"""Compact detection TensorBoard helpers.

Contains write_tensorboard_model_summary() and write_tensorboard_scalars().
"""

from models.runtime import unwrap_module
from models.training_utils import add_tensorboard_scalar

from .compact_utils import module_param_count
from .config import ConfigCompactDetect as Config
from .model import OpticalCompactDetector


def write_tensorboard_model_summary(writer, student, detector, sample_shape):
    if writer is None:
        return
    student = unwrap_module(student)
    detector = unwrap_module(detector)
    detector_lines = [f"- detector.{name}: {module_param_count(child):,}" for name, child in detector.named_children()]
    student_lines = [f"- student.{name}: {module_param_count(child):,}" for name, child in student.named_children()]
    flow_text = "\n".join([
        "Input grayscale image",
        "-> OpticalStudent",
        "-> CompactOpticalDetector stem",
        "-> heatmap head / wh head / offset head",
        "-> center decode + NMS",
        "",
        f"Input tensor: {tuple(sample_shape)}",
        f"Detection tensor stride: {Config.OUTPUT_STRIDE}",
        f"SLM init mode: {getattr(Config, 'SLM_INIT_MODE', 'unknown')}",
        f"Visualization seed: {getattr(Config, 'VIS_SEED', 'unknown')}",
        f"Teacher feature warmup epochs: {getattr(Config, 'COMPACT_TEACHER_WARMUP_EPOCHS', 0)}",
        f"Teacher feature warmup subset size: {getattr(Config, 'COMPACT_TEACHER_WARMUP_SUBSET_SIZE', 0)}",
        f"Teacher feature warmup raw student: {getattr(Config, 'COMPACT_TEACHER_WARMUP_RAW_STUDENT', True)}",
        "Output tensors: heatmap=(B,C,H/4,W/4), wh=(B,2,H/4,W/4), offset=(B,2,H/4,W/4)",
    ])
    param_text = "\n".join([
        f"Student parameters: {module_param_count(student):,}",
        f"Compact detector parameters: {module_param_count(detector):,}",
        f"Combined route parameters: {module_param_count(OpticalCompactDetector(student, detector)):,}",
        "",
        "Detector module parameters:",
        *detector_lines,
        "",
        "Student module parameters:",
        *student_lines,
    ])
    writer.add_text("Model/data_flow", flow_text, 0)
    writer.add_text("Model/parameters", param_text, 0)


def write_tensorboard_scalars(writer, step, train_loss, train_stats, val_losses=None, val_metrics=None, lr=None):
    add_tensorboard_scalar(writer, "Loss/train_total", train_loss, step)
    for key, value in train_stats.items():
        add_tensorboard_scalar(writer, f"Loss/train_{key}", value, step)
    if val_losses is not None:
        for key, value in val_losses.items():
            add_tensorboard_scalar(writer, f"Loss/val_{key}", value, step)
    if val_metrics is not None:
        for key, value in val_metrics.items():
            add_tensorboard_scalar(writer, f"Metrics/{key}", value, step)
    add_tensorboard_scalar(writer, "LR/current", lr, step)
