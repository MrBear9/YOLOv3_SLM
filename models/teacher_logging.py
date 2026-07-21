"""Teacher training logging & initialization helpers.

Contains bootstrap_runtime(), log_all_parameters(), and
write_teacher_tensorboard_scalars() split out from the original
train_teacher_helpers.py so each module stays focused and small.
"""

import os

import torch

from models.runtime import init_log_file, log_to_file
from models.teacher import build_teacher
from models.training_utils import add_tensorboard_scalar
from models.yolov8.config_v8 import ConfigYOLOv8Anchor as Config
from models.yolov8.head_v8 import build_detector_head


def bootstrap_runtime():
    Config.initialize()
    Config.print_config()
    init_log_file(Config)
    log_to_file(Config, f"Log file path: {Config.LOG_FILE}")
    log_to_file(Config, f"Visualization save path: {Config.TEACHER_OUTPUT_DIR}")
    log_to_file(Config, f"Class info: {Config.CLASS_NAMES}, Num classes: {Config.NUM_CLASSES}")


def log_all_parameters():
    log_to_file(Config, "=" * 80)
    log_to_file(Config, "Optical teacher lightweight multi-scale detector configuration")
    log_to_file(Config, "=" * 80)
    log_to_file(Config, f"Dataset: {Config.YAML_PATH}")
    log_to_file(Config, f"Classes: {Config.CLASS_NAMES}")
    log_to_file(Config, f"Image size / batch / epochs: {Config.IMG_SIZE} / {Config.BATCH_SIZE} / {Config.EPOCHS}")
    log_to_file(
        Config,
        f"Head factory: type={Config.DETECTOR_HEAD_TYPE}, protocol={Config.DETECTION_PROTOCOL}, light_base_ch={Config.YOLO_LIGHT_BASE_CH}, "
        f"yolov8_base_ch={Config.YOLOV8_BASE_CHANNELS}, c2f_blocks={Config.YOLOV8_C2F_BLOCKS}",
    )
    active_strides = Config.ANCHOR_FREE_STRIDES if Config.DETECTION_PROTOCOL == "anchor_free_tal" else Config.STRIDES
    log_to_file(Config, f"Strides: {active_strides}")
    if Config.DETECTION_PROTOCOL == "anchor":
        log_to_file(Config, f"Anchor source: {Config.ANCHOR_SOURCE}")
        log_to_file(Config, f"Anchors: {Config.ANCHORS}")
        log_to_file(Config, f"Loss weights box/obj/noobj/cls: {Config.BOX_WEIGHT_BASE}/{Config.OBJ_WEIGHT_BASE}/{Config.NOOBJ_WEIGHT_BASE}/{Config.CLS_WEIGHT_BASE}")
    else:
        log_to_file(
            Config,
            f"Anchor-free TAL/DFL: head_ch={Config.ANCHOR_FREE_HEAD_CH}, reg_max={Config.ANCHOR_FREE_REG_MAX}, "
            f"topk={Config.TAL_TOPK}, alpha={Config.TAL_ALPHA}, beta={Config.TAL_BETA}",
        )
        log_to_file(
            Config,
            f"Light P2: fusion_ch={Config.ANCHOR_FREE_P2_FUSION_CH}, head_ch={Config.ANCHOR_FREE_P2_HEAD_CH}",
        )
        log_to_file(Config, "Anchor-free classification: BCE with TAL IoU soft targets")
    if Config.DETECTION_PROTOCOL == "anchor":
        log_to_file(Config, f"Focal alpha/gamma: {Config.FOCAL_ALPHA}/{Config.FOCAL_GAMMA}")
    if Config.DETECTION_PROTOCOL == "anchor":
        log_to_file(
            Config,
            f"Anchor matching: mode={Config.ANCHOR_MATCH_MODE}, ratio_thresh={Config.ANCHOR_MATCH_RATIO_THRESH}, "
            f"neighbor_cells={Config.ASSIGN_NEIGHBOR_CELLS}, simota_iou={Config.ANCHOR_MATCH_IOU_THRESH}, "
            f"center_radius={Config.CENTER_PRIOR_RADIUS}, top_n={Config.SIMOTA_TOP_N}, max_assign={Config.SIMOTA_MAX_ASSIGN}",
        )
    log_to_file(Config, f"Hard negative mining: ratio={Config.HARD_NEG_RATIO}, min={Config.HARD_NEG_MIN}")
    log_to_file(Config, f"Box decode range: {Config.BOX_DECODE_RANGE}")
    log_to_file(Config, f"LR teacher/detector: {Config.PHASE1_TEACHER_LR}/{Config.PHASE1_DETECTOR_LR} -> {Config.PHASE2_TEACHER_LR}/{Config.PHASE2_DETECTOR_LR}")
    log_to_file(Config, f"Detection conf/nms/max_det: {Config.CONF_THRESH}/{Config.NMS_THRESH}/{Config.MAX_DET}")
    log_to_file(Config, f"Metric conf/nms/max_det: {Config.METRIC_CONF_THRESH}/{Config.METRIC_NMS_THRESH}/{Config.METRIC_MAX_DET}")
    log_to_file(
        Config,
        f"Training data: letterbox=True, augment={Config.TRAIN_AUGMENT}, hflip={Config.AUG_HFLIP_PROB}, "
        f"rotate={Config.AUG_ROTATE_DEG}, scale={Config.AUG_SCALE_MIN}-{Config.AUG_SCALE_MAX}, translate={Config.AUG_TRANSLATE}",
    )
    log_to_file(
        Config,
        f"Small-soldier Copy-Paste: enabled={Config.SOLDIER_COPY_PASTE}, prob={Config.SOLDIER_COPY_PASTE_PROB}, "
        f"max_objects={Config.SOLDIER_COPY_PASTE_MAX_OBJECTS}, area_max={Config.SOLDIER_COPY_PASTE_AREA_MAX}, "
        f"scale={Config.SOLDIER_COPY_PASTE_SCALE_MIN}-{Config.SOLDIER_COPY_PASTE_SCALE_MAX}, "
        f"ioa_max={Config.SOLDIER_COPY_PASTE_IOA_MAX}, max_existing={Config.SOLDIER_COPY_PASTE_MAX_EXISTING}, "
        f"disable_last_epochs={Config.SOLDIER_COPY_PASTE_DISABLE_LAST_EPOCHS}",
    )
    log_to_file(Config, f"Output: {Config.TEACHER_OUTPUT_DIR}")
    teacher = build_teacher(Config)
    detector = build_detector_head(Config, in_channels=1, out_channels=Config.get_detector_output_channels())
    arch_lower = str(Config.TEACHER_ARCH).strip().lower()
    log_to_file(Config, f"Teacher arch: {Config.TEACHER_ARCH}")
    if arch_lower in {"convteacher_v2", "v2"}:
        log_to_file(
            Config,
            f"V2 CVOCA teacher: synthetic_wavelengths={Config.TEACHER_V2_SYNTHETIC_WAVELENGTHS}, "
            f"complex_kernel={Config.TEACHER_V2_COMPLEX_KERNEL_SIZE}",
        )
    log_to_file(Config, f"Teacher parameters: {sum(p.numel() for p in teacher.parameters() if p.requires_grad):,}")
    log_to_file(Config, f"Detector head type: {Config.DETECTOR_HEAD_TYPE}")
    log_to_file(Config, f"Detector parameters: {sum(p.numel() for p in detector.parameters() if p.requires_grad):,}")
    if arch_lower in {"convteacher_v3", "v3"}:
        log_to_file(Config, f"V3 residual_scale={Config.TEACHER_V3_RESIDUAL_SCALE}")
        log_to_file(
            Config,
            f"V3 physical regularization gate/residual/output: "
            f"{Config.TEACHER_V3_GATE_SPARSITY_WEIGHT}/"
            f"{Config.TEACHER_V3_RESIDUAL_L1_WEIGHT}/"
            f"{Config.TEACHER_V3_OUTPUT_DEVIATION_WEIGHT}",
        )
    if getattr(Config, "ENABLE_FEATURE_DISTILL", False):
        log_to_file(Config, f"Feature distillation: weight={Config.FEATURE_DISTILL_WEIGHT}")
    log_to_file(Config, f"AMP: enabled={Config.ENABLE_AMP}, dtype={Config.AMP_DTYPE}")
    log_to_file(
        Config,
        f"Teacher SLM-cipher loss: weight={Config.TEACHER_SLM_CIPHER_LOSS_WEIGHT}, "
        f"blur_kernel={Config.TEACHER_SLM_CIPHER_BLUR_KERNEL}, "
        f"tv_target={Config.TEACHER_SLM_CIPHER_TV_TARGET}, "
        f"hf_target={Config.TEACHER_SLM_CIPHER_HF_TARGET}, "
        f"range_floor={Config.TEACHER_SLM_CIPHER_RANGE_FLOOR}, "
        f"mean_floor={Config.TEACHER_SLM_CIPHER_MEAN_FLOOR}, "
        f"peak_limit={Config.TEACHER_SLM_CIPHER_PEAK_LIMIT}, "
        f"edge_limit={Config.TEACHER_SLM_CIPHER_EDGE_LIMIT}",
    )
    log_to_file(Config, "=" * 80)


def write_teacher_tensorboard_scalars(writer, step, train_losses, val_losses=None, val_metrics=None, lr=None):
    if writer is None:
        return
    for key, value in train_losses.items():
        if key.startswith("copy_paste_"):
            add_tensorboard_scalar(writer, f"Augmentation/CopyPaste/{key.removeprefix('copy_paste_')}", value, step)
        elif key.startswith("positive_"):
            assignment_name = key.removeprefix("positive_")
            if assignment_name.startswith("class_"):
                cls_id = int(assignment_name.removeprefix("class_"))
                assignment_name = f"class/{Config.CLASS_NAMES.get(cls_id, f'class_{cls_id}')}"
            add_tensorboard_scalar(writer, f"Assignment/train_positive/{assignment_name}", value, step)
        else:
            add_tensorboard_scalar(writer, f"Loss/train_{key}", value, step)
    if val_losses is not None:
        for key, value in val_losses.items():
            add_tensorboard_scalar(writer, f"Loss/val_{key}", value, step)
    if val_metrics is not None:
        for key in ("precision", "recall", "f1", "map50", "precision_op", "recall_op", "f1_op"):
            add_tensorboard_scalar(writer, f"Metrics/{key}", val_metrics.get(key), step)
        op_conf = float(getattr(Config, "CONF_THRESH", 0.35))
        threshold_tag = f"conf_{op_conf:g}"
        for key in ("precision_op", "recall_op", "f1_op"):
            add_tensorboard_scalar(writer, f"MetricsOperating/{threshold_tag}/{key}", val_metrics.get(key), step)
        for cls_id, values in val_metrics.get("per_class", {}).items():
            cls_name = Config.CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            for key in ("ap50", "precision", "recall", "f1", "confidence"):
                add_tensorboard_scalar(writer, f"MetricsPerClass/{cls_name}/{key}", values.get(key), step)
            add_tensorboard_scalar(writer, f"MetricsPerClass/{cls_name}/gt_count", values.get("gt_count"), step)
            for size_name, recall in values.get("size_recall", {}).items():
                add_tensorboard_scalar(writer, f"MetricsPerClass/{cls_name}/size_{size_name}_recall", recall, step)
                add_tensorboard_scalar(
                    writer, f"MetricsPerClass/{cls_name}/size_{size_name}_gt_count",
                    values.get("size_gt_count", {}).get(size_name), step,
                )
            pr_data = val_metrics.get("pr_data", {}).get(cls_id, {})
            if pr_data.get("confidence"):
                writer.add_pr_curve(
                    f"PRCurve/{cls_name}",
                    labels=torch.tensor(pr_data["label"], dtype=torch.int32),
                    predictions=torch.tensor(pr_data["confidence"], dtype=torch.float32),
                    global_step=step, num_thresholds=127,
                )
        for size_name, value in val_metrics.get("size_recall", {}).items():
            add_tensorboard_scalar(writer, f"MetricsBySize/{size_name}/recall", value, step)
            add_tensorboard_scalar(
                writer, f"MetricsBySize/{size_name}/gt_count",
                val_metrics.get("size_gt_count", {}).get(size_name), step,
            )
    add_tensorboard_scalar(writer, "LR/current", lr, step)
