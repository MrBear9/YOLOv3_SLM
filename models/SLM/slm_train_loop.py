"""SLM student training main loop.

Contains the train() function for optical_slm_yolov8_head.py.
Supporting helpers live in:
- slm_utils.py         : config, scheduling, visualization, phase grad, checkpoint, log_config
- slm_train_setup.py   : setup_training() — model/data/criterion construction
- slm_train_epoch.py   : run_epoch() — single epoch logic
"""

import os
import warnings

import torch

from models.SLM.config_slm import ConfigSLM as Config
from models.SLM.slm_utils import (
    configure_student_norm_for_stage,
    save_slm_component_curves,
    stage_schedule,
)
from models.SLM.slm_train_epoch import run_epoch
from models.SLM.slm_train_setup import setup_training
from models.SLM.utils_slm import collect_slm_statistics, save_detector_best, save_student_best
from models.monitoring import snapshot_parameters
from models.runtime import (
    cleanup_distributed,
    init_distributed_mode,
    init_epoch_log_table,
    log_to_file,
)
from models.training_utils import save_training_curves

warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")


def train():
    local_rank, use_ddp = init_distributed_mode(Config)
    is_main = local_rank == 0

    ctx = setup_training(is_main, use_ddp)

    best_student_loss = float("inf")
    best_detector_loss = float("inf")
    best_map50 = -1.0
    best_student_map50 = -1.0
    global_epoch = 0

    # 初始化参数快照用于变化追踪
    if is_main:
        snapshot_parameters(ctx["student"])
        snapshot_parameters(ctx["detector"])
    deployment_norm_mode = Config.STUDENT_NORM_MODE
    init_epoch_log_table(Config)

    for stage_name, stage_epochs in stage_schedule():
        if stage_epochs <= 0:
            continue
        stage_norm_mode = configure_student_norm_for_stage(stage_name, deployment_norm_mode)
        stage_weights = Config.get_stage_loss_weights(stage_name)
        ctx["student_raw"].enable_norm = bool(Config.ENABLE_STUDENT_NORM and stage_norm_mode != "none")

        # Configure trainable components per stage
        if stage_name == "phase_focus":
            _set_trainable_both(ctx["student"], ctx["detector"], student_trainable=True, detector_trainable=False)
        elif stage_name == "detector_focus":
            _set_trainable_both(ctx["student"], ctx["detector"], student_trainable=False, detector_trainable=True)
        else:
            _set_trainable_both(ctx["student"], ctx["detector"], student_trainable=True, detector_trainable=True)

        # Build optimizer & scheduler for this stage
        from models.SLM.utils_slm import build_stage_optimizer
        from models.SLM.slm_utils import build_stage_scheduler
        optimizer = build_stage_optimizer(Config, ctx["student_raw"], ctx["detector_raw"], stage_name)
        scheduler = build_stage_scheduler(optimizer, stage_epochs)
        ctx["optimizer"] = optimizer
        ctx["scheduler"] = scheduler
        ctx["is_main"] = is_main

        norm_is_deployment_ready = Config.ENABLE_STUDENT_NORM and stage_norm_mode != "none"
        log_to_file(
            Config,
            f"Start stage={stage_name}, epochs={stage_epochs}, student_norm_mode={stage_norm_mode}, "
            f"student_enable_norm={ctx['student_raw'].enable_norm}, "
            f"scheduler={Config.LR_SCHEDULER if scheduler is not None else 'none'}",
        )
        detector_no_improve = 0

        for _ in range(stage_epochs):
            best_map50, best_student_map50, best_student_loss, best_detector_loss, no_improve_delta = run_epoch(
                global_epoch,
                stage_name,
                stage_weights,
                norm_is_deployment_ready,
                ctx,
                best_map50,
                best_student_map50,
                best_student_loss,
                best_detector_loss,
            )
            global_epoch += 1
            detector_no_improve += no_improve_delta
            should_stop_stage = (
                stage_name == "detector_focus"
                and Config.ENABLE_DETECTOR_FOCUS_EARLY_STOP
                and Config.DETECTOR_FOCUS_EARLY_STOP_PATIENCE > 0
                and detector_no_improve >= Config.DETECTOR_FOCUS_EARLY_STOP_PATIENCE
            )
            if use_ddp:
                stop_tensor = torch.tensor(
                    int(should_stop_stage) if is_main else 0,
                    dtype=torch.int32,
                    device=ctx["device"],
                )
                torch.distributed.broadcast(stop_tensor, src=0)
                should_stop_stage = bool(stop_tensor.item())
            if should_stop_stage:
                log_to_file(
                    Config,
                    f"Early stopping detector_focus after {detector_no_improve} epochs without mAP50 improvement. "
                    f"Best mAP50={best_map50:.4f}.",
                )
                break

    _save_fallback_checkpoints(ctx, global_epoch)
    _finalize(ctx)


def _set_trainable_both(student, detector, student_trainable, detector_trainable):
    from models.SLM.utils_slm import set_trainable
    set_trainable(student, student_trainable)
    if student_trainable:
        set_trainable(student.module.slm1 if hasattr(student, "module") else student.slm1, Config.TRAIN_SLM1)
        set_trainable(student.module.slm2 if hasattr(student, "module") else student.slm2, Config.TRAIN_SLM2)
    set_trainable(detector, detector_trainable)


def _save_fallback_checkpoints(ctx, global_epoch):
    """Save fallback checkpoints if no best was selected during training."""
    is_main = ctx["is_main"]
    student_raw = ctx["student_raw"]
    detector_raw = ctx["detector_raw"]
    history = ctx["history"]

    if is_main and not os.path.exists(Config.get_detector_best_path()):
        fallback_loss = history["train_total"][-1] if history["train_total"] else 0.0
        slm_stats = collect_slm_statistics(student_raw)
        save_detector_best(
            detector_raw,
            Config.get_detector_best_path(),
            global_epoch + 1,
            fallback_loss,
            extra={
                "train_loss": fallback_loss,
                "val_loss": None,
                "val_map50": None,
                "selection_metric": "fallback_last_train_loss",
                "recommended_inference_checkpoint": True,
                "paired_student_source": "student_state_dict",
                "paired_student_epoch": global_epoch + 1,
                "paired_student_stage": "fallback_last_epoch",
                "detector_head_type": Config.DETECTOR_HEAD_TYPE,
                "detection_protocol": Config.DETECTION_PROTOCOL,
                "anchor_match_mode": Config.ANCHOR_MATCH_MODE,
                "student_norm_mode": Config.STUDENT_NORM_MODE,
                "student_norm_schedule": Config.STUDENT_NORM_SCHEDULE,
                "slm_stats": slm_stats,
                "slm_quality_passed": False,
            },
            student=student_raw,
            config=Config,
        )
        save_student_best(
            Config,
            student_raw,
            Config.get_student_best_path(),
            global_epoch + 1,
            fallback_loss,
            extra={
                "train_loss": fallback_loss,
                "val_loss": None,
                "val_map50": None,
                "slm_stats": slm_stats,
                "paired_with_detector_best": True,
                "paired_detector_checkpoint": Config.get_detector_best_path(),
                "paired_student_epoch": global_epoch + 1,
                "paired_student_stage": "fallback_last_epoch",
                "selection_metric": "fallback_last_train_loss",
                "recommended_inference_checkpoint": False,
                "student_norm_mode": Config.STUDENT_NORM_MODE,
                "student_norm_schedule": Config.STUDENT_NORM_SCHEDULE,
                "slm_quality_passed": False,
            },
        )
        log_to_file(
            Config,
            "No validation-selected detector best was saved; wrote fallback detector_best.pth with the final "
            "student_state_dict for checkpoint compatibility.",
        )
    if is_main and not os.path.exists(Config.get_student_best_path()):
        slm_stats = collect_slm_statistics(student_raw)
        log_to_file(
            Config,
            "No paired optical_student_best.pth was saved because detector_best.pth was never updated and fallback "
            f"student mirror creation failed. Last stats: {slm_stats}",
        )


def _finalize(ctx):
    """Save training curves, close TensorBoard, log completion."""
    is_main = ctx["is_main"]
    history = ctx["history"]
    tensorboard_writer = ctx["tensorboard_writer"]

    if is_main:
        save_training_curves(history, Config.OUTPUT_DIR, op_conf_threshold=Config.CONF_THRESH)
        save_slm_component_curves(history, Config.OUTPUT_DIR)
        if tensorboard_writer is not None:
            tensorboard_writer.close()
    log_to_file(Config, "=" * 80)
    log_to_file(Config, "Training complete")
    log_to_file(Config, f"Recommended inference checkpoint: {Config.get_detector_best_path()}")
    log_to_file(Config, f"Paired SLM student mirror for phase extraction: {Config.get_student_best_path()}")

    if ctx["use_ddp"]:
        cleanup_distributed()
