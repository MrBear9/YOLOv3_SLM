import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.SLM.config_slm import ConfigSLM as Config
from models.SLM.dataset_slm import SLMFeatureDataset, slm_collate_fn
from models.SLM.evaluation_slm import evaluate_slm_detector, save_slm_detection_visualization
from models.SLM.losses_slm import CompositeOpticalFeatureLoss, detection_response_loss
from models.SLM.optical_layers import OpticalStudent
from models.SLM.utils_slm import (
    build_stage_optimizer,
    collect_slm_statistics,
    load_student_checkpoint,
    load_teacher_detector_checkpoint,
    save_detector_best,
    save_student_best,
    set_trainable,
)
from models.runtime import init_epoch_log_table, init_log_file, log_epoch_table_row, log_to_file
from models.teacher import build_teacher
from models.training_utils import save_training_curves
from models.yolov8.head_v8 import YOLOv8AnchorHead
from models.yolov8.loss_anchor_v8 import YOLOv3AnchorLossForV8Head


def configure_backends():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = Config.ENABLE_CUDNN_BENCHMARK
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = Config.ENABLE_TF32
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = Config.ENABLE_TF32


def stage_schedule():
    return [
        ("student_only", Config.STUDENT_ONLY_EPOCHS),
        ("student_adapt_max", Config.STUDENT_ADAPT_MAX_EPOCHS),
        ("detector_only", Config.DETECTOR_ONLY_EPOCHS),
        ("joint_balance", Config.JOINT_EPOCHS),
    ]


def prepare_batch(batch, device):
    gray = batch["gray_tensor"].to(device, non_blocking=Config.PIN_MEMORY)
    rgb = batch["rgb_tensor"].to(device, non_blocking=Config.PIN_MEMORY)
    if Config.ENABLE_CHANNELS_LAST and torch.cuda.is_available():
        gray = gray.contiguous(memory_format=torch.channels_last)
        rgb = rgb.contiguous(memory_format=torch.channels_last)
    return gray, rgb, batch["targets"]


def configure_student_norm_for_stage(stage_name, deployment_norm_mode):
    schedule = str(getattr(Config, "STUDENT_NORM_SCHEDULE", "always")).lower()
    if schedule == "none":
        stage_norm_mode = "none"
    elif schedule == "late" and stage_name == "student_only":
        stage_norm_mode = Config.STUDENT_NORM_EARLY_MODE
    else:
        stage_norm_mode = deployment_norm_mode
    Config.STUDENT_NORM_MODE = stage_norm_mode
    return stage_norm_mode


def build_stage_scheduler(optimizer, stage_epochs):
    if str(getattr(Config, "LR_SCHEDULER", "")).lower() != "cosineannealinglr":
        return None
    return CosineAnnealingLR(optimizer, T_max=max(int(stage_epochs), 1), eta_min=Config.ETA_MIN)


def log_config():
    log_to_file(Config, "=" * 80)
    log_to_file(Config, "Optical SLM student training with YOLOv8-head teacher")
    log_to_file(Config, "=" * 80)
    log_to_file(Config, f"Dataset: {Config.YAML_PATH}")
    log_to_file(Config, f"Output: {Config.OUTPUT_DIR}")
    log_to_file(Config, f"Teacher detector checkpoint: {Config.TEACHER_DETECTOR_CHECKPOINT}")
    log_to_file(Config, f"Teacher arch: {Config.TEACHER_ARCH}")
    log_to_file(Config, f"Classes: {Config.CLASS_NAMES}")
    log_to_file(
        Config,
        f"Epochs student/adapt/detector/joint: "
        f"{Config.STUDENT_ONLY_EPOCHS}/{Config.STUDENT_ADAPT_MAX_EPOCHS}/{Config.DETECTOR_ONLY_EPOCHS}/{Config.JOINT_EPOCHS}",
    )
    log_to_file(Config, f"Save paired detector best: {Config.get_detector_best_path()}")
    log_to_file(Config, f"Save paired student mirror: {Config.get_student_best_path()}")
    log_to_file(
        Config,
        "Recommended inference checkpoint: detector_best.pth, because it carries the detector and "
        "the paired student_state_dict from the same best-mAP epoch.",
    )
    log_to_file(
        Config,
        f"LR student/phase/adapt_student/adapt_phase/detector/joint_detector: "
        f"{Config.STUDENT_LR}/{Config.PHASE_PARAM_LR}/{Config.ADAPT_STUDENT_LR}/"
        f"{Config.ADAPT_PHASE_PARAM_LR}/{Config.DETECTOR_LR}/{Config.JOINT_DETECTOR_LR}",
    )
    log_to_file(Config, f"LR scheduler: {Config.LR_SCHEDULER}, eta_min={Config.ETA_MIN}")
    log_to_file(Config, f"Validation interval: {Config.VAL_INTERVAL}")
    log_to_file(Config, f"Visualization: split={Config.VIS_DATASET_SPLIT}, interval={Config.VIS_INTERVAL}")
    log_to_file(Config, f"SLM init mode: {Config.SLM_INIT_MODE}")
    log_to_file(Config, f"SLM init checkpoint: {Config.SLM_INIT_CHECKPOINT}")
    log_to_file(
        Config,
        f"Student normalization: enabled={Config.ENABLE_STUDENT_NORM}, schedule={Config.STUDENT_NORM_SCHEDULE}, "
        f"early_mode={Config.STUDENT_NORM_EARLY_MODE}, deployment_mode={Config.STUDENT_NORM_MODE}, "
        f"percentile={Config.STUDENT_NORM_PERCENTILE}",
    )
    log_to_file(
        Config,
        "Feature loss weights full/low1/low2/ssim/grad/freq/pearson: "
        f"{Config.LOSS_FULL_WEIGHT}/{Config.LOSS_LOW1_WEIGHT}/{Config.LOSS_LOW2_WEIGHT}/"
        f"{Config.LOSS_SSIM_WEIGHT}/{Config.LOSS_GRAD_WEIGHT}/{Config.LOSS_FREQ_WEIGHT}/{Config.LOSS_PEARSON_WEIGHT}",
    )
    log_to_file(
        Config,
        f"Stage loss weights student/adapt/joint_feature/joint_detection: "
        f"{Config.FEATURE_LOSS_WEIGHT_STUDENT}/{Config.FEATURE_LOSS_WEIGHT_ADAPT}/"
        f"{Config.FEATURE_LOSS_WEIGHT_JOINT}/{Config.DETECTION_LOSS_WEIGHT_JOINT}",
    )
    log_to_file(
        Config,
        f"Detector early stop: patience={Config.DETECTOR_EARLY_STOP_PATIENCE}, "
        f"min_delta={Config.DETECTOR_EARLY_STOP_MIN_DELTA}",
    )
    log_to_file(
        Config,
        "Phase regularization weight/targets diversity/std/span/circular: "
        f"{Config.LOSS_PHASE_DIVERSITY_WEIGHT}/{Config.PHASE_STD_TARGET}/"
        f"{Config.PHASE_SPAN_TARGET}/{Config.PHASE_CIRCULAR_STD_TARGET}",
    )
    log_to_file(Config, "Checkpoint payload intentionally omits a 'phase' key for SLM extraction compatibility.")


def train():
    Config.initialize()
    init_log_file(Config)
    configure_backends()
    log_config()
    device = torch.device(Config.DEVICE)

    teacher = build_teacher(Config).to(device)
    reference_detector = YOLOv8AnchorHead(Config, in_channels=1, out_channels=Config.get_detector_output_channels()).to(device)
    checkpoint_info = load_teacher_detector_checkpoint(teacher, reference_detector, Config.TEACHER_DETECTOR_CHECKPOINT, device)
    log_to_file(Config, f"Loaded teacher/detector checkpoint: {checkpoint_info}")
    set_trainable(teacher, False)
    set_trainable(reference_detector, False)
    teacher.eval()
    reference_detector.eval()

    student = OpticalStudent(Config).to(device)
    init_mode = str(Config.SLM_INIT_MODE).strip().lower()
    if init_mode in {"checkpoint", "vortex_checkpoint"}:
        student_info = load_student_checkpoint(student, Config.SLM_INIT_CHECKPOINT, device)
        log_to_file(Config, f"Initialized SLM student from checkpoint: {student_info}")
        if student_info["loaded"] == 0:
            log_to_file(Config, "SLM checkpoint initialization loaded 0 tensors; using current initialized phases.")
    else:
        log_to_file(Config, f"Initialized SLM student with mode={init_mode}")
    detector = YOLOv8AnchorHead(Config, in_channels=1, out_channels=Config.get_detector_output_channels()).to(device)
    detector.load_state_dict(reference_detector.state_dict(), strict=False)
    if Config.ENABLE_CHANNELS_LAST and torch.cuda.is_available():
        student = student.to(memory_format=torch.channels_last)
        detector = detector.to(memory_format=torch.channels_last)

    train_dataset = SLMFeatureDataset(Config, split="train")
    loader_kwargs = {
        "batch_size": Config.BATCH_SIZE,
        "shuffle": True,
        "num_workers": Config.NUM_WORKERS,
        "pin_memory": Config.PIN_MEMORY,
        "collate_fn": slm_collate_fn,
    }
    if Config.NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = Config.PERSISTENT_WORKERS
        loader_kwargs["prefetch_factor"] = Config.PREFETCH_FACTOR
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_dataset = None
    val_loader = None
    try:
        val_dataset = SLMFeatureDataset(Config, split="val")
        if len(val_dataset) > 0:
            val_kwargs = {
                "batch_size": Config.BATCH_SIZE,
                "shuffle": False,
                "num_workers": Config.NUM_WORKERS,
                "pin_memory": Config.PIN_MEMORY,
                "collate_fn": slm_collate_fn,
            }
            if Config.NUM_WORKERS > 0:
                val_kwargs["persistent_workers"] = Config.PERSISTENT_WORKERS
                val_kwargs["prefetch_factor"] = Config.PREFETCH_FACTOR
            val_loader = DataLoader(val_dataset, **val_kwargs)
            log_to_file(Config, f"Validation dataset: {len(val_dataset)} images")
    except Exception as exc:
        log_to_file(Config, f"Validation dataset unavailable: {exc}")
    vis_dataset = val_dataset if Config.VIS_DATASET_SPLIT == "val" and val_dataset is not None and len(val_dataset) > 0 else train_dataset
    vis_prefix = "val" if vis_dataset is val_dataset else "train"

    feature_criterion = CompositeOpticalFeatureLoss(Config)
    detection_criterion = YOLOv3AnchorLossForV8Head(Config)
    best_student_loss = float("inf")
    best_detector_loss = float("inf")
    best_map50 = -1.0
    best_student_map50 = -1.0
    history = {"train_total": [], "val_total": [], "precision": [], "recall": [], "f1": [], "map50": []}
    global_epoch = 0
    deployment_norm_mode = Config.STUDENT_NORM_MODE
    init_epoch_log_table(Config)

    for stage_name, stage_epochs in stage_schedule():
        if stage_epochs <= 0:
            continue
        stage_norm_mode = configure_student_norm_for_stage(stage_name, deployment_norm_mode)
        if stage_name in {"student_only", "student_adapt_max"}:
            set_trainable(student, True)
            set_trainable(detector, False)
        elif stage_name == "detector_only":
            set_trainable(student, False)
            set_trainable(detector, True)
        else:
            set_trainable(student, True)
            set_trainable(detector, True)
        optimizer = build_stage_optimizer(Config, student, detector, stage_name)
        scheduler = build_stage_scheduler(optimizer, stage_epochs)
        norm_is_deployment_ready = Config.ENABLE_STUDENT_NORM and stage_norm_mode != "none"
        log_to_file(
            Config,
            f"Start stage={stage_name}, epochs={stage_epochs}, student_norm_mode={stage_norm_mode}, "
            f"scheduler={Config.LR_SCHEDULER if scheduler is not None else 'none'}",
        )
        detector_no_improve = 0

        for _ in range(stage_epochs):
            student.train(stage_name != "detector_only")
            detector.train(stage_name not in {"student_only", "student_adapt_max"})
            epoch_total = 0.0
            epoch_feature = 0.0
            epoch_detection = 0.0
            epoch_response = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {global_epoch + 1}/{Config.EPOCHS} [{stage_name}]", leave=True):
                gray, rgb, targets = prepare_batch(batch, device)
                optimizer.zero_grad()
                with torch.no_grad():
                    teacher_feature = teacher(rgb)
                student_feature = student(gray)

                feature_loss, _ = feature_criterion(student_feature, teacher_feature.detach(), student)
                response_loss, _ = detection_response_loss(Config, reference_detector, student_feature, teacher_feature.detach())
                if stage_name in {"student_only", "student_adapt_max"} and Config.DETECTION_LOSS_WEIGHT_STUDENT <= 0:
                    detection_loss = torch.zeros((), device=device)
                else:
                    predictions = detector(student_feature)
                    detection_loss, _ = detection_criterion(predictions, targets)

                if stage_name == "student_only":
                    total_loss = (
                        feature_loss * Config.FEATURE_LOSS_WEIGHT_STUDENT
                        + detection_loss * Config.DETECTION_LOSS_WEIGHT_STUDENT
                        + response_loss
                    )
                elif stage_name == "student_adapt_max":
                    total_loss = feature_loss * Config.FEATURE_LOSS_WEIGHT_ADAPT + response_loss
                elif stage_name == "detector_only":
                    total_loss = detection_loss * Config.DETECTION_LOSS_WEIGHT_DETECTOR
                else:
                    total_loss = (
                        feature_loss * Config.FEATURE_LOSS_WEIGHT_JOINT
                        + detection_loss * Config.DETECTION_LOSS_WEIGHT_JOINT
                        + response_loss
                    )

                total_loss.backward()
                optimizer.step()
                epoch_total += float(total_loss.detach().item())
                epoch_feature += float(feature_loss.detach().item())
                epoch_detection += float(detection_loss.detach().item())
                epoch_response += float(response_loss.detach().item())

            if scheduler is not None:
                scheduler.step()

            num_batches = max(len(train_loader), 1)
            avg_total = epoch_total / num_batches
            avg_feature = epoch_feature / num_batches
            avg_detection = epoch_detection / num_batches
            avg_response = epoch_response / num_batches
            history["train_total"].append(avg_total)
            display_epoch = global_epoch + 1
            slm_stats = collect_slm_statistics(student)
            slm_ok = (
                slm_stats["slm1_wrapped_std"] >= Config.PHASE_BEST_MIN_STD
                and slm_stats["slm2_wrapped_std"] >= Config.PHASE_BEST_MIN_STD
                and slm_stats["slm1_circular_std"] >= Config.PHASE_BEST_MIN_CIRCULAR_STD
                and slm_stats["slm2_circular_std"] >= Config.PHASE_BEST_MIN_CIRCULAR_STD
                and slm_stats["slm1_near_boundary_ratio"] <= Config.PHASE_BEST_MAX_NEAR_BOUNDARY_RATIO
                and slm_stats["slm2_near_boundary_ratio"] <= Config.PHASE_BEST_MAX_NEAR_BOUNDARY_RATIO
                and slm_stats["slm1_wrapped_span"] >= Config.PHASE_BEST_MIN_SPAN
                and slm_stats["slm2_wrapped_span"] >= Config.PHASE_BEST_MIN_SPAN
            )
            val_losses = None
            val_metrics = None
            if val_loader is not None and ((global_epoch + 1) % Config.VAL_INTERVAL == 0):
                val_losses, val_metrics = evaluate_slm_detector(
                    Config,
                    teacher,
                    student,
                    detector,
                    val_loader,
                    detection_criterion,
                    feature_criterion,
                    device,
                    stage_name,
                )
                history["val_total"].append(val_losses["total"])
                history["precision"].append(val_metrics["precision"])
                history["recall"].append(val_metrics["recall"])
                history["f1"].append(val_metrics["f1"])
                history["map50"].append(val_metrics["map50"])
            else:
                for key in ("val_total", "precision", "recall", "f1", "map50"):
                    history[key].append(np.nan)

            student_score_is_best = False
            if norm_is_deployment_ready and slm_ok and stage_name == "student_adapt_max":
                student_score_is_best = avg_feature < best_student_loss
            elif norm_is_deployment_ready and slm_ok and stage_name == "joint_balance":
                if val_metrics is not None:
                    student_score_is_best = val_metrics["map50"] > best_student_map50 + Config.DETECTOR_EARLY_STOP_MIN_DELTA
                else:
                    student_score_is_best = avg_total < best_student_loss
            if student_score_is_best:
                best_student_loss = avg_total
                if val_metrics is not None:
                    best_student_map50 = val_metrics["map50"]
                log_to_file(
                    Config,
                    f"Tracked best normalized SLM student candidate: epoch={display_epoch}, train_loss={avg_total:.6f}, "
                    f"val_loss={val_losses['total']:.6f}, map50={best_student_map50:.4f}" if val_metrics is not None else f"Tracked best normalized SLM student candidate: epoch={display_epoch}, train_loss={avg_total:.6f}",
                )
            elif stage_name == "student_only" and not norm_is_deployment_ready:
                log_to_file(
                    Config,
                    "Student-only phase was trained without deployment normalization; standalone "
                    "optical_student_best.pth is deferred until normalized detector/joint training.",
                )

            detector_score_is_best = False
            if stage_name in {"detector_only", "joint_balance"} and val_metrics is not None and val_metrics["map50"] > best_map50 + Config.DETECTOR_EARLY_STOP_MIN_DELTA:
                best_map50 = val_metrics["map50"]
                detector_score_is_best = True
                detector_no_improve = 0
            elif val_metrics is not None and stage_name == "detector_only":
                detector_no_improve += 1
            elif val_metrics is None and avg_total < best_detector_loss:
                detector_score_is_best = True
            if stage_name in {"detector_only", "joint_balance"} and detector_score_is_best:
                best_detector_loss = avg_total
                save_detector_best(
                    detector,
                    Config.get_detector_best_path(),
                    display_epoch,
                    avg_total,
                    extra={
                        "train_loss": avg_total,
                        "val_loss": val_losses["total"] if val_losses is not None else None,
                        "val_map50": best_map50 if val_metrics is not None else None,
                        "selection_metric": "detector_val_map50" if val_metrics is not None else "detector_train_loss",
                        "recommended_inference_checkpoint": True,
                        "paired_student_source": "student_state_dict",
                        "paired_student_epoch": display_epoch,
                        "paired_student_stage": stage_name,
                        "student_norm_mode": Config.STUDENT_NORM_MODE,
                        "student_norm_schedule": Config.STUDENT_NORM_SCHEDULE,
                        "slm_stats": slm_stats,
                        "slm_quality_passed": bool(slm_ok),
                    },
                    student=student,
                    config=Config,
                )
                save_student_best(
                    Config,
                    student,
                    Config.get_student_best_path(),
                    display_epoch,
                    avg_total,
                    extra={
                        "train_loss": avg_total,
                        "val_loss": val_losses["total"] if val_losses is not None else None,
                        "slm_stats": slm_stats,
                        "val_map50": best_map50 if val_metrics is not None else None,
                        "paired_with_detector_best": True,
                        "paired_detector_checkpoint": Config.get_detector_best_path(),
                        "paired_student_epoch": display_epoch,
                        "paired_student_stage": stage_name,
                        "selection_metric": "paired_detector_val_map50" if val_metrics is not None else "paired_detector_train_loss",
                        "recommended_inference_checkpoint": False,
                        "student_norm_mode": Config.STUDENT_NORM_MODE,
                        "student_norm_schedule": Config.STUDENT_NORM_SCHEDULE,
                        "slm_quality_passed": bool(slm_ok),
                    },
                )
                best_student_map50 = max(best_student_map50, best_map50)
                log_to_file(
                    Config,
                    f"Saved best detector + paired SLM student: epoch={display_epoch}, train_loss={avg_total:.6f}, "
                    f"val_loss={val_losses['total']:.6f}, map50={best_map50:.4f}, slm_quality_passed={slm_ok}" if val_metrics is not None else f"Saved best detector + paired SLM student: epoch={display_epoch}, train_loss={avg_total:.6f}, slm_quality_passed={slm_ok}",
                )

            if global_epoch % Config.VIS_INTERVAL == 0:
                save_slm_detection_visualization(
                    Config,
                    global_epoch,
                    teacher,
                    student,
                    detector,
                    vis_dataset,
                    Config.VISUALIZATION_DIR,
                    prefix=vis_prefix,
                    device=device,
                )
            save_training_curves(history, Config.OUTPUT_DIR)

            log_to_file(
                Config,
                f"Epoch {display_epoch:03d} [{stage_name}] total={avg_total:.6f} "
                f"feature={avg_feature:.6f} detection={avg_detection:.6f} response={avg_response:.6f} "
                f"slm_std={slm_stats['slm1_wrapped_std']:.3f}/{slm_stats['slm2_wrapped_std']:.3f} "
                f"slm_circ={slm_stats['slm1_circular_std']:.3f}/{slm_stats['slm2_circular_std']:.3f} "
                f"slm_near={slm_stats['slm1_near_boundary_ratio']:.3f}/{slm_stats['slm2_near_boundary_ratio']:.3f} "
                f"slm_span={slm_stats['slm1_wrapped_span']:.3f}/{slm_stats['slm2_wrapped_span']:.3f}"
            )
            log_epoch_table_row(
                Config,
                epoch=global_epoch,
                phase=stage_name,
                train_loss=avg_total,
                val_loss=val_losses["total"] if val_losses is not None else None,
                precision=val_metrics["precision"] if val_metrics is not None else None,
                recall=val_metrics["recall"] if val_metrics is not None else None,
                f1_score=val_metrics["f1"] if val_metrics is not None else None,
                map50=val_metrics["map50"] if val_metrics is not None else None,
                lr=max(group["lr"] for group in optimizer.param_groups),
                best_status=Config.EPOCH_TABLE_BEST_MARK if detector_score_is_best else "",
            )
            global_epoch += 1
            if (
                stage_name == "detector_only"
                and Config.DETECTOR_EARLY_STOP_PATIENCE > 0
                and detector_no_improve >= Config.DETECTOR_EARLY_STOP_PATIENCE
            ):
                log_to_file(
                    Config,
                    f"Early stopping detector_only after {detector_no_improve} epochs without mAP50 improvement. "
                    f"Best mAP50={best_map50:.4f}.",
                )
                break

    if not os.path.exists(Config.get_detector_best_path()):
        fallback_loss = history["train_total"][-1] if history["train_total"] else 0.0
        slm_stats = collect_slm_statistics(student)
        save_detector_best(
            detector,
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
                "student_norm_mode": Config.STUDENT_NORM_MODE,
                "student_norm_schedule": Config.STUDENT_NORM_SCHEDULE,
                "slm_stats": slm_stats,
                "slm_quality_passed": False,
            },
            student=student,
            config=Config,
        )
        save_student_best(
            Config,
            student,
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
    if not os.path.exists(Config.get_student_best_path()):
        slm_stats = collect_slm_statistics(student)
        log_to_file(
            Config,
            "No paired optical_student_best.pth was saved because detector_best.pth was never updated and fallback "
            f"student mirror creation failed. Last stats: {slm_stats}",
        )
    save_training_curves(history, Config.OUTPUT_DIR)
    log_to_file(Config, "=" * 80)
    log_to_file(Config, "Training complete")
    log_to_file(Config, f"Recommended inference checkpoint: {Config.get_detector_best_path()}")
    log_to_file(Config, f"Paired SLM student mirror for phase extraction: {Config.get_student_best_path()}")


if __name__ == "__main__":
    train()
