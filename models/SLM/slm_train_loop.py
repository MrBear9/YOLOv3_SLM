"""SLM student training main loop.

Contains the train() function for optical_slm_yolov8_head.py.
Supporting helpers live in models/SLM/slm_utils.py.
"""

import os
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.SLM.config_slm import ConfigSLM as Config
from models.SLM.dataset_slm import SLMFeatureDataset, slm_collate_fn
from models.SLM.evaluation_slm import evaluate_slm_detector, save_slm_detection_visualization
from models.SLM.losses_slm import CompositeOpticalFeatureLoss, detection_response_loss, input_privacy_loss
from models.SLM.optical_layers import OpticalStudent
from models.SLM.slm_utils import (
    build_stage_scheduler,
    collect_phase_grad_norm,
    collect_phase_snapshot,
    collect_phase_update_norm,
    clip_phase_grad_norm,
    configure_backends,
    configure_student_norm_for_stage,
    log_config,
    prepare_batch,
    save_current_student_checkpoint,
    save_slm_component_curves,
    stage_schedule,
    write_slm_tensorboard_scalars,
)
from models.SLM.utils_slm import (
    build_stage_optimizer,
    collect_slm_statistics,
    load_student_checkpoint,
    load_teacher_detector_checkpoint,
    save_detector_best,
    save_student_best,
    set_trainable,
)
from models.runtime import (
    cleanup_distributed,
    get_runtime_device,
    init_distributed_mode,
    init_epoch_log_table,
    init_log_file,
    log_epoch_table_row,
    log_to_file,
    wrap_data_parallel,
)
from models.teacher import build_teacher
from models.training_utils import create_tensorboard_writer, save_training_curves
from models.yolov8.head_v8 import build_detector_head
from models.yolov8.loss_anchor_v8 import YOLOv3AnchorLossForV8Head

warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")


def train():
    local_rank, use_ddp = init_distributed_mode(Config)
    is_main = local_rank == 0

    Config.initialize()
    init_log_file(Config)
    configure_backends()
    log_config()
    device = get_runtime_device(Config)

    teacher = build_teacher(Config).to(device)
    reference_detector = build_detector_head(Config, in_channels=1).to(device)
    checkpoint_info = load_teacher_detector_checkpoint(teacher, reference_detector, Config.TEACHER_DETECTOR_CHECKPOINT, device)
    log_to_file(Config, f"Loaded teacher/detector checkpoint: {checkpoint_info}")
    set_trainable(teacher, False)
    set_trainable(reference_detector, False)
    teacher.eval()
    reference_detector.eval()

    student = OpticalStudent(Config).to(device)
    init_mode = str(Config.SLM_INIT_MODE).strip().lower()
    if init_mode in {"checkpoint", "vortex_checkpoint", "double_helix_checkpoint", "dh_psf_checkpoint"}:
        student_info = load_student_checkpoint(student, Config.SLM_INIT_CHECKPOINT, device)
        log_to_file(Config, f"Initialized SLM student from checkpoint: {student_info}")
        if student_info["loaded"] == 0:
            log_to_file(Config, "SLM checkpoint initialization loaded 0 tensors; using current initialized phases.")
    else:
        log_to_file(Config, f"Initialized SLM student with mode={init_mode}")
    detector = build_detector_head(Config, in_channels=1).to(device)
    detector.load_state_dict(reference_detector.state_dict(), strict=False)
    if Config.ENABLE_CHANNELS_LAST and torch.cuda.is_available():
        student = student.to(memory_format=torch.channels_last)
        detector = detector.to(memory_format=torch.channels_last)

    # 保留原始模型引用（用于优化器构建和评估函数）
    student_raw = student
    detector_raw = detector

    # DDP 包装可训练模型
    student = wrap_data_parallel(Config, student, module_name="OpticalStudent", find_unused_parameters=False)
    detector = wrap_data_parallel(Config, detector, module_name="Detector", find_unused_parameters=False)

    train_dataset = SLMFeatureDataset(Config, split="train")
    train_sampler = None
    if use_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        log_to_file(Config, f"Using DistributedSampler for DDP training")
    loader_kwargs = {
        "batch_size": Config.BATCH_SIZE,
        "shuffle": train_sampler is None,
        "sampler": train_sampler,
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
    val_sampler = None
    try:
        val_dataset = SLMFeatureDataset(Config, split="val")
        if len(val_dataset) > 0:
            if use_ddp:
                from torch.utils.data.distributed import DistributedSampler
                val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
            val_kwargs = {
                "batch_size": Config.BATCH_SIZE,
                "shuffle": False,
                "sampler": val_sampler,
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
    history = {
        "train_total": [],
        "train_feature": [],
        "train_detection": [],
        "train_response": [],
        "train_privacy": [],
        "val_total": [],
        "val_feature": [],
        "val_detection": [],
        "val_response": [],
        "val_privacy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "map50": [],
        "precision_op": [],
        "recall_op": [],
        "f1_op": [],
    }
    global_epoch = 0
    deployment_norm_mode = Config.STUDENT_NORM_MODE
    init_epoch_log_table(Config)
    tensorboard_writer = create_tensorboard_writer(Config, Config.OUTPUT_DIR, log_to_file) if is_main else None

    for stage_name, stage_epochs in stage_schedule():
        if stage_epochs <= 0:
            continue
        stage_norm_mode = configure_student_norm_for_stage(stage_name, deployment_norm_mode)
        stage_weights = Config.get_stage_loss_weights(stage_name)
        student_raw.enable_norm = bool(Config.ENABLE_STUDENT_NORM and stage_norm_mode != "none")
        if stage_name == "phase_focus":
            set_trainable(student, True)
            set_trainable(detector, False)
        elif stage_name == "detector_focus":
            set_trainable(student, False)
            set_trainable(detector, True)
        else:
            set_trainable(student, True)
            set_trainable(detector, True)
        optimizer = build_stage_optimizer(Config, student_raw, detector_raw, stage_name)
        scheduler = build_stage_scheduler(optimizer, stage_epochs)
        norm_is_deployment_ready = Config.ENABLE_STUDENT_NORM and stage_norm_mode != "none"
        log_to_file(
            Config,
            f"Start stage={stage_name}, epochs={stage_epochs}, student_norm_mode={stage_norm_mode}, "
            f"student_enable_norm={student_raw.enable_norm}, "
            f"scheduler={Config.LR_SCHEDULER if scheduler is not None else 'none'}",
        )
        detector_no_improve = 0

        for _ in range(stage_epochs):
            if use_ddp and train_sampler is not None:
                train_sampler.set_epoch(global_epoch)
            student.train(stage_name != "detector_focus")
            detector.train(stage_name != "phase_focus")
            epoch_total_t = torch.zeros((), device=device)
            epoch_feature_t = torch.zeros((), device=device)
            epoch_detection_t = torch.zeros((), device=device)
            epoch_response_t = torch.zeros((), device=device)
            epoch_privacy_t = torch.zeros((), device=device)
            phase_snapshot = collect_phase_snapshot(student_raw)
            epoch_phase_grad_norm = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {global_epoch + 1}/{Config.EPOCHS} [{stage_name}]", leave=True, disable=not is_main):
                gray, rgb, targets = prepare_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                teacher_feature = None
                if stage_weights["feature"] > 0 or stage_weights["response"] > 0:
                    with torch.no_grad():
                        teacher_feature = teacher(rgb)
                if stage_name == "detector_focus":
                    with torch.no_grad():
                        student_feature = student_raw(gray)
                else:
                    student_feature = student(gray)

                zero = torch.zeros((), device=device, dtype=student_feature.dtype)
                if stage_weights["feature"] > 0:
                    feature_loss, _ = feature_criterion(student_feature, teacher_feature.detach(), student_raw, stage_name=stage_name)
                else:
                    feature_loss = zero
                if stage_weights["response"] > 0:
                    response_loss, _ = detection_response_loss(Config, reference_detector, student_feature, teacher_feature.detach())
                else:
                    response_loss = zero
                if stage_weights["privacy"] > 0:
                    privacy_loss, _ = input_privacy_loss(Config, student_feature, gray)
                else:
                    privacy_loss = zero
                if stage_weights["detection"] > 0:
                    predictions = detector(student_feature)
                    detection_loss, _ = detection_criterion(predictions, targets)
                else:
                    detection_loss = zero

                total_loss = (
                    feature_loss * stage_weights["feature"]
                    + detection_loss * stage_weights["detection"]
                    + response_loss * stage_weights["response"]
                    + privacy_loss * stage_weights["privacy"]
                )

                total_loss.backward()
                phase_grad_norm = collect_phase_grad_norm(student_raw)
                if stage_name in {"joint_fit", "norm_joint"}:
                    clipped_norm = clip_phase_grad_norm(student_raw, Config.PHASE_GRAD_CLIP_NORM)
                    if clipped_norm is not None:
                        phase_grad_norm = min(clipped_norm, Config.PHASE_GRAD_CLIP_NORM)
                epoch_phase_grad_norm += phase_grad_norm
                optimizer.step()
                epoch_total_t += total_loss.detach()
                epoch_feature_t += feature_loss.detach()
                epoch_detection_t += detection_loss.detach()
                epoch_response_t += response_loss.detach()
                epoch_privacy_t += privacy_loss.detach()

            if scheduler is not None:
                scheduler.step()

            num_batches = max(len(train_loader), 1)
            avg_total = float(epoch_total_t.item()) / num_batches
            avg_feature = float(epoch_feature_t.item()) / num_batches
            avg_detection = float(epoch_detection_t.item()) / num_batches
            avg_response = float(epoch_response_t.item()) / num_batches
            avg_privacy = float(epoch_privacy_t.item()) / num_batches
            avg_phase_grad_norm = epoch_phase_grad_norm / num_batches
            phase_update_norm, phase_update_rel = collect_phase_update_norm(student_raw, phase_snapshot)
            history["train_total"].append(avg_total)
            history["train_feature"].append(avg_feature)
            history["train_detection"].append(avg_detection)
            history["train_response"].append(avg_response)
            history["train_privacy"].append(avg_privacy)
            display_epoch = global_epoch + 1
            slm_stats = collect_slm_statistics(student_raw)
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
                    response_detector=reference_detector,
                )
                history["val_total"].append(val_losses["total"])
                history["val_feature"].append(val_losses["feature"])
                history["val_detection"].append(val_losses["detection"])
                history["val_response"].append(val_losses["response"])
                history["val_privacy"].append(val_losses["privacy"])
                history["precision"].append(val_metrics["precision"])
                history["recall"].append(val_metrics["recall"])
                history["f1"].append(val_metrics["f1"])
                history["map50"].append(val_metrics["map50"])
                history["precision_op"].append(val_metrics["precision_op"])
                history["recall_op"].append(val_metrics["recall_op"])
                history["f1_op"].append(val_metrics["f1_op"])
            else:
                for key in ("val_total", "val_feature", "val_detection", "val_response", "val_privacy",
                            "precision", "recall", "f1", "map50",
                            "precision_op", "recall_op", "f1_op"):
                    history[key].append(np.nan)
            current_lr = max(group["lr"] for group in optimizer.param_groups)

            if is_main:
                write_slm_tensorboard_scalars(
                    tensorboard_writer,
                    display_epoch,
                    stage_name,
                    {
                        "total": avg_total,
                        "feature": avg_feature,
                        "detection": avg_detection,
                        "response": avg_response,
                        "privacy": avg_privacy,
                    },
                    val_losses=val_losses,
                    val_metrics=val_metrics,
                    slm_stats=slm_stats,
                    phase_grad_norm=avg_phase_grad_norm,
                    phase_update_norm=phase_update_norm,
                    phase_update_rel=phase_update_rel,
                    lr=current_lr,
                )

            if is_main:
                save_current_student_checkpoint(
                    student_raw,
                    Config.get_student_current_path(),
                    display_epoch,
                    avg_total,
                    stage_name,
                    slm_stats,
                    slm_ok,
                    Config,
                    val_losses=val_losses,
                    val_metrics=val_metrics,
                    phase_grad_norm=avg_phase_grad_norm,
                    phase_update_norm=phase_update_norm,
                    phase_update_rel=phase_update_rel,
                    mirror_path=Config.get_student_best_path() if stage_name == "phase_focus" else None,
                )

            student_score_is_best = False
            if norm_is_deployment_ready and slm_ok and stage_name == "norm_joint":
                if val_metrics is not None:
                    student_score_is_best = val_metrics["map50"] > best_student_map50 + Config.DETECTOR_FOCUS_EARLY_STOP_MIN_DELTA
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
            elif stage_name == "phase_focus" and not norm_is_deployment_ready:
                log_to_file(
                    Config,
                    "Phase-focus stage was trained without deployment normalization; optical_student_current.pth "
                    "and a phase-focus optical_student_best.pth mirror were saved for phase inspection.",
                )

            detector_score_is_best = False
            detector_stages = {"detector_focus", "joint_fit", "norm_joint"}
            if stage_name in detector_stages and val_metrics is not None and val_metrics["map50"] > best_map50 + Config.DETECTOR_FOCUS_EARLY_STOP_MIN_DELTA:
                best_map50 = val_metrics["map50"]
                detector_score_is_best = True
                detector_no_improve = 0
            elif val_metrics is not None and stage_name == "detector_focus":
                detector_no_improve += 1
            elif stage_name in detector_stages and val_metrics is None and avg_total < best_detector_loss:
                detector_score_is_best = True
            if stage_name in detector_stages and detector_score_is_best:
                best_detector_loss = avg_total
                if is_main:
                    save_detector_best(
                        detector_raw,
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
                        student=student_raw,
                        config=Config,
                    )
                    save_student_best(
                        Config,
                        student_raw,
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

            if is_main and global_epoch % Config.VIS_INTERVAL == 0:
                save_slm_detection_visualization(
                    Config,
                    global_epoch,
                    teacher,
                    student_raw,
                    detector_raw,
                    vis_dataset,
                    Config.VISUALIZATION_DIR,
                    prefix=vis_prefix,
                    device=device,
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
                lr=current_lr,
                best_status=Config.EPOCH_TABLE_BEST_MARK if detector_score_is_best else "",
            )
            # 每个 epoch 结束后同步所有 rank
            if use_ddp:
                torch.distributed.barrier()
            global_epoch += 1
            if (
                stage_name == "detector_focus"
                and Config.ENABLE_DETECTOR_FOCUS_EARLY_STOP
                and Config.DETECTOR_FOCUS_EARLY_STOP_PATIENCE > 0
                and detector_no_improve >= Config.DETECTOR_FOCUS_EARLY_STOP_PATIENCE
            ):
                log_to_file(
                    Config,
                    f"Early stopping detector_focus after {detector_no_improve} epochs without mAP50 improvement. "
                    f"Best mAP50={best_map50:.4f}.",
                )
                break

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
    if is_main:
        save_training_curves(history, Config.OUTPUT_DIR)
        save_slm_component_curves(history, Config.OUTPUT_DIR)
        if tensorboard_writer is not None:
            tensorboard_writer.close()
    log_to_file(Config, "=" * 80)
    log_to_file(Config, "Training complete")
    log_to_file(Config, f"Recommended inference checkpoint: {Config.get_detector_best_path()}")
    log_to_file(Config, f"Paired SLM student mirror for phase extraction: {Config.get_student_best_path()}")

    if use_ddp:
        cleanup_distributed()
