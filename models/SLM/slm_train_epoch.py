"""SLM per-epoch training logic.

Contains run_epoch() which executes one training epoch (with optional
validation) and returns updated best-model tracking values.
"""

import numpy as np
import torch
from tqdm import tqdm

from models.SLM.config_slm import ConfigSLM as Config
from models.SLM.evaluation_slm import evaluate_slm_detector, save_slm_detection_visualization
from models.SLM.losses_slm import detection_response_loss, input_privacy_loss
from models.SLM.slm_utils import (
    collect_phase_grad_norm,
    collect_phase_snapshot,
    collect_phase_update_norm,
    clip_phase_grad_norm,
    prepare_batch,
    save_current_student_checkpoint,
    write_slm_tensorboard_scalars,
)
from models.SLM.utils_slm import (
    collect_slm_statistics,
    save_detector_best,
    save_student_best,
)
from models.runtime import log_epoch_table_row, log_to_file


def run_epoch(
    global_epoch,
    stage_name,
    stage_weights,
    norm_is_deployment_ready,
    ctx,
    best_map50,
    best_student_map50,
    best_student_loss,
    best_detector_loss,
):
    """Run one training epoch. Returns updated best values and detector_no_improve delta."""
    device = ctx["device"]
    student = ctx["student"]
    detector = ctx["detector"]
    student_raw = ctx["student_raw"]
    teacher = ctx["teacher"]
    reference_detector = ctx["reference_detector"]
    train_loader = ctx["train_loader"]
    val_loader = ctx["val_loader"]
    vis_dataset = ctx["vis_dataset"]
    vis_prefix = ctx["vis_prefix"]
    feature_criterion = ctx["feature_criterion"]
    detection_criterion = ctx["detection_criterion"]
    optimizer = ctx["optimizer"]
    scheduler = ctx["scheduler"]
    history = ctx["history"]
    tensorboard_writer = ctx["tensorboard_writer"]
    is_main = ctx["is_main"]
    use_ddp = ctx["use_ddp"]
    train_sampler = ctx["train_sampler"]

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

    # Validation
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

    # TensorBoard & checkpoint
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

    # Student best tracking
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

    # Detector best tracking
    detector_score_is_best = False
    detector_no_improve_delta = 0
    detector_stages = {"detector_focus", "joint_fit", "norm_joint"}
    if stage_name in detector_stages and val_metrics is not None and val_metrics["map50"] > best_map50 + Config.DETECTOR_FOCUS_EARLY_STOP_MIN_DELTA:
        best_map50 = val_metrics["map50"]
        detector_score_is_best = True
    elif val_metrics is not None and stage_name == "detector_focus":
        detector_no_improve_delta = 1
    elif stage_name in detector_stages and val_metrics is None and avg_total < best_detector_loss:
        detector_score_is_best = True
    if stage_name in detector_stages and detector_score_is_best:
        best_detector_loss = avg_total
        if is_main:
            save_detector_best(
                detector_raw := ctx["detector_raw"],
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

    # Visualization
    if is_main and global_epoch % Config.VIS_INTERVAL == 0:
        save_slm_detection_visualization(
            Config,
            global_epoch,
            teacher,
            student_raw,
            ctx["detector_raw"],
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
    if use_ddp:
        torch.distributed.barrier()

    return best_map50, best_student_map50, best_student_loss, best_detector_loss, detector_no_improve_delta
