"""SLM per-epoch training logic.

Contains run_epoch() which executes one training epoch (with optional
validation) and returns updated best-model tracking values.
"""

from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm

from models.SLM.config_slm import ConfigSLM as Config
from models.SLM.evaluation_slm import evaluate_slm_detector, save_slm_detection_visualization
from models.SLM.losses_slm import detection_response_loss, phase_regularization_loss
from models.SLM.slm_utils import (
    collect_phase_grad_norm,
    collect_phase_grad_norms,
    collect_phase_snapshot,
    collect_phase_update_norm,
    collect_phase_update_norms,
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
from models.monitoring import (
    snapshot_parameters,
    write_confusion_matrix,
    write_gradient_monitoring,
    write_parameter_monitoring,
)
from models.runtime import gather_detection_results, log_epoch_table_row, log_to_file, unwrap_module
from models.yolov8.feature_adapter import prepare_slm_detector_feature


def _collect_slm_val_detections(config, student, detector, val_loader, device):
    """Collect all validation detections and targets for confusion matrix."""
    from models.yolov8.detection_protocol import decode_detections

    student = unwrap_module(student)
    detector = unwrap_module(detector)
    student.eval()
    detector.eval()
    all_dets = []
    all_targets = []
    amp_enabled = bool(getattr(config, "ENABLE_AMP", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(getattr(config, "AMP_DTYPE", "float16")).lower() in {"bf16", "bfloat16"} else torch.float16
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled) if device.type == "cuda" else nullcontext()
    with torch.no_grad():
        for batch in val_loader:
            gray = batch["gray_tensor"].to(device, non_blocking=config.PIN_MEMORY)
            if config.ENABLE_CHANNELS_LAST and torch.cuda.is_available():
                gray = gray.contiguous(memory_format=torch.channels_last)
            with amp_ctx:
                student_feature = student(gray) if hasattr(student, '__call__') else student
                predictions = detector(prepare_slm_detector_feature(config, student_feature))
            detections = decode_detections(
                config, predictions,
                conf_thresh=getattr(config, "METRIC_CONF_THRESH", config.CONF_THRESH),
                nms_thresh=getattr(config, "METRIC_NMS_THRESH", config.NMS_THRESH),
                max_det=getattr(config, "METRIC_MAX_DET", config.MAX_DET),
            )
            targets = batch["targets"]
            for i, dets in enumerate(detections):
                all_dets.append(np.array(dets) if not isinstance(dets, np.ndarray) else dets)
                gt = targets[i]
                all_targets.append(gt.cpu().numpy() if isinstance(gt, torch.Tensor) else gt)
    return all_dets, all_targets


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
    phase_only_stage = stage_name in {"phase_focus", "phase_refine", "phase_coarse", "phase_mid"}
    student.train(stage_name != "detector_focus")
    detector.train(not phase_only_stage)
    epoch_total_t = torch.zeros((), device=device)
    epoch_feature_t = torch.zeros((), device=device)
    epoch_feature_global_t = torch.zeros((), device=device)
    epoch_feature_roi_t = torch.zeros((), device=device)
    epoch_feature_background_t = torch.zeros((), device=device)
    epoch_detection_t = torch.zeros((), device=device)
    epoch_response_t = torch.zeros((), device=device)
    epoch_phase_regularization_t = torch.zeros((), device=device)
    phase_snapshot = collect_phase_snapshot(student_raw)
    epoch_phase_grad_norm = 0.0
    # Use list to accumulate across batches 閳?dict built at epoch end
    _phase_grad_norms_accum = []

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
            feature_loss, feature_stats = feature_criterion(
                student_feature,
                teacher_feature.detach(),
                student_raw,
                stage_name=stage_name,
                targets=targets,
            )
        else:
            feature_loss = zero
            feature_stats = {"feature_global": 0.0, "feature_roi": 0.0, "feature_background": 0.0}
        if stage_weights["response"] > 0:
            response_loss, _ = detection_response_loss(Config, reference_detector, student_feature, teacher_feature.detach())
        else:
            response_loss = zero
        if stage_weights["phase_regularization"] > 0:
            phase_regularization_loss_value, _ = phase_regularization_loss(Config, student_raw)
        else:
            phase_regularization_loss_value = zero
        if stage_weights["detection"] > 0:
            predictions = detector(prepare_slm_detector_feature(Config, student_feature))
            detection_loss, _ = detection_criterion(predictions, targets)
        else:
            detection_loss = zero

        total_loss = (
            feature_loss * stage_weights["feature"]
            + detection_loss * stage_weights["detection"]
            + response_loss * stage_weights["response"]
            + phase_regularization_loss_value * stage_weights["phase_regularization"]
        )

        total_loss.backward()
        _phase_grad_norms_accum.append(collect_phase_grad_norms(student_raw))
        phase_grad_norm = collect_phase_grad_norm(student_raw)
        if stage_name in {"phase_refine", "phase_coarse", "phase_mid", "joint_fit", "norm_joint"}:
            clipped_norm = clip_phase_grad_norm(student_raw, Config.PHASE_GRAD_CLIP_NORM)
            if clipped_norm is not None:
                phase_grad_norm = min(clipped_norm, Config.PHASE_GRAD_CLIP_NORM)
        epoch_phase_grad_norm += phase_grad_norm
        optimizer.step()
        epoch_total_t += total_loss.detach()
        epoch_feature_t += feature_loss.detach()
        epoch_feature_global_t += feature_stats["feature_global"]
        epoch_feature_roi_t += feature_stats["feature_roi"]
        epoch_feature_background_t += feature_stats["feature_background"]
        epoch_detection_t += detection_loss.detach()
        epoch_response_t += response_loss.detach()
        epoch_phase_regularization_t += phase_regularization_loss_value.detach()

    if scheduler is not None:
        scheduler.step()

    num_batches = max(len(train_loader), 1)
    avg_total = float(epoch_total_t.item()) / num_batches
    avg_feature = float(epoch_feature_t.item()) / num_batches
    avg_feature_global = float(epoch_feature_global_t.item()) / num_batches
    avg_feature_roi = float(epoch_feature_roi_t.item()) / num_batches
    avg_feature_background = float(epoch_feature_background_t.item()) / num_batches
    avg_detection = float(epoch_detection_t.item()) / num_batches
    avg_response = float(epoch_response_t.item()) / num_batches
    avg_phase_regularization = float(epoch_phase_regularization_t.item()) / num_batches
    avg_phase_grad_norm = epoch_phase_grad_norm / num_batches

    # Merge per-batch grad norms into epoch-level stats
    from collections import defaultdict
    epoch_phase_grad_norms = defaultdict(float)
    for batch_norms in _phase_grad_norms_accum:
        for k, v in batch_norms.items():
            epoch_phase_grad_norms[k] += v
    for k in epoch_phase_grad_norms:
        epoch_phase_grad_norms[k] /= num_batches

    phase_update_norm, phase_update_rel = collect_phase_update_norm(student_raw, phase_snapshot)
    phase_layer_updates = collect_phase_update_norms(student_raw, phase_snapshot)
    phase_layer_stats = {
        layer_name: {
            "grad_norm": epoch_phase_grad_norms[layer_name],
            **phase_layer_updates[layer_name],
        }
        for layer_name in epoch_phase_grad_norms
    }
    history["train_total"].append(avg_total)
    history["train_feature"].append(avg_feature)
    history["train_detection"].append(avg_detection)
    history["train_response"].append(avg_response)
    history["train_phase_regularization"].append(avg_phase_regularization)
    display_epoch = global_epoch + 1
    slm_stats = collect_slm_statistics(student_raw)
    # Direct-SGD has no preferred phase histogram; diagnostics cannot veto a
    # checkpoint selected by validation detection performance.
    slm_ok = True

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
        history["val_phase_regularization"].append(val_losses["phase_regularization"])
        history["precision"].append(val_metrics["precision"])
        history["recall"].append(val_metrics["recall"])
        history["f1"].append(val_metrics["f1"])
        history["map50"].append(val_metrics["map50"])
        history["precision_op"].append(val_metrics["precision_op"])
        history["recall_op"].append(val_metrics["recall_op"])
        history["f1_op"].append(val_metrics["f1_op"])
    else:
        for key in ("val_total", "val_feature", "val_detection", "val_response", "val_phase_regularization",
                     "precision", "recall", "f1", "map50",
                     "precision_op", "recall_op", "f1_op"):
            history[key].append(np.nan)
    current_lr = max(group["lr"] for group in optimizer.param_groups)
    lr_by_group = {}
    for group in optimizer.param_groups:
        group_name = group.get("slm_group", "detector" if stage_name == "detector_focus" else "other")
        lr_by_group[group_name] = max(lr_by_group.get(group_name, 0.0), float(group["lr"]))

    cm_dets = cm_targets = None
    should_write_cm = val_loader is not None and (global_epoch + 1) % Config.VAL_INTERVAL == 0
    if should_write_cm:
        cm_dets, cm_targets = _collect_slm_val_detections(Config, student, detector, val_loader, device)
        cm_dets, cm_targets = gather_detection_results(cm_dets, cm_targets)

    # TensorBoard & checkpoint
    if is_main:
        write_slm_tensorboard_scalars(
            tensorboard_writer,
            display_epoch,
            stage_name,
            {
                "total": avg_total,
                "feature": avg_feature,
                "feature_global": avg_feature_global,
                "feature_roi": avg_feature_roi,
                "feature_background": avg_feature_background,
                "detection": avg_detection,
                "response": avg_response,
                "phase_regularization": avg_phase_regularization,
            },
            val_losses=val_losses,
            val_metrics=val_metrics,
            slm_stats=slm_stats,
            phase_grad_norm=avg_phase_grad_norm,
            phase_update_norm=phase_update_norm,
            phase_update_rel=phase_update_rel,
            phase_layer_stats=phase_layer_stats,
            lr=current_lr,
            lr_by_group=lr_by_group,
        )
        # 閳光偓閳光偓 濮婎垰瀹?& 閸欏倹鏆熼惄鎴炵ゴ 閳光偓閳光偓
        write_gradient_monitoring(tensorboard_writer, student, display_epoch, prefix="Grad/Student")
        write_gradient_monitoring(tensorboard_writer, detector, display_epoch, prefix="Grad/Detector")
        write_parameter_monitoring(tensorboard_writer, student, display_epoch, prefix="Param/Student")
        write_parameter_monitoring(tensorboard_writer, detector, display_epoch, prefix="Param/Detector")
        # 閳光偓閳光偓 濞ｉ攱绌惌鈺呮█閿涘牊鐦℃稉顏堢崣鐠?epoch閿涘鏀㈤埞鈧?
        if should_write_cm:
            write_confusion_matrix(
                tensorboard_writer, cm_dets, cm_targets,
                Config.NUM_CLASSES, Config.CLASS_NAMES, display_epoch,
                iou_threshold=Config.METRIC_IOU_THRESHOLD,
                conf_threshold=getattr(Config, "CONF_THRESH", 0.35),
                prefix="ConfusionMatrix",
                image_size=Config.RESOLUTION,
            )

    phase_only_checkpoint_due = (
        stage_name in {"phase_focus", "phase_refine", "phase_coarse", "phase_mid"}
        and Config.VIS_INTERVAL > 0
        and display_epoch % Config.VIS_INTERVAL == 0
    )
    if is_main and (stage_name not in {"phase_focus", "phase_refine", "phase_coarse", "phase_mid"} or phase_only_checkpoint_due):
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
            mirror_path=None,
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
    elif phase_only_checkpoint_due and not norm_is_deployment_ready:
        log_to_file(
            Config,
            "Phase-only checkpoint saved for phase inspection without deployment normalization.",
        )

    # Detector best tracking
    detector_score_is_best = False
    detector_no_improve_delta = 0
    detector_stages = {"detector_focus", "phase_refine", "phase_coarse", "phase_mid", "joint_fit", "norm_joint"}
    if stage_name in detector_stages and val_metrics is not None and val_metrics["map50"] > best_map50 + Config.DETECTOR_FOCUS_EARLY_STOP_MIN_DELTA:
        best_map50 = val_metrics["map50"]
        detector_score_is_best = True
    elif val_metrics is not None and stage_name in {"detector_focus", "phase_refine", "phase_coarse", "phase_mid", "joint_fit"}:
        detector_no_improve_delta = 1
    elif (
        stage_name in detector_stages
        and val_loader is None
        and val_metrics is None
        and avg_total < best_detector_loss
    ):
        detector_score_is_best = True
    best_checkpoint_path = None
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
                    "detector_head_type": Config.DETECTOR_HEAD_TYPE,
                    "detection_protocol": "anchor_free_tal",
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
            best_checkpoint_path = Config.get_detector_best_path()
        best_student_map50 = max(best_student_map50, best_map50)

    # Visualization
    if is_main and Config.VIS_INTERVAL > 0 and display_epoch % Config.VIS_INTERVAL == 0:
        save_slm_detection_visualization(
            Config,
            display_epoch,
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
        precision_op=val_metrics["precision_op"] if val_metrics is not None else None,
    )
    if use_ddp:
        torch.distributed.barrier()

    return best_map50, best_student_map50, best_student_loss, best_detector_loss, detector_no_improve_delta, best_checkpoint_path
