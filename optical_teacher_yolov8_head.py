import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dataset import YOLODataset, build_class_balanced_train_sampler, identity_collate
from models.runtime import (
    append_plain_log,
    get_dataloader_kwargs,
    get_runtime_device,
    init_epoch_log_table,
    init_log_file,
    log_epoch_table_row,
    log_to_file,
    prepare_batch,
    unwrap_module,
    wrap_data_parallel,
)
from models.teacher import build_teacher
from models.teacher_guidance import compute_teacher_guidance_loss, compute_teacher_guidance_loss_v3, build_feature_distillation_loss
from models.training_utils import (
    build_optimizer_from_model,
    initialize_teacher_weights,
    save_training_curves,
    set_detector_trainable,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.yolov8.config_v8 import ConfigYOLOv8Anchor as Config
from models.yolov8.head_v8 import TeacherWithYOLOv8AnchorDetector, YOLOv8AnchorHead, build_detector_head
from models.yolov8.loss_anchor_v8 import YOLOv3AnchorLossForV8Head
from models.yolov8.metrics_anchor_v8 import evaluate_model_anchor_v8
from models.yolov8.visualization_anchor_v8 import save_detection_visualization_anchor_v8


def bootstrap_runtime():
    Config.initialize()
    Config.print_config()
    init_log_file(Config)
    log_to_file(Config, f"Log file path: {Config.LOG_FILE}")
    log_to_file(Config, f"Visualization save path: {Config.TEACHER_OUTPUT_DIR}")
    log_to_file(Config, f"Class info: {Config.CLASS_NAMES}, Num classes: {Config.NUM_CLASSES}")


def log_all_parameters():
    log_to_file(Config, "=" * 80)
    log_to_file(Config, "Optical teacher YOLOv8-style head + YOLOv3 anchor loss configuration")
    log_to_file(Config, "=" * 80)
    log_to_file(Config, f"Dataset: {Config.YAML_PATH}")
    log_to_file(Config, f"Classes: {Config.CLASS_NAMES}")
    log_to_file(Config, f"Image size / batch / epochs: {Config.IMG_SIZE} / {Config.BATCH_SIZE} / {Config.EPOCHS}")
    log_to_file(Config, f"Head: YOLOv8 style C2f/PAN, anchor-formatted output, base_ch={Config.YOLOV8_BASE_CHANNELS}, c2f_blocks={Config.YOLOV8_C2F_BLOCKS}")
    log_to_file(Config, f"Strides: {Config.STRIDES}")
    log_to_file(Config, f"Anchor source: {Config.ANCHOR_SOURCE}")
    log_to_file(Config, f"Anchors: {Config.ANCHORS}")
    log_to_file(Config, f"Loss weights box/obj/noobj/cls: {Config.BOX_WEIGHT_BASE}/{Config.OBJ_WEIGHT_BASE}/{Config.NOOBJ_WEIGHT_BASE}/{Config.CLS_WEIGHT_BASE}")
    log_to_file(Config, f"Focal alpha/gamma: {Config.FOCAL_ALPHA}/{Config.FOCAL_GAMMA}")
    log_to_file(Config, f"Anchor matching: ratio_thresh={Config.ANCHOR_MATCH_RATIO_THRESH}, "
                f"neighbor_cells={Config.ASSIGN_NEIGHBOR_CELLS}")
    log_to_file(Config, f"Hard negative mining: ratio={Config.HARD_NEG_RATIO}, min={Config.HARD_NEG_MIN}")
    log_to_file(Config, f"Box decode range: {Config.BOX_DECODE_RANGE}")
    log_to_file(Config, f"LR teacher/detector: {Config.PHASE1_TEACHER_LR}/{Config.PHASE1_DETECTOR_LR} -> {Config.PHASE2_TEACHER_LR}/{Config.PHASE2_DETECTOR_LR} -> {Config.PHASE3_TEACHER_LR}/{Config.PHASE3_DETECTOR_LR}")
    log_to_file(Config, f"Detection conf/nms/max_det: {Config.CONF_THRESH}/{Config.NMS_THRESH}/{Config.MAX_DET}")
    log_to_file(Config, f"Metric conf/nms/max_det: {Config.METRIC_CONF_THRESH}/{Config.METRIC_NMS_THRESH}/{Config.METRIC_MAX_DET}")
    log_to_file(Config, f"Output: {Config.TEACHER_OUTPUT_DIR}")
    teacher = build_teacher(Config)
    detector = build_detector_head(Config, in_channels=1, out_channels=Config.get_detector_output_channels())
    log_to_file(Config, f"Teacher arch: {Config.TEACHER_ARCH}")
    log_to_file(Config, f"Teacher parameters: {sum(p.numel() for p in teacher.parameters() if p.requires_grad):,}")
    log_to_file(Config, f"Detector head type: {Config.DETECTOR_HEAD_TYPE}")
    log_to_file(Config, f"Detector parameters: {sum(p.numel() for p in detector.parameters() if p.requires_grad):,}")
    arch_lower = str(Config.TEACHER_ARCH).strip().lower()
    if arch_lower in {"convteacher_v3", "v3"}:
        log_to_file(Config, f"V3 residual_scale={Config.TEACHER_V3_RESIDUAL_SCALE}, "
                    f"bg_identity_weight={Config.TEACHER_V3_BG_IDENTITY_WEIGHT}, "
                    f"grad_consistency_weight={Config.TEACHER_V3_GRAD_CONSISTENCY_WEIGHT}")
    elif arch_lower in {"convteacher_unet", "unet"}:
        log_to_file(Config, f"UNet base_ch={Config.TEACHER_UNET_BASE_CHANNELS}, "
                    f"residual_scale={Config.TEACHER_UNET_RESIDUAL_SCALE}, "
                    f"bg_identity_weight={Config.TEACHER_V3_BG_IDENTITY_WEIGHT}, "
                    f"grad_consistency_weight={Config.TEACHER_V3_GRAD_CONSISTENCY_WEIGHT}")
    if getattr(Config, "ENABLE_FEATURE_DISTILL", False):
        log_to_file(Config, f"Feature distillation: weight={Config.FEATURE_DISTILL_WEIGHT}")
    log_to_file(Config, "=" * 80)


def train():
    bootstrap_runtime()
    log_all_parameters()
    device = get_runtime_device(Config)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = Config.ENABLE_CUDNN_BENCHMARK
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = Config.ENABLE_TF32
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = Config.ENABLE_TF32

    log_to_file(Config, f"Using device: {device}")
    teacher = build_teacher(Config)
    loaded_teacher, teacher_message = initialize_teacher_weights(Config, teacher, device)
    log_to_file(Config, teacher_message)
    freeze_teacher = Config.FREEZE_TEACHER and loaded_teacher
    for p in teacher.parameters():
        p.requires_grad = not freeze_teacher
    log_to_file(Config, f"Teacher status: {'frozen' if freeze_teacher else 'trainable'}")

    arch_lower = str(Config.TEACHER_ARCH).strip().lower()
    is_v3 = arch_lower in {"convteacher_v3", "v3", "convteacher_unet", "unet"}
    if is_v3:
        log_to_file(Config, f"Residual+gate teacher ({arch_lower}): "
                    f"bg_identity_weight={Config.TEACHER_V3_BG_IDENTITY_WEIGHT}, "
                    f"grad_consistency_weight={Config.TEACHER_V3_GRAD_CONSISTENCY_WEIGHT}")

    detector = build_detector_head(Config, in_channels=1, out_channels=Config.get_detector_output_channels())
    model = wrap_data_parallel(Config, TeacherWithYOLOv8AnchorDetector(Config, teacher=teacher, detector=detector), module_name="TeacherWithDetector")
    set_detector_trainable(model, True)

    distill_loss_fn = None
    enable_distill = bool(getattr(Config, "ENABLE_FEATURE_DISTILL", False))
    if enable_distill and is_v3:
        distill_loss_fn = build_feature_distillation_loss(Config).to(device)
        log_to_file(Config, f"Feature distillation enabled, weight={Config.FEATURE_DISTILL_WEIGHT}")

    train_dataset = YOLODataset(Config, split="train")
    train_sampler = None
    if Config.USE_CLASS_BALANCED_SAMPLER:
        train_sampler, sampler_summary = build_class_balanced_train_sampler(Config, train_dataset)
        log_to_file(Config, f"Class balanced sampler: {sampler_summary}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        collate_fn=identity_collate,
        **get_dataloader_kwargs(Config, shuffle=True, sampler=train_sampler),
    )

    val_loader = None
    val_dataset = None
    try:
        val_dataset = YOLODataset(Config, split="val")
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                collate_fn=identity_collate,
                **get_dataloader_kwargs(Config, shuffle=False),
            )
    except Exception as exc:
        log_to_file(Config, f"Validation dataset unavailable: {exc}")

    criterion = YOLOv3AnchorLossForV8Head(Config)
    vis_dataset = val_dataset if Config.VIS_DATASET_SPLIT == "val" and val_dataset is not None and len(val_dataset) > 0 else train_dataset
    vis_prefix = "val" if vis_dataset is val_dataset else "train"
    vis_dir = os.path.join(Config.TEACHER_OUTPUT_DIR, "visualizations")
    teacher_best_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "teacher_best.pth")
    teacher_final_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "teacher_final.pth")
    joint_best_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "teacher_detector_best.pth")
    joint_final_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "teacher_detector_final.pth")

    history = {"train_total": [], "val_total": [], "precision": [], "recall": [], "f1": [], "map50": []}
    best_loss = float("inf")
    best_map50 = -1.0
    current_phase = None
    optimizer = None
    scheduler = None

    log_to_file(Config, "=" * 60)
    log_to_file(Config, "Training YOLOv8-style head with legacy YOLOv3 anchor loss")
    log_to_file(Config, "=" * 60)
    init_epoch_log_table(Config)

    for epoch in range(Config.EPOCHS):
        model.train()
        train_component_sums = {"total": 0.0, "box": 0.0, "obj": 0.0, "noobj": 0.0, "cls": 0.0}
        stage_settings = Config.get_stage_settings(epoch)
        phase = criterion.set_epoch_weights(epoch)
        if phase != current_phase:
            current_phase = phase
            optimizer = build_optimizer_from_model(Config, model, teacher_lr=stage_settings["teacher_lr"], detector_lr=stage_settings["detector_lr"])
            if phase == "locate_gt":
                phase_epochs = Config.STAGE1_LOCATE_EPOCHS
            elif phase == "texture_detail":
                phase_epochs = Config.STAGE2_TEXTURE_EPOCHS
            else:
                phase_epochs = Config.STAGE3_BALANCE_EPOCHS
            scheduler = CosineAnnealingLR(optimizer, T_max=phase_epochs, eta_min=Config.ETA_MIN)
            log_to_file(Config, f"Epoch {epoch}: phase={phase}, teacher_lr={stage_settings['teacher_lr']:.6g}, detector_lr={stage_settings['detector_lr']:.6g}, cosine_T_max={phase_epochs}")

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [{phase}]", leave=True):
            batch_images, batch_targets = prepare_batch(Config, batch, device)
            optimizer.zero_grad()
            if enable_distill and distill_loss_fn is not None:
                teacher_features, predictions, teacher_aux, det_features = model(
                    batch_images, return_feature=True, return_teacher_aux=True, return_det_features=True
                )
            else:
                teacher_features, predictions = model(batch_images, return_feature=True)

            loss, loss_stats = criterion(predictions, batch_targets)
            if is_v3:
                feature_loss, _ = compute_teacher_guidance_loss_v3(Config, teacher_features, batch_images, batch_targets)
            else:
                feature_loss, _ = compute_teacher_guidance_loss(Config, teacher_features, batch_targets, stage_settings=stage_settings)
            loss = loss + feature_loss

            if enable_distill and distill_loss_fn is not None:
                distill_loss, _ = distill_loss_fn(teacher_aux, det_features)
                loss = loss + distill_loss * Config.FEATURE_DISTILL_WEIGHT

            loss.backward()
            optimizer.step()
            train_component_sums["total"] += float(loss.detach().item())
            for key in ("box", "obj", "noobj", "cls"):
                train_component_sums[key] += loss_stats.get(key, 0.0)

        avg_train = {key: value / max(len(train_loader), 1) for key, value in train_component_sums.items()}
        history["train_total"].append(avg_train["total"])
        scheduler.step()
        current_lr = max(group["lr"] for group in optimizer.param_groups)
        val_losses = None
        val_metrics = None
        if val_loader is not None and ((epoch + 1) % Config.VAL_INTERVAL == 0):
            val_losses, val_metrics = evaluate_model_anchor_v8(Config, model, val_loader, criterion, device, stage_settings=stage_settings, is_v3=is_v3)
            history["val_total"].append(val_losses["total"])
            history["precision"].append(val_metrics["precision"])
            history["recall"].append(val_metrics["recall"])
            history["f1"].append(val_metrics["f1"])
            history["map50"].append(val_metrics["map50"])
        else:
            for key in ("val_total", "precision", "recall", "f1", "map50"):
                history[key].append(np.nan)

        is_best = False
        if val_metrics is not None and val_metrics["map50"] > best_map50:
            best_map50 = val_metrics["map50"]
            is_best = True
        elif val_loader is None and avg_train["total"] < best_loss:
            is_best = True

        if is_best:
            if val_metrics is None:
                best_loss = avg_train["total"]
            model_core = unwrap_module(model)
            torch.save(model_core.detector.state_dict(), os.path.join(Config.TEACHER_OUTPUT_DIR, "detector_best.pth"))
            if Config.SAVE_TEACHER_WEIGHTS:
                torch.save(model_core.teacher.state_dict(), teacher_best_path)
                torch.save(
                    {
                        "teacher_state_dict": model_core.teacher.state_dict(),
                        "detector_state_dict": model_core.detector.state_dict(),
                        "epoch": epoch,
                        "loss": avg_train["total"],
                        "val_map50": best_map50 if val_metrics is not None else None,
                        "teacher_arch": Config.TEACHER_ARCH,
                        "head_type": "yolov8_style_head_yolov3_anchor_loss",
                    },
                    joint_best_path,
                )

        if epoch % Config.VIS_INTERVAL == 0:
            save_detection_visualization_anchor_v8(Config, epoch, model, vis_dataset, vis_dir, prefix=vis_prefix, device=device)

        log_epoch_table_row(
            Config,
            epoch=epoch,
            phase=phase,
            train_loss=avg_train["total"],
            val_loss=val_losses["total"] if val_losses is not None else None,
            precision=val_metrics["precision"] if val_metrics is not None else None,
            recall=val_metrics["recall"] if val_metrics is not None else None,
            f1_score=val_metrics["f1"] if val_metrics is not None else None,
            map50=val_metrics["map50"] if val_metrics is not None else None,
            lr=current_lr,
            best_status=Config.EPOCH_TABLE_BEST_MARK if is_best else "",
        )
        save_training_curves(history, Config.TEACHER_OUTPUT_DIR)

    model_core = unwrap_module(model)
    model_save_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "detector_final.pth")
    torch.save(model_core.detector.state_dict(), model_save_path)
    if Config.SAVE_TEACHER_WEIGHTS:
        torch.save(model_core.teacher.state_dict(), teacher_final_path)
        torch.save(
            {
                "teacher_state_dict": model_core.teacher.state_dict(),
                "detector_state_dict": model_core.detector.state_dict(),
                "epoch": Config.EPOCHS - 1,
                "loss": history["train_total"][-1] if history["train_total"] else None,
                "val_map50": best_map50 if best_map50 >= 0 else None,
                "teacher_arch": Config.TEACHER_ARCH,
                "head_type": "yolov8_style_head_yolov3_anchor_loss",
            },
            joint_final_path,
        )

    append_plain_log(Config, Config.get_epoch_table_separator())
    log_to_file(Config, "=" * 60)
    log_to_file(Config, "Training complete")
    log_to_file(Config, f"Best detector model saved to: {os.path.join(Config.TEACHER_OUTPUT_DIR, 'detector_best.pth')}")
    log_to_file(Config, f"Final detector model saved to: {model_save_path}")
    log_to_file(Config, f"Teacher output directory: {Config.TEACHER_OUTPUT_DIR}")
    log_to_file(Config, "=" * 60)


if __name__ == "__main__":
    train()
