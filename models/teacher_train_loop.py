"""Teacher training main loop.

Contains the train() function for optical_teacher_yolov8_head.py.
Supporting helpers live in models/teacher_logging.py.
"""

import os
import warnings
from contextlib import nullcontext

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dataset import YOLODataset, build_class_balanced_train_sampler, identity_collate
from models.runtime import (
    append_plain_log,
    cleanup_distributed,
    get_dataloader_kwargs,
    get_runtime_device,
    init_distributed_mode,
    init_epoch_log_table,
    init_log_file,
    log_epoch_table_row,
    log_to_file,
    prepare_batch,
    seed_training,
    unwrap_module,
    wrap_data_parallel,
)
from models.teacher import build_teacher
from models.teacher_guidance import (
    build_feature_distillation_loss,
    teacher_slm_cipher_loss,
)
from models.teacher_logging import bootstrap_runtime, log_all_parameters, write_teacher_tensorboard_scalars
from models.training_utils import (
    add_tensorboard_scalar,
    build_optimizer_from_model,
    create_tensorboard_writer,
    initialize_teacher_weights,
    load_joint_teacher_detector_checkpoint,
    save_training_curves,
    set_detector_trainable,
)
from models.monitoring import (
    snapshot_parameters,
    write_confusion_matrix,
    write_gradient_monitoring,
    write_parameter_monitoring,
)
from models.yolov8.config_v8 import ConfigYOLOv8Anchor as Config
from models.yolov8.head_v8 import TeacherWithDetector, build_detector_head
from models.yolov8.detection_protocol import build_detection_criterion, decode_detections
from models.yolov8.metrics_anchor_v8 import evaluate_model_anchor_v8
from models.yolov8.visualization_anchor_v8 import save_detection_visualization_anchor_v8

# 过滤 DDP 梯度 stride 警告（DDP 内部桶布局与梯度布局的已知差异，不影响正确性和性能）
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")


def _checkpoint_payload(model, epoch, loss_value, val_map50):
    """Build a checkpoint that remains consumable by every downstream stage."""
    model_core = unwrap_module(model)
    teacher = model_core.teacher
    payload = {
        "teacher_state_dict": teacher.state_dict(),
        "detector_state_dict": model_core.detector.state_dict(),
        "epoch": int(epoch),
        "loss": None if loss_value is None else float(loss_value),
        "val_map50": None if val_map50 is None else float(val_map50),
        "teacher_arch": Config.TEACHER_ARCH,
        "head_type": Config.DETECTOR_HEAD_TYPE,
        "detection_protocol": "anchor_free_tal",
        "architecture_revision": getattr(teacher, "architecture_revision", "contextdw_v1"),
        "teacher_v4_base_channels": getattr(Config, "TEACHER_V4_BASE_CHANNELS", 32),
        "teacher_v4_depths": getattr(Config, "TEACHER_V4_DEPTHS", (3, 5, 5, 3)),
        "global_local_context_grid": Config.GLOBAL_LOCAL_CONTEXT_GRID,
        "teacher_depths": Config.GLOBAL_LOCAL_TEACHER_DEPTHS,
        "detector_depths": Config.GLOBAL_LOCAL_DETECTOR_DEPTHS,
    }
    return payload


def _module_param_count(module, trainable_only=False):
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def _write_teacher_tensorboard_model_summary(writer, model):
    if writer is None:
        return
    model_core = unwrap_module(model)
    teacher = model_core.teacher
    detector = model_core.detector
    teacher_lines = [f"- teacher.{name}: {_module_param_count(child):,}" for name, child in teacher.named_children()]
    detector_lines = [f"- detector.{name}: {_module_param_count(child):,}" for name, child in detector.named_children()]
    flow_text = "\n".join([
        "Teacher detection training route",
        "",
        "RGB image -> Teacher -> teacher feature -> detector",
        "",
        "Training output:",
        "teacher feature plus YOLOv8-style detector predictions",
        "",
        f"Teacher architecture: {Config.TEACHER_ARCH}",
        f"Detector head type: {Config.DETECTOR_HEAD_TYPE}",
        f"Input tensor: (B, 1, {Config.RESOLUTION[0]}, {Config.RESOLUTION[1]})",
        f"Detection strides: {Config.ANCHOR_FREE_STRIDES}",
        f"AMP enabled: {Config.ENABLE_AMP}",
        f"Feature distillation enabled: {getattr(Config, 'ENABLE_FEATURE_DISTILL', False)}",
    ])
    param_text = "\n".join([
        f"TeacherWithDetector parameters: {_module_param_count(model_core):,}",
        f"Teacher parameters: {_module_param_count(teacher):,}",
        f"Detector parameters: {_module_param_count(detector):,}",
        "",
        f"Teacher trainable parameters now: {_module_param_count(teacher, trainable_only=True):,}",
        f"Detector trainable parameters now: {_module_param_count(detector, trainable_only=True):,}",
        "",
        "Teacher module parameters:",
        *teacher_lines,
        "",
        "Detector module parameters:",
        *detector_lines,
    ])
    writer.add_text("Model/data_flow", flow_text, 0)
    writer.add_text("Model/parameters", param_text, 0)


def _write_physical_teacher_phase_maps(writer, step, teacher_aux):
    """Record predicted V2 phase maps without retaining their training graph."""
    if writer is None or teacher_aux is None:
        return
    for index, phase_map in enumerate(teacher_aux.get("phase_maps", ()), start=1):
        # add_image expects CHW; retain only the first sample, not its batch axis.
        phase = phase_map[0].detach().float().cpu()
        phase_image = (phase + torch.pi) / (2.0 * torch.pi)
        writer.add_image(f"TeacherV2/phase_map_{index}", phase_image.clamp(0.0, 1.0), step)
        add_tensorboard_scalar(writer, f"TeacherV2/phase_map_{index}_abs_mean", phase.abs().mean(), step)


def _collect_teacher_val_detections(config, model, val_loader, device):
    """Collect all validation detections and targets for confusion matrix."""
    from models.runtime import prepare_batch

    # This helper runs on rank 0 only. Bypass DDP so no unmatched collectives
    # are started while the other ranks wait at the end-of-epoch barrier.
    model = unwrap_module(model)
    model.eval()
    all_dets = []
    all_targets = []
    amp_enabled = bool(getattr(config, "ENABLE_AMP", True)) and device.type == "cuda"
    amp_dtype_name = str(getattr(config, "AMP_DTYPE", "float16")).strip().lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting detections for CM", leave=False, disable=True):
            batch_images, batch_targets = prepare_batch(config, batch, device)
            with amp_ctx:
                _, predictions = model(batch_images, return_feature=True)
            detections = decode_detections(
                config, predictions,
                conf_thresh=getattr(config, "METRIC_CONF_THRESH", config.CONF_THRESH),
                nms_thresh=getattr(config, "METRIC_NMS_THRESH", config.NMS_THRESH),
                max_det=getattr(config, "METRIC_MAX_DET", config.MAX_DET),
            )
            for i, dets in enumerate(detections):
                all_dets.append(np.array(dets) if not isinstance(dets, np.ndarray) else dets)
                all_targets.append(batch_targets[i].cpu().numpy() if isinstance(batch_targets[i], torch.Tensor) else batch_targets[i])
    return all_dets, all_targets


def train():
    local_rank, use_ddp = init_distributed_mode(Config)
    is_main = local_rank == 0

    bootstrap_runtime()
    log_all_parameters()
    seed_training(
        int(getattr(Config, "TRAIN_SEED", 42)) + int(local_rank),
        deterministic=bool(getattr(Config, "DETERMINISTIC_TRAINING", False)),
    )
    log_to_file(
        Config,
        f"Reproducibility: seed={int(getattr(Config, 'TRAIN_SEED', 42))}, "
        f"deterministic={bool(getattr(Config, 'DETERMINISTIC_TRAINING', False))}, "
        f"rank={local_rank}",
    )
    device = get_runtime_device(Config)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = Config.ENABLE_CUDNN_BENCHMARK
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = Config.ENABLE_TF32
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = Config.ENABLE_TF32

    if use_ddp:
        world_size = torch.distributed.get_world_size()
        log_to_file(Config, f"Using device: {device} (local_rank={local_rank}, world_size={world_size}, use_ddp={use_ddp})")
    else:
        log_to_file(Config, f"Using device: {device} (local_rank={local_rank}, use_ddp={use_ddp})")
    amp_enabled = bool(getattr(Config, "ENABLE_AMP", True)) and device.type == "cuda"
    amp_dtype_name = str(getattr(Config, "AMP_DTYPE", "float16")).strip().lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    amp_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    amp_context = (
        lambda: torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
        if device.type == "cuda"
        else nullcontext
    )
    if amp_enabled:
        log_to_file(Config, f"Using AMP autocast dtype={amp_dtype}")
    teacher = build_teacher(Config)

    arch_lower = str(Config.TEACHER_ARCH).strip().lower()
    is_v3 = arch_lower in {"convteacher_v3", "v3"}
    is_physical_v2 = arch_lower in {"convteacher_v2", "v2", "physical_teacher_v4", "v4"}

    detector = build_detector_head(Config, in_channels=1, out_channels=Config.get_detector_output_channels())
    resume_checkpoint = None
    if Config.get_teacher_init_mode() == "joint_checkpoint":
        resume_checkpoint, init_message = load_joint_teacher_detector_checkpoint(
            Config, teacher, detector, Config.get_teacher_init_checkpoint(), device
        )
        if resume_checkpoint is None:
            raise RuntimeError(init_message)
        loaded_teacher = True
        log_to_file(Config, init_message)
    else:
        loaded_teacher, teacher_message = initialize_teacher_weights(Config, teacher, device)
        log_to_file(Config, teacher_message)
    freeze_teacher = Config.FREEZE_TEACHER and loaded_teacher
    for p in teacher.parameters():
        p.requires_grad = not freeze_teacher
    log_to_file(Config, f"Teacher status: {'frozen' if freeze_teacher else 'trainable'}")
    model = wrap_data_parallel(Config, TeacherWithDetector(Config, teacher=teacher, detector=detector), module_name="TeacherWithDetector")
    set_detector_trainable(model, True)

    distill_loss_fn = None
    enable_distill = bool(getattr(Config, "ENABLE_FEATURE_DISTILL", False))
    if enable_distill:
        distill_loss_fn = build_feature_distillation_loss(Config).to(device)
        log_to_file(Config, f"Feature distillation enabled, weight={Config.FEATURE_DISTILL_WEIGHT}")

    train_dataset = YOLODataset(Config, split="train")
    train_sampler = None
    if use_ddp and Config.USE_CLASS_BALANCED_SAMPLER:
        train_sampler, sampler_summary = build_class_balanced_train_sampler(
            Config, train_dataset,
            num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank(),
        )
        log_to_file(Config, f"DDP class balanced sampler: {sampler_summary}")
    elif use_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        log_to_file(Config, f"Using DistributedSampler for DDP training")
    elif Config.USE_CLASS_BALANCED_SAMPLER:
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

    criterion = build_detection_criterion(Config)
    vis_dataset = val_dataset if Config.VIS_DATASET_SPLIT == "val" and val_dataset is not None and len(val_dataset) > 0 else train_dataset
    vis_prefix = "val" if vis_dataset is val_dataset else "train"
    vis_dir = os.path.join(Config.TEACHER_OUTPUT_DIR, "visualizations")
    joint_best_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "teacher_detector_best.pth")
    joint_final_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "teacher_detector_final.pth")

    history = {"train_total": [], "val_total": [], "precision": [], "recall": [], "f1": [], "map50": [], "precision_op": [], "recall_op": [], "f1_op": []}
    best_loss = float("inf")
    best_map50 = float(resume_checkpoint.get("val_map50", -1.0)) if resume_checkpoint is not None else -1.0
    early_stop_map50 = best_map50
    no_improve_epochs = 0
    last_epoch = -1
    current_phase = None
    optimizer = None
    scheduler = None

    log_to_file(Config, "=" * 60)
    log_to_file(Config, f"Training {Config.DETECTOR_HEAD_TYPE} detector with anchor_free_tal protocol")
    log_to_file(Config, "=" * 60)
    init_epoch_log_table(Config)
    tensorboard_writer = create_tensorboard_writer(Config, Config.TEACHER_OUTPUT_DIR, log_to_file) if is_main else None
    if is_main:
        _write_teacher_tensorboard_model_summary(tensorboard_writer, model)
        snapshot_parameters(model)  # 初始化参数快照用于变化追踪

    start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1 if resume_checkpoint is not None else 0
    if resume_checkpoint is not None and is_main:
        checkpoint_path = os.path.abspath(Config.get_teacher_init_checkpoint())
        if checkpoint_path != os.path.abspath(joint_best_path):
            torch.save(resume_checkpoint, joint_best_path)
        log_to_file(Config, f"Protected baseline checkpoint at mAP50={best_map50:.4f}; resuming from epoch {start_epoch}.")

    for epoch in range(start_epoch, Config.EPOCHS):
        last_epoch = epoch
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        model.train()
        last_teacher_aux = None
        train_component_sums = {
            "total": 0.0,
            "box": 0.0,
            "obj": 0.0,
            "noobj": 0.0,
            "cls": 0.0,
            "dfl": 0.0,
            "positive_total": 0.0,
            "positive_small": 0.0,
            "slm_cipher": 0.0,
            "slm_tv": 0.0,
            "slm_hf": 0.0,
            "slm_range": 0.0,
            "slm_mean": 0.0,
            "slm_peak": 0.0,
            "slm_edge": 0.0,
        }
        stage_settings = Config.get_stage_settings(epoch)
        if resume_checkpoint is not None:
            stage_settings = {
                **stage_settings,
                "teacher_lr": Config.JOINT_RESUME_TEACHER_LR,
                "detector_lr": Config.JOINT_RESUME_DETECTOR_LR,
            }
        phase = criterion.set_epoch_weights(epoch)
        if phase != current_phase:
            current_phase = phase
            optimizer = build_optimizer_from_model(Config, model, teacher_lr=stage_settings["teacher_lr"], detector_lr=stage_settings["detector_lr"])
            remaining = Config.EPOCHS - epoch
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=Config.ETA_MIN)
            log_to_file(Config, f"Epoch {epoch}: phase={phase}, teacher_lr={stage_settings['teacher_lr']:.6g}, detector_lr={stage_settings['detector_lr']:.6g}, cosine_T_max={remaining}")

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [{phase}]", leave=True, disable=not is_main):
            batch_images, batch_targets = prepare_batch(Config, batch, device)
            optimizer.zero_grad()
            use_distill = enable_distill and distill_loss_fn is not None
            use_slm_cipher = Config.TEACHER_SLM_CIPHER_LOSS_WEIGHT > 0
            with amp_context():
                if use_distill:
                    teacher_features, predictions, teacher_aux, det_features = model(
                        batch_images, return_feature=True, return_teacher_aux=True, return_det_features=True
                    )
                elif is_v3 or is_physical_v2 or use_slm_cipher:
                    teacher_features, predictions, teacher_aux = model(
                        batch_images, return_feature=True, return_teacher_aux=True
                    )
                else:
                    teacher_features, predictions = model(batch_images, return_feature=True)
                    teacher_aux = None

                if is_physical_v2 and is_main:
                    last_teacher_aux = {"phase_maps": tuple(
                        p[:1].detach() for p in teacher_aux.get("phase_maps", ())
                    )}

                loss, loss_stats = criterion(predictions, batch_targets)

                if use_distill:
                    distill_loss, _ = distill_loss_fn(teacher_aux, det_features)
                    loss = loss + distill_loss * Config.FEATURE_DISTILL_WEIGHT

                if use_slm_cipher:
                    slm_cipher_loss, slm_cipher_stats = teacher_slm_cipher_loss(Config, teacher_aux)
                    loss = loss + slm_cipher_loss
                    for key in ("slm_cipher", "slm_tv", "slm_hf", "slm_range", "slm_mean", "slm_peak", "slm_edge"):
                        train_component_sums[key] += slm_cipher_stats[key]


                if is_v3 and teacher_aux is not None:
                    gate_sparsity = teacher_aux["gate"].mean()
                    residual_l1 = teacher_aux["residual"].abs().mean()
                    output_deviation = ((1.0 - teacher_aux["det_feature"]) - teacher_aux["gray"]).abs().mean()
                    loss = (
                        loss
                        + Config.TEACHER_V3_GATE_SPARSITY_WEIGHT * gate_sparsity
                        + Config.TEACHER_V3_RESIDUAL_L1_WEIGHT * residual_l1
                        + Config.TEACHER_V3_OUTPUT_DEVIATION_WEIGHT * output_deviation
                    )

            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            train_component_sums["total"] += float(loss.detach().item())
            for key in ("box", "obj", "noobj", "cls", "dfl", "positive_total", "positive_small"):
                train_component_sums[key] += loss_stats.get(key, 0.0)
            for cls_id in range(Config.NUM_CLASSES):
                key = f"positive_class_{cls_id}"
                train_component_sums[key] = train_component_sums.get(key, 0.0) + loss_stats.get(key, 0.0)

        avg_train = {key: value / max(len(train_loader), 1) for key, value in train_component_sums.items()}
        copy_paste_stats = train_dataset.get_copy_paste_stats(reset=True)
        avg_train.update({f"copy_paste_{key}": value for key, value in copy_paste_stats.items()})
        history["train_total"].append(avg_train["total"])
        scheduler.step()
        current_lr = max(group["lr"] for group in optimizer.param_groups)
        val_losses = None
        val_metrics = None
        if val_loader is not None and ((epoch + 1) % Config.VAL_INTERVAL == 0):
            val_losses, val_metrics = evaluate_model_anchor_v8(Config, model, val_loader, criterion, device)
            history["val_total"].append(val_losses["total"])
            history["precision"].append(val_metrics["precision"])
            history["recall"].append(val_metrics["recall"])
            history["f1"].append(val_metrics["f1"])
            history["map50"].append(val_metrics["map50"])
            history["precision_op"].append(val_metrics["precision_op"])
            history["recall_op"].append(val_metrics["recall_op"])
            history["f1_op"].append(val_metrics["f1_op"])
        else:
            for key in ("val_total", "precision", "recall", "f1", "map50", "precision_op", "recall_op", "f1_op"):
                history[key].append(np.nan)
        if is_main:
            write_teacher_tensorboard_scalars(
                tensorboard_writer,
                epoch + 1,
                avg_train,
                val_losses=val_losses,
                val_metrics=val_metrics,
                lr=current_lr,
            )
            # ── 梯度 & 参数监测 ──
            write_gradient_monitoring(tensorboard_writer, model, epoch + 1, prefix="Grad")
            write_parameter_monitoring(tensorboard_writer, model, epoch + 1, prefix="Param")
            _write_physical_teacher_phase_maps(tensorboard_writer, epoch + 1, last_teacher_aux)
            # ── 混淆矩阵（每 VIS_INTERVAL 个 epoch 或验证时）──
            if val_loader is not None and (epoch + 1) % max(Config.VIS_INTERVAL, 1) == 0:
                cm_dets, cm_targets = _collect_teacher_val_detections(Config, model, val_loader, device)
                write_confusion_matrix(
                    tensorboard_writer, cm_dets, cm_targets,
                    Config.NUM_CLASSES, Config.CLASS_NAMES, epoch + 1,
                    iou_threshold=Config.METRIC_IOU_THRESHOLD,
                    conf_threshold=getattr(Config, "CONF_THRESH", 0.35),
                    prefix="ConfusionMatrix",
                    image_size=Config.RESOLUTION,
                )

        is_best = False
        significant_improvement = False
        if val_metrics is not None:
            current_map50 = val_metrics["map50"]
            if current_map50 > best_map50:
                best_map50 = current_map50
                is_best = True
            if current_map50 > early_stop_map50 + Config.TEACHER_EARLY_STOP_MIN_DELTA:
                early_stop_map50 = current_map50
                significant_improvement = True
        elif val_loader is None and avg_train["total"] < best_loss:
            is_best = True

        if is_best:
            if val_metrics is None:
                best_loss = avg_train["total"]
            if is_main:
                torch.save(
                    _checkpoint_payload(
                        model,
                        epoch=epoch,
                        loss_value=avg_train["total"],
                        val_map50=best_map50 if val_metrics is not None else None,
                    ),
                    joint_best_path,
                )
        if significant_improvement:
            no_improve_epochs = 0
        elif val_metrics is not None:
            no_improve_epochs += Config.VAL_INTERVAL

        if is_main and epoch % Config.VIS_INTERVAL == 0:
            save_detection_visualization_anchor_v8(Config, epoch, unwrap_module(model), vis_dataset, vis_dir, prefix=vis_prefix, device=device)

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
            precision_op=val_metrics["precision_op"] if val_metrics is not None else None,
        )
        # 每个 epoch 结束后同步所有 rank（唯一 barrier，确保所有 rank 完成本 epoch 的全部工作后再进入下一 epoch）
        # Rank 0 owns early-stopping state; all ranks must take the same exit path.
        should_stop = bool(
            val_metrics is not None
            and epoch + 1 >= int(getattr(Config, "TEACHER_MIN_EPOCHS", 50))
            and Config.TEACHER_EARLY_STOP_PATIENCE > 0
            and no_improve_epochs >= Config.TEACHER_EARLY_STOP_PATIENCE
        ) if is_main else False
        if use_ddp:
            stop_tensor = torch.tensor(int(should_stop), device=device, dtype=torch.int32)
            torch.distributed.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())

        if should_stop:
            log_to_file(
                Config,
                f"Early stopping teacher after {no_improve_epochs} epochs without mAP50 improvement. "
                f"Best mAP50={best_map50:.4f}.",
            )
            break

    if is_main:
        torch.save(
            _checkpoint_payload(
                model,
                epoch=last_epoch,
                loss_value=history["train_total"][-1] if history["train_total"] else None,
                val_map50=best_map50 if best_map50 >= 0 else None,
            ),
            joint_final_path,
        )

    append_plain_log(Config, Config.get_epoch_table_separator())
    log_to_file(Config, "=" * 60)
    log_to_file(Config, "Training complete")
    log_to_file(Config, f"Best model saved to: {joint_best_path}")
    log_to_file(Config, f"Final model saved to: {joint_final_path}")
    log_to_file(Config, f"Teacher output directory: {Config.TEACHER_OUTPUT_DIR}")
    log_to_file(Config, "=" * 60)
    if is_main:
        save_training_curves(history, Config.TEACHER_OUTPUT_DIR, op_conf_threshold=Config.CONF_THRESH)
        if tensorboard_writer is not None:
            tensorboard_writer.close()

    if use_ddp:
        cleanup_distributed()
