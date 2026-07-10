"""Compact optical detection training main loop.

Contains the train() function. Supporting helpers live in:
- compact_utils.py     : init, optimizer, checkpoint
- compact_tensorboard.py : TensorBoard writing
- compact_vis.py       : visualization
"""

from contextlib import nullcontext

import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.SLM.dataset_slm import SLMFeatureDataset, slm_collate_fn
from models.SLM.losses_slm import CompositeOpticalFeatureLoss
from models.SLM.optical_layers import OpticalStudent
from models.SLM.utils_slm import load_student_checkpoint, set_trainable
from models.teacher import build_teacher
from models.runtime import (
    cleanup_distributed,
    get_runtime_device,
    init_distributed_mode,
    init_epoch_log_table,
    log_epoch_table_row,
    log_to_file,
    unwrap_module,
)
from models.training_utils import create_tensorboard_writer

from .compact_tensorboard import write_tensorboard_model_summary, write_tensorboard_scalars
from .compact_utils import (
    build_optimizer,
    init_compact_log_file,
    load_teacher_feature_checkpoint,
    save_checkpoint,
)
from .compact_vis import add_tensorboard_teacher_feature, add_tensorboard_visualization, save_compact_visualization_png
from .config import ConfigCompactDetect as Config
from .losses import CenterDetectionLoss
from .metrics import evaluate_center_detector
from .model import CompactOpticalDetector, OpticalCompactDetector


def build_warmup_subset(dataset):
    subset_size = int(getattr(Config, "COMPACT_TEACHER_WARMUP_SUBSET_SIZE", 0))
    if subset_size <= 0 or bool(getattr(Config, "SINGLE_IMAGE_TRAINING", False)) or subset_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(int(getattr(Config, "VIS_SEED", 20260709)))
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def train():
    Config.initialize()
    _, use_ddp = init_distributed_mode(Config)
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    init_compact_log_file()
    if is_main:
        init_epoch_log_table(Config)
    if not use_ddp and torch.cuda.is_available() and len(Config.GPU_IDS) > 1:
        original_gpu_ids = list(Config.GPU_IDS)
        Config.GPU_IDS = [Config.GPU_IDS[0]]
        Config.DEVICE = f"cuda:{Config.GPU_IDS[0]}"
        log_to_file(
            Config,
            f"Multiple GPUs detected {original_gpu_ids}, but DataParallel is disabled for the optical FFT route. "
            "Using single GPU for this python launch; use torchrun for multi-GPU DDP.",
        )
    device = get_runtime_device(Config)

    warmup_epochs = max(int(getattr(Config, "COMPACT_TEACHER_WARMUP_EPOCHS", 0)), 0)
    teacher = None
    feature_criterion = None
    if warmup_epochs > 0:
        teacher = build_teacher(Config).to(device)
        teacher_info = load_teacher_feature_checkpoint(teacher, Config.TEACHER_DETECTOR_CHECKPOINT, device)
        set_trainable(teacher, False)
        teacher.eval()
        feature_criterion = CompositeOpticalFeatureLoss(Config)
        log_to_file(Config, f"Loaded teacher for compact feature warmup: {teacher_info}")
        log_to_file(
            Config,
            f"Compact teacher warmup: epochs={warmup_epochs}, weight={Config.COMPACT_TEACHER_WARMUP_WEIGHT}, "
            f"raw_student={Config.COMPACT_TEACHER_WARMUP_RAW_STUDENT}",
        )

    train_dataset = SLMFeatureDataset(Config, split="train")
    val_dataset = SLMFeatureDataset(Config, split="val")
    warmup_dataset = build_warmup_subset(train_dataset) if warmup_epochs > 0 else train_dataset
    train_sampler = None
    warmup_sampler = None
    val_sampler = None
    if use_ddp:
        from torch.utils.data.distributed import DistributedSampler
        warmup_sampler = DistributedSampler(warmup_dataset, shuffle=True, drop_last=False)
        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            drop_last=not bool(getattr(Config, "SINGLE_IMAGE_TRAINING", False)),
        )
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    loader_kwargs = {
        "batch_size": Config.BATCH_SIZE,
        "num_workers": Config.NUM_WORKERS,
        "pin_memory": Config.PIN_MEMORY,
        "collate_fn": slm_collate_fn,
    }
    if Config.NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = Config.PERSISTENT_WORKERS
        loader_kwargs["prefetch_factor"] = Config.PREFETCH_FACTOR
    warmup_loader = DataLoader(
        warmup_dataset,
        shuffle=warmup_sampler is None,
        sampler=warmup_sampler,
        **loader_kwargs,
    )
    train_loader = DataLoader(train_dataset, shuffle=train_sampler is None, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, sampler=val_sampler, **loader_kwargs)

    student_raw = OpticalStudent(Config).to(device)
    pretrained_student = Config.COMPACT_PRETRAINED_STUDENT
    if pretrained_student:
        info = load_student_checkpoint(student_raw, pretrained_student, device)
        log_to_file(Config, f"Student checkpoint load: {info}")
    set_trainable(student_raw, bool(Config.COMPACT_TRAIN_STUDENT))

    detector_raw = CompactOpticalDetector(Config, in_channels=1).to(device)
    if Config.ENABLE_CHANNELS_LAST and torch.cuda.is_available():
        student_raw = student_raw.to(memory_format=torch.channels_last)
        detector_raw = detector_raw.to(memory_format=torch.channels_last)

    optimizer = build_optimizer(student_raw, detector_raw)
    if dist.is_initialized():
        student = torch.nn.parallel.DistributedDataParallel(
            student_raw,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=False,
        )
        detector = torch.nn.parallel.DistributedDataParallel(
            detector_raw,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=False,
        )
        log_to_file(Config, f"CompactOpticalStudent wrapped with DDP on GPU {device.index} (world_size={dist.get_world_size()})")
        log_to_file(Config, f"CompactDetector wrapped with DDP on GPU {device.index} (world_size={dist.get_world_size()})")
    else:
        student = student_raw
        detector = detector_raw

    model_for_count = OpticalCompactDetector(student_raw, detector_raw)
    student_params = sum(p.numel() for p in student_raw.parameters())
    detector_params = sum(p.numel() for p in detector_raw.parameters())
    total_params = sum(p.numel() for p in model_for_count.parameters())
    log_to_file(Config, f"Train images: {len(train_dataset)}, val images: {len(val_dataset)}")
    if warmup_epochs > 0 and len(warmup_dataset) != len(train_dataset):
        log_to_file(
            Config,
            f"Compact teacher warmup subset: {len(warmup_dataset)}/{len(train_dataset)} images "
            f"(seed={getattr(Config, 'VIS_SEED', 'unknown')})",
        )
    if bool(getattr(Config, "SINGLE_IMAGE_TRAINING", False)):
        log_to_file(
            Config,
            f"Compact single-image training enabled: image={Config.SINGLE_IMAGE_PATH}, "
            f"label={Config.SINGLE_IMAGE_LABEL_PATH or 'auto'}, train_repeat={Config.SINGLE_IMAGE_REPEAT}",
        )
    log_to_file(Config, f"Compact detector parameters: {detector_params:,}")
    log_to_file(Config, f"Student parameters: {student_params:,}, combined train route: {total_params:,}")

    criterion = CenterDetectionLoss(Config)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    tensorboard_writer = create_tensorboard_writer(Config, Config.OUTPUT_DIR, log_to_file) if is_main else None
    if is_main:
        write_tensorboard_model_summary(tensorboard_writer, student_raw, detector_raw, (1, 1, Config.IMG_SIZE, Config.IMG_SIZE))
        add_tensorboard_teacher_feature(tensorboard_writer, 0, val_dataset, teacher, device)
        add_tensorboard_visualization(tensorboard_writer, 0, val_dataset, student, detector, device)
        save_compact_visualization_png(0, val_dataset, student, detector, device)
    if dist.is_initialized():
        dist.barrier()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(Config.EPOCHS, 1), eta_min=1e-6)
    amp_enabled = bool(getattr(Config, "ENABLE_AMP", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(Config.AMP_DTYPE).lower() in {"bf16", "bfloat16"} else torch.float16
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled) if device.type == "cuda" else nullcontext()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_map50 = -1.0
    best_loss = float("inf")
    for epoch in range(Config.EPOCHS):
        in_teacher_warmup = epoch < warmup_epochs
        phase_name = "teacher_warmup" if in_teacher_warmup else "compact"
        current_loader = warmup_loader if in_teacher_warmup else train_loader
        current_sampler = warmup_sampler if in_teacher_warmup else train_sampler
        if current_sampler is not None:
            current_sampler.set_epoch(epoch)
        student.train(bool(Config.COMPACT_TRAIN_STUDENT))
        detector.train(not in_teacher_warmup)
        set_trainable(detector, not in_teacher_warmup)
        epoch_loss_t = torch.zeros((), device=device)
        stat_keys = (
            ("feature_total", "full", "low1", "low2", "ssim", "grad", "freq", "pearson", "slm_smooth", "slm_diversity")
            if in_teacher_warmup
            else ("heatmap", "wh", "offset")
        )
        epoch_stats_t = {key: torch.zeros((), device=device) for key in stat_keys}

        for batch in tqdm(current_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [{phase_name}]", leave=True, disable=not is_main):
            images = batch["gray_tensor"].to(device, non_blocking=Config.PIN_MEMORY)
            optimizer.zero_grad(set_to_none=True)
            if in_teacher_warmup:
                rgb = batch["rgb_tensor"].to(device, non_blocking=Config.PIN_MEMORY)
                if Config.ENABLE_CHANNELS_LAST and torch.cuda.is_available():
                    rgb = rgb.contiguous(memory_format=torch.channels_last)
                fp32_ctx = torch.amp.autocast(device_type="cuda", enabled=False) if device.type == "cuda" else nullcontext()
                with fp32_ctx:
                    with torch.no_grad():
                        teacher_feature = teacher(rgb.float())
                    if Config.COMPACT_TEACHER_WARMUP_RAW_STUDENT:
                        from .compact_vis import forward_student_with_norm
                        features = forward_student_with_norm(student, images.float(), enable_norm=False)
                    else:
                        features = student(images.float())
                    feature_loss, stats = feature_criterion(features, teacher_feature.detach(), unwrap_module(student), stage_name="phase_focus")
                    loss = feature_loss * float(Config.COMPACT_TEACHER_WARMUP_WEIGHT)
            else:
                targets = [target.to(device, non_blocking=Config.PIN_MEMORY) for target in batch["targets"]]
                with amp_ctx:
                    features = student(images)
                    pred = detector(features)
                    loss, stats = criterion(pred, targets)
            scaler.scale(loss).backward()
            if Config.COMPACT_GRAD_CLIP_NORM and Config.COMPACT_GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None],
                    Config.COMPACT_GRAD_CLIP_NORM,
                )
            scaler.step(optimizer)
            scaler.update()
            epoch_loss_t += loss.detach()
            for key in epoch_stats_t:
                epoch_stats_t[key] += torch.as_tensor(stats.get(key, 0.0), device=device)

        scheduler.step()
        batch_count_t = torch.tensor(float(max(len(current_loader), 1)), device=device)
        if dist.is_initialized():
            dist.all_reduce(epoch_loss_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_count_t, op=dist.ReduceOp.SUM)
            for value in epoch_stats_t.values():
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
        train_loss = float((epoch_loss_t / batch_count_t.clamp(min=1.0)).item())
        epoch_stats = {key: float((value / batch_count_t.clamp(min=1.0)).item()) for key, value in epoch_stats_t.items()}

        val_losses = None
        val_metrics = None
        if not in_teacher_warmup and (epoch + 1) % Config.VAL_INTERVAL == 0:
            val_losses, val_metrics = evaluate_center_detector(Config, student, detector, val_loader, criterion, device)
        if is_main and Config.VIS_INTERVAL > 0 and (epoch + 1) % Config.VIS_INTERVAL == 0:
            add_tensorboard_visualization(tensorboard_writer, epoch + 1, val_dataset, student, detector, device)
        if is_main and Config.VIS_FILE_INTERVAL > 0 and (epoch + 1) % Config.VIS_FILE_INTERVAL == 0:
            save_compact_visualization_png(epoch + 1, val_dataset, student, detector, device)
        if dist.is_initialized():
            dist.barrier()

        current_lr = max(group["lr"] for group in optimizer.param_groups)
        is_best = False
        if not in_teacher_warmup and is_main and val_metrics is not None and val_metrics["map50"] > best_map50:
            best_map50 = val_metrics["map50"]
            is_best = True
        elif not in_teacher_warmup and is_main and val_metrics is None and train_loss < best_loss:
            is_best = True
        if not in_teacher_warmup and train_loss < best_loss:
            best_loss = train_loss
        if is_main and is_best:
            save_checkpoint(Config.get_detector_best_path(), student, detector, epoch + 1, train_loss, metrics=val_metrics)
            log_to_file(Config, f"Saved best compact detector: epoch={epoch + 1}, map50={best_map50:.4f}, train_loss={train_loss:.6f}")
        if is_main and not in_teacher_warmup and Config.SAVE_INTERVAL > 0 and (epoch + 1) % Config.SAVE_INTERVAL == 0:
            save_checkpoint(Config.get_detector_final_path(), student, detector, epoch + 1, train_loss, metrics=val_metrics)

        if is_main:
            write_tensorboard_scalars(tensorboard_writer, epoch + 1, train_loss, epoch_stats, val_losses=val_losses, val_metrics=val_metrics, lr=current_lr)
            log_epoch_table_row(
                Config,
                epoch=epoch,
                phase=phase_name,
                train_loss=train_loss,
                val_loss=val_losses["total"] if val_losses is not None else None,
                precision=val_metrics["precision"] if val_metrics is not None else None,
                recall=val_metrics["recall"] if val_metrics is not None else None,
                f1_score=val_metrics["f1"] if val_metrics is not None else None,
                map50=val_metrics["map50"] if val_metrics is not None else None,
                lr=current_lr,
                best_status=Config.EPOCH_TABLE_BEST_MARK if is_best else "",
            )

    if is_main:
        save_checkpoint(Config.get_detector_final_path(), student, detector, Config.EPOCHS, best_loss)
        if tensorboard_writer is not None:
            tensorboard_writer.close()
        log_to_file(Config, f"Training complete. Best checkpoint: {Config.get_detector_best_path()}")
    if dist.is_initialized():
        cleanup_distributed()
