import os
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn


def init_distributed_mode(config):
    """Initialize DDP only when launched by torchrun. Returns (local_rank, use_ddp)."""
    gpu_ids = getattr(config, "GPU_IDS", [])
    if len(gpu_ids) <= 1 or not torch.cuda.is_available():
        return 0, False
    required_env = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    if not all(name in os.environ for name in required_env):
        return 0, False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, False
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # 增加 NCCL 超时到 30 分钟，避免长 epoch 后误超时
    timeout_ms = int(os.environ.get("NCCL_TIMEOUT_MS", 30 * 60 * 1000))
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(f"cuda:{local_rank}"),
        timeout=timedelta(milliseconds=timeout_ms),
    )
    config.GPU_IDS = [local_rank]
    config.DEVICE = f"cuda:{local_rank}"
    return local_rank, True


def cleanup_distributed():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def init_log_file(config):
    os.makedirs(config.LOG_ROOT_DIR, exist_ok=True)
    config.TRAIN_START_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        with open(config.LOG_FILE, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("光学教师YOLOv8训练日志\n")
            f.write("=" * 80 + "\n")
            f.write(f"训练时间: {config.TRAIN_START_TIME}\n")
            f.write("=" * 80 + "\n\n")


def log_to_file(config, message, also_print=True):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_message = f"{timestamp} {message}"
    is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    if is_main and not config.should_skip_file_log(message):
        with open(config.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
    if also_print and is_main:
        print(log_message)


def append_plain_log(config, message=""):
    is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    if is_main:
        with open(config.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def get_runtime_device(config):
    if torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()
        return torch.device(f"cuda:{local_rank}")
    return torch.device(f"cuda:{config.GPU_IDS[0]}") if config.GPU_IDS else torch.device("cpu")


def should_use_channels_last(config):
    return torch.cuda.is_available() and getattr(config, "ENABLE_CHANNELS_LAST", False)


def prepare_tensor_for_device(config, tensor, device, *, channels_last=False):
    tensor = tensor.to(device, non_blocking=config.PIN_MEMORY)
    if not tensor.is_floating_point():
        return tensor
    if channels_last and tensor.dim() == 4 and should_use_channels_last(config):
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor.contiguous()


def prepare_conv_tensor(config, tensor):
    if tensor.is_floating_point() and tensor.dim() == 4 and should_use_channels_last(config):
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor.contiguous() if tensor.is_floating_point() else tensor


def _make_params_contiguous(module):
    """确保所有参数在内存中连续排列，避免 DDP 的 Grad strides 警告。"""
    for name, param in module.named_parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()
    for name, buf in module.named_buffers():
        if buf.is_floating_point() and not buf.is_contiguous():
            buf.data = buf.data.contiguous()


def wrap_data_parallel(config, module, module_name="module", find_unused_parameters=False):
    device = get_runtime_device(config)
    module = module.to(device)
    if should_use_channels_last(config):
        module = module.to(memory_format=torch.channels_last)
    if torch.distributed.is_initialized():
        _make_params_contiguous(module)
        module = nn.parallel.DistributedDataParallel(
            module, device_ids=[device.index], output_device=device.index,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=True,
        )
        world_size = torch.distributed.get_world_size()
        log_to_file(config, f"{module_name} wrapped with DDP on GPU {device.index} (world_size={world_size})")
    elif len(config.GPU_IDS) > 1:
        module = nn.DataParallel(module, device_ids=config.GPU_IDS, output_device=config.GPU_IDS[0])
        log_to_file(config, f"{module_name} wrapped with DataParallel on GPUs: {config.GPU_IDS}")
    return module


def unwrap_module(module):
    if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return module.module
    return module


def get_dataloader_kwargs(config, shuffle=False, sampler=None):
    kwargs = {
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": config.NUM_WORKERS,
        "pin_memory": config.PIN_MEMORY,
    }
    if config.NUM_WORKERS > 0:
        kwargs["persistent_workers"] = config.PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = config.PREFETCH_FACTOR
    return kwargs


def init_epoch_log_table(config):
    is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    if is_main:
        append_plain_log(config, "")
        append_plain_log(config, config.get_epoch_table_separator())
        append_plain_log(config, config.get_epoch_table_header())
        append_plain_log(config, config.get_epoch_table_separator())


def _format_table_value(value, width, decimals=4):
    if value is None:
        return f"{'N/A':<{width}}"
    try:
        if np.isnan(value):
            return f"{'N/A':<{width}}"
    except TypeError:
        pass
    return f"{value:<{width}.{decimals}f}"


def log_epoch_table_row(config, epoch, phase, train_loss, val_loss, precision, recall, f1_score, map50, lr, best_status):
    is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    if is_main:
        phase_text = str(phase)[: config.EPOCH_TABLE_PHASE_WIDTH - 1]
        append_plain_log(
            config,
            f"{epoch + 1:<{config.EPOCH_TABLE_EPOCH_WIDTH}}"
            f"{phase_text:<{config.EPOCH_TABLE_PHASE_WIDTH}}"
            f"{_format_table_value(train_loss, config.EPOCH_TABLE_TRAIN_LOSS_WIDTH)}"
            f"{_format_table_value(val_loss, config.EPOCH_TABLE_VAL_LOSS_WIDTH)}"
            f"{_format_table_value(precision, config.EPOCH_TABLE_METRIC_WIDTH, 3)}"
            f"{_format_table_value(recall, config.EPOCH_TABLE_METRIC_WIDTH, 3)}"
            f"{_format_table_value(f1_score, config.EPOCH_TABLE_METRIC_WIDTH, 3)}"
            f"{_format_table_value(map50, config.EPOCH_TABLE_METRIC_WIDTH, 3)}"
            f"{_format_table_value(lr, config.EPOCH_TABLE_LR_WIDTH, 6)}"
            f"{str(best_status):<{config.EPOCH_TABLE_BEST_WIDTH}}",
        )


def prepare_batch(config, batch, device):
    batch_images = []
    batch_targets = []
    for img_tensor, targets in batch:
        batch_images.append(img_tensor)
        batch_targets.append(targets)
    batch_images = prepare_tensor_for_device(config, torch.stack(batch_images), device, channels_last=True)
    return batch_images, batch_targets
