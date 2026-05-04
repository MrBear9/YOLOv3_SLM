import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn


def init_log_file(config):
    os.makedirs(config.LOG_ROOT_DIR, exist_ok=True)
    config.TRAIN_START_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(config.LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("光学教师YOLOv8训练日志\n")
        f.write("=" * 80 + "\n")
        f.write(f"训练时间: {config.TRAIN_START_TIME}\n")
        f.write("=" * 80 + "\n\n")


def log_to_file(config, message, also_print=True):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_message = f"{timestamp} {message}"
    if not config.should_skip_file_log(message):
        with open(config.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
    if also_print:
        print(log_message)


def append_plain_log(config, message=""):
    with open(config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def get_runtime_device(config):
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


def wrap_data_parallel(config, module, module_name="module"):
    device = get_runtime_device(config)
    module = module.to(device)
    if should_use_channels_last(config):
        module = module.to(memory_format=torch.channels_last)
    if len(config.GPU_IDS) > 1:
        module = nn.DataParallel(module, device_ids=config.GPU_IDS, output_device=config.GPU_IDS[0])
        log_to_file(config, f"{module_name} wrapped with DataParallel on GPUs: {config.GPU_IDS}")
    return module


def unwrap_module(module):
    return module.module if isinstance(module, nn.DataParallel) else module


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
