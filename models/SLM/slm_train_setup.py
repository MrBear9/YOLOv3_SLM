"""SLM training setup: model construction, data loading, criterion.

Contains setup_training() which initializes everything and returns a
context dict consumed by train() and run_epoch().
"""

import torch
from torch.utils.data import DataLoader

from models.SLM.config_slm import ConfigSLM as Config
from models.SLM.dataset_slm import SLMFeatureDataset, slm_collate_fn
from models.SLM.losses_slm import CompositeOpticalFeatureLoss, detection_response_loss, input_privacy_loss
from models.SLM.optical_layers import OpticalStudent
from models.SLM.slm_utils import configure_backends, log_config
from models.SLM.utils_slm import (
    load_student_checkpoint,
    load_teacher_detector_checkpoint,
    set_trainable,
)
from models.runtime import (
    DistributedEvalSampler,
    get_runtime_device,
    init_log_file,
    log_to_file,
    wrap_data_parallel,
)
from models.teacher import build_teacher
from models.training_utils import create_tensorboard_writer
from models.yolov8.head_v8 import build_detector_head
from models.yolov8.detection_protocol import build_detection_criterion


def setup_training(is_main, use_ddp):
    """Initialize everything and return a context dict for training."""
    Config.initialize()
    init_log_file(Config)
    configure_backends()
    log_config()
    device = get_runtime_device(Config)

    # Teacher + reference detector (frozen)
    teacher = build_teacher(Config).to(device)
    reference_detector = build_detector_head(Config, in_channels=1).to(device)
    checkpoint_info = load_teacher_detector_checkpoint(
        teacher, reference_detector, Config.TEACHER_DETECTOR_CHECKPOINT, device
    )
    log_to_file(Config, f"Loaded teacher/detector checkpoint: {checkpoint_info}")
    set_trainable(teacher, False)
    set_trainable(reference_detector, False)
    teacher.eval()
    reference_detector.eval()

    # Student + trainable detector
    multi_head_enabled = bool(getattr(Config, "SLM_MULTI_HEAD_ENABLED", False))
    if multi_head_enabled:
        from models.SLM.multi_head_slm import MultiHeadOpticalStudent
        student = MultiHeadOpticalStudent(Config).to(device)
        num_heads = int(getattr(Config, "SLM_MULTI_HEAD_NUM_HEADS", 4))
        log_to_file(Config, f"Multi-head SLM enabled: K={num_heads} virtual pairs → {sum(p.numel() for p in student.parameters()):,} params")
    else:
        student = OpticalStudent(Config).to(device)
    init_mode = str(Config.SLM_INIT_MODE).strip().lower()
    if init_mode == "checkpoint":
        student_info = load_student_checkpoint(student, Config.SLM_INIT_CHECKPOINT, device)
        log_to_file(Config, f"Initialized SLM student from checkpoint: {student_info}")
        if student_info["loaded"] == 0:
            raise RuntimeError(
                "SLM_INIT_MODE='checkpoint' requires a compatible SLM_INIT_CHECKPOINT; "
                f"loaded no tensors from {Config.SLM_INIT_CHECKPOINT!r}."
            )
    else:
        log_to_file(Config, f"Initialized SLM student with mode={init_mode}")
    detector = build_detector_head(Config, in_channels=1).to(device)
    detector.load_state_dict(reference_detector.state_dict(), strict=False)
    if Config.ENABLE_CHANNELS_LAST and torch.cuda.is_available():
        student = student.to(memory_format=torch.channels_last)
        detector = detector.to(memory_format=torch.channels_last)

    student_raw = student
    detector_raw = detector
    # Layer-wise ablations freeze one SLM after DDP has registered all parameters.
    # Let DDP mark that layer unused instead of waiting for gradients that never arrive.
    num_layers = int(getattr(Config, "NUM_LAYERS", 2))
    student_find_unused = not all(
        Config.is_trainable(i) if hasattr(Config, "is_trainable")
        else getattr(Config, f"TRAIN_SLM{i}", True)
        for i in range(1, num_layers + 1)
    )
    student = wrap_data_parallel(
        Config,
        student,
        module_name="OpticalStudent",
        find_unused_parameters=student_find_unused,
    )
    detector = wrap_data_parallel(Config, detector, module_name="Detector", find_unused_parameters=False)
    if student_find_unused:
        log_to_file(
            Config,
            "OpticalStudent DDP unused-parameter detection enabled for a single-SLM ablation.",
        )

    # Datasets & loaders
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
                val_sampler = DistributedEvalSampler(val_dataset)
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
    detection_criterion = build_detection_criterion(Config)

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

    tensorboard_writer = create_tensorboard_writer(Config, Config.OUTPUT_DIR, log_to_file) if is_main else None

    return {
        "device": device,
        "teacher": teacher,
        "reference_detector": reference_detector,
        "student": student,
        "detector": detector,
        "student_raw": student_raw,
        "detector_raw": detector_raw,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "vis_dataset": vis_dataset,
        "vis_prefix": vis_prefix,
        "feature_criterion": feature_criterion,
        "detection_criterion": detection_criterion,
        "history": history,
        "tensorboard_writer": tensorboard_writer,
        "use_ddp": use_ddp,
        "train_sampler": train_sampler,
    }
