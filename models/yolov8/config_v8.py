'''
@File    :   config_v8.py
@Time    :   2026/07/03 18:09:48
@Author  :   Mr.Bear9 
@Github  :   https://github.com/MrBear9
'''


import os
from datetime import datetime

import torch
import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_project_path(path):
    if not path or os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def load_class_names(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", ["object"])
    return {i: name for i, name in enumerate(names)}, len(names)


class ConfigYOLOv8Anchor:
    # =========================================================================
    # Common — paths, device, I/O
    # =========================================================================
    YAML_PATH = r"data/military/data.yaml"
    CLASS_NAMES = None
    NUM_CLASSES = None
    TEACHER_OUTPUT_DIR = r"output/Tv1_light"
    LOG_ROOT_DIR = None
    LOG_FILE = None
    TIMESTAMP = None
    TRAIN_START_TIME = None

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_IDS = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

    # =========================================================================
    # Training scale
    # =========================================================================
    # Shared image/feature canvas as (height, width). Example: (1080, 1920).
    RESOLUTION = (640, 640)
    BATCH_SIZE = 8
    ANCHOR_FREE_STRIDES = [4, 8, 16, 32]

    STAGE1_LOCATE_EPOCHS = 40
    STAGE2_BALANCE_EPOCHS = 200
    EPOCHS = STAGE1_LOCATE_EPOCHS + STAGE2_BALANCE_EPOCHS

    # =========================================================================
    # TEACHER_ARCH = "convteacher" | "v1"  (deeper semantic projection)
    # =========================================================================
    TEACHER_V1_BASE_CHANNELS = 32
    TEACHER_V1_C2F_BLOCKS = 3

    # =========================================================================
    # TEACHER_ARCH = "convteacher_v2" | "v2"  (Fourier + C2fCIB + V3 output)
    # =========================================================================
    TEACHER_ARCH = "convteacher"
    TEACHER_V2_BASE_CHANNELS = 32
    TEACHER_V2_C2F_BLOCKS = 3
    TEACHER_V2_FOURIER_BANDS = 8
    TEACHER_V2_FOURIER_LOW_PASS_SIGMA = 0.5
    TEACHER_V2_RESIDUAL_SCALE = 0.30

    # =========================================================================
    # TEACHER_ARCH = "convteacher_v3" | "v3"  (residual + gate)
    # =========================================================================
    TEACHER_V3_BASE_CHANNELS = 24
    TEACHER_V3_C2F_BLOCKS = 2
    TEACHER_V3_RESIDUAL_SCALE = 0.30
    TEACHER_V3_GATE_SPARSITY_WEIGHT = 0.003
    TEACHER_V3_RESIDUAL_L1_WEIGHT = 0.001
    TEACHER_V3_OUTPUT_DEVIATION_WEIGHT = 0.02

    # =========================================================================
    # TEACHER_ARCH = "convteacher"     (legacy V1, sigmoid heatmap)
    # =========================================================================
    # (uses its own hard-coded channels)

    # =========================================================================
    # Teacher init / freeze "scratch"(default) | "checkpoint" | "joint_checkpoint" 
    # =========================================================================
    TEACHER_INIT_MODE = "joint_checkpoint"
    TEACHER_INIT_CHECKPOINT = r"output/Tv1_light_0.8398/teacher_detector_best.pth"
    FREEZE_TEACHER = False

    # =========================================================================
    # DETECTOR_HEAD_TYPE = "light" | "compact"
    # =========================================================================
    DETECTOR_HEAD_TYPE = "light"

    # -------- Shared backbone params (also used as fallback by teacher_guidance) --------
    YOLOV8_BASE_CHANNELS = 32
    YOLOV8_C2F_BLOCKS = 3

    # -------- DETECTOR_HEAD_TYPE = "light" --------
    YOLO_LIGHT_BASE_CH = 8  # 方案A: halved from 16
    DETECTOR_USE_COORDCONV = True
    ANCHOR_FREE_HEAD_CH = 24
    ANCHOR_FREE_P2_FUSION_CH = 16
    ANCHOR_FREE_P2_HEAD_CH = 16
    ANCHOR_FREE_REG_MAX = 16
    ANCHOR_FREE_BOX_WEIGHT = 7.5
    ANCHOR_FREE_CLS_WEIGHT = 0.5
    ANCHOR_FREE_DFL_WEIGHT = 1.5
    ANCHOR_FREE_PRE_NMS_TOPK = 3000
    TAL_TOPK = 10
    TAL_ALPHA = 0.5
    TAL_BETA = 6.0
    DETECTOR_INVERT_FEATURE = True   # invert teacher feature (dark→bright) before detector

    # -------- DETECTOR_HEAD_TYPE = "compact" | "center_detect" --------
    # Anchor-free center-point detector (CompactOpticalDetector).
    # Uses heatmap + box-size + center-offset heads; CenterDetectionLoss.
    # COMPACT_MODEL_VERSION: "v1" (109K params) or "v2" (~32K, Level-4 decoupled 3-network).
    COMPACT_MODEL_VERSION = "v2"
    COMPACT_BASE_CH = 16
    COMPACT_HEAD_CH = 32
    COMPACT_DILATIONS = (1, 2, 4)
    COMPACT_DETECTOR_LR = 3e-4
    COMPACT_WEIGHT_DECAY = 3e-5
    COMPACT_GRAD_CLIP_NORM = 5.0
    HEATMAP_LOSS_WEIGHT = 1.0
    WH_LOSS_WEIGHT = 0.08
    OFFSET_LOSS_WEIGHT = 1.0
    # Level-4 V2 decoupled loss weights (obj + cls replace heatmap)
    OBJ_LOSS_WEIGHT = 1.0
    CLS_LOSS_WEIGHT = 1.0
    MIN_HEATMAP_RADIUS = 1
    HEATMAP_RADIUS_SCALE = 0.35
    # Compact decode/metric thresholds (separate from anchor-head counterparts)
    COMPACT_CONF_THRESH = 0.30
    COMPACT_NMS_THRESH = 0.45
    COMPACT_MAX_DET = 100
    COMPACT_METRIC_CONF_THRESH = 0.001
    COMPACT_METRIC_NMS_THRESH = 0.50
    COMPACT_METRIC_MAX_DET = 100
    COMPACT_METRIC_PRE_NMS_TOPK = 300
    COMPACT_METRIC_IOU_THRESHOLD = 0.5

    # =========================================================================
    # Feature distillation (teacher → detector)
    # =========================================================================
    ENABLE_FEATURE_DISTILL = True
    FEATURE_DISTILL_WEIGHT = 0.5

    # =========================================================================
    # Teacher ciphertext regularization (weakened)
    #   TV/HF: relaxed to allow more texture detail for SLM modulation
    #   Range/Mean: kept as floor to prevent all-dark output
    #   Peak/Edge: relaxed to allow localized brightness for target highlighting
    # =========================================================================
    TEACHER_SLM_CIPHER_LOSS_WEIGHT = 0.04
    TEACHER_SLM_CIPHER_BLUR_KERNEL = 15
    TEACHER_SLM_CIPHER_TV_TARGET = 0.045
    TEACHER_SLM_CIPHER_HF_TARGET = 0.075
    TEACHER_SLM_CIPHER_RANGE_FLOOR = 0.28
    TEACHER_SLM_CIPHER_MEAN_FLOOR = 0.52
    TEACHER_SLM_CIPHER_PEAK_LIMIT = 0.88
    TEACHER_SLM_CIPHER_EDGE_LIMIT = 0.75
    TEACHER_SLM_CIPHER_TV_WEIGHT = 0.5
    TEACHER_SLM_CIPHER_HF_WEIGHT = 0.6
    TEACHER_SLM_CIPHER_RANGE_WEIGHT = 0.6
    TEACHER_SLM_CIPHER_MEAN_WEIGHT = 0.8
    TEACHER_SLM_CIPHER_PEAK_WEIGHT = 0.6
    TEACHER_SLM_CIPHER_EDGE_WEIGHT = 0.4
    OPTICAL_FIELD_EPS = 1e-8
    OPTICAL_NORM_EPS = 1e-6

    # =========================================================================
    # Optimizer & LR schedule
    # =========================================================================
    PHASE1_TEACHER_LR = 4e-4
    PHASE1_DETECTOR_LR = 3e-4
    PHASE2_TEACHER_LR = 1.5e-4
    PHASE2_DETECTOR_LR = 1e-4
    # Short low-LR refinement from the protected 0.8398 joint checkpoint.
    JOINT_RESUME_TEACHER_LR = 3e-5
    JOINT_RESUME_DETECTOR_LR = 2e-5
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-3
    OPTIMIZER = "AdamW"
    LR_SCHEDULER = "CosineAnnealingLR"
    ETA_MIN = 1e-6

    # =========================================================================
    # Detection post-process
    # =========================================================================
    CONF_THRESH = 0.35
    NMS_THRESH = 0.35
    MAX_DET = 20
    AGNOSTIC_NMS = False

    # =========================================================================
    # Validation
    # =========================================================================
    VAL_INTERVAL = 2
    TEACHER_EARLY_STOP_PATIENCE = 20
    TEACHER_EARLY_STOP_MIN_DELTA = 0.002
    METRIC_IOU_THRESHOLD = 0.5
    METRIC_CONF_THRESH = 0.001
    METRIC_NMS_THRESH = 0.50
    METRIC_MAX_DET = 300

    # =========================================================================
    # Visualization
    # =========================================================================
    VIS_INTERVAL = 5
    VIS_BATCH_SIZE = 4
    VIS_DPI = 130
    VIS_DATASET_SPLIT = "val"
    VIS_SEED = 20260506
    VIS_CONF_THRESH = 0.50
    VIS_NMS_THRESH = 0.35
    VIS_MAX_DET = 20

    # =========================================================================
    # Data loading
    # =========================================================================
    USE_CLASS_BALANCED_SAMPLER = True
    SAMPLER_SEED = 20260716
    CLASS_BALANCE_POWER = 0.6
    MAX_CLASS_BALANCE_GAIN = 3.0
    MAJORITY_ONLY_IMAGE_WEIGHT = 0.45
    EMPTY_IMAGE_SAMPLE_WEIGHT = 0.7
    MIN_IMAGE_SAMPLE_WEIGHT = 0.35

    # Conservative box-aware augmentation for grayscale optical inputs.
    TRAIN_AUGMENT = True
    AUG_HFLIP_PROB = 0.5
    AUG_ROTATE_DEG = 5.0
    AUG_SCALE_MIN = 0.8
    AUG_SCALE_MAX = 1.25
    AUG_TRANSLATE = 0.08
    AUG_BRIGHTNESS = 0.15
    AUG_CONTRAST = 0.15
    AUG_GAMMA = 0.15
    AUG_BLUR_PROB = 0.08
    AUG_NOISE_PROB = 0.10
    AUG_NOISE_STD = 0.01

    # Targeted training-only augmentation for the remaining small-soldier gap.
    SOLDIER_COPY_PASTE = True
    SOLDIER_CLASS_ID = 1
    SOLDIER_COPY_PASTE_PROB = 0.12
    SOLDIER_COPY_PASTE_MAX_OBJECTS = 1
    # Keep this gentle augmentation active throughout checkpoint refinement.
    SOLDIER_COPY_PASTE_AREA_MAX = 32 * 32
    SOLDIER_COPY_PASTE_SCALE_MIN = 0.90
    SOLDIER_COPY_PASTE_SCALE_MAX = 1.10
    SOLDIER_COPY_PASTE_IOA_MAX = 0.15
    SOLDIER_COPY_PASTE_EDGE_FEATHER = 2

    # Windows 使用 spawn 创建多进程，开销远大于 Linux 的 fork，需要降低 worker 数量
    _IS_WINDOWS = os.name == "nt"
    NUM_WORKERS = (0 if _IS_WINDOWS else min(4, os.cpu_count() or 0))
    PIN_MEMORY = torch.cuda.is_available()
    PERSISTENT_WORKERS = False
    PREFETCH_FACTOR = (0 if _IS_WINDOWS else 2)
    DATALOADER_TIMEOUT = (0 if _IS_WINDOWS else 300)
    ENABLE_CUDNN_BENCHMARK = True
    ENABLE_CHANNELS_LAST = True
    ENABLE_TF32 = True
    ENABLE_AMP = True
    AMP_DTYPE = "float16"

    # =========================================================================
    # Log / table formatting
    # =========================================================================
    EPOCH_TABLE_EPOCH_WIDTH = 8
    EPOCH_TABLE_PHASE_WIDTH = 18
    EPOCH_TABLE_TRAIN_LOSS_WIDTH = 13
    EPOCH_TABLE_VAL_LOSS_WIDTH = 13
    EPOCH_TABLE_METRIC_WIDTH = 11
    EPOCH_TABLE_LR_WIDTH = 12
    EPOCH_TABLE_BEST_WIDTH = 8
    EPOCH_TABLE_BEST_MARK = "Yes"
    SKIP_FILE_LOG_MESSAGES = ("best checkpoint updated", "saved best model", "best model saved")

    @classmethod
    def initialize(cls):
        if not isinstance(cls.RESOLUTION, (tuple, list)) or len(cls.RESOLUTION) != 2:
            raise ValueError("RESOLUTION must be a (height, width) pair.")
        cls.RESOLUTION = tuple(int(value) for value in cls.RESOLUTION)
        if min(cls.RESOLUTION) < 1:
            raise ValueError("RESOLUTION height and width must be positive.")
        cls.YAML_PATH = resolve_project_path(cls.YAML_PATH)
        cls.TEACHER_OUTPUT_DIR = resolve_project_path(cls.TEACHER_OUTPUT_DIR)
        cls.CLASS_NAMES, cls.NUM_CLASSES = load_class_names(cls.YAML_PATH)
        os.makedirs(cls.TEACHER_OUTPUT_DIR, exist_ok=True)
        cls.LOG_ROOT_DIR = os.path.join(cls.TEACHER_OUTPUT_DIR, "logs")
        os.makedirs(cls.LOG_ROOT_DIR, exist_ok=True)
        cls.TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.TRAIN_START_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cls.LOG_FILE = os.path.join(cls.LOG_ROOT_DIR, f"training_log_{cls.TIMESTAMP}.txt")

    @classmethod
    def get_detector_output_channels(cls):
        return 3 * (5 + cls.NUM_CLASSES)

    @classmethod
    def get_teacher_init_mode(cls):
        mode = str(cls.TEACHER_INIT_MODE).strip().lower()
        return mode if mode in {"scratch", "checkpoint", "joint_checkpoint"} else "scratch"

    @classmethod
    def get_teacher_init_checkpoint(cls):
        if cls.get_teacher_init_mode() not in {"checkpoint", "joint_checkpoint"}:
            return None
        checkpoint_path = str(cls.TEACHER_INIT_CHECKPOINT).strip()
        return checkpoint_path if checkpoint_path else None

    @classmethod
    def get_stage_settings(cls, epoch):
        if epoch < cls.STAGE1_LOCATE_EPOCHS:
            return {
                "phase": "locate_gt",
                "teacher_lr": cls.PHASE1_TEACHER_LR,
                "detector_lr": cls.PHASE1_DETECTOR_LR,
            }
        return {
            "phase": "balance_refine",
            "teacher_lr": cls.PHASE2_TEACHER_LR,
            "detector_lr": cls.PHASE2_DETECTOR_LR,
        }

    @classmethod
    def get_dynamic_weights(cls, epoch):
        """Return phase name only; kept for backward compatibility."""
        return {"phase": cls.get_stage_settings(epoch)["phase"]}

    @classmethod
    def should_skip_file_log(cls, message):
        return any(token in message for token in cls.SKIP_FILE_LOG_MESSAGES)

    @classmethod
    def get_epoch_table_columns(cls):
        return [
            ("Epoch", cls.EPOCH_TABLE_EPOCH_WIDTH),
            ("Phase", cls.EPOCH_TABLE_PHASE_WIDTH),
            ("Train Loss", cls.EPOCH_TABLE_TRAIN_LOSS_WIDTH),
            ("Val Loss", cls.EPOCH_TABLE_VAL_LOSS_WIDTH),
            ("Precision", cls.EPOCH_TABLE_METRIC_WIDTH),
            ("Recall", cls.EPOCH_TABLE_METRIC_WIDTH),
            ("F1", cls.EPOCH_TABLE_METRIC_WIDTH),
            ("mAP50", cls.EPOCH_TABLE_METRIC_WIDTH),
            ("LR", cls.EPOCH_TABLE_LR_WIDTH),
            ("Best", cls.EPOCH_TABLE_BEST_WIDTH),
        ]

    @classmethod
    def get_epoch_table_separator(cls):
        return "-" * sum(width for _, width in cls.get_epoch_table_columns())

    @classmethod
    def get_epoch_table_header(cls):
        return "".join(f"{title:<{width}}" for title, width in cls.get_epoch_table_columns())

    @classmethod
    def print_config(cls):
        print("=" * 80)
        print("光学教师YOLOv8训练配置")
        print("=" * 80)
        print(f"Device: {cls.DEVICE}")
        print(f"Resolution (H, W): {cls.RESOLUTION}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Num Classes: {cls.NUM_CLASSES}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print("=" * 80)
