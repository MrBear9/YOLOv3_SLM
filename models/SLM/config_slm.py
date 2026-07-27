import os
from datetime import datetime

import numpy as np
import torch

from models.SLM.config_optical import OpticalConfig
from models.yolov8.config_v8 import load_class_names, resolve_project_path


class ConfigSLM(OpticalConfig):
    # =========================================================================
    # Common paths, device, I/O
    # =========================================================================
    YAML_PATH = r"data/military/data.yaml"
    CLASS_NAMES = None
    NUM_CLASSES = None
    # Keep this run separate from the frozen-SLM baseline (mAP50=0.4647).
    OUTPUT_DIR = r"output/SLM_Tv1_light_direct_sgd"
    VISUALIZATION_DIR = None
    LOG_ROOT_DIR = None
    LOG_FILE = None
    TIMESTAMP = None
    TRAIN_START_TIME = None

    TEACHER_DETECTOR_CHECKPOINT = r"output/Tv1_light_srgb_linear_invertFalse/teacher_detector_best.pth"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_IDS = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

    # =========================================================================
    # Training scale
    # =========================================================================
    BATCH_SIZE = 8
    ANCHOR_FREE_STRIDES = [4, 8, 16, 32]

    # Match the proven two-stage baseline budget before testing any
    # detector-guided SLM refinement. The detector stage can stop early once
    # validation mAP50 has genuinely plateaued.
    PHASE_FOCUS_EPOCHS = 145
    DETECTOR_FOCUS_EPOCHS = 135
    # The current direct-SGD run peaks during detector_focus. Keep its learned
    # SLM/detector pair intact while the joint objectives are reworked.
    JOINT_FIT_EPOCHS = 0
    NORM_JOINT_EPOCHS = 0
    EPOCHS = PHASE_FOCUS_EPOCHS + DETECTOR_FOCUS_EPOCHS + JOINT_FIT_EPOCHS + NORM_JOINT_EPOCHS

    # =========================================================================
    # SLM optical parameters
    # =========================================================================
    WAVELENGTH = 532e-9
    PIXEL_SIZE = 6.4e-6
    # PROP_DISTANCE per-layer: see OpticalConfig.prop_distance(layer_idx)
    # Options: "phase", "amp_phase".
    SLM_MODE = "phase"
    # Optical/teacher canvas as (height, width). Example: (1080, 1920).
    RESOLUTION = (640, 640)
    OPTICAL_FIELD_EPS = 1e-8
    OPTICAL_NORM_EPS = 1e-6
    # DMD gray code is the incident intensity; must match teacher training.
    INPUT_INTENSITY_MODE = "srgb"

    # -------- Phase parameterisation --------
    # "direct_sgd" is the validated continuous phase optimization used by
    # HolographSLM. "direct_sgb" remains a spelling-compatible alias.
    # "direct_sgd_pyramid" adds a zero-initialized multi-scale residual while
    # retaining the direct continuous phase path. "multiscale_mlp" remains a
    # pure pyramid ablation.
    SLM_PHASE_PARAM_MODE = "direct_sgd"
    # Multiplier applied to the pyramid residual before direct phase addition.
    # Starting below one keeps the direct-SGD path dominant at early epochs.
    SLM_DIRECT_SGD_PYRAMID_SCALE = 0.25

    # Per-layer block/freq overrides: see OpticalConfig accessors
    #   phase_block_grid(layer_idx), phase_mlp_num_freqs(layer_idx), …
    # Per-layer trainable flag and LR multipliers: see OpticalConfig accessors
    #   is_trainable(layer_idx), layer_lr_mult(layer_idx, stage)

    # -------- Plan D: multi-head virtual SLM (training-time capacity boost) --------
    SLM_MULTI_HEAD_ENABLED = False
    SLM_MULTI_HEAD_NUM_HEADS = 2
    SLM_MULTI_HEAD_FUSION = "mean" # "learned_gate"

    # -------- Student normalization --------
    ENABLE_STUDENT_NORM = True
    # Options: "joint_and_norm", "norm_joint_only", "always", "none".
    STUDENT_NORM_SCHEDULE = "norm_joint_only"
    # Options: "max", "percentile", "mean", "none".
    STUDENT_NORM_MODE = "percentile"
    STUDENT_NORM_PERCENTILE = 0.990
    STUDENT_OUTPUT_CLAMP_MAX = 2.5
    STUDENT_OUTPUT_BLUR_KERNEL = 1

    # -------- SLM phase init --------
    # Options: zero, random, vortex, checkpoint.
    # Direct-SGD starts from HolographSLM's small continuous random phase map.
    SLM_INIT_MODE = "random"
    SLM_DIRECT_SGD_INIT_RANGE_RAD = 0.5
    SLM_INIT_NOISE_STD = 0.02
    SLM_INIT_CHECKPOINT = r"output/OpticalSLM_YOLOv8Head_student/optical_student_best.pth"
    # Per-layer vortex charge and radial curvature: see OpticalConfig.vortex_init.
    # Vortex is always one global phase singularity; it is never tiled.

    # -------- SLM drive calibration --------
    SLM_PHASE_LEVELS = 256
    # Hardware calibration is applied only when exporting a displayed phase map.
    # Direct-SGD optimizes exp(1j * raw_phase) continuously in simulation.
    SIMULATE_PHASE_QUANTIZATION = False
    SLM_GRAY_INVERTED = True
    SLM_EXPORT_PHASE_OFFSET_RAD = float(np.pi)
    # Optional .npy/.csv/.txt measured gray-to-phase curve in radians.
    # A one-column file is sampled uniformly over gray codes; two columns are
    # interpreted as (gray_code, measured_phase_radians).
    SLM_GRAY_TO_PHASE_LUT = r""

    # =========================================================================
    # TEACHER_ARCH  (must match teacher-training checkpoint)
    # Options: "convteacher"/"v1", "convteacher_v2"/"v2", "convteacher_v3"/"v3".
    # =========================================================================
    TEACHER_ARCH = "convteacher"

    # -------- TEACHER_ARCH = "convteacher" / "v1" --------
    TEACHER_V1_BASE_CHANNELS = 32
    TEACHER_V1_C2F_BLOCKS = 3

    # -------- TEACHER_ARCH = "convteacher_v2" --------
    TEACHER_V2_BASE_CHANNELS = 32
    TEACHER_V2_C2F_BLOCKS = 3
    TEACHER_V2_FOURIER_BANDS = 8
    TEACHER_V2_FOURIER_LOW_PASS_SIGMA = 0.5
    TEACHER_V2_RESIDUAL_SCALE = 0.30

    # -------- TEACHER_ARCH = "convteacher_v3" --------
    TEACHER_V3_BASE_CHANNELS = 24
    TEACHER_V3_C2F_BLOCKS = 2
    TEACHER_V3_RESIDUAL_SCALE = 0.30
    TEACHER_V3_GATE_SPARSITY_WEIGHT = 0.003
    TEACHER_V3_RESIDUAL_L1_WEIGHT = 0.001
    TEACHER_V3_OUTPUT_DEVIATION_WEIGHT = 0.02

    # =========================================================================
    # DETECTOR_HEAD_TYPE  (must match teacher-training checkpoint)
    # Options: "light".
    # =========================================================================
    DETECTOR_HEAD_TYPE = "light"

    # -------- Shared backbone params (fallback for non-light head types) --------
    YOLOV8_BASE_CHANNELS = 32
    YOLOV8_C2F_BLOCKS = 3

    # -------- DETECTOR_HEAD_TYPE = "light" --------
    YOLO_LIGHT_BASE_CH = 8  # A: halved from 16
    DETECTOR_USE_COORDCONV = True
    DETECTOR_INVERT_FEATURE = False   # invert teacher feature (dark→bright) before detector
    # The SLM detector baseline was validated on the raw optical intensity.
    # Keep it independent from teacher-side feature inversion.
    SLM_DETECTOR_INVERT_FEATURE = False
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

    # =========================================================================
    # SLM feature loss (student teacher feature matching)
    # =========================================================================
    LOSS_FULL_WEIGHT = 0.10
    LOSS_LOW1_WEIGHT = 0.35
    LOSS_LOW2_WEIGHT = 0.20
    LOSS_SSIM_WEIGHT = 0.25
    LOSS_GRAD_WEIGHT = 0.00
    LOSS_FREQ_WEIGHT = 0.00
    LOSS_PEARSON_WEIGHT = 0.20
    FEATURE_LOSS_PREFILTER_KERNEL = 1
    ENABLE_FEATURE_DOMAIN_ALIGNMENT = True
    # Options: "mean_std", "minmax"/"min_max", "none".
    FEATURE_DOMAIN_ALIGN_MODE = "minmax"

    # -------- Privacy / optical obfuscation loss --------
    PRIVACY_CORR_TARGET = 0.15
    PRIVACY_SSIM_TARGET = 0.20

    # =========================================================================
    # Stage loss weights
    # =========================================================================
    FEATURE_LOSS_WEIGHT_PHASE_FOCUS = 1.1
    DETECTION_LOSS_WEIGHT_PHASE_FOCUS = 0.0
    RESPONSE_LOSS_WEIGHT_PHASE_FOCUS = 0.0
    PRIVACY_LOSS_WEIGHT_PHASE_FOCUS = 0.0

    FEATURE_LOSS_WEIGHT_DETECTOR_FOCUS = 0.0
    DETECTION_LOSS_WEIGHT_DETECTOR_FOCUS = 1.0
    RESPONSE_LOSS_WEIGHT_DETECTOR_FOCUS = 0.0
    PRIVACY_LOSS_WEIGHT_DETECTOR_FOCUS = 0.0

    FEATURE_LOSS_WEIGHT_JOINT = 0.15
    DETECTION_LOSS_WEIGHT_JOINT = 1.00
    RESPONSE_LOSS_WEIGHT_JOINT = 0.00
    PRIVACY_LOSS_WEIGHT_JOINT = 0.00

    FEATURE_LOSS_WEIGHT_NORM_JOINT = 0.10
    DETECTION_LOSS_WEIGHT_NORM_JOINT = 1.00
    RESPONSE_LOSS_WEIGHT_NORM_JOINT = 0.00
    PRIVACY_LOSS_WEIGHT_NORM_JOINT = 0.00

    # =========================================================================
    # Optimizer & LR schedule
    # =========================================================================
    PHASE_FOCUS_PHASE_PARAM_LR = 3e-3
    DETECTOR_LR = 3e-4
    JOINT_PHASE_PARAM_LR = 5e-4
    JOINT_DETECTOR_LR = 5e-5
    NORM_JOINT_PHASE_PARAM_LR = 4e-4
    NORM_JOINT_DETECTOR_LR = 5e-5
    PHASE_GRAD_CLIP_NORM = 2.0
    WEIGHT_DECAY = 3e-5
    PHASE_WEIGHT_DECAY = 0.0
    # Options: "CosineAnnealingLR", "none".
    LR_SCHEDULER = "CosineAnnealingLR"
    ETA_MIN = 1e-6

    ENABLE_DETECTOR_FOCUS_EARLY_STOP = True
    DETECTOR_FOCUS_EARLY_STOP_PATIENCE = 20
    DETECTOR_FOCUS_EARLY_STOP_MIN_DELTA = 0.002

    # =========================================================================
    # Detection post-process
    # =========================================================================
    CONF_THRESH = 0.5
    NMS_THRESH = 0.45
    MAX_DET = 10
    METRIC_CONF_THRESH = 0.001
    METRIC_NMS_THRESH = 0.50
    METRIC_MAX_DET = 300
    AGNOSTIC_NMS = False

    # =========================================================================
    # Validation
    # =========================================================================
    VAL_INTERVAL = 1
    METRIC_IOU_THRESHOLD = 0.5

    # =========================================================================
    # Visualization
    # =========================================================================
    VIS_INTERVAL = 5
    VIS_BATCH_SIZE = 4
    VIS_DPI = 130
    # Options: "val", "train".
    VIS_DATASET_SPLIT = "val"
    VIS_SEED = 20260710
    VIS_CONF_THRESH = 0.5
    VIS_NMS_THRESH = 0.35
    VIS_MAX_DET = 5

    # Repeat each train-set entry for tiny subset experiments; validation is unchanged.
    TRAIN_DATASET_REPEAT = 1

    # =========================================================================
    # Data loading
    # =========================================================================
    _IS_WINDOWS = os.name == "nt"
    NUM_WORKERS = (0 if _IS_WINDOWS else min(12, os.cpu_count() or 0))
    PIN_MEMORY = torch.cuda.is_available()
    PERSISTENT_WORKERS = (not _IS_WINDOWS)
    PREFETCH_FACTOR = (0 if _IS_WINDOWS else 4)
    ENABLE_CHANNELS_LAST = True
    ENABLE_TF32 = True
    ENABLE_CUDNN_BENCHMARK = True

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
    SKIP_FILE_LOG_MESSAGES = ("best checkpoint updated",)

    @classmethod
    def initialize(cls):
        if not isinstance(cls.RESOLUTION, (tuple, list)) or len(cls.RESOLUTION) != 2:
            raise ValueError("RESOLUTION must be a (height, width) pair.")
        cls.RESOLUTION = tuple(int(value) for value in cls.RESOLUTION)
        if min(cls.RESOLUTION) < 1:
            raise ValueError("RESOLUTION height and width must be positive.")
        cls.EPOCHS = cls.PHASE_FOCUS_EPOCHS + cls.DETECTOR_FOCUS_EPOCHS + cls.JOINT_FIT_EPOCHS + cls.NORM_JOINT_EPOCHS
        cls.YAML_PATH = resolve_project_path(cls.YAML_PATH)
        cls.OUTPUT_DIR = resolve_project_path(cls.OUTPUT_DIR)
        cls.TEACHER_DETECTOR_CHECKPOINT = resolve_project_path(cls.TEACHER_DETECTOR_CHECKPOINT)
        cls.SLM_INIT_CHECKPOINT = resolve_project_path(cls.SLM_INIT_CHECKPOINT)
        cls.SLM_GRAY_TO_PHASE_LUT = resolve_project_path(cls.SLM_GRAY_TO_PHASE_LUT)
        cls.CLASS_NAMES, cls.NUM_CLASSES = load_class_names(cls.YAML_PATH)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        cls.LOG_ROOT_DIR = os.path.join(cls.OUTPUT_DIR, "logs")
        cls.VISUALIZATION_DIR = os.path.join(cls.OUTPUT_DIR, "visualizations")
        os.makedirs(cls.LOG_ROOT_DIR, exist_ok=True)
        os.makedirs(cls.VISUALIZATION_DIR, exist_ok=True)
        cls.TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.LOG_FILE = os.path.join(cls.LOG_ROOT_DIR, f"training_log_{cls.TIMESTAMP}.txt")

    @classmethod
    def get_detector_output_channels(cls):
        return 3 * (5 + cls.NUM_CLASSES)

    @classmethod
    def get_stage_loss_weights(cls, stage_name):
        if stage_name == "phase_focus":
            return {
                "feature": cls.FEATURE_LOSS_WEIGHT_PHASE_FOCUS,
                "detection": cls.DETECTION_LOSS_WEIGHT_PHASE_FOCUS,
                "response": cls.RESPONSE_LOSS_WEIGHT_PHASE_FOCUS,
                "privacy": cls.PRIVACY_LOSS_WEIGHT_PHASE_FOCUS,
            }
        if stage_name == "detector_focus":
            return {
                "feature": cls.FEATURE_LOSS_WEIGHT_DETECTOR_FOCUS,
                "detection": cls.DETECTION_LOSS_WEIGHT_DETECTOR_FOCUS,
                "response": cls.RESPONSE_LOSS_WEIGHT_DETECTOR_FOCUS,
                "privacy": cls.PRIVACY_LOSS_WEIGHT_DETECTOR_FOCUS,
            }
        if stage_name == "norm_joint":
            return {
                "feature": cls.FEATURE_LOSS_WEIGHT_NORM_JOINT,
                "detection": cls.DETECTION_LOSS_WEIGHT_NORM_JOINT,
                "response": cls.RESPONSE_LOSS_WEIGHT_NORM_JOINT,
                "privacy": cls.PRIVACY_LOSS_WEIGHT_NORM_JOINT,
            }
        return {
            "feature": cls.FEATURE_LOSS_WEIGHT_JOINT,
            "detection": cls.DETECTION_LOSS_WEIGHT_JOINT,
            "response": cls.RESPONSE_LOSS_WEIGHT_JOINT,
            "privacy": cls.PRIVACY_LOSS_WEIGHT_JOINT,
        }

    @classmethod
    def get_student_best_path(cls):
        return os.path.join(cls.OUTPUT_DIR, "optical_student_best.pth")

    @classmethod
    def get_student_current_path(cls):
        return os.path.join(cls.OUTPUT_DIR, "optical_student_current.pth")

    @classmethod
    def get_detector_best_path(cls):
        return os.path.join(cls.OUTPUT_DIR, "detector_best.pth")

    @classmethod
    def get_loss_curve_path(cls):
        return os.path.join(cls.OUTPUT_DIR, "loss_curve.png")

    @classmethod
    def get_epoch_table_columns(cls):
        return [
            ("Epoch", cls.EPOCH_TABLE_EPOCH_WIDTH),
            ("Stage", cls.EPOCH_TABLE_PHASE_WIDTH),
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
    def should_skip_file_log(cls, message):
        return any(token in message for token in cls.SKIP_FILE_LOG_MESSAGES)
