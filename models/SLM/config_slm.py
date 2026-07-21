import os
from datetime import datetime

import numpy as np
import torch

from models.SLM.config_optical import OpticalConfig
from models.yolov8.config_v8 import load_anchor_groups, load_class_names, resolve_project_path


class ConfigSLM(OpticalConfig):
    # =========================================================================
    # Common paths, device, I/O
    # =========================================================================
    YAML_PATH = r"data/military/data.yaml"
    CLASS_NAMES = None
    NUM_CLASSES = None
    OUTPUT_DIR = r"output/SLM_Tv1_light"
    VISUALIZATION_DIR = None
    LOG_ROOT_DIR = None
    LOG_FILE = None
    TIMESTAMP = None
    TRAIN_START_TIME = None

    TEACHER_DETECTOR_CHECKPOINT = r"output/Tv1_light_free_blance_0.8309/teacher_detector_best.pth"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_IDS = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

    # =========================================================================
    # Training scale
    # =========================================================================
    IMG_SIZE = 640
    BATCH_SIZE = 8
    STRIDES = [8, 16, 32]

    PHASE_FOCUS_EPOCHS = 245
    DETECTOR_FOCUS_EPOCHS = 135
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
    RESOLUTION = (640, 640)
    OPTICAL_FIELD_EPS = 1e-8
    OPTICAL_NORM_EPS = 1e-6

    # -------- Phase parameterisation --------
    # "direct": single flat phase_raw parameter (legacy / compatible)
    # "multiscale_mlp": Plan A+C multi-scale pyramid + neural-field MLP
    SLM_PHASE_PARAM_MODE = "multiscale_mlp"

    # Per-layer block/freq overrides: see OpticalConfig accessors
    #   phase_block_grid(layer_idx), phase_mlp_num_freqs(layer_idx), …
    # Per-layer trainable flag and LR multipliers: see OpticalConfig accessors
    #   is_trainable(layer_idx), layer_lr_mult(layer_idx, stage)

    # -------- Plan D: multi-head virtual SLM (training-time capacity boost) --------
    SLM_MULTI_HEAD_ENABLED = False
    SLM_MULTI_HEAD_NUM_HEADS = 4
    SLM_MULTI_HEAD_FUSION = "mean"

    # -------- Student normalization --------
    ENABLE_STUDENT_NORM = True
    # Options: "joint_and_norm", "norm_joint_only", "always", "none".
    STUDENT_NORM_SCHEDULE = "joint_and_norm"
    # Options: "max", "percentile", "mean", "none".
    STUDENT_NORM_MODE = "percentile"
    STUDENT_NORM_PERCENTILE = 0.990
    STUDENT_OUTPUT_CLAMP_MAX = 2.5
    STUDENT_OUTPUT_BLUR_KERNEL = 1

    # -------- SLM phase init --------
    # Options: zero, random, vortex, dh_psf/double_helix_psf, checkpoint,
    # vortex_checkpoint, dh_psf_checkpoint/double_helix_checkpoint.
    # "zero": flat phase (tiny noise), no range bias 鈥?lets the phase learn
    #   freely in early stages before diversity constraints ramp up.
    SLM_INIT_MODE = "vortex"
    SLM_INIT_NOISE_STD = 0.02
    SLM_INIT_CHECKPOINT = r"output/OpticalSLM_YOLOv8Head_student/optical_student_best.pth"
    # Per-layer vortex/DH-PSF init: see OpticalConfig.vortex_init(layer_idx), dh_psf_init(layer_idx)
    # Number of tiled vortex phase cells: 1.0 -> single cell, 2.0 -> 2 x 2 array, 3.0 -> 3 x 3 array.
    SLM_VORTEX_PERIODS = 1.0
    # When True and periods > 1, alternate charge sign in a checkerboard across cells.
    SLM_VORTEX_ALTERNATE_CHARGE = True
    # Number of tiled DH-PSF phase cells: 1.0 -> single cell, 2.0 -> 2 x 2 array, 3.0 -> 3 x 3 array.
    SLM_DH_PSF_PERIODS = 1.0
    # Spiral topological charge inside each DH-PSF cell; this is not the array count.
    SLM_DH_PSF_CHARGE = 1.0
    SLM_DH_PSF_RADIAL_SCALE = 6.0
    SLM_DH_PSF_SADDLE_SCALE = 0.08
    SLM_DH_PSF_SPIRAL_OFFSET = 0.0
    SLM_DH_PSF_APERTURE_RADIUS = 2.0
    # Per-layer rotation/handedness: see OpticalConfig.dh_psf_init(layer_idx)

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
    TEACHER_V2_SYNTHETIC_WAVELENGTHS = 3
    TEACHER_V2_COMPLEX_KERNEL_SIZE = 5

    # -------- TEACHER_ARCH = "convteacher_v3" --------
    TEACHER_V3_BASE_CHANNELS = 24
    TEACHER_V3_C2F_BLOCKS = 2
    TEACHER_V3_RESIDUAL_SCALE = 0.30

    # =========================================================================
    # DETECTOR_HEAD_TYPE  (must match teacher-training checkpoint)
    # Options: "light", "yolov8_anchor".
    # =========================================================================
    DETECTOR_HEAD_TYPE = "light"
    # Must match the teacher checkpoint: "anchor_free_tal" or legacy "anchor".
    DETECTION_PROTOCOL = "anchor_free_tal"

    # -------- DETECTOR_HEAD_TYPE = "yolov8_anchor" --------
    YOLOV8_BASE_CHANNELS = 32
    YOLOV8_C2F_BLOCKS = 3

    # -------- DETECTOR_HEAD_TYPE = "light" --------
    YOLO_LIGHT_BASE_CH = 8  # A: halved from 16
    ANCHOR_FREE_HEAD_CH = 24
    ANCHOR_FREE_REG_MAX = 16
    ANCHOR_FREE_BOX_WEIGHT = 7.5
    ANCHOR_FREE_CLS_WEIGHT = 0.5
    ANCHOR_FREE_DFL_WEIGHT = 1.5
    ANCHOR_FREE_PRE_NMS_TOPK = 3000
    TAL_TOPK = 10
    TAL_ALPHA = 0.5
    TAL_BETA = 5.0

    # =========================================================================
    # Anchors
    # =========================================================================
    DEFAULT_ANCHORS = [
        [[26, 23], [47, 49], [100, 67]],
        [[103, 169], [203, 107], [351, 177]],
        [[241, 354], [534, 299], [568, 528]],
    ]
    ANCHOR_CONFIG_PATH = r"output/anchor_clustering/yolo_anchors.yaml"
    USE_EXTERNAL_ANCHORS = True
    ANCHORS = None
    ANCHOR_SOURCE = "default"

    # =========================================================================
    # Anchor assignment
    # =========================================================================
    # Choose explicitly between "ratio" and "yolo7_simota" before training.
    ANCHOR_MATCH_MODE = "yolo7_simota"
    ANCHOR_MATCH_RATIO_THRESH = 3.5
    ASSIGN_NEIGHBOR_CELLS = True
    NOOBJ_IGNORE_IOU = 0.68
    ANCHOR_MATCH_IOU_THRESH = 0.20
    CENTER_PRIOR_RADIUS = 2.5
    CENTER_PRIOR_WEIGHT = 0.50
    SIMOTA_TOP_N = 20
    SIMOTA_MAX_ASSIGN = 15
    SIMOTA_OBJ_POS_THRESH = 0.05
    SIMOTA_USE_SIZE_WEIGHT_OVERRIDE = True
    SIMOTA_SMALL_OBJ_WEIGHT = 1.5
    SIMOTA_MEDIUM_OBJ_WEIGHT = 1.0
    SIMOTA_LARGE_OBJ_WEIGHT = 0.8

    # =========================================================================
    # Box decode
    # =========================================================================
    BOX_DECODE_RANGE = 2.0

    # =========================================================================
    # Detection loss weights
    # =========================================================================
    BOX_WEIGHT_BASE = 5.0
    OBJ_WEIGHT_BASE = 2.0
    NOOBJ_WEIGHT_BASE = 2.0
    CLS_WEIGHT_BASE = 1.8
    LOSS_UNCERTAINTY_WEIGHTING = False

    # -------- Object size weighting --------
    SMALL_OBJ_AREA = 32 * 32
    LARGE_OBJ_AREA = 128 * 128

    # =========================================================================
    # Focal loss
    # =========================================================================
    FOCAL_ALPHA = 0.35
    FOCAL_GAMMA = 2.0

    # =========================================================================
    # Hard negative mining
    # =========================================================================
    HARD_NEG_RATIO = 30
    HARD_NEG_MIN = 512

    # =========================================================================
    # SLM feature loss (student teacher feature matching)
    # =========================================================================
    LOSS_FULL_WEIGHT = 0.03
    LOSS_LOW1_WEIGHT = 0.25
    LOSS_LOW2_WEIGHT = 0.15
    LOSS_SSIM_WEIGHT = 0.35
    LOSS_GRAD_WEIGHT = 0.10
    LOSS_FREQ_WEIGHT = 0.08
    LOSS_PEARSON_WEIGHT = 0.45
    LOSS_PHASE_SMOOTH_WEIGHT = 0.000
    LOSS_PHASE_DIVERSITY_WEIGHT = 0.015
    PHASE_SMOOTH_WEIGHT_PHASE_FOCUS = 0.001
    PHASE_DIVERSITY_WEIGHT_PHASE_FOCUS = 0.03
    PHASE_SMOOTH_WEIGHT_DETECTOR_FOCUS = 0.0
    PHASE_DIVERSITY_WEIGHT_DETECTOR_FOCUS = 0.0
    PHASE_SMOOTH_WEIGHT_JOINT = 0.02
    PHASE_DIVERSITY_WEIGHT_JOINT = 0.03
    PHASE_SMOOTH_WEIGHT_NORM_JOINT = 0.02
    PHASE_DIVERSITY_WEIGHT_NORM_JOINT = 0.08
    FEATURE_LOSS_PREFILTER_KERNEL = 1
    ENABLE_FEATURE_DOMAIN_ALIGNMENT = True
    # Options: "mean_std", "minmax"/"min_max", "none".
    FEATURE_DOMAIN_ALIGN_MODE = "minmax"

    # -------- Privacy / optical obfuscation loss --------
    PRIVACY_CORR_TARGET = 0.15
    PRIVACY_SSIM_TARGET = 0.20

    # -------- Phase quality constraints --------
    PHASE_STD_TARGET = 0.40 # 0.60 -> 0.40
    PHASE_SPAN_TARGET = 2.50 # 3.50 -> 2.50
    PHASE_CIRCULAR_STD_TARGET = 0.35 # 0.50 -> 0.35
    PHASE_NEAR_BOUNDARY_LIMIT = 0.85
    PHASE_NEAR_BOUNDARY_EPS = 0.05
    PHASE_BEST_MIN_STD = 0.01 # 0.10 -> 0.01
    PHASE_BEST_MIN_CIRCULAR_STD = 0.01 # 0.15 -> 0.01
    PHASE_BEST_MAX_NEAR_BOUNDARY_RATIO = 0.90
    PHASE_BEST_MIN_SPAN = 0.01 # 0.01 -> 0.01

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

    FEATURE_LOSS_WEIGHT_JOINT = 0.08
    DETECTION_LOSS_WEIGHT_JOINT = 1.00
    RESPONSE_LOSS_WEIGHT_JOINT = 0.03
    PRIVACY_LOSS_WEIGHT_JOINT = 0.00

    FEATURE_LOSS_WEIGHT_NORM_JOINT = 0.05
    DETECTION_LOSS_WEIGHT_NORM_JOINT = 1.00
    RESPONSE_LOSS_WEIGHT_NORM_JOINT = 0.02
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
    DETECTOR_FOCUS_EARLY_STOP_PATIENCE = 18
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
        cls.EPOCHS = cls.PHASE_FOCUS_EPOCHS + cls.DETECTOR_FOCUS_EPOCHS + cls.JOINT_FIT_EPOCHS + cls.NORM_JOINT_EPOCHS
        cls.YAML_PATH = resolve_project_path(cls.YAML_PATH)
        cls.OUTPUT_DIR = resolve_project_path(cls.OUTPUT_DIR)
        cls.TEACHER_DETECTOR_CHECKPOINT = resolve_project_path(cls.TEACHER_DETECTOR_CHECKPOINT)
        cls.SLM_INIT_CHECKPOINT = resolve_project_path(cls.SLM_INIT_CHECKPOINT)
        cls.ANCHOR_CONFIG_PATH = resolve_project_path(cls.ANCHOR_CONFIG_PATH)
        cls.CLASS_NAMES, cls.NUM_CLASSES = load_class_names(cls.YAML_PATH)
        cls.ANCHORS = [[anchor.copy() for anchor in layer] for layer in cls.DEFAULT_ANCHORS]
        cls.ANCHOR_SOURCE = "default"
        if cls.USE_EXTERNAL_ANCHORS:
            try:
                cls.ANCHORS = load_anchor_groups(cls.ANCHOR_CONFIG_PATH)
                cls.ANCHOR_SOURCE = cls.ANCHOR_CONFIG_PATH
            except Exception as exc:
                cls.ANCHORS = [[anchor.copy() for anchor in layer] for layer in cls.DEFAULT_ANCHORS]
                cls.ANCHOR_SOURCE = f"default (external load failed: {exc})"
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
    def get_phase_regularization_weights(cls, stage_name):
        if stage_name == "phase_focus":
            return {
                "smooth": cls.PHASE_SMOOTH_WEIGHT_PHASE_FOCUS,
                "diversity": cls.PHASE_DIVERSITY_WEIGHT_PHASE_FOCUS,
            }
        if stage_name == "detector_focus":
            return {
                "smooth": cls.PHASE_SMOOTH_WEIGHT_DETECTOR_FOCUS,
                "diversity": cls.PHASE_DIVERSITY_WEIGHT_DETECTOR_FOCUS,
            }
        if stage_name == "norm_joint":
            return {
                "smooth": cls.PHASE_SMOOTH_WEIGHT_NORM_JOINT,
                "diversity": cls.PHASE_DIVERSITY_WEIGHT_NORM_JOINT,
            }
        return {
            "smooth": cls.PHASE_SMOOTH_WEIGHT_JOINT,
            "diversity": cls.PHASE_DIVERSITY_WEIGHT_JOINT,
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
