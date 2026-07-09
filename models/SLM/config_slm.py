import os
from datetime import datetime

import numpy as np
import torch

from models.yolov8.config_v8 import load_anchor_groups, load_class_names, resolve_project_path


class ConfigSLM:
    # =========================================================================
    # Common — paths, device, I/O
    # =========================================================================
    YAML_PATH = r"data/military/data.yaml"
    CLASS_NAMES = None
    NUM_CLASSES = None
    OUTPUT_DIR = r"output/SLM_Tv1_light_branch"
    VISUALIZATION_DIR = None
    LOG_ROOT_DIR = None
    LOG_FILE = None
    TIMESTAMP = None
    TRAIN_START_TIME = None

    TEACHER_DETECTOR_CHECKPOINT = r"output/Tv1_light_branch/teacher_detector_best.pth"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_IDS = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

    # =========================================================================
    # Training scale
    # =========================================================================
    IMG_SIZE = 640
    BATCH_SIZE = 16
    STRIDES = [8, 16, 32]

    PHASE_FOCUS_EPOCHS = 145
    DETECTOR_FOCUS_EPOCHS = 35
    JOINT_FIT_EPOCHS = 100
    NORM_JOINT_EPOCHS = 30
    EPOCHS = PHASE_FOCUS_EPOCHS + DETECTOR_FOCUS_EPOCHS + JOINT_FIT_EPOCHS + NORM_JOINT_EPOCHS

    # =========================================================================
    # SLM optical parameters
    # =========================================================================
    WAVELENGTH = 532e-9
    PIXEL_SIZE = 6.4e-6
    PROP_DISTANCE_1 = 0.10 # 10cm
    PROP_DISTANCE_2 = 0.20 # 20cm
    # Options: "phase", "amp_phase".
    SLM_MODE = "phase"
    RESOLUTION = (640, 640)
    OPTICAL_FIELD_EPS = 1e-8
    OPTICAL_NORM_EPS = 1e-6

    # -------- Phase parameterisation --------
    # "direct": single flat phase_raw parameter (legacy / compatible)
    # "multiscale_mlp": Plan A+C — multi-scale pyramid + neural-field MLP
    SLM_PHASE_PARAM_MODE = "multiscale_mlp"
    # Plan A — multi-scale pyramid levels (80→160→320 for 640 resolution)
    SLM_PHASE_NUM_SCALES = 3
    # Plan B — overlapping block-wise learning (replaces the finest scale)
    SLM_PHASE_USE_BLOCKWISE = True
    SLM_PHASE_BLOCK_GRID = 5       # 5×5 = 25 blocks (set 1 to disable)
    SLM_PHASE_BLOCK_OVERLAP = 20   # overlap pixels for boundary blending
    SLM_PHASE_BLOCK_INNER_SCALES = 2  # mini-pyramid layers inside each block (1 = single scale)
    # Plan C — coordinate MLP architecture
    SLM_PHASE_MLP_HIDDEN = 64
    SLM_PHASE_MLP_NUM_FREQS = 6
    SLM_PHASE_MLP_LAYERS = 3

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
    # "zero": flat phase (tiny noise), no range bias — lets the phase learn
    #   freely in early stages before diversity constraints ramp up.
    SLM_INIT_MODE = "zero"
    SLM_INIT_NOISE_STD = 0.02
    SLM_INIT_CHECKPOINT = r"output/OpticalSLM_YOLOv8Head_student/optical_student_best.pth"
    SLM_VORTEX_CHARGE_1 = 1.0
    SLM_VORTEX_CHARGE_2 = -1.0
    SLM_VORTEX_RADIAL_SCALE_1 = 0.35
    SLM_VORTEX_RADIAL_SCALE_2 = -0.25
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
    SLM_DH_PSF_ROTATION_1 = 0.0
    SLM_DH_PSF_ROTATION_2 = np.pi / 2
    SLM_DH_PSF_HANDEDNESS_1 = 1.0
    SLM_DH_PSF_HANDEDNESS_2 = -1.0

    # =========================================================================
    # TEACHER_ARCH  (must match teacher-training checkpoint)
    # Options: "convteacher"/"v1", "convteacher_v2"/"v2", "convteacher_v3"/"v3".
    # =========================================================================
    TEACHER_ARCH = "convteacher_v2"

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
    # Options: "light_branch", "light", "yolov8_anchor".
    # =========================================================================
    DETECTOR_HEAD_TYPE = "light"

    # -------- DETECTOR_HEAD_TYPE = "yolov8_anchor" --------
    YOLOV8_BASE_CHANNELS = 32
    YOLOV8_C2F_BLOCKS = 3

    # -------- DETECTOR_HEAD_TYPE = "light" --------
    YOLO_LIGHT_BASE_CH = 16

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
    # Options: "auto", "ratio", "yolo7_simota".
    # "auto" keeps ratio matching for legacy heads and enables YOLOv7-style
    # SimOTA matching for DETECTOR_HEAD_TYPE="light".
    ANCHOR_MATCH_MODE = "auto"
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
    SMALL_OBJ_WEIGHT = 0.8
    MEDIUM_OBJ_WEIGHT = 1.0
    LARGE_OBJ_WEIGHT = 1.5

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
    # SLM feature loss (student → teacher matching)
    # =========================================================================
    LOSS_FULL_WEIGHT = 0.05
    LOSS_LOW1_WEIGHT = 0.20
    LOSS_LOW2_WEIGHT = 0.10
    LOSS_SSIM_WEIGHT = 0.30
    LOSS_GRAD_WEIGHT = 0.25
    LOSS_FREQ_WEIGHT = 0.05
    LOSS_PEARSON_WEIGHT = 0.40
    LOSS_PHASE_SMOOTH_WEIGHT = 0.3
    LOSS_PHASE_DIVERSITY_WEIGHT = 0.15
    PHASE_SMOOTH_WEIGHT_PHASE_FOCUS = 0.003
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
    PHASE_STD_TARGET = 0.60
    PHASE_SPAN_TARGET = 3.50
    PHASE_CIRCULAR_STD_TARGET = 0.50
    PHASE_NEAR_BOUNDARY_LIMIT = 0.85
    PHASE_NEAR_BOUNDARY_EPS = 0.05
    PHASE_BEST_MIN_STD = 0.10
    PHASE_BEST_MIN_CIRCULAR_STD = 0.15
    PHASE_BEST_MAX_NEAR_BOUNDARY_RATIO = 0.90
    PHASE_BEST_MIN_SPAN = 1.10

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
    DETECTOR_FOCUS_EARLY_STOP_PATIENCE = 8
    DETECTOR_FOCUS_EARLY_STOP_MIN_DELTA = 0.002

    # =========================================================================
    # Detection post-process
    # =========================================================================
    CONF_THRESH = 0.7
    NMS_THRESH = 0.25
    MAX_DET = 5
    METRIC_CONF_THRESH = 0.001
    METRIC_NMS_THRESH = 0.50
    METRIC_MAX_DET = 300
    AGNOSTIC_NMS = False
    ENABLE_CONTAINMENT_SUPPRESSION = False
    ENABLE_WBF = False

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
    VIS_SEED = 20260504
    VIS_CONF_THRESH = 0.5
    VIS_NMS_THRESH = 0.35
    VIS_MAX_DET = 5
    VIS_SHOW_BEST_MATCHED_ANCHORS = False
    VIS_MAX_GT_ANCHOR_OVERLAYS = 2

    # =========================================================================
    # Data loading
    # =========================================================================
    # Windows 使用 spawn 创建多进程，开销远大于 Linux 的 fork，需要降低 worker 数量
    _IS_WINDOWS = os.name == "nt"
    NUM_WORKERS = (4 if _IS_WINDOWS else min(12, os.cpu_count() or 0))
    PIN_MEMORY = torch.cuda.is_available()
    PERSISTENT_WORKERS = (not _IS_WINDOWS)
    PREFETCH_FACTOR = (2 if _IS_WINDOWS else 4)
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
    def apply_runtime_overrides(cls):
        overrides = {
            "OPTICAL_SLM_YAML_PATH": ("YAML_PATH", str),
            "OPTICAL_SLM_OUTPUT_DIR": ("OUTPUT_DIR", str),
            "OPTICAL_SLM_TEACHER_DETECTOR_CHECKPOINT": ("TEACHER_DETECTOR_CHECKPOINT", str),
            "OPTICAL_SLM_DETECTOR_HEAD_TYPE": ("DETECTOR_HEAD_TYPE", str),
            "OPTICAL_SLM_INIT_MODE": ("SLM_INIT_MODE", str),
            "OPTICAL_SLM_INIT_CHECKPOINT": ("SLM_INIT_CHECKPOINT", str),
            "OPTICAL_SLM_VORTEX_PERIODS": ("SLM_VORTEX_PERIODS", float),
            "OPTICAL_SLM_VORTEX_CHARGE_1": ("SLM_VORTEX_CHARGE_1", float),
            "OPTICAL_SLM_VORTEX_CHARGE_2": ("SLM_VORTEX_CHARGE_2", float),
            "OPTICAL_SLM_VORTEX_RADIAL_SCALE_1": ("SLM_VORTEX_RADIAL_SCALE_1", float),
            "OPTICAL_SLM_VORTEX_RADIAL_SCALE_2": ("SLM_VORTEX_RADIAL_SCALE_2", float),
            "OPTICAL_SLM_DH_PSF_PERIODS": ("SLM_DH_PSF_PERIODS", float),
            "OPTICAL_SLM_DH_PSF_CHARGE": ("SLM_DH_PSF_CHARGE", float),
            "OPTICAL_SLM_DH_PSF_RADIAL_SCALE": ("SLM_DH_PSF_RADIAL_SCALE", float),
            "OPTICAL_SLM_DH_PSF_SADDLE_SCALE": ("SLM_DH_PSF_SADDLE_SCALE", float),
            "OPTICAL_SLM_DH_PSF_SPIRAL_OFFSET": ("SLM_DH_PSF_SPIRAL_OFFSET", float),
            "OPTICAL_SLM_DH_PSF_APERTURE_RADIUS": ("SLM_DH_PSF_APERTURE_RADIUS", float),
            "OPTICAL_SLM_DH_PSF_ROTATION_1": ("SLM_DH_PSF_ROTATION_1", float),
            "OPTICAL_SLM_DH_PSF_ROTATION_2": ("SLM_DH_PSF_ROTATION_2", float),
            "OPTICAL_SLM_DH_PSF_HANDEDNESS_1": ("SLM_DH_PSF_HANDEDNESS_1", float),
            "OPTICAL_SLM_DH_PSF_HANDEDNESS_2": ("SLM_DH_PSF_HANDEDNESS_2", float),
            "OPTICAL_SLM_TEACHER_ARCH": ("TEACHER_ARCH", str),
            "OPTICAL_SLM_TEACHER_V1_BASE_CHANNELS": ("TEACHER_V1_BASE_CHANNELS", int),
            "OPTICAL_SLM_TEACHER_V1_C2F_BLOCKS": ("TEACHER_V1_C2F_BLOCKS", int),
            "OPTICAL_SLM_TEACHER_V2_BASE_CHANNELS": ("TEACHER_V2_BASE_CHANNELS", int),
            "OPTICAL_SLM_TEACHER_V2_C2F_BLOCKS": ("TEACHER_V2_C2F_BLOCKS", int),
            "OPTICAL_SLM_TEACHER_V2_SYNTHETIC_WAVELENGTHS": ("TEACHER_V2_SYNTHETIC_WAVELENGTHS", int),
            "OPTICAL_SLM_TEACHER_V2_COMPLEX_KERNEL_SIZE": ("TEACHER_V2_COMPLEX_KERNEL_SIZE", int),
            "OPTICAL_SLM_TEACHER_V3_BASE_CHANNELS": ("TEACHER_V3_BASE_CHANNELS", int),
            "OPTICAL_SLM_TEACHER_V3_C2F_BLOCKS": ("TEACHER_V3_C2F_BLOCKS", int),
            "OPTICAL_SLM_YOLO_LIGHT_BASE_CH": ("YOLO_LIGHT_BASE_CH", int),
            "OPTICAL_SLM_TEACHER_V3_RESIDUAL_SCALE": ("TEACHER_V3_RESIDUAL_SCALE", float),
            "OPTICAL_SLM_ANCHOR_MATCH_MODE": ("ANCHOR_MATCH_MODE", str),
            "OPTICAL_SLM_ANCHOR_MATCH_IOU_THRESH": ("ANCHOR_MATCH_IOU_THRESH", float),
            "OPTICAL_SLM_CENTER_PRIOR_RADIUS": ("CENTER_PRIOR_RADIUS", float),
            "OPTICAL_SLM_CENTER_PRIOR_WEIGHT": ("CENTER_PRIOR_WEIGHT", float),
            "OPTICAL_SLM_SIMOTA_TOP_N": ("SIMOTA_TOP_N", int),
            "OPTICAL_SLM_SIMOTA_MAX_ASSIGN": ("SIMOTA_MAX_ASSIGN", int),
            "OPTICAL_SLM_SIMOTA_OBJ_POS_THRESH": ("SIMOTA_OBJ_POS_THRESH", float),
            "OPTICAL_SLM_STUDENT_NORM_SCHEDULE": ("STUDENT_NORM_SCHEDULE", str),
            "OPTICAL_SLM_STUDENT_NORM_MODE": ("STUDENT_NORM_MODE", str),
            "OPTICAL_SLM_STUDENT_NORM_PERCENTILE": ("STUDENT_NORM_PERCENTILE", float),
            "OPTICAL_SLM_STUDENT_OUTPUT_CLAMP_MAX": ("STUDENT_OUTPUT_CLAMP_MAX", float),
            "OPTICAL_SLM_STUDENT_OUTPUT_BLUR_KERNEL": ("STUDENT_OUTPUT_BLUR_KERNEL", int),
            "OPTICAL_SLM_PHASE_FOCUS_PHASE_PARAM_LR": ("PHASE_FOCUS_PHASE_PARAM_LR", float),
            "OPTICAL_SLM_DETECTOR_LR": ("DETECTOR_LR", float),
            "OPTICAL_SLM_JOINT_PHASE_PARAM_LR": ("JOINT_PHASE_PARAM_LR", float),
            "OPTICAL_SLM_JOINT_DETECTOR_LR": ("JOINT_DETECTOR_LR", float),
            "OPTICAL_SLM_NORM_JOINT_PHASE_PARAM_LR": ("NORM_JOINT_PHASE_PARAM_LR", float),
            "OPTICAL_SLM_NORM_JOINT_DETECTOR_LR": ("NORM_JOINT_DETECTOR_LR", float),
            "OPTICAL_SLM_PHASE_GRAD_CLIP_NORM": ("PHASE_GRAD_CLIP_NORM", float),
            "OPTICAL_SLM_LOSS_SSIM_WEIGHT": ("LOSS_SSIM_WEIGHT", float),
            "OPTICAL_SLM_LOSS_GRAD_WEIGHT": ("LOSS_GRAD_WEIGHT", float),
            "OPTICAL_SLM_LOSS_FREQ_WEIGHT": ("LOSS_FREQ_WEIGHT", float),
            "OPTICAL_SLM_LOSS_PEARSON_WEIGHT": ("LOSS_PEARSON_WEIGHT", float),
            "OPTICAL_SLM_FEATURE_LOSS_WEIGHT_PHASE_FOCUS": ("FEATURE_LOSS_WEIGHT_PHASE_FOCUS", float),
            "OPTICAL_SLM_DETECTION_LOSS_WEIGHT_PHASE_FOCUS": ("DETECTION_LOSS_WEIGHT_PHASE_FOCUS", float),
            "OPTICAL_SLM_RESPONSE_LOSS_WEIGHT_PHASE_FOCUS": ("RESPONSE_LOSS_WEIGHT_PHASE_FOCUS", float),
            "OPTICAL_SLM_PRIVACY_LOSS_WEIGHT_PHASE_FOCUS": ("PRIVACY_LOSS_WEIGHT_PHASE_FOCUS", float),
            "OPTICAL_SLM_FEATURE_LOSS_WEIGHT_DETECTOR_FOCUS": ("FEATURE_LOSS_WEIGHT_DETECTOR_FOCUS", float),
            "OPTICAL_SLM_DETECTION_LOSS_WEIGHT_DETECTOR_FOCUS": ("DETECTION_LOSS_WEIGHT_DETECTOR_FOCUS", float),
            "OPTICAL_SLM_RESPONSE_LOSS_WEIGHT_DETECTOR_FOCUS": ("RESPONSE_LOSS_WEIGHT_DETECTOR_FOCUS", float),
            "OPTICAL_SLM_PRIVACY_LOSS_WEIGHT_DETECTOR_FOCUS": ("PRIVACY_LOSS_WEIGHT_DETECTOR_FOCUS", float),
            "OPTICAL_SLM_FEATURE_LOSS_WEIGHT_JOINT": ("FEATURE_LOSS_WEIGHT_JOINT", float),
            "OPTICAL_SLM_DETECTION_LOSS_WEIGHT_JOINT": ("DETECTION_LOSS_WEIGHT_JOINT", float),
            "OPTICAL_SLM_RESPONSE_LOSS_WEIGHT_JOINT": ("RESPONSE_LOSS_WEIGHT_JOINT", float),
            "OPTICAL_SLM_PRIVACY_LOSS_WEIGHT_JOINT": ("PRIVACY_LOSS_WEIGHT_JOINT", float),
            "OPTICAL_SLM_FEATURE_LOSS_WEIGHT_NORM_JOINT": ("FEATURE_LOSS_WEIGHT_NORM_JOINT", float),
            "OPTICAL_SLM_DETECTION_LOSS_WEIGHT_NORM_JOINT": ("DETECTION_LOSS_WEIGHT_NORM_JOINT", float),
            "OPTICAL_SLM_RESPONSE_LOSS_WEIGHT_NORM_JOINT": ("RESPONSE_LOSS_WEIGHT_NORM_JOINT", float),
            "OPTICAL_SLM_PRIVACY_LOSS_WEIGHT_NORM_JOINT": ("PRIVACY_LOSS_WEIGHT_NORM_JOINT", float),
            "OPTICAL_SLM_PHASE_SMOOTH_WEIGHT_PHASE_FOCUS": ("PHASE_SMOOTH_WEIGHT_PHASE_FOCUS", float),
            "OPTICAL_SLM_PHASE_DIVERSITY_WEIGHT_PHASE_FOCUS": ("PHASE_DIVERSITY_WEIGHT_PHASE_FOCUS", float),
            "OPTICAL_SLM_PHASE_SMOOTH_WEIGHT_DETECTOR_FOCUS": ("PHASE_SMOOTH_WEIGHT_DETECTOR_FOCUS", float),
            "OPTICAL_SLM_PHASE_DIVERSITY_WEIGHT_DETECTOR_FOCUS": ("PHASE_DIVERSITY_WEIGHT_DETECTOR_FOCUS", float),
            "OPTICAL_SLM_PHASE_SMOOTH_WEIGHT_JOINT": ("PHASE_SMOOTH_WEIGHT_JOINT", float),
            "OPTICAL_SLM_PHASE_DIVERSITY_WEIGHT_JOINT": ("PHASE_DIVERSITY_WEIGHT_JOINT", float),
            "OPTICAL_SLM_PHASE_SMOOTH_WEIGHT_NORM_JOINT": ("PHASE_SMOOTH_WEIGHT_NORM_JOINT", float),
            "OPTICAL_SLM_PHASE_DIVERSITY_WEIGHT_NORM_JOINT": ("PHASE_DIVERSITY_WEIGHT_NORM_JOINT", float),
            "OPTICAL_SLM_FEATURE_DOMAIN_ALIGN_MODE": ("FEATURE_DOMAIN_ALIGN_MODE", str),
            "OPTICAL_SLM_PRIVACY_CORR_TARGET": ("PRIVACY_CORR_TARGET", float),
            "OPTICAL_SLM_PRIVACY_SSIM_TARGET": ("PRIVACY_SSIM_TARGET", float),
            "OPTICAL_SLM_DETECTOR_FOCUS_EARLY_STOP_PATIENCE": ("DETECTOR_FOCUS_EARLY_STOP_PATIENCE", int),
            "OPTICAL_SLM_DETECTOR_FOCUS_EARLY_STOP_MIN_DELTA": ("DETECTOR_FOCUS_EARLY_STOP_MIN_DELTA", float),
            "OPTICAL_SLM_LR_SCHEDULER": ("LR_SCHEDULER", str),
            "OPTICAL_SLM_ETA_MIN": ("ETA_MIN", float),
            "OPTICAL_SLM_BATCH_SIZE": ("BATCH_SIZE", int),
            "OPTICAL_SLM_PHASE_FOCUS_EPOCHS": ("PHASE_FOCUS_EPOCHS", int),
            "OPTICAL_SLM_DETECTOR_FOCUS_EPOCHS": ("DETECTOR_FOCUS_EPOCHS", int),
            "OPTICAL_SLM_JOINT_FIT_EPOCHS": ("JOINT_FIT_EPOCHS", int),
            "OPTICAL_SLM_NORM_JOINT_EPOCHS": ("NORM_JOINT_EPOCHS", int),
            "OPTICAL_SLM_NUM_WORKERS": ("NUM_WORKERS", int),
            "OPTICAL_SLM_PHASE_PARAM_MODE": ("SLM_PHASE_PARAM_MODE", str),
        }
        for env_name, (attr, caster) in overrides.items():
            value = os.environ.get(env_name)
            if value:
                setattr(cls, attr, caster(value))

        align = os.environ.get("OPTICAL_SLM_ENABLE_FEATURE_DOMAIN_ALIGNMENT")
        if align:
            cls.ENABLE_FEATURE_DOMAIN_ALIGNMENT = align.strip().lower() in {"1", "true", "yes", "on"}
        early_stop = os.environ.get("OPTICAL_SLM_ENABLE_DETECTOR_FOCUS_EARLY_STOP")
        if early_stop:
            cls.ENABLE_DETECTOR_FOCUS_EARLY_STOP = early_stop.strip().lower() in {"1", "true", "yes", "on"}
        bool_overrides = {
            "OPTICAL_SLM_VORTEX_ALTERNATE_CHARGE": "SLM_VORTEX_ALTERNATE_CHARGE",
        }
        for env_name, attr in bool_overrides.items():
            value = os.environ.get(env_name)
            if value:
                setattr(cls, attr, value.strip().lower() in {"1", "true", "yes", "on"})

    @classmethod
    def initialize(cls):
        cls.apply_runtime_overrides()
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
