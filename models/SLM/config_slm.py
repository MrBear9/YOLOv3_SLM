import os
from datetime import datetime

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
    # Random-init 10 cm + 10 cm run using the v5 primary schedule.
    OUTPUT_DIR = r"output/SLM_Tv2_light_10cm_roi_v10_scratch_v5_schedule"
    VISUALIZATION_DIR = None
    LOG_ROOT_DIR = None
    LOG_FILE = None
    TIMESTAMP = None
    TRAIN_START_TIME = None

    TEACHER_DETECTOR_CHECKPOINT = r"output/Tv2_light_20cm/teacher_detector_best.pth"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_IDS = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

    # =========================================================================
    # Training scale
    # =========================================================================
    BATCH_SIZE = 8
    ANCHOR_FREE_STRIDES = [4, 8, 16, 32]

    # SLM_INIT_MODE is inherited from OpticalConfig and remains ``random``.
    # Fit the optical mapping and detector from zero before any task-aware
    # phase update; no previous student or detector checkpoint is loaded.
    PHASE_FOCUS_EPOCHS = 100
    DETECTOR_FOCUS_EPOCHS = 80
    # Keep optional SLM2-only refinement disabled in the primary scratch run.
    PHASE_REFINE_EPOCHS = 0
    # Preserve the v5 final 30-epoch joint fitting stage for a direct schedule
    # comparison; it starts from the detector-focus validation best pair.
    JOINT_FIT_EPOCHS = 30
    # Deployment normalization remains a separate hardware experiment.
    NORM_JOINT_EPOCHS = 0
    EPOCHS = (
        PHASE_FOCUS_EPOCHS + DETECTOR_FOCUS_EPOCHS + PHASE_REFINE_EPOCHS
        + JOINT_FIT_EPOCHS + NORM_JOINT_EPOCHS
    )

    # Per-layer block/freq overrides: see OpticalConfig accessors
    #   phase_block_grid(layer_idx), phase_mlp_num_freqs(layer_idx), …
    # Per-layer trainable flag and LR multipliers: see OpticalConfig accessors
    #   is_trainable(layer_idx), layer_lr_mult(layer_idx, stage)

    # Restore the validation-best paired state before every detector-aware
    # optical stage, so a regressing ablation cannot affect the next one.
    RESTORE_BEST_PAIRED_BEFORE_REFINEMENT = True

    # Per-layer vortex charge and radial curvature: see OpticalConfig.vortex_init.
    # Vortex is always one global phase singularity; it is never tiled.

    # =========================================================================
    # TEACHER_ARCH  (must match teacher-training checkpoint)
    # Options: "convteacher"/"v1", "convteacher_v2"/"v2", "convteacher_v3"/"v3".
    # =========================================================================
    TEACHER_ARCH = "convteacher_v2"

    # -------- TEACHER_ARCH = "convteacher" / "v1" --------
    TEACHER_V1_BASE_CHANNELS = 32
    TEACHER_V1_C2F_BLOCKS = 3

    # -------- TEACHER_ARCH = "convteacher_v2" (phase-only SLM/ASM teacher) --------
    TEACHER_V2_BASE_CHANNELS = 32
    TEACHER_V2_C2F_BLOCKS = 3
    TEACHER_V2_FOURIER_BANDS = 8
    TEACHER_V2_FOURIER_LOW_PASS_SIGMA = 0.5
    # These remain fixed to the loaded V2 teacher checkpoint. Student distance
    # is intentionally different in this 20 cm optical-geometry ablation.
    TEACHER_V2_NUM_SLM_LAYERS = 2
    TEACHER_V2_WAVELENGTH = 532e-9
    TEACHER_V2_PIXEL_SIZE = 6.4e-6
    TEACHER_V2_PROP_DISTANCES = (0.20, 0.20)

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
    LOSS_FULL_WEIGHT = 0.20
    LOSS_LOW1_WEIGHT = 0.25
    LOSS_LOW2_WEIGHT = 0.10
    LOSS_SSIM_WEIGHT = 0.25
    LOSS_GRAD_WEIGHT = 0.10
    LOSS_FREQ_WEIGHT = 0.00
    LOSS_PEARSON_WEIGHT = 0.10
    FEATURE_LOSS_PREFILTER_KERNEL = 1
    ENABLE_FEATURE_DOMAIN_ALIGNMENT = True
    # Options: "mean_std", "minmax"/"min_max", "none".
    FEATURE_DOMAIN_ALIGN_MODE = "minmax"
    # Preserve detection-relevant local structure rather than only matching
    # full-frame feature statistics. Boxes are expanded slightly for context.
    ENABLE_TARGET_ROI_FEATURE_LOSS = True
    LOSS_TARGET_ROI_WEIGHT = 0.75
    TARGET_ROI_CONTEXT_SCALE = 1.20
    TARGET_ROI_FEATHER_KERNEL = 9
    # Macro mAP gives each class equal value. Aircraft and soldier are the
    # current weak classes; warship already has strong AP and large boxes.
    TARGET_ROI_CLASS_WEIGHTS = {0: 1.0, 1: 1.25, 2: 1.35, 3: 0.75}

    # =========================================================================
    # Stage loss weights
    # =========================================================================
    FEATURE_LOSS_WEIGHT_PHASE_FOCUS = 1.1
    DETECTION_LOSS_WEIGHT_PHASE_FOCUS = 0.0
    RESPONSE_LOSS_WEIGHT_PHASE_FOCUS = 0.0
    FEATURE_LOSS_WEIGHT_DETECTOR_FOCUS = 0.0
    DETECTION_LOSS_WEIGHT_DETECTOR_FOCUS = 1.0
    RESPONSE_LOSS_WEIGHT_DETECTOR_FOCUS = 0.0

    # Preserve optical image/ROI correspondence while the frozen detector
    # supplies task gradients to the final SLM.
    FEATURE_LOSS_WEIGHT_PHASE_REFINE = 0.50
    DETECTION_LOSS_WEIGHT_PHASE_REFINE = 1.00
    RESPONSE_LOSS_WEIGHT_PHASE_REFINE = 0.00
    PHASE_REGULARIZATION_WEIGHT_PHASE_REFINE = 0.00

    FEATURE_LOSS_WEIGHT_JOINT = 0.10
    DETECTION_LOSS_WEIGHT_JOINT = 1.00
    # Per-image max-normalized response matching did not improve validation
    # mAP, so it remains disabled for this route.
    RESPONSE_LOSS_WEIGHT_JOINT = 0.00
    PHASE_REGULARIZATION_WEIGHT_JOINT = 0.00

    FEATURE_LOSS_WEIGHT_NORM_JOINT = 0.20
    DETECTION_LOSS_WEIGHT_NORM_JOINT = 1.00
    RESPONSE_LOSS_WEIGHT_NORM_JOINT = 0.00
    PHASE_REGULARIZATION_WEIGHT_NORM_JOINT = 0.03

    # All phase constraints are evaluated on exp(j * phase), so 0 and 2pi
    # remain physically identical. Circular variance is a bounded modulation
    # measure; TV/high-pass suppress non-deployable pixel noise.
    PHASE_TARGET_CIRCULAR_VARIANCE = 0.25
    PHASE_CIRCULAR_TV_WEIGHT = 0.04
    PHASE_CIRCULAR_HIGH_FREQ_WEIGHT = 0.02
    PHASE_CIRCULAR_VARIANCE_WEIGHT = 1.00
    PHASE_HIGH_FREQ_KERNEL = 5

    # =========================================================================
    # Optimizer & LR schedule
    # =========================================================================
    PHASE_FOCUS_PHASE_PARAM_LR = 3e-3
    # v8's best checkpoint was reached with this clipped SLM2-only step.
    PHASE_REFINE_PHASE_PARAM_LR = 1e-4
    DETECTOR_LR = 3e-4
    JOINT_PHASE_PARAM_LR = 3e-4
    JOINT_DETECTOR_LR = 1e-5
    NORM_JOINT_PHASE_PARAM_LR = 2e-4
    NORM_JOINT_DETECTOR_LR = 5e-5
    PHASE_GRAD_CLIP_NORM = 2.0
    WEIGHT_DECAY = 3e-5
    PHASE_WEIGHT_DECAY = 0.0
    # Options: "CosineAnnealingLR", "none".
    LR_SCHEDULER = "CosineAnnealingLR"
    ETA_MIN = 1e-5
    # Joint fitting keeps detector adaptation conservative while phase updates
    # decay only to this floor instead of the global scheduler's 1e-5 floor.
    JOINT_LR_SCHEDULER = "cosine_phase_floor"
    JOINT_PHASE_ETA_MIN = 5e-5

    # Preserve all 30 v5 joint epochs for a direct schedule comparison.
    ENABLE_DETECTOR_FOCUS_EARLY_STOP = False
    DETECTOR_FOCUS_EARLY_STOP_PATIENCE = 20
    # SLM ablations often improve by <0.002 mAP; retain real improvements.
    DETECTOR_FOCUS_EARLY_STOP_MIN_DELTA = 1e-5

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
    # Detector fitting converges slowly, while SLM2 refinements peak early.
    # Use the cheaper cadence for the long stage and denser selection for the
    # short, directly comparable residual ablations.
    VAL_INTERVAL = 5
    PHASE_REFINE_VAL_INTERVAL = 2
    METRIC_IOU_THRESHOLD = 0.5
    # Phase-focus has no detection objective and its mAP is necessarily zero;
    # skip a full validation sweep there. Detector-aware stages still validate.
    VALIDATE_PHASE_ONLY = False

    # Expensive per-parameter TensorBoard diagnostics are sampled at this
    # interval. Core losses, mAP, per-class AP, and SLM summaries remain
    # available every epoch.
    PARAMETER_MONITOR_INTERVAL = 10
    # 0 records phase gradient norms from the final train batch only.
    PHASE_GRAD_MONITOR_BATCH_INTERVAL = 0

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

    # Fix model initialization, optical random phase, sampler order, and
    # DataLoader workers so small mAP changes are comparable across runs.
    TRAIN_SEED = 20260813
    DETERMINISTIC_TRAINING = True

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
    ENABLE_CUDNN_BENCHMARK = False

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
        cls.EPOCHS = (
            cls.PHASE_FOCUS_EPOCHS + cls.DETECTOR_FOCUS_EPOCHS
            + cls.PHASE_REFINE_EPOCHS + cls.JOINT_FIT_EPOCHS
            + cls.NORM_JOINT_EPOCHS
        )
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
                "phase_regularization": 0.0,
            }
        if stage_name == "phase_refine":
            return {
                "feature": cls.FEATURE_LOSS_WEIGHT_PHASE_REFINE,
                "detection": cls.DETECTION_LOSS_WEIGHT_PHASE_REFINE,
                "response": cls.RESPONSE_LOSS_WEIGHT_PHASE_REFINE,
                "phase_regularization": cls.PHASE_REGULARIZATION_WEIGHT_PHASE_REFINE,
            }
        if stage_name == "detector_focus":
            return {
                "feature": cls.FEATURE_LOSS_WEIGHT_DETECTOR_FOCUS,
                "detection": cls.DETECTION_LOSS_WEIGHT_DETECTOR_FOCUS,
                "response": cls.RESPONSE_LOSS_WEIGHT_DETECTOR_FOCUS,
                "phase_regularization": 0.0,
            }
        if stage_name == "norm_joint":
            return {
                "feature": cls.FEATURE_LOSS_WEIGHT_NORM_JOINT,
                "detection": cls.DETECTION_LOSS_WEIGHT_NORM_JOINT,
                "response": cls.RESPONSE_LOSS_WEIGHT_NORM_JOINT,
                "phase_regularization": cls.PHASE_REGULARIZATION_WEIGHT_NORM_JOINT,
            }
        return {
            "feature": cls.FEATURE_LOSS_WEIGHT_JOINT,
            "detection": cls.DETECTION_LOSS_WEIGHT_JOINT,
            "response": cls.RESPONSE_LOSS_WEIGHT_JOINT,
            "phase_regularization": cls.PHASE_REGULARIZATION_WEIGHT_JOINT,
        }

    @classmethod
    def validation_interval(cls, stage_name):
        """Return the stage-specific validation cadence, with a safe default."""
        stage_key = f"{str(stage_name).upper()}_VAL_INTERVAL"
        return max(int(getattr(cls, stage_key, cls.VAL_INTERVAL)), 1)

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
