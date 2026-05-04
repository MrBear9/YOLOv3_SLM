import os
from datetime import datetime

import numpy as np
import torch

from models.yolov8.config_v8 import load_anchor_groups, load_class_names, resolve_project_path


class ConfigSLM:
    YAML_PATH = r"data\military\data.yaml"
    CLASS_NAMES = None
    NUM_CLASSES = None
    OUTPUT_DIR = r"output\OpticalSLM_YOLOv8Head_student"
    VISUALIZATION_DIR = None
    LOG_ROOT_DIR = None
    LOG_FILE = None
    TIMESTAMP = None
    TRAIN_START_TIME = None

    TEACHER_DETECTOR_CHECKPOINT = r"output\OpticalTeacherYOLO_YOLOv8Head\teacher_detector_best.pth"
    DETECTOR_INIT_MODE = "checkpoint"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_IDS = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    IMG_SIZE = 640
    BATCH_SIZE = 8
    STUDENT_ONLY_EPOCHS = 60
    DETECTOR_ONLY_EPOCHS = 120
    JOINT_EPOCHS = 60
    EPOCHS = STUDENT_ONLY_EPOCHS + DETECTOR_ONLY_EPOCHS + JOINT_EPOCHS

    STRIDES = [8, 16, 32]
    YOLOV8_BASE_CHANNELS = 32
    YOLOV8_C2F_BLOCKS = 2
    DEFAULT_ANCHORS = [
        [[26, 23], [47, 49], [100, 67]],
        [[103, 169], [203, 107], [351, 177]],
        [[241, 354], [534, 299], [568, 528]],
    ]
    ANCHOR_CONFIG_PATH = r"output\anchor_clustering\yolo_anchors.yaml"
    USE_EXTERNAL_ANCHORS = True
    ANCHORS = None
    ANCHOR_SOURCE = "default"
    POSITIVE_ANCHOR_IOU = 0.35
    MAX_POSITIVE_ANCHORS = 1
    NOOBJ_IGNORE_IOU = 0.68
    SMALL_OBJ_AREA = 32 * 32
    LARGE_OBJ_AREA = 128 * 128
    SMALL_OBJ_WEIGHT = 0.8
    MEDIUM_OBJ_WEIGHT = 1.0
    LARGE_OBJ_WEIGHT = 1.5

    BOX_WEIGHT_BASE = 1.32
    OBJ_WEIGHT_BASE = 0.56
    NOOBJ_WEIGHT_BASE = 0.26
    CLS_WEIGHT_BASE = 0.32
    FOCAL_ALPHA = 0.28
    FOCAL_GAMMA = 1.6

    STUDENT_LR = 1e-3
    PHASE_PARAM_LR = 2e-3
    DETECTOR_LR = 4e-4
    JOINT_STUDENT_LR = 5e-5
    JOINT_PHASE_PARAM_LR = 1e-4
    JOINT_DETECTOR_LR = 2e-4
    WEIGHT_DECAY = 3e-5

    FEATURE_LOSS_WEIGHT_STUDENT = 1.0
    DETECTION_LOSS_WEIGHT_STUDENT = 0.0
    DETECTION_LOSS_WEIGHT_DETECTOR = 1.0
    FEATURE_LOSS_WEIGHT_JOINT = 0.35
    DETECTION_LOSS_WEIGHT_JOINT = 1.2
    RESPONSE_LOSS_WEIGHT = 0.0

    LOSS_FULL_WEIGHT = 0.2
    LOSS_LOW1_WEIGHT = 0.8
    LOSS_LOW2_WEIGHT = 0.4
    LOSS_SSIM_WEIGHT = 0.3
    LOSS_GRAD_WEIGHT = 0.25
    LOSS_FREQ_WEIGHT = 0.15
    LOSS_PHASE_SMOOTH_WEIGHT = 0.005
    LOSS_PHASE_DIVERSITY_WEIGHT = 0.20
    PHASE_STD_TARGET = 1.20
    PHASE_SPAN_TARGET = 5.20
    PHASE_BEST_MIN_STD = 0.45
    PHASE_BEST_MIN_SPAN = np.pi

    WAVELENGTH = 532e-9
    PIXEL_SIZE = 6.4e-6
    PROP_DISTANCE_1 = 0.01
    PROP_DISTANCE_2 = 0.02
    SLM_MODE = "phase"
    RESOLUTION = (640, 640)
    OPTICAL_FIELD_EPS = 1e-8
    OPTICAL_NORM_EPS = 1e-6
    ENABLE_STUDENT_NORM = True

    CONF_THRESH = 0.5
    NMS_THRESH = 0.35
    MAX_DET = 5
    AGNOSTIC_NMS = False
    ENABLE_CONTAINMENT_SUPPRESSION = False
    ENABLE_WBF = False
    METRIC_IOU_THRESHOLD = 0.5

    VAL_INTERVAL = 1  # 验证间隔
    VIS_INTERVAL = 5  # 可视化间隔
    VIS_BATCH_SIZE = 4
    VIS_DPI = 130
    VIS_DATASET_SPLIT = "val"
    VIS_SEED = 20260504
    VIS_CONF_THRESH = 0.5  # 可视化时的置信度阈值
    VIS_NMS_THRESH = 0.35  # 可视化时的NMS阈值
    VIS_MAX_DET = 5  # 可视化时的最大检测框数量
    VIS_SHOW_BEST_MATCHED_ANCHORS = False  # 是否显示最佳匹配anchor
    VIS_MAX_GT_ANCHOR_OVERLAYS = 2  # 最大显示的GT anchor数量
    NUM_WORKERS = min(8, os.cpu_count() or 0)
    PIN_MEMORY = torch.cuda.is_available()
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 4
    ENABLE_CHANNELS_LAST = True
    ENABLE_TF32 = True
    ENABLE_CUDNN_BENCHMARK = True

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
            "OPTICAL_SLM_BATCH_SIZE": ("BATCH_SIZE", int),
            "OPTICAL_SLM_EPOCHS": ("EPOCHS", int),
            "OPTICAL_SLM_NUM_WORKERS": ("NUM_WORKERS", int),
        }
        for env_name, (attr, caster) in overrides.items():
            value = os.environ.get(env_name)
            if value:
                setattr(cls, attr, caster(value))

    @classmethod
    def initialize(cls):
        cls.apply_runtime_overrides()
        cls.YAML_PATH = resolve_project_path(cls.YAML_PATH)
        cls.OUTPUT_DIR = resolve_project_path(cls.OUTPUT_DIR)
        cls.TEACHER_DETECTOR_CHECKPOINT = resolve_project_path(cls.TEACHER_DETECTOR_CHECKPOINT)
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
    def get_student_best_path(cls):
        return os.path.join(cls.OUTPUT_DIR, "optical_student_best.pth")

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
