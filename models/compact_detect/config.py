import os
from datetime import datetime

import torch
import yaml

from models.SLM.config_slm import ConfigSLM


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


class ConfigCompactDetect(ConfigSLM):
    """Configuration for the independent compact optical detector."""

    OUTPUT_DIR = r"output/SLM_compact_detect"
    VISUALIZATION_DIR = None
    LOG_ROOT_DIR = None
    LOG_FILE = None
    TIMESTAMP = None
    TRAIN_START_TIME = None

    EPOCHS = 220
    BATCH_SIZE = 12
    IMG_SIZE = 640
    OUTPUT_STRIDE = 4

    COMPACT_BASE_CH = 16
    COMPACT_DILATIONS = (1, 2, 4)
    COMPACT_HEAD_CH = 32
    COMPACT_PRETRAINED_STUDENT = r""
    COMPACT_TRAIN_STUDENT = True
    COMPACT_TEACHER_WARMUP_EPOCHS = 5
    COMPACT_TEACHER_WARMUP_WEIGHT = 1.0
    COMPACT_TEACHER_WARMUP_RAW_STUDENT = True
    COMPACT_PHASE_LR = 4e-4
    COMPACT_DETECTOR_LR = 3e-4
    COMPACT_WEIGHT_DECAY = 3e-5
    COMPACT_GRAD_CLIP_NORM = 5.0

    HEATMAP_LOSS_WEIGHT = 1.0
    WH_LOSS_WEIGHT = 0.08
    OFFSET_LOSS_WEIGHT = 1.0
    MIN_HEATMAP_RADIUS = 1
    HEATMAP_RADIUS_SCALE = 0.35

    CONF_THRESH = 0.30
    NMS_THRESH = 0.45
    MAX_DET = 100
    DECODE_PRE_NMS_TOPK = 400
    METRIC_CONF_THRESH = 0.01
    METRIC_NMS_THRESH = 0.50
    METRIC_MAX_DET = 100
    METRIC_PRE_NMS_TOPK = 300
    METRIC_IOU_THRESHOLD = 0.5

    VAL_INTERVAL = 1
    VIS_SEED = 20260709
    VIS_INTERVAL = 1
    VIS_MAX_IMAGES = 2
    VIS_FILE_INTERVAL = 10
    VIS_FILE_MAX_IMAGES = 3
    SAVE_INTERVAL = 20
    ENABLE_AMP = True
    AMP_DTYPE = "float16"
    ENABLE_CHANNELS_LAST = False

    SINGLE_IMAGE_TRAINING = False
    SINGLE_IMAGE_PATH = r"data\military\test\images\train_025317.jpg"
    SINGLE_IMAGE_LABEL_PATH = r"data\military\test\labels\train_025317.txt"
    SINGLE_IMAGE_REPEAT = 1000

    EPOCH_TABLE_EPOCH_WIDTH = 8
    EPOCH_TABLE_PHASE_WIDTH = 14
    EPOCH_TABLE_TRAIN_LOSS_WIDTH = 13
    EPOCH_TABLE_VAL_LOSS_WIDTH = 13
    EPOCH_TABLE_METRIC_WIDTH = 11
    EPOCH_TABLE_LR_WIDTH = 12
    EPOCH_TABLE_BEST_WIDTH = 8
    EPOCH_TABLE_BEST_MARK = "Yes"
    SKIP_FILE_LOG_MESSAGES = ("best checkpoint updated",)

    @classmethod
    def apply_runtime_overrides(cls):
        super().apply_runtime_overrides()
        overrides = {
            "OPTICAL_COMPACT_OUTPUT_DIR": ("OUTPUT_DIR", str),
            "OPTICAL_COMPACT_EPOCHS": ("EPOCHS", int),
            "OPTICAL_COMPACT_BATCH_SIZE": ("BATCH_SIZE", int),
            "OPTICAL_COMPACT_BASE_CH": ("COMPACT_BASE_CH", int),
            "OPTICAL_COMPACT_HEAD_CH": ("COMPACT_HEAD_CH", int),
            "OPTICAL_COMPACT_TEACHER_WARMUP_EPOCHS": ("COMPACT_TEACHER_WARMUP_EPOCHS", int),
            "OPTICAL_COMPACT_TEACHER_WARMUP_WEIGHT": ("COMPACT_TEACHER_WARMUP_WEIGHT", float),
            "OPTICAL_COMPACT_PHASE_LR": ("COMPACT_PHASE_LR", float),
            "OPTICAL_COMPACT_DETECTOR_LR": ("COMPACT_DETECTOR_LR", float),
            "OPTICAL_COMPACT_PRETRAINED_STUDENT": ("COMPACT_PRETRAINED_STUDENT", str),
            "OPTICAL_COMPACT_CONF_THRESH": ("CONF_THRESH", float),
            "OPTICAL_COMPACT_NMS_THRESH": ("NMS_THRESH", float),
            "OPTICAL_COMPACT_MAX_DET": ("MAX_DET", int),
            "OPTICAL_COMPACT_METRIC_CONF_THRESH": ("METRIC_CONF_THRESH", float),
            "OPTICAL_COMPACT_METRIC_MAX_DET": ("METRIC_MAX_DET", int),
            "OPTICAL_COMPACT_METRIC_PRE_NMS_TOPK": ("METRIC_PRE_NMS_TOPK", int),
            "OPTICAL_COMPACT_VIS_SEED": ("VIS_SEED", int),
            "OPTICAL_COMPACT_VIS_INTERVAL": ("VIS_INTERVAL", int),
            "OPTICAL_COMPACT_VIS_MAX_IMAGES": ("VIS_MAX_IMAGES", int),
            "OPTICAL_COMPACT_VIS_FILE_INTERVAL": ("VIS_FILE_INTERVAL", int),
            "OPTICAL_COMPACT_VIS_FILE_MAX_IMAGES": ("VIS_FILE_MAX_IMAGES", int),
            "OPTICAL_COMPACT_SINGLE_IMAGE_PATH": ("SINGLE_IMAGE_PATH", str),
            "OPTICAL_COMPACT_SINGLE_IMAGE_LABEL_PATH": ("SINGLE_IMAGE_LABEL_PATH", str),
            "OPTICAL_COMPACT_SINGLE_IMAGE_REPEAT": ("SINGLE_IMAGE_REPEAT", int),
        }
        for env_name, (attr, caster) in overrides.items():
            value = os.environ.get(env_name)
            if value:
                setattr(cls, attr, caster(value))
        train_student = os.environ.get("OPTICAL_COMPACT_TRAIN_STUDENT")
        if train_student:
            cls.COMPACT_TRAIN_STUDENT = train_student.strip().lower() in {"1", "true", "yes", "on"}
        warmup_raw = os.environ.get("OPTICAL_COMPACT_TEACHER_WARMUP_RAW_STUDENT")
        if warmup_raw:
            cls.COMPACT_TEACHER_WARMUP_RAW_STUDENT = warmup_raw.strip().lower() in {"1", "true", "yes", "on"}
        single_image = os.environ.get("OPTICAL_COMPACT_SINGLE_IMAGE_TRAINING")
        if single_image:
            cls.SINGLE_IMAGE_TRAINING = single_image.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def initialize(cls):
        cls.apply_runtime_overrides()
        cls.YAML_PATH = resolve_project_path(cls.YAML_PATH)
        cls.OUTPUT_DIR = resolve_project_path(cls.OUTPUT_DIR)
        cls.TEACHER_DETECTOR_CHECKPOINT = resolve_project_path(cls.TEACHER_DETECTOR_CHECKPOINT)
        cls.SLM_INIT_CHECKPOINT = resolve_project_path(cls.SLM_INIT_CHECKPOINT)
        cls.COMPACT_PRETRAINED_STUDENT = resolve_project_path(cls.COMPACT_PRETRAINED_STUDENT)
        cls.SINGLE_IMAGE_PATH = resolve_project_path(cls.SINGLE_IMAGE_PATH)
        cls.SINGLE_IMAGE_LABEL_PATH = resolve_project_path(cls.SINGLE_IMAGE_LABEL_PATH)
        cls.CLASS_NAMES, cls.NUM_CLASSES = load_class_names(cls.YAML_PATH)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        cls.LOG_ROOT_DIR = os.path.join(cls.OUTPUT_DIR, "logs")
        cls.VISUALIZATION_DIR = os.path.join(cls.OUTPUT_DIR, "visualizations")
        os.makedirs(cls.LOG_ROOT_DIR, exist_ok=True)
        os.makedirs(cls.VISUALIZATION_DIR, exist_ok=True)
        cls.TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.TRAIN_START_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cls.LOG_FILE = os.path.join(cls.LOG_ROOT_DIR, f"training_log_{cls.TIMESTAMP}.txt")
        cls.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cls.GPU_IDS = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

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
    def get_detector_best_path(cls):
        return os.path.join(cls.OUTPUT_DIR, "compact_detector_best.pth")

    @classmethod
    def get_detector_final_path(cls):
        return os.path.join(cls.OUTPUT_DIR, "compact_detector_final.pth")
