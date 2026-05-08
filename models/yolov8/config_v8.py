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


def load_anchor_groups(anchor_yaml_path):
    with open(anchor_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    anchors = cfg.get("anchors")
    if anchors is None or not isinstance(anchors, list) or len(anchors) != 3:
        raise ValueError(f"'anchors' must contain exactly 3 layers: {anchor_yaml_path}")
    normalized = []
    for layer_idx, layer_anchors in enumerate(anchors):
        if not isinstance(layer_anchors, list) or len(layer_anchors) != 3:
            raise ValueError(f"Layer {layer_idx} must contain exactly 3 anchors: {anchor_yaml_path}")
        layer_values = []
        for anchor_idx, anchor in enumerate(layer_anchors):
            if not isinstance(anchor, (list, tuple)) or len(anchor) != 2:
                raise ValueError(f"Anchor {anchor_idx} in layer {layer_idx} must be [w, h]: {anchor_yaml_path}")
            w = int(anchor[0])
            h = int(anchor[1])
            if w <= 0 or h <= 0:
                raise ValueError(f"Anchor {anchor_idx} in layer {layer_idx} must be positive: {anchor_yaml_path}")
            layer_values.append([w, h])
        normalized.append(layer_values)
    return normalized


class ConfigYOLOv8Anchor:
    YAML_PATH = r"data\military\data.yaml"
    CLASS_NAMES = None
    NUM_CLASSES = None
    TEACHER_OUTPUT_DIR = r"output\OpticalTeacherYOLO_YOLOv8Head"
    LOG_ROOT_DIR = None
    LOG_FILE = None
    TIMESTAMP = None
    TRAIN_START_TIME = None

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_IDS = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    IMG_SIZE = 640
    BATCH_SIZE = 8
    STAGE1_LOCATE_EPOCHS = 60
    STAGE2_TEXTURE_EPOCHS = 80
    STAGE3_BALANCE_EPOCHS = 200
    EPOCHS = STAGE1_LOCATE_EPOCHS + STAGE2_TEXTURE_EPOCHS + STAGE3_BALANCE_EPOCHS

    STRIDES = [8, 16, 32]
    YOLOV8_BASE_CHANNELS = 32
    YOLOV8_C2F_BLOCKS = 2
    DEFAULT_ANCHORS = [
        [[26, 23], [47, 49], [100, 67]],
        [[103, 169], [203, 107], [351, 177]],
        [[241, 354], [534, 299], [568, 528]],
    ]
    ANCHOR_CONFIG_PATH = r"output/anchor_clustering/yolo_anchors.yaml"
    USE_EXTERNAL_ANCHORS = True
    ANCHORS = None
    ANCHOR_SOURCE = "default"
    POSITIVE_ANCHOR_IOU = 0.35
    MAX_POSITIVE_ANCHORS = 1
    NOOBJ_IGNORE_IOU = 0.68

    BOX_WEIGHT_BASE = 1.45
    OBJ_WEIGHT_BASE = 0.72
    NOOBJ_WEIGHT_BASE = 0.18
    CLS_WEIGHT_BASE = 0.35
    FOCAL_ALPHA = 0.28
    FOCAL_GAMMA = 1.6

    PHASE1_TEACHER_LR = 4e-4
    PHASE1_DETECTOR_LR = 3e-4
    PHASE2_TEACHER_LR = 2e-4
    PHASE2_DETECTOR_LR = 2e-4
    PHASE3_TEACHER_LR = 1.5e-4
    PHASE3_DETECTOR_LR = 1e-4
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 3e-5
    OPTIMIZER = "Adam"

    TEACHER_INIT_MODE = "scratch"
    TEACHER_INIT_CHECKPOINT = r"output/OpticalTeacherYOLO/teacher_final.pth"
    FREEZE_TEACHER = False
    SAVE_TEACHER_WEIGHTS = True

    FEATURE_HEATMAP_WEIGHT_BASE = 0.06
    FEATURE_CONTRAST_WEIGHT_BASE = 0.024
    FEATURE_SPARSITY_WEIGHT_BASE = 0.012
    FEATURE_TV_WEIGHT_BASE = 0.004
    PHASE1_TEACHER_GUIDANCE_SCALE = 1.5 # 第一阶段教师指导缩放
    PHASE2_TEACHER_GUIDANCE_SCALE = 3.0 # 第二阶段教师指导缩放
    PHASE3_TEACHER_GUIDANCE_SCALE = 2.5 # 第三阶段教师指导缩放
    FEATURE_FOREGROUND_TARGET = 0.74 # 前景目标阈值
    FEATURE_BACKGROUND_TARGET = 0.05 # 背景目标阈值
    FEATURE_HEATMAP_SIGMA = 0.35
    FEATURE_BOX_FILL_VALUE = 0.14
    FEATURE_CORE_FILL_VALUE = 0.40
    FEATURE_CORE_RATIO = 0.48
    FEATURE_LARGE_OBJECT_AREA = 160 * 160
    FEATURE_LARGE_BOX_FILL_BOOST = 0.05
    FEATURE_LARGE_CORE_FILL_BOOST = 0.08
    FEATURE_LARGE_CORE_RATIO_BOOST = 0.12

    SMALL_OBJ_AREA = 32 * 32
    LARGE_OBJ_AREA = 128 * 128
    SMALL_OBJ_WEIGHT = 0.8
    MEDIUM_OBJ_WEIGHT = 1.0
    LARGE_OBJ_WEIGHT = 1.5

    CONF_THRESH = 0.35 # 置信度阈值
    NMS_THRESH = 0.35 # NMS阈值
    MAX_DET = 20 # 最大检测框数量
    AGNOSTIC_NMS = False
    ENABLE_CONTAINMENT_SUPPRESSION = True
    CONTAINMENT_SUPPRESS_RATIO = 0.90
    CONTAINMENT_AREA_RATIO_LIMIT = 0.50
    ENABLE_WBF = True
    WBF_IOU_THRESH = 0.72
    WBF_SKIP_CONF_THRESH = 0.08
    WBF_SCORE_POWER = 1.5
    WBF_MAX_CANDIDATES = 12
    WBF_CENTER_DIST_FACTOR = 0.22
    WBF_SIZE_RATIO_LIMIT = 1.8

    VAL_INTERVAL = 2
    METRIC_IOU_THRESHOLD = 0.5
    METRIC_CONF_THRESH = 0.001 # 验证集置信度阈值
    METRIC_NMS_THRESH = 0.50 # 验证集NMS阈值
    METRIC_MAX_DET = 300 # 验证集最大检测框数量

    VIS_INTERVAL = 5 # 可视化间隔
    VIS_BATCH_SIZE = 4 # 可视化批量大小
    VIS_DPI = 130 # 可视化DPI
    VIS_DATASET_SPLIT = "val"
    VIS_SEED = 20260506
    VIS_CONF_THRESH = 0.35 # 可视化置信度阈值
    VIS_NMS_THRESH = 0.35 # 可视化NMS阈值
    VIS_MAX_DET = 20 # 可视化最大检测框数量
    VIS_SHOW_BEST_MATCHED_ANCHORS = True # 是否显示最佳匹配anchor
    VIS_MAX_GT_ANCHOR_OVERLAYS = 2 # 最大显示的GT anchor数量

    USE_CLASS_BALANCED_SAMPLER = True
    CLASS_BALANCE_POWER = 0.6
    MAX_CLASS_BALANCE_GAIN = 3.0
    MAJORITY_ONLY_IMAGE_WEIGHT = 0.45
    EMPTY_IMAGE_SAMPLE_WEIGHT = 0.7
    MIN_IMAGE_SAMPLE_WEIGHT = 0.35

    NUM_WORKERS = min(12, os.cpu_count() or 0)
    PIN_MEMORY = torch.cuda.is_available()
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 4
    ENABLE_CUDNN_BENCHMARK = True
    ENABLE_CHANNELS_LAST = True
    ENABLE_TF32 = True

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
    def apply_runtime_overrides(cls):
        overrides = {
            "OPTICAL_TEACHER_YAML_PATH": ("YAML_PATH", str),
            "OPTICAL_TEACHER_OUTPUT_DIR": ("TEACHER_OUTPUT_DIR", str),
            "OPTICAL_TEACHER_INIT_MODE": ("TEACHER_INIT_MODE", str),
            "OPTICAL_TEACHER_INIT_CHECKPOINT": ("TEACHER_INIT_CHECKPOINT", str),
            "OPTICAL_TEACHER_BATCH_SIZE": ("BATCH_SIZE", int),
            "OPTICAL_TEACHER_EPOCHS": ("EPOCHS", int),
            "OPTICAL_TEACHER_NUM_WORKERS": ("NUM_WORKERS", int),
        }
        for env_name, (attr, caster) in overrides.items():
            value = os.environ.get(env_name)
            if value:
                setattr(cls, attr, caster(value))

        freeze_teacher = os.environ.get("OPTICAL_TEACHER_FREEZE_TEACHER")
        if freeze_teacher:
            cls.FREEZE_TEACHER = freeze_teacher.strip().lower() in {"1", "true", "yes", "on"}

        use_sampler = os.environ.get("OPTICAL_TEACHER_USE_CLASS_BALANCED_SAMPLER")
        if use_sampler:
            cls.USE_CLASS_BALANCED_SAMPLER = use_sampler.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def initialize(cls):
        cls.apply_runtime_overrides()
        cls.YAML_PATH = resolve_project_path(cls.YAML_PATH)
        cls.TEACHER_OUTPUT_DIR = resolve_project_path(cls.TEACHER_OUTPUT_DIR)
        cls.CLASS_NAMES, cls.NUM_CLASSES = load_class_names(cls.YAML_PATH)
        cls.ANCHORS = [[anchor.copy() for anchor in layer] for layer in cls.DEFAULT_ANCHORS]
        cls.ANCHOR_SOURCE = "default"
        if cls.USE_EXTERNAL_ANCHORS:
            try:
                anchor_config_path = resolve_project_path(cls.ANCHOR_CONFIG_PATH)
                cls.ANCHORS = load_anchor_groups(anchor_config_path)
                cls.ANCHOR_SOURCE = anchor_config_path
            except Exception as exc:
                cls.ANCHORS = [[anchor.copy() for anchor in layer] for layer in cls.DEFAULT_ANCHORS]
                cls.ANCHOR_SOURCE = f"default (external load failed: {exc})"
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
        return mode if mode in {"scratch", "checkpoint"} else "scratch"

    @classmethod
    def get_teacher_init_checkpoint(cls):
        if cls.get_teacher_init_mode() != "checkpoint":
            return None
        checkpoint_path = str(cls.TEACHER_INIT_CHECKPOINT).strip()
        return checkpoint_path if checkpoint_path else None

    @classmethod
    def get_stage_settings(cls, epoch):
        if epoch < cls.STAGE1_LOCATE_EPOCHS:
            return {
                "phase": "locate_gt",
                "box_weight": cls.BOX_WEIGHT_BASE * 1.35,
                "obj_weight": cls.OBJ_WEIGHT_BASE * 1.15,
                "noobj_weight": cls.NOOBJ_WEIGHT_BASE,
                "cls_weight": cls.CLS_WEIGHT_BASE * 0.72,
                "size_weights": {
                    "small": cls.SMALL_OBJ_WEIGHT,
                    "medium": cls.MEDIUM_OBJ_WEIGHT,
                    "large": cls.LARGE_OBJ_WEIGHT,
                },
                "teacher_lr": cls.PHASE1_TEACHER_LR,
                "detector_lr": cls.PHASE1_DETECTOR_LR,
                "teacher_guidance_scale": cls.PHASE1_TEACHER_GUIDANCE_SCALE,
            }
        if epoch < cls.STAGE1_LOCATE_EPOCHS + cls.STAGE2_TEXTURE_EPOCHS:
            return {
                "phase": "texture_detail",
                "box_weight": cls.BOX_WEIGHT_BASE * 1.02,
                "obj_weight": cls.OBJ_WEIGHT_BASE,
                "noobj_weight": cls.NOOBJ_WEIGHT_BASE * 0.92,
                "cls_weight": cls.CLS_WEIGHT_BASE * 1.35,
                "size_weights": {
                    "small": 1.05,
                    "medium": 1.0,
                    "large": 1.10,
                },
                "teacher_lr": cls.PHASE2_TEACHER_LR,
                "detector_lr": cls.PHASE2_DETECTOR_LR,
                "teacher_guidance_scale": cls.PHASE2_TEACHER_GUIDANCE_SCALE,
            }
        return {
            "phase": "balance_refine",
            "box_weight": cls.BOX_WEIGHT_BASE,
            "obj_weight": cls.OBJ_WEIGHT_BASE,
            "noobj_weight": cls.NOOBJ_WEIGHT_BASE * 1.08,
            "cls_weight": cls.CLS_WEIGHT_BASE * 1.12,
            "size_weights": {
                "small": 1.0,
                "medium": 1.0,
                "large": 1.0,
            },
            "teacher_lr": cls.PHASE3_TEACHER_LR,
            "detector_lr": cls.PHASE3_DETECTOR_LR,
            "teacher_guidance_scale": cls.PHASE3_TEACHER_GUIDANCE_SCALE,
        }

    @classmethod
    def get_dynamic_weights(cls, epoch):
        return cls.get_stage_settings(epoch)

    @classmethod
    def get_teacher_guidance_weights(cls, stage_settings=None):
        if stage_settings is None:
            stage_settings = cls.get_stage_settings(0)
        scale = float(stage_settings["teacher_guidance_scale"])
        return {
            "heatmap": cls.FEATURE_HEATMAP_WEIGHT_BASE * scale,
            "contrast": cls.FEATURE_CONTRAST_WEIGHT_BASE * scale,
            "sparsity": cls.FEATURE_SPARSITY_WEIGHT_BASE * scale,
            "tv": cls.FEATURE_TV_WEIGHT_BASE * scale,
        }

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
        print(f"Image Size: {cls.IMG_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Num Classes: {cls.NUM_CLASSES}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Class Names: {cls.CLASS_NAMES}")
        print("=" * 80)
