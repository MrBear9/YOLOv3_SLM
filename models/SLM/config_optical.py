"""光学传播层父类配置 — per-layer 配置用字典结构，消除 key 爆炸。

使用方式: ``ConfigSLM(OpticalConfig)`` 继承，然后通过访问器方法取值。
"""

import math

class OpticalConfig:
    """光学传播的共享默认值 + per-layer 可覆盖配置。

    Per-layer 配置通过 ``{layer_idx: value}`` 字典组织，访问器方法自动
    fallback 到 layer 1 的值。新增 3 层/4 层只需改 ``NUM_LAYERS`` +
    在对应 dict 中加 key，无需新增任何配置项。
    """

    # ═══════════════════════════════════════════════════════════════════════
    # 传播层数  2 | 3 | 4
    # ═══════════════════════════════════════════════════════════════════════
    NUM_LAYERS = 2

    # Optical canvas and shared physical constants.
    # Spatial sizes are always expressed as (height, width).
    RESOLUTION = (640, 640)
    WAVELENGTH = 532e-9
    PIXEL_SIZE = 6.4e-6
    SLM_MODE = "phase"
    INPUT_INTENSITY_MODE = "srgb"
    OPTICAL_FIELD_EPS = 1e-8
    OPTICAL_NORM_EPS = 1e-6

    # Keep the full-resolution direct phase primary; the pyramid is a
    # deliberately smaller multi-scale residual during static-phase fitting.
    SLM_PHASE_PARAM_MODE = "direct_sgd_pyramid"
    SLM_DIRECT_SGD_PYRAMID_SCALE = 0.5
    # Optional virtual multi-head optical path.
    SLM_MULTI_HEAD_ENABLED = False
    SLM_MULTI_HEAD_NUM_HEADS = 2
    SLM_MULTI_HEAD_FUSION = "mean"

    # Student optical output normalization.
    ENABLE_STUDENT_NORM = True
    STUDENT_NORM_SCHEDULE = "norm_joint_only"
    STUDENT_NORM_MODE = "percentile"
    STUDENT_NORM_PERCENTILE = 0.990
    STUDENT_OUTPUT_CLAMP_MAX = 3.5
    STUDENT_OUTPUT_BLUR_KERNEL = 1

    # SLM phase initialization and hardware export.
    SLM_INIT_MODE = "random"
    SLM_DIRECT_SGD_INIT_RANGE_RAD = 0.5
    SLM_INIT_NOISE_STD = 0.02
    SLM_INIT_CHECKPOINT = r"output/SLM_Tv2_light_phase_refine/detector_best.pth"
    SLM_PHASE_LEVELS = 256
    SIMULATE_PHASE_QUANTIZATION = False
    SLM_GRAY_INVERTED = True
    SLM_EXPORT_PHASE_OFFSET_RAD = float(math.pi)
    SLM_GRAY_TO_PHASE_LUT = r""

    # ═══════════════════════════════════════════════════════════════════════
    # 共享默认值（所有层共用，除非 per-layer dict 覆盖）
    # ═══════════════════════════════════════════════════════════════════════
    PHASE_NUM_SCALES = 5
    PHASE_USE_BLOCKWISE = True
    PHASE_MLP_HIDDEN = 32
    PHASE_MLP_LAYERS = 2

    # ═══════════════════════════════════════════════════════════════════════
    # Per-layer 配置  {layer_idx (1-based): value}
    # ═══════════════════════════════════════════════════════════════════════
    # Student geometry for the current ablation: SLM1 -> 10 cm -> SLM2 -> 10 cm.
    # Teacher V2 geometry is configured separately in ConfigSLM as 20/20 cm.
    PROP_DISTANCE = {1: 0.10, 2: 0.10, 3: 0.20, 4: 0.20}
    TRAIN_LAYER  = {1: True, 2: True, 3: True, 4: True}

    # Phase field block parameters
    # Accept an int for square panels or (rows, cols) for rectangular panels.
    PHASE_BLOCK_GRID         = {1: 6, 2: 4, 3: 5, 4: 7}
    PHASE_BLOCK_OVERLAP      = {1: 8, 2: 8, 3: 8, 4: 8}
    PHASE_BLOCK_INNER_SCALES = {1: 3, 2: 2, 3: 2, 4: 2}
    PHASE_MLP_NUM_FREQS      = {1: 10, 2: 6, 3: 6, 4: 6}

    # Per-layer LR multipliers (× base LR)
    PHASE_FOCUS_LR_MULT = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    JOINT_LR_MULT       = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    NORM_JOINT_LR_MULT  = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    # Vortex 初始化
    VORTEX_CHARGE       = {1: 3.0, 2: 3.0,  3: 4.0, 4: 4.0}
    VORTEX_RADIAL_SCALE = {1: 0.35, 2: 0.25, 3: 0.35, 4: 0.45}

    # DH-PSF 初始化

    # ═══════════════════════════════════════════════════════════════════════
    # 便捷访问器（带 fallback）
    # ═══════════════════════════════════════════════════════════════════════

    @classmethod
    def _layer_val(cls, dct, layer_idx, default=None):
        """从 dict 取 per-layer 值，fallback 到 layer 1 的值或 default。"""
        if layer_idx in dct:
            return dct[layer_idx]
        if default is not None:
            return default
        return dct.get(1)

    @classmethod
    def prop_distance(cls, layer_idx):
        return cls._layer_val(cls.PROP_DISTANCE, layer_idx, 0.10)

    @classmethod
    def is_trainable(cls, layer_idx):
        return cls._layer_val(cls.TRAIN_LAYER, layer_idx, True)

    @classmethod
    def phase_block_grid(cls, layer_idx):
        return cls._layer_val(cls.PHASE_BLOCK_GRID, layer_idx, 4)

    @classmethod
    def phase_block_overlap(cls, layer_idx):
        return cls._layer_val(cls.PHASE_BLOCK_OVERLAP, layer_idx, 8)

    @classmethod
    def phase_block_inner_scales(cls, layer_idx):
        return cls._layer_val(cls.PHASE_BLOCK_INNER_SCALES, layer_idx, 2)

    @classmethod
    def phase_mlp_num_freqs(cls, layer_idx):
        return cls._layer_val(cls.PHASE_MLP_NUM_FREQS, layer_idx, 6)

    @classmethod
    def layer_lr_mult(cls, layer_idx, stage):
        """stage: 'phase_focus' | 'joint' | 'norm_joint'"""
        mapping = {
            "phase_focus": cls.PHASE_FOCUS_LR_MULT,
            "joint": cls.JOINT_LR_MULT,
            "norm_joint": cls.NORM_JOINT_LR_MULT,
        }
        dct = mapping.get(stage, cls.JOINT_LR_MULT)
        return cls._layer_val(dct, layer_idx, 1.0)

    @classmethod
    def vortex_init(cls, layer_idx):
        charge = cls._layer_val(cls.VORTEX_CHARGE, layer_idx, 1.0)
        radial_scale = cls._layer_val(cls.VORTEX_RADIAL_SCALE, layer_idx, 0.35)
        return charge, radial_scale
