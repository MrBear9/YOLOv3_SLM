"""光学传播层父类配置 — per-layer 配置用字典结构，消除 key 爆炸。

使用方式: ``ConfigSLM(OpticalConfig)`` 继承，然后通过访问器方法取值。
"""

import math

from models.SLM.physical_defaults import (
    DEFAULT_DMD_PIXEL_PITCH,
    DEFAULT_DMD_RESOLUTION,
    DEFAULT_PROPAGATION_DISTANCES,
    DEFAULT_SLM_LAYER_PROFILES,
    SLM_PROFILES,
    active_pixel_shape,
    aperture_size,
)

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
    # DMD input: 640 pixels * 5.4 um = 3.456 mm in both axes.
    RESOLUTION = DEFAULT_DMD_RESOLUTION
    DMD_PIXEL_PITCH = DEFAULT_DMD_PIXEL_PITCH
    WAVELENGTH = 532e-9
    # Real panel facts and fitted numerical ASM sampling are independent.
    SLM_PROFILES = SLM_PROFILES
    SLM_LAYER_PROFILES = dict(DEFAULT_SLM_LAYER_PROFILES)
    # Compatibility scalar. New propagation code calls sampling_pitch(layer).
    PIXEL_SIZE = SLM_PROFILES["magicholo_4p5"]["effective_sampling_pitch"]
    SLM_MODE = "phase"
    INPUT_INTENSITY_MODE = "srgb"
    OPTICAL_FIELD_EPS = 1e-8
    OPTICAL_NORM_EPS = 1e-6

    # Direct phase optimization on the simulation grid; Adam is the optimizer.
    # The phase starts randomly and receives gradients directly; no pyramid or
    # neural-field residual participates in the current experiment.
    SLM_PHASE_PARAM_MODE = "direct_sgd"
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
    # A validated paired checkpoint can be used as a non-regression starting
    # point for a short optical refinement.  ``ConfigSLM`` selects the actual
    # checkpoint for a matching optical geometry; keep random as the general
    # default for direct phase optimization.
    SLM_INIT_MODE = "random"
    SLM_DIRECT_SGD_INIT_RANGE_RAD = 0.5
    SLM_INIT_NOISE_STD = 0.02
    SLM_INIT_CHECKPOINT = r"output/SLM_Tv2_light_phase_refine/detector_best.pth"
    SLM_PHASE_LEVELS = 256
    SIMULATE_PHASE_QUANTIZATION = False
    SIMULATE_HARDWARE_PIXEL_GRID = True
    SLM_GRAY_INVERTED = True
    SLM_EXPORT_PHASE_OFFSET_RAD = float(math.pi)
    SLM_GRAY_TO_PHASE_LUT = r""

    # ═══════════════════════════════════════════════════════════════════════
    # 可选相位金字塔（当前隔离且不生效）
    # 仅当 SLM_PHASE_PARAM_MODE 显式改为 "direct_sgd_pyramid" 时，
    # 下面参数才会进入模型；保留供未来实验使用。
    # ═══════════════════════════════════════════════════════════════════════
    SLM_DIRECT_SGD_PYRAMID_SCALE = 0.5
    PHASE_NUM_SCALES = 5
    PHASE_USE_BLOCKWISE = True
    PHASE_MLP_HIDDEN = 32
    PHASE_MLP_LAYERS = 2

    # ═══════════════════════════════════════════════════════════════════════
    # Per-layer 配置  {layer_idx (1-based): value}
    # ═══════════════════════════════════════════════════════════════════════
    # Student geometry: SLM1 -> 20 cm -> SLM2 -> 10 cm -> sensor plane.
    PROP_DISTANCE = {
        1: DEFAULT_PROPAGATION_DISTANCES[0],
        2: DEFAULT_PROPAGATION_DISTANCES[1],
        3: 0.20,
        4: 0.20,
    }

    # Which optical layers each phase-only stage may update.  The refinement
    # stage deliberately gives the final SLM its own detector-aware update;
    # SLM1 remains the stable image/detail encoder at that point.
    PHASE_TRAINABLE_LAYERS = {
        "phase_focus": {1: True, 2: True, 3: True, 4: True},
        "phase_refine": {1: False, 2: True, 3: True, 4: True},
    }

    # Phase field block parameters
    # Accept an int for square panels or (rows, cols) for rectangular panels.
    PHASE_BLOCK_GRID         = {1: 6, 2: 4, 3: 5, 4: 7}
    PHASE_BLOCK_OVERLAP      = {1: 8, 2: 8, 3: 8, 4: 8}
    PHASE_BLOCK_INNER_SCALES = {1: 3, 2: 2, 3: 2, 4: 2}
    PHASE_MLP_NUM_FREQS      = {1: 10, 2: 6, 3: 6, 4: 6}

    # Per-layer LR multipliers (× base LR)
    PHASE_FOCUS_LR_MULT = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    PHASE_REFINE_LR_MULT = {1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0}
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
    def slm_profile(cls, layer_idx):
        name = cls._layer_val(cls.SLM_LAYER_PROFILES, layer_idx)
        if name not in cls.SLM_PROFILES:
            raise ValueError(f"Unknown SLM profile for layer {layer_idx}: {name!r}.")
        return cls.SLM_PROFILES[name]

    @classmethod
    def slm_profile_name(cls, layer_idx):
        return cls._layer_val(cls.SLM_LAYER_PROFILES, layer_idx)

    @classmethod
    def hardware_pixel_pitch(cls, layer_idx):
        return float(cls.slm_profile(layer_idx)["hardware_pixel_pitch"])

    @classmethod
    def sampling_pitch(cls, layer_idx):
        return float(cls.slm_profile(layer_idx)["effective_sampling_pitch"])

    @classmethod
    def dmd_aperture(cls):
        return aperture_size(cls.RESOLUTION, cls.DMD_PIXEL_PITCH)

    @classmethod
    def slm_active_shape(cls, layer_idx):
        return active_pixel_shape(cls.slm_profile_name(layer_idx), cls.dmd_aperture())

    @classmethod
    def validate_optical_geometry(cls):
        if tuple(cls.RESOLUTION) != tuple(DEFAULT_DMD_RESOLUTION):
            raise ValueError("This bench configuration expects the DMD canvas to be 640x640.")
        for layer_idx in range(1, int(cls.NUM_LAYERS) + 1):
            shape = cls.slm_active_shape(layer_idx)
            pitch = cls.hardware_pixel_pitch(layer_idx)
            represented = tuple(count * pitch for count in shape)
            error = max(abs(a - b) for a, b in zip(represented, cls.dmd_aperture()))
            if error > pitch / 2 + 1e-12:
                raise ValueError(f"SLM{layer_idx} cannot cover the DMD aperture within half a hardware pixel.")

    @classmethod
    def is_trainable(cls, layer_idx, stage=None):
        """Return the layer policy for a phase-only stage."""
        stage = str(stage or "phase_focus")
        layer_policy = cls.PHASE_TRAINABLE_LAYERS.get(stage)
        if layer_policy is None:
            return True
        return cls._layer_val(layer_policy, layer_idx, True)

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
        """stage: 'phase_focus' | 'phase_refine' | 'joint' | 'norm_joint'"""
        mapping = {
            "phase_focus": cls.PHASE_FOCUS_LR_MULT,
            "phase_refine": cls.PHASE_REFINE_LR_MULT,
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
