"""光学传播层父类配置 — per-layer 配置用字典结构，消除 key 爆炸。

使用方式: ``ConfigSLM(OpticalConfig)`` 继承，然后通过访问器方法取值。
"""

import numpy as np


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
    PROP_DISTANCE = {1: 0.10, 2: 0.10, 3: 0.20, 4: 0.20}
    TRAIN_LAYER  = {1: True, 2: True, 3: True, 4: True}

    # Phase field block parameters
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
    DH_PSF_ROTATION    = {1: 0.0, 2: np.pi / 2, 3: 0.0, 4: np.pi / 2}
    DH_PSF_HANDEDNESS  = {1: 1.0, 2: -1.0,     3: 1.0, 4: -1.0}

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

    @classmethod
    def dh_psf_init(cls, layer_idx):
        rotation = cls._layer_val(cls.DH_PSF_ROTATION, layer_idx, 0.0)
        handedness = cls._layer_val(cls.DH_PSF_HANDEDNESS, layer_idx, 1.0)
        return rotation, handedness
