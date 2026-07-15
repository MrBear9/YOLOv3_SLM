"""models.teacher - Optical teacher model architectures."""

__version__ = "0.1.0"

from .architectures import CVOCAConvTeacherV2, ConvTeacher, ConvTeacherV3
from .building_blocks import (
    CVOCAStage,
    FeedbackGuidance,
    RawImageBridge,
    SqueezeExcite,
    SwiGLUGate,
    SyntheticWavelengthComplexConv,
    TeacherBottleneck,
    TeacherC2f,
    TeacherConvBNAct,
    TeacherResidualBlock,
    TeacherSPPF,
)


def build_teacher(config=None):
    arch = str(getattr(config, "TEACHER_ARCH", "convteacher_v2") if config is not None else "convteacher_v2").strip().lower()
    if arch in {"convteacher", "v1"}:
        c = int(getattr(config, "TEACHER_V1_BASE_CHANNELS", 32) if config is not None else 32)
        b = int(getattr(config, "TEACHER_V1_C2F_BLOCKS", 3) if config is not None else 3)
        return ConvTeacher(base_channels=c, c2f_blocks=b)
    if arch in {"convteacher_v2", "v2"}:
        c = int(getattr(config, "TEACHER_V2_BASE_CHANNELS", 24) if config is not None else 24)
        b = int(getattr(config, "TEACHER_V2_C2F_BLOCKS", 2) if config is not None else 2)
        w = int(getattr(config, "TEACHER_V2_SYNTHETIC_WAVELENGTHS", 3) if config is not None else 3)
        k = int(getattr(config, "TEACHER_V2_COMPLEX_KERNEL_SIZE", 5) if config is not None else 5)
        return CVOCAConvTeacherV2(base_channels=c, c2f_blocks=b, synthetic_wavelengths=w, complex_kernel_size=k)
    if arch in {"convteacher_v3", "v3"}:
        c = int(getattr(config, "TEACHER_V3_BASE_CHANNELS", 24) if config is not None else 24)
        b = int(getattr(config, "TEACHER_V3_C2F_BLOCKS", 2) if config is not None else 2)
        s = float(getattr(config, "TEACHER_V3_RESIDUAL_SCALE", 0.30) if config is not None else 0.30)
        return ConvTeacherV3(base_channels=c, c2f_blocks=b, residual_scale=s)
    raise ValueError(f"Unsupported TEACHER_ARCH: {arch}")


__all__ = [
    "build_teacher",
    "ConvTeacher",
    "ConvTeacherV3",
    "CVOCAConvTeacherV2",
    "FeedbackGuidance",
    "RawImageBridge",
    "SqueezeExcite",
    "SwiGLUGate",
    "TeacherResidualBlock",
    "TeacherConvBNAct",
    "TeacherBottleneck",
    "TeacherC2f",
    "TeacherSPPF",
    "SyntheticWavelengthComplexConv",
    "CVOCAStage",
]
