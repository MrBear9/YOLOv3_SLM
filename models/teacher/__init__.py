"""models.teacher - Optical teacher model architectures."""

__version__ = "0.1.0"

from .architectures import CVOCAConvTeacherV2, ConvTeacher, ConvTeacherV3
from .building_blocks import (
    C2fCIB,
    CIB,
    FeedbackGuidance,
    RawImageBridge,
    SqueezeExcite,
    SwiGLUGate,
    TeacherBottleneck,
    TeacherC2f,
    TeacherConvBNAct,
    TeacherResidualBlock,
    TeacherSPPF,
)
from .fourier_layers import FourierOpticalLayer


def build_teacher(config=None):
    arch = str(getattr(config, "TEACHER_ARCH", "convteacher_v2") if config is not None else "convteacher_v2").strip().lower()
    if arch in {"convteacher", "v1"}:
        c = int(getattr(config, "TEACHER_V1_BASE_CHANNELS", 32) if config is not None else 32)
        b = int(getattr(config, "TEACHER_V1_C2F_BLOCKS", 3) if config is not None else 3)
        return ConvTeacher(base_channels=c, c2f_blocks=b)
    if arch in {"convteacher_v2", "v2"}:
        c = int(getattr(config, "TEACHER_V2_BASE_CHANNELS", 32) if config is not None else 32)
        b = int(getattr(config, "TEACHER_V2_C2F_BLOCKS", 3) if config is not None else 3)
        fb = int(getattr(config, "TEACHER_V2_FOURIER_BANDS", 8) if config is not None else 8)
        fs = float(getattr(config, "TEACHER_V2_FOURIER_LOW_PASS_SIGMA", 0.5) if config is not None else 0.5)
        rs = float(getattr(config, "TEACHER_V2_RESIDUAL_SCALE", 0.30) if config is not None else 0.30)
        return CVOCAConvTeacherV2(
            base_channels=c,
            c2f_blocks=b,
            fourier_bands=fb,
            fourier_low_pass_sigma=fs,
            residual_scale=rs,
        )
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
    "C2fCIB",
    "CIB",
    "FeedbackGuidance",
    "FourierOpticalLayer",
    "RawImageBridge",
    "SqueezeExcite",
    "SwiGLUGate",
    "TeacherResidualBlock",
    "TeacherConvBNAct",
    "TeacherBottleneck",
    "TeacherC2f",
    "TeacherSPPF",
]
