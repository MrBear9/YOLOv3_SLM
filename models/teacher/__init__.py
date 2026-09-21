"""models.teacher - Optical teacher model architectures."""

__version__ = "0.1.0"

from .architectures import ConvTeacher, ConvTeacherV3
from .physical_simulator import PhysicalSLMSimulator
from .physical_teacher_v2 import PhysicallyConstrainedTeacherV2
from .physical_teacher_v4 import PhysicallyConstrainedTeacherV4
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


def configure_teacher_checkpoint(config, path):
    """Restore architecture metadata before constructing a frozen teacher."""
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    arch = checkpoint.get("teacher_arch", "convteacher_v2")
    if arch not in {"convteacher", "v1", "convteacher_v2", "v2", "convteacher_v3", "v3", "physical_teacher_v4", "v4"}:
        raise ValueError(f"Unsupported teacher checkpoint: {arch}. Old transfer V4 is a static pair, not the new CNN teacher.")
    config.TEACHER_ARCH = arch
    for key, attr in {
        "teacher_v4_base_channels": "TEACHER_V4_BASE_CHANNELS",
        "teacher_v4_depths": "TEACHER_V4_DEPTHS",
        "global_local_context_grid": "GLOBAL_LOCAL_CONTEXT_GRID",
        "teacher_depths": "GLOBAL_LOCAL_TEACHER_DEPTHS",
        "detector_depths": "GLOBAL_LOCAL_DETECTOR_DEPTHS",
    }.items():
        if key in checkpoint:
            setattr(config, attr, checkpoint[key])
    return arch


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
        return PhysicallyConstrainedTeacherV2(
            config=config,
            base_channels=c,
            c2f_blocks=b,
            fourier_bands=fb,
            fourier_low_pass_sigma=fs,
        )
    if arch in {"convteacher_v3", "v3"}:
        c = int(getattr(config, "TEACHER_V3_BASE_CHANNELS", 24) if config is not None else 24)
        b = int(getattr(config, "TEACHER_V3_C2F_BLOCKS", 2) if config is not None else 2)
        s = float(getattr(config, "TEACHER_V3_RESIDUAL_SCALE", 0.30) if config is not None else 0.30)
        return ConvTeacherV3(base_channels=c, c2f_blocks=b, residual_scale=s)
    if arch in {"physical_teacher_v4", "v4"}:
        return PhysicallyConstrainedTeacherV4(config)
    raise ValueError(f"Unsupported TEACHER_ARCH: {arch}")


__all__ = [
    "build_teacher",
    "ConvTeacher",
    "ConvTeacherV3",
    "PhysicallyConstrainedTeacherV2",
    "PhysicallyConstrainedTeacherV4",
    "PhysicalSLMSimulator",
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
