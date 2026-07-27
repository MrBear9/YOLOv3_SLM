"""Compatibility exports for the modular optical SLM implementation."""

from models.SLM.asm_propagation import ASMPropagation
from models.SLM.optical_student import OpticalStudent, OpticalStudentWithDetector
from models.SLM.phase_parameterization import (
    FourierFeatureField,
    MultiScalePhaseField,
    _get_phase_cfg,
    _resolve_prop_distance,
)
from models.SLM.slm_modulation import SLMLayer

__all__ = [
    "ASMPropagation",
    "FourierFeatureField",
    "MultiScalePhaseField",
    "OpticalStudent",
    "OpticalStudentWithDetector",
    "SLMLayer",
    "_get_phase_cfg",
    "_resolve_prop_distance",
]