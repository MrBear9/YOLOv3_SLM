"""Differentiable phase-only SLM propagation used by teacher V2."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from models.SLM.asm_propagation import ASMPropagation
from models.SLM.physical_defaults import (
    DEFAULT_PIXEL_SIZE,
    DEFAULT_PROPAGATION_DISTANCES,
    DEFAULT_WAVELENGTH,
)
from models.SLM.slm_modulation import apply_phase_modulation


class PhysicalSLMSimulator(nn.Module):
    """Apply predicted phase maps through the same phase-only ASM model as SLM.

    The phase maps are supplied by the teacher network rather than stored as
    independent SLM parameters.  This keeps the teacher output inside the
    physically reachable space of a cascaded phase-only optical system.
    """

    def __init__(self, config):
        super().__init__()
        resolution = tuple(int(value) for value in getattr(config, "RESOLUTION", (640, 640)))
        if len(resolution) != 2 or min(resolution) < 1:
            raise ValueError("TEACHER_V2 physical simulation requires a positive (height, width) RESOLUTION.")

        self.num_layers = int(getattr(config, "TEACHER_V2_NUM_SLM_LAYERS", 2))
        if self.num_layers < 1:
            raise ValueError("TEACHER_V2_NUM_SLM_LAYERS must be at least 1.")

        self.wavelength = float(getattr(config, "TEACHER_V2_WAVELENGTH", DEFAULT_WAVELENGTH))
        self.pixel_size = getattr(config, "TEACHER_V2_PIXEL_SIZE", DEFAULT_PIXEL_SIZE)
        distances = getattr(config, "TEACHER_V2_PROP_DISTANCES", DEFAULT_PROPAGATION_DISTANCES)
        if isinstance(distances, (int, float)):
            distances = (float(distances),) * self.num_layers
        self.propagation_distances = tuple(float(value) for value in distances)
        if len(self.propagation_distances) != self.num_layers:
            raise ValueError(
                "TEACHER_V2_PROP_DISTANCES must contain one distance for each "
                "TEACHER_V2_NUM_SLM_LAYERS entry."
            )
        if self.wavelength <= 0 or any(distance == 0 for distance in self.propagation_distances):
            raise ValueError("Teacher V2 wavelength must be positive and propagation distances must be non-zero.")

        optics_config = SimpleNamespace(
            WAVELENGTH=self.wavelength,
            PIXEL_SIZE=self.pixel_size,
            RESOLUTION=resolution,
        )
        self.propagations = nn.ModuleList(
            ASMPropagation(optics_config, distance) for distance in self.propagation_distances
        )
        self.field_epsilon = float(getattr(config, "OPTICAL_FIELD_EPS", 1e-8))

    def forward(self, intensity, phase_maps):
        if len(phase_maps) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} phase maps, got {len(phase_maps)}.")
        if intensity.ndim != 4 or intensity.shape[1] != 1:
            raise ValueError("PhysicalSLMSimulator expects intensity with shape [B, 1, H, W].")

        # FFT propagation uses complex64 so it remains valid under float16 AMP.
        with torch.autocast(device_type=intensity.device.type, enabled=False):
            incident = intensity.float()
            field = torch.complex(
                torch.sqrt(incident.clamp_min(0.0) + self.field_epsilon),
                torch.zeros_like(incident),
            )
            for phase_map, propagation in zip(phase_maps, self.propagations):
                if phase_map.shape != incident.shape:
                    raise ValueError(
                        f"Phase map shape {tuple(phase_map.shape)} does not match input intensity "
                        f"shape {tuple(incident.shape)}."
                    )
                field = apply_phase_modulation(field, phase_map.float())
                field = propagation(field)
            return torch.abs(field).square()

    def physics_metadata(self):
        return {
            "num_slm_layers": self.num_layers,
            "wavelength": self.wavelength,
            "pixel_size": self.pixel_size,
            "propagation_distances": self.propagation_distances,
        }
