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
from models.SLM.slm_modulation import apply_phase_modulation, complex_intensity, resample_phase_map
from models.SLM.physical_grid import PhysicalGrid, physical_coordinates


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
        pixel_sizes = getattr(config, "TEACHER_V2_SAMPLING_PITCHES", None)
        if pixel_sizes is None:
            pixel_sizes = getattr(config, "TEACHER_V2_PIXEL_SIZE", DEFAULT_PIXEL_SIZE)
        if isinstance(pixel_sizes, (int, float)):
            pixel_sizes = (float(pixel_sizes),) * self.num_layers
        self.sampling_pitches = tuple(float(value) for value in pixel_sizes)
        if len(self.sampling_pitches) != self.num_layers or any(value <= 0 for value in self.sampling_pitches):
            raise ValueError(
                "TEACHER_V2_SAMPLING_PITCHES must contain one positive pitch "
                "for each TEACHER_V2_NUM_SLM_LAYERS entry."
            )
        # Compatibility attribute for older logs/checkpoints.
        self.pixel_size = self.sampling_pitches[0]
        distances = getattr(config, "TEACHER_V2_PROP_DISTANCES", DEFAULT_PROPAGATION_DISTANCES[:self.num_layers])
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
            WAVELENGTH=self.wavelength, RESOLUTION=resolution,
            ASM_PADDING_FACTOR=getattr(config, "ASM_PADDING_FACTOR", 2),
            ASM_SUPERSAMPLE_FACTOR=getattr(config, "ASM_SUPERSAMPLE_FACTOR", 1),
            OPTICAL_COORDINATE_MODE=getattr(config, "OPTICAL_COORDINATE_MODE", "legacy"),
        )
        self.geometry = None
        if physical_coordinates(config):
            profiles = config.TEACHER_V2_SLM_PROFILES
            if len(profiles) != self.num_layers:
                raise ValueError("Teacher SLM profiles must match its number of layers.")
            self.geometry = PhysicalGrid(config, self.sampling_pitches,
                [config.SLM_PROFILES[name]["hardware_pixel_pitch"] for name in profiles],
                [config.SLM_PROFILES[name]["resolution"] for name in profiles])
        self.propagations = nn.ModuleList(
            ASMPropagation(optics_config, distance, pixel_size=pitch,
                resolution=self.geometry.shapes[i] if self.geometry is not None else None)
            for i, (distance, pitch) in enumerate(zip(self.propagation_distances, self.sampling_pitches))
        )
        self.field_epsilon = float(getattr(config, "OPTICAL_FIELD_EPS", 1e-8))
        hardware_shapes = getattr(config, "TEACHER_V2_ACTIVE_PIXEL_SHAPES", None)
        if hardware_shapes is None:
            hardware_shapes = (resolution,) * self.num_layers
        self.hardware_shapes = tuple(tuple(int(value) for value in shape) for shape in hardware_shapes)
        if self.geometry is not None:
            self.hardware_shapes = self.geometry.active_shapes
        if len(self.hardware_shapes) != self.num_layers:
            raise ValueError("TEACHER_V2_ACTIVE_PIXEL_SHAPES must contain one shape per SLM layer.")
        self.simulate_hardware_grid = bool(getattr(config, "SIMULATE_HARDWARE_PIXEL_GRID", True))

    def forward(self, intensity, phase_maps):
        if len(phase_maps) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} phase maps, got {len(phase_maps)}.")
        if intensity.ndim != 4 or intensity.shape[1] != 1:
            raise ValueError("PhysicalSLMSimulator expects intensity with shape [B, 1, H, W].")

        # FFT propagation uses complex64 so it remains valid under float16 AMP.
        with torch.autocast(device_type=intensity.device.type, enabled=False):
            incident = intensity.float()
            if self.geometry is not None:
                field = self.geometry.incident_field(incident, self.field_epsilon)
                for i, (phase_map, propagation) in enumerate(zip(phase_maps, self.propagations)):
                    if phase_map.shape != incident.shape:
                        raise ValueError("Teacher phase maps must match the DMD intensity shape.")
                    if i:
                        field = self.geometry.transfers[i - 1](field)
                    field = self.geometry.modulate(field, phase_map, i)
                    field = propagation(field)
                return self.geometry.intensity(field)
            field = torch.complex(
                torch.sqrt(incident.clamp_min(0.0) + self.field_epsilon),
                torch.zeros_like(incident),
            )
            for phase_map, propagation, hardware_shape in zip(
                phase_maps, self.propagations, self.hardware_shapes
            ):
                if phase_map.shape != incident.shape:
                    raise ValueError(
                        f"Phase map shape {tuple(phase_map.shape)} does not match input intensity "
                        f"shape {tuple(incident.shape)}."
                    )
                phase_map = phase_map.float()
                if self.simulate_hardware_grid and tuple(phase_map.shape[-2:]) != hardware_shape:
                    phase_map = resample_phase_map(phase_map, hardware_shape)
                    phase_map = resample_phase_map(phase_map, incident.shape[-2:])
                field = apply_phase_modulation(field, phase_map)
                field = propagation(field)
            return complex_intensity(field)

    def physics_metadata(self):
        return {
            "physical_grid": self.geometry.metadata() if self.geometry is not None else None,
            "num_slm_layers": self.num_layers,
            "wavelength": self.wavelength,
            "sampling_pitches": self.sampling_pitches,
            "hardware_active_shapes": self.hardware_shapes,
            "simulate_hardware_pixel_grid": self.simulate_hardware_grid,
            "propagation_distances": self.propagation_distances,
            "asm_padding_factor": self.propagations[0].padding_factor,
            "asm_supersample_factor": self.propagations[0].supersample_factor,
        }
