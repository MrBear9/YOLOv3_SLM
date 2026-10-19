"""Shared metre-based coordinates for dynamic and static phase-only optics.

Resampling is an approximation to continuous-field evaluation, not scaled ASM.
Numerical convergence must be checked by refining the sampling and field extent.
No camera calibration, relay inversion, lens or measured pupil is implied.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.SLM.slm_modulation import resample_phase_map


def physical_coordinates(config):
    return getattr(config, "OPTICAL_COORDINATE_MODE", "legacy") == "physical"


class CoordinateSampler(nn.Module):
    """Centred pixel coordinates; zero outside the source, no implicit zoom.

Complex fields are interpolated componentwise. When downsampling a field,
frequencies above the destination Nyquist limit are removed first. Phase
pixels instead use nearest sampling (piecewise constant SLM modulation).
"""
    def __init__(self, source_shape, source_pitch, target_shape, target_pitch,
                 *, mode="bilinear", bandlimit=False):
        super().__init__()
        self.source_shape = tuple(source_shape)
        self.identity = tuple(source_shape) == tuple(target_shape) and source_pitch == target_pitch
        self.mode = mode
        axes = [(torch.arange(n, dtype=torch.float64) - (n - 1) / 2) * target_pitch
                for n in target_shape]
        yy, xx = torch.meshgrid(*axes, indexing="ij")
        grid = torch.stack((2 * xx / (source_shape[1] * source_pitch),
                            2 * yy / (source_shape[0] * source_pitch)), -1).float()[None]
        self.register_buffer("grid", grid, persistent=False)
        cutoff = None
        if bandlimit and target_pitch > source_pitch * (1 + 1e-12):
            fy = torch.fft.fftfreq(source_shape[0], d=source_pitch)
            fx = torch.fft.fftfreq(source_shape[1], d=source_pitch)
            cutoff = ((fy[:, None].abs() <= .5 / target_pitch) &
                      (fx[None, :].abs() <= .5 / target_pitch))[None, None]
        self.register_buffer("cutoff", cutoff, persistent=False)

    def forward(self, value):
        if tuple(value.shape[-2:]) != self.source_shape:
            raise ValueError(f"Coordinate sampler expected {self.source_shape}, got {value.shape[-2:]}.")
        if self.identity:
            return value
        if self.cutoff is not None:
            filtered = torch.fft.ifft2(torch.fft.fft2(value) * self.cutoff)
            value = filtered if value.is_complex() else filtered.real
        dtype = value.real.dtype if value.is_complex() else value.dtype
        grid = self.grid.to(dtype=dtype).expand(value.shape[0], -1, -1, -1)
        def sample(x):
            return F.grid_sample(x, grid, mode=self.mode, padding_mode="zeros", align_corners=False)
        if value.is_complex():
            return torch.complex(sample(value.real), sample(value.imag))
        return sample(value)


class PhysicalGrid(nn.Module):
    """Geometry shared by teacher and student; phase parameters stay separate."""
    def __init__(self, config, pitches, hardware_pitches, panel_shapes):
        super().__init__()
        self.resolution = tuple(config.RESOLUTION)
        self.pitches = tuple(float(p) for p in pitches)
        self.hardware_pitches = tuple(float(p) for p in hardware_pitches)
        self.panel_shapes = tuple(tuple(s) for s in panel_shapes)
        self.magnification = float(config.DMD_RELAY_MAGNIFICATION)
        self.input_pitch = float(config.DMD_PIXEL_PITCH) * self.magnification
        self.reference_extent = tuple(n * self.input_pitch for n in self.resolution)
        self.active_shapes = tuple(tuple(round(l / p) for l in self.reference_extent)
                                   for p in self.hardware_pitches)
        for shape, panel in zip(self.active_shapes, self.panel_shapes):
            if any(n < 1 or n > full for n, full in zip(shape, panel)):
                raise ValueError("The reference phase aperture must fit the SLM panel.")
        if any(p > hp * (1 + 1e-10) for p, hp in zip(self.pitches, self.hardware_pitches)):
            raise ValueError("physical coordinates require sampling no coarser than SLM pixels; "
                             "use coordinate_mode: legacy for old 13/6.4 um checkpoints.")
        factor = float(config.OPTICAL_FIELD_EXTENT_FACTOR)
        multiple = int(config.OPTICAL_GRID_MULTIPLE)
        if factor < 1 or multiple < 1 or self.input_pitch <= 0:
            raise ValueError("Invalid physical grid settings.")
        extent = tuple(l * factor for l in self.reference_extent)
        shapes = []
        # Earlier planes must cover the complete destination window. Otherwise
        # a zero-padded resampler would invent unknown propagated field samples.
        for pitch in reversed(self.pitches):
            shape = tuple(math.ceil(l / pitch / multiple - 1e-10) * multiple for l in extent)
            if max(shape) > config.OPTICAL_MAX_GRID_SIZE:
                raise ValueError(f"Physical grid {shape} exceeds OPTICAL_MAX_GRID_SIZE; check optics YAML.")
            shapes.append(shape)
            extent = tuple(n * pitch for n in shape)
        self.shapes = tuple(reversed(shapes))
        self.input_sampler = CoordinateSampler(self.resolution, self.input_pitch,
                                              self.shapes[0], self.pitches[0], bandlimit=True)
        self.transfers = nn.ModuleList(CoordinateSampler(self.shapes[i - 1], self.pitches[i - 1],
            self.shapes[i], self.pitches[i], bandlimit=True) for i in range(1, len(self.pitches)))
        self.phase_samplers = nn.ModuleList(CoordinateSampler(active, hardware, shape, pitch, mode="nearest")
            for active, hardware, shape, pitch in zip(self.active_shapes, self.hardware_pitches,
                                                       self.shapes, self.pitches))
        # Ideal point-intensity readout on a fixed physical ROI. It is deliberately
        # not interpolation of amplitude and not a real camera pixel integration.
        self.readout = CoordinateSampler(self.shapes[-1], self.pitches[-1],
                                         self.resolution, self.input_pitch)
        for i, (shape, pitch, panel, hp) in enumerate(zip(self.shapes, self.pitches,
                                                        self.panel_shapes, self.hardware_pitches)):
            axes = [(torch.arange(n) - (n - 1) / 2) * pitch for n in shape]
            mask = ((axes[0][:, None].abs() < panel[0] * hp / 2) &
                    (axes[1][None, :].abs() < panel[1] * hp / 2)).float()[None, None]
            self.register_buffer(f"panel_mask_{i}", mask, persistent=False)
        # Persistent identity prevents a same-shaped tensor from silently
        # acquiring another physical meaning after a YAML edit.
        identity = [1, *self.resolution, self.input_pitch, self.magnification, len(self.pitches),
                    config.ASM_PADDING_FACTOR, config.ASM_SUPERSAMPLE_FACTOR]
        for values in zip(self.pitches, self.hardware_pitches, self.panel_shapes,
                          self.active_shapes, self.shapes):
            pitch, hp, panel, active, shape = values
            identity.extend([pitch, hp, *panel, *active, *shape])
        self.register_buffer("geometry_signature", torch.tensor(identity, dtype=torch.float64))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "geometry_signature"
        value = state_dict.get(key)
        if value is None or value.shape != self.geometry_signature.shape or not torch.equal(
                value.to(self.geometry_signature), self.geometry_signature):
            error_msgs.append(f"{prefix}physical geometry mismatch: use the checkpoint's optics YAML "
                              "or train a new teacher/student pair; legacy weights cannot be reused silently.")
            state_dict = dict(state_dict)
            state_dict[key] = self.geometry_signature
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def incident_field(self, intensity, epsilon):
        if tuple(intensity.shape[-2:]) != self.resolution or intensity.shape[1] != 1:
            raise ValueError("Incident intensity must use the configured DMD canvas and one channel.")
        amplitude = torch.sqrt(intensity.float().clamp_min(0) + epsilon) / self.magnification
        # Ideal relay amplitude scaling preserves integrated power at M != 1.
        return self.input_sampler(torch.complex(amplitude, torch.zeros_like(amplitude)))

    def modulate(self, field, phase, index):
        native = resample_phase_map(phase.float(), self.active_shapes[index])
        phasor = torch.complex(torch.cos(native), torch.sin(native))
        # Outside the exported patch use reference phase zero, NOT an opaque
        # aperture. Only the finite panel boundary blocks the field.
        modulation = 1 + self.phase_samplers[index](phasor - 1)
        return field * modulation * getattr(self, f"panel_mask_{index}")

    def intensity(self, field):
        return self.readout(field.real.square() + field.imag.square())

    def metadata(self):
        return dict(coordinate_mode="physical", reference_extent_m=self.reference_extent,
                    grid_shapes=self.shapes, sampling_pitches_m=self.pitches,
                    active_shapes=self.active_shapes, panel_shapes=self.panel_shapes,
                    readout="ideal centre ROI intensity samples; no camera calibration",
                    readout_shape=self.resolution, readout_pitch_m=self.input_pitch)


def validate_geometry_state(module, state):
    """Call BEFORE partial-loading helpers fill missing keys from current state."""
    expected = {k: v for k, v in module.state_dict().items() if k.endswith("geometry_signature")}
    actual = {k: v for k, v in state.items() if k.endswith("geometry_signature")}
    if bool(expected) != bool(actual) or set(expected) != set(actual):
        raise RuntimeError("Physical geometry mismatch (legacy versus physical checkpoint); "
                           "select the matching optics YAML or retrain the teacher/student pair.")
    for key, value in expected.items():
        if actual[key].shape != value.shape or not torch.equal(actual[key].to(value), value):
            raise RuntimeError("Physical geometry mismatch: checkpoint and optics YAML differ.")
