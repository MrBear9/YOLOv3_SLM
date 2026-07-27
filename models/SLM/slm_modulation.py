'''
@File    :   slm_modulation.py
@Time    :   2026/07/27 12:50:01
@Author  :   Mr.Bear9 
@Github  :   https://github.com/MrBear9
'''

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.SLM.phase_parameterization import MultiScalePhaseField


def apply_phase_modulation(field, phase):
    """Apply continuous phase-only modulation to a complex optical field."""
    return field * torch.exp(1j * phase)

class SLMLayer(nn.Module):
    def __init__(self, config, resolution=None, mode=None, layer_index=1):
        super().__init__()
        self.config = config
        resolution = config.RESOLUTION if resolution is None else resolution
        mode = config.SLM_MODE if mode is None else mode
        assert mode in {"phase", "amp_phase"}
        self.mode = mode
        self.layer_index = layer_index
        self.phase_levels = int(getattr(config, "SLM_PHASE_LEVELS", 256))
        if self.phase_levels < 2:
            raise ValueError("SLM_PHASE_LEVELS must be at least 2.")
        self.simulate_quantization = bool(getattr(config, "SIMULATE_PHASE_QUANTIZATION", True))
        self.gray_inverted = bool(getattr(config, "SLM_GRAY_INVERTED", False))
        lut_gray, lut_phase = self._load_gray_to_phase_lut()
        self.register_buffer("lut_gray", lut_gray, persistent=True)
        self.register_buffer("lut_phase", lut_phase, persistent=True)

        # Normalize old names so checkpoints keep their flat ``phase_raw`` key.
        param_mode = str(getattr(config, "SLM_PHASE_PARAM_MODE", "direct_sgd")).strip().lower()
        if param_mode in {"direct", "direct_sgd", "direct_sgb"}:
            param_mode = "direct_sgd"
        elif param_mode not in {"direct_sgd_pyramid", "multiscale_mlp"}:
            raise ValueError(
                "SLM_PHASE_PARAM_MODE must be direct_sgd (or direct_sgb), "
                "direct_sgd_pyramid, or multiscale_mlp; "
                f"got {param_mode!r}."
            )
        self._phase_param_mode = param_mode

        if param_mode == "multiscale_mlp":
            self.phase_field = MultiScalePhaseField(config, resolution, layer_index=layer_index)
            # Seed the finest scale with the chosen init pattern
            init_phase = self._initial_phase(resolution)
            self.phase_field.set_base_phase(init_phase)
            self.register_parameter("phase_raw", None)
        elif param_mode == "direct_sgd_pyramid":
            # Keep the HolographSLM direct variable as the primary phase. The
            # zero-start pyramid is a residual, so a loaded direct checkpoint
            # begins from exactly its previously validated modulation.
            self.phase_raw = nn.Parameter(self._initial_phase(resolution))
            self.phase_field = MultiScalePhaseField(config, resolution, layer_index=layer_index)
            self.phase_field.zero_residual()
        else:
            # Legacy / direct mode
            self.phase_raw = nn.Parameter(self._initial_phase(resolution))
            self.phase_field = None

        if mode == "amp_phase":
            self.amp_raw = nn.Parameter(torch.rand(1, 1, *resolution))
        else:
            self.register_parameter("amp_raw", None)

    def _load_gray_to_phase_lut(self):
        path = str(getattr(self.config, "SLM_GRAY_TO_PHASE_LUT", "") or "").strip()
        if not path:
            return torch.empty(0), torch.empty(0)
        lut_path = Path(path)
        if not lut_path.is_file():
            raise FileNotFoundError(f"SLM_GRAY_TO_PHASE_LUT not found: {lut_path}")
        if lut_path.suffix.lower() == ".npy":
            values = np.asarray(np.load(lut_path), dtype=np.float32)
        else:
            values = np.asarray(np.loadtxt(lut_path, delimiter=","), dtype=np.float32)
        if values.ndim == 1:
            gray = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
            phase = values
        elif values.ndim == 2 and values.shape[1] == 2:
            gray = values[:, 0]
            phase = values[:, 1]
            gray_scale = 255.0 if gray.max() > 1.0 else 1.0
            gray = gray / gray_scale
        else:
            raise ValueError("SLM_GRAY_TO_PHASE_LUT must contain phase values or (gray, phase) pairs.")
        order = np.argsort(gray)
        gray, phase = gray[order], phase[order]
        if gray.size < 2 or np.any(np.diff(gray) <= 0) or np.any(np.diff(phase) < 0):
            raise ValueError("SLM_GRAY_TO_PHASE_LUT requires increasing gray and non-decreasing phase values.")
        return torch.from_numpy(gray), torch.from_numpy(phase)

    def _phase_grid(self, resolution):
        height, width = resolution
        y = torch.linspace(-1.0, 1.0, height)
        x = torch.linspace(-1.0, 1.0, width)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return height, width, yy, xx

    def _wrap_with_noise(self, phase, height, width):
        noise_std = float(getattr(self.config, "SLM_INIT_NOISE_STD", 0.0))
        if noise_std > 0:
            phase = phase + torch.randn_like(phase) * noise_std
        return torch.remainder(phase, 2 * np.pi).contiguous().view(1, 1, height, width)

    def _initial_phase(self, resolution):
        init_mode = str(getattr(self.config, "SLM_INIT_MODE", "zero")).lower()
        if init_mode == "zero":
            height, width = resolution
            noise_std = float(getattr(self.config, "SLM_INIT_NOISE_STD", 0.02))
            if noise_std < 0:
                raise ValueError("SLM_INIT_NOISE_STD must be non-negative.")
            phase = torch.zeros(1, 1, height, width)
            if noise_std > 0:
                phase = phase + torch.randn_like(phase) * noise_std
            return torch.remainder(phase, 2 * np.pi)
        if init_mode == "random":
            if self._phase_param_mode in {"direct_sgd", "direct_sgd_pyramid"}:
                init_range = float(getattr(self.config, "SLM_DIRECT_SGD_INIT_RANGE_RAD", 0.5))
                if init_range <= 0:
                    raise ValueError("SLM_DIRECT_SGD_INIT_RANGE_RAD must be positive.")
                return torch.empty(1, 1, *resolution).uniform_(-init_range, init_range)
            return torch.rand(1, 1, *resolution) * (2 * np.pi)
        if init_mode == "vortex":
            height, width, yy, xx = self._phase_grid(resolution)
            if hasattr(self.config, "vortex_init"):
                charge, radial_scale = self.config.vortex_init(self.layer_index)
            else:
                charge = float(getattr(self.config, f"SLM_VORTEX_CHARGE_{self.layer_index}", 1.0))
                radial_scale = float(getattr(self.config, f"SLM_VORTEX_RADIAL_SCALE_{self.layer_index}", 0.35))
            theta = torch.atan2(yy, xx)
            radius2 = xx.square() + yy.square()
            phase = charge * theta + radial_scale * np.pi * radius2
            return self._wrap_with_noise(phase, height, width)
        if init_mode == "checkpoint":
            # The train setup replaces this neutral phase with the checkpoint.
            return torch.zeros(1, 1, *resolution)
        raise ValueError(
            "SLM_INIT_MODE must be one of: zero, random, vortex, checkpoint; "
            f"got {init_mode!r}."
        )

    def _raw_phase(self):
        """Return the continuous phase used by direct-SGD or the pyramid mode."""
        if self._phase_param_mode == "multiscale_mlp" and self.phase_field is not None:
            return self.phase_field()
        if self._phase_param_mode == "direct_sgd_pyramid":
            scale = float(getattr(self.config, "SLM_DIRECT_SGD_PYRAMID_SCALE", 0.25))
            return self.phase_raw + scale * self.phase_field()
        return self.phase_raw

    def direct_phase(self):
        """Return the full-resolution direct-SGD component, when present."""
        return self.phase_raw if self.phase_raw is not None else None

    def pyramid_phase(self):
        """Return the unscaled pyramid residual, when the mode contains one."""
        return self.phase_field() if self.phase_field is not None else None

    def wrapped_phase(self):
        """Simulation phase represented in the conventional [0, 2pi) interval."""
        return torch.remainder(self._raw_phase(), 2 * np.pi)

    def phase_to_gray(self, phase=None):
        """Convert a hardware/export phase map to the calibrated gray drive."""
        phase = self.hardware_export_phase() if phase is None else torch.remainder(phase, 2 * np.pi)
        if self.lut_phase.numel() == 0:
            gray = phase / (2 * np.pi)
        else:
            phase_grid = self.lut_phase.to(device=phase.device, dtype=phase.dtype)
            gray_grid = self.lut_gray.to(device=phase.device, dtype=phase.dtype)
            flat_phase = phase.flatten().clamp(phase_grid[0], phase_grid[-1])
            upper = torch.searchsorted(phase_grid, flat_phase).clamp(1, phase_grid.numel() - 1)
            lower = upper - 1
            p0, p1 = phase_grid[lower], phase_grid[upper]
            g0, g1 = gray_grid[lower], gray_grid[upper]
            gray = (g0 + (flat_phase - p0) * (g1 - g0) / (p1 - p0).clamp_min(1e-8)).view_as(phase)
        return 1.0 - gray if self.gray_inverted else gray

    def gray_to_phase(self, gray):
        gray = gray.clamp(0.0, 1.0)
        if self.gray_inverted:
            gray = 1.0 - gray
        if self.lut_phase.numel() == 0:
            return gray * (2 * np.pi)
        gray_grid = self.lut_gray.to(device=gray.device, dtype=gray.dtype)
        phase_grid = self.lut_phase.to(device=gray.device, dtype=gray.dtype)
        flat_gray = gray.flatten()
        upper = torch.searchsorted(gray_grid, flat_gray).clamp(1, gray_grid.numel() - 1)
        lower = upper - 1
        g0, g1 = gray_grid[lower], gray_grid[upper]
        p0, p1 = phase_grid[lower], phase_grid[upper]
        return (p0 + (flat_gray - g0) * (p1 - p0) / (g1 - g0).clamp_min(1e-8)).view_as(gray)

    def effective_phase(self):
        """Legacy name for the phase used by the numerical propagation model."""
        return self.simulation_phase()

    def simulation_phase(self):
        """Return the phase used during propagation, without hardware quantization."""
        if self._phase_param_mode in {"direct_sgd", "direct_sgd_pyramid"}:
            return self._raw_phase()
        # Preserve the legacy hardware-in-the-loop behavior for the pyramid path.
        gray = self.phase_to_gray(self.wrapped_phase())
        if self.simulate_quantization:
            scaled = gray * (self.phase_levels - 1)
            rounded = torch.round(scaled)
            gray = gray + (rounded / (self.phase_levels - 1) - gray).detach()
        return self.gray_to_phase(gray)

    def hardware_export_phase(self):
        """Return the display phase using HolographSLM's +pi export origin."""
        offset = float(getattr(self.config, "SLM_EXPORT_PHASE_OFFSET_RAD", np.pi))
        return torch.remainder(self._raw_phase() + offset, 2 * np.pi)

    def phase_to_gray_uint8(self):
        """Return the actual finite-level drive image for SLM export."""
        gray = self.phase_to_gray().detach().clamp(0.0, 1.0)
        return torch.round(gray * (self.phase_levels - 1)).to(torch.uint8)

    def centered_phase(self):
        wrapped = self.wrapped_phase()
        return torch.atan2(torch.sin(wrapped), torch.cos(wrapped))

    def forward(self, field):
        phase = self.simulation_phase()
        out = apply_phase_modulation(field, phase)
        if self.mode == "amp_phase":
            out = out * torch.sigmoid(self.amp_raw)
        return out


