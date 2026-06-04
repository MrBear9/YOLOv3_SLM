import numpy as np
import torch
import torch.nn as nn


class SLMLayer(nn.Module):
    def __init__(self, config, resolution=None, mode=None, layer_index=1):
        super().__init__()
        self.config = config
        resolution = config.RESOLUTION if resolution is None else resolution
        mode = config.SLM_MODE if mode is None else mode
        assert mode in {"phase", "amp_phase"}
        self.mode = mode
        self.layer_index = layer_index
        self.phase_raw = nn.Parameter(self._initial_phase(resolution))
        if mode == "amp_phase":
            self.amp_raw = nn.Parameter(torch.rand(1, 1, *resolution))
        else:
            self.register_parameter("amp_raw", None)

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
        init_mode = str(getattr(self.config, "SLM_INIT_MODE", "random")).lower()
        if init_mode in {"vortex", "vortex_checkpoint"}:
            height, width, yy, xx = self._phase_grid(resolution)
            theta = torch.atan2(yy, xx)
            radius2 = xx.square() + yy.square()
            if self.layer_index == 1:
                charge = float(getattr(self.config, "SLM_VORTEX_CHARGE_1", 1.0))
                radial_scale = float(getattr(self.config, "SLM_VORTEX_RADIAL_SCALE_1", 0.35))
            else:
                charge = float(getattr(self.config, "SLM_VORTEX_CHARGE_2", -1.0))
                radial_scale = float(getattr(self.config, "SLM_VORTEX_RADIAL_SCALE_2", -0.25))
            phase = charge * theta + radial_scale * np.pi * radius2
            return self._wrap_with_noise(phase, height, width)
        if init_mode in {"double_helix", "double_helix_psf", "dh_psf", "double_helix_checkpoint", "dh_psf_checkpoint"}:
            height, width, yy, xx = self._phase_grid(resolution)
            periods = max(float(getattr(self.config, "SLM_DH_PSF_PERIODS", 2.0)), 1.0)
            charge = float(getattr(self.config, "SLM_DH_PSF_CHARGE", 1.0))
            radial_scale = float(getattr(self.config, "SLM_DH_PSF_RADIAL_SCALE", 0.20))
            saddle_scale = float(getattr(self.config, "SLM_DH_PSF_SADDLE_SCALE", 0.08))
            spiral_offset = float(getattr(self.config, "SLM_DH_PSF_SPIRAL_OFFSET", 0.0))
            aperture_radius = float(getattr(self.config, "SLM_DH_PSF_APERTURE_RADIUS", 2.0))
            if self.layer_index == 1:
                rotation = float(getattr(self.config, "SLM_DH_PSF_ROTATION_1", 0.0))
                handedness = float(getattr(self.config, "SLM_DH_PSF_HANDEDNESS_1", 1.0))
            else:
                rotation = float(getattr(self.config, "SLM_DH_PSF_ROTATION_2", np.pi / 2))
                handedness = float(getattr(self.config, "SLM_DH_PSF_HANDEDNESS_2", -1.0))

            cell_x = torch.remainder((xx + 1.0) * periods / 2.0, 1.0) * 2.0 - 1.0
            cell_y = torch.remainder((yy + 1.0) * periods / 2.0, 1.0) * 2.0 - 1.0
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)
            xr = cell_x * cos_r - cell_y * sin_r
            yr = cell_x * sin_r + cell_y * cos_r

            shifted_x = xr - spiral_offset
            shifted_y = yr
            theta = torch.atan2(shifted_y, shifted_x)
            radius2 = shifted_x.square() + shifted_y.square()
            radius = torch.sqrt(radius2 + 1e-8)

            # DH-PSF-like seed: a single spiral phase term plus dense quadratic
            # radial rings. This produces the visible spiral-ring phase cell used
            # as an initialization prior; periods > 1 tiles that cell into an array.
            phase = (
                handedness * charge * theta
                + radial_scale * np.pi * radius2
                + saddle_scale * np.pi * (xr.square() - yr.square())
            )
            aperture = radius <= aperture_radius
            if aperture.any():
                outside_fill = phase[aperture].mean()
                phase = torch.where(aperture, phase, outside_fill + 0.15 * np.pi * (xr.square() + yr.square()))
            return self._wrap_with_noise(phase, height, width)
        return torch.rand(1, 1, *resolution) * 2 * np.pi

    def wrapped_phase(self):
        return torch.remainder(self.phase_raw, 2 * np.pi)

    def centered_phase(self):
        wrapped = self.wrapped_phase()
        return torch.atan2(torch.sin(wrapped), torch.cos(wrapped))

    def forward(self, field):
        mod = torch.exp(1j * self.wrapped_phase())
        if self.mode == "amp_phase":
            mod = mod * torch.sigmoid(self.amp_raw)
        return field * mod


class ASMPropagation(nn.Module):
    def __init__(self, config, distance, wavelength=None, pixel_size=None, resolution=None):
        super().__init__()
        wavelength = config.WAVELENGTH if wavelength is None else wavelength
        pixel_size = config.PIXEL_SIZE if pixel_size is None else pixel_size
        resolution = config.RESOLUTION if resolution is None else resolution
        fx = torch.fft.fftfreq(resolution[0], pixel_size)
        fy = torch.fft.fftfreq(resolution[1], pixel_size)
        fx_grid, fy_grid = torch.meshgrid(fx, fy, indexing="ij")
        k2 = 1 / wavelength ** 2 - fx_grid ** 2 - fy_grid ** 2
        k2 = torch.clamp(k2, min=0)
        self.register_buffer("H", torch.exp(1j * 2 * np.pi * distance * torch.sqrt(k2)))

    def forward(self, field):
        return torch.fft.ifft2(torch.fft.fft2(field) * self.H)


class OpticalStudent(nn.Module):
    def __init__(self, config, enable_norm=None):
        super().__init__()
        self.config = config
        self.slm1 = SLMLayer(config, layer_index=1)
        self.prop1 = ASMPropagation(config, config.PROP_DISTANCE_1)
        self.slm2 = SLMLayer(config, layer_index=2)
        self.prop2 = ASMPropagation(config, config.PROP_DISTANCE_2)
        self.enable_norm = config.ENABLE_STUDENT_NORM if enable_norm is None else enable_norm

    def forward(self, intensity):
        amp = torch.sqrt(intensity.clamp(min=0) + self.config.OPTICAL_FIELD_EPS)
        field = torch.complex(amp, torch.zeros_like(amp))
        field = self.prop1(self.slm1(field))
        field = self.prop2(self.slm2(field))
        out = torch.abs(field) ** 2
        if self.enable_norm:
            norm_mode = str(getattr(self.config, "STUDENT_NORM_MODE", "mean")).lower()
            if norm_mode == "max":
                scale = out.amax(dim=[2, 3], keepdim=True)
            elif norm_mode == "percentile":
                flat = out.flatten(2)
                q = float(getattr(self.config, "STUDENT_NORM_PERCENTILE", 0.995))
                scale = torch.quantile(flat, q, dim=2, keepdim=True).view(out.shape[0], out.shape[1], 1, 1)
            elif norm_mode == "none":
                scale = torch.ones_like(out.mean(dim=[2, 3], keepdim=True))
            else:
                scale = out.mean(dim=[2, 3], keepdim=True)
            out = out / (scale + self.config.OPTICAL_NORM_EPS)
        return out


class OpticalStudentWithDetector(nn.Module):
    def __init__(self, config, detector, enable_norm=None):
        super().__init__()
        self.student = OpticalStudent(config, enable_norm=enable_norm)
        self.detector = detector

    def forward(self, x, return_feature=False):
        feature = self.student(x)
        preds = self.detector(feature)
        if return_feature:
            return feature, preds
        return preds


if __name__ == "__main__":
    from pathlib import Path
    import sys

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from models.SLM.config_slm import ConfigSLM as Config

    output_dir = Path("output/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    original_values = {
        "RESOLUTION": Config.RESOLUTION,
        "IMG_SIZE": Config.IMG_SIZE,
        "SLM_INIT_MODE": Config.SLM_INIT_MODE,
        "SLM_INIT_NOISE_STD": Config.SLM_INIT_NOISE_STD,
        "SLM_DH_PSF_PERIODS": Config.SLM_DH_PSF_PERIODS,
    }

    Config.RESOLUTION = (512, 512)
    Config.IMG_SIZE = 512
    Config.SLM_INIT_MODE = "dh_psf"
    Config.SLM_INIT_NOISE_STD = 0.0

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.0))
    for row, periods in enumerate((1.0, 2.0)):
        Config.SLM_DH_PSF_PERIODS = periods
        student = OpticalStudent(Config)
        for col, layer_name in enumerate(("slm1", "slm2")):
            phase = getattr(student, layer_name).wrapped_phase().detach().squeeze().cpu().numpy()
            ax = axes[row, col]
            im = ax.imshow(phase, cmap="turbo", vmin=0.0, vmax=2 * np.pi)
            ax.set_title(f"{layer_name} dh_psf periods={periods:g}")
            ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04, label="phase [0, 2pi]")
    fig.savefig(output_dir / "slm_dh_psf_initial_phase_test_2.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for key, value in original_values.items():
        setattr(Config, key, value)
    print(f"Saved DH-PSF initialization preview to: {output_dir / 'slm_dh_psf_initial_phase_test_2.png'}")
