import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASMPropagation(nn.Module):
    """Band-limited ASM using neural-holography's centered-field convention.

    The input field is centered in image coordinates. For linear convolution,
    it is center-padded before ``ifftshift -> FFT -> H -> IFFT -> fftshift``
    and center-cropped afterwards. P controls padding; K repeats complex pixels
    on a pitch/K grid, then averages complex blocks back to the input grid.
    K does not add phase parameters or model camera intensity integration.
    """

    def __init__(self, config, distance, wavelength=None, pixel_size=None,
                 resolution=None, linear_conv=True):
        super().__init__()
        padding_factor = getattr(config, "ASM_PADDING_FACTOR", 2)
        supersample_factor = getattr(config, "ASM_SUPERSAMPLE_FACTOR", 1)
        for name, value in (("ASM_PADDING_FACTOR", padding_factor),
                            ("ASM_SUPERSAMPLE_FACTOR", supersample_factor)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        # Retain the explicit legacy opt-out used by numerical diagnostics.
        self.padding_factor = padding_factor if linear_conv else 1
        self.supersample_factor = supersample_factor
        self._linear_conv = self.padding_factor > 1

        wavelength = config.WAVELENGTH if wavelength is None else wavelength
        pixel_size = config.PIXEL_SIZE if pixel_size is None else pixel_size
        resolution = config.RESOLUTION if resolution is None else resolution
        height, width = int(resolution[0]), int(resolution[1])
        if isinstance(pixel_size, (tuple, list)):
            dy, dx = float(pixel_size[0]), float(pixel_size[1])
        else:
            dy = dx = float(pixel_size)
        wavelength, distance = float(wavelength), float(distance)
        if height < 1 or width < 1 or dy <= 0 or dx <= 0 or wavelength <= 0 or distance == 0:
            raise ValueError("ASM resolution, pixel size, wavelength, and non-zero distance must be valid.")
        self.pixel_size = (dy, dx)
        dy, dx = dy / supersample_factor, dx / supersample_factor
        self.fine_pixel_size = (dy, dx)
        self.wavelength = wavelength
        self.distance = distance

        self.fine_resolution = (height * supersample_factor, width * supersample_factor)
        padded_height, padded_width = (n * self.padding_factor for n in self.fine_resolution)
        y_len, x_len = padded_height * dy, padded_width * dx

        fy_max = 1.0 / math.sqrt((2.0 * distance / y_len) ** 2 + 1.0) / wavelength
        fx_max = 1.0 / math.sqrt((2.0 * distance / x_len) ** 2 + 1.0) / wavelength
        if getattr(config, "OPTICAL_COORDINATE_MODE", "legacy") == "physical":
            # Physical mode uses the exact DFT bins. Compute the relative phase
            # in float64 to avoid losing diffraction phase in the large k*z
            # carrier; the omitted carrier is a spatially uniform phase only.
            fy_exact = np.fft.fftshift(np.fft.fftfreq(padded_height, d=dy))
            fx_exact = np.fft.fftshift(np.fft.fftfreq(padded_width, d=dx))
            radial2 = fy_exact[:, None] ** 2 + fx_exact[None, :] ** 2
            root = np.sqrt(np.maximum(wavelength ** -2 - radial2, 0))
            relative_phase = -2 * math.pi * distance * radial2 / (root + 1 / wavelength)
            valid = ((radial2 <= wavelength ** -2) &
                     (np.abs(fy_exact[:, None]) < fy_max) & (np.abs(fx_exact[None, :]) < fx_max))
            transfer_centered = torch.from_numpy((valid * np.exp(1j * relative_phase)).astype(np.complex64))
        else:
            # Match the validated neural-holography ordering: centered H, then ifftshift.
            fy = np.linspace(
                -1 / (2 * dy) + 0.5 / (2 * y_len),
                1 / (2 * dy) - 0.5 / (2 * y_len),
                padded_height,
            )
            fx = np.linspace(
                -1 / (2 * dx) + 0.5 / (2 * x_len),
                1 / (2 * dx) - 0.5 / (2 * x_len),
                padded_width,
            )
            FX, FY = np.meshgrid(fx, fy)
            propagating = 1.0 / wavelength ** 2 - (FX ** 2 + FY ** 2)
            # Match the validated neural-holography implementation: quantize the
            # phase-per-metre grid to float32 before applying the propagation distance.
            phase_per_metre = torch.from_numpy(
                2.0 * math.pi * np.sqrt(np.clip(propagating, 0.0, None))
            ).to(dtype=torch.float32)
            phase = phase_per_metre * distance
            band = (propagating >= 0.0) & (np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)
            band = torch.from_numpy(band.astype(np.float32))
            transfer_centered = torch.complex(band * torch.cos(phase), band * torch.sin(phase))
        self.register_buffer(
            "H", torch.fft.ifftshift(transfer_centered, dim=(-2, -1)).unsqueeze(0).unsqueeze(0)
        )
        self.input_resolution = (height, width)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # H is a cached physical operator, not a learned parameter. Never let an
        # old checkpoint silently override the geometry selected in optics.yaml.
        key = prefix + "H"
        saved = state_dict.get(key)
        if saved is not None and (
            saved.shape != self.H.shape or not torch.allclose(
                saved.to(device=self.H.device, dtype=self.H.dtype), self.H,
                rtol=1e-5, atol=1e-6,
            )
        ):
            error_msgs.append(
                f"{key}: checkpoint ASM kernel differs from configured optics "
                f"(wavelength={self.wavelength:g} m, pitch={self.pixel_size} m, "
                f"distance={self.distance:g} m, P={self.padding_factor}, K={self.supersample_factor}). Use the checkpoint's original "
                "configs/optics.yaml (legacy sampling: 13.0 / 6.4 um), or train "
                "a new teacher/student pair for the new geometry."
            )
            # Preserve the configured kernel even when load_state_dict raises.
            state_dict[key] = self.H
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, field):
        if tuple(field.shape[-2:]) != self.input_resolution:
            raise ValueError(
                f"ASM expected field spatial shape {self.input_resolution}, got {tuple(field.shape[-2:])}."
            )
        k = self.supersample_factor
        if k > 1:
            field = field.repeat_interleave(k, dim=-2).repeat_interleave(k, dim=-1)
        if self._linear_conv:
            field = self._center_pad(field, self.H.shape[-2:])
        spectrum = torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm="ortho")
        out = torch.fft.fftshift(torch.fft.ifft2(spectrum * self.H, norm="ortho"), dim=(-2, -1))
        if self._linear_conv:
            out = self._center_crop(out, self.fine_resolution)
        if k > 1:
            height, width = self.input_resolution
            out = out.reshape(*out.shape[:-2], height, k, width, k).mean(dim=(-3, -1))
        return out

    @staticmethod
    def _center_pad(field, target_shape):
        height, width = field.shape[-2:]
        target_height, target_width = target_shape
        diff_h, diff_w = target_height - height, target_width - width
        top = (diff_h + height % 2) // 2
        bottom = (diff_h + 1 - height % 2) // 2
        left = (diff_w + width % 2) // 2
        right = (diff_w + 1 - width % 2) // 2
        return F.pad(field, (left, right, top, bottom))

    @staticmethod
    def _center_crop(field, target_shape):
        target_height, target_width = target_shape
        height, width = field.shape[-2:]
        diff_h, diff_w = height - target_height, width - target_width
        top = (diff_h + 1 - target_height % 2) // 2
        left = (diff_w + 1 - target_width % 2) // 2
        return field[..., top:top + target_height, left:left + target_width]

