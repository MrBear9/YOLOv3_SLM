import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASMPropagation(nn.Module):
    """Band-limited ASM using neural-holography's centered-field convention.

    The input field is centered in image coordinates. For linear convolution,
    it is center-padded before ``ifftshift -> FFT -> H -> IFFT -> fftshift``
    and center-cropped afterwards. This preserves the optical axis for square
    and rectangular SLM panels alike.
    """

    def __init__(self, config, distance, wavelength=None, pixel_size=None,
                 resolution=None, linear_conv=True):
        super().__init__()
        self._linear_conv = bool(linear_conv)

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
        self.wavelength = wavelength
        self.distance = distance

        padded_height, padded_width = (
            (height * 2, width * 2) if self._linear_conv else (height, width)
        )
        y_len, x_len = padded_height * dy, padded_width * dx

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
        fy_max = 1.0 / math.sqrt((2.0 * distance / y_len) ** 2 + 1.0) / wavelength
        fx_max = 1.0 / math.sqrt((2.0 * distance / x_len) ** 2 + 1.0) / wavelength
        band = (propagating >= 0.0) & (np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)
        band = torch.from_numpy(band.astype(np.float32))
        transfer_centered = torch.complex(band * torch.cos(phase), band * torch.sin(phase))
        self.register_buffer(
            "H", torch.fft.ifftshift(transfer_centered, dim=(-2, -1)).unsqueeze(0).unsqueeze(0)
        )
        self.input_resolution = (height, width)

    def forward(self, field):
        if tuple(field.shape[-2:]) != self.input_resolution:
            raise ValueError(
                f"ASM expected field spatial shape {self.input_resolution}, got {tuple(field.shape[-2:])}."
            )
        if self._linear_conv:
            field = self._center_pad(field, self.H.shape[-2:])
        spectrum = torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm="ortho")
        out = torch.fft.fftshift(torch.fft.ifft2(spectrum * self.H, norm="ortho"), dim=(-2, -1))
        if self._linear_conv:
            out = self._center_crop(out, self.input_resolution)
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

