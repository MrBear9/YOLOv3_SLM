"""Optional exact-DFT ASM kernel; selected explicitly with asm_kernel: fftfreq.

Only kernel construction lives here. ASMPropagation owns the shared padding,
FFT, cropping and checkpoint validation for both kernels.
"""
import math

import numpy as np
import torch


def build_fftfreq_kernel(shape, pitch, wavelength, distance, band_limits):
    """Return the centred complex64 transfer function (uniform carrier omitted)."""
    height, width = shape
    dy, dx = pitch
    fy_max, fx_max = band_limits
    fy = np.fft.fftshift(np.fft.fftfreq(height, d=dy))
    fx = np.fft.fftshift(np.fft.fftfreq(width, d=dx))
    radial2 = fy[:, None] ** 2 + fx[None, :] ** 2
    root = np.sqrt(np.maximum(wavelength ** -2 - radial2, 0))
    # Stable float64 form of 2*pi*z*(sqrt(lambda^-2-f^2)-lambda^-1).
    relative_phase = -2 * math.pi * distance * radial2 / (root + 1 / wavelength)
    valid = ((radial2 <= wavelength ** -2) &
             (np.abs(fy[:, None]) < fy_max) & (np.abs(fx[None, :]) < fx_max))
    return torch.from_numpy((valid * np.exp(1j * relative_phase)).astype(np.complex64))
