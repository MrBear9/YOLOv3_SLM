"""Fourier optical layers for SLM-friendly teacher features.

These layers replace the physical-optical CVOCA/complex-convolution blocks with
learnable frequency-domain filters.  They are resolution-agnostic and explicitly
low-pass constrained so the teacher output stays smooth and physically realisable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierOpticalLayer(nn.Module):
    """Learnable radial frequency filter with a residual path.

    The filter is represented as a weighted sum of fixed radial Gaussian basis
    functions.  Only the per-channel band weights are learned, so the layer is
    resolution-agnostic and does not require a fixed input size.

    A learnable low-pass envelope is applied to the final amplitude filter to
    guarantee that high frequencies are attenuated.  This keeps the teacher
    output smooth, which is important for downstream SLM student training.

    Args:
        channels: number of input/output channels.
        num_bands: number of radial frequency basis functions.
        init_low_pass_sigma: initial standard deviation of the low-pass envelope.
    """

    def __init__(self, channels, num_bands=8, init_low_pass_sigma=0.5):
        super().__init__()
        self.channels = int(channels)
        self.num_bands = max(int(num_bands), 1)

        # Per-channel weights for the radial basis bands.
        self.band_weights = nn.Parameter(torch.zeros(self.channels, self.num_bands))

        # Learnable low-pass sigma (constrained positive via softplus).
        self.low_pass_sigma = nn.Parameter(torch.tensor(float(init_low_pass_sigma)))

        # Fixed radial basis centres and width.
        centers = torch.linspace(0.0, 0.5, self.num_bands)
        self.register_buffer("band_centers", centers)
        # Width covers the [0, 0.5] interval comfortably.
        if self.num_bands > 1:
            width = 0.5 / (self.num_bands - 1)
        else:
            width = 0.25
        self.register_buffer("band_width", torch.tensor(width))

        # Optional channel-wise 1x1 projection after IFFT.
        self.post = nn.Sequential(
            nn.BatchNorm2d(self.channels),
            nn.Conv2d(self.channels, self.channels, kernel_size=1, groups=self.channels, bias=False),
            nn.SiLU(),
        )

    def _frequency_grid(self, h, w, device, dtype):
        """Return normalized radial frequency grid for rfft2 output shape."""
        u = torch.fft.fftfreq(h, device=device, dtype=dtype)
        v = torch.fft.rfftfreq(w, device=device, dtype=dtype)
        U, V = torch.meshgrid(u, v, indexing="ij")
        return torch.sqrt(U * U + V * V)

    def _make_amplitude_filter(self, h, w, device, dtype):
        """Build [C, H, W//2+1] real-valued amplitude filter."""
        radius = self._frequency_grid(h, w, device, dtype).unsqueeze(0)  # [1, H, W//2+1]
        centers = self.band_centers.view(-1, 1, 1)                        # [B, 1, 1]
        width = self.band_width

        # Radial Gaussian basis functions.
        basis = torch.exp(-((radius - centers) ** 2) / (2.0 * width ** 2))  # [B, H, W//2+1]
        # Normalise basis so they form a smooth partition of the frequency axis.
        basis = basis / (basis.sum(dim=0, keepdim=True) + 1e-6)

        # Softmax across bands gives positive, sum-to-one weights per channel.
        weights = torch.softmax(self.band_weights, dim=1)  # [C, B]
        amp_filter = torch.einsum("cb,bhw->chw", weights, basis)  # [C, H, W//2+1]

        # Low-pass envelope: high frequencies are always attenuated.
        sigma = F.softplus(self.low_pass_sigma)
        low_pass = torch.exp(-(radius ** 2) / (2.0 * sigma ** 2))
        return amp_filter * low_pass  # [C, H, W//2+1]

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        original_dtype = x.dtype

        # cuFFT in half precision only supports power-of-two sizes.
        # Teacher feature maps such as 80x80 are not powers of two, so we
        # fall back to float32 for the FFT and convert back afterwards.
        needs_fp32_fft = x.is_cuda and original_dtype == torch.float16
        x_fft_input = x.float() if needs_fp32_fft else x

        x_fft = torch.fft.rfft2(x_fft_input, norm="ortho")

        amp_filter = self._make_amplitude_filter(H, W, x_fft_input.device, x_fft_input.dtype)
        # Apply amplitude filter (broadcast over batch).
        y_fft = x_fft * amp_filter.unsqueeze(0)

        y = torch.fft.irfft2(y_fft, s=(H, W), norm="ortho")

        if needs_fp32_fft:
            y = y.to(original_dtype)

        y = self.post(y)

        # Residual connection: at init the filter is near all-pass and post is
        # near identity, so the layer starts as an approximate no-op.
        return y + x
