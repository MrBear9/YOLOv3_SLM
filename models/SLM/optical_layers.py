import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_prop_distance(config, layer_idx):
    """Resolve propagation distance for a given layer."""
    if hasattr(config, "prop_distance"):
        return config.prop_distance(layer_idx)
    return getattr(config, f"PROP_DISTANCE_{layer_idx}", 0.10)


def _get_phase_cfg(config, key, layer_index, default):
    """Resolve a per-layer phase config, preferring OpticalConfig accessors.

    Priority:
    1. ``config.phase_{key}(layer_index)``  — OpticalConfig accessor
    2. ``config.SLM{layer_index}_PHASE_{key}`` — old flat attr
    3. ``config.SLM_PHASE_{key}`` — shared fallback
    4. ``default``
    """
    accessor_name = f"phase_{key.lower()}"
    if hasattr(config, accessor_name):
        return getattr(config, accessor_name)(layer_index)
    old_per_layer = f"SLM{layer_index}_PHASE_{key}"
    if hasattr(config, old_per_layer):
        return getattr(config, old_per_layer)
    old_shared = f"SLM_PHASE_{key}"
    if hasattr(config, old_shared):
        return getattr(config, old_shared)
    return default


# ═══════════════════════════════════════════════════════════════════════════
# Plan C: Fourier Feature Neural Field — coordinate-based smooth phase generator
# ═══════════════════════════════════════════════════════════════════════════

class FourierFeatureField(nn.Module):
    """Implicit neural representation of phase via per-pixel MLP (1×1 convs).

    Takes a coordinate grid with Fourier feature positional encoding and
    produces a smooth phase map.  The MLP's inductive bias toward smooth
    functions acts as a natural regulariser for optical phase patterns.
    """

    def __init__(self, resolution, hidden_dim=64, num_frequencies=6, num_layers=3):
        super().__init__()
        h, w = resolution
        # Fourier feature encoding: sin/cos of powers-of-pi-scaled coords
        in_channels = 2 + 4 * num_frequencies  # (y, x) + 2 * (sin, cos) * num_freq
        self.num_frequencies = num_frequencies

        layers = []
        prev_dim = in_channels
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(prev_dim, hidden_dim, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        layers.append(nn.Conv2d(prev_dim, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

        # Register a persistent coordinate grid so it is always on the right device
        y = torch.linspace(-1.0, 1.0, h)
        x = torch.linspace(-1.0, 1.0, w)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid_2ch = torch.stack([yy, xx], dim=0).unsqueeze(0)  # (1, 2, h, w)
        self.register_buffer("coord_grid", grid_2ch, persistent=True)

        # Precompute Fourier features as a buffer (deterministic, never changes)
        encoded = self._fourier_features(grid_2ch, num_frequencies)
        self.register_buffer("fourier_encoded", encoded, persistent=False)

        self._init_near_zero()

    def _init_near_zero(self):
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1e-4)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _fourier_features(grid, num_frequencies):
        """Encode (y, x) coords with sin/cos at power-of-pi frequencies."""
        features = [grid]
        for i in range(num_frequencies):
            freq = (2 ** i) * torch.pi
            features.append(torch.sin(freq * grid))
            features.append(torch.cos(freq * grid))
        return torch.cat(features, dim=1)  # (B, in_channels, h, w)

    def forward(self):
        return self.net(self.fourier_encoded)  # (1, 1, h, w)


# ═══════════════════════════════════════════════════════════════════════════
# Plan A + B+ + C: Multi-Scale Pyramid + Block-wise (with mini-pyramid) + Neural Field
# ═══════════════════════════════════════════════════════════════════════════

class MultiScalePhaseField(nn.Module):
    """Hierarchical phase parameterisation combining Plans A, B+ and C.

    Plan C — FourierFeatureField provides a smooth global phase base.
    Plan A — learnable parameters at multiple resolutions add global detail.
    Plan B+ — the finest scale is split into overlapping blocks, each with
             its own mini-pyramid (coarse → fine), stitched with feathering.

    ``phase = mlp_field() + sum(upsample(global_scales)) + block_stitch(blocks)``

    The final wrapped phase is SLM-compatible [0, 2π).
    """

    def __init__(self, config, resolution, layer_index=1):
        super().__init__()
        self.resolution = resolution
        self.layer_index = layer_index

        # Shared settings (may come from OpticalConfig or ConfigSLM)
        num_scales = int(getattr(config, "SLM_PHASE_NUM_SCALES",
                                getattr(config, "PHASE_NUM_SCALES", 4)))
        mlp_hidden = int(getattr(config, "SLM_PHASE_MLP_HIDDEN",
                                getattr(config, "PHASE_MLP_HIDDEN", 64)))
        mlp_layers = int(getattr(config, "SLM_PHASE_MLP_LAYERS",
                                getattr(config, "PHASE_MLP_LAYERS", 3)))
        mlp_freqs = _get_phase_cfg(config, "MLP_NUM_FREQS", layer_index, 6)

        # --- Plan C: smooth neural-field base ---
        self.mlp_field = FourierFeatureField(
            resolution,
            hidden_dim=mlp_hidden,
            num_frequencies=mlp_freqs,
            num_layers=mlp_layers,
        )

        # --- Plan B config ---
        self.use_blockwise = bool(getattr(config, "SLM_PHASE_USE_BLOCKWISE",
                                          getattr(config, "PHASE_USE_BLOCKWISE", True)))
        block_grid = _get_phase_cfg(config, "BLOCK_GRID", layer_index, 4)
        block_overlap = int(_get_phase_cfg(config, "BLOCK_OVERLAP", layer_index, 16))
        h, w = resolution

        # --- Plan A: global multi-scale parameters (coarse → medium-fine) ---
        # The finest scale is handled by Plan B when blockwise is enabled.
        if self.use_blockwise:
            # Global scales: exclude the finest, it's replaced by blocks
            global_scales_count = max(num_scales - 1, 1)
        else:
            global_scales_count = num_scales

        scale_resolutions = []
        for i in range(global_scales_count):
            # Use num_scales (not global_scales_count) so the coarsest factor
            # stays the same regardless of blockwise — blocks replace the
            # finest scale, not duplicate it.
            factor = 2 ** (num_scales - 1 - i)
            sh = max(h // factor, 4)
            sw = max(w // factor, 4)
            scale_resolutions.append((sh, sw))
        self.scale_resolutions = scale_resolutions

        self.scale_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, sh, sw))
            for sh, sw in scale_resolutions
        ])

        # --- Plan B: overlapping block-wise parameters (finest scale) ---
        if self.use_blockwise:
            if isinstance(block_grid, (tuple, list)):
                if len(block_grid) != 2:
                    raise ValueError("PHASE_BLOCK_GRID must be an int or (rows, cols).")
                grid_h, grid_w = (int(block_grid[0]), int(block_grid[1]))
            else:
                grid_h = grid_w = int(block_grid)
            if grid_h < 1 or grid_w < 1:
                raise ValueError("PHASE_BLOCK_GRID dimensions must be positive.")
            self.block_grid = (grid_h, grid_w)
            self.block_overlap = max(block_overlap, 0)
            self._feather_slices = self._build_block_slices()

            n_blocks = grid_h * grid_w
            self._n_blocks = n_blocks

            # --- Plan B+: mini-pyramid inside each block ---
            # Each block gets its own coarse→fine scale decomposition so that
            # different spatial regions can learn different frequency mixes.
            # block_inner_scales=1  →  single scale (legacy, same as before)
            # block_inner_scales=2  →  coarse (//4) + fine (//1)   lightweight
            block_inner_scales = _get_phase_cfg(config, "BLOCK_INNER_SCALES", layer_index, 2)
            self._block_inner_scales = max(int(block_inner_scales), 1)

            factors = [2 ** (self._block_inner_scales - k) for k in range(self._block_inner_scales - 1)] + [1]
            self._block_inner_resolutions = []

            self.blocks = nn.ParameterList()
            self._block_offsets = []
            for _, _, _, _, block_h, block_w in self._feather_slices:
                inner_resolutions = [
                    (max(block_h // f, 2), max(block_w // f, 2)) for f in factors
                ]
                self._block_offsets.append(len(self.blocks))
                self._block_inner_resolutions.append(inner_resolutions)
                for sh, sw in inner_resolutions:
                    self.blocks.append(nn.Parameter(torch.zeros(1, 1, sh, sw)))
        else:
            self.blocks = None
            self._n_blocks = 0
            self._block_inner_scales = 1
            self._block_offsets = []

    def _build_block_slices(self):
        """Partition arbitrary rectangular resolutions without stride truncation."""
        H, W = self.resolution
        n_h, n_w = self.block_grid
        slices = []
        y_edges = [round(i * H / n_h) for i in range(n_h + 1)]
        x_edges = [round(i * W / n_w) for i in range(n_w + 1)]
        before = self.block_overlap // 2
        after = self.block_overlap - before
        for i in range(n_h):
            for j in range(n_w):
                y_start = max(0, y_edges[i] - (before if i > 0 else 0))
                y_end = min(H, y_edges[i + 1] + (after if i < n_h - 1 else 0))
                x_start = max(0, x_edges[j] - (before if j > 0 else 0))
                x_end = min(W, x_edges[j + 1] + (after if j < n_w - 1 else 0))
                slices.append((y_start, y_end, x_start, x_end, y_end - y_start, x_end - x_start))
        return slices

    def _build_feather_weights(self, y_len, x_len, i, j, device=None):
        """Linear ramp weights that fade to zero at overlapping edges."""
        if device is None:
            device = self.scale_params[0].device
        wy = torch.ones(y_len, device=device)
        wx = torch.ones(x_len, device=device)
        ov_y = min(self.block_overlap, y_len)
        ov_x = min(self.block_overlap, x_len)
        n_h, n_w = self.block_grid

        if ov_y == 0 and ov_x == 0:
            return wy[:, None] * wx[None, :]
        if i > 0:
            wy[:ov_y] = torch.linspace(0, 1, ov_y, device=device)
        if i < n_h - 1:
            wy[-ov_y:] = torch.linspace(1, 0, ov_y, device=device)
        if j > 0:
            wx[:ov_x] = torch.linspace(0, 1, ov_x, device=device)
        if j < n_w - 1:
            wx[-ov_x:] = torch.linspace(1, 0, ov_x, device=device)
        return wy[:, None] * wx[None, :]

    def _stitch_blocks(self):
        """Stitch overlapping blocks into a single phase map with blending.

        Each block's inner pyramid scales are summed (after upsampling to the
        tile size) before feather-blending into the output.  This lets each
        block express coarse trend + fine detail independently.
        """
        H, W = self.resolution
        device = self.blocks[0].device
        phase = torch.zeros(1, 1, H, W, device=device)
        weight = torch.zeros(1, 1, H, W, device=device)
        n_inner = self._block_inner_scales

        for block_idx in range(self._n_blocks):
            y_start, y_end, x_start, x_end, bh, bw = self._feather_slices[block_idx]

            # Sum inner pyramid scales for this block
            block_phase = None
            for k in range(n_inner):
                param = self.blocks[self._block_offsets[block_idx] + k]
                upsampled = F.interpolate(param, size=(bh, bw),
                                          mode="bilinear", align_corners=False)
                if block_phase is None:
                    block_phase = upsampled
                else:
                    block_phase = block_phase + upsampled

            i, j = divmod(block_idx, self.block_grid[1])
            w = self._build_feather_weights(
                bh, bw, i, j, device=device
            ).to(dtype=block_phase.dtype)

            phase[:, :, y_start:y_end, x_start:x_end] += block_phase * w[None, None, :, :]
            weight[:, :, y_start:y_end, x_start:x_end] += w[None, None, :, :]

        return phase / (weight + 1e-8)

    def set_base_phase(self, phase):
        """Seed the finest-scale parameters with the initial phase pattern.

        In blockwise mode only the finest inner scale receives the spatial
        crop. Coarser scales start at zero so summing the pyramid preserves
        the requested initial phase instead of multiplying it by the number
        of inner scales.
        """
        target_h, target_w = self.resolution
        with torch.no_grad():
            if self.use_blockwise and self.blocks is not None:
                n_inner = self._block_inner_scales
                base = F.interpolate(phase, size=(target_h, target_w),
                                      mode="bilinear", align_corners=False)
                for block_idx in range(self._n_blocks):
                    y_start, y_end, x_start, x_end, _, _ = self._feather_slices[block_idx]
                    crop = base[:, :, y_start:y_end, x_start:x_end]
                    inner_resolutions = self._block_inner_resolutions[block_idx]
                    # Keep the seed in one scale; the other scales remain free residuals.
                    for k in range(n_inner):
                        param = self.blocks[self._block_offsets[block_idx] + k]
                        if k == n_inner - 1:
                            param.copy_(
                                F.interpolate(crop, size=inner_resolutions[k],
                                              mode="bilinear", align_corners=False)
                            )
                        else:
                            param.zero_()
            else:
                self.scale_params[-1].copy_(
                    F.interpolate(phase, size=(target_h, target_w),
                                  mode="bilinear", align_corners=False)
                )

    def forward(self):
        h, w = self.resolution
        device = self.scale_params[0].device

        # Plan C: smooth base from neural field
        phase = self.mlp_field()

        # Plan A: global multi-scale detail
        for param in self.scale_params:
            phase = phase + F.interpolate(param, size=(h, w),
                                          mode="bilinear", align_corners=False)

        # Plan B: block-wise finest-scale detail
        if self.use_blockwise and self.blocks is not None:
            phase = phase + self._stitch_blocks()

        return phase


# ═══════════════════════════════════════════════════════════════════════════
# SLM Layer
# ═══════════════════════════════════════════════════════════════════════════

class SLMLayer(nn.Module):
    def __init__(self, config, resolution=None, mode=None, layer_index=1):
        super().__init__()
        self.config = config
        resolution = config.RESOLUTION if resolution is None else resolution
        mode = config.SLM_MODE if mode is None else mode
        assert mode in {"phase", "amp_phase"}
        self.mode = mode
        self.layer_index = layer_index

        # Choose phase parameterisation
        param_mode = str(getattr(config, "SLM_PHASE_PARAM_MODE", "direct")).lower()
        self._phase_param_mode = param_mode

        if param_mode == "multiscale_mlp":
            self.phase_field = MultiScalePhaseField(config, resolution, layer_index=layer_index)
            # Seed the finest scale with the chosen init pattern
            init_phase = self._initial_phase(resolution)
            self.phase_field.set_base_phase(init_phase)
            self.register_parameter("phase_raw", None)
        else:
            # Legacy / direct mode
            self.phase_raw = nn.Parameter(self._initial_phase(resolution))
            self.phase_field = None

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
        """Return the un-wrapped phase tensor regardless of parameterisation."""
        if self._phase_param_mode == "multiscale_mlp" and self.phase_field is not None:
            return self.phase_field()
        return self.phase_raw

    def wrapped_phase(self):
        return torch.remainder(self._raw_phase(), 2 * np.pi)

    def centered_phase(self):
        wrapped = self.wrapped_phase()
        return torch.atan2(torch.sin(wrapped), torch.cos(wrapped))

    def forward(self, field):
        mod = torch.exp(1j * self.wrapped_phase())
        if self.mode == "amp_phase":
            mod = mod * torch.sigmoid(self.amp_raw)
        return field * mod


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

class OpticalStudent(nn.Module):
    """N-layer optical student: SLM1→Prop1→SLM2→Prop2→...→SLM_N→Prop_N.

    Number of layers controlled by ``config.NUM_LAYERS`` (default 2).
    Each layer has its own ``SLMLayer`` and ``ASMPropagation``.
    Backward-compatible ``slm1``/``slm2``/``prop1``/``prop2`` attributes
    are set for the first two layers.
    """

    def __init__(self, config, enable_norm=None):
        super().__init__()
        self.config = config
        self.num_layers = int(getattr(config, "NUM_LAYERS", 2))

        for layer_idx in range(1, self.num_layers + 1):
            slm = SLMLayer(config, layer_index=layer_idx)
            dist = _resolve_prop_distance(config, layer_idx)
            prop = ASMPropagation(config, dist)
            setattr(self, f"slm{layer_idx}", slm)
            setattr(self, f"prop{layer_idx}", prop)

        self.enable_norm = (
            config.ENABLE_STUDENT_NORM if enable_norm is None else enable_norm
        )

    def forward(self, intensity):
        amp = torch.sqrt(intensity.clamp(min=0) + self.config.OPTICAL_FIELD_EPS)
        field = torch.complex(amp, torch.zeros_like(amp))
        for layer_idx in range(1, self.num_layers + 1):
            field = getattr(self, f"slm{layer_idx}")(field)
            field = getattr(self, f"prop{layer_idx}")(field)
        out = torch.abs(field) ** 2
        blur_kernel = int(getattr(self.config, "STUDENT_OUTPUT_BLUR_KERNEL", 1))
        if blur_kernel > 1:
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            out = F.avg_pool2d(out, kernel_size=blur_kernel, stride=1, padding=blur_kernel // 2)
        if self.enable_norm:
            out = self._apply_norm(out)
        return out

    def _apply_norm(self, out):
        """Per-sample normalisation (shared with MultiHeadOpticalStudent)."""
        norm_mode = str(getattr(self.config, "STUDENT_NORM_MODE", "mean")).lower()
        if norm_mode == "max":
            scale = out.amax(dim=[2, 3], keepdim=True)
        elif norm_mode == "percentile":
            flat = out.flatten(2)
            q = float(getattr(self.config, "STUDENT_NORM_PERCENTILE", 0.995))
            scale = torch.quantile(flat, q, dim=2, keepdim=True).view(
                out.shape[0], out.shape[1], 1, 1
            )
        elif norm_mode == "none":
            scale = torch.ones_like(out.mean(dim=[2, 3], keepdim=True))
        else:
            scale = out.mean(dim=[2, 3], keepdim=True)
        out = out / (scale + self.config.OPTICAL_NORM_EPS)
        clamp_max = float(getattr(self.config, "STUDENT_OUTPUT_CLAMP_MAX", 0.0))
        if clamp_max > 0:
            out = out.clamp(max=clamp_max)
        return out

    # ── iteration interface (used by losses / stats / save) ──────────────

    def all_slm_layers(self):
        """Yield ``(name, slm_layer)`` for every SLM layer."""
        for i in range(1, self.num_layers + 1):
            yield f"slm{i}", getattr(self, f"slm{i}")


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
