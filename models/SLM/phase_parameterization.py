'''
@File    :   phase_parameterization.py
@Time    :   2026/07/27 12:49:49
@Author  :   Mr.Bear9 
@Github  :   https://github.com/MrBear9
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


# Helpers

def _resolve_prop_distance(config, layer_idx):
    """Resolve propagation distance for a given layer."""
    if hasattr(config, "prop_distance"):
        return config.prop_distance(layer_idx)
    return getattr(config, f"PROP_DISTANCE_{layer_idx}", 0.10)


def _get_phase_cfg(config, key, layer_index, default):
    """Resolve a per-layer phase config, preferring OpticalConfig accessors.

    Priority:
    1. ``config.phase_{key}(layer_index)``  鈥?OpticalConfig accessor
    2. ``config.SLM{layer_index}_PHASE_{key}`` 鈥?old flat attr
    3. ``config.SLM_PHASE_{key}`` 鈥?shared fallback
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


# Plan C: Fourier Feature Neural Field 鈥?coordinate-based smooth phase generator

class FourierFeatureField(nn.Module):
    """Implicit neural representation of phase via per-pixel MLP (1脳1 convs).

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


# Plan A + B+ + C: Multi-Scale Pyramid + Block-wise (with mini-pyramid) + Neural Field

class MultiScalePhaseField(nn.Module):
    """Hierarchical phase parameterisation combining Plans A, B+ and C.

    Plan C 鈥?FourierFeatureField provides a smooth global phase base.
    Plan A 鈥?learnable parameters at multiple resolutions add global detail.
    Plan B+ 鈥?the finest scale is split into overlapping blocks, each with
             its own mini-pyramid (coarse 鈫?fine), stitched with feathering.

    ``phase = mlp_field() + sum(upsample(global_scales)) + block_stitch(blocks)``

    The final wrapped phase is SLM-compatible [0, 2蟺).
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

        # --- Plan A: global multi-scale parameters (coarse 鈫?medium-fine) ---
        # The finest scale is handled by Plan B when blockwise is enabled.
        if self.use_blockwise:
            # Global scales: exclude the finest, it's replaced by blocks
            global_scales_count = max(num_scales - 1, 1)
        else:
            global_scales_count = num_scales

        scale_resolutions = []
        for i in range(global_scales_count):
            # Use num_scales (not global_scales_count) so the coarsest factor
            # stays the same regardless of blockwise 鈥?blocks replace the
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
            # Each block gets its own coarse鈫抐ine scale decomposition so that
            # different spatial regions can learn different frequency mixes.
            # block_inner_scales=1  鈫? single scale (legacy, same as before)
            # block_inner_scales=2  鈫? coarse (//4) + fine (//1)   lightweight
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

    def zero_residual(self):
        """Make the initial residual zero without disabling MLP gradients."""
        with torch.no_grad():
            for parameter in self.scale_params:
                parameter.zero_()
            if self.blocks is not None:
                for parameter in self.blocks:
                    parameter.zero_()
            # A zero output projection gives an exactly zero initial residual.
            # Keep preceding layers at their small random initialization so the
            # final projection receives non-zero features and can learn.
            output_layer = self.mlp_field.net[-1]
            output_layer.weight.zero_()
            if output_layer.bias is not None:
                output_layer.bias.zero_()

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

# SLM Layer


