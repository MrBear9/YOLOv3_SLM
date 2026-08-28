'''
@File    :   optical_student.py
@Time    :   2026/07/27 12:50:13
@Author  :   Mr.Bear9 
@Github  :   https://github.com/MrBear9
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SLM.asm_propagation import ASMPropagation
from models.SLM.phase_parameterization import _resolve_prop_distance
from models.SLM.slm_modulation import SLMLayer, complex_intensity

class OpticalStudent(nn.Module):
    """N-layer optical student: SLM1鈫扨rop1鈫扴LM2鈫扨rop2鈫?..鈫扴LM_N鈫扨rop_N.

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
            sampling_pitch = (
                config.sampling_pitch(layer_idx)
                if hasattr(config, "sampling_pitch") else config.PIXEL_SIZE
            )
            prop = ASMPropagation(config, dist, pixel_size=sampling_pitch)
            setattr(self, f"slm{layer_idx}", slm)
            setattr(self, f"prop{layer_idx}", prop)

        self.enable_norm = (
            config.ENABLE_STUDENT_NORM if enable_norm is None else enable_norm
        )

    def _propagate_field(self, intensity):
        """Propagate an incident intensity through the configured SLM cascade."""
        amp = torch.sqrt(intensity.clamp(min=0) + self.config.OPTICAL_FIELD_EPS)
        field = torch.complex(amp, torch.zeros_like(amp))
        for layer_idx in range(1, self.num_layers + 1):
            field = getattr(self, f"slm{layer_idx}")(field)
            field = getattr(self, f"prop{layer_idx}")(field)
        return field

    def _postprocess_intensity(self, intensity):
        """Apply the optional blur and detector-domain normalisation."""
        out = intensity
        blur_kernel = int(getattr(self.config, "STUDENT_OUTPUT_BLUR_KERNEL", 1))
        if blur_kernel > 1:
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            out = F.avg_pool2d(out, kernel_size=blur_kernel, stride=1, padding=blur_kernel // 2)
        if self.enable_norm:
            out = self._apply_norm(out)
        return out

    def forward_with_optical_field(self, intensity):
        """Return detector input plus the final, pre-normalisation optical field.

        ``raw_intensity`` is the physical simulated output :math:`|U|^2`
        immediately after the final propagation.  It deliberately precedes
        output blur, normalisation, clamping, detector polarity conversion,
        and any display-only contrast stretch.
        """
        field = self._propagate_field(intensity)
        raw_intensity = complex_intensity(field)
        return self._postprocess_intensity(raw_intensity), {
            "field": field,
            "raw_intensity": raw_intensity,
        }

    def forward(self, intensity):
        feature, _ = self.forward_with_optical_field(intensity)
        return feature

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

    # 鈹€鈹€ iteration interface (used by losses / stats / save) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

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
