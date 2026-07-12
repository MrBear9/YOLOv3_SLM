"""YOLOv8 detector head public API.

Re-exports build_detector_head() and TeacherWithDetector().
Actual head implementations live in detection_heads.py,
building blocks live in building_blocks.py.
"""

import torch.nn as nn

from models.runtime import prepare_conv_tensor
from models.teacher import build_teacher

from .detection_heads import (
    EnhancedYOLOv8AnchorHead,
    YOLOLightHead,
    YOLOv8AnchorHead,
)


def build_detector_head(config, in_channels=1, out_channels=None):
    """Factory: build the configured detector head type.

    Supported head_type values:
      - "yolov8_anchor" (default): EnhancedYOLOv8AnchorHead with ECA + deeper branches
      - "yolov8_anchor_legacy": original YOLOv8AnchorHead (kept for comparison)
      - "light" / "yolo_light": YOLOLightHead
    """
    head_type = str(getattr(config, "DETECTOR_HEAD_TYPE", "yolov8_anchor")).strip().lower()
    out_channels = config.get_detector_output_channels() if out_channels is None else out_channels
    if head_type in {"light", "yolo_light"}:
        base_ch = int(getattr(config, "YOLO_LIGHT_BASE_CH", 8))
        return YOLOLightHead(config, in_channels=in_channels, out_channels=out_channels, base_ch=base_ch)
    base_ch = int(getattr(config, "YOLOV8_BASE_CHANNELS", 32))
    c2f_blocks = int(getattr(config, "YOLOV8_C2F_BLOCKS", 3))
    if head_type in {"yolov8_anchor_legacy", "legacy"}:
        return YOLOv8AnchorHead(config, in_channels=in_channels, out_channels=out_channels, base_ch=base_ch, c2f_blocks=c2f_blocks)
    if head_type in {"yolov8_anchor", "enhanced"}:
        return EnhancedYOLOv8AnchorHead(config, in_channels=in_channels, out_channels=out_channels, base_ch=base_ch, c2f_blocks=c2f_blocks)
    raise ValueError(
        f"Unsupported DETECTOR_HEAD_TYPE={head_type!r}; choose 'light', "
        "'yolov8_anchor', or 'yolov8_anchor_legacy'."
    )


class TeacherWithDetector(nn.Module):
    """Teacher + Detector wrapper.

    Supports both YOLOv8AnchorHead and YOLOLightHead via
    ``build_detector_head(config)``.  The detector type is controlled by
    ``DETECTOR_HEAD_TYPE`` in config.
    """

    def __init__(self, config, teacher=None, detector=None):
        super().__init__()
        self.config = config
        self.teacher = build_teacher(config) if teacher is None else teacher
        self.detector = build_detector_head(config, in_channels=1) if detector is None else detector

    def forward(self, x, return_feature=False, return_teacher_aux=False, return_det_features=False):
        x = prepare_conv_tensor(self.config, x)
        need_aux = return_teacher_aux or return_det_features
        teacher_out = self.teacher(x, return_aux=need_aux)
        teacher_feature = teacher_out["det_feature"] if need_aux else teacher_out

        # Conditionally invert for YOLO detector: optical features have dark targets
        # (low values), but YOLO expects bright targets (high values).  When enabled,
        # we invert with detached min/max so every pixel gets a uniform gradient of
        # -1 — no gradient concentration at extreme pixels.  The original value range
        # is preserved (no normalisation to [0,1]) so the detector's BatchNorm
        # distributions stay in a reasonable regime.
        if getattr(self.config, "DETECTOR_INVERT_FEATURE", True):
            t_min = teacher_feature.amin(dim=(2, 3), keepdim=True)
            t_max = teacher_feature.amax(dim=(2, 3), keepdim=True)
            det_input = t_max.detach() + t_min.detach() - teacher_feature
        else:
            det_input = teacher_feature

        det_out = self.detector(prepare_conv_tensor(self.config, det_input), return_features=return_det_features)
        if return_det_features:
            detections, det_features = det_out
        else:
            detections, det_features = det_out, None

        if not return_feature and not need_aux:
            return detections
        result = []
        if return_feature:
            result.append(teacher_feature)
        result.append(detections)
        if return_teacher_aux:
            result.append(teacher_out)
        if return_det_features:
            result.append(det_features)
        return tuple(result)
