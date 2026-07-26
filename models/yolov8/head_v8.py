"""YOLOv8 detector head public API.

Re-exports build_detector_head() and TeacherWithDetector().
Actual head implementations live in detection_heads.py,
building blocks live in building_blocks.py.

Note: YOLOv8AnchorHead / EnhancedYOLOv8AnchorHead were removed from the
factory because they depend on the legacy anchor protocol (ratio/SimOTA)
which has been deprecated. See docs/HeadIdea/deprecated-matching-strategies.md.
"""

import torch.nn as nn

from models.runtime import prepare_conv_tensor
from models.teacher import build_teacher

from .detection_heads import YOLOLightHead
from .feature_adapter import prepare_detector_feature


def build_detector_head(config, in_channels=1, out_channels=None):
    """Factory: build the configured detector head type.

    Supported head_type values:
      - "light" / "yolo_light": YOLOLightHead (anchor-free TAL)
      - "compact" / "center_detect": CompactOpticalDetector

    The *out_channels* parameter is accepted for backward compatibility
    but is no longer used by any current head type.
    """
    head_type = str(getattr(config, "DETECTOR_HEAD_TYPE", "light")).strip().lower()
    if head_type in {"light", "yolo_light"}:
        base_ch = int(getattr(config, "YOLO_LIGHT_BASE_CH", 8))
        return YOLOLightHead(config, in_channels=in_channels, base_ch=base_ch)
    if head_type in {"compact", "center_detect"}:
        from models.compact_detect.model import CompactOpticalDetector           # V1 (anchor-free, 109 K)
        from models.compact_detect.model_v2 import CompactOpticalDetectorV2      # V2 (Level-4 decoupled, ~32 K)
        version = str(getattr(config, "COMPACT_MODEL_VERSION", "v2")).strip().lower()
        if version in {"v2", "2"}:
            return CompactOpticalDetectorV2(config, in_channels=in_channels)
        return CompactOpticalDetector(config, in_channels=in_channels)
    # Legacy anchor head types ("yolov8_anchor", "yolov8_anchor_legacy") are no
    # longer supported — the anchor protocol (ratio/SimOTA) was removed.
    # Use "light" for anchor-free TAL detection.
    if head_type in {"yolov8_anchor", "yolov8_anchor_legacy", "enhanced", "legacy"}:
        raise ValueError(
            f"DETECTOR_HEAD_TYPE={head_type!r} requires the legacy anchor protocol "
            f"which has been removed. Use 'light' or 'compact' instead."
        )
    raise ValueError(
        f"Unsupported DETECTOR_HEAD_TYPE={head_type!r}; choose 'light' or 'compact'."
    )


class TeacherWithDetector(nn.Module):
    """Teacher + Detector wrapper.

    Supports YOLOLightHead (anchor-free TAL) and CompactOpticalDetector
    via ``build_detector_head(config)``.  The detector type is controlled
    by ``DETECTOR_HEAD_TYPE`` in config.
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
        det_input = prepare_detector_feature(self.config, teacher_feature)

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
