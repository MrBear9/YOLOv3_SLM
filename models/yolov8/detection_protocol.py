"""Detection protocol — anchor-free TAL (Task-Aligned Learning) only.

The legacy anchor-based protocol (ratio / SimOTA matching) was removed.
See docs/HeadIdea/deprecated-matching-strategies.md for details.
"""

from .anchor_free import AnchorFreeTALLoss, decode_anchor_free


def build_detection_criterion(config):
    return AnchorFreeTALLoss(config)


def decode_detections(config, predictions, conf_thresh=None, nms_thresh=None, max_det=None):
    return decode_anchor_free(config, predictions, conf_thresh=conf_thresh, nms_thresh=nms_thresh, max_det=max_det)
