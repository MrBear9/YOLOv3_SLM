"""Shared construction and decoding for anchor and anchor-free experiments."""

from .anchor_free import AnchorFreeTALLoss, decode_anchor_free
from .decode_anchor_v8 import decode_detections_anchor_v8
from .loss_anchor_v8 import YOLOv3AnchorLossForV8Head


def uses_anchor_free(config):
    return str(getattr(config, "DETECTION_PROTOCOL", "anchor_free_tal")).strip().lower() == "anchor_free_tal"


def build_detection_criterion(config):
    if uses_anchor_free(config) and str(getattr(config, "DETECTOR_HEAD_TYPE", "light")).strip().lower() not in {"light", "yolo_light"}:
        raise ValueError("anchor_free_tal currently requires DETECTOR_HEAD_TYPE='light'.")
    return AnchorFreeTALLoss(config) if uses_anchor_free(config) else YOLOv3AnchorLossForV8Head(config)


def decode_detections(config, predictions, conf_thresh=None, nms_thresh=None, max_det=None):
    decoder = decode_anchor_free if uses_anchor_free(config) else decode_detections_anchor_v8
    return decoder(config, predictions, conf_thresh=conf_thresh, nms_thresh=nms_thresh, max_det=max_det)
