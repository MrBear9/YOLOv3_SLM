"""Version-dispatch factories for compact detection.

Provides three functions that let training loops pick the right
criterion, decode function, and loss-stat keys based on
``COMPACT_MODEL_VERSION`` without hard-coding branch logic everywhere.

Usage::

    from .factory import build_compact_criterion, build_compact_decode_fn, get_compact_loss_keys

    criterion  = build_compact_criterion(Config)
    decode_fn  = build_compact_decode_fn(Config)
    stat_keys  = get_compact_loss_keys(Config)
"""

__all__ = [
    "build_compact_criterion",
    "build_compact_decode_fn",
    "get_compact_loss_keys",
]


def _compact_version(config):
    return str(getattr(config, "COMPACT_MODEL_VERSION", "v1")).strip().lower()


def build_compact_criterion(config):
    """Return the appropriate loss module for the configured version.

    V1 → ``CenterDetectionLoss``   (heatmap + wh + offset)
    V2 → ``DecoupledCenterLoss``    (obj + cls + wh + offset)
    """
    if _compact_version(config) in {"v2", "2"}:
        from .losses import DecoupledCenterLoss
        return DecoupledCenterLoss(config)
    from .losses import CenterDetectionLoss
    return CenterDetectionLoss(config)


def build_compact_decode_fn(config):
    """Return the appropriate decode *function* for the configured version.

    V1 → ``decode_center_detections``
    V2 → ``decode_decoupled``
    """
    if _compact_version(config) in {"v2", "2"}:
        from .decode import decode_decoupled
        return decode_decoupled
    from .decode import decode_center_detections
    return decode_center_detections


def get_compact_loss_keys(config):
    """Return the tuple of loss-stat dict keys for TensorBoard / logging.

    V1 → ``("heatmap", "wh", "offset", "feature_total")``
    V2 → ``("obj", "cls", "wh", "offset", "feature_total")``
    """
    if _compact_version(config) in {"v2", "2"}:
        return ("obj", "cls", "wh", "offset", "feature_total")
    return ("heatmap", "wh", "offset", "feature_total")
