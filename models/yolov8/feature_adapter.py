"""Shared detector-domain transform for teacher and optical features."""


def prepare_detector_feature(config, feature):
    """Apply the detector's feature-polarity convention with usable gradients."""
    if not bool(getattr(config, "DETECTOR_INVERT_FEATURE", True)):
        return feature
    feature_min = feature.amin(dim=(2, 3), keepdim=True)
    feature_max = feature.amax(dim=(2, 3), keepdim=True)
    # Detaching extrema gives every feature pixel the same inversion gradient.
    return feature_max.detach() + feature_min.detach() - feature
