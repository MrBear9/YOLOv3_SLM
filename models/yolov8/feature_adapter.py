"""Shared detector-domain transform for teacher and optical features."""


def prepare_detector_feature(config, feature, invert=None):
    """Apply a feature-polarity convention with usable gradients."""
    if invert is None:
        invert = getattr(config, "DETECTOR_INVERT_FEATURE", True)
    if not bool(invert):
        return feature
    feature_min = feature.amin(dim=(2, 3), keepdim=True)
    feature_max = feature.amax(dim=(2, 3), keepdim=True)
    # Detaching extrema gives every feature pixel the same inversion gradient.
    return feature_max.detach() + feature_min.detach() - feature


def prepare_slm_detector_feature(config, feature):
    """Use the detector-domain convention validated for the SLM branch."""
    invert = getattr(config, "SLM_DETECTOR_INVERT_FEATURE", False)
    return prepare_detector_feature(config, feature, invert=invert)
