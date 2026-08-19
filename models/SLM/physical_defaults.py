"""Shared bench geometry for the DMD, physical SLMs, and numerical ASM grid.

Hardware pitch and numerical sampling pitch are deliberately separate.  The
latter is an empirical propagation-model parameter inherited from the
hardware-validated HolographSLM experiments; it must not be used to decide how
many physical pixels to illuminate or export.
"""

DEFAULT_WAVELENGTH = 532e-9
DEFAULT_DMD_RESOLUTION = (640, 640)
DEFAULT_DMD_PIXEL_PITCH = 5.4e-6
# Current cascaded bench: SLM1 -> 10 cm -> SLM2 -> 10 cm -> sensor plane.
DEFAULT_PROPAGATION_DISTANCES = (0.10, 0.10)

SLM_PROFILES = {
    "zk_weixing_8p0": {
        "model": "中科微兴",
        "hardware_pixel_pitch": 8.0e-6,
        "effective_sampling_pitch": 11.0e-6,
    },
    "magicholo_4p5": {
        "model": "MagicHolo",
        "hardware_pixel_pitch": 4.5e-6,
        "effective_sampling_pitch": 6.4e-6,
    },
}

DEFAULT_SLM_LAYER_PROFILES = {
    1: "zk_weixing_8p0",
    2: "magicholo_4p5",
}

# Compatibility default for callers that still accept one scalar pitch.
DEFAULT_PIXEL_SIZE = SLM_PROFILES[DEFAULT_SLM_LAYER_PROFILES[2]]["effective_sampling_pitch"]


def aperture_size(resolution=DEFAULT_DMD_RESOLUTION, pixel_pitch=DEFAULT_DMD_PIXEL_PITCH):
    """Return the illuminated physical aperture as ``(height, width)`` metres."""
    height, width = (int(value) for value in resolution)
    if min(height, width) < 1 or float(pixel_pitch) <= 0:
        raise ValueError("DMD resolution and pixel pitch must be positive.")
    return height * float(pixel_pitch), width * float(pixel_pitch)


def active_pixel_shape(profile_name, aperture_hw=None):
    """Return SLM pixels covering the DMD aperture, rounded to whole pixels."""
    if profile_name not in SLM_PROFILES:
        raise ValueError(f"Unknown SLM profile: {profile_name!r}.")
    aperture_hw = aperture_size() if aperture_hw is None else aperture_hw
    pitch = SLM_PROFILES[profile_name]["hardware_pixel_pitch"]
    return tuple(int(round(float(length) / pitch)) for length in aperture_hw)

