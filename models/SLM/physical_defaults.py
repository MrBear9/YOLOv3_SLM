"""Shared training optics defaults loaded from the main project YAML.

Hardware pitch determines the illuminated/exported pixel count. ASM sampling
defaults to that pitch; explicit overrides reproduce older numerical models.
"""

import argparse
import math
import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def add_optics_argument(parser):
    """Register the shared YAML selector on a public command-line parser."""
    parser.allow_abbrev = False
    parser.add_argument("--optics", type=Path, default=None,
                        help="Optical YAML path (default: project configs/optics.yaml).")


def resolve_optics_path(argv=None):
    """Resolve before config classes bind their constants at import time.

    torchrun forwards script arguments to every worker. Parsing only --optics
    here makes training, evaluation and exports use one YAML without mutable
    environment state or reloading already-imported model configurations.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_optics_argument(parser)
    args, _ = parser.parse_known_args(argv)
    path = args.optics if args.optics is not None else PROJECT_ROOT / "configs/optics.yaml"
    return path.expanduser().resolve()


OPTICS_CONFIG_PATH = resolve_optics_path()


def _positive(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be positive and finite.")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite.")
    return value


def _shape(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(
        type(v) is not int or v < 1 for v in value
    ):
        raise ValueError(f"{label} must be [positive integer height, width].")
    return tuple(value)


def load_optics(path=OPTICS_CONFIG_PATH):
    """Validate a YAML file and resolve its units without mutating configs."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Optics YAML must contain a mapping.")
    try:
        resolution = _shape(data["resolution"], "resolution")
        dmd_pitch = _positive(data["dmd"]["pixel_pitch_um"], "dmd.pixel_pitch_um") * 1e-6
        magnification = _positive(data["dmd"].get("relay_magnification", 1), "dmd.relay_magnification")
        hardware, propagation, export = (data[k] for k in ("hardware", "propagation", "phase_export"))
        coordinate_mode = propagation.get("coordinate_mode", "legacy")
        if coordinate_mode not in ("physical", "legacy"):
            raise ValueError("propagation.coordinate_mode must be physical or legacy.")
        extent_factor = _positive(propagation.get("field_extent_factor", 1.5), "field_extent_factor")
        if extent_factor < 1:
            raise ValueError("field_extent_factor must be at least one.")
        grid_multiple = propagation.get("grid_multiple", 32)
        max_grid_size = propagation.get("max_grid_size", 2048)
        if any(type(v) is not int or v < 1 for v in (grid_multiple, max_grid_size)):
            raise ValueError("grid_multiple and max_grid_size must be positive integers.")
        factors = {}
        for key, default in (("padding_factor", 2), ("supersample_factor", 1)):
            value = propagation.get(key, default)
            if type(value) is not int or value < 1:
                raise ValueError(f"propagation.{key} must be a positive integer.")
            factors[key] = value
        wavelength = _positive(hardware["wavelength_nm"], "hardware.wavelength_nm") * 1e-9
        sampling = propagation["sampling_pitch_um"]
        if not isinstance(sampling, dict) or set(sampling) - set(hardware["profiles"]):
            raise ValueError("sampling_pitch_um must map known profile names to pitches or null.")
        profiles = {}
        for name, entry in hardware["profiles"].items():
            pitch = _positive(entry["pixel_pitch_um"], f"{name}.pixel_pitch_um") * 1e-6
            override = sampling.get(name)
            fill = _positive(entry["fill_factor_pct"], f"{name}.fill_factor_pct")
            if fill > 100:
                raise ValueError(f"{name}.fill_factor_pct must not exceed 100.")
            profiles[name] = {
                "model": str(entry["model"]),
                "resolution": _shape(entry["resolution"], f"{name}.resolution"),
                "hardware_pixel_pitch": pitch,
                "effective_sampling_pitch": pitch if override is None else _positive(
                    override, f"sampling_pitch_um.{name}"
                ) * 1e-6,
                "fill_factor_pct": fill,
            }
        layers = propagation["layers"]
        if not isinstance(layers, list) or not 1 <= len(layers) <= 4:
            raise ValueError("Configure one to four propagation layers; unused layers are reserved.")
        num_layers = propagation.get("num_layers", min(2, len(layers)))
        if type(num_layers) is not int or not 1 <= num_layers <= len(layers):
            raise ValueError("propagation.num_layers must select one to four configured layers.")
        layer_profiles, distances = {}, []
        for index, layer in enumerate(layers, 1):
            name = layer["profile"]
            if name not in profiles:
                raise ValueError(f"Unknown SLM profile for layer {index}: {name!r}.")
            profile = profiles[name]
            active = tuple(round(n * dmd_pitch * magnification / profile["hardware_pixel_pitch"]) for n in resolution)
            if any(n < 1 or n > full for n, full in zip(active, profile["resolution"])):
                raise ValueError(f"Layer {index}: illuminated aperture does not fit the SLM panel.")
            layer_profiles[index] = name
            distances.append(_positive(layer["distance_cm"], f"layer {index}.distance_cm") * 1e-2)
        if type(export["levels"]) is not int or export["levels"] != 256:
            raise ValueError("Current hardware export requires 256 gray levels.")
        if type(export["gray_inverted"]) is not bool:
            raise ValueError("phase_export.gray_inverted must be boolean.")
        offset = float(export["phase_offset_pi"]) * math.pi
        if not math.isfinite(offset):
            raise ValueError("phase_export.phase_offset_pi must be finite.")
        lut = export["gray_to_phase_lut"]
        if lut:
            lut = Path(lut)
            lut = str((path.parent / lut).resolve()) if not lut.is_absolute() else str(lut)
        return dict(
            resolution=resolution, dmd_pixel_pitch=dmd_pitch, wavelength=wavelength,
            profiles=profiles, layer_profiles=layer_profiles, distances=tuple(distances),
            phase_levels=export["levels"], gray_inverted=export["gray_inverted"],
            phase_offset=offset, gray_to_phase_lut=lut or "",
            coordinate_mode=coordinate_mode, relay_magnification=magnification,
            num_layers=num_layers,
            field_extent_factor=extent_factor, grid_multiple=grid_multiple, max_grid_size=max_grid_size,
            **factors,
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid optics YAML {path}: {error}") from error


OPTICS = load_optics()


def save_optics_snapshot(output_dir, num_layers=None):
    """Save a replayable YAML with the run; only the main DDP rank writes it."""
    if int(os.environ.get("RANK", "0")) != 0:
        return
    source = yaml.safe_load(OPTICS_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    source["propagation"].setdefault("num_layers", OPTICS["num_layers"])
    if num_layers is not None:
        source["propagation"]["num_layers"] = int(num_layers)
    if OPTICS["gray_to_phase_lut"]:
        source["phase_export"]["gray_to_phase_lut"] = OPTICS["gray_to_phase_lut"]
    destination = Path(output_dir) / "optics.yaml"
    if destination.exists():
        previous = yaml.safe_load(destination.read_text(encoding="utf-8-sig"))
        previous["propagation"].setdefault("num_layers", min(2, len(previous["propagation"]["layers"])))
        if previous != source:
            raise ValueError(f"{destination} belongs to different optics; select a new output directory.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(source, allow_unicode=True, sort_keys=False), encoding="utf-8")


DEFAULT_WAVELENGTH = OPTICS["wavelength"]
DEFAULT_DMD_RESOLUTION = OPTICS["resolution"]
DEFAULT_DMD_PIXEL_PITCH = OPTICS["dmd_pixel_pitch"]
DEFAULT_PROPAGATION_DISTANCES = OPTICS["distances"]
SLM_PROFILES = OPTICS["profiles"]
DEFAULT_SLM_LAYER_PROFILES = OPTICS["layer_profiles"]
# Compatibility scalar; actual propagation uses the per-layer pitches.
DEFAULT_PIXEL_SIZE = SLM_PROFILES[DEFAULT_SLM_LAYER_PROFILES[1]]["effective_sampling_pitch"]


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
    profile = SLM_PROFILES[profile_name]
    shape = tuple(int(round(float(length) / profile["hardware_pixel_pitch"])) for length in aperture_hw)
    if any(n < 1 or n > full for n, full in zip(shape, profile["resolution"])):
        raise ValueError(f"Illuminated aperture does not fit SLM profile {profile_name!r}.")
    return shape

