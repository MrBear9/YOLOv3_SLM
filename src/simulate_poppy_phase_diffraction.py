"""532 nm plane-wave diffraction: POPPY Fresnel versus exact scalar ASM.

Run from the project root, for example::

    python src/simulate_poppy_phase_diffraction.py --phase-dir output/A6000/SLM_layer_adaptive_scratch_2gpu_seed42/slm

Optional lens focal plane and axial scan (distances in mm)::

    python src/simulate_poppy_phase_diffraction.py --phase-dir PATH --focal-length-mm 100 --scan-mm 10 200 21

Dependencies: python -m pip install poppy astropy numpy matplotlib
The default three-method comparison also uses the project's existing torch installation.

Input: 2-D finite phase arrays in RADIANS, not grayscale drive values. The
Supported filenames are slm{1,2}_hardware_export_phase.npy and calibration
slm{1,2}_phase_rad.npy (native hardware pixels, before grayscale quantization).
For calibration exports, the nearest config.json within two parent directories
provides physical pitches, wavelength and distances unless overridden on CLI.
Physical pitches
come from phase_export_metadata.json or explicit --pitch{1,2}-um. Without
metadata, defaults are the project's 8 and 4.5 um hardware pitches. For raw
or simulation arrays, explicitly provide pitches appropriate to those arrays.

Single-SLM cases REMOVE the other SLM (including its aperture). All cases
use the same incident irradiance and end at the same bench camera plane:
SLM1-only propagates z12+z23; SLM2-only receives a fresh collimated beam and
propagates z23; dual propagates z12, applies SLM2, then propagates z23.
The optional lens is placed immediately after the last active SLM.

This is ideal scalar, paraxial Fresnel propagation, not a reproduction of
training's ASM/resampling/normalization. It excludes DMD image modulation,
SLM calibration errors, polarization, relay optics and oblique incidence.
Without a lens, outputs are diffraction intensities, not automatically PSFs.
Increase --oversample and reduce --grid-pitch-um to check convergence.
Documentation: https://poppy-optics.readthedocs.io/en/latest/fresnel.html

By default also computes independent NumPy angular-spectrum propagation:
H = exp(i*2*pi*z/lambda*(sqrt(1-lambda**2*(fx**2+fy**2))-1)).
The constant carrier phase is removed; evanescent terms decay for positive z.
ASM and POPPY use identical input fields, finite SLM apertures, grid and padding.
No additional frequency band-limit is applied. Both FFT methods require padding
and sampling convergence checks; this is not the training ASM implementation.
camera_poppy_vs_asm.png/.npz and JSON errors compare the same camera plane.
Optional axial scans include both methods. Optional lens plots remain POPPY-only
(the lens changes sampling); they are not included in the ASM comparison.
Use --skip-asm to retain the original POPPY-only workflow.

Also saves camera_three_methods.png, camera_three_methods_difference.png and
camera_three_methods.npz, directly calling models.SLM.asm_propagation.ASMPropagation.
This additional controlled comparison applies the project band mask to all three
methods at each segment, uses the same external padding, and center-crops all
outputs only at the final camera. The project module runs with linear_conv=False
to avoid double padding. Its original frequency grid and complex64 arithmetic
are retained. This tests propagation implementations on common physical sampling,
not the full training model with its effective pitches and per-layer cropping.
Lens and axial-scan outputs do not include the project module. The original
two-method outputs remain unbandlimited; do not mix their metrics with the
controlled three-method metrics. --skip-project-comparison disables the latter.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import astropy.units as u
    import poppy
except ImportError as exc:
    raise SystemExit("Install dependencies: python -m pip install poppy astropy numpy matplotlib") from exc


def positive(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be a finite positive number.")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase-dir", type=Path, help="Directory with hardware phase NPYs and optional export metadata.")
    parser.add_argument("--phase1", type=Path)
    parser.add_argument("--phase2", type=Path)
    parser.add_argument("--pitch1-um", type=positive)
    parser.add_argument("--pitch2-um", type=positive)
    parser.add_argument("--wavelength-nm", type=positive, help="Calibration config value or 532 nm by default.")
    parser.add_argument("--z12-mm", type=positive, help="Calibration config value or 200 mm by default.")
    parser.add_argument("--z23-mm", type=positive, help="Calibration config value or 100 mm by default.")
    parser.add_argument("--grid-pitch-um", type=positive, help="Common computational pitch; default min(hardware pitches)/2.")
    parser.add_argument("--oversample", type=int, default=2, help="FFT padding factor, >=2 (default 2).")
    parser.add_argument("--focal-length-mm", type=positive, help="Optional ideal lens after last active SLM; save focal-plane intensities.")
    parser.add_argument("--scan-mm", nargs=3, type=float, metavar=("START", "STOP", "COUNT"), help="Optional axial scan measured from last active SLM; no lens.")
    parser.add_argument("--output", type=Path, default=Path("output/poppy_phase_diffraction"))
    parser.add_argument("--skip-asm", action="store_true", help="Disable the default camera and axial-scan ASM comparison.")
    parser.add_argument("--skip-project-comparison", action="store_true",
                        help="Skip the additional controlled three-method camera comparison.")
    args = parser.parse_args()
    if args.oversample < 2:
        parser.error("--oversample must be >=2.")
    if args.scan_mm:
        start, stop, count = args.scan_mm
        if not all(np.isfinite(args.scan_mm)) or not 0 < start < stop or count < 2 or count != int(count):
            parser.error("--scan-mm needs 0 < START < STOP and an integer COUNT >=2.")
    if not args.phase_dir and not (args.phase1 and args.phase2):
        parser.error("Provide --phase-dir or both --phase1 and --phase2.")
    return args


def load_phase(path):
    phase = np.load(path, allow_pickle=False)
    if phase.ndim != 2 or min(phase.shape) < 2 or not np.isrealobj(phase) or not np.isfinite(phase).all():
        raise ValueError(f"{path}: expected a finite real 2-D phase array in radians.")
    if not np.issubdtype(phase.dtype, np.floating):
        raise ValueError(f"{path}: integer arrays may be grayscale drive values; supply floating-point radian phases.")
    return phase.astype(np.float64)


def apply_phase(wave, phase, pitch, wavelength):
    """Pixel-held phase mask with finite rectangular aperture on physical coordinates.

    Sample nearest hardware pixel (no interpolation across a wrapped-phase seam).
    POPPY accepts optical path difference in metres: OPD = phase*lambda/(2*pi).
    """
    y, x = wave.coordinates()
    h, w = phase.shape
    row = np.floor(y / pitch + h / 2).astype(int)
    col = np.floor(x / pitch + w / 2).astype(int)
    inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    opd = np.zeros(wave.shape, dtype=float)
    opd[inside] = phase[row[inside], col[inside]] * wavelength / (2 * np.pi)
    wave *= poppy.ArrayOpticalElement(
        transmission=inside.astype(float), opd=opd,
        pixelscale=wave.pixelscale, name="Finite SLM phase mask",
    )
    return wave


def intensity_data(wave):
    pitch = wave.pixelscale.to_value(u.m / u.pixel)
    # POPPY preserves summed pixel power. A lens can change pixel area, so
    # convert to irradiance relative to the original unit-amplitude plane wave.
    intensity = np.asarray(wave.intensity, dtype=float) * (wave.incident_pitch_m / pitch)**2
    y, x = wave.coordinates()
    extent = [x[0, 0]-pitch/2, x[0, -1]+pitch/2, y[0, 0]-pitch/2, y[-1, 0]+pitch/2]
    return intensity, np.asarray(extent) * 1000, pitch


def propagate_asm(field, pitch, wavelength, distance):
    """Forward exact scalar angular spectrum on a fixed padded grid.

    Drop only the uniform propagating carrier phase. Use a stable expression
    for sqrt(1-q)-1 near q=0 to avoid subtractive cancellation.
    """
    if distance < 0 or pitch <= 0 or wavelength <= 0:
        raise ValueError("ASM needs z >= 0, pitch > 0 and wavelength > 0.")
    if distance == 0:
        return np.array(field, dtype=complex, copy=True)
    fy = np.fft.fftfreq(field.shape[0], d=pitch)
    fx = np.fft.fftfreq(field.shape[1], d=pitch)
    q = wavelength**2 * (fy[:, None]**2 + fx[None, :]**2)
    root = np.sqrt((1-q).astype(complex))
    relative_kz = -q/(root+1)
    transfer = np.exp(2j*np.pi*distance/wavelength * relative_kz)
    return np.fft.ifft2(np.fft.fft2(field) * transfer)


def compare_asm_camera(cameras, asm_bases, pitch, wavelength, z12, z23, output):
    """Compare raw irradiance, with one shared colour scale and no per-image norm."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    arrays, stats, panels = {}, {}, []
    for col, (label, wave) in enumerate(cameras.items()):
        reference, extent, output_pitch = intensity_data(wave)
        if reference.shape != asm_bases[label].shape or not np.isclose(output_pitch, pitch, rtol=1e-9, atol=0):
            plt.close(fig)
            raise ValueError("POPPY changed output sampling; fixed-grid ASM comparison requires the same grid. "
                             "Use shorter distances or --skip-asm for this geometry.")
        z = z23 + (z12 if label == "SLM1_only" else 0)
        field = propagate_asm(asm_bases[label], pitch, wavelength, z)
        intensity = np.abs(field)**2
        difference = intensity-reference
        energy = max(float(np.sum(reference**2)), 1e-300)
        power = max(float(reference.sum()), 1e-300)
        stats[label] = {
            "relative_intensity_l2_error": float(np.sqrt(np.sum(difference**2)/energy)),
            "mean_absolute_intensity_error": float(np.mean(np.abs(difference))),
            "max_absolute_intensity_error": float(np.max(np.abs(difference))),
            "asm_peak_irradiance": float(intensity.max()),
            "asm_integrated_relative_power_m2": float(intensity.sum()*pitch**2),
            "relative_integrated_power_difference": float((intensity.sum()-reference.sum())/power),
        }
        print(f"{label} ASM vs POPPY: relative intensity L2 = {stats[label]['relative_intensity_l2_error']:.4%}")
        for row, (method, values) in enumerate((("POPPY Fresnel", reference), ("ASM", intensity),
                                               ("Absolute difference", np.abs(difference)))):
            im = axes[row, col].imshow(values, origin="lower", extent=extent,
                                       cmap="magma" if row == 2 else "viridis", vmin=0)
            axes[row, col].set(title=f"{label.replace('_', ' ')} | {method}", xlabel="x (mm)", ylabel="y (mm)")
            panels.append((row, im, float(values.max())))
        arrays[label+"_poppy"] = reference.astype(np.float32)
        arrays[label+"_asm"] = intensity.astype(np.float32)
        arrays[label+"_asm_minus_poppy"] = difference.astype(np.float32)
        arrays[label+"_extent_mm"] = extent
    maximum = max(v for row, _, v in panels if row < 2)
    error_maximum = max(v for row, _, v in panels if row == 2)
    for row, im, _ in panels:
        im.set_clim(0, max(error_maximum if row == 2 else maximum, 1e-15))
    fig.colorbar(axes[0, 0].images[0], ax=axes[:2, :], label="I / incident irradiance (shared scale)", shrink=.7)
    fig.colorbar(axes[2, 0].images[0], ax=axes[2, :], label="|I_ASM - I_POPPY| (separate scale)", shrink=.8)
    fig.savefig(output / "camera_poppy_vs_asm.png", dpi=250)
    plt.close(fig)
    arrays["pitch_m"] = np.asarray(pitch)
    np.savez_compressed(output / "camera_poppy_vs_asm.npz", **arrays)
    return stats


def save_comparison(waves, output, name, view_half_width_mm=None):
    data = [intensity_data(wave) for wave in waves.values()]
    maximum = max(float(item[0].max()) for item in data)
    arrays = {}
    stats = {}
    fig, axes = plt.subplots(2, len(data), figsize=(12, 7), constrained_layout=True)
    for col, ((label, wave), (intensity, extent, pitch)) in enumerate(zip(waves.items(), data)):
        arrays[label] = intensity.astype(np.float32)
        arrays[label + "_extent_mm"] = extent
        arrays[label + "_pitch_m"] = np.asarray(pitch)
        peak = np.unravel_index(np.argmax(intensity), intensity.shape)
        y, x = wave.coordinates()
        # Integral uses incident amplitude=1, so this is a relative power in m^2.
        border = np.zeros(intensity.shape, dtype=bool)
        width = max(1, min(intensity.shape)//20)
        border[:width] = border[-width:] = True
        border[:, :width] = border[:, -width:] = True
        total = float(intensity.sum())
        stats[label] = {"peak_irradiance_relative_to_incident": float(intensity.max()),
                        "integrated_relative_power_m2": total*pitch**2,
                        "peak_xy_mm": [float(x[peak]*1000), float(y[peak]*1000)],
                        "outer_5percent_power_fraction": float(intensity[border].sum()/max(total, 1e-300))}
        for row in range(2):
            kwargs = {"vmin": 0, "vmax": maximum} if row == 0 else {"norm": LogNorm(vmin=1e-5, vmax=1)}
            displayed = intensity if row == 0 else np.maximum(intensity/max(maximum, 1e-300), 1e-5)
            im = axes[row, col].imshow(displayed, origin="lower", extent=extent, cmap="viridis", **kwargs)
            axes[row, col].set(title=label.replace("_", " "), xlabel="x (mm)", ylabel="y (mm)")
            if view_half_width_mm is not None:
                axes[row, col].set_xlim(-view_half_width_mm, view_half_width_mm)
                axes[row, col].set_ylim(-view_half_width_mm, view_half_width_mm)
        print(f"{label}: peak={intensity.max():.5g}; edge power={stats[label]['outer_5percent_power_fraction']:.3%}")
    fig.colorbar(axes[0, 0].images[0], ax=axes[0, :], label="I / incident irradiance (shared scale)", shrink=.8)
    fig.colorbar(im, ax=axes[1, :], label="I / global peak (shared log scale)", shrink=.8)
    fig.savefig(output / f"{name}.png", dpi=250)
    plt.close(fig)
    np.savez_compressed(output / f"{name}.npz", **arrays)
    return stats


def compare_project_camera(first, second, pitch, wavelength, z12, z23, crop_size, output):
    """Controlled comparison calling the project's actual ASMPropagation class.

    All methods start with identical sampled fields, use the already-padded grid,
    share the exact binary band mask from the project's H, and crop only at the
    final camera. The project uses linear_conv=False to avoid double padding.
    Its native frequency grid and float32 transfer phase are intentionally kept:
    differences against NumPy include these implementation choices.
    This isolates propagators; it does NOT reproduce the full training pipeline
    (effective sampling pitches, per-layer crop, image illumination, etc.).
    """
    import torch
    from models.SLM.asm_propagation import ASMPropagation

    shape = first.shape
    methods = ("POPPY Fresnel", "NumPy ASM", "Project ASM")
    cases = ("SLM1_only", "SLM2_only", "Dual_SLM")
    start = (shape[0]-crop_size)//2
    crop = (slice(start, start+crop_size), slice(start, start+crop_size))
    y, x = first.coordinates()
    extent = np.array([x[start, start]-pitch/2, x[start, start+crop_size-1]+pitch/2,
                       y[start, start]-pitch/2, y[start+crop_size-1, start]+pitch/2])*1000
    stats, arrays, masks = {}, {}, {}
    figures = [plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True) for _ in range(2)]
    intensity_panels, difference_panels = [], []

    def propagate_all(fields, distance):
        # CPU avoids tying diagnostics to the training device; the original
        # project's complex64 transfer function and forward pass are unchanged.
        module = ASMPropagation(SimpleNamespace(), distance, wavelength=wavelength,
                                pixel_size=pitch, resolution=shape, linear_conv=False)
        mask = module.H[0, 0].abs().numpy() > 0
        masks[str(distance)] = {"retained_frequency_fraction": float(mask.mean())}
        result = []
        for index, field in enumerate(fields):
            if index == 0:
                wave = copy.deepcopy(first)
                wave.wavefront = field.copy()
                wave.propagate_fresnel(distance*u.m)
                if wave.shape != shape or not np.isclose(wave.pixelscale.to_value(u.m/u.pixel), pitch, rtol=1e-9, atol=0):
                    raise ValueError("Controlled comparison requires fixed POPPY sampling; shorten distances or use --skip-project-comparison.")
                propagated = wave.wavefront
            elif index == 1:
                propagated = propagate_asm(field, pitch, wavelength, distance)
            else:
                with torch.no_grad():
                    propagated = module(torch.as_tensor(field, dtype=torch.complex64)[None, None])[0, 0].numpy()
            if index < 2:
                propagated = np.fft.ifft2(np.fft.fft2(propagated)*mask)
            result.append(propagated)
        return result

    for col, case in enumerate(cases):
        entrance = second.wavefront if case == "SLM2_only" else first.wavefront
        fields = [entrance]*3
        if case == "Dual_SLM":
            fields = propagate_all(fields, z12)
            fields = [field*second.wavefront for field in fields]
        fields = propagate_all(fields, z23 + (z12 if case == "SLM1_only" else 0))
        intensities = [np.abs(field[crop])**2 for field in fields]
        del fields
        stats[case] = {"methods": {}, "pairwise": {}}
        for row, (method, intensity) in enumerate(zip(methods, intensities)):
            key = ("poppy", "numpy_asm", "project_asm")[row]
            arrays[case+"_"+key] = intensity.astype(np.float32)
            stats[case]["methods"][key] = {"peak": float(intensity.max()),
                                          "cropped_power_m2": float(intensity.sum()*pitch**2)}
            ax = figures[0][1][row, col]
            im = ax.imshow(intensity, origin="lower", extent=extent, cmap="viridis", vmin=0)
            ax.set(title=f"{case.replace('_',' ')} | {method}", xlabel="x (mm)", ylabel="y (mm)")
            intensity_panels.append((im, float(intensity.max())))
        for row, (a, b, name) in enumerate(((0, 1, "NumPy - POPPY"), (0, 2, "Project - POPPY"),
                                           (1, 2, "Project - NumPy"))):
            difference = intensities[b]-intensities[a]
            relative = float(np.linalg.norm(difference)/max(np.linalg.norm(intensities[a]), 1e-300))
            stats[case]["pairwise"][name] = {"relative_intensity_l2": relative,
                                              "mae": float(np.mean(np.abs(difference)))}
            ax = figures[1][1][row, col]
            im = ax.imshow(np.abs(difference), origin="lower", extent=extent, cmap="magma", vmin=0)
            ax.set(title=f"{case.replace('_',' ')} | {name}\nRelative L2: {relative:.2%}", xlabel="x (mm)", ylabel="y (mm)")
            difference_panels.append((im, float(np.abs(difference).max())))
        print(f"{case} controlled comparison: {stats[case]['pairwise']}", flush=True)
    for (fig, axes), panels, filename, label in zip(
        figures, (intensity_panels, difference_panels),
        ("camera_three_methods", "camera_three_methods_difference"),
        ("I / incident irradiance", "Absolute intensity difference"),
    ):
        maximum = max(peak for _, peak in panels)
        for im, _ in panels:
            im.set_clim(0, max(maximum, 1e-15))
        fig.colorbar(panels[0][0], ax=axes, label=label+" (shared scale)", shrink=.75)
        fig.savefig(output / f"{filename}.png", dpi=250)
        plt.close(fig)
    np.savez_compressed(output / "camera_three_methods.npz", **arrays, extent_mm=extent, pitch_m=pitch)
    return {"implementation": "models.SLM.asm_propagation.ASMPropagation; CPU complex64; unmodified",
            "common_conditions": "Same incident fields, hardware pitch mapping, external padding, per-segment project band mask, final-only center crop",
            "project_linear_conv": False, "output_crop_shape": [crop_size, crop_size], "band_masks": masks,
            "scope": "Controlled propagation comparison, not full training pipeline reproduction; original project frequency grid and precision retained",
            "metrics": stats}


def resolve_phase_paths(args):
    paths = []
    for i in (1, 2):
        explicit = getattr(args, f"phase{i}")
        if explicit is not None:
            path = explicit
        else:
            candidates = [args.phase_dir / f"slm{i}_{suffix}.npy"
                          for suffix in ("hardware_export_phase", "phase_rad")]
            available = [p for p in candidates if p.is_file()]
            if not available:
                raise FileNotFoundError("No phase array found. Expected: " + ", ".join(map(str, candidates)))
            if len(available) > 1:
                raise ValueError(f"Multiple phase exports found; select one with --phase{i}: {available}")
            path = available[0]
        if not path.is_file():
            raise FileNotFoundError(f"Phase file not found: {path}")
        paths.append(path)
    return paths


def calibration_config(paths):
    configs = []
    for path in paths:
        found = None
        if path.name.endswith("_phase_rad.npy"):
            for directory in list(path.resolve().parents)[:3]:
                candidate = directory / "config.json"
                if candidate.is_file():
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                    if "slm1_pitch_um" in data and "slm2_pitch_um" in data:
                        found = candidate
                        break
        configs.append(found)
    if configs[0] != configs[1]:
        raise ValueError("Phase files have different calibration configs; use a pair from the same calibration run.")
    if configs[0]:
        print(f"Calibration config: {configs[0]}")
        return json.loads(configs[0].read_text(encoding="utf-8"))
    return {}


def main():
    args = parse_args()
    metadata_path = args.phase_dir / "phase_export_metadata.json" if args.phase_dir else None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path and metadata_path.exists() else {}
    paths = resolve_phase_paths(args)
    calibration = calibration_config(paths)
    for attr, key, fallback in (("wavelength_nm", "wavelength_nm", 532.),
                                ("z12_mm", "slm1_to_slm2_mm", 200.),
                                ("z23_mm", "slm2_to_camera_mm", 100.)):
        if getattr(args, attr) is None:
            setattr(args, attr, positive(str(calibration.get(key, fallback))))
    print("Phase files: " + ", ".join(map(str, paths)))
    phases = [load_phase(path) for path in paths]
    pitches = []
    for i, path in enumerate(paths, 1):
        explicit = getattr(args, f"pitch{i}_um")
        if "hardware_export_phase" not in path.name and not path.name.endswith("_phase_rad.npy") and explicit is None:
            raise ValueError(f"For {path.name}, specify --pitch{i}-um explicitly; simulation and hardware grids differ.")
        layer = metadata.get("slm_layers", {}).get(f"slm{i}", {})
        default_pitch = float(calibration.get(f"slm{i}_pitch_um", (8., 4.5)[i-1]))*1e-6
        pitches.append(explicit*1e-6 if explicit else float(layer.get("hardware_pixel_pitch_m", default_pitch)))
    if not all(np.isfinite(p) and p > 0 for p in pitches):
        raise ValueError("Invalid physical pixel pitch in metadata.")
    wavelength = args.wavelength_nm * 1e-9
    grid_pitch = (args.grid_pitch_um * 1e-6) if args.grid_pitch_um else min(pitches)/2
    if grid_pitch > min(pitches):
        raise ValueError("Computational pitch must not exceed the smallest hardware pitch.")
    size = max(max(phase.shape)*pitch for phase, pitch in zip(phases, pitches))
    npix = int(np.ceil(size/grid_pitch))
    npix += npix % 2
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"POPPY {poppy.__version__}; hardware pitches: {np.array(pitches)*1e6} um")
    print(f"Computational array: {npix*args.oversample} squared, pitch {grid_pitch*1e6:g} um")

    def fresh():
        wave = poppy.FresnelWavefront(beam_radius=npix*grid_pitch/2*u.m,
                                     wavelength=wavelength*u.m, npix=npix, oversample=args.oversample)
        wave.incident_pitch_m = grid_pitch
        return wave

    # Store fields immediately after their last active phase mask.
    first = apply_phase(fresh(), phases[0], pitches[0], wavelength)
    second = apply_phase(fresh(), phases[1], pitches[1], wavelength)
    dual = copy.deepcopy(first)
    dual.propagate_fresnel(args.z12_mm*1e-3*u.m)
    apply_phase(dual, phases[1], pitches[1], wavelength)
    bases = {"SLM1_only": first, "SLM2_only": second, "Dual_SLM": dual}
    cameras = {}
    for label, base in bases.items():
        wave = copy.deepcopy(base)
        distance = args.z23_mm + (args.z12_mm if label == "SLM1_only" else 0)
        wave.propagate_fresnel(distance*1e-3*u.m)
        cameras[label] = wave
    summary = {"poppy_version": poppy.__version__, "phase_files": [str(p.resolve()) for p in paths],
               "phase_shapes": [list(p.shape) for p in phases], "hardware_pitch_m": pitches,
               "grid_pitch_m": grid_pitch, "npix_unpadded": npix, "oversample": args.oversample,
               "wavelength_nm": args.wavelength_nm, "z12_mm": args.z12_mm, "z23_mm": args.z23_mm,
               "single_case_convention": "Other SLM and aperture removed; same bench camera; fresh plane wave for SLM2-only",
               "camera": save_comparison(cameras, args.output, "camera_diffraction")}
    asm_bases = None
    if not args.skip_asm:
        # Start from the exact same sampled entrance fields; dual's intermediate
        # propagation is independently recomputed by ASM, never reused from POPPY.
        asm_dual = propagate_asm(first.wavefront, grid_pitch, wavelength, args.z12_mm*1e-3)
        asm_dual *= second.wavefront
        asm_bases = {"SLM1_only": first.wavefront, "SLM2_only": second.wavefront, "Dual_SLM": asm_dual}
        summary["asm_comparison"] = {
            "method": "Exact scalar angular spectrum; decaying evanescent waves; carrier phase removed",
            "bandlimit": "No additional bandlimit; identical spatial padding to POPPY",
            "scope": "Camera plane and optional free-space axial scan; lens focal plane excluded",
            "metrics": compare_asm_camera(cameras, asm_bases, grid_pitch, wavelength,
                                          args.z12_mm*1e-3, args.z23_mm*1e-3, args.output),
        }
    del cameras
    if not args.skip_project_comparison and not args.skip_asm:
        summary["controlled_three_method_comparison"] = compare_project_camera(
            first, second, grid_pitch, wavelength, args.z12_mm*1e-3,
            args.z23_mm*1e-3, npix, args.output)
    if args.focal_length_mm:
        focused = {}
        for label, base in bases.items():
            wave = copy.deepcopy(base)
            wave *= poppy.QuadraticLens(args.focal_length_mm*1e-3*u.m)
            wave.propagate_fresnel(args.focal_length_mm*1e-3*u.m)
            focused[label] = wave
        summary["lens_focal_length_mm"] = args.focal_length_mm
        # Show the central diffraction lobes rather than a nearly invisible point
        # on a millimetre-wide field; NPZ arrays still contain the entire plane.
        zoom_mm = 8*wavelength*(args.focal_length_mm*1e-3)/min(
            min(phase.shape)*pitch for phase, pitch in zip(phases, pitches)
        )*1000
        summary["lens_plot_half_width_mm"] = zoom_mm
        summary["lens_focal_plane"] = save_comparison(focused, args.output, "lens_focal_plane", zoom_mm)
        del focused
    if args.scan_mm:
        start, stop, count = args.scan_mm
        distances = np.linspace(start, stop, int(count))
        peaks, asm_peaks = [], []
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for label, base in bases.items():
            values = []
            for distance in distances:
                wave = copy.deepcopy(base)
                wave.propagate_fresnel(distance*1e-3*u.m)
                pitch = wave.pixelscale.to_value(u.m/u.pixel)
                values.append(float(wave.intensity.max()) * (grid_pitch/pitch)**2)
            peaks.append(values)
            line, = ax.plot(distances, values, label=label.replace("_", " ")+" POPPY")
            if asm_bases is not None:
                asm_values = [float(np.max(np.abs(propagate_asm(
                    asm_bases[label], grid_pitch, wavelength, distance*1e-3))**2)) for distance in distances]
                asm_peaks.append(asm_values)
                ax.plot(distances, asm_values, linestyle="--", color=line.get_color(),
                        label=label.replace("_", " ")+" ASM")
        ax.set(xlabel="Distance after last active SLM (mm)", ylabel="Peak I / incident irradiance",
               title="Axial peak scan (not a focus criterion)")
        ax.legend(fontsize=7, ncol=2)
        fig.savefig(args.output / "axial_scan.png", dpi=250)
        plt.close(fig)
        np.savez_compressed(args.output / "axial_scan.npz", distance_mm=distances, peaks=peaks,
                            asm_peaks=np.asarray(asm_peaks).reshape(-1, len(distances)), cases=list(bases))
        summary["scan_mm"] = args.scan_mm
    (args.output / "simulation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output.resolve()}")
    print("Check padding/sampling convergence before interpreting fine spots as physical foci.")


if __name__ == "__main__":
    main()
