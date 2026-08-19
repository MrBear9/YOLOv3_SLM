"""Export trained SLM phases with explicit simulation and hardware conventions."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from models.SLM.utils_slm import load_student_checkpoint


def export_student_phase_images(config, checkpoint_path, output_dir, device="cpu"):
    """Export raw, simulation, hardware phase arrays and native SLM PNGs per layer."""
    from models.SLM.multi_head_slm import MultiHeadOpticalStudent
    from models.SLM.optical_layers import OpticalStudent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    student_cls = MultiHeadOpticalStudent if bool(getattr(config, "SLM_MULTI_HEAD_ENABLED", False)) else OpticalStudent
    student = student_cls(config).to(device)
    info = load_student_checkpoint(student, str(checkpoint_path), torch.device(device))
    if info["loaded"] == 0:
        raise RuntimeError(f"No compatible SLM tensors found in {checkpoint_path!r}.")
    saved = []
    for layer_name, slm in student.all_slm_layers():
        raw = slm._raw_phase().detach().squeeze(0).squeeze(0).cpu().numpy()
        direct = slm.direct_phase()
        pyramid = slm.pyramid_phase()
        simulation = slm.simulation_phase().detach().squeeze(0).squeeze(0).cpu().numpy()
        hardware = slm.hardware_export_phase().detach().squeeze(0).squeeze(0).cpu().numpy()
        gray = slm.phase_to_gray_uint8().squeeze(0).squeeze(0).cpu().numpy()
        np.save(output_dir / f"{layer_name}_raw_phase.npy", raw)
        if direct is not None:
            np.save(output_dir / f"{layer_name}_direct_phase.npy", direct.detach().squeeze().cpu().numpy())
        if pyramid is not None:
            np.save(output_dir / f"{layer_name}_pyramid_phase.npy", pyramid.detach().squeeze().cpu().numpy())
        np.save(output_dir / f"{layer_name}_simulation_phase.npy", simulation)
        np.save(output_dir / f"{layer_name}_hardware_export_phase.npy", hardware)
        path = output_dir / f"{layer_name}.png"
        Image.fromarray(gray, mode="L").save(path)
        saved.append(path)
    metadata = {
        "phase_parameterization": str(getattr(config, "SLM_PHASE_PARAM_MODE", "direct_sgd")),
        "simulation_convention": "direct_sgd: exp(1j * raw_phase)",
        "hardware_export_convention": "hardware_phase=(raw_phase+offset) mod 2pi",
        "hardware_export_offset_rad": float(getattr(config, "SLM_EXPORT_PHASE_OFFSET_RAD", np.pi)),
        "gray_inverted": bool(getattr(config, "SLM_GRAY_INVERTED", False)),
        "phase_levels": int(getattr(config, "SLM_PHASE_LEVELS", 256)),
        "simulate_hardware_pixel_grid": bool(getattr(config, "SIMULATE_HARDWARE_PIXEL_GRID", False)),
        "direct_sgd_pyramid_scale": float(getattr(config, "SLM_DIRECT_SGD_PYRAMID_SCALE", 0.0)),
        "simulation_resolution_hw": [int(config.RESOLUTION[0]), int(config.RESOLUTION[1])],
        "dmd_pixel_pitch_m": float(getattr(config, "DMD_PIXEL_PITCH", 0.0)),
        "dmd_aperture_m": list(config.dmd_aperture()) if hasattr(config, "dmd_aperture") else None,
        "slm_layers": {
            f"slm{index}": {
                "profile": config.slm_profile_name(index),
                "hardware_pixel_pitch_m": config.hardware_pixel_pitch(index),
                "effective_sampling_pitch_m": config.sampling_pitch(index),
                "active_shape_hw": list(config.slm_active_shape(index)),
            }
            for index in range(1, int(getattr(config, "NUM_LAYERS", 2)) + 1)
        } if hasattr(config, "slm_profile_name") else {},
        "gray_to_phase_lut": str(getattr(config, "SLM_GRAY_TO_PHASE_LUT", "") or ""),
    }
    with (output_dir / "phase_export_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return saved, info
