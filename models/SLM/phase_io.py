"""Export trained SLM phases through the same calibrated drive chain used in training."""

from pathlib import Path

from PIL import Image
import torch

from models.SLM.utils_slm import load_student_checkpoint


def export_student_phase_images(config, checkpoint_path, output_dir, device="cpu"):
    """Load a student checkpoint and save one finite-level grayscale PNG per SLM layer."""
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
        gray = slm.phase_to_gray_uint8().squeeze(0).squeeze(0).cpu().numpy()
        path = output_dir / f"{layer_name}_gray.png"
        Image.fromarray(gray, mode="L").save(path)
        saved.append(path)
    return saved, info
