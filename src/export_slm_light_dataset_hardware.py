"""Export a static optical-student + Light checkpoint for the hardware bench.

Unlike Teacher V2, the optical student does not predict a phase for every
scene.  Each physical SLM therefore receives one checkpoint-owned, static
phase image while the numbered DMD inputs advance through the dataset split.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.class_display import class_name_for_id
from models.SLM.config_slm import ConfigSLM as Config
from models.SLM.dataset_slm import SLMFeatureDataset, slm_collate_fn
from models.SLM.optical_layers import OpticalStudent
from models.teacher_guidance import enhance_feature_for_display
from models.runtime import get_dataloader_kwargs
from models.yolov8.detection_protocol import decode_detections
from models.yolov8.feature_adapter import prepare_slm_detector_feature
from models.yolov8.head_v8 import build_detector_head


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export numbered DMD inputs, one static phase image per SLM, and "
            "simulated Light detections for a labelled dataset split."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Paired optical-student + Light checkpoint (usually detector_best.pth).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/military/data.yaml"),
        help="Dataset YAML file (default: data/military/data.yaml).",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Dataset split whose DMD inputs are exported (default: test).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Root directory for the hardware export.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda, cuda:0, or cpu (default: optical-student configuration).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Simulation batch size used while exporting previews (default: 2).",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="Confidence threshold for exported simulated detections.",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=None,
        help="NMS IoU threshold for exported simulated detections.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only export the first N images, useful for a hardware trial.",
    )
    parser.add_argument(
        "--lut",
        type=Path,
        default=None,
        help="Optional shared gray-to-phase LUT (.npy, .csv, or .txt).",
    )
    gray_group = parser.add_mutually_exclusive_group()
    gray_group.add_argument(
        "--gray-inverted",
        dest="gray_inverted",
        action="store_true",
        help="Force inverted SLM gray drives.",
    )
    gray_group.add_argument(
        "--no-gray-inverted",
        dest="gray_inverted",
        action="store_false",
        help="Force non-inverted SLM gray drives.",
    )
    parser.set_defaults(gray_inverted=None)
    parser.add_argument(
        "--phase-levels",
        type=int,
        default=None,
        help="Number of available SLM drive levels, 2 to 256 (default: checkpoint configuration).",
    )
    parser.add_argument(
        "--export-phase-offset-rad",
        type=float,
        default=None,
        help="Override the hardware phase offset before wrapping to [0, 2pi).",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if args.phase_levels is not None and not 2 <= args.phase_levels <= 256:
        raise ValueError("--phase-levels must be between 2 and 256.")
    for name in ("conf_threshold", "nms_threshold"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")


def load_checkpoint(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Expected a checkpoint dictionary, got {type(checkpoint).__name__}.")
    for key in ("student_state_dict", "detector_state_dict"):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint is missing {key!r}: {path}")
    if int(checkpoint.get("num_heads", 1)) != 1:
        raise ValueError(
            "Hardware export requires a single-head OpticalStudent checkpoint. "
            "A multi-head checkpoint contains several simultaneous phase paths "
            "that one physical SLM cascade cannot display."
        )
    checkpoint_layers = int(checkpoint.get("num_layers", Config.NUM_LAYERS))
    if checkpoint_layers != int(Config.NUM_LAYERS):
        raise ValueError(
            f"Checkpoint contains {checkpoint_layers} SLM layers, but the current "
            f"hardware configuration contains {Config.NUM_LAYERS}."
        )
    checkpoint_head = str(checkpoint.get("head_type", "light")).strip().lower()
    if checkpoint_head not in {"light", "yolo_light"}:
        raise ValueError(
            f"Checkpoint head_type is {checkpoint_head!r}; this exporter requires the Light head."
        )
    return checkpoint


def normalize_state_dict(state_dict: dict, prefix: str) -> dict:
    normalized = {}
    for raw_key, value in state_dict.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if key.startswith(prefix):
            key = key[len(prefix):]
        normalized[key] = value
    return normalized


def restore_checkpoint(
    student: torch.nn.Module,
    detector: torch.nn.Module,
    checkpoint: dict,
) -> dict:
    """Strictly restore learned tensors while keeping the selected hardware LUT."""
    student_state = normalize_state_dict(checkpoint["student_state_dict"], "student.")
    # LUT buffers describe the current physical calibration and may intentionally
    # differ from the buffers stored during training.
    lut_keys = {key for key in student.state_dict() if key.endswith(("lut_gray", "lut_phase"))}
    for key in lut_keys:
        student_state.pop(key, None)
    incompatible = student.load_state_dict(student_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys if key not in lut_keys]
    if missing or unexpected:
        raise RuntimeError(
            "Student checkpoint is incompatible with the current optical configuration: "
            f"missing={missing}, unexpected={unexpected}."
        )

    detector_state = normalize_state_dict(checkpoint["detector_state_dict"], "detector.")
    detector.load_state_dict(detector_state, strict=True)

    if "student_enable_norm" in checkpoint:
        student.enable_norm = bool(checkpoint["student_enable_norm"])
    if "student_norm_mode" in checkpoint:
        student.config.STUDENT_NORM_MODE = str(checkpoint["student_norm_mode"])
    if "student_norm_schedule" in checkpoint:
        student.config.STUDENT_NORM_SCHEDULE = str(checkpoint["student_norm_schedule"])
    return {
        "student_tensors": len(student_state),
        "detector_tensors": len(detector_state),
        "calibration_buffers_from_current_config": sorted(lut_keys),
    }


def build_output_directories(root: Path, layer_count: int) -> dict[str, Path]:
    directories = {
        "input": root / "input",
        "raw_optical_intensity": root / "raw_optical_intensity",
        "detector_feature": root / "detector_feature",
        "detection": root / "detection",
        "json": root / "json",
    }
    directories.update({f"slm{index}": root / f"slm{index}" for index in range(1, layer_count + 1)})
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def tensor_to_u8_image(values: torch.Tensor, enhance: bool = False) -> Image.Image:
    array = values.detach().float().cpu().numpy()
    if enhance:
        array = enhance_feature_for_display(array)
    else:
        array = np.clip(array, 0.0, 1.0)
    return Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="L")


def export_static_phase_images(
    student: OpticalStudent,
    directories: dict[str, Path],
) -> list[dict]:
    phase_records = []
    for layer_index, (layer_name, slm) in enumerate(student.all_slm_layers(), start=1):
        gray = slm.phase_to_gray().detach().clamp(0.0, 1.0)
        level_count = int(slm.phase_levels)
        quantized = torch.round(gray * (level_count - 1)) / (level_count - 1)
        gray_u8 = torch.round(quantized * 255.0).to(torch.uint8)[0, 0].cpu().numpy()
        phase = slm.hardware_export_phase().detach()[0, 0].float().cpu().numpy()
        phase_path = directories[layer_name] / "phase.png"
        Image.fromarray(gray_u8, mode="L").save(phase_path)
        phase_records.append(
            {
                "layer": layer_index,
                "name": layer_name,
                "profile": Config.slm_profile_name(layer_index),
                "hardware_pixel_pitch_m": Config.hardware_pixel_pitch(layer_index),
                "effective_sampling_pitch_m": Config.sampling_pitch(layer_index),
                "active_shape_hw": list(Config.slm_active_shape(layer_index)),
                "gray_drive_png": str(phase_path.relative_to(directories["input"].parent)).replace("\\", "/"),
                "phase_levels": level_count,
                "gray_inverted": bool(slm.gray_inverted),
                "hardware_phase_min_rad": float(phase.min()),
                "hardware_phase_max_rad": float(phase.max()),
            }
        )
    return phase_records


def detections_to_records(detections: np.ndarray) -> list[dict]:
    records = []
    for cx, cy, width, height, confidence, class_id_value in detections:
        class_id = int(class_id_value)
        records.append(
            {
                "class_id": class_id,
                "class_name": str(Config.CLASS_NAMES[class_id]),
                "confidence": float(confidence),
                "xywh_pixels": [float(cx), float(cy), float(width), float(height)],
            }
        )
    return records


def ground_truth_to_records(targets: torch.Tensor, image_size_hw: tuple[int, int]) -> list[dict]:
    image_h, image_w = image_size_hw
    records = []
    for class_id_value, cx, cy, width, height in targets.tolist():
        class_id = int(class_id_value)
        records.append(
            {
                "class_id": class_id,
                "class_name": str(Config.CLASS_NAMES[class_id]),
                "xywh_pixels": [
                    float(cx * image_w),
                    float(cy * image_h),
                    float(width * image_w),
                    float(height * image_h),
                ],
            }
        )
    return records


def draw_detection_boxes(
    input_image: Image.Image,
    detections: np.ndarray,
    ground_truth: torch.Tensor,
) -> Image.Image:
    canvas = input_image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    image_w, image_h = canvas.size

    for class_id_value, cx, cy, width, height in ground_truth.tolist():
        class_id = int(class_id_value)
        x1, y1 = (cx - width / 2.0) * image_w, (cy - height / 2.0) * image_h
        x2, y2 = (cx + width / 2.0) * image_w, (cy + height / 2.0) * image_h
        label = f"GT {class_name_for_id(Config.CLASS_NAMES, class_id)}"
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
        draw.text((x1, max(0.0, y1 - 14.0)), label, fill="lime", stroke_width=1, stroke_fill="black")

    for cx, cy, width, height, confidence, class_id_value in detections:
        class_id = int(class_id_value)
        x1, y1 = cx - width / 2.0, cy - height / 2.0
        x2, y2 = cx + width / 2.0, cy + height / 2.0
        label = f"{class_name_for_id(Config.CLASS_NAMES, class_id)} {confidence:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, max(0.0, y1 - 14.0)), label, fill="red", stroke_width=1, stroke_fill="black")
    return canvas


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.checkpoint = resolve_project_path(args.checkpoint)
    args.data = resolve_project_path(args.data)
    args.output = resolve_project_path(args.output)
    if args.lut is not None:
        args.lut = resolve_project_path(args.lut)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if args.lut is not None and not args.lut.is_file():
        raise FileNotFoundError(f"LUT not found: {args.lut}")

    Config.YAML_PATH = str(args.data)
    if args.lut is not None:
        Config.SLM_GRAY_TO_PHASE_LUT = str(args.lut)
    if args.gray_inverted is not None:
        Config.SLM_GRAY_INVERTED = bool(args.gray_inverted)
    if args.phase_levels is not None:
        Config.SLM_PHASE_LEVELS = int(args.phase_levels)
    if args.export_phase_offset_rad is not None:
        Config.SLM_EXPORT_PHASE_OFFSET_RAD = float(args.export_phase_offset_rad)
    Config.initialize()

    checkpoint = load_checkpoint(args.checkpoint)
    device = torch.device(args.device or Config.DEVICE)
    student = OpticalStudent(Config).to(device).eval()
    detector = build_detector_head(Config, in_channels=1).to(device).eval()
    restore_info = restore_checkpoint(student, detector, checkpoint)

    dataset = SLMFeatureDataset(Config, split=args.split)
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset split {args.split!r} is empty.")
    selected_count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    selected_dataset = dataset if selected_count == len(dataset) else Subset(dataset, range(selected_count))
    dataloader = DataLoader(
        selected_dataset,
        batch_size=args.batch_size,
        collate_fn=slm_collate_fn,
        **get_dataloader_kwargs(Config),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    directories = build_output_directories(args.output, int(Config.NUM_LAYERS))
    phase_records = export_static_phase_images(student, directories)

    conf_threshold = float(Config.CONF_THRESH if args.conf_threshold is None else args.conf_threshold)
    nms_threshold = float(Config.NMS_THRESH if args.nms_threshold is None else args.nms_threshold)
    amp_enabled = bool(getattr(Config, "ENABLE_AMP", True)) and device.type == "cuda"
    amp_dtype_name = str(getattr(Config, "AMP_DTYPE", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16

    samples = []
    next_id = 1
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=f"Exporting static SLM + light {args.split}"):
            gray = batch["gray_tensor"].to(device, non_blocking=Config.PIN_MEMORY)
            if Config.ENABLE_CHANNELS_LAST and device.type == "cuda":
                gray = gray.contiguous(memory_format=torch.channels_last)
            amp_context = (
                torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                feature, optical = student.forward_with_optical_field(gray)
                detector_feature = prepare_slm_detector_feature(Config, feature)
                predictions = detector(detector_feature)
            batch_detections = decode_detections(
                Config,
                predictions,
                conf_thresh=conf_threshold,
                nms_thresh=nms_threshold,
                max_det=Config.MAX_DET,
            )

            for batch_index, (targets, source_path, detections) in enumerate(
                zip(batch["targets"], batch["image_paths"], batch_detections)
            ):
                file_id = f"{next_id:04d}"
                next_id += 1
                input_image = tensor_to_u8_image(batch["gray_tensor"][batch_index, 0])
                raw_image = tensor_to_u8_image(optical["raw_intensity"][batch_index, 0], enhance=True)
                feature_image = tensor_to_u8_image(detector_feature[batch_index, 0], enhance=True)
                detections_array = np.asarray(detections, dtype=np.float32).reshape(-1, 6)

                input_path = directories["input"] / f"{file_id}.png"
                raw_path = directories["raw_optical_intensity"] / f"{file_id}.png"
                feature_path = directories["detector_feature"] / f"{file_id}.png"
                detection_path = directories["detection"] / f"{file_id}.png"
                json_path = directories["json"] / f"{file_id}.json"
                input_image.save(input_path)
                raw_image.save(raw_path)
                feature_image.save(feature_path)
                draw_detection_boxes(input_image, detections_array, targets).save(detection_path)

                record = {
                    "id": file_id,
                    "source_image": str(Path(source_path).resolve()),
                    "input_dmd": str(input_path.relative_to(args.output)).replace("\\", "/"),
                    "raw_optical_intensity_preview": str(raw_path.relative_to(args.output)).replace("\\", "/"),
                    "detector_feature_preview": str(feature_path.relative_to(args.output)).replace("\\", "/"),
                    "detection_visualization": str(detection_path.relative_to(args.output)).replace("\\", "/"),
                    "detections": detections_to_records(detections_array),
                    "ground_truth": ground_truth_to_records(targets, tuple(Config.RESOLUTION)),
                }
                json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                samples.append(record)

    manifest = {
        "export_type": "static_optical_student_plus_light",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_map50": checkpoint.get("val_map50"),
        "checkpoint_restore": restore_info,
        "dataset_yaml": str(args.data),
        "split": args.split,
        "exported_images": len(samples),
        "class_names": {str(key): str(value) for key, value in Config.CLASS_NAMES.items()},
        "simulation_resolution_hw": list(Config.RESOLUTION),
        "dmd_pixel_pitch_m": float(Config.DMD_PIXEL_PITCH),
        "dmd_aperture_m": list(Config.dmd_aperture()),
        "input_intensity_mode": str(Config.INPUT_INTENSITY_MODE),
        "student_enable_norm": bool(student.enable_norm),
        "student_norm_mode": str(Config.STUDENT_NORM_MODE),
        "student_output_blur_kernel": int(Config.STUDENT_OUTPUT_BLUR_KERNEL),
        "detector_feature_inverted": bool(Config.SLM_DETECTOR_INVERT_FEATURE),
        "confidence_threshold": conf_threshold,
        "nms_iou_threshold": nms_threshold,
        "phase_parameterization": str(Config.SLM_PHASE_PARAM_MODE),
        "phase_export_offset_rad": float(Config.SLM_EXPORT_PHASE_OFFSET_RAD),
        "gray_to_phase_lut": str(Config.SLM_GRAY_TO_PHASE_LUT or "ideal_linear"),
        "static_slm_phases": phase_records,
        "phase_usage": (
            "Each SLM directory contains exactly one shared phase.png. Keep both static phase images "
            "displayed while advancing all numbered files in input/."
        ),
        "preview_note": (
            "raw_optical_intensity/ and detector_feature/ are contrast-enhanced PNG previews; "
            "they are not quantitative camera calibration files."
        ),
        "numbering": "The same 0001-based ID identifies input, previews, detection, and json files.",
        "samples": samples,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Exported {len(samples)} {args.split} DMD inputs to: {args.output}")
    for phase_record in phase_records:
        print(
            f"{phase_record['name'].upper()} static phase: "
            f"{phase_record['gray_drive_png']} ({phase_record['active_shape_hw'][1]}x"
            f"{phase_record['active_shape_hw'][0]})"
        )
    print("Each SLM has one shared phase.png; it is not repeated for every DMD input.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
