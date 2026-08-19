"""Export input-dependent Teacher V2 SLM drives for every image in val or test."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

# Allow direct execution with ``python src/export_teacher_v2_dataset_hardware.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dataset import YOLODataset
from models.SLM.slm_modulation import resample_phase_map
from models.teacher import build_teacher
from models.teacher_guidance import enhance_feature_for_display
from models.yolov8.config_v8 import ConfigYOLOv8Anchor as Config
from models.yolov8.detection_protocol import decode_detections
from models.yolov8.feature_adapter import prepare_detector_feature
from models.yolov8.head_v8 import build_detector_head
from src.predict_teacher_v2_single_image import (
    detections_to_records,
    load_checkpoint,
    load_phase_lut,
    phase_to_gray,
    prepare_scene,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Teacher V2 letterboxed inputs, SLM gray drives, and detections for a dataset split."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Teacher V2 + light detector checkpoint.")
    parser.add_argument("--data", type=Path, default=Path("data/military/data.yaml"), help="Dataset YAML file.")
    parser.add_argument("--split", choices=("val", "test"), default="test", help="Dataset split to export.")
    parser.add_argument("--output", type=Path, required=True, help="Root directory for numbered hardware export files.")
    parser.add_argument("--device", default="cpu", help="cuda, cuda:0, or cpu (default: cpu for local hardware export).")
    parser.add_argument("--conf-threshold", type=float, default=None, help="Detection confidence threshold.")
    parser.add_argument("--nms-threshold", type=float, default=None, help="NMS IoU threshold.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of images to export, useful for a hardware trial.")
    parser.add_argument("--lut", type=Path, default=None, help="Optional calibrated gray-to-phase LUT (.npy, .csv, or .txt).")
    parser.add_argument("--gray-inverted", action="store_true", help="Invert SLM gray drive after LUT conversion.")
    parser.add_argument("--phase-levels", type=int, default=256, help="Number of SLM drive levels (2 to 256, default: 256).")
    parser.add_argument(
        "--export-phase-offset-rad", type=float, default=math.pi,
        help="Hardware phase offset before wrapping to [0, 2pi), default: pi.",
    )
    return parser.parse_args()


def build_output_directories(root, phase_count):
    directories = {
        "input": root / "input",
        "teacher_feature": root / "teacher_feature",
        "detection": root / "detection",
        "json": root / "json",
    }
    directories.update({f"slm{index}": root / f"slm{index}" for index in range(1, phase_count + 1)})
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def phase_map_to_drive_image(phase_map, target_shape, lut_phase, lut_gray, args):
    centered_phase = resample_phase_map(
        phase_map.float(), target_shape
    )[0, 0].detach().cpu().numpy()
    hardware_phase = np.remainder(centered_phase + args.export_phase_offset_rad, 2.0 * math.pi)
    gray = phase_to_gray(hardware_phase, lut_phase, lut_gray, args.gray_inverted)
    # Quantize to the actual available hardware levels before writing the PNG.
    gray = np.rint(gray * (args.phase_levels - 1)) / (args.phase_levels - 1)
    return np.rint(gray * 255.0).astype(np.uint8)


def teacher_feature_to_image(feature):
    """Convert one teacher feature map into a contrast-enhanced 8-bit image."""
    feature_np = feature[0, 0].detach().float().cpu().numpy()
    display = enhance_feature_for_display(feature_np)
    return Image.fromarray(np.rint(display * 255.0).astype(np.uint8), mode="L")


def draw_prediction_boxes(image, detections):
    """Render one red box and a class-confidence label for every prediction."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for cx, cy, width, height, confidence, class_id in detections:
        x1, y1 = cx - width / 2, cy - height / 2
        x2, y2 = cx + width / 2, cy + height / 2
        label = f"{Config.CLASS_NAMES[int(class_id)]} {confidence:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        label_box = draw.textbbox((x1, y1), label)
        label_height = label_box[3] - label_box[1]
        text_y = max(0, y1 - label_height - 4)
        label_bottom = max(text_y + label_height + 3, min(max(y1, 0), canvas.height))
        draw.rectangle((x1, text_y, x1 + (label_box[2] - label_box[0]) + 4, label_bottom), fill="red")
        draw.text((x1 + 2, text_y + 1), label, fill="white")
    return canvas


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if args.lut is not None and not args.lut.is_file():
        raise FileNotFoundError(f"LUT not found: {args.lut}")
    if not 2 <= args.phase_levels <= 256:
        raise ValueError("--phase-levels must be between 2 and 256.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    for name in ("conf_threshold", "nms_threshold"):
        if (value := getattr(args, name)) is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")

    Config.YAML_PATH = str(args.data.resolve())
    Config.initialize()
    dataset = YOLODataset(Config, yaml_path=Config.YAML_PATH, split=args.split)
    source_paths = dataset.files[:args.limit]
    if not source_paths:
        raise RuntimeError(f"Dataset split {args.split!r} is empty.")

    device = torch.device(args.device or Config.DEVICE)
    teacher = build_teacher(Config).to(device).eval()
    detector = build_detector_head(Config, in_channels=1).to(device).eval()
    checkpoint = load_checkpoint(args.checkpoint, teacher, detector, device)
    lut_phase, lut_gray = load_phase_lut(args.lut)
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    directories = None
    with torch.inference_mode():
        for position, source_path in enumerate(tqdm(source_paths, desc=f"Exporting {args.split}"), start=1):
            file_id = f"{position:04d}"
            input_image, input_tensor = prepare_scene(source_path, Config.RESOLUTION)
            # Keep phase prediction and complex ASM propagation in float32.
            teacher_aux = teacher(input_tensor.to(device), return_aux=True)
            predictions = detector(prepare_detector_feature(Config, teacher_aux["det_feature"]))
            detections = decode_detections(Config, predictions, args.conf_threshold, args.nms_threshold)[0]
            phase_maps = teacher_aux.get("phase_maps", ())
            if directories is None:
                directories = build_output_directories(output_root, len(phase_maps))
            elif len(phase_maps) != sum(key.startswith("slm") for key in directories):
                raise RuntimeError("Teacher returned an inconsistent number of phase maps across images.")

            input_path = directories["input"] / f"{file_id}.png"
            feature_path = directories["teacher_feature"] / f"{file_id}.png"
            detection_path = directories["detection"] / f"{file_id}.png"
            json_path = directories["json"] / f"{file_id}.json"
            input_image.save(input_path)
            teacher_feature_to_image(teacher_aux["det_feature"]).save(feature_path)
            draw_prediction_boxes(input_image, detections).save(detection_path)
            phase_files = []
            for layer_index, phase_map in enumerate(phase_maps, start=1):
                phase_path = directories[f"slm{layer_index}"] / f"{file_id}.png"
                target_shape = Config.TEACHER_V2_ACTIVE_PIXEL_SHAPES[layer_index - 1]
                Image.fromarray(
                    phase_map_to_drive_image(phase_map, target_shape, lut_phase, lut_gray, args), mode="L"
                ).save(phase_path)
                phase_files.append(str(phase_path.relative_to(output_root)).replace("\\", "/"))

            record = {
                "id": file_id,
                "source_image": str(Path(source_path).resolve()),
                "input_letterboxed": str(input_path.relative_to(output_root)).replace("\\", "/"),
                "teacher_feature": str(feature_path.relative_to(output_root)).replace("\\", "/"),
                "slm_gray_drives": phase_files,
                "detection_visualization": str(detection_path.relative_to(output_root)).replace("\\", "/"),
                "detections": detections_to_records(detections),
            }
            json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            manifest.append(record)

    export_info = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_map50": checkpoint.get("val_map50"),
        "teacher_arch": checkpoint.get("teacher_arch"),
        "dataset_yaml": str(args.data.resolve()),
        "split": args.split,
        "exported_images": len(manifest),
        "simulation_resolution_hw": list(Config.RESOLUTION),
        "dmd_pixel_pitch_m": Config.DMD_PIXEL_PITCH,
        "slm_profiles": list(Config.TEACHER_V2_SLM_PROFILES),
        "slm_hardware_pixel_pitches_m": list(Config.TEACHER_V2_HARDWARE_PIXEL_PITCHES),
        "slm_effective_sampling_pitches_m": list(Config.TEACHER_V2_SAMPLING_PITCHES),
        "slm_active_shapes_hw": [list(shape) for shape in Config.TEACHER_V2_ACTIVE_PIXEL_SHAPES],
        "input_intensity_mode": Config.INPUT_INTENSITY_MODE,
        "phase_export_offset_rad": args.export_phase_offset_rad,
        "phase_levels": args.phase_levels,
        "gray_inverted": args.gray_inverted,
        "gray_to_phase_lut": str(args.lut.resolve()) if args.lut else "ideal_linear",
        "numbering": "The same 0001-based ID identifies matching files in input, teacher_feature, slm1, slm2, detection, and json.",
        "samples": manifest,
    }
    (output_root / "manifest.json").write_text(json.dumps(export_info, indent=2), encoding="utf-8")
    print(f"Exported {len(manifest)} {args.split} images to: {output_root}")
    print("Directories: input, teacher_feature, slm1..slmN, detection, json; each uses the same 0001-based ID.")


if __name__ == "__main__":
    main()
