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

from models.SLM.physical_defaults import add_optics_argument
from models.class_display import class_name_for_id
from src.hardware_detection_iou import add_detection_iou
from models.dataset import YOLODataset, letterbox_image_targets
from models.SLM.slm_modulation import resample_phase_map
from models.teacher import build_teacher, configure_teacher_checkpoint
from models.teacher_guidance import optical_preview_to_uint8
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
    add_optics_argument(parser)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Teacher V2 + light detector checkpoint.")
    parser.add_argument("--data", type=Path, default=Path("data/military/data.yaml"), help="Dataset YAML file.")
    parser.add_argument("--split", choices=("val", "test"), default="test", help="Dataset split to export.")
    parser.add_argument("--output", type=Path, required=True, help="Root directory for numbered hardware export files.")
    parser.add_argument("--device", default="cpu", help="cuda, cuda:0, or cpu (default: cpu for local hardware export).")
    parser.add_argument("--conf-threshold", type=float, default=None, help="Detection confidence threshold.")
    parser.add_argument("--nms-threshold", type=float, default=None, help="NMS IoU threshold.")
    parser.add_argument("--match-iou-threshold", type=float, default=0.5,
                        help="Same-class one-to-one IoU matching threshold for JSON diagnostics (default: 0.5).")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of images to export, useful for a hardware trial.")
    parser.add_argument("--color-map", action="store_true",
                        help="Render optical output previews with training viridis colors; default: grayscale. Inputs and SLM drives stay grayscale.")
    parser.add_argument("--lut", type=Path, default=Path(Config.SLM_GRAY_TO_PHASE_LUT) if Config.SLM_GRAY_TO_PHASE_LUT else None, help="Calibrated LUT; defaults to optics YAML.")
    parser.add_argument("--gray-inverted", action=argparse.BooleanOptionalAction, default=Config.SLM_GRAY_INVERTED, help="Gray polarity; defaults to optics YAML.")
    parser.add_argument("--phase-levels", type=int, default=Config.SLM_PHASE_LEVELS, help="Number of SLM drive levels (2 to 256); defaults to optics YAML.")
    parser.add_argument(
        "--export-phase-offset-rad", type=float, default=Config.SLM_EXPORT_PHASE_OFFSET_RAD,
        help="Hardware phase offset before wrapping to [0, 2pi); defaults to optics YAML.",
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


def teacher_feature_to_image(feature, color_map=False):
    """Convert one teacher feature map into a contrast-enhanced 8-bit image."""
    feature_np = feature[0, 0].detach().float().cpu().numpy()
    return Image.fromarray(optical_preview_to_uint8(feature_np, color_map=color_map))


def load_letterboxed_ground_truth(source_path, label_path, resolution):
    """Load YOLO labels and apply the exact letterbox transform used for inference."""
    targets = []
    if label_path.is_file():
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                box = [float(value) for value in parts[1:5]]
            except ValueError as exc:
                raise ValueError(f"Invalid YOLO label in {label_path} at line {line_number}.") from exc
            if not 0 <= class_id < len(Config.CLASS_NAMES):
                raise ValueError(f"Class ID {class_id} in {label_path} is outside the configured class range.")
            targets.append([class_id, *box])

    target_tensor = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 5), dtype=torch.float32)
    with Image.open(source_path) as source:
        _, target_tensor = letterbox_image_targets(source.convert("RGB"), target_tensor, resolution)
    return target_tensor


def ground_truth_to_records(targets, image_size):
    """Convert letterboxed normalized YOLO targets to the exported pixel convention."""
    height, width = image_size
    return [
        {
            "class_id": int(class_id),
            "class_name": Config.CLASS_NAMES[int(class_id)],
            "xywh_pixels": [
                float(cx * width), float(cy * height), float(box_width * width), float(box_height * height),
            ],
        }
        for class_id, cx, cy, box_width, box_height in targets.tolist()
    ]


def draw_detection_boxes(image, detections, ground_truth):
    """Render colored GT/prediction annotations on a grayscale scene."""
    canvas = image.convert("L").convert("RGB")
    draw = ImageDraw.Draw(canvas)
    image_width, image_height = canvas.size

    for class_id, cx, cy, width, height in ground_truth.tolist():
        x1, y1 = (cx - width / 2) * image_width, (cy - height / 2) * image_height
        x2, y2 = (cx + width / 2) * image_width, (cy + height / 2) * image_height
        label = f"GT {class_name_for_id(Config.CLASS_NAMES, int(class_id))}"
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
        label_box = draw.textbbox((x1, y1), label)
        label_height = label_box[3] - label_box[1]
        text_y = max(0, y1 - label_height - 4)
        draw.rectangle((x1, text_y, x1 + (label_box[2] - label_box[0]) + 4, text_y + label_height + 3), fill="lime")
        draw.text((x1 + 2, text_y + 1), label, fill="black")

    for cx, cy, width, height, confidence, class_id in detections:
        x1, y1 = cx - width / 2, cy - height / 2
        x2, y2 = cx + width / 2, cy + height / 2
        label = (
            f"Pred {class_name_for_id(Config.CLASS_NAMES, int(class_id))} "
            f"{confidence:.2f}"
        )
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
    if not 0 < args.match_iou_threshold <= 1:
        raise ValueError("--match-iou-threshold must be in (0, 1].")
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
    configure_teacher_checkpoint(Config, args.checkpoint)
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
            # Export the actual optical input, using the teacher's channel mean
            # rather than PIL's different RGB luminance conversion. No contrast stretch.
            gray_input = input_tensor.mean(dim=1)[0].clamp(0.0, 1.0)
            input_image = Image.fromarray(
                np.rint(gray_input.cpu().numpy() * 255.0).astype(np.uint8)
            )
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
            ground_truth = load_letterboxed_ground_truth(
                source_path,
                Path(dataset.get_label_path(source_path)),
                Config.RESOLUTION,
            )
            input_image.save(input_path)
            teacher_feature_to_image(teacher_aux["det_feature"], color_map=args.color_map).save(feature_path)
            draw_detection_boxes(input_image, detections, ground_truth).save(detection_path)
            phase_files = []
            panel_files = []
            for layer_index, phase_map in enumerate(phase_maps, start=1):
                phase_path = directories[f"slm{layer_index}"] / f"{file_id}.png"
                target_shape = Config.TEACHER_V2_ACTIVE_PIXEL_SHAPES[layer_index - 1]
                Image.fromarray(
                    phase_map_to_drive_image(phase_map, target_shape, lut_phase, lut_gray, args), mode="L"
                ).save(phase_path)
                phase_files.append(str(phase_path.relative_to(output_root)).replace("\\", "/"))
                if Config.OPTICAL_COORDINATE_MODE == "physical":
                    profile = Config.SLM_PROFILES[Config.TEACHER_V2_SLM_PROFILES[layer_index - 1]]
                    panel_shape = profile["resolution"]
                    background = phase_map_to_drive_image(torch.zeros_like(phase_map), (1, 1),
                                                          lut_phase, lut_gray, args).item()
                    drive = phase_map_to_drive_image(phase_map, target_shape, lut_phase, lut_gray, args)
                    panel = np.full(panel_shape, background, dtype=np.uint8)
                    top, left = ((n - m) // 2 for n, m in zip(panel_shape, target_shape))
                    panel[top:top + target_shape[0], left:left + target_shape[1]] = drive
                    panel_path = output_root / f"slm{layer_index}_panel" / f"{file_id}.png"
                    panel_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(panel, mode="L").save(panel_path)
                    panel_files.append(str(panel_path.relative_to(output_root)).replace("\\", "/"))

            record = {
                "id": file_id,
                "source_image": str(Path(source_path).resolve()),
                "input_letterboxed": str(input_path.relative_to(output_root)).replace("\\", "/"),
                "teacher_feature": str(feature_path.relative_to(output_root)).replace("\\", "/"),
                "slm_gray_drives": phase_files,
                "slm_full_panel_drives": panel_files,
                "physical_grid": teacher.simulator.geometry.metadata() if teacher.simulator.geometry is not None else None,
                "detection_visualization": str(detection_path.relative_to(output_root)).replace("\\", "/"),
                "detections": detections_to_records(detections),
                "ground_truth": ground_truth_to_records(ground_truth, input_image.size[::-1]),
            }
            add_detection_iou(record, args.match_iou_threshold)
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
        "input_png_mode": "L",
        "optical_preview_colormap": "viridis" if args.color_map else "gray",
        "optical_preview_png_mode": "RGB" if args.color_map else "L",
        "input_gray_conversion": "teacher input tensor channel mean, clamped to [0,1], rounded to uint8",
        "detection_png_mode": "RGB",
        "detection_annotations": "Grayscale scene; GT: green boxes; Pred: red boxes; visualization only",
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
