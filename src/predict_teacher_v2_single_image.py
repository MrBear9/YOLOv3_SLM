"""Predict input-dependent Teacher V2 phases and detect one image.

The exported phase PNGs are generated for one specified input image.  They are
not fixed phases that can be reused for a different input image.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

# Allow direct execution with ``python src/predict_teacher_v2_single_image.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dataset import image_to_intensity_tensor, letterbox_image_targets
from models.teacher import build_teacher
from models.yolov8.config_v8 import ConfigYOLOv8Anchor as Config
from models.yolov8.detection_protocol import decode_detections
from models.yolov8.feature_adapter import prepare_detector_feature
from models.yolov8.head_v8 import build_detector_head


def parse_args():
    parser = argparse.ArgumentParser(description="Export Teacher V2 phases and detect one input image.")
    parser.add_argument("--image", type=Path, required=True, help="Input scene image, displayed to the DMD/SLM system.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Teacher + light detector checkpoint.")
    parser.add_argument("--output", type=Path, default=Path("output/single_teacher_v2"), help="Output directory.")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu (default: training config).")
    parser.add_argument("--conf-threshold", type=float, default=None, help="Detection confidence threshold.")
    parser.add_argument("--nms-threshold", type=float, default=None, help="NMS IoU threshold.")
    parser.add_argument(
        "--captured-feature", type=Path, default=None,
        help="Optional camera image captured after the loaded SLM phases. It is detected by the light head.",
    )
    parser.add_argument(
        "--lut", type=Path, default=None,
        help="Optional calibrated gray-to-phase LUT (.npy, .csv, or .txt).",
    )
    parser.add_argument("--gray-inverted", action="store_true", help="Invert SLM gray drive after LUT conversion.")
    parser.add_argument("--phase-levels", type=int, default=256, help="SLM gray levels (default: 256).")
    parser.add_argument(
        "--export-phase-offset-rad", type=float, default=math.pi,
        help="Hardware phase offset before wrapping to [0, 2pi), default: pi.",
    )
    return parser.parse_args()


def load_checkpoint(path, teacher, detector, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Expected a checkpoint dictionary, got {type(checkpoint).__name__}.")
    for key, module in (("teacher_state_dict", teacher), ("detector_state_dict", detector)):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint is missing {key!r}: {path}")
        module.load_state_dict(checkpoint[key], strict=True)
    return checkpoint


def load_phase_lut(path):
    if path is None:
        return None, None
    if path.suffix.lower() == ".npy":
        values = np.asarray(np.load(path), dtype=np.float32)
    else:
        values = np.asarray(np.loadtxt(path, delimiter=","), dtype=np.float32)
    if values.ndim == 1:
        gray = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
        phase = values
    elif values.ndim == 2 and values.shape[1] == 2:
        gray, phase = values[:, 0], values[:, 1]
        gray = gray / (255.0 if gray.max() > 1.0 else 1.0)
    else:
        raise ValueError("--lut must contain phase values or (gray, phase) pairs.")
    order = np.argsort(phase)
    phase, gray = phase[order], gray[order]
    if phase.size < 2 or np.any(np.diff(phase) < 0):
        raise ValueError("The LUT must contain at least two non-decreasing phase values.")
    return phase, gray


def phase_to_gray(phase_0_2pi, lut_phase, lut_gray, inverted):
    if lut_phase is None:
        gray = phase_0_2pi / (2.0 * math.pi)
    else:
        gray = np.interp(phase_0_2pi, lut_phase, lut_gray)
    gray = np.clip(gray, 0.0, 1.0)
    return 1.0 - gray if inverted else gray


def prepare_scene(image_path, resolution):
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        letterboxed, _ = letterbox_image_targets(source, torch.zeros((0, 5)), resolution)
    tensor = image_to_intensity_tensor(letterboxed, mode=Config.INPUT_INTENSITY_MODE).unsqueeze(0)
    return letterboxed, tensor


def prepare_camera_feature(image_path, resolution):
    with Image.open(image_path) as image:
        image = image.convert("L").resize((resolution[1], resolution[0]), Image.Resampling.BILINEAR)
    return image, torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)


def detections_to_records(detections):
    return [
        {
            "class_id": int(class_id),
            "class_name": Config.CLASS_NAMES[int(class_id)],
            "confidence": float(confidence),
            "xywh_pixels": [float(cx), float(cy), float(width), float(height)],
        }
        for cx, cy, width, height, confidence, class_id in detections
    ]


def draw_detections(image, detections):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for cx, cy, width, height, confidence, class_id in detections:
        x1, y1 = cx - width / 2, cy - height / 2
        x2, y2 = cx + width / 2, cy + height / 2
        label = f"{Config.CLASS_NAMES[int(class_id)]}: {confidence:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, max(0, y1 - 14)), label, fill="red", stroke_width=1, stroke_fill="black")
    return canvas


def export_phase_maps(phase_maps, output_dir, args):
    lut_phase, lut_gray = load_phase_lut(args.lut)
    saved = []
    for index, phase_map in enumerate(phase_maps, start=1):
        centered = phase_map[0, 0].detach().float().cpu().numpy()
        hardware_phase = np.remainder(centered + args.export_phase_offset_rad, 2.0 * math.pi)
        gray = phase_to_gray(hardware_phase, lut_phase, lut_gray, args.gray_inverted)
        # The PNG is 8-bit; the exact continuous phase remains available as NPY.
        gray_u8 = np.rint(gray * 255.0).astype(np.uint8)
        np.save(output_dir / f"phase_map_{index}_centered_rad.npy", centered)
        np.save(output_dir / f"phase_map_{index}_hardware_phase_0_2pi_rad.npy", hardware_phase)
        Image.fromarray(gray_u8, mode="L").save(output_dir / f"phase_map_{index}_slm_gray.png")
        saved.append({
            "layer": index,
            "centered_phase_rad": f"phase_map_{index}_centered_rad.npy",
            "hardware_phase_rad": f"phase_map_{index}_hardware_phase_0_2pi_rad.npy",
            "slm_gray_png": f"phase_map_{index}_slm_gray.png",
        })
    return saved


def main():
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.captured_feature is not None and not args.captured_feature.is_file():
        raise FileNotFoundError(f"Captured feature image not found: {args.captured_feature}")
    if args.lut is not None and not args.lut.is_file():
        raise FileNotFoundError(f"LUT not found: {args.lut}")
    if not 2 <= args.phase_levels <= 256:
        raise ValueError("--phase-levels must be between 2 and 256 because PNG output is 8-bit.")

    Config.initialize()
    device = torch.device(args.device or Config.DEVICE)
    teacher = build_teacher(Config).to(device).eval()
    detector = build_detector_head(Config, in_channels=1).to(device).eval()
    checkpoint = load_checkpoint(args.checkpoint, teacher, detector, device)

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_image, scene_tensor = prepare_scene(args.image, Config.RESOLUTION)
    scene_image.save(output_dir / "input_letterboxed.png")

    with torch.inference_mode():
        teacher_aux = teacher(scene_tensor.to(device), return_aux=True)
        simulation_predictions = detector(prepare_detector_feature(Config, teacher_aux["det_feature"]))
        simulation_detections = decode_detections(
            Config, simulation_predictions, args.conf_threshold, args.nms_threshold
        )[0]
        if args.captured_feature is not None:
            camera_image, camera_tensor = prepare_camera_feature(args.captured_feature, Config.RESOLUTION)
            hardware_predictions = detector(prepare_detector_feature(Config, camera_tensor.to(device)))
            hardware_detections = decode_detections(
                Config, hardware_predictions, args.conf_threshold, args.nms_threshold
            )[0]
        else:
            camera_image, hardware_detections = None, None

    phase_files = export_phase_maps(teacher_aux["phase_maps"], output_dir, args)
    draw_detections(scene_image, simulation_detections).save(output_dir / "detection_simulation.png")
    (output_dir / "detections_simulation.json").write_text(
        json.dumps(detections_to_records(simulation_detections), indent=2), encoding="utf-8"
    )
    if camera_image is not None:
        draw_detections(camera_image, hardware_detections).save(output_dir / "detection_hardware.png")
        (output_dir / "detections_hardware.json").write_text(
            json.dumps(detections_to_records(hardware_detections), indent=2), encoding="utf-8"
        )

    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_map50": checkpoint.get("val_map50"),
        "teacher_arch": checkpoint.get("teacher_arch"),
        "resolution_hw": list(Config.RESOLUTION),
        "input_intensity_mode": Config.INPUT_INTENSITY_MODE,
        "slm_layers": phase_files,
        "phase_export_offset_rad": args.export_phase_offset_rad,
        "gray_inverted": args.gray_inverted,
        "gray_to_phase_lut": str(args.lut.resolve()) if args.lut else "ideal_linear",
        "note": "Each phase map is valid only for input_letterboxed.png and its registered optical input.",
    }
    (output_dir / "export_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Exported {len(phase_files)} input-dependent phase maps to: {output_dir}")
    print(f"Simulation detections: {len(simulation_detections)} -> {output_dir / 'detection_simulation.png'}")
    if camera_image is not None:
        print(f"Hardware-capture detections: {len(hardware_detections)} -> {output_dir / 'detection_hardware.png'}")


if __name__ == "__main__":
    main()

"""
# 单图“教师 V2 相位预测 + light 检测”脚本：
# [src/predict_teacher_v2_single_image.py (line 1)](E:\\pythonProject\\YOLOv3_SLM\\src\\predict_teacher_v2_single_image.py:1)
# 服务器运行示例：
# python src/predict_teacher_v2_single_image.py `
#   --image path/to/scene.png `
#   --checkpoint output/Tv2_light/teacher_detector_best.pth `
#   --output output/single_image_test
# 输出包括：
# 两层指定图像对应的 SLM 灰度相位图：phase_map_1_slm_gray.png、phase_map_2_slm_gray.png
# 精确相位数组：*.npy
# 仿真光学输出的检测图：detection_simulation.png
# 检测框与置信度：detections_simulation.json
# 相位、输入和硬件设置记录：export_metadata.json
# 将两张相位图依次加载至两块 SLM，并保证输入图、DMD 显示方式、640×640 letterbox、波长、像元尺寸、传播距离均与训练一致。相位图只对本次指定输入图有效，换图后必须重新预测并加载相位。
# 若你拍到经过 SLM 系统后的相机灰度图，可追加：
# --captured-feature path/to/camera_capture.png
# 脚本会额外输出基于真实相机结果的 detection_hardware.png
"""