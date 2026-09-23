"""Evaluate a paired optical-student + light-detector checkpoint on val or test."""

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, hsv_to_rgb
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow direct execution with ``python src/evaluate_slm_light_dataset.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.SLM.config_slm import ConfigSLM as Config
from models.SLM.dataset_slm import SLMFeatureDataset, slm_collate_fn
from models.SLM.optical_layers import OpticalStudent
from models.SLM.utils_slm import load_student_detector_checkpoint
from models.geometry import bbox_iou_xywh
from models.detection_metrics_offline import add_multi_iou_metrics
from models.coco_offline import evaluate_coco, require_coco
from models.monitoring import (
    _render_confusion_matrix_image,
    _render_normalized_confusion_matrix_image,
    compute_detection_confusion_matrix,
)
from models.review_candidates import write_review_lists
from models.runtime import get_dataloader_kwargs
from models.yolov8.detection_protocol import decode_detections
from models.yolov8.feature_adapter import prepare_slm_detector_feature
from models.yolov8.head_v8 import build_detector_head
from models.yolov8.metrics_anchor_v8 import compute_average_precision, compute_pr_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an optical student + light detector on a labelled dataset split.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Paired detector_best.pth with student and detector states.")
    parser.add_argument("--data", type=Path, default=Path("data/military/data.yaml"), help="Dataset YAML file.")
    parser.add_argument("--split", choices=("val", "test"), default="test", help="Dataset split to evaluate.")
    parser.add_argument("--output", type=Path, default=Path("output/slm_light_dataset_eval"), help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch-size override.")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu (default: student config).")
    parser.add_argument("--conf-threshold", type=float, default=None, help="Confidence for the count confusion matrix.")
    parser.add_argument("--nms-threshold", type=float, default=None, help="NMS IoU threshold for decoding.")
    parser.add_argument("--iou-threshold", type=float, default=None, help="IoU for confusion/review matching only; AP50 and AP50:95 use fixed thresholds.")
    parser.add_argument(
        "--optical-field-samples", type=int, default=0,
        help="Save raw final optical intensity plus complex-field real/imaginary parts for the first N split images (0 disables).",
    )
    parser.add_argument("--skip-coco", action="store_true", help="Skip official COCO evaluation (enabled by default).")
    return parser.parse_args()


def load_checkpoint_metadata(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_student(config, checkpoint):
    """Match a multi-head checkpoint automatically; optical settings remain config-owned."""
    checkpoint_heads = int(checkpoint.get("num_heads", 1)) if isinstance(checkpoint, dict) else 1
    if checkpoint_heads > 1:
        from models.SLM.multi_head_slm import MultiHeadOpticalStudent

        config.SLM_MULTI_HEAD_ENABLED = True
        config.SLM_MULTI_HEAD_NUM_HEADS = checkpoint_heads
        return MultiHeadOpticalStudent(config)
    config.SLM_MULTI_HEAD_ENABLED = False
    return OpticalStudent(config)


def evaluate_student(config, student, detector, dataloader, device):
    """Compute the same class-wise AP50 matching protocol used by SLM training."""
    student.eval()
    detector.eval()
    all_detections, all_targets, source_paths = [], [], []
    storage = {class_id: [] for class_id in range(config.NUM_CLASSES)}
    gt_counts = {class_id: 0 for class_id in range(config.NUM_CLASSES)}
    amp_enabled = bool(getattr(config, "ENABLE_AMP", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(getattr(config, "AMP_DTYPE", "float16")).lower() in {"bf16", "bfloat16"} else torch.float16
    amp_context = torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled) if device.type == "cuda" else nullcontext()

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating SLM + light"):
            gray = batch["gray_tensor"].to(device, non_blocking=config.PIN_MEMORY)
            if config.ENABLE_CHANNELS_LAST and device.type == "cuda":
                gray = gray.contiguous(memory_format=torch.channels_last)
            with amp_context:
                feature = student(gray)
                predictions = detector(prepare_slm_detector_feature(config, feature))
            detections = decode_detections(
                config,
                predictions,
                conf_thresh=config.METRIC_CONF_THRESH,
                nms_thresh=config.METRIC_NMS_THRESH,
                max_det=config.METRIC_MAX_DET,
            )
            for targets, sample_detections, source_path in zip(
                batch["targets"], detections, batch["image_paths"],
            ):
                targets_np = targets.cpu().numpy()
                detections_np = np.asarray(sample_detections, dtype=np.float32).reshape(-1, 6)
                all_targets.append(targets_np)
                all_detections.append(detections_np)
                source_paths.append(Path(source_path).resolve())
                gt_by_class = {}
                for target in targets_np:
                    if target.shape[0] < 5 or target[3] <= 0 or target[4] <= 0:
                        continue
                    class_id = int(target[0])
                    gt_by_class.setdefault(class_id, []).append(target[1:5] * np.asarray(
                        [config.RESOLUTION[1], config.RESOLUTION[0], config.RESOLUTION[1], config.RESOLUTION[0]], dtype=np.float32
                    ))
                    gt_counts[class_id] += 1
                for class_id in range(config.NUM_CLASSES):
                    class_detections = sorted(
                        (det for det in detections_np if int(det[5]) == class_id), key=lambda det: float(det[4]), reverse=True
                    )
                    matched_gt = set()
                    for detection in class_detections:
                        best_iou, best_index = 0.0, -1
                        for index, gt_box in enumerate(gt_by_class.get(class_id, ())):
                            if index in matched_gt:
                                continue
                            iou = float(bbox_iou_xywh(
                                torch.from_numpy(detection[:4]).float().unsqueeze(0),
                                torch.from_numpy(np.asarray(gt_box)).float().unsqueeze(0),
                            ).item())
                            if iou > best_iou:
                                best_iou, best_index = iou, index
                        is_tp = best_index >= 0 and best_iou >= 0.5
                        storage[class_id].append((float(detection[4]), float(is_tp)))
                        if is_tp:
                            matched_gt.add(best_index)

    per_class, ap_values = {}, []
    for class_id in range(config.NUM_CLASSES):
        ap50 = compute_average_precision(storage[class_id], gt_counts[class_id])
        if ap50 is not None:
            ap_values.append(ap50)
        per_class[class_id] = {
            "class_name": config.CLASS_NAMES[class_id],
            "ap50": float(ap50 or 0.0),
            "gt_count": int(gt_counts[class_id]),
            **compute_pr_summary(storage[class_id], gt_counts[class_id]),
        }
    metrics = {
        "map50": float(np.mean(ap_values)) if ap_values else 0.0,
        "per_class": per_class,
    }
    add_multi_iou_metrics(metrics, all_detections, all_targets, config)
    return all_detections, all_targets, source_paths, metrics


def _display_range(values, lower=2.0, upper=98.0):
    """Return a robust display range without modifying the saved raw array."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, (lower, upper))
    if high <= low + 1e-12:
        low, high = float(finite.min()), float(finite.max())
    if high <= low + 1e-12:
        high = low + 1.0
    return float(low), float(high)


def _complex_field_to_rgb(real, imag, magnitude_scale):
    """Encode the 2-D (Re(U), Im(U)) direction as hue and |U| as brightness."""
    magnitude = np.hypot(real, imag)
    phase = np.arctan2(imag, real)
    hue = np.mod(phase, 2.0 * np.pi) / (2.0 * np.pi)
    value = np.clip(magnitude / max(float(magnitude_scale), 1e-12), 0.0, 1.0)
    hsv = np.stack((hue, np.ones_like(hue), value), axis=-1)
    return hsv_to_rgb(hsv)


def _draw_complex_mapping_legend(axis):
    """Draw a compact 2-D Re(U)/Im(U) colour key for the complex RGB panel."""
    coords = np.linspace(-1.0, 1.0, 161)
    real, imag = np.meshgrid(coords, coords)
    rgb = _complex_field_to_rgb(real, imag, magnitude_scale=1.0)
    outside = np.hypot(real, imag) > 1.0
    rgb[outside] = 1.0
    axis.imshow(rgb, origin="lower", extent=(-1.0, 1.0, -1.0, 1.0))
    axis.axhline(0.0, color="black", linewidth=0.35, alpha=0.45)
    axis.axvline(0.0, color="black", linewidth=0.35, alpha=0.45)
    axis.set_xlabel(r"Re$(U)$", fontsize=6, labelpad=1)
    axis.set_ylabel(r"Im$(U)$", fontsize=6, labelpad=1)
    axis.tick_params(axis="both", labelsize=5, length=1)


def _save_mapping_legend(path):
    """Save the two scalar colourbars and the complex Re/Im colour key separately."""
    figure, axes = plt.subplots(
        1, 3, figsize=(9.0, 2.7),
        gridspec_kw={"width_ratios": (1.25, 1.25, 1.0)},
        constrained_layout=True,
    )
    axes[0].set_title("DMD input intensity", fontsize=10)
    figure.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap="gray"),
        cax=axes[0], orientation="horizontal",
    )
    axes[0].set_xlabel("Intensity", fontsize=8)

    axes[1].set_title(r"Raw optical intensity $|U|^2$", fontsize=10)
    figure.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap="magma"),
        cax=axes[1], orientation="horizontal",
    )
    axes[1].set_xlabel("Preview-normalised display value", fontsize=8)

    axes[2].set_title("2-D complex mapping", fontsize=10)
    _draw_complex_mapping_legend(axes[2])
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_optical_field_samples(config, student, dataset, device, output_dir, sample_count):
    """Save labelled optical-field previews and raw-intensity statistics."""
    if sample_count <= 0:
        return None
    if not hasattr(student, "forward_with_optical_field"):
        raise ValueError(
            "--optical-field-samples currently requires a single-head OpticalStudent checkpoint. "
            "A multi-head model has multiple final complex fields, so it has no unique Re(U)/Im(U) image."
        )

    output_dir = output_dir / "optical_fields"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(int(sample_count), len(dataset))
    was_training = student.training
    student.eval()
    summary = {
        "definition": "raw_intensity = |U_final|^2 before blur, normalisation, clamping, detector conversion, or display stretch",
        "display_note": "PNG previews use robust display ranges; raw arrays are not saved.",
        "raw_numeric_arrays_saved": False,
        "mapping_legend": "mapping_legend.png",
        "student_output_normalised": bool(getattr(student, "enable_norm", False)),
        "student_norm_mode": str(getattr(config, "STUDENT_NORM_MODE", "none")),
        "output_blur_kernel": int(getattr(config, "STUDENT_OUTPUT_BLUR_KERNEL", 1)),
        "samples": [],
    }
    _save_mapping_legend(output_dir / summary["mapping_legend"])

    with torch.inference_mode():
        for sample_index in range(sample_count):
            sample = dataset[sample_index]
            gray = sample["gray_tensor"].unsqueeze(0).to(device, non_blocking=config.PIN_MEMORY)
            if config.ENABLE_CHANNELS_LAST and device.type == "cuda":
                gray = gray.contiguous(memory_format=torch.channels_last)
            _, optical = student.forward_with_optical_field(gray)
            raw_intensity = optical["raw_intensity"][0, 0].detach().float().cpu().numpy()
            field = optical["field"][0, 0].detach().cpu()
            real = field.real.float().numpy()
            imag = field.imag.float().numpy()
            input_intensity = sample["gray_tensor"][0].detach().float().cpu().numpy()

            stem = f"sample_{sample_index:02d}_{Path(sample['image_path']).stem}"
            raw_low, raw_high = _display_range(raw_intensity)
            magnitude = np.hypot(real, imag)
            complex_display_scale = max(float(np.percentile(magnitude, 99)), 1e-12)
            complex_rgb = _complex_field_to_rgb(real, imag, complex_display_scale)
            figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
            panels = (
                (input_intensity, "DMD input intensity", "gray", 0.0, 1.0),
                (raw_intensity, r"Raw optical intensity $|U|^2$", "magma", raw_low, raw_high),
                (complex_rgb, r"Complex field: 2-D Re$(U)$/Im$(U)$ mapping", None, None, None),
            )
            for axis, (array, title, cmap, vmin, vmax) in zip(axes, panels):
                axis.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
                axis.set_title(title)
                axis.axis("off")
            figure.savefig(output_dir / f"{stem}_optical_fields.png", dpi=160)
            plt.close(figure)

            summary["samples"].append({
                "sample_index": sample_index,
                "source_image": str(sample["image_path"]),
                "preview": f"{stem}_optical_fields.png",
                "raw_intensity_stats": {
                    "min": float(raw_intensity.min()), "max": float(raw_intensity.max()),
                    "mean": float(raw_intensity.mean()), "p01": float(np.percentile(raw_intensity, 1)),
                    "p50": float(np.percentile(raw_intensity, 50)), "p99": float(np.percentile(raw_intensity, 99)),
                    "preview_vmin": raw_low, "preview_vmax": raw_high,
                },
                "complex_mapping": {
                    "encoding": "hue=atan2(Im(U), Re(U)); brightness=clip(|U| / p99(|U|), 0, 1)",
                    "magnitude_p99": complex_display_scale,
                },
            })
    if was_training:
        student.train()
    (output_dir / "optical_field_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"directory": str(output_dir), "samples": sample_count}


def save_outputs(output_dir, confusion, metrics, checkpoint, restore_info, args, dataset_size, review_summary, optical_field_summary):
    class_names = Config.CLASS_NAMES
    labels = [class_names[index] for index in range(Config.NUM_CLASSES)] + ["background"]
    foreground = confusion[:Config.NUM_CLASSES, :Config.NUM_CLASSES]
    percentages = np.divide(foreground * 100.0, foreground.sum(axis=1, keepdims=True), out=np.zeros_like(foreground), where=foreground.sum(axis=1, keepdims=True) > 0)
    Image.fromarray(_render_confusion_matrix_image(
        confusion, class_names,
        title="Detection confusion matrix",
    )).save(output_dir / "confusion_matrix_counts_with_background.png")
    Image.fromarray(_render_normalized_confusion_matrix_image(
        confusion, class_names,
        title="Foreground class consistency (%)",
    )).save(output_dir / "confusion_matrix_normalized_foreground_percent.png")
    np.savetxt(output_dir / "confusion_matrix_counts_with_background.csv", confusion.astype(np.int64), delimiter=",", fmt="%d")
    np.savetxt(output_dir / "confusion_matrix_normalized_foreground_percent.csv", percentages, delimiter=",", fmt="%.2f")
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_recorded_val_map50": checkpoint.get("val_map50"), "checkpoint_restore": restore_info,
        "dataset_yaml": str(args.data.resolve()), "split": args.split, "images": dataset_size,
        "resolution_hw": list(Config.RESOLUTION), "metric_iou_threshold": args.iou_threshold,
        "matrix_confidence_threshold": args.conf_threshold, "metric_decode_confidence_threshold": Config.METRIC_CONF_THRESH,
        "metric_decode_nms_threshold": Config.METRIC_NMS_THRESH, "labels_counts_matrix": labels,
        "labels_normalized_matrix": labels[:-1], "metrics": metrics, "review_summary": review_summary,
        "optical_field_summary": optical_field_summary,
    }
    (output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    if not args.skip_coco:
        require_coco()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.optical_field_samples < 0:
        raise ValueError("--optical-field-samples must be non-negative.")
    for name in ("conf_threshold", "nms_threshold", "iou_threshold"):
        if (value := getattr(args, name)) is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")

    Config.YAML_PATH = str(args.data.resolve())
    Config.initialize()
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.nms_threshold is not None:
        Config.METRIC_NMS_THRESH = args.nms_threshold
    if args.iou_threshold is not None:
        Config.METRIC_IOU_THRESHOLD = args.iou_threshold
    args.conf_threshold = float(Config.CONF_THRESH if args.conf_threshold is None else args.conf_threshold)
    args.iou_threshold = float(Config.METRIC_IOU_THRESHOLD)
    device = torch.device(args.device or Config.DEVICE)
    checkpoint = load_checkpoint_metadata(args.checkpoint, device)
    if not isinstance(checkpoint, dict) or "student_state_dict" not in checkpoint or "detector_state_dict" not in checkpoint:
        raise KeyError("--checkpoint must contain paired student_state_dict and detector_state_dict.")

    student = build_student(Config, checkpoint).to(device)
    detector = build_detector_head(Config, in_channels=1).to(device)
    restore_info = load_student_detector_checkpoint(student, detector, str(args.checkpoint), device)
    if restore_info["detector_loaded"] != restore_info["detector_total"]:
        raise RuntimeError(
            "Detector checkpoint architecture does not match the current shared light head: "
            f"loaded {restore_info['detector_loaded']}/{restore_info['detector_total']} tensors."
        )
    dataset = SLMFeatureDataset(Config, split=args.split)
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset split {args.split!r} is empty.")
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    optical_field_summary = save_optical_field_samples(
        Config, student, dataset, device, output_dir, args.optical_field_samples,
    )
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, collate_fn=slm_collate_fn, **get_dataloader_kwargs(Config))
    detections, targets, source_paths, metrics = evaluate_student(Config, student, detector, dataloader, device)
    if not args.skip_coco:
        metrics["coco"] = evaluate_coco(detections, targets, source_paths, Config, output_dir)
    confusion, class_tp, class_fp, class_fn = compute_detection_confusion_matrix(
        detections, targets, Config.NUM_CLASSES, iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold, image_size=Config.RESOLUTION,
    )
    review_summary = write_review_lists(
        output_dir, source_paths, detections, targets, Config, device,
    )
    save_outputs(
        output_dir, confusion, metrics, checkpoint, restore_info, args, len(dataset), review_summary,
        optical_field_summary,
    )
    print(f"Evaluated {len(dataset)} {args.split} images on {device}.")
    print(f"mAP50: {metrics['map50']:.4f}")
    print(f"mAP50:95 (101-point): {metrics['map50_95']:.4f}")
    print(f"mAP75 (101-point): {metrics['map75']:.4f}")
    print(f"Count matrix (with background): {output_dir / 'confusion_matrix_counts_with_background.png'}")
    print(f"Normalized foreground matrix: {output_dir / 'confusion_matrix_normalized_foreground_percent.png'}")
    print("per-class TP:", class_tp.astype(int))
    print("per-class FP:", class_fp.astype(int))
    print("per-class FN:", class_fn.astype(int))
    print(
        f"Review lists: {review_summary['unrecognized_images']} unrecognized -> "
        f"{output_dir / review_summary['unrecognized_list']}; "
        f"{review_summary['map_error_images']} mAP-error images -> "
        f"{output_dir / review_summary['map_error_list']}"
    )
    if optical_field_summary is not None:
        print(
            f"Optical fields: {optical_field_summary['samples']} sample(s) -> "
            f"{optical_field_summary['directory']}"
        )


if __name__ == "__main__":
    main()
