import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.SLM.config_slm import ConfigSLM as Config
from models.SLM.dataset_slm import SLMFeatureDataset, slm_collate_fn
from models.SLM.optical_layers import OpticalStudent
from models.SLM.utils_slm import load_matching_state
from models.geometry import bbox_iou_xywh
from models.teacher_guidance import enhance_feature_for_display
from models.yolov8.decode_anchor_v8 import decode_detections_anchor_v8
from models.yolov8.head_v8 import YOLOv8AnchorHead
from models.yolov8.metrics_anchor_v8 import compute_average_precision


DEFAULT_OUTPUT_DIR = ROOT / "Optical_yolo_detect" / "Optical_SLM_yolov8_head_model_results"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("student_state_dict", "detector_state_dict", "model_state_dict", "state_dict", "model"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def extract_student_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("student_state_dict", "model_state_dict", "state_dict", "model"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def extract_detector_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("detector_state_dict", "model_state_dict", "state_dict", "model"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def safe_torch_load(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_student_from_checkpoint(student, checkpoint, checkpoint_path):
    loaded, total = load_matching_state(student, extract_student_state_dict(checkpoint), prefixes=("student.",))
    norm_enabled = checkpoint.get("student_enable_norm") if isinstance(checkpoint, dict) else None
    if norm_enabled is not None:
        student.enable_norm = bool(norm_enabled)
    if loaded == 0:
        raise RuntimeError(f"No compatible student weights found: {checkpoint_path}")
    return {
        "loaded": loaded,
        "total": total,
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "loss": checkpoint.get("loss") if isinstance(checkpoint, dict) else None,
        "train_loss": checkpoint.get("train_loss") if isinstance(checkpoint, dict) else None,
        "val_loss": checkpoint.get("val_loss") if isinstance(checkpoint, dict) else None,
        "val_map50": checkpoint.get("val_map50") if isinstance(checkpoint, dict) else None,
        "selection_metric": checkpoint.get("selection_metric") if isinstance(checkpoint, dict) else None,
        "paired_with_detector_best": checkpoint.get("paired_with_detector_best") if isinstance(checkpoint, dict) else None,
    }


def load_student(student, checkpoint_path, device):
    checkpoint = safe_torch_load(checkpoint_path, device)
    return load_student_from_checkpoint(student, checkpoint, checkpoint_path)


def load_detector(detector, checkpoint_path, device):
    checkpoint = safe_torch_load(checkpoint_path, device)
    loaded, total = load_matching_state(detector, extract_detector_state_dict(checkpoint), prefixes=("detector.",))
    if loaded == 0:
        raise RuntimeError(f"No compatible detector weights found: {checkpoint_path}")
    return {
        "loaded": loaded,
        "total": total,
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "loss": checkpoint.get("loss") if isinstance(checkpoint, dict) else None,
        "train_loss": checkpoint.get("train_loss") if isinstance(checkpoint, dict) else None,
        "val_loss": checkpoint.get("val_loss") if isinstance(checkpoint, dict) else None,
        "val_map50": checkpoint.get("val_map50") if isinstance(checkpoint, dict) else None,
        "selection_metric": checkpoint.get("selection_metric") if isinstance(checkpoint, dict) else None,
        "checkpoint": checkpoint,
    }


def update_metrics(state, detections, targets, img_size, iou_thresh):
    gt_by_class = {}
    for gt in targets:
        cls_id = int(gt[0].item())
        box = [float(gt[1] * img_size), float(gt[2] * img_size), float(gt[3] * img_size), float(gt[4] * img_size)]
        gt_by_class.setdefault(cls_id, []).append(box)
        state["gt_counts"][cls_id] += 1
        state["gt"] += 1
    matched = {cls_id: set() for cls_id in gt_by_class}
    for det in sorted(detections, key=lambda item: item[4], reverse=True):
        cls_id = int(det[5])
        best_iou, best_idx = 0.0, -1
        for idx, gt_box in enumerate(gt_by_class.get(cls_id, [])):
            if idx in matched.get(cls_id, set()):
                continue
            iou = float(bbox_iou_xywh(torch.tensor(det[:4]).float().unsqueeze(0), torch.tensor(gt_box).float().unsqueeze(0)).item())
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        is_tp = best_iou >= iou_thresh
        state["ap_storage"][cls_id].append((float(det[4]), 1.0 if is_tp else 0.0))
        if is_tp:
            state["tp"] += 1
            matched.setdefault(cls_id, set()).add(best_idx)
        else:
            state["fp"] += 1
    for cls_id, boxes in gt_by_class.items():
        state["fn"] += len(boxes) - len(matched.get(cls_id, set()))


def save_vis(path, gray, optical_feature, targets, detections):
    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    img_np = gray.squeeze().detach().cpu().numpy()
    feature_np = enhance_feature_for_display(optical_feature.squeeze().detach().cpu().numpy())
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(feature_np, cmap="magma")
    axes[1].set_title("Optical feature")
    axes[2].imshow(img_np, cmap="gray")
    axes[2].set_title("Ground Truth")
    axes[3].imshow(img_np, cmap="gray")
    axes[3].set_title("Predictions")
    for target in targets:
        cls_id, cx, cy, w, h = target.tolist()
        x1 = (cx - w / 2) * Config.IMG_SIZE
        y1 = (cy - h / 2) * Config.IMG_SIZE
        axes[2].add_patch(
            plt.Rectangle((x1, y1), w * Config.IMG_SIZE, h * Config.IMG_SIZE, fill=False, edgecolor="lime", linewidth=1.8, linestyle="--")
        )
        axes[2].text(
            x1,
            y1 + 10,
            str(Config.CLASS_NAMES[int(cls_id)]),
            color="lime",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="black", alpha=0.35, edgecolor="none"),
        )
    for det in detections:
        cx, cy, w, h, conf, cls_id = det
        x1 = cx - w / 2
        y1 = cy - h / 2
        color = plt.cm.tab20(int(cls_id) / max(Config.NUM_CLASSES, 1))
        axes[3].add_patch(plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=1.6))
        axes[3].text(
            x1,
            y1 + 10,
            f"{Config.CLASS_NAMES[int(cls_id)]} {conf:.2f}",
            color=color,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="black", alpha=0.35, edgecolor="none"),
        )
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run(args):
    Config.YAML_PATH = str(args.yaml)
    Config.OUTPUT_DIR = str(args.output_dir)
    Config.IMG_SIZE = args.img_size
    Config.BATCH_SIZE = args.batch_size
    Config.CONF_THRESH = args.conf_thresh
    Config.NMS_THRESH = args.nms_thresh
    Config.MAX_DET = args.max_det
    Config.initialize()
    device = torch.device(args.device)
    student = OpticalStudent(Config).to(device)
    detector = YOLOv8AnchorHead(Config, in_channels=1, out_channels=Config.get_detector_output_channels()).to(device)
    detector_info = load_detector(detector, args.detector_checkpoint, device)
    detector_checkpoint = detector_info.pop("checkpoint")
    paired_student_available = isinstance(detector_checkpoint, dict) and "student_state_dict" in detector_checkpoint
    if args.prefer_paired_student and paired_student_available:
        student_info = load_student_from_checkpoint(student, detector_checkpoint, args.detector_checkpoint)
        student_source = "detector_checkpoint.student_state_dict"
        student_source_path = args.detector_checkpoint
        student_source_note = "Using paired student from detector_best.pth; --student-checkpoint was ignored."
    else:
        student_info = load_student(student, args.student_checkpoint, device)
        student_source = "student_checkpoint"
        student_source_path = args.student_checkpoint
        if args.prefer_paired_student:
            student_source_note = "No paired student_state_dict found in detector checkpoint; fell back to --student-checkpoint."
        else:
            student_source_note = "--no-prefer-paired-student was set; using --student-checkpoint."
    student.eval()
    detector.eval()

    dataset = SLMFeatureDataset(Config, split=args.split)
    if args.max_eval_images > 0:
        dataset.entries = dataset.entries[: args.max_eval_images]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=slm_collate_fn)
    vis_dir = args.output_dir / "visualizations"
    ensure_dir(vis_dir)
    state = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "gt": 0,
        "ap_storage": {cls_id: [] for cls_id in range(Config.NUM_CLASSES)},
        "gt_counts": {cls_id: 0 for cls_id in range(Config.NUM_CLASSES)},
    }
    saved = 0
    seen = 0
    with torch.no_grad():
        for batch in loader:
            gray = batch["gray_tensor"].to(device)
            if Config.ENABLE_CHANNELS_LAST and torch.cuda.is_available():
                gray = gray.contiguous(memory_format=torch.channels_last)
            student_features = student(gray)
            preds = detector(student_features)
            detections = decode_detections_anchor_v8(Config, preds, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, max_det=args.max_det)
            for idx in range(gray.shape[0]):
                seen += 1
                targets = batch["targets"][idx]
                update_metrics(state, detections[idx], targets, Config.IMG_SIZE, args.metric_iou_thresh)
                if args.max_vis_images < 0 or saved < args.max_vis_images:
                    stem = Path(batch["image_paths"][idx]).stem
                    save_vis(
                        vis_dir / f"{seen:04d}_{stem}.png",
                        gray[idx].cpu(),
                        student_features[idx].cpu(),
                        targets,
                        detections[idx],
                    )
                    saved += 1
    precision = state["tp"] / (state["tp"] + state["fp"] + 1e-6)
    recall = state["tp"] / (state["tp"] + state["fn"] + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    ap_values = []
    for cls_id in range(Config.NUM_CLASSES):
        ap = compute_average_precision(state["ap_storage"][cls_id], state["gt_counts"][cls_id])
        if ap is not None:
            ap_values.append(ap)
    map50 = float(np.mean(ap_values)) if ap_values else 0.0
    summary = (
        "Optical SLM YOLOv8-head simulation summary\n"
        f"Detector checkpoint: {args.detector_checkpoint}\n"
        f"Requested student checkpoint: {args.student_checkpoint}\n"
        f"Actual student source: {student_source}\n"
        f"Actual student source path: {student_source_path}\n"
        f"Student source note: {student_source_note}\n"
        f"Student loaded: {student_info['loaded']}/{student_info['total']}\n"
        f"Student epoch/loss/train_loss/val_loss/val_mAP50: "
        f"{student_info['epoch']}/{student_info['loss']}/{student_info['train_loss']}/{student_info['val_loss']}/{student_info['val_map50']}\n"
        f"Student selection metric / paired flag: {student_info['selection_metric']}/{student_info['paired_with_detector_best']}\n"
        f"Detector loaded: {detector_info['loaded']}/{detector_info['total']}\n"
        f"Detector epoch/loss/train_loss/val_loss/val_mAP50: "
        f"{detector_info['epoch']}/{detector_info['loss']}/{detector_info['train_loss']}/{detector_info['val_loss']}/{detector_info['val_map50']}\n"
        f"Detector selection metric: {detector_info['selection_metric']}\n"
        f"Paired student in detector checkpoint: {paired_student_available}\n"
        f"Images: {seen}\n"
        f"Precision: {precision:.6f}\n"
        f"Recall: {recall:.6f}\n"
        f"F1: {f1:.6f}\n"
        f"mAP: {map50:.6f}\n"
        f"mAP50: {map50:.6f}\n"
        f"TP: {state['tp']} FP: {state['fp']} FN: {state['fn']}\n"
        f"Visualizations: {vis_dir}\n"
    )
    summary_path = args.output_dir / "slm_yolov8_head_simulation_summary.txt"
    ensure_dir(args.output_dir)
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate Optical SLM student + YOLOv8-head detector.")
    parser.add_argument("--student-checkpoint", type=Path, default=ROOT / "output" / "OpticalSLM_YOLOv8Head_student" / "optical_student_best.pth")
    parser.add_argument("--detector-checkpoint", type=Path, default=ROOT / "output" / "OpticalSLM_YOLOv8Head_student" / "detector_best.pth")
    parser.add_argument("--yaml", type=Path, default=ROOT / "data" / "military" / "data.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--conf-thresh", type=float, default=0.5)
    parser.add_argument("--nms-thresh", type=float, default=0.35)
    parser.add_argument("--max-det", type=int, default=5)
    parser.add_argument("--metric-iou-thresh", type=float, default=0.5)
    parser.add_argument("--max-eval-images", type=int, default=-1)
    parser.add_argument("--max-vis-images", type=int, default=20)
    parser.add_argument("--prefer-paired-student", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.student_checkpoint = args.student_checkpoint.resolve()
    args.detector_checkpoint = args.detector_checkpoint.resolve()
    args.yaml = args.yaml.resolve()
    args.output_dir = args.output_dir.resolve()
    ensure_dir(args.output_dir)
    run(args)


if __name__ == "__main__":
    main()
