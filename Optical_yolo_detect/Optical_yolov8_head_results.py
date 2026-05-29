import argparse
import csv
import os
import sys
from datetime import datetime
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

from models.dataset import YOLODataset, identity_collate
from models.geometry import bbox_iou_xywh
from models.runtime import prepare_batch
from models.teacher_guidance import enhance_feature_for_display
from models.yolov8.config_v8 import ConfigYOLOv8Anchor as Config
from models.yolov8.decode_anchor_v8 import decode_detections_anchor_v8
from models.yolov8.head_v8 import TeacherWithDetector, build_detector_head


DEFAULT_CHECKPOINT = ROOT / "output" / "OpticalTeacherYOLO_YOLOv8Head" / "teacher_detector_final.pth"
DEFAULT_OUTPUT_DIR = ROOT / "Optical_yolo_detect" / "Optical_yolov8_head_results"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("teacher_state_dict", "detector_state_dict", "model_state_dict", "state_dict", "model"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def load_matching_state(module, state_dict):
    target_state = module.state_dict()
    compatible = {}
    for raw_key, value in state_dict.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if key in target_state and target_state[key].shape == value.shape:
            compatible[key] = value
    if compatible:
        module.load_state_dict({**target_state, **compatible}, strict=False)
    return len(compatible), len(target_state)


def load_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    info = {
        "teacher_loaded": 0,
        "teacher_total": len(model.teacher.state_dict()),
        "detector_loaded": 0,
        "detector_total": len(model.detector.state_dict()),
        "mode": "unknown",
    }

    if isinstance(checkpoint, dict) and "teacher_state_dict" in checkpoint and "detector_state_dict" in checkpoint:
        info["teacher_loaded"], info["teacher_total"] = load_matching_state(model.teacher, unwrap_state_dict(checkpoint["teacher_state_dict"]))
        info["detector_loaded"], info["detector_total"] = load_matching_state(model.detector, unwrap_state_dict(checkpoint["detector_state_dict"]))
        info["mode"] = "joint_teacher_detector"
        return info

    state_dict = unwrap_state_dict(checkpoint)
    teacher_state = {}
    detector_state = {}
    for raw_key, value in state_dict.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if key.startswith("teacher."):
            teacher_state[key[len("teacher."):]] = value
        elif key.startswith("detector."):
            detector_state[key[len("detector."):]] = value
        else:
            detector_state[key] = value

    if teacher_state:
        info["teacher_loaded"], info["teacher_total"] = load_matching_state(model.teacher, teacher_state)
    if detector_state:
        info["detector_loaded"], info["detector_total"] = load_matching_state(model.detector, detector_state)
    info["mode"] = "mixed_or_detector_only"
    if info["teacher_loaded"] == 0 and info["detector_loaded"] == 0:
        raise RuntimeError(f"No compatible weights found in checkpoint: {checkpoint_path}")
    return info


def compute_average_precision(detections, total_gt):
    if total_gt == 0:
        return None
    if len(detections) == 0:
        return 0.0
    detections = sorted(detections, key=lambda item: item[0], reverse=True)
    tp = np.array([item[1] for item in detections], dtype=np.float32)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


def init_metric_state(num_classes):
    return {
        "metric_storage": {cls_id: [] for cls_id in range(num_classes)},
        "gt_counts": {cls_id: 0 for cls_id in range(num_classes)},
        "det_counts": {cls_id: 0 for cls_id in range(num_classes)},
        "total_tp": 0,
        "total_fp": 0,
        "total_fn": 0,
    }


def update_metric_state(metric_state, sample_detections, sample_targets, img_size, iou_thresh):
    gt_by_class = {}
    for gt in sample_targets:
        if gt.shape[0] < 5 or gt[3] <= 0 or gt[4] <= 0:
            continue
        cls_id = int(gt[0].item())
        gt_box = [
            float(gt[1].item() * img_size),
            float(gt[2].item() * img_size),
            float(gt[3].item() * img_size),
            float(gt[4].item() * img_size),
        ]
        gt_by_class.setdefault(cls_id, []).append(gt_box)
        metric_state["gt_counts"][cls_id] += 1

    matched = {cls_id: set() for cls_id in gt_by_class}
    for det in sorted(sample_detections, key=lambda item: item[4], reverse=True):
        cls_id = int(det[5])
        metric_state["det_counts"][cls_id] += 1
        best_iou, best_gt_idx = 0.0, -1
        for gt_idx, gt_box in enumerate(gt_by_class.get(cls_id, [])):
            if gt_idx in matched.get(cls_id, set()):
                continue
            iou = float(bbox_iou_xywh(torch.tensor(det[:4]).float().unsqueeze(0), torch.tensor(gt_box).float().unsqueeze(0)).item())
            if iou > best_iou:
                best_iou, best_gt_idx = iou, gt_idx
        is_tp = best_iou >= iou_thresh
        metric_state["metric_storage"][cls_id].append((float(det[4]), 1.0 if is_tp else 0.0))
        if is_tp:
            metric_state["total_tp"] += 1
            matched.setdefault(cls_id, set()).add(best_gt_idx)
        else:
            metric_state["total_fp"] += 1

    for cls_id, gt_boxes in gt_by_class.items():
        metric_state["total_fn"] += len(gt_boxes) - len(matched.get(cls_id, set()))


def finalize_metrics(metric_state, class_names):
    total_tp = metric_state["total_tp"]
    total_fp = metric_state["total_fp"]
    total_fn = metric_state["total_fn"]
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    per_class = []
    ap_values = []
    for cls_id in range(len(class_names)):
        ap = compute_average_precision(metric_state["metric_storage"][cls_id], metric_state["gt_counts"][cls_id])
        if ap is not None:
            ap_values.append(ap)
        per_class.append(
            {
                "class_id": cls_id,
                "class_name": class_names.get(cls_id, str(cls_id)),
                "gt_count": metric_state["gt_counts"][cls_id],
                "det_count": metric_state["det_counts"][cls_id],
                "ap50": ap,
            }
        )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "map50": float(np.mean(ap_values)) if ap_values else 0.0,
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_fn": int(total_fn),
        "per_class": per_class,
    }


def save_metrics_csv(path, metrics):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "gt_count", "det_count", "ap50"])
        writer.writeheader()
        for row in metrics["per_class"]:
            row = row.copy()
            row["ap50"] = "" if row["ap50"] is None else f"{row['ap50']:.6f}"
            writer.writerow(row)


def save_visualization(path, image_tensor, teacher_feature, targets, detections, class_names, img_size, title):
    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    img_np = image_tensor.squeeze(0).detach().cpu().numpy()
    feature_np = enhance_feature_for_display(teacher_feature.squeeze().detach().cpu().numpy())
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(feature_np, cmap="magma")
    axes[1].set_title("Teacher feature")
    axes[2].imshow(img_np, cmap="gray")
    axes[2].set_title("Ground Truth")
    axes[3].imshow(img_np, cmap="gray")
    axes[3].set_title("Predictions")

    for target in targets:
        cls_id, cx, cy, w, h = target.tolist()
        x1 = (cx - w / 2) * img_size
        y1 = (cy - h / 2) * img_size
        axes[2].add_patch(plt.Rectangle((x1, y1), w * img_size, h * img_size, fill=False, edgecolor="lime", linewidth=1.8))
        axes[2].text(x1, y1 - 4, class_names.get(int(cls_id), str(int(cls_id))), color="lime", fontsize=8)

    for det in detections:
        cx, cy, w, h, conf, cls_id = det
        x1 = cx - w / 2
        y1 = cy - h / 2
        color = plt.cm.tab20(int(cls_id) / max(len(class_names), 1))
        axes[3].add_patch(plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=1.6))
        axes[3].text(
            x1,
            y1 - 5,
            f"{class_names.get(int(cls_id), str(int(cls_id)))}: {conf:.2f}",
            color=color,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.35),
        )

    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def evaluate(args):
    Config.YAML_PATH = str(args.yaml)
    Config.TEACHER_OUTPUT_DIR = str(args.output_dir)
    Config.IMG_SIZE = args.img_size
    Config.BATCH_SIZE = args.batch_size
    Config.CONF_THRESH = args.conf_thresh
    Config.NMS_THRESH = args.nms_thresh
    Config.MAX_DET = args.max_det
    Config.initialize()
    if args.anchor_yaml is not None:
        Config.ANCHOR_CONFIG_PATH = str(args.anchor_yaml)
        Config.initialize()

    device = torch.device(args.device)
    dataset = YOLODataset(Config, split=args.split)
    if args.max_eval_images > 0:
        dataset.files = dataset.files[: args.max_eval_images]
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=identity_collate)

    teacher_detector = TeacherWithDetector(
        Config,
        detector=build_detector_head(Config, in_channels=1),
    ).to(device)
    checkpoint_info = load_checkpoint(teacher_detector, args.checkpoint, device)
    teacher_detector.eval()

    output_dir = args.output_dir
    vis_dir = output_dir / "visualizations"
    ensure_dir(vis_dir)
    metric_state = init_metric_state(Config.NUM_CLASSES)
    saved_visualizations = []
    evaluated_images = 0

    with torch.no_grad():
        for batch in dataloader:
            images, targets_list = prepare_batch(Config, batch, device)
            teacher_features, predictions = teacher_detector(images, return_feature=True)
            detections = decode_detections_anchor_v8(
                Config,
                predictions,
                conf_thresh=args.conf_thresh,
                nms_thresh=args.nms_thresh,
                max_det=args.max_det,
                img_size=args.img_size,
            )
            for idx in range(images.shape[0]):
                evaluated_images += 1
                targets = targets_list[idx]
                sample_detections = detections[idx]
                update_metric_state(metric_state, sample_detections, targets, args.img_size, args.metric_iou_thresh)
                if args.max_vis_images < 0 or len(saved_visualizations) < args.max_vis_images:
                    source_path = Path(dataset.files[evaluated_images - 1])
                    save_path = vis_dir / f"{evaluated_images:04d}_{source_path.stem}.png"
                    save_visualization(
                        save_path,
                        images[idx].cpu(),
                        teacher_features[idx].cpu(),
                        targets,
                        sample_detections,
                        Config.CLASS_NAMES,
                        args.img_size,
                        source_path.name,
                    )
                    saved_visualizations.append(save_path)

    metrics = finalize_metrics(metric_state, Config.CLASS_NAMES)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"yolov8_head_results_summary_{timestamp}.txt"
    metrics_csv_path = output_dir / f"yolov8_head_per_class_metrics_{timestamp}.csv"
    save_metrics_csv(metrics_csv_path, metrics)
    summary = build_summary(args, metrics, checkpoint_info, evaluated_images, saved_visualizations, metrics_csv_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary)
    print(f"Summary saved to: {summary_path}")
    print(f"Per-class CSV saved to: {metrics_csv_path}")


def build_summary(args, metrics, checkpoint_info, evaluated_images, saved_visualizations, metrics_csv_path):
    lines = [
        "Optical YOLOv8 Head Results Summary",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Model: YOLOv8-style head + YOLOv3 anchor loss",
        f"Checkpoint: {args.checkpoint}",
        f"Dataset yaml: {args.yaml}",
        f"Split: {args.split}",
        f"Output directory: {args.output_dir}",
        f"Image size: {args.img_size}",
        f"Confidence threshold: {args.conf_thresh}",
        f"NMS threshold: {args.nms_thresh}",
        f"Max detections: {args.max_det}",
        f"Metric IoU threshold: {args.metric_iou_thresh}",
        f"Evaluated images: {evaluated_images}",
        f"Saved visualizations: {len(saved_visualizations)}",
        f"Per-class CSV: {metrics_csv_path}",
        (
            "Checkpoint load: "
            f"mode={checkpoint_info['mode']}, "
            f"teacher={checkpoint_info['teacher_loaded']}/{checkpoint_info['teacher_total']}, "
            f"detector={checkpoint_info['detector_loaded']}/{checkpoint_info['detector_total']}"
        ),
        "",
        "[Overall Metrics]",
        f"Precision: {metrics['precision']:.6f}",
        f"Recall: {metrics['recall']:.6f}",
        f"F1: {metrics['f1']:.6f}",
        f"mAP50: {metrics['map50']:.6f}",
        f"TP: {metrics['total_tp']}",
        f"FP: {metrics['total_fp']}",
        f"FN: {metrics['total_fn']}",
        "",
        "[Per-class AP50]",
    ]
    for row in metrics["per_class"]:
        ap_text = "N/A" if row["ap50"] is None else f"{row['ap50']:.6f}"
        lines.append(
            f"- class_id={row['class_id']}, class_name={row['class_name']}, "
            f"gt_count={row['gt_count']}, det_count={row['det_count']}, ap50={ap_text}"
        )
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and visualize YOLOv8-style optical teacher detector checkpoints.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--yaml", type=Path, default=ROOT / "data" / "military" / "data.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anchor-yaml", type=Path, default=ROOT / "output" / "anchor_clustering" / "yolo_anchors.yaml")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--conf-thresh", type=float, default=0.5)
    parser.add_argument("--nms-thresh", type=float, default=0.35)
    parser.add_argument("--max-det", type=int, default=5)
    parser.add_argument("--metric-iou-thresh", type=float, default=0.5)
    parser.add_argument("--max-eval-images", type=int, default=-1)
    parser.add_argument("--max-vis-images", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.yaml = args.yaml.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.anchor_yaml is not None:
        args.anchor_yaml = args.anchor_yaml.resolve()
    ensure_dir(args.output_dir)
    evaluate(args)


if __name__ == "__main__":
    main()
