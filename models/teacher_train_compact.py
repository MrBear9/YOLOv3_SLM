"""Teacher + Compact Detector joint training loop.

RGB image → Teacher (v1/v2/v3) → 1ch feature → CompactOpticalDetector → center-point predictions.

This module is selected automatically when DETECTOR_HEAD_TYPE is "compact" or
"center_detect" in ConfigYOLOv8Anchor.  It shares the same teacher backbone
and YOLODataset data pipeline as teacher_train_loop.py, but replaces the
anchor-based YOLOv8 head with the anchor-free CompactOpticalDetector and
uses CenterDetectionLoss.
"""

import os
import warnings
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.class_display import class_name_for_id
from models.compact_detect.factory import build_compact_criterion, build_compact_decode_fn, get_compact_loss_keys
from models.compact_detect.metrics import _distributed_merge_metrics, compute_average_precision
from models.yolov8.head_v8 import build_detector_head
from models.dataset import YOLODataset, build_class_balanced_train_sampler, identity_collate
from models.geometry import bbox_iou_matrix_xywh
from models.monitoring import (
    snapshot_parameters,
    write_confusion_matrix,
    write_gradient_monitoring,
    write_parameter_monitoring,
)
from models.runtime import (
    cleanup_distributed,
    get_dataloader_kwargs,
    get_runtime_device,
    init_distributed_mode,
    init_epoch_log_table,
    log_epoch_table_row,
    log_to_file,
    prepare_batch,
    prepare_conv_tensor,
)
from models.teacher import build_teacher
from models.training_utils import (
    create_tensorboard_writer,
    initialize_teacher_weights,
)
from models.yolov8.config_v8 import ConfigYOLOv8Anchor as Config

warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _module_param_count(module, trainable_only=False):
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def _write_compact_tensorboard_model_summary(writer, teacher, detector):
    """Log teacher + compact detector architecture to TensorBoard."""
    if writer is None:
        return
    teacher_lines = [
        f"- teacher.{name}: {_module_param_count(child):,}"
        for name, child in teacher.named_children()
    ]
    detector_lines = [
        f"- detector.{name}: {_module_param_count(child):,}"
        for name, child in detector.named_children()
    ]
    flow_text = "\n".join([
        "Teacher + Compact detection training route",
        "",
        "RGB image → Teacher → 1ch feature → CompactOpticalDetector → center-point predictions",
        "",
        "Training output:",
        "heatmap + box-size + center-offset heads (anchor-free)",
        "",
        f"Teacher architecture: {Config.TEACHER_ARCH}",
        f"Detector head type: compact (center-point, anchor-free)",
        f"Input tensor: (B, 1, {Config.RESOLUTION[0]}, {Config.RESOLUTION[1]})",
        f"Output stride: 4",
        f"AMP enabled: {Config.ENABLE_AMP}",
    ])
    param_text = "\n".join([
        f"Teacher parameters: {_module_param_count(teacher):,}",
        f"  trainable: {_module_param_count(teacher, trainable_only=True):,}",
        f"Compact detector parameters: {_module_param_count(detector):,}",
        f"  trainable: {_module_param_count(detector, trainable_only=True):,}",
        "",
        "Teacher module parameters:",
        *teacher_lines,
        "",
        "Detector module parameters:",
        *detector_lines,
    ])
    writer.add_text("Model/data_flow", flow_text, 0)
    writer.add_text("Model/parameters", param_text, 0)


def _collect_compact_val_detections(teacher, detector, val_loader, device):
    """Collect compact detections and targets for confusion matrix.

    Returns (all_detections, all_targets) suitable for
    ``write_confusion_matrix``.
    """
    teacher.eval()
    detector.eval()
    all_dets = []
    all_targets = []
    decode_fn = build_compact_decode_fn(Config)
    amp_enabled = bool(getattr(Config, "ENABLE_AMP", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(Config.AMP_DTYPE).lower() in {"bf16", "bfloat16"} else torch.float16
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting detections for CM", leave=False, disable=True):
            batch_images, batch_targets = prepare_batch(Config, batch, device)
            with amp_ctx:
                features = teacher(prepare_conv_tensor(Config, batch_images))
                pred = detector(features)
            detections = decode_fn(
                Config, pred,
                conf_thresh=getattr(Config, "COMPACT_METRIC_CONF_THRESH", Config.COMPACT_CONF_THRESH),
                nms_thresh=getattr(Config, "COMPACT_METRIC_NMS_THRESH", Config.COMPACT_NMS_THRESH),
                max_det=getattr(Config, "COMPACT_METRIC_MAX_DET", Config.COMPACT_MAX_DET),
                pre_nms_topk=getattr(Config, "COMPACT_METRIC_PRE_NMS_TOPK", None),
            )
            for i, dets in enumerate(detections):
                all_dets.append(np.array(dets) if not isinstance(dets, np.ndarray) else dets)
                all_targets.append(
                    batch_targets[i].cpu().numpy()
                    if isinstance(batch_targets[i], torch.Tensor)
                    else batch_targets[i]
                )
    return all_dets, all_targets


def _write_compact_tensorboard_predictions(writer, step, teacher, detector, dataset, device, max_images=4):
    """Draw compact detector predictions on fixed images and log to TensorBoard.

    Uses a fixed seed so the same images are shown every interval, making it
    easy to track how predictions evolve across epochs.
    """
    if writer is None or dataset is None or len(dataset) == 0:
        return
    teacher.eval()
    detector.eval()

    max_images = min(max_images, len(dataset), int(getattr(Config, "VIS_MAX_IMAGES", 4)))
    generator = torch.Generator().manual_seed(int(getattr(Config, "VIS_SEED", 20260506)))
    indices = torch.randperm(len(dataset), generator=generator)[:max_images].tolist()

    amp_enabled = bool(getattr(Config, "ENABLE_AMP", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(Config.AMP_DTYPE).lower() in {"bf16", "bfloat16"} else torch.float16
    fp32_ctx = torch.amp.autocast(device_type="cuda", enabled=False) if device.type == "cuda" else nullcontext()

    rendered = []
    with torch.no_grad():
        for idx in indices:
            img_tensor, targets = dataset[idx]
            # img_tensor: (C, H, W) in [0, 1] — C=3 (RGB) or C=1 (gray)
            img_arr = img_tensor.cpu().numpy()
            if img_arr.ndim == 3 and img_arr.shape[0] == 1:
                # Grayscale (1, H, W) → (H, W)
                img_arr = (img_arr.squeeze(0) * 255).clip(0, 255).astype(np.uint8)
                canvas = Image.fromarray(img_arr, mode="L").convert("RGB")
            elif img_arr.ndim == 3 and img_arr.shape[0] == 3:
                # RGB (3, H, W) → (H, W, 3)
                img_arr = (img_arr.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                canvas = Image.fromarray(img_arr).convert("RGB")
            else:
                # Fallback: handle unexpected shapes
                img_arr = img_arr.squeeze()
                if img_arr.ndim == 2:
                    img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
                    canvas = Image.fromarray(img_arr, mode="L").convert("RGB")
                else:
                    image_h, image_w = Config.RESOLUTION
                    canvas = Image.new("RGB", (image_w, image_h), (128, 128, 128))
            draw = ImageDraw.Draw(canvas)

            # Inference
            inp = img_tensor.unsqueeze(0).to(device)
            with fp32_ctx:
                features = teacher(prepare_conv_tensor(Config, inp))
                pred = detector(features)
            decode_vis_fn = build_compact_decode_fn(Config)
            dets_list = decode_vis_fn(
                Config, pred,
                conf_thresh=getattr(Config, "COMPACT_CONF_THRESH", 0.30),
                nms_thresh=getattr(Config, "COMPACT_NMS_THRESH", 0.45),
                max_det=getattr(Config, "COMPACT_MAX_DET", 100),
            )

            # Draw GT boxes (green)
            for gt in targets:
                if gt.numel() < 5 or gt[3] <= 0 or gt[4] <= 0:
                    continue
                cls_id = int(gt[0].item())
                cx, cy, w, h = gt[1].item(), gt[2].item(), gt[3].item(), gt[4].item()
                x1 = int((cx - w / 2) * canvas.width)
                y1 = int((cy - h / 2) * canvas.height)
                x2 = int((cx + w / 2) * canvas.width)
                y2 = int((cy + h / 2) * canvas.height)
                draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=2)
                cls_name = class_name_for_id(Config.CLASS_NAMES, cls_id, str(cls_id))
                draw.text((x1, max(y1 - 10, 0)), cls_name, fill=(0, 180, 0))

            # Draw predictions (red)
            for det in dets_list[0]:
                cx, cy, bw, bh = float(det[0]), float(det[1]), float(det[2]), float(det[3])
                score = det[4]
                cls_id = int(det[5])
                x1 = max(0, int(cx - bw / 2))
                y1 = max(0, int(cy - bh / 2))
                x2 = min(canvas.width - 1, int(cx + bw / 2))
                y2 = min(canvas.height - 1, int(cy + bh / 2))
                if x2 <= x1 or y2 <= y1:
                    continue
                draw.rectangle([x1, y1, x2, y2], outline=(220, 40, 40), width=2)
                cls_name = class_name_for_id(Config.CLASS_NAMES, cls_id, str(cls_id))
                draw.text((x1, y2 + 2), f"{cls_name} {score:.2f}", fill=(220, 40, 40))

            canvas = canvas.resize((640, 640), Image.LANCZOS)
            rendered.append(np.array(canvas).transpose(2, 0, 1))  # (3, H, W)

    grid = np.stack(rendered, axis=0)  # (N, 3, H, W)
    writer.add_images("Predictions/compact", grid, step, dataformats="NCHW")


def _evaluate_teacher_compact(teacher, detector, val_loader, criterion, device):
    """Run validation for teacher+compact detector with YOLODataset format.

    Returns (losses_dict, metrics_dict) with keys:
      losses:  {"total", "heatmap", "wh", "offset"}
      metrics: {"precision", "recall", "f1", "map50"}
    """
    teacher.eval()
    detector.eval()

    loss_keys = [k for k in get_compact_loss_keys(Config) if k != "feature_total"]
    decode_val_fn = build_compact_decode_fn(Config)

    metric_storage = {cls_id: [] for cls_id in range(Config.NUM_CLASSES)}
    gt_counts = {cls_id: 0 for cls_id in range(Config.NUM_CLASSES)}
    loss_totals = {"total": 0.0}
    for k in loss_keys:
        loss_totals[k] = 0.0
    total_tp = total_fp = total_fn = 0
    total_tp_op = total_fp_op = total_fn_op = 0
    num_batches = 0
    is_main = not dist.is_initialized() or dist.get_rank() == 0

    amp_enabled = bool(getattr(Config, "ENABLE_AMP", True)) and device.type == "cuda"
    amp_dtype_name = str(getattr(Config, "AMP_DTYPE", "float16")).strip().lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
        if device.type == "cuda"
        else nullcontext()
    )
    metric_iou = float(getattr(Config, "COMPACT_METRIC_IOU_THRESHOLD", 0.5))
    op_conf = float(getattr(Config, "COMPACT_CONF_THRESH", 0.30))

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating (compact)", leave=False, disable=not is_main):
            batch_images, batch_targets = prepare_batch(Config, batch, device)
            with amp_ctx:
                features = teacher(prepare_conv_tensor(Config, batch_images))
                pred = detector(features)
                loss, stats = criterion(pred, batch_targets)
            loss_totals["total"] += float(loss.detach().item())
            for k in loss_keys:
                loss_totals[k] += float(stats.get(k, 0.0))
            num_batches += 1

            detections = decode_val_fn(
                Config, pred,
                conf_thresh=getattr(Config, "COMPACT_METRIC_CONF_THRESH", Config.COMPACT_CONF_THRESH),
                nms_thresh=getattr(Config, "COMPACT_METRIC_NMS_THRESH", Config.COMPACT_NMS_THRESH),
                max_det=getattr(Config, "COMPACT_METRIC_MAX_DET", Config.COMPACT_MAX_DET),
                pre_nms_topk=getattr(Config, "COMPACT_METRIC_PRE_NMS_TOPK", None),
            )

            for sample_idx, sample_dets in enumerate(detections):
                gt_by_class = {}
                target_tensor = batch_targets[sample_idx]
                for gt_idx in range(target_tensor.shape[0]):
                    gt = target_tensor[gt_idx]
                    if gt.shape[0] < 5 or gt[3] <= 0 or gt[4] <= 0:
                        continue
                    cls_id = int(gt[0].item())
                    image_h, image_w = Config.RESOLUTION
                    gt_box = [
                        float(gt[1].item() * image_w),
                        float(gt[2].item() * image_h),
                        float(gt[3].item() * image_w),
                        float(gt[4].item() * image_h),
                    ]
                    gt_by_class.setdefault(cls_id, []).append(gt_box)
                    gt_counts[cls_id] += 1

                dets_by_class = {}
                for det in sample_dets:
                    dets_by_class.setdefault(int(det[5]), []).append(det)

                # Prediction-only classes are false positives too; excluding them
                # inflates AP and makes it disagree with the confusion matrix.
                for cls_id in set(gt_by_class) | set(dets_by_class):
                    gt_boxes_list = gt_by_class.get(cls_id, [])
                    dets = sorted(dets_by_class.get(cls_id, []), key=lambda d: d[4], reverse=True)
                    if not gt_boxes_list:
                        metric_storage[cls_id].extend((float(det[4]), 0.0) for det in dets)
                        total_fp += len(dets)
                        continue
                    if not dets:
                        total_fn += len(gt_boxes_list)
                        continue
                    gt_boxes = torch.tensor(gt_boxes_list, dtype=torch.float32, device=device)
                    det_boxes = torch.from_numpy(np.stack([d[:4] for d in dets])).to(device=device, dtype=torch.float32)
                    iou_matrix = bbox_iou_matrix_xywh(det_boxes, gt_boxes)
                    matched_gt = set()
                    for det_idx, det in enumerate(dets):
                        ious = iou_matrix[det_idx].clone()
                        for matched_idx in matched_gt:
                            ious[matched_idx] = -1.0
                        best_iou, best_gt_idx = ious.max(dim=0)
                        is_tp = float(best_iou.item()) >= metric_iou
                        metric_storage[cls_id].append((float(det[4]), 1.0 if is_tp else 0.0))
                        if is_tp:
                            total_tp += 1
                            matched_gt.add(int(best_gt_idx.item()))
                        else:
                            total_fp += 1
                    total_fn += len(gt_boxes_list) - len(matched_gt)

                for cls_id in set(gt_by_class) | set(dets_by_class):
                    gt_boxes_list = gt_by_class.get(cls_id, [])
                    op_dets = sorted(
                        (det for det in dets_by_class.get(cls_id, []) if float(det[4]) >= op_conf),
                        key=lambda det: det[4], reverse=True,
                    )
                    if not gt_boxes_list:
                        total_fp_op += len(op_dets)
                        continue
                    if not op_dets:
                        total_fn_op += len(gt_boxes_list)
                        continue
                    gt_boxes = torch.tensor(gt_boxes_list, dtype=torch.float32, device=device)
                    det_boxes = torch.from_numpy(np.stack([det[:4] for det in op_dets])).to(device=device, dtype=torch.float32)
                    iou_matrix = bbox_iou_matrix_xywh(det_boxes, gt_boxes)
                    matched_gt = set()
                    for det_idx in range(len(op_dets)):
                        ious = iou_matrix[det_idx].clone()
                        for matched_idx in matched_gt:
                            ious[matched_idx] = -1.0
                        best_iou, best_gt_idx = ious.max(dim=0)
                        if float(best_iou.item()) >= metric_iou:
                            total_tp_op += 1
                            matched_gt.add(int(best_gt_idx.item()))
                        else:
                            total_fp_op += 1
                    total_fn_op += len(gt_boxes_list) - len(matched_gt)

    num_batches = max(num_batches, 1)

    # Distributed merge (reuse compact metrics helper)
    if dist.is_initialized():
        payload = {"metric_storage": metric_storage, "gt_counts": gt_counts}
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, payload)
        merged_storage = {cls_id: [] for cls_id in metric_storage.keys()}
        merged_gt_counts = {cls_id: 0 for cls_id in gt_counts.keys()}
        for item in gathered:
            for cls_id, values in item["metric_storage"].items():
                merged_storage[cls_id].extend(values)
            for cls_id, value in item["gt_counts"].items():
                merged_gt_counts[cls_id] += int(value)
        metric_storage = merged_storage
        gt_counts = merged_gt_counts

        stat_parts = [loss_totals.get("total", 0.0)]
        for k in loss_keys:
            stat_parts.append(loss_totals.get(k, 0.0))
        stat_parts.extend([
            float(total_tp), float(total_fp), float(total_fn),
            float(total_tp_op), float(total_fp_op), float(total_fn_op),
            float(num_batches),
        ])
        stat_tensor = torch.tensor(stat_parts, dtype=torch.float64, device=device)
        dist.all_reduce(stat_tensor, op=dist.ReduceOp.SUM)
        loss_totals = {"total": float(stat_tensor[0].item())}
        for i, k in enumerate(loss_keys):
            loss_totals[k] = float(stat_tensor[1 + i].item())
        base = 1 + len(loss_keys)
        total_tp = int(stat_tensor[base + 0].item())
        total_fp = int(stat_tensor[base + 1].item())
        total_fn = int(stat_tensor[base + 2].item())
        total_tp_op = int(stat_tensor[base + 3].item())
        total_fp_op = int(stat_tensor[base + 4].item())
        total_fn_op = int(stat_tensor[base + 5].item())
        num_batches = int(max(stat_tensor[base + 6].item(), 1.0))

    losses = {key: value / num_batches for key, value in loss_totals.items()}
    precision_metric = total_tp / (total_tp + total_fp + 1e-6)
    recall_metric = total_tp / (total_tp + total_fn + 1e-6)
    f1_metric = 2.0 * precision_metric * recall_metric / (precision_metric + recall_metric + 1e-6)
    precision = total_tp_op / (total_tp_op + total_fp_op + 1e-6)
    recall = total_tp_op / (total_tp_op + total_fn_op + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)

    ap_values = []
    for cls_id in range(Config.NUM_CLASSES):
        ap = compute_average_precision(metric_storage[cls_id], gt_counts[cls_id])
        if ap is not None:
            ap_values.append(ap)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "map50": float(np.mean(ap_values)) if ap_values else 0.0,
        "precision_metric": float(precision_metric),
        "recall_metric": float(recall_metric),
        "f1_metric": float(f1_metric),
    }
    return losses, metrics


def _build_compact_optimizer(teacher, detector, teacher_lr=0.0):
    """Build optimizer with separate param groups for teacher and compact detector.

    Teacher params use *teacher_lr*; detector params use COMPACT_DETECTOR_LR.
    Teacher bias/norm (dim < 2) params get weight_decay=0.
    """
    teacher_trainable = [p for p in teacher.parameters() if p.requires_grad]
    detector_params = [p for p in detector.parameters() if p.requires_grad]

    teacher_decay = []
    teacher_no_decay = []
    for p in teacher_trainable:
        if p.dim() >= 2:
            teacher_decay.append(p)
        else:
            teacher_no_decay.append(p)

    compact_lr = float(getattr(Config, "COMPACT_DETECTOR_LR", 3e-4))
    compact_wd = float(getattr(Config, "COMPACT_WEIGHT_DECAY", 3e-5))

    param_groups = []
    if teacher_decay:
        param_groups.append({"params": teacher_decay, "lr": teacher_lr, "weight_decay": Config.WEIGHT_DECAY})
    if teacher_no_decay:
        param_groups.append({"params": teacher_no_decay, "lr": teacher_lr, "weight_decay": 0.0})
    if detector_params:
        param_groups.append({"params": detector_params, "lr": compact_lr, "weight_decay": compact_wd})

    return torch.optim.AdamW(param_groups, lr=compact_lr, weight_decay=compact_wd)


def _save_compact_checkpoint(path, teacher, detector, epoch, loss, metrics=None):
    """Save teacher + compact detector checkpoint."""
    torch.save(
        {
            "teacher_state_dict": teacher.state_dict(),
            "detector_state_dict": detector.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "val_map50": metrics["map50"] if metrics is not None else None,
            "teacher_arch": Config.TEACHER_ARCH,
            "head_type": "compact_center_detect",
        },
        path,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main training entry point
# ═══════════════════════════════════════════════════════════════════════════

def train():
    """Train teacher (v1/v2/v3) + CompactOpticalDetector jointly."""
    Config.initialize()
    local_rank, use_ddp = init_distributed_mode(Config)
    is_main = local_rank == 0

    # ── Initialisation (compact-specific, no temporary model builds) ────
    Config.initialize()
    Config.print_config()
    from models.runtime import init_log_file
    init_log_file(Config)
    log_to_file(Config, f"Log file path: {Config.LOG_FILE}")
    log_to_file(Config, f"Output dir: {Config.TEACHER_OUTPUT_DIR}")
    log_to_file(Config, f"Class info: {Config.CLASS_NAMES}, Num classes: {Config.NUM_CLASSES}")
    log_to_file(Config, "=" * 80)
    log_to_file(Config, "Teacher + CompactOpticalDetector configuration")
    log_to_file(Config, "=" * 80)
    log_to_file(Config, f"Dataset: {Config.YAML_PATH}")
    log_to_file(Config, f"Resolution (H, W) / batch / epochs: {Config.RESOLUTION} / {Config.BATCH_SIZE} / {Config.EPOCHS}")
    log_to_file(Config, f"Teacher arch: {Config.TEACHER_ARCH}, detector: compact (center-point)")
    log_to_file(Config, f"Compact: base_ch={Config.COMPACT_BASE_CH}, head_ch={Config.COMPACT_HEAD_CH}, dilations={Config.COMPACT_DILATIONS}")
    log_to_file(Config, f"Loss weights: heatmap={Config.HEATMAP_LOSS_WEIGHT}, wh={Config.WH_LOSS_WEIGHT}, offset={Config.OFFSET_LOSS_WEIGHT}, obj={getattr(Config, 'OBJ_LOSS_WEIGHT', 'N/A')}, cls={getattr(Config, 'CLS_LOSS_WEIGHT', 'N/A')}")
    log_to_file(Config, f"Strides: {Config.ANCHOR_FREE_STRIDES}")
    log_to_file(
        Config,
        f"LR stage1 teacher/detector: {Config.PHASE1_TEACHER_LR}/{Config.COMPACT_DETECTOR_LR}  "
        f"stage2: {Config.PHASE2_TEACHER_LR}/{Config.COMPACT_DETECTOR_LR}",
    )
    log_to_file(Config, f"Detection conf/nms/max_det: {Config.COMPACT_CONF_THRESH}/{Config.COMPACT_NMS_THRESH}/{Config.COMPACT_MAX_DET}")
    log_to_file(Config, f"Metric conf/nms/max_det: {Config.COMPACT_METRIC_CONF_THRESH}/{Config.COMPACT_METRIC_NMS_THRESH}/{Config.COMPACT_METRIC_MAX_DET}")

    device = get_runtime_device(Config)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = Config.ENABLE_CUDNN_BENCHMARK
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = Config.ENABLE_TF32
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = Config.ENABLE_TF32

    if use_ddp:
        world_size = torch.distributed.get_world_size()
        log_to_file(Config, f"Using device: {device} (local_rank={local_rank}, world_size={world_size}, use_ddp={use_ddp})")
    else:
        log_to_file(Config, f"Using device: {device} (local_rank={local_rank}, use_ddp={use_ddp})")

    amp_enabled = bool(getattr(Config, "ENABLE_AMP", True)) and device.type == "cuda"
    amp_dtype_name = str(getattr(Config, "AMP_DTYPE", "float16")).strip().lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    amp_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    amp_context = (
        lambda: torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
        if device.type == "cuda"
        else nullcontext
    )
    if amp_enabled:
        log_to_file(Config, f"Using AMP autocast dtype={amp_dtype}")

    # ── Teacher ──────────────────────────────────────────────────────────
    teacher = build_teacher(Config)
    loaded_teacher, teacher_message = initialize_teacher_weights(Config, teacher, device)
    teacher = teacher.to(device)
    if Config.ENABLE_CHANNELS_LAST and device.type == "cuda":
        teacher = teacher.to(memory_format=torch.channels_last)
    log_to_file(Config, teacher_message)
    freeze_teacher = Config.FREEZE_TEACHER and loaded_teacher
    for p in teacher.parameters():
        p.requires_grad = not freeze_teacher
    log_to_file(Config, f"Teacher status: {'frozen' if freeze_teacher else 'trainable'}")

    # ── Compact detector ─────────────────────────────────────────────────
    detector = build_detector_head(Config, in_channels=1).to(device)
    log_to_file(
        Config,
        f"CompactOpticalDetector: base_ch={Config.COMPACT_BASE_CH}, "
        f"head_ch={Config.COMPACT_HEAD_CH}, dilations={Config.COMPACT_DILATIONS}",
    )

    # ── Data ─────────────────────────────────────────────────────────────
    train_dataset = YOLODataset(Config, split="train")
    train_sampler = None
    if use_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        log_to_file(Config, "Using DistributedSampler for DDP training")
    elif Config.USE_CLASS_BALANCED_SAMPLER:
        train_sampler, sampler_summary = build_class_balanced_train_sampler(Config, train_dataset)
        log_to_file(Config, f"Class balanced sampler: {sampler_summary}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        collate_fn=identity_collate,
        **get_dataloader_kwargs(Config, shuffle=True, sampler=train_sampler),
    )

    val_loader = None
    val_dataset = None
    try:
        val_dataset = YOLODataset(Config, split="val")
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                collate_fn=identity_collate,
                **get_dataloader_kwargs(Config, shuffle=False),
            )
    except Exception as exc:
        log_to_file(Config, f"Validation dataset unavailable: {exc}")

    # ── Criterion ────────────────────────────────────────────────────────
    criterion = build_compact_criterion(Config)
    log_to_file(Config, f"Criterion: {type(criterion).__name__} (anchor-free)")

    # ── Logging ──────────────────────────────────────────────────────────
    log_to_file(Config, "=" * 60)
    log_to_file(Config, "Training Teacher + CompactOpticalDetector (center-point, anchor-free)")
    log_to_file(Config, f"Train images: {len(train_dataset)}, val images: {len(val_dataset) if val_dataset else 0}")
    log_to_file(Config, f"Teacher arch: {Config.TEACHER_ARCH}, detector: compact")
    log_to_file(
        Config,
        f"Teacher params: {_module_param_count(teacher):,}  "
        f"(trainable: {_module_param_count(teacher, trainable_only=True):,})",
    )
    log_to_file(
        Config,
        f"Compact detector params: {_module_param_count(detector):,}  "
        f"(trainable: {_module_param_count(detector, trainable_only=True):,})",
    )
    log_to_file(Config, "=" * 60)

    init_epoch_log_table(Config)
    tensorboard_writer = create_tensorboard_writer(Config, Config.TEACHER_OUTPUT_DIR, log_to_file) if is_main else None
    if is_main:
        _write_compact_tensorboard_model_summary(tensorboard_writer, teacher, detector)
        snapshot_parameters(teacher)
        snapshot_parameters(detector)

    # ── Training state ───────────────────────────────────────────────────
    best_loss = float("inf")
    best_map50 = -1.0
    no_improve_epochs = 0
    last_epoch = -1
    current_phase = None
    optimizer = None
    scheduler = None
    avg_train = {"total": 0.0}
    joint_best_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "teacher_detector_best.pth")
    joint_final_path = os.path.join(Config.TEACHER_OUTPUT_DIR, "teacher_detector_final.pth")

    for epoch in range(Config.EPOCHS):
        last_epoch = epoch
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        teacher.train(bool(not freeze_teacher))
        detector.train()

        stage_settings = Config.get_stage_settings(epoch)
        phase = stage_settings["phase"]
        if phase != current_phase:
            current_phase = phase
            optimizer = _build_compact_optimizer(teacher, detector, teacher_lr=stage_settings["teacher_lr"])
            remaining = Config.EPOCHS - epoch
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=Config.ETA_MIN)
            log_to_file(
                Config,
                f"Epoch {epoch}: phase={phase}, teacher_lr={stage_settings['teacher_lr']:.6g}, "
                f"detector_lr={Config.COMPACT_DETECTOR_LR:.6g}, cosine_T_max={remaining}",
            )

        epoch_loss = 0.0
        train_loss_keys = [k for k in get_compact_loss_keys(Config) if k != "feature_total"]
        epoch_train_stats = {k: 0.0 for k in train_loss_keys}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [{phase}]", leave=True, disable=not is_main):
            batch_images, batch_targets = prepare_batch(Config, batch, device)
            optimizer.zero_grad()

            with amp_context():
                features = teacher(prepare_conv_tensor(Config, batch_images))
                predictions = detector(features)
                loss, loss_stats = criterion(predictions, batch_targets)

            amp_scaler.scale(loss).backward()

            grad_clip = float(getattr(Config, "COMPACT_GRAD_CLIP_NORM", 0.0) or 0.0)
            if grad_clip > 0:
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None],
                    grad_clip,
                )

            amp_scaler.step(optimizer)
            amp_scaler.update()

            epoch_loss += float(loss.detach().item())
            for k in train_loss_keys:
                epoch_train_stats[k] += float(loss_stats.get(k, 0.0))

        n_batches = max(len(train_loader), 1)
        avg_train = {"total": epoch_loss / n_batches}
        for k in train_loss_keys:
            avg_train[k] = epoch_train_stats[k] / n_batches

        if scheduler is not None:
            scheduler.step()
        current_lr = max(group["lr"] for group in optimizer.param_groups)

        # ── Validation ───────────────────────────────────────────────────
        val_losses = None
        val_metrics = None
        if val_loader is not None and ((epoch + 1) % Config.VAL_INTERVAL == 0):
            val_losses, val_metrics = _evaluate_teacher_compact(
                teacher, detector, val_loader, criterion, device,
            )

        # ── TensorBoard ──────────────────────────────────────────────────
        if is_main and tensorboard_writer is not None:
            writer = tensorboard_writer
            writer.add_scalar("Loss/train_total", avg_train["total"], epoch + 1)
            for k in train_loss_keys:
                writer.add_scalar(f"Loss/train_{k}", avg_train[k], epoch + 1)
            if val_losses is not None:
                writer.add_scalar("Loss/val_total", val_losses["total"], epoch + 1)
                for k in train_loss_keys:
                    writer.add_scalar(f"Loss/val_{k}", val_losses.get(k, 0.0), epoch + 1)
            if val_metrics is not None:
                for key in ("precision", "recall", "f1", "map50"):
                    if key in val_metrics:
                        writer.add_scalar(f"Metrics/{key}", val_metrics[key], epoch + 1)
            writer.add_scalar("LR", current_lr, epoch + 1)

            # ── 梯度 & 参数监测 ──
            write_gradient_monitoring(tensorboard_writer, teacher, epoch + 1, prefix="Grad/Teacher")
            write_gradient_monitoring(tensorboard_writer, detector, epoch + 1, prefix="Grad/Detector")
            write_parameter_monitoring(tensorboard_writer, teacher, epoch + 1, prefix="Param/Teacher")
            write_parameter_monitoring(tensorboard_writer, detector, epoch + 1, prefix="Param/Detector")

            # ── 混淆矩阵 ──
            if val_loader is not None and (epoch + 1) % max(Config.VIS_INTERVAL, 1) == 0:
                cm_dets, cm_targets = _collect_compact_val_detections(
                    teacher, detector, val_loader, device,
                )
                write_confusion_matrix(
                    tensorboard_writer, cm_dets, cm_targets,
                    Config.NUM_CLASSES, Config.CLASS_NAMES, epoch + 1,
                    iou_threshold=getattr(Config, "COMPACT_METRIC_IOU_THRESHOLD", 0.5),
                    conf_threshold=getattr(Config, "COMPACT_CONF_THRESH", 0.30),
                    prefix="ConfusionMatrix",
                    image_size=Config.RESOLUTION,
                )

            # ── 预测可视化 ──
            if val_dataset is not None and (epoch + 1) % max(Config.VIS_INTERVAL, 1) == 0:
                _write_compact_tensorboard_predictions(
                    tensorboard_writer, epoch + 1, teacher, detector,
                    val_dataset, device,
                    max_images=getattr(Config, "VIS_MAX_IMAGES", 4),
                )

        # ── Best checkpoint ──────────────────────────────────────────────
        is_best = False
        if val_metrics is not None and val_metrics.get("map50", -1.0) > best_map50 + Config.TEACHER_EARLY_STOP_MIN_DELTA:
            best_map50 = val_metrics["map50"]
            is_best = True
        elif val_loader is None and avg_train["total"] < best_loss:
            is_best = True

        if is_best:
            if val_metrics is None:
                best_loss = avg_train["total"]
            if is_main:
                _save_compact_checkpoint(
                    joint_best_path, teacher, detector, epoch, avg_train["total"], metrics=val_metrics,
                )
                log_to_file(Config, f"Saved best compact detector: epoch={epoch + 1}, map50={best_map50:.4f}, train_loss={avg_train['total']:.6f}")
            no_improve_epochs = 0
        elif val_metrics is not None:
            no_improve_epochs += Config.VAL_INTERVAL

        # ── Epoch log ────────────────────────────────────────────────────
        log_epoch_table_row(
            Config,
            epoch=epoch,
            phase=phase,
            train_loss=avg_train["total"],
            val_loss=val_losses["total"] if val_losses is not None else None,
            precision=val_metrics["precision"] if val_metrics is not None else None,
            recall=val_metrics["recall"] if val_metrics is not None else None,
            f1_score=val_metrics["f1"] if val_metrics is not None else None,
            map50=val_metrics["map50"] if val_metrics is not None else None,
            lr=current_lr,
            best_status=Config.EPOCH_TABLE_BEST_MARK if is_best else "",
            precision_op=None,
        )

        if use_ddp:
            torch.distributed.barrier()

        # ── Early stopping ───────────────────────────────────────────────
        if (
            val_metrics is not None
            and Config.TEACHER_EARLY_STOP_PATIENCE > 0
            and no_improve_epochs >= Config.TEACHER_EARLY_STOP_PATIENCE
        ):
            log_to_file(
                Config,
                f"Early stopping after {no_improve_epochs} epochs without mAP50 improvement. "
                f"Best mAP50={best_map50:.4f}.",
            )
            break

    # ── Final checkpoint ─────────────────────────────────────────────────
    if is_main:
        _save_compact_checkpoint(
            joint_final_path, teacher, detector, last_epoch, avg_train.get("total", 0.0),
        )
        if tensorboard_writer is not None:
            tensorboard_writer.close()

    log_to_file(Config, "=" * 60)
    log_to_file(Config, "Training complete")
    log_to_file(Config, f"Best model saved to: {joint_best_path}")
    log_to_file(Config, f"Final model saved to: {joint_final_path}")
    log_to_file(Config, "=" * 60)

    if use_ddp:
        cleanup_distributed()
