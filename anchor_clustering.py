"""
Anchor box clustering utility for the military YOLO dataset.

Highlights:
- Supports clustering over multiple splits, e.g. train + val.
- Uses IoU-distance k-means, which is more suitable for anchor clustering than Euclidean k-means.
- Compares clustered anchors against the current default anchors.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm


DEFAULT_ANCHORS = [
    [26, 23], [47, 49], [100, 67],
    [103, 169], [203, 107], [351, 177],
    [241, 354], [534, 299], [568, 528],
]


def load_dataset_info(yaml_path):
    """Load dataset configuration and resolve the dataset root path."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg.get("path", "")
    if not data_root:
        data_root = os.path.dirname(os.path.abspath(yaml_path))
    elif not os.path.exists(data_root):
        data_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(yaml_path)), data_root))

    return cfg, data_root


def collect_bounding_boxes(data_root, splits):
    """Collect normalized width/height boxes from one or more dataset splits."""
    all_boxes = []
    split_stats = {}

    for split in splits:
        label_dir = os.path.join(data_root, split, "labels")
        if not os.path.exists(label_dir):
            print(f"Warning: label directory does not exist, skipped: {label_dir}")
            split_stats[split] = {"files": 0, "boxes": 0}
            continue

        label_files = sorted(f for f in os.listdir(label_dir) if f.endswith(".txt"))
        split_box_count = 0

        print(f"Collecting bounding boxes from split '{split}'...")
        for label_file in tqdm(label_files, leave=False):
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    width = float(parts[3])
                    height = float(parts[4])
                    if width <= 0.0 or height <= 0.0:
                        continue

                    all_boxes.append([width, height])
                    split_box_count += 1

        split_stats[split] = {"files": len(label_files), "boxes": split_box_count}

    return np.asarray(all_boxes, dtype=np.float32), split_stats


def box_iou_matrix(boxes, anchors):
    """Compute IoU matrix for width/height boxes assuming the same center."""
    boxes = np.asarray(boxes, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=np.float32)

    inter_w = np.minimum(boxes[:, None, 0], anchors[None, :, 0])
    inter_h = np.minimum(boxes[:, None, 1], anchors[None, :, 1])
    inter_area = inter_w * inter_h

    box_area = boxes[:, 0:1] * boxes[:, 1:2]
    anchor_area = anchors[:, 0] * anchors[:, 1]
    union_area = box_area + anchor_area - inter_area + 1e-9
    return inter_area / union_area


def kmeans_anchors_iou(boxes, num_anchors=9, img_size=640, max_iter=100, seed=42):
    """Cluster anchors with IoU-distance k-means."""
    if len(boxes) == 0:
        raise ValueError("No valid bounding boxes found for anchor clustering.")

    boxes_pixels = np.asarray(boxes, dtype=np.float32) * float(img_size)
    rng = np.random.default_rng(seed)
    clusters = boxes_pixels[rng.choice(len(boxes_pixels), size=num_anchors, replace=False)].copy()
    last_assignments = None

    for _ in range(max_iter):
        distances = 1.0 - box_iou_matrix(boxes_pixels, clusters)
        assignments = distances.argmin(axis=1)

        if last_assignments is not None and np.array_equal(assignments, last_assignments):
            break

        for cluster_idx in range(num_anchors):
            mask = assignments == cluster_idx
            if mask.any():
                clusters[cluster_idx] = np.median(boxes_pixels[mask], axis=0)

        last_assignments = assignments

    anchors = clusters / float(img_size)
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]
    return anchors


def calculate_avg_iou(boxes, anchors, img_size=640):
    """Calculate average best-anchor IoU."""
    if len(boxes) == 0:
        return 0.0

    boxes_pixels = np.asarray(boxes, dtype=np.float32) * float(img_size)
    anchors_pixels = np.asarray(anchors, dtype=np.float32) * float(img_size)
    ious = box_iou_matrix(boxes_pixels, anchors_pixels)
    return float(ious.max(axis=1).mean())


def format_anchors_for_yolo(anchors, img_size=640, num_layers=3):
    """Format anchors as pixel anchors grouped for P3/P4/P5."""
    anchors = np.asarray(anchors, dtype=np.float32)
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]

    if len(anchors) % num_layers != 0:
        raise ValueError("The number of anchors must be divisible by the number of layers.")

    anchors_per_layer = len(anchors) // num_layers
    yolo_anchors = []

    for layer_idx in range(num_layers):
        layer_anchors = anchors[layer_idx * anchors_per_layer:(layer_idx + 1) * anchors_per_layer]
        anchor_list = []
        for anchor in layer_anchors:
            w = int(round(anchor[0] * img_size))
            h = int(round(anchor[1] * img_size))
            anchor_list.append([w, h])
        yolo_anchors.append(anchor_list)

    return yolo_anchors


def plot_anchor_distribution(boxes, anchors, img_size, output_path):
    """Plot box size distribution with clustered anchors overlaid."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    boxes_pixels = np.asarray(boxes, dtype=np.float32) * float(img_size)
    anchors_pixels = np.asarray(anchors, dtype=np.float32) * float(img_size)

    plt.figure(figsize=(12, 8))
    plt.scatter(
        boxes_pixels[:, 0],
        boxes_pixels[:, 1],
        alpha=0.2,
        s=8,
        label="Dataset Boxes",
        color="steelblue",
    )
    plt.scatter(
        anchors_pixels[:, 0],
        anchors_pixels[:, 1],
        s=220,
        marker="x",
        color="crimson",
        linewidth=3,
        label="Clustered Anchors",
    )

    for anchor in anchors_pixels:
        plt.annotate(
            f"{int(round(anchor[0]))}x{int(round(anchor[1]))}",
            (anchor[0], anchor[1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    max_dim = float(max(boxes_pixels[:, 0].max(), boxes_pixels[:, 1].max()))
    plt.xlim(0, max_dim * 1.1)
    plt.ylim(0, max_dim * 1.1)
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.title("Anchor Clustering Distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_results(output_dir, args, split_stats, boxes, anchors, avg_iou, default_avg_iou):
    """Save plain-text summary and YOLO anchor config."""
    yolo_formatted = format_anchors_for_yolo(anchors, img_size=args.img_size)
    output_file = os.path.join(output_dir, "anchor_clustering_results.txt")
    yolo_config_file = os.path.join(output_dir, "yolo_anchors.yaml")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== YOLO Anchor Clustering Analysis Results ===\n\n")
        f.write(f"Dataset YAML: {args.yaml}\n")
        f.write(f"Splits: {', '.join(args.splits)}\n")
        f.write(f"Image size: {args.img_size}\n")
        f.write(f"Number of anchors: {args.num_anchors}\n")
        f.write(f"Number of bounding boxes: {len(boxes)}\n")
        f.write(f"Average IoU (clustered anchors): {avg_iou:.4f}\n")
        f.write(f"Average IoU (default anchors): {default_avg_iou:.4f}\n\n")
        f.write("Split statistics:\n")
        for split, stats in split_stats.items():
            f.write(f"- {split}: files={stats['files']}, boxes={stats['boxes']}\n")
        f.write("\nClustered anchors (pixels):\n")
        for i, anchor in enumerate(np.round(anchors * args.img_size).astype(int), start=1):
            f.write(f"Anchor {i}: {anchor[0]}x{anchor[1]}\n")
        f.write("\nYOLO anchor groups:\n")
        for i, layer_anchors in enumerate(yolo_formatted):
            f.write(f"P{3 + i}: {layer_anchors}\n")

    config_payload = {
        "dataset_yaml": args.yaml,
        "splits": args.splits,
        "img_size": args.img_size,
        "num_anchors": args.num_anchors,
        "num_boxes": int(len(boxes)),
        "avg_iou": float(avg_iou),
        "default_avg_iou": float(default_avg_iou),
        "recommended": bool(avg_iou >= default_avg_iou),
        "anchors": yolo_formatted,
    }
    with open(yolo_config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_payload, f, allow_unicode=True, sort_keys=False)

    return output_file, yolo_config_file


def print_anchor_report(title, anchors, img_size):
    """Print anchor sizes in a readable way."""
    print(f"\n=== {title} ===")
    for idx, anchor in enumerate(np.round(np.asarray(anchors) * img_size).astype(int), start=1):
        print(f"Anchor {idx}: {anchor[0]}x{anchor[1]}")


def main():
    parser = argparse.ArgumentParser(description="YOLO anchor clustering analysis")
    parser.add_argument("--yaml", type=str, default="data/military/data.yaml", help="Dataset YAML file path")
    parser.add_argument("--num-anchors", type=int, default=9, help="Number of anchors")
    parser.add_argument("--img-size", type=int, default=640, help="Training image size")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Dataset splits to cluster together, e.g. --splits train val",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/anchor_clustering",
        help="Directory to save reports and config",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg, data_root = load_dataset_info(args.yaml)
    print(f"Output directory: {args.output_dir}")
    print(f"Dataset root: {data_root}")
    print(f"Number of classes: {cfg.get('nc', 'Unknown')}")
    print(f"Class names: {cfg.get('names', 'Unknown')}")
    print(f"Clustering splits: {args.splits}")

    boxes, split_stats = collect_bounding_boxes(data_root, args.splits)
    if len(boxes) == 0:
        raise RuntimeError("No bounding box data found in the requested splits.")

    print(f"Collected {len(boxes)} bounding boxes")
    anchors = kmeans_anchors_iou(boxes, num_anchors=args.num_anchors, img_size=args.img_size)
    avg_iou = calculate_avg_iou(boxes, anchors, img_size=args.img_size)
    default_anchors = np.asarray(DEFAULT_ANCHORS, dtype=np.float32) / float(args.img_size)
    default_avg_iou = calculate_avg_iou(boxes, default_anchors, img_size=args.img_size)

    print(f"Average IoU (clustered anchors): {avg_iou:.4f}")
    print(f"Average IoU (default anchors):   {default_avg_iou:.4f}")
    print_anchor_report("Clustered Anchors", anchors, args.img_size)
    print_anchor_report("Default Anchors", default_anchors, args.img_size)

    plot_path = os.path.join(args.output_dir, "anchor_distribution.png")
    plot_anchor_distribution(boxes, anchors, args.img_size, plot_path)
    output_file, yolo_config_file = save_results(
        args.output_dir,
        args,
        split_stats,
        boxes,
        anchors,
        avg_iou,
        default_avg_iou,
    )

    print("\nYOLO config format:")
    print("anchors:")
    for i, layer_anchors in enumerate(format_anchors_for_yolo(anchors, img_size=args.img_size)):
        print(f"  # P{3 + i}/{[8, 16, 32][i]}")
        print(f"  - {layer_anchors}")

    recommendation = "recommended" if avg_iou >= default_avg_iou else "not recommended"
    print(f"\nThis clustered anchor set is {recommendation} compared with the default anchors.")
    print(f"- Analysis results: {output_file}")
    print(f"- Distribution plot: {plot_path}")
    print(f"- YOLO config file: {yolo_config_file}")


if __name__ == "__main__":
    main()
