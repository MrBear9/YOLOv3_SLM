"""Compose Fashion-MNIST samples into a compact multi-object YOLO dataset."""

import argparse
import gzip
import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = (
    "T-shirt_top", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle_boot",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/fashion"), help="Source directory containing the four IDX gzip files.")
    parser.add_argument("--output-root", type=Path, default=Path("data/fashion_multi_1to4"), help="Destination for the composed YOLO dataset.")
    parser.add_argument("--val-size", type=int, default=6000, help="Original training samples reserved for validation.")
    parser.add_argument("--canvas-size", type=int, default=224, help="Square output image size in pixels.")
    parser.add_argument("--min-targets", type=int, default=1, help="Minimum targets in each composed image.")
    parser.add_argument("--max-targets", type=int, default=4, help="Maximum targets in each composed image.")
    parser.add_argument("--min-object-size", type=int, default=32, help="Desired minimum foreground-box side before letterboxing.")
    parser.add_argument("--max-cell-fill", type=float, default=0.84, help="Maximum fraction of a layout cell occupied by one target.")
    parser.add_argument("--foreground-threshold", type=int, default=20, help="Threshold used to crop and label garments.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic composition seed.")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated PNG/TXT files in --output-root.")
    return parser.parse_args()


def read_idx_images(path):
    with gzip.open(path, "rb") as handle:
        magic, count, height, width = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Expected image IDX magic 2051 in {path}, got {magic}.")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    if data.size != count * height * width:
        raise ValueError(f"Unexpected image payload size in {path}.")
    return data.reshape(count, height, width)


def read_idx_labels(path):
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Expected label IDX magic 2049 in {path}, got {magic}.")
        labels = np.frombuffer(handle.read(), dtype=np.uint8)
    if labels.size != count:
        raise ValueError(f"Unexpected label payload size in {path}.")
    return labels


def foreground_crop(image, threshold):
    ys, xs = np.nonzero(image > threshold)
    if xs.size == 0:
        return image
    return image[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1]


def layout_cells(target_count, canvas_size, rng):
    """Return non-overlapping cells while retaining useful object scale."""
    if target_count == 1:
        return [(0, 0, canvas_size, canvas_size)]
    if target_count == 2:
        split = canvas_size // 2
        if rng.random() < 0.5:
            return [(0, 0, split, canvas_size), (split, 0, canvas_size, canvas_size)]
        return [(0, 0, canvas_size, split), (0, split, canvas_size, canvas_size)]
    split = canvas_size // 2
    cells = [
        (0, 0, split, split), (split, 0, canvas_size, split),
        (0, split, split, canvas_size), (split, split, canvas_size, canvas_size),
    ]
    rng.shuffle(cells)
    return cells[:target_count]


def resize_object(crop, cell_width, cell_height, args, rng):
    source_height, source_width = crop.shape
    available_width = max(1, int(cell_width * args.max_cell_fill))
    available_height = max(1, int(cell_height * args.max_cell_fill))
    maximum_scale = min(available_width / source_width, available_height / source_height)
    minimum_scale = args.min_object_size / min(source_width, source_height)
    if minimum_scale <= maximum_scale:
        scale = float(rng.uniform(max(minimum_scale, 0.72 * maximum_scale), maximum_scale))
    else:
        # Preserve aspect ratio for unusually slender garments. Any short-side
        # exception is counted in conversion_metadata.json.
        scale = maximum_scale
    width = max(args.min_object_size, int(round(source_width * scale)))
    height = max(args.min_object_size, int(round(source_height * scale)))
    width = min(available_width, width)
    height = min(available_height, height)
    return np.asarray(
        Image.fromarray(crop, mode="L").resize((width, height), Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )


def compose_scene(samples, args, rng):
    canvas = np.zeros((args.canvas_size, args.canvas_size), dtype=np.uint8)
    records = []
    for (source_image, class_id), (x0, y0, x1, y1) in zip(
        samples, layout_cells(len(samples), args.canvas_size, rng)
    ):
        patch = resize_object(
            foreground_crop(source_image, args.foreground_threshold),
            x1 - x0, y1 - y0, args, rng,
        )
        patch_height, patch_width = patch.shape
        paste_x = int(rng.integers(x0, x1 - patch_width + 1))
        paste_y = int(rng.integers(y0, y1 - patch_height + 1))
        region = canvas[paste_y:paste_y + patch_height, paste_x:paste_x + patch_width]
        np.maximum(region, patch, out=region)

        ys, xs = np.nonzero(patch > args.foreground_threshold)
        if xs.size == 0:
            left, right, top, bottom = 0, patch_width - 1, 0, patch_height - 1
        else:
            left, right = int(xs.min()), int(xs.max())
            top, bottom = int(ys.min()), int(ys.max())
        box_left, box_right = paste_x + left, paste_x + right + 1
        box_top, box_bottom = paste_y + top, paste_y + bottom + 1
        box_width, box_height = box_right - box_left, box_bottom - box_top
        records.append({
            "class_id": int(class_id),
            "x": (box_left + box_right) / (2.0 * args.canvas_size),
            "y": (box_top + box_bottom) / (2.0 * args.canvas_size),
            "width": box_width / args.canvas_size,
            "height": box_height / args.canvas_size,
            "width_pixels": box_width,
            "height_pixels": box_height,
        })
    return canvas, records


def prepare_split_dirs(root, split, overwrite):
    images_dir, labels_dir = root / split / "images", root / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    existing = list(images_dir.glob("fashion_multi_*.png")) + list(labels_dir.glob("fashion_multi_*.txt"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{root / split} already contains {len(existing)} generated files; "
            "choose another --output-root or pass --overwrite."
        )
    if overwrite:
        for path in existing:
            path.unlink()
    return images_dir, labels_dir


def target_group_sizes(sample_count, minimum, maximum, rng):
    sizes, remaining = [], sample_count
    while remaining:
        size = min(int(rng.integers(minimum, maximum + 1)), remaining)
        sizes.append(size)
        remaining -= size
    return sizes


def write_split(root, split, images, labels, args, split_seed):
    images_dir, labels_dir = prepare_split_dirs(root, split, args.overwrite)
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(len(images))
    group_sizes = target_group_sizes(len(order), args.min_targets, args.max_targets, rng)
    cursor, object_sizes = 0, []
    for scene_index, group_size in enumerate(group_sizes):
        indices = order[cursor:cursor + group_size]
        cursor += group_size
        canvas, records = compose_scene([(images[index], labels[index]) for index in indices], args, rng)
        stem = f"fashion_multi_{scene_index:05d}"
        Image.fromarray(canvas, mode="L").save(images_dir / f"{stem}.png")
        lines = [
            f"{item['class_id']} {item['x']:.8f} {item['y']:.8f} {item['width']:.8f} {item['height']:.8f}"
            for item in records
        ]
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        object_sizes.extend((item["width_pixels"], item["height_pixels"]) for item in records)
    short_sides = [min(width, height) for width, height in object_sizes]
    return {
        "source_objects": int(len(images)),
        "composite_images": len(group_sizes),
        "objects_per_image": {str(key): value for key, value in sorted(Counter(group_sizes).items())},
        "mean_objects_per_image": float(len(images) / len(group_sizes)),
        "minimum_box_short_side_pixels": int(min(short_sides)),
        "objects_below_requested_minimum": int(sum(size < args.min_object_size for size in short_sides)),
    }


def write_dataset_yaml(output_root):
    try:
        dataset_path = output_root.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        dataset_path = output_root.as_posix()
    names = "\n".join(f"  - {name}" for name in CLASS_NAMES)
    content = (
        "# Multi-object Fashion-MNIST detection dataset.\n"
        f"path: {dataset_path}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n{names}\n\n"
        "# Composition already controls target scale; do not shrink the full scene again.\n"
        "augmentation:\n  canvas_scale:\n    min: 1.0\n    max: 1.0\n"
        "  affine_scale_min: 1.0\n  affine_scale_max: 1.0\n"
        "  affine_rotate_deg: 0.0\n  affine_translate: 0.0\ncopy_paste: false\n"
    )
    (output_root / "data.yaml").write_text(content, encoding="utf-8")


def validate_args(args, training_count):
    if not 0 < args.val_size < training_count:
        raise ValueError(f"--val-size must be between 1 and {training_count - 1}.")
    if args.canvas_size < 64:
        raise ValueError("--canvas-size must be at least 64.")
    if not 1 <= args.min_targets <= args.max_targets <= 4:
        raise ValueError("Target counts must satisfy 1 <= min <= max <= 4.")
    if not 4 <= args.min_object_size < args.canvas_size:
        raise ValueError("--min-object-size must be at least 4 and smaller than --canvas-size.")
    if not 0.5 <= args.max_cell_fill <= 0.95:
        raise ValueError("--max-cell-fill must be between 0.5 and 0.95.")
    if not 0 <= args.foreground_threshold <= 255:
        raise ValueError("--foreground-threshold must be between 0 and 255.")


def main():
    args = parse_args()
    source_root, output_root = args.root.resolve(), args.output_root.resolve()
    train_images = read_idx_images(source_root / "train-images-idx3-ubyte.gz")
    train_labels = read_idx_labels(source_root / "train-labels-idx1-ubyte.gz")
    test_images = read_idx_images(source_root / "t10k-images-idx3-ubyte.gz")
    test_labels = read_idx_labels(source_root / "t10k-labels-idx1-ubyte.gz")
    validate_args(args, len(train_images))
    if len(train_images) != len(train_labels) or len(test_images) != len(test_labels):
        raise ValueError("Image and label counts do not match.")
    if train_labels.max(initial=0) >= len(CLASS_NAMES) or test_labels.max(initial=0) >= len(CLASS_NAMES):
        raise ValueError("Found a Fashion-MNIST label outside the expected 0--9 range.")

    output_root.mkdir(parents=True, exist_ok=True)
    train_end = len(train_images) - args.val_size
    counts = {
        "train": write_split(output_root, "train", train_images[:train_end], train_labels[:train_end], args, args.seed),
        "val": write_split(output_root, "val", train_images[train_end:], train_labels[train_end:], args, args.seed + 1),
        "test": write_split(output_root, "test", test_images, test_labels, args, args.seed + 2),
    }
    write_dataset_yaml(output_root)
    metadata = {
        "source": "Fashion-MNIST IDX gzip", "source_root": str(source_root),
        "composition": {
            "seed": args.seed, "canvas_size": args.canvas_size,
            "targets_per_image": [args.min_targets, args.max_targets],
            "requested_minimum_object_side_pixels": args.min_object_size,
            "max_cell_fill": args.max_cell_fill,
            "foreground_threshold": args.foreground_threshold,
        },
        "classes": CLASS_NAMES, "counts": counts,
    }
    (output_root / "conversion_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Created multi-object Fashion-MNIST under {output_root}")
    for split, stats in counts.items():
        print(
            f"  {split}: {stats['source_objects']} objects -> {stats['composite_images']} images, "
            f"mean={stats['mean_objects_per_image']:.2f} objects/image, "
            f"min short side={stats['minimum_box_short_side_pixels']} px"
        )


if __name__ == "__main__":
    main()
