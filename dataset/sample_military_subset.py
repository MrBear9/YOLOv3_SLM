"""Create a reproducible YOLO subset from data/military/data.yaml.

Examples:
  python dataset/sample_military_subset.py --train-count 100 --val-count 20
  python dataset/sample_military_subset.py --train-count 100 --val-count 20 --balanced
  python dataset/sample_military_subset.py --train-count 20 --val-count 5 --class military_tank
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_names(raw_names):
    if isinstance(raw_names, dict):
        return [str(raw_names[index]) for index in sorted(raw_names, key=int)]
    return [str(name) for name in raw_names]


def resolve_dataset_root(config, yaml_path):
    root = Path(config.get("path", yaml_path.parent))
    if root.is_absolute():
        return root
    project_candidate = PROJECT_ROOT / root
    return project_candidate if project_candidate.exists() else yaml_path.parent / root


def read_labels(label_path, class_count):
    labels = []
    if not label_path.exists():
        return labels
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:5]]
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}") from exc
        if not 0 <= class_id < class_count:
            raise ValueError(f"Class id {class_id} at {label_path}:{line_number} is outside the YAML class range.")
        labels.append((class_id, values))
    return labels


def collect_candidates(images_dir, class_count, require_labels):
    candidates = []
    image_paths = sorted(
        image_path for image_path in images_dir.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES
    )
    for index, image_path in enumerate(image_paths, start=1):
        if require_labels and index % 10000 == 0:
            print(f"  Scanned {index}/{len(image_paths)} images in {images_dir}", flush=True)
        label_path = images_dir.parent / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        labels = read_labels(label_path, class_count) if require_labels else None
        if not require_labels or labels:
            candidates.append({"image": image_path, "label": label_path, "labels": labels})
    return candidates


def select_random(candidates, count, rng):
    if count > len(candidates):
        raise ValueError(f"Requested {count} images, but only {len(candidates)} labelled images are available.")
    return rng.sample(candidates, count)


def select_single_class(candidates, count, class_id, rng):
    eligible = [candidate for candidate in candidates if any(label[0] == class_id for label in candidate["labels"])]
    if count > len(eligible):
        raise ValueError(
            f"Requested {count} images for class {class_id}, but only {len(eligible)} labelled images are available."
        )
    return rng.sample(eligible, count)


def select_balanced(candidates, count, class_count, rng):
    quotient, remainder = divmod(count, class_count)
    quotas = {class_id: quotient + (class_id < remainder) for class_id in range(class_count)}
    pools = {
        class_id: [candidate for candidate in candidates if any(label[0] == class_id for label in candidate["labels"])]
        for class_id in range(class_count)
    }
    selected = []
    selected_paths = set()
    # Scarcer classes pick first so shared images remain available to them.
    for class_id in sorted(range(class_count), key=lambda key: len(pools[key])):
        available = [candidate for candidate in pools[class_id] if candidate["image"] not in selected_paths]
        if quotas[class_id] > len(available):
            raise ValueError(
                f"Cannot select a balanced subset: class {class_id} needs {quotas[class_id]} unique images, "
                f"but only {len(available)} remain."
            )
        picked = rng.sample(available, quotas[class_id])
        selected.extend(picked)
        selected_paths.update(candidate["image"] for candidate in picked)
    rng.shuffle(selected)
    return selected


def write_sample(sample, destination_images, destination_labels, single_class_id=None):
    image_path = sample["image"]
    shutil.copy2(image_path, destination_images / image_path.name)
    labels = sample["labels"]
    if labels is None:
        if sample["label"].exists():
            shutil.copy2(sample["label"], destination_labels / f"{image_path.stem}.txt")
        return
    if single_class_id is not None:
        labels = [(0, values) for class_id, values in labels if class_id == single_class_id]
    label_lines = ["{} {:.6f} {:.6f} {:.6f} {:.6f}".format(class_id, *values) for class_id, values in labels]
    (destination_labels / f"{image_path.stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")


def summarize_labels(samples, class_count, single_class_id=None):
    counts = Counter()
    for sample in samples:
        labels = sample["labels"]
        if labels is None:
            labels = read_labels(sample["label"], class_count)
        for class_id, _ in labels:
            if single_class_id is None or class_id == single_class_id:
                counts[0 if single_class_id is not None else class_id] += 1
    return dict(sorted(counts.items()))


def select_split(candidates, count, class_count, rng, balanced, single_class_id):
    if single_class_id is not None:
        return select_single_class(candidates, count, single_class_id, rng)
    if balanced:
        return select_balanced(candidates, count, class_count, rng)
    return select_random(candidates, count, rng)


def create_subset(args):
    yaml_path = resolve_project_path(args.yaml).resolve()
    with yaml_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    names = load_names(config["names"])
    class_count = len(names)
    single_class_id = None
    if args.class_name is not None:
        try:
            single_class_id = int(args.class_name)
        except ValueError:
            if args.class_name not in names:
                raise ValueError(f"Unknown class '{args.class_name}'. Choices: {', '.join(names)}")
            single_class_id = names.index(args.class_name)
        if not 0 <= single_class_id < class_count:
            raise ValueError(f"Class id must be in [0, {class_count - 1}].")
    if args.balanced and single_class_id is not None:
        raise ValueError("--balanced and --class cannot be used together.")

    dataset_root = resolve_dataset_root(config, yaml_path)
    rng = random.Random(args.seed)
    selected_by_split = {}
    for split, count in (("train", args.train_count), ("val", args.val_count)):
        if split not in config:
            raise ValueError(f"Source YAML does not define the required '{split}' split.")
        images_dir = Path(config[split])
        images_dir = images_dir if images_dir.is_absolute() else dataset_root / images_dir
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        require_labels = args.balanced or single_class_id is not None
        candidates = collect_candidates(images_dir, class_count, require_labels)
        selected_by_split[split] = select_split(
            candidates, count, class_count, rng, args.balanced, single_class_id
        )

    output_dir = resolve_project_path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    if args.dry_run:
        for split, samples in selected_by_split.items():
            print(
                f"{split}: images={len(samples)}, "
                f"label_instances={summarize_labels(samples, class_count, single_class_id)}"
            )
        return

    output_names = [names[single_class_id]] if single_class_id is not None else names
    for split, samples in selected_by_split.items():
        image_destination = output_dir / split / "images"
        label_destination = output_dir / split / "labels"
        image_destination.mkdir(parents=True, exist_ok=True)
        label_destination.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            write_sample(sample, image_destination, label_destination, single_class_id)
        print(
            f"{split}: images={len(samples)}, "
            f"label_instances={summarize_labels(samples, class_count, single_class_id)}"
        )

    relative_output = output_dir.resolve().relative_to(PROJECT_ROOT).as_posix()
    output_yaml = {
        "path": relative_output,
        "train": "train/images",
        "val": "val/images",
        "nc": len(output_names),
        "names": output_names,
    }
    output_yaml_path = output_dir / "data.yaml"
    with output_yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(output_yaml, handle, allow_unicode=True, sort_keys=False)
    print(f"Created subset YAML: {output_yaml_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Sample a train/val YOLO subset with a fixed random seed.")
    parser.add_argument("--yaml", default="data/military/data.yaml", help="Source YOLO data.yaml path.")
    parser.add_argument("--output", default="dataset/military", help="Output dataset directory.")
    parser.add_argument("--train-count", type=int, required=True, help="Number of training images to sample.")
    parser.add_argument("--val-count", type=int, required=True, help="Number of validation images to sample.")
    parser.add_argument("--class", dest="class_name", help="Use one class name or id only; labels are remapped to class 0.")
    parser.add_argument("--balanced", action="store_true", help="Allocate the requested image count evenly across classes.")
    parser.add_argument("--seed", type=int, default=20260712, help="Sampling seed.")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected split statistics without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory.")
    args = parser.parse_args()
    if args.train_count < 1 or args.val_count < 1:
        parser.error("--train-count and --val-count must both be at least 1.")
    return args


if __name__ == "__main__":
    create_subset(parse_args())
