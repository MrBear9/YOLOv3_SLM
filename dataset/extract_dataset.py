import argparse
import random
import shutil
import yaml
from pathlib import Path


def extract_dataset(yaml_path, split, num, output_dir):
    # Read data.yaml
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    if split not in data:
        raise ValueError(f"Split '{split}' not found in {yaml_path}. Available: {[k for k in data if k in ('train', 'val', 'test')]}")

    dataset_root = Path(yaml_path).parent
    split_path = Path(data[split])
    # data.yaml uses relative paths from its own directory
    images_dir = dataset_root / split_path

    # Handle both "path/train/images" and "train/images" formats
    if not images_dir.exists():
        images_dir = Path(data.get('path', '.')) / split_path
        if not images_dir.is_absolute():
            images_dir = dataset_root / images_dir
    images_dir = images_dir.resolve()

    labels_dir = images_dir.parent / 'labels'

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Gather all image files
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])

    if num > len(image_files):
        raise ValueError(f"Requested {num} images, but only {len(image_files)} available in {split}")

    # Random sampling
    selected = random.sample(image_files, num)

    # Prepare output directory
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    digits = max(4, len(str(num)))  # at least 4-digit zero-padding like Origin0001

    for i, img_path in enumerate(selected, start=1):
        ext = img_path.suffix
        new_name = f"Origin{i:0{digits}d}"

        # Copy image
        dst_img = out / f"{new_name}{ext}"
        shutil.copy2(img_path, dst_img)
        print(f"[{i:0{digits}d}/{num}] {img_path.name} -> {dst_img.name}")

        # Copy label if exists
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            dst_label = out / f"{new_name}.txt"
            shutil.copy2(label_path, dst_label)
        else:
            print(f"  Warning: label not found for {img_path.name}")

    print(f"\nDone. {num} images + labels extracted to {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract random samples from YOLO dataset')
    parser.add_argument('--yaml', type=str, default='data/military/data.yaml',
                        help='Path to data.yaml (default: data/military/data.yaml)')
    parser.add_argument('--split', type=str, required=True,
                        choices=['train', 'val', 'test'],
                        help='Which split to sample from')
    parser.add_argument('--num', type=int, required=True,
                        help='Number of images to extract')
    parser.add_argument('--output', type=str, default='Optical_yolo_detect/image_Origin',
                        help='Output directory (default: Optical_yolo_detect/image_Origin)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    random.seed(args.seed)
    extract_dataset(args.yaml, args.split, args.num, args.output)
