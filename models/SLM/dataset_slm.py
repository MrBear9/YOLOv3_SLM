import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import yaml

from models.dataset import letterbox_image_targets


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_data_path(path):
    if not path or os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def infer_label_path(image_path):
    image_path = Path(image_path)
    label_name = image_path.with_suffix(".txt").name
    candidates = []
    parent_parts = list(image_path.parent.parts)
    for idx in range(len(parent_parts) - 1, -1, -1):
        if parent_parts[idx].lower() == "images":
            candidates.append(Path(*parent_parts[:idx], "labels", *parent_parts[idx + 1:]) / label_name)
            break
    if image_path.parent.name.lower() == "images":
        candidates.append(image_path.parent.parent / "labels" / label_name)
    candidates.append(image_path.with_suffix(".txt"))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


class SLMFeatureDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        with open(config.YAML_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if bool(getattr(config, "SINGLE_IMAGE_TRAINING", False)):
            image_path = resolve_data_path(getattr(config, "SINGLE_IMAGE_PATH", ""))
            if not image_path:
                raise ValueError("SINGLE_IMAGE_TRAINING is enabled, but SINGLE_IMAGE_PATH is empty.")
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Single-image training file not found: {image_path}")
            label_path = resolve_data_path(getattr(config, "SINGLE_IMAGE_LABEL_PATH", ""))
            if not label_path:
                label_path = infer_label_path(image_path)
            repeat = max(int(getattr(config, "SINGLE_IMAGE_REPEAT", 1)), 1) if split == "train" else 1
            self.entries = [
                {
                    "image_path": image_path,
                    "label_path": label_path,
                }
                for _ in range(repeat)
            ]
            return
        root = cfg.get("path", ".")
        if not os.path.isabs(root):
            root = os.path.join(PROJECT_ROOT, root)
        split_rel = cfg.get(split)
        if split_rel is None:
            raise ValueError(f"Split '{split}' not found in yaml: {config.YAML_PATH}")
        images_dir = split_rel if os.path.isabs(split_rel) else os.path.join(root, split_rel)
        labels_dir = os.path.join(os.path.dirname(images_dir), "labels")
        self.entries = []
        for name in sorted(os.listdir(images_dir)):
            image_path = os.path.join(images_dir, name)
            if not os.path.isfile(image_path) or os.path.splitext(name)[1].lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                continue
            self.entries.append(
                {
                    "image_path": image_path,
                    "label_path": os.path.join(labels_dir, os.path.splitext(name)[0] + ".txt"),
                }
            )
        if split == "train":
            repeat = max(int(getattr(config, "TRAIN_DATASET_REPEAT", 1)), 1)
            self.entries *= repeat

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img = Image.open(entry["image_path"]).convert("RGB")
        targets = []
        label_path = entry["label_path"]
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        targets.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 5), dtype=torch.float32)
        img, targets = letterbox_image_targets(img, targets, self.config.IMG_SIZE)
        gray_tensor = TF.to_tensor(TF.to_grayscale(img, num_output_channels=1))
        rgb_tensor = gray_tensor
        return {
            "gray_tensor": gray_tensor,
            "rgb_tensor": rgb_tensor,
            "targets": targets,
            "image_path": entry["image_path"],
        }


def slm_collate_fn(batch):
    return {
        "gray_tensor": torch.stack([item["gray_tensor"] for item in batch], dim=0),
        "rgb_tensor": torch.stack([item["rgb_tensor"] for item in batch], dim=0),
        "targets": [item["targets"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }
