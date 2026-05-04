import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SLMFeatureDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        with open(config.YAML_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
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
        self.gray = transforms.Compose([transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)), transforms.Grayscale(1), transforms.ToTensor()])
        self.rgb = transforms.Compose([transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)), transforms.Grayscale(1), transforms.ToTensor()])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img = Image.open(entry["image_path"]).convert("RGB")
        gray_tensor = self.gray(img)
        rgb_tensor = self.rgb(img)
        targets = []
        label_path = entry["label_path"]
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        targets.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 5), dtype=torch.float32)
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
