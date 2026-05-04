import os
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def identity_collate(batch):
    return batch


class YOLODataset(Dataset):
    def __init__(self, config, yaml_path=None, split="train"):
        self.config = config
        if yaml_path is None:
            yaml_path = config.YAML_PATH
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        root = cfg["path"]
        if not os.path.isabs(root):
            root = os.path.join(PROJECT_ROOT, root)
        img_dir = os.path.join(root, f"{split}/images")
        self.label_dir = os.path.join(root, f"{split}/labels")
        self.files = sorted(
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
        self.img_size = config.IMG_SIZE
        self.num_classes = config.NUM_CLASSES
        self._sampling_metadata = None
        self.transform = transforms.Compose(
            [transforms.Resize((self.img_size, self.img_size)), transforms.Grayscale(1), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.files)

    def get_label_path(self, img_path):
        return os.path.join(self.label_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")

    def get_sampling_metadata(self):
        if self._sampling_metadata is not None:
            return self._sampling_metadata
        image_class_counters = []
        class_box_counts = Counter()
        empty_image_count = 0
        for img_path in self.files:
            label_path = self.get_label_path(img_path)
            image_class_counter = Counter()
            if os.path.exists(label_path):
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        try:
                            cls_id = int(parts[0])
                        except ValueError:
                            continue
                        if 0 <= cls_id < self.num_classes:
                            image_class_counter[cls_id] += 1
                            class_box_counts[cls_id] += 1
            if len(image_class_counter) == 0:
                empty_image_count += 1
            image_class_counters.append(image_class_counter)
        self._sampling_metadata = {
            "image_class_counters": image_class_counters,
            "class_box_counts": class_box_counts,
            "empty_image_count": empty_image_count,
        }
        return self._sampling_metadata

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        label_path = self.get_label_path(img_path)
        targets = []
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        targets.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        if targets:
            targets = torch.tensor(targets, dtype=torch.float32)
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)
        return img_tensor, targets


def build_class_balanced_train_sampler(config, dataset):
    metadata = dataset.get_sampling_metadata()
    class_box_counts = metadata["class_box_counts"]
    if len(class_box_counts) == 0:
        return None, {"enabled": False, "reason": "no_valid_labels"}

    majority_class_id, majority_count = class_box_counts.most_common(1)[0]
    class_gains = {}
    for cls_id, cls_count in class_box_counts.items():
        raw_gain = (majority_count / max(cls_count, 1)) ** config.CLASS_BALANCE_POWER
        class_gains[cls_id] = min(config.MAX_CLASS_BALANCE_GAIN, max(1.0, raw_gain))

    image_weights = []
    boosted_images = 0
    majority_only_images = 0
    for image_class_counter in metadata["image_class_counters"]:
        if len(image_class_counter) == 0:
            weight = config.EMPTY_IMAGE_SAMPLE_WEIGHT
        else:
            total_boxes = sum(image_class_counter.values())
            weight = sum(box_count * class_gains.get(cls_id, 1.0) for cls_id, box_count in image_class_counter.items()) / max(total_boxes, 1)
            if len(image_class_counter) == 1 and majority_class_id in image_class_counter:
                majority_only_images += 1
                weight *= config.MAJORITY_ONLY_IMAGE_WEIGHT
            if weight > 1.05:
                boosted_images += 1
        image_weights.append(max(config.MIN_IMAGE_SAMPLE_WEIGHT, weight))

    weights_tensor = torch.tensor(image_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights=weights_tensor, num_samples=len(dataset), replacement=True)
    summary = {
        "enabled": True,
        "majority_class_name": config.CLASS_NAMES.get(majority_class_id, str(majority_class_id)),
        "majority_count": int(majority_count),
        "boosted_images": boosted_images,
        "majority_only_images": majority_only_images,
        "empty_images": metadata["empty_image_count"],
        "min_weight": round(float(weights_tensor.min().item()), 4),
        "max_weight": round(float(weights_tensor.max().item()), 4),
        "mean_weight": round(float(weights_tensor.mean().item()), 4),
    }
    return sampler, summary
