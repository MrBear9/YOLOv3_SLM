import os
import math
import random
from collections import Counter

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms import functional as TF
import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def identity_collate(batch):
    return batch


def letterbox_image_targets(img, targets, image_size, fill=114):
    src_w, src_h = img.size
    scale = min(image_size / src_w, image_size / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left, top = (image_size - new_w) // 2, (image_size - new_h) // 2
    canvas = Image.new("RGB", (image_size, image_size), color=(fill, fill, fill))
    canvas.paste(resized, (left, top))
    if targets.numel():
        targets = targets.clone()
        targets[:, 1] = (targets[:, 1] * src_w * scale + left) / image_size
        targets[:, 2] = (targets[:, 2] * src_h * scale + top) / image_size
        targets[:, 3] = targets[:, 3] * src_w * scale / image_size
        targets[:, 4] = targets[:, 4] * src_h * scale / image_size
    return canvas, targets


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
        self.augment = split == "train" and bool(getattr(config, "TRAIN_AUGMENT", False))

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
        label_path = self.get_label_path(img_path)
        targets = []
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        targets.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 5), dtype=torch.float32)
        img, targets = self._letterbox(img, targets)
        if self.augment:
            img, targets = self._augment(img, targets)
        img_tensor = TF.to_tensor(TF.to_grayscale(img, num_output_channels=1))
        if self.augment and random.random() < float(getattr(self.config, "AUG_NOISE_PROB", 0.10)):
            std = float(getattr(self.config, "AUG_NOISE_STD", 0.01))
            img_tensor = (img_tensor + torch.randn_like(img_tensor) * std).clamp_(0.0, 1.0)
        return img_tensor, targets

    def _letterbox(self, img, targets):
        return letterbox_image_targets(img, targets, self.img_size)

    def _augment(self, img, targets):
        if random.random() < float(getattr(self.config, "AUG_HFLIP_PROB", 0.5)):
            img = TF.hflip(img)
            if targets.numel():
                targets[:, 1] = 1.0 - targets[:, 1]

        angle = random.uniform(-float(getattr(self.config, "AUG_ROTATE_DEG", 5.0)), float(getattr(self.config, "AUG_ROTATE_DEG", 5.0)))
        scale = random.uniform(float(getattr(self.config, "AUG_SCALE_MIN", 0.8)), float(getattr(self.config, "AUG_SCALE_MAX", 1.25)))
        translate_frac = float(getattr(self.config, "AUG_TRANSLATE", 0.08))
        tx = random.uniform(-translate_frac, translate_frac) * self.img_size
        ty = random.uniform(-translate_frac, translate_frac) * self.img_size
        boxes = self._targets_to_xyxy(targets)
        shear = [0.0, 0.0]
        img = v2.functional.affine(img, angle, [round(tx), round(ty)], scale, shear, fill=114)
        boxes = v2.functional.affine(boxes, angle, [round(tx), round(ty)], scale, shear)
        targets = self._xyxy_to_targets(targets, boxes)

        brightness = float(getattr(self.config, "AUG_BRIGHTNESS", 0.15))
        contrast = float(getattr(self.config, "AUG_CONTRAST", 0.15))
        gamma = float(getattr(self.config, "AUG_GAMMA", 0.15))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(1.0 - brightness, 1.0 + brightness))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(1.0 - contrast, 1.0 + contrast))
        gamma_value = random.uniform(1.0 - gamma, 1.0 + gamma)
        img = img.point([round(255.0 * ((i / 255.0) ** gamma_value)) for i in range(256)] * 3)
        if random.random() < float(getattr(self.config, "AUG_BLUR_PROB", 0.08)):
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.8)))
        return img, targets

    def _targets_to_xyxy(self, targets):
        boxes = targets[:, 1:5].clone()
        xyxy = torch.stack((boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                            boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2), 1) * self.img_size
        return tv_tensors.BoundingBoxes(xyxy, format="XYXY", canvas_size=(self.img_size, self.img_size))

    def _xyxy_to_targets(self, targets, boxes):
        boxes = torch.as_tensor(boxes).clamp(0, self.img_size)
        wh = boxes[:, 2:4] - boxes[:, 0:2]
        keep = (wh[:, 0] >= 2.0) & (wh[:, 1] >= 2.0)
        targets = targets[keep].clone()
        if targets.numel():
            boxes, wh = boxes[keep], wh[keep]
            targets[:, 1:3] = ((boxes[:, 0:2] + boxes[:, 2:4]) / 2.0) / self.img_size
            targets[:, 3:5] = wh / self.img_size
        return targets


class DistributedWeightedSampler(Sampler):
    """Draw one deterministic weighted global sample and shard it across DDP ranks."""

    def __init__(self, weights, num_replicas, rank, seed=0, drop_last=True):
        self.weights = weights
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        if drop_last:
            self.num_samples = len(weights) // num_replicas
        else:
            self.num_samples = math.ceil(len(weights) / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=generator).tolist()
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def build_class_balanced_train_sampler(config, dataset, num_replicas=None, rank=None):
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
    if num_replicas is not None and rank is not None:
        sampler = DistributedWeightedSampler(
            weights_tensor, num_replicas=num_replicas, rank=rank,
            seed=int(getattr(config, "SAMPLER_SEED", 20260716)), drop_last=True,
        )
    else:
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
