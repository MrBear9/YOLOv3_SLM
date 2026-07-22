import os
import math
import multiprocessing as mp
import random
from collections import Counter

import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms import functional as TF
import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def identity_collate(batch):
    return batch


def letterbox_content_bounds(image_size, output_size):
    src_w, src_h = image_size
    scale = min(output_size / src_w, output_size / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    left, top = (output_size - new_w) // 2, (output_size - new_h) // 2
    return left, top, left + new_w, top + new_h


def letterbox_image_targets(img, targets, image_size, fill=114):
    src_w, src_h = img.size
    scale = min(image_size / src_w, image_size / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left, top, _, _ = letterbox_content_bounds((src_w, src_h), image_size)
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
        self.copy_paste_enabled = self.augment and bool(getattr(config, "SOLDIER_COPY_PASTE", False))
        self.copy_paste_class = int(getattr(config, "SOLDIER_CLASS_ID", 1))
        self._copy_paste_donors = []
        self._copy_paste_attempted = mp.Value("q", 0)
        self._copy_paste_images = mp.Value("q", 0)
        self._copy_paste_objects = mp.Value("q", 0)

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
        copy_paste_donors = []
        donor_max_area = float(getattr(self.config, "SOLDIER_COPY_PASTE_AREA_MAX", 32 * 32))
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
                            if (
                                self.copy_paste_enabled and cls_id == self.copy_paste_class
                                and len(parts) >= 5
                            ):
                                box = tuple(float(value) for value in parts[1:5])
                                if box[2] * box[3] * self.img_size ** 2 <= donor_max_area:
                                    copy_paste_donors.append((img_path, box))
            if len(image_class_counter) == 0:
                empty_image_count += 1
            image_class_counters.append(image_class_counter)
        self._sampling_metadata = {
            "image_class_counters": image_class_counters,
            "class_box_counts": class_box_counts,
            "empty_image_count": empty_image_count,
            "copy_paste_donors": copy_paste_donors,
        }
        self._copy_paste_donors = copy_paste_donors
        return self._sampling_metadata

    def __getitem__(self, idx):
        if self.copy_paste_enabled and not self._copy_paste_donors:
            self.get_sampling_metadata()
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
        content_bounds = letterbox_content_bounds(img.size, self.img_size)
        img, targets = self._letterbox(img, targets)
        if self.augment:
            img, targets = self._copy_paste_small_soldiers(img, targets, content_bounds)
            img, targets = self._augment(img, targets)
        img_tensor = TF.to_tensor(TF.to_grayscale(img, num_output_channels=1))
        if self.augment and random.random() < float(getattr(self.config, "AUG_NOISE_PROB", 0.10)):
            std = float(getattr(self.config, "AUG_NOISE_STD", 0.01))
            img_tensor = (img_tensor + torch.randn_like(img_tensor) * std).clamp_(0.0, 1.0)
        return img_tensor, targets

    def _letterbox(self, img, targets):
        return letterbox_image_targets(img, targets, self.img_size)

    def get_copy_paste_stats(self, reset=False):
        values = {
            "attempted_images": self._copy_paste_attempted.value,
            "successful_images": self._copy_paste_images.value,
            "pasted_objects": self._copy_paste_objects.value,
        }
        if reset:
            for counter in (self._copy_paste_attempted, self._copy_paste_images, self._copy_paste_objects):
                with counter.get_lock():
                    counter.value = 0
        return values

    @staticmethod
    def _box_ioa(candidate, existing):
        if existing.numel() == 0:
            return candidate.new_zeros(0)
        lt = torch.maximum(candidate[:2], existing[:, :2])
        rb = torch.minimum(candidate[2:], existing[:, 2:])
        intersection = (rb - lt).clamp(min=0).prod(-1)
        candidate_area = (candidate[2:] - candidate[:2]).clamp(min=1).prod()
        return intersection / candidate_area

    def _copy_paste_small_soldiers(self, img, targets, content_bounds=None):
        if not self._copy_paste_donors or random.random() >= float(
            getattr(self.config, "SOLDIER_COPY_PASTE_PROB", 0.12)
        ):
            return img, targets
        with self._copy_paste_attempted.get_lock():
            self._copy_paste_attempted.value += 1

        existing = torch.as_tensor(self._targets_to_xyxy(targets)).clone()
        pasted = []
        max_objects = int(getattr(self.config, "SOLDIER_COPY_PASTE_MAX_OBJECTS", 1))
        max_ioa = float(getattr(self.config, "SOLDIER_COPY_PASTE_IOA_MAX", 0.15))
        scale_min = float(getattr(self.config, "SOLDIER_COPY_PASTE_SCALE_MIN", 0.9))
        scale_max = float(getattr(self.config, "SOLDIER_COPY_PASTE_SCALE_MAX", 1.10))
        attempts = max_objects * 8
        content_left, content_top, content_right, content_bottom = content_bounds or (0, 0, self.img_size, self.img_size)
        for _ in range(attempts):
            if len(pasted) >= max_objects:
                break
            donor_path, donor_box = random.choice(self._copy_paste_donors)
            with Image.open(donor_path) as donor_image:
                donor_image = donor_image.convert("RGB")
                dw, dh = donor_image.size
                cx, cy, bw, bh = donor_box
                crop_box = (
                    max(0, round((cx - bw / 2) * dw)), max(0, round((cy - bh / 2) * dh)),
                    min(dw, round((cx + bw / 2) * dw)), min(dh, round((cy + bh / 2) * dh)),
                )
                crop = donor_image.crop(crop_box).copy()
            if crop.width < 2 or crop.height < 2:
                continue
            scale = random.uniform(scale_min, scale_max)
            paste_w = max(2, round(donor_box[2] * self.img_size * scale))
            paste_h = max(2, round(donor_box[3] * self.img_size * scale))
            if paste_w >= content_right - content_left or paste_h >= content_bottom - content_top:
                continue
            x1 = random.randint(content_left, content_right - paste_w)
            y1 = random.randint(content_top, content_bottom - paste_h)
            candidate = existing.new_tensor([x1, y1, x1 + paste_w, y1 + paste_h])
            overlaps_existing = existing.numel() and self._box_ioa(candidate, existing).max().item() > max_ioa
            if overlaps_existing:
                continue

            crop = crop.resize((paste_w, paste_h), Image.Resampling.BILINEAR)
            feather = int(getattr(self.config, "SOLDIER_COPY_PASTE_EDGE_FEATHER", 2))
            mask = Image.new("L", (paste_w, paste_h), 0)
            inset = min(feather, max((min(paste_w, paste_h) - 1) // 2, 0))
            ImageDraw.Draw(mask).rectangle((inset, inset, paste_w - 1 - inset, paste_h - 1 - inset), fill=255)
            if inset > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(feather))
            img.paste(crop, (x1, y1), mask)
            existing = torch.cat((existing, candidate.unsqueeze(0)), dim=0)
            pasted.append([
                self.copy_paste_class,
                (x1 + paste_w / 2) / self.img_size, (y1 + paste_h / 2) / self.img_size,
                paste_w / self.img_size, paste_h / self.img_size,
            ])

        if pasted:
            targets = torch.cat((targets, targets.new_tensor(pasted)), dim=0)
            with self._copy_paste_images.get_lock():
                self._copy_paste_images.value += 1
            with self._copy_paste_objects.get_lock():
                self._copy_paste_objects.value += len(pasted)
        return img, targets

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
        "copy_paste_donors": len(metadata.get("copy_paste_donors", [])),
    }
    return sampler, summary
