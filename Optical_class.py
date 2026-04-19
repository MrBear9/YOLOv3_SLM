import os
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class ConfigYOLO:
    DEFAULT_NUM_CLASSES = 4
    DEFAULT_IMG_SIZE = 640
    DEFAULT_RESOLUTION = (640, 640)
    DEFAULT_OPTICAL_MODE = "phase"
    DEFAULT_ENABLE_CONSTRAINT = True
    DEFAULT_TEACHER_INIT_MODE = "checkpoint_or_random"
    DEFAULT_FREEZE_TEACHER = True
    DEFAULT_TEACHER_DEVICE = "cpu"
    DEFAULT_SLM_VORTEX_CHARGE = 0
    DEFAULT_VORTEX_PERTURBATION = 0.1

    PROP_DISTANCE_1 = 0.01
    PROP_DISTANCE_2 = 0.02
    WAVELENGTH = 532e-9
    PIXEL_SIZE = 6.4e-6

    YOLO_HEAD_IN_CHANNELS = 1
    YOLO_HEAD_BASE_CHANNELS = 32
    YOLO_NUM_ANCHORS = 3

    OPTICAL_FIELD_EPS = 1e-8
    OPTICAL_NORM_EPS = 1e-6
    TARGET_EPS = 1e-6

    LOSS_BOX_WEIGHT = 0.05
    LOSS_OBJ_WEIGHT = 1.5
    LOSS_NOOBJ_WEIGHT = 0.5
    LOSS_CLS_WEIGHT = 0.15

    @classmethod
    def detector_output_channels(cls, num_classes):
        return cls.YOLO_NUM_ANCHORS * (4 + 1 + num_classes)


def load_anchor_groups(anchor_yaml_path):
    """Load grouped YOLO anchors from an external yaml file."""
    with open(anchor_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    anchors = cfg.get("anchors")
    if anchors is None:
        raise ValueError(f"'anchors' not found in anchor config: {anchor_yaml_path}")
    if not isinstance(anchors, list) or len(anchors) != 3:
        raise ValueError(f"'anchors' must contain exactly 3 layers: {anchor_yaml_path}")

    normalized = []
    for layer_idx, layer_anchors in enumerate(anchors):
        if not isinstance(layer_anchors, list) or len(layer_anchors) != 3:
            raise ValueError(f"Layer {layer_idx} must contain exactly 3 anchors: {anchor_yaml_path}")

        layer_values = []
        for anchor_idx, anchor in enumerate(layer_anchors):
            if not isinstance(anchor, (list, tuple)) or len(anchor) != 2:
                raise ValueError(f"Anchor {anchor_idx} in layer {layer_idx} must be [w, h]: {anchor_yaml_path}")

            w = int(anchor[0])
            h = int(anchor[1])
            if w <= 0 or h <= 0:
                raise ValueError(f"Anchor {anchor_idx} in layer {layer_idx} must be positive: {anchor_yaml_path}")
            layer_values.append([w, h])
        normalized.append(layer_values)

    return normalized


def resolve_anchor_groups(default_anchors, use_external_anchors=False, anchor_config_path=None):
    """Resolve anchors from defaults or an optional external yaml config."""
    anchors = [[anchor.copy() for anchor in layer] for layer in default_anchors]
    anchor_source = "default"
    if use_external_anchors:
        try:
            anchors = load_anchor_groups(anchor_config_path)
            anchor_source = anchor_config_path
        except Exception as exc:
            anchors = [[anchor.copy() for anchor in layer] for layer in default_anchors]
            anchor_source = f"default (external load failed: {exc})"
    return anchors, anchor_source


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("teacher_state_dict", "model_state_dict", "state_dict", "model"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def load_teacher_checkpoint(teacher, checkpoint_path, device="cpu"):
    if not checkpoint_path:
        return False, "Teacher checkpoint: not configured"
    if not os.path.exists(checkpoint_path):
        return False, f"Teacher checkpoint not found: {checkpoint_path}"

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = extract_state_dict(checkpoint)
    teacher_state = teacher.state_dict()
    compatible_state = {}

    for key, value in state_dict.items():
        normalized_key = key[8:] if key.startswith("teacher.") else key
        if normalized_key in teacher_state and teacher_state[normalized_key].shape == value.shape:
            compatible_state[normalized_key] = value

    if len(compatible_state) == 0:
        return False, f"No compatible ConvTeacher weights found in: {checkpoint_path}"

    teacher.load_state_dict({**teacher_state, **compatible_state}, strict=False)
    return True, f"Loaded {len(compatible_state)} teacher tensors from: {checkpoint_path}"


class SLMLayer(nn.Module):
    """Spatial light modulator layer with optional vortex initialization."""

    def __init__(self, resolution, mode="phase", vortex_charge=0, vortex_perturbation=0.1):
        super().__init__()
        assert mode in ["phase", "amp_phase"]
        self.mode = mode
        self.vortex_charge = vortex_charge
        self.vortex_perturbation = vortex_perturbation

        self.phase_raw = nn.Parameter(self._init_phase(resolution))

        if self.mode == "amp_phase":
            self.amp_raw = nn.Parameter(torch.rand(1, 1, *resolution))
        else:
            self.register_parameter("amp_raw", None)

    def _init_phase(self, resolution):
        if self.vortex_charge == 0:
            return torch.rand(1, 1, *resolution) * 2 * np.pi

        height, width = resolution
        y = torch.linspace(-1.0, 1.0, height)
        x = torch.linspace(-1.0, 1.0, width)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        theta = torch.atan2(yy, xx)
        vortex_phase = self.vortex_charge * theta
        noise = torch.randn_like(vortex_phase) * (self.vortex_perturbation * np.pi)

        init_phase = torch.remainder(vortex_phase + noise, 2 * np.pi)
        return init_phase.unsqueeze(0).unsqueeze(0)

    def forward(self, field):
        phase = torch.remainder(self.phase_raw, 2 * np.pi)
        mod = torch.exp(1j * phase)

        if self.mode == "amp_phase":
            amp = torch.sigmoid(self.amp_raw)
            mod = mod * amp

        return field * mod


class ASMPropagation(nn.Module):
    """Angular spectrum propagation."""

    def __init__(self, distance, wavelength, pixel_size, resolution):
        super().__init__()
        fx = torch.fft.fftfreq(resolution[0], pixel_size)
        fy = torch.fft.fftfreq(resolution[1], pixel_size)
        FX, FY = torch.meshgrid(fx, fy, indexing="ij")
        k2 = 1 / wavelength**2 - FX**2 - FY**2
        k2 = torch.clamp(k2, min=0)
        H = torch.exp(1j * 2 * np.pi * distance * torch.sqrt(k2))
        self.register_buffer("H", H)

    def forward(self, field):
        return torch.fft.ifft2(torch.fft.fft2(field) * self.H)


class ConvTeacher(nn.Module):
    """Teacher network that generates a dense constraint map."""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.project = nn.Conv2d(32, 1, kernel_size=1, bias=False)

    def forward(self, x):
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)

        f = self.conv1(x)
        f = self.conv2(f)
        f = self.conv3(f)
        f = self.refine(f)
        f = self.project(f)
        f = torch.abs(f)
        f = F.interpolate(f, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(f)


class OpticalFrontend(nn.Module):
    """Two-layer optical frontend."""

    def __init__(
        self,
        resolution=None,
        mode=None,
        slm1_vortex_charge=None,
        slm2_vortex_charge=None,
        vortex_perturbation=None,
    ):
        super().__init__()
        resolution = resolution or ConfigYOLO.DEFAULT_RESOLUTION
        mode = mode or ConfigYOLO.DEFAULT_OPTICAL_MODE
        slm1_vortex_charge = (
            ConfigYOLO.DEFAULT_SLM_VORTEX_CHARGE if slm1_vortex_charge is None else slm1_vortex_charge
        )
        slm2_vortex_charge = (
            ConfigYOLO.DEFAULT_SLM_VORTEX_CHARGE if slm2_vortex_charge is None else slm2_vortex_charge
        )
        vortex_perturbation = (
            ConfigYOLO.DEFAULT_VORTEX_PERTURBATION
            if vortex_perturbation is None
            else vortex_perturbation
        )
        self.slm1 = SLMLayer(
            resolution,
            mode,
            vortex_charge=slm1_vortex_charge,
            vortex_perturbation=vortex_perturbation,
        )
        self.prop1 = ASMPropagation(
            ConfigYOLO.PROP_DISTANCE_1,
            ConfigYOLO.WAVELENGTH,
            ConfigYOLO.PIXEL_SIZE,
            resolution,
        )
        self.slm2 = SLMLayer(
            resolution,
            mode,
            vortex_charge=slm2_vortex_charge,
            vortex_perturbation=vortex_perturbation,
        )
        self.prop2 = ASMPropagation(
            ConfigYOLO.PROP_DISTANCE_2,
            ConfigYOLO.WAVELENGTH,
            ConfigYOLO.PIXEL_SIZE,
            resolution,
        )
        self.enable_norm = False

    def forward(self, intensity):
        amp = torch.sqrt(intensity.clamp(min=0) + ConfigYOLO.OPTICAL_FIELD_EPS)
        field = torch.complex(amp, torch.zeros_like(amp))

        field = self.prop1(self.slm1(field))
        field = self.prop2(self.slm2(field))

        out = torch.abs(field) ** 2
        if self.enable_norm:
            out = out / (out.mean(dim=[2, 3], keepdim=True) + ConfigYOLO.OPTICAL_NORM_EPS)
        return out


class LightConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class YOLOLightHead(nn.Module):
    """Lightweight FPN-style detector head."""

    def __init__(self, in_channels=None, out_channels=None):
        super().__init__()
        in_channels = ConfigYOLO.YOLO_HEAD_IN_CHANNELS if in_channels is None else in_channels
        out_channels = (
            ConfigYOLO.detector_output_channels(ConfigYOLO.DEFAULT_NUM_CLASSES)
            if out_channels is None
            else out_channels
        )
        base_ch = ConfigYOLO.YOLO_HEAD_BASE_CHANNELS

        self.init_conv = LightConvBlock(in_channels, base_ch, kernel_size=3, stride=1)
        self.down_to_p5 = nn.Sequential(
            LightConvBlock(base_ch, base_ch * 2, stride=2),
            LightConvBlock(base_ch * 2, base_ch * 4, stride=2),
            LightConvBlock(base_ch * 4, base_ch * 8, stride=2),
            LightConvBlock(base_ch * 8, base_ch * 8, stride=2),
            LightConvBlock(base_ch * 8, base_ch * 8, stride=2),
        )
        self.up_p5_to_p4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse_p4 = LightConvBlock(base_ch * 8 + base_ch * 8, base_ch * 4)

        self.up_p4_to_p3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse_p3 = LightConvBlock(base_ch * 4 + base_ch * 8, base_ch * 2)

        self.head_p5 = nn.Conv2d(base_ch * 8, out_channels, 1)
        self.head_p4 = nn.Conv2d(base_ch * 4, out_channels, 1)
        self.head_p3 = nn.Conv2d(base_ch * 2, out_channels, 1)

    def forward(self, x):
        x_init = self.init_conv(x)

        x320 = self.down_to_p5[0](x_init)
        x160 = self.down_to_p5[1](x320)
        x80 = self.down_to_p5[2](x160)
        x40 = self.down_to_p5[3](x80)
        p5 = self.down_to_p5[4](x40)

        p5_up = self.up_p5_to_p4(p5)
        p4_fuse = torch.cat([p5_up, x40], dim=1)
        p4 = self.fuse_p4(p4_fuse)

        p4_up = self.up_p4_to_p3(p4)
        p3_fuse = torch.cat([p4_up, x80], dim=1)
        p3 = self.fuse_p3(p3_fuse)

        p5_out = self.head_p5(p5)
        p4_out = self.head_p4(p4)
        p3_out = self.head_p3(p3)
        return p3_out, p4_out, p5_out


class OpticalYOLOv3(nn.Module):
    """Optical frontend + detector head + ConvTeacher constraint."""

    def __init__(
        self,
        num_classes=None,
        img_size=None,
        optical_mode=None,
        enable_constraint=None,
        teacher=None,
        teacher_checkpoint=None,
        teacher_init_mode=None,
        freeze_teacher=None,
        teacher_device=None,
        slm1_vortex_charge=None,
        slm2_vortex_charge=None,
        vortex_perturbation=None,
    ):
        super().__init__()
        num_classes = ConfigYOLO.DEFAULT_NUM_CLASSES if num_classes is None else num_classes
        img_size = ConfigYOLO.DEFAULT_IMG_SIZE if img_size is None else img_size
        optical_mode = ConfigYOLO.DEFAULT_OPTICAL_MODE if optical_mode is None else optical_mode
        enable_constraint = (
            ConfigYOLO.DEFAULT_ENABLE_CONSTRAINT if enable_constraint is None else enable_constraint
        )
        teacher_init_mode = (
            ConfigYOLO.DEFAULT_TEACHER_INIT_MODE if teacher_init_mode is None else teacher_init_mode
        )
        freeze_teacher = ConfigYOLO.DEFAULT_FREEZE_TEACHER if freeze_teacher is None else freeze_teacher
        teacher_device = ConfigYOLO.DEFAULT_TEACHER_DEVICE if teacher_device is None else teacher_device
        self.img_size = img_size
        self.enable_constraint = enable_constraint
        self.freeze_teacher = freeze_teacher
        self.teacher_checkpoint = teacher_checkpoint
        self.teacher_init_mode = teacher_init_mode
        self.teacher_loaded = False
        self.teacher_status_message = ""

        self.optical_frontend = OpticalFrontend(
            resolution=(img_size, img_size),
            mode=optical_mode,
            slm1_vortex_charge=slm1_vortex_charge,
            slm2_vortex_charge=slm2_vortex_charge,
            vortex_perturbation=vortex_perturbation,
        )

        out_channels = ConfigYOLO.detector_output_channels(num_classes)
        self.detector = YOLOLightHead(
            in_channels=ConfigYOLO.YOLO_HEAD_IN_CHANNELS,
            out_channels=out_channels,
        )

        self.teacher = teacher if teacher is not None else ConvTeacher()
        self.num_classes = num_classes
        self._initialize_teacher(teacher_device)

    def _initialize_teacher(self, device):
        init_mode = self.teacher_init_mode.lower()
        if init_mode not in {"checkpoint", "checkpoint_or_random", "random"}:
            raise ValueError(
                "teacher_init_mode must be one of: checkpoint, checkpoint_or_random, random"
            )

        status_messages = []
        if init_mode in {"checkpoint", "checkpoint_or_random"}:
            loaded, message = load_teacher_checkpoint(self.teacher, self.teacher_checkpoint, device=device)
            self.teacher_loaded = loaded
            status_messages.append(message)
            if not loaded and init_mode == "checkpoint":
                raise FileNotFoundError(message)
        else:
            status_messages.append("Teacher initialized from random weights.")

        if not self.teacher_loaded and init_mode == "checkpoint_or_random":
            status_messages.append("Teacher fallback: random initialization.")

        for param in self.teacher.parameters():
            param.requires_grad = not self.freeze_teacher

        if self.freeze_teacher:
            self.teacher.eval()

        status_messages.append(f"Teacher status: {'frozen' if self.freeze_teacher else 'trainable'}")
        self.teacher_status_message = " | ".join(status_messages)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_teacher:
            self.teacher.eval()
        return self

    def forward(self, x):
        if x.shape[1] == 3:
            intensity = x.mean(dim=1, keepdim=True)
        else:
            intensity = x

        optical_feature = self.optical_frontend(intensity)

        if self.enable_constraint:
            teacher_context = torch.no_grad() if self.freeze_teacher else nullcontext()
            with teacher_context:
                constraint_target = self.teacher(intensity)
        else:
            constraint_target = None

        p3, p4, p5 = self.detector(optical_feature)
        return p3, p4, p5, optical_feature, constraint_target

    def enable_normalization(self, enable=True):
        self.optical_frontend.enable_norm = enable

    def enable_constraint_loss(self, enable=True):
        self.enable_constraint = enable


class YOLOLoss(nn.Module):
    def __init__(self, box_weight=None, obj_weight=None, noobj_weight=None, cls_weight=None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.bce = nn.BCEWithLogitsLoss()
        self.box_weight = ConfigYOLO.LOSS_BOX_WEIGHT if box_weight is None else box_weight
        self.obj_weight = ConfigYOLO.LOSS_OBJ_WEIGHT if obj_weight is None else obj_weight
        self.noobj_weight = ConfigYOLO.LOSS_NOOBJ_WEIGHT if noobj_weight is None else noobj_weight
        self.cls_weight = ConfigYOLO.LOSS_CLS_WEIGHT if cls_weight is None else cls_weight

    def forward(self, pred, target, batch_size):
        pred = pred.permute(0, 2, 3, 1).reshape(batch_size, pred.shape[2], pred.shape[3], 3, -1)

        obj_mask = target[..., 4] == 1
        noobj_mask = target[..., 4] == 0

        if obj_mask.any():
            pred_pos = pred[obj_mask]
            target_pos = target[obj_mask]

            xy_loss = self.bce(pred_pos[:, 0:2], target_pos[:, 0:2])
            wh_loss = self.smooth_l1(pred_pos[:, 2:4], target_pos[:, 2:4])
            box_loss = xy_loss + wh_loss

            obj_loss_pos = self.bce(pred_pos[:, 4], target_pos[:, 4])
            cls_loss = self.bce(pred_pos[:, 5:], target_pos[:, 5:])
        else:
            box_loss = pred[..., :4].sum() * 0.0
            obj_loss_pos = pred[..., 4].sum() * 0.0
            cls_loss = pred[..., 5:].sum() * 0.0

        if noobj_mask.any():
            noobj_loss = self.bce(pred[noobj_mask][:, 4], target[noobj_mask][:, 4])
        else:
            noobj_loss = pred[..., 4].sum() * 0.0

        obj_loss = obj_loss_pos + self.noobj_weight * noobj_loss
        total_loss = self.box_weight * box_loss + self.obj_weight * obj_loss + self.cls_weight * cls_loss
        return total_loss, box_loss, obj_loss, cls_loss


def build_target(targets, anchors, stride, num_classes, img_size, device):
    batch_size = targets.shape[0]
    h, w = img_size // stride, img_size // stride
    num_anchors = len(anchors)
    eps = ConfigYOLO.TARGET_EPS

    target_tensor = torch.zeros((batch_size, h, w, num_anchors, 5 + num_classes), device=device)

    for b in range(batch_size):
        for t in targets[b]:
            cls, cx, cy, bw, bh = t
            if bw.item() <= 0 or bh.item() <= 0:
                continue

            gx = cx * w
            gy = cy * h
            gw = torch.clamp(bw * img_size / stride, min=eps)
            gh = torch.clamp(bh * img_size / stride, min=eps)

            i = min(w - 1, max(0, int(gx.item())))
            j = min(h - 1, max(0, int(gy.item())))

            best_idx = 0
            best_iou = 0.0
            for a_idx, (aw, ah) in enumerate(anchors):
                aw_s = aw / stride
                ah_s = ah / stride
                inter = min(gw.item(), aw_s) * min(gh.item(), ah_s)
                union = gw.item() * gh.item() + aw_s * ah_s - inter
                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
                    best_idx = a_idx

            aw, ah = anchors[best_idx]
            aw_s = aw / stride
            ah_s = ah / stride
            tx = gx - i
            ty = gy - j
            tw = torch.log(gw / aw_s + eps)
            th = torch.log(gh / ah_s + eps)

            target_tensor[b, j, i, best_idx, 0:4] = torch.stack([tx, ty, tw, th])
            target_tensor[b, j, i, best_idx, 4] = 1
            target_tensor[b, j, i, best_idx, 5 + int(cls.item())] = 1

    return target_tensor


def visualize_optical_detection(model, dataloader, device, save_path="optical_detection.png", num_samples=4):
    model.eval()

    samples = []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            imgs, targets = batch
            if isinstance(imgs, (list, tuple)):
                imgs = torch.stack(imgs)
            for idx in range(min(num_samples, len(imgs))):
                samples.append((imgs[idx].to(device), targets[idx]))
        else:
            batch_items = list(batch)[:num_samples]
            imgs, targets = zip(*batch_items)
            imgs = torch.stack(imgs).to(device)
            samples.extend(list(zip(imgs, targets)))
        break

    fig, axes = plt.subplots(len(samples), 4, figsize=(16, 4 * max(1, len(samples))))
    if len(samples) == 1:
        axes = axes.reshape(1, -1)

    for idx, (img, target) in enumerate(samples):
        with torch.no_grad():
            p3, p4, p5, optical_feat, constraint_target = model(img.unsqueeze(0))

        img_np = img.permute(1, 2, 0).cpu().numpy()
        optical_np = optical_feat.squeeze().cpu().numpy()
        teacher_np = (
            constraint_target.squeeze().cpu().numpy()
            if constraint_target is not None
            else np.zeros_like(optical_np)
        )
        prediction_np = torch.sigmoid(p3[0, 4::(5 + model.num_classes)]).max(dim=0).values.cpu().numpy()

        axes[idx, 0].imshow(img_np)
        axes[idx, 0].set_title(f"Input {idx + 1}")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(optical_np, cmap="hot")
        axes[idx, 1].set_title("Optical Feature")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(teacher_np, cmap="hot")
        axes[idx, 2].set_title("Teacher Constraint")
        axes[idx, 2].axis("off")

        axes[idx, 3].imshow(prediction_np, cmap="hot")
        axes[idx, 3].set_title("Prediction Response")
        axes[idx, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to: {save_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = OpticalYOLOv3(num_classes=4, img_size=640, optical_mode="phase").to(device)
    x = torch.randn(2, 3, 640, 640).to(device)
    p3, p4, p5, optical_feat, constraint_target = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Optical feature shape: {optical_feat.shape}")
    if constraint_target is not None:
        print(f"Teacher constraint shape: {constraint_target.shape}")
    print(f"P3 shape: {p3.shape}")
    print(f"P4 shape: {p4.shape}")
    print(f"P5 shape: {p5.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
