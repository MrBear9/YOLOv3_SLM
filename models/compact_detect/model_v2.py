"""Level-4 fully-decoupled three-network detector (Compact V2).

Replaces the old ECA+FPN+dilation-fusion V2 with three independent
task-specific networks:

  BoxNet — 1→8→16→32 (stride-4) → wh(2) + offset(2)
  ObjNet — 1→8→16→32 (stride-4) → obj(1)      (binary objectness)
  ClsNet — 1→8→16→32 (stride-4) → cls(N)      (class logits)

Each network receives its own learnable 1×1 view of the single-channel
optical input.  Total params ≈ 32–36 K (vs 109 K for V1, 144 K for old V2).

References:
  docs/HeadIdea/检测头多通道复用的临时修改意见.md  §8.2 层级 4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ConvBNAct


# ═══════════════════════════════════════════════════════════════════════════════
# Lightweight building blocks
# ═══════════════════════════════════════════════════════════════════════════════

class MiniBackbone(nn.Module):
    """Tiny stride-4 backbone: 1 → 8 → 16 → 32.  ≈ 5.9 K params."""

    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, 8, 3, stride=2),   # H/2
            ConvBNAct(8, 16, 3, stride=2),            # H/4
            ConvBNAct(16, 32, 3),                      # H/4 (same stride)
        )

    def forward(self, x):
        return self.stem(x)


class MiniHead(nn.Module):
    """Detection head: 32 → 16 (ConvBNAct 3×3) → out (Conv 1×1).  ≈ 4.7 K params."""

    def __init__(self, out_channels):
        super().__init__()
        self.head = nn.Sequential(
            ConvBNAct(32, 16, 3),
            nn.Conv2d(16, out_channels, 1),
        )

    def forward(self, x):
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Level-4 fully-decoupled detector
# ═══════════════════════════════════════════════════════════════════════════════

class CompactOpticalDetectorV2(nn.Module):
    """Level-4 decoupled three-network anchor-free detector.

    Three independent task networks, each with its own learnable 1×1
    view convolution and dedicated backbone + head.

    Output dict:
      "obj"    — (B, 1, H/4, W/4)  raw logits  (binary objectness)
      "cls"    — (B, N, H/4, W/4)  raw logits  (class scores)
      "wh"     — (B, 2, H/4, W/4)  softplus    (box width / height)
      "offset" — (B, 2, H/4, W/4)  raw         (sub-pixel centre offset)
    """

    def __init__(self, config, in_channels=1):
        super().__init__()
        self.config = config
        num_classes = int(config.NUM_CLASSES)

        # ── Learnable per-task views (2 params each) ──
        self.view_box = nn.Conv2d(in_channels, 1, 1, bias=True)
        self.view_obj = nn.Conv2d(in_channels, 1, 1, bias=True)
        self.view_cls = nn.Conv2d(in_channels, 1, 1, bias=True)

        # ── Independent task backbones ──
        self.box_net = MiniBackbone(1)
        self.obj_net = MiniBackbone(1)
        self.cls_net = MiniBackbone(1)

        # ── Task heads ──
        self.wh_head = MiniHead(2)           # width, height
        self.offset_head = MiniHead(2)       # dx, dy
        self.obj_head = MiniHead(1)          # objectness logit
        self.cls_head = MiniHead(num_classes)  # class logits

        self._init_heads()

    def _init_heads(self):
        # Objectness: focal-loss prior so initial output ≈ sigmoid(-2.19) ≈ 0.1
        nn.init.constant_(self.obj_head.head[-1].bias, -2.19)
        nn.init.normal_(self.obj_head.head[-1].weight, std=0.001)

        # Box heads: small normal init, zero bias
        for head in (self.wh_head.head[-1], self.offset_head.head[-1]):
            nn.init.normal_(head.weight, std=0.001)
            nn.init.zeros_(head.bias)

        # Class head: zero bias (uniform prior), small normal weights
        nn.init.normal_(self.cls_head.head[-1].weight, std=0.001)
        nn.init.zeros_(self.cls_head.head[-1].bias)

    def forward(self, x):
        # Per-task view projections
        feat_box = self.box_net(self.view_box(x))
        feat_obj = self.obj_net(self.view_obj(x))
        feat_cls = self.cls_net(self.view_cls(x))

        return {
            "obj":    self.obj_head(feat_obj),
            "cls":    self.cls_head(feat_cls),
            "wh":     F.softplus(self.wh_head(feat_box)),
            "offset": self.offset_head(feat_box),
        }
