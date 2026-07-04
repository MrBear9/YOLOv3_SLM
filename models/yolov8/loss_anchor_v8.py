import torch
import torch.nn as nn
import torch.nn.functional as F

from models.geometry import bbox_iou_matrix_xywh, bbox_iou_xywh, weighted_mean
from models.losses import SigmoidFocalLoss


def decode_boxes_to_absolute(pred_boxes, anchors, stride, box_decode_range=2.0):
    """Decode predicted box offsets to absolute coordinates.

    box_decode_range controls the reach of each grid cell:
      1.0 → sigmoid       → [0, 1]      (legacy, single-cell)
      2.0 → sigmoid*2-0.5 → [-0.5, 1.5] (neighbor-cell compatible)
    """
    grid_h, grid_w = pred_boxes.shape[1], pred_boxes.shape[2]
    device = pred_boxes.device
    dtype = pred_boxes.dtype
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=dtype),
        torch.arange(grid_w, device=device, dtype=dtype),
        indexing="ij",
    )
    grid_x = grid_x.contiguous().view(1, grid_h, grid_w, 1)
    grid_y = grid_y.contiguous().view(1, grid_h, grid_w, 1)
    anchor_tensor = anchors.contiguous().view(1, 1, 1, 3, 2).to(device=device, dtype=dtype)
    half = (box_decode_range - 1.0) / 2.0
    x = (torch.sigmoid(pred_boxes[..., 0]) * box_decode_range - half + grid_x) * stride
    y = (torch.sigmoid(pred_boxes[..., 1]) * box_decode_range - half + grid_y) * stride
    w = torch.exp(torch.clamp(pred_boxes[..., 2], min=-8.0, max=8.0)) * anchor_tensor[..., 0]
    h = torch.exp(torch.clamp(pred_boxes[..., 3], min=-8.0, max=8.0)) * anchor_tensor[..., 1]
    return torch.stack([x, y, w, h], dim=-1)


def _focal_bce_unreduced(logits, targets, alpha, gamma):
    """Per-element focal BCE loss (for hard negative mining)."""
    bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    prob = torch.sigmoid(logits.float())
    p_t = targets.float() * prob + (1.0 - targets.float()) * (1.0 - prob)
    alpha_t = targets.float() * alpha + (1.0 - targets.float()) * (1.0 - alpha)
    return alpha_t * torch.pow(1.0 - p_t, gamma) * bce


class YOLOv3AnchorLossForV8Head(nn.Module):
    """Anchor-based YOLO loss with ratio matching or YOLOv7-style SimOTA."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.anchors = torch.tensor(config.ANCHORS, dtype=torch.float32)
        self.num_classes = config.NUM_CLASSES
        self.strides = config.STRIDES
        self.noobj_weight = config.NOOBJ_WEIGHT_BASE
        self.focal_alpha = config.FOCAL_ALPHA
        self.focal_gamma = config.FOCAL_GAMMA
        self.focal_loss = SigmoidFocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="mean")
        self.size_weights = {"small": 1.0, "medium": 1.0, "large": 1.0}
        self._active_match_mode = "ratio"
        self.last_components = {"total": 0.0, "box": 0.0, "obj": 0.0, "noobj": 0.0, "cls": 0.0}
        self.label_smoothing = float(getattr(config, "LABEL_SMOOTHING", 0.0))

        # Uncertainty-weighted multi-task loss (learnable task precisions)
        # Initialised to match static weights so training starts identically.
        if getattr(config, "LOSS_UNCERTAINTY_WEIGHTING", True):
            import math
            bw = float(getattr(config, "BOX_WEIGHT_BASE", 5.0))
            ow = float(getattr(config, "OBJ_WEIGHT_BASE", 2.0))
            cw = float(getattr(config, "CLS_WEIGHT_BASE", 1.8))
            self.log_var_box = nn.Parameter(torch.tensor(-math.log(max(bw, 1e-3))))
            self.log_var_obj = nn.Parameter(torch.tensor(-math.log(max(ow, 1e-3))))
            self.log_var_cls = nn.Parameter(torch.tensor(-math.log(max(cw, 1e-3))))
        else:
            self.box_weight = float(getattr(config, "BOX_WEIGHT_BASE", 5.0))
            self.obj_weight = float(getattr(config, "OBJ_WEIGHT_BASE", 2.0))
            self.cls_weight = float(getattr(config, "CLS_WEIGHT_BASE", 1.8))
            self.log_var_box = self.log_var_obj = self.log_var_cls = None

    def set_epoch_weights(self, epoch):
        weights = self.config.get_dynamic_weights(epoch)
        self.noobj_weight = weights["noobj_weight"]
        if self.log_var_box is None:
            self.box_weight = weights["box_weight"]
            self.obj_weight = weights["obj_weight"]
            self.cls_weight = weights["cls_weight"]
        self.size_weights = weights.get("size_weights", self.size_weights)
        return weights["phase"]

    def _resolve_anchor_match_mode(self):
        mode = str(getattr(self.config, "ANCHOR_MATCH_MODE", "auto")).strip().lower()
        head_type = str(getattr(self.config, "DETECTOR_HEAD_TYPE", "")).strip().lower()
        if mode in {"auto", "default", ""}:
            return "yolo7_simota" if head_type in {"light", "yolo_light"} else "ratio"
        if mode in {"simota", "yolo7", "yolo7_simota", "neighbor_simota", "ota"}:
            return "yolo7_simota"
        return "ratio"

    def _get_size_weight(self, width, height):
        area = float(width * height)
        if self._active_match_mode == "yolo7_simota" and getattr(self.config, "SIMOTA_USE_SIZE_WEIGHT_OVERRIDE", True):
            if area >= self.config.LARGE_OBJ_AREA:
                return float(getattr(self.config, "SIMOTA_LARGE_OBJ_WEIGHT", 0.8))
            if area >= self.config.SMALL_OBJ_AREA:
                return float(getattr(self.config, "SIMOTA_MEDIUM_OBJ_WEIGHT", 1.0))
            return float(getattr(self.config, "SIMOTA_SMALL_OBJ_WEIGHT", 1.5))
        if area >= self.config.LARGE_OBJ_AREA:
            return self.size_weights["large"]
        if area >= self.config.SMALL_OBJ_AREA:
            return self.size_weights["medium"]
        return self.size_weights["small"]

    def _match_anchors_by_ratio(self, tw, th, prepared_scales):
        """Ratio-based anchor matching (Temple-style).

        Returns list of (ratio, scale_idx, anchor_idx) sorted best-first.
        """
        candidates = []
        for scale_idx, scale_data in enumerate(prepared_scales):
            for anchor_idx in range(3):
                aw, ah = scale_data["anchors"][anchor_idx]
                rw = max(tw / (aw + 1e-6), aw / (tw + 1e-6))
                rh = max(th / (ah + 1e-6), ah / (th + 1e-6))
                candidates.append((max(rw, rh), scale_idx, anchor_idx))
        candidates.sort(key=lambda item: item[0])
        return candidates

    def _vectorized_ratio_match(self, gt_boxes, gt_cls_ids, gt_batch_idx, batch_size, prepared_scales, ratio_thresh):
        """向量化锚点匹配：一次性处理所有 GT box，消除 Python 循环。

        gt_boxes: [total_gt, 4] (cx_px, cy_px, w_px, h_px)
        gt_cls_ids: [total_gt] int
        gt_batch_idx: [total_gt] int
        """
        device = gt_boxes.device
        assign_neighbor = getattr(self.config, "ASSIGN_NEIGHBOR_CELLS", True)

        # 预计算所有 anchor 的 (w, h) 和 scale 元信息
        all_aw, all_ah = [], []
        scale_indices, anchor_indices, scale_gw, scale_gh, scale_stride = [], [], [], [], []
        for si, sd in enumerate(prepared_scales):
            for ai in range(3):
                all_aw.append(sd["anchors"][ai][0].item())
                all_ah.append(sd["anchors"][ai][1].item())
                scale_indices.append(si)
                anchor_indices.append(ai)
                scale_gw.append(sd["grid_w"])
                scale_gh.append(sd["grid_h"])
                scale_stride.append(float(sd["stride"]))

        aw_all = torch.tensor(all_aw, device=device, dtype=torch.float32)  # [9]
        ah_all = torch.tensor(all_ah, device=device, dtype=torch.float32)  # [9]
        num_anchors = len(all_aw)

        tw = gt_boxes[:, 2]  # [N]
        th = gt_boxes[:, 3]  # [N]

        # 计算所有 (GT, anchor) 的 ratio [N, 9]
        rw = torch.max(tw.unsqueeze(1) / (aw_all.unsqueeze(0) + 1e-6),
                        aw_all.unsqueeze(0) / (tw.unsqueeze(1) + 1e-6))
        rh = torch.max(th.unsqueeze(1) / (ah_all.unsqueeze(0) + 1e-6),
                        ah_all.unsqueeze(0) / (th.unsqueeze(1) + 1e-6))
        ratios = torch.max(rw, rh)  # [N, 9]

        # 确定有效匹配：ratio < ratio_thresh 的存在时只用它们，否则用所有
        below_thresh = ratios < ratio_thresh  # [N, 9]
        has_valid = below_thresh.any(dim=1)    # [N]
        use_mask = torch.where(has_valid.unsqueeze(1), below_thresh, torch.ones_like(ratios, dtype=torch.bool))

        # 获取匹配的 (gt_idx, anchor_global_idx) 对
        gt_idx, anchor_gidx = torch.where(use_mask)
        if gt_idx.numel() == 0:
            return

        num_pairs = len(gt_idx)
        pair_idx = torch.arange(num_pairs, device=device)
        matched_ratios = ratios[gt_idx, anchor_gidx]  # [M]
        matched_cls = gt_cls_ids[gt_idx]               # [M]
        matched_boxes = gt_boxes[gt_idx]               # [M, 4]
        batch_of_gt = gt_batch_idx[gt_idx]          # [M] batch index per pair

        # 预计算 scale 元信息（用 list 索引，只做一次）
        si_list = [scale_indices[i] for i in anchor_gidx.tolist()]
        ai_list = [anchor_indices[i] for i in anchor_gidx.tolist()]
        gw_list = [scale_gw[i] for i in anchor_gidx.tolist()]
        gh_list = [scale_gh[i] for i in anchor_gidx.tolist()]
        stride_list = [scale_stride[i] for i in anchor_gidx.tolist()]

        matched_si = torch.tensor(si_list, device=device)
        matched_ai = torch.tensor(ai_list, device=device)
        matched_gw = torch.tensor(gw_list, device=device)
        matched_gh = torch.tensor(gh_list, device=device)

        # 计算 grid 坐标
        cx_px = matched_boxes[:, 0]
        cy_px = matched_boxes[:, 1]
        gx = cx_px / torch.tensor(stride_list, device=device)
        gy = cy_px / torch.tensor(stride_list, device=device)
        gi = gx.long().clamp_min(0).clamp_max(matched_gw - 1)
        gj = gy.long().clamp_min(0).clamp_max(matched_gh - 1)

        # 展开邻居 cell
        offsets = [(0, 0)]
        if assign_neighbor:
            offsets += [(-1, 0), (1, 0), (0, -1), (0, 1)]

        all_gi, all_gj, all_pair_idx = [], [], []
        for dx, dy in offsets:
            ni = gi + dx
            nj = gj + dy
            valid = (ni >= 0) & (ni < matched_gw) & (nj >= 0) & (nj < matched_gh)
            if dx != 0 or dy != 0:
                if dx == -1:
                    valid = valid & (gx - gi.float() < 0.5) & (gi > 0)
                elif dx == 1:
                    valid = valid & ((gi + 1).float() - gx < 0.5) & (gi < matched_gw - 1)
                elif dy == -1:
                    valid = valid & (gy - gj.float() < 0.5) & (gj > 0)
                elif dy == 1:
                    valid = valid & ((gj + 1).float() - gy < 0.5) & (gj < matched_gh - 1)
            all_gi.append(ni[valid])
            all_gj.append(nj[valid])
            all_pair_idx.append(pair_idx[valid])

        if not any(t.numel() > 0 for t in all_gi):
            return

        final_gi = torch.cat(all_gi)
        final_gj = torch.cat(all_gj)
        final_pair = torch.cat(all_pair_idx)

        # 用 pair_idx 回查原始匹配信息
        final_gt = gt_idx[final_pair]
        final_ag = anchor_gidx[final_pair]
        final_si = matched_si[final_pair]
        final_ai = matched_ai[final_pair]

        # 计算 size_weight
        areas = gt_boxes[:, 2] * gt_boxes[:, 3]
        large_area = float(getattr(self.config, "LARGE_OBJ_AREA", 1024))
        small_area = float(getattr(self.config, "SMALL_OBJ_AREA", 76))
        size_weight = torch.ones(len(gt_boxes), device=device)
        size_weight[areas >= large_area] = self.size_weights["large"]
        size_weight[(areas >= small_area) & (areas < large_area)] = self.size_weights["medium"]
        size_weight[areas < small_area] = self.size_weights["small"]

        # 批量分配 target
        for si_idx in range(len(prepared_scales)):
            mask = final_si == si_idx
            if not mask.any():
                continue
            sd = prepared_scales[si_idx]
            b_idx = batch_of_gt[final_pair[mask]]
            ai_idx = final_ai[mask]
            yi = final_gj[mask]
            xi = final_gi[mask]

            sd["target_boxes_abs"][b_idx, yi, xi, ai_idx] = gt_boxes[final_gt[mask]]
            sd["target_obj"][b_idx, yi, xi, ai_idx] = 1.0
            sd["target_cls"][b_idx, yi, xi, ai_idx].zero_()
            sd["target_cls"][b_idx, yi, xi, ai_idx, matched_cls[final_pair[mask]]] = 1.0
            sd["target_match_ratio"][b_idx, yi, xi, ai_idx] = matched_ratios[final_pair[mask]]
            sd["target_scale_weight"][b_idx, yi, xi, ai_idx] = size_weight[final_gt[mask]]

    def _neighbor_offsets(self, cx, cy, gw, gh):
        """Return grid offsets for neighbor-cell assignment."""
        gi, gj = int(cx), int(cy)
        offsets = [(gi, gj)]
        assign_neighbor = getattr(self.config, "ASSIGN_NEIGHBOR_CELLS", True)
        if not assign_neighbor:
            return offsets
        if cx - gi < 0.5 and gi > 0:
            offsets.append((gi - 1, gj))
        if (gi + 1) - cx < 0.5 and gi < gw - 1:
            offsets.append((gi + 1, gj))
        if cy - gj < 0.5 and gj > 0:
            offsets.append((gi, gj - 1))
        if (gj + 1) - cy < 0.5 and gj < gh - 1:
            offsets.append((gi, gj + 1))
        return offsets

    def _match_anchors_yolo7_simota(self, gt_idx, tx, ty, tw, th, prepared_scales):
        """GPU-batched YOLOv7-style center candidates + SimOTA top-k for one GT."""
        config = self.config
        img_size = float(config.IMG_SIZE)
        radius = float(getattr(config, "CENTER_PRIOR_RADIUS", 2.5))
        radius = max(radius, 0.5)
        radius_cells = int(torch.ceil(torch.as_tensor(radius)).item())
        center_weight = float(getattr(config, "CENTER_PRIOR_WEIGHT", 0.5))
        iou_thresh = float(getattr(config, "ANCHOR_MATCH_IOU_THRESH", 0.20))
        top_n = int(getattr(config, "SIMOTA_TOP_N", 20))
        max_assign = int(getattr(config, "SIMOTA_MAX_ASSIGN", 15))
        device = tw.device
        dtype = tw.dtype
        gt_box = torch.stack([tx * img_size, ty * img_size, tw, th]).view(1, 4)

        candidate_chunks = []
        for scale_idx, scale_data in enumerate(prepared_scales):
            gw, gh = scale_data["grid_w"], scale_data["grid_h"]
            stride = float(scale_data["stride"])
            gx = tx * gw
            gy = ty * gh
            base_x = torch.floor(gx).long()
            base_y = torch.floor(gy).long()
            offsets = torch.arange(-radius_cells, radius_cells + 1, device=device)
            grid_y, grid_x = torch.meshgrid(base_y + offsets, base_x + offsets, indexing="ij")
            valid = (grid_x >= 0) & (grid_x < gw) & (grid_y >= 0) & (grid_y < gh)
            if not valid.any():
                continue

            dx = (grid_x.to(dtype) + 0.5) - gx
            dy = (grid_y.to(dtype) + 0.5) - gy
            center_dist = torch.sqrt(dx * dx + dy * dy)
            valid = valid & (center_dist <= radius * 1.4143)
            if not valid.any():
                continue

            grid_x = grid_x[valid].long()
            grid_y = grid_y[valid].long()
            center_dist = center_dist[valid].to(dtype)
            num_cells = grid_x.numel()
            anchors = scale_data["anchors"].to(device=device, dtype=dtype)
            anchor_idx = torch.arange(3, device=device).repeat(num_cells)
            cell_x = grid_x.repeat_interleave(3)
            cell_y = grid_y.repeat_interleave(3)
            dist = center_dist.repeat_interleave(3)
            anchor_wh = anchors[anchor_idx]
            cand_box = torch.stack(
                [
                    (cell_x.to(dtype) + 0.5) * stride,
                    (cell_y.to(dtype) + 0.5) * stride,
                    anchor_wh[:, 0],
                    anchor_wh[:, 1],
                ],
                dim=1,
            )
            iou = bbox_iou_xywh(cand_box, gt_box.expand_as(cand_box)).clamp(min=0.0, max=1.0)
            norm_dist = torch.clamp(dist / radius, max=2.0)
            cost = -iou + center_weight * norm_dist
            candidate_chunks.append(
                {
                    "cost": cost,
                    "iou": iou,
                    "scale_idx": torch.full_like(anchor_idx, scale_idx),
                    "anchor_idx": anchor_idx,
                    "grid_x": cell_x,
                    "grid_y": cell_y,
                }
            )

        if not candidate_chunks:
            return []

        cost = torch.cat([chunk["cost"] for chunk in candidate_chunks])
        iou = torch.cat([chunk["iou"] for chunk in candidate_chunks])
        scale_idx = torch.cat([chunk["scale_idx"] for chunk in candidate_chunks])
        anchor_idx = torch.cat([chunk["anchor_idx"] for chunk in candidate_chunks])
        grid_x = torch.cat([chunk["grid_x"] for chunk in candidate_chunks])
        grid_y = torch.cat([chunk["grid_y"] for chunk in candidate_chunks])

        eligible = iou >= iou_thresh
        ranked_idx = torch.where(eligible)[0] if eligible.any() else torch.arange(iou.numel(), device=device)
        ranked_ious = iou[ranked_idx]
        top_count = min(max(top_n, 1), ranked_ious.numel())
        top_ious = torch.topk(ranked_ious, k=top_count, largest=True).values
        dynamic_k = int(torch.ceil(top_ious.sum()).clamp(min=1, max=max_assign).item())
        dynamic_k = min(dynamic_k, ranked_idx.numel())
        selected_local = torch.topk(cost[ranked_idx], k=dynamic_k, largest=False).indices
        selected = ranked_idx[selected_local]
        selected_iou = iou[selected].detach().cpu().tolist()
        selected_scale = scale_idx[selected].long().detach().cpu().tolist()
        selected_anchor = anchor_idx[selected].long().detach().cpu().tolist()
        selected_x = grid_x[selected].long().detach().cpu().tolist()
        selected_y = grid_y[selected].long().detach().cpu().tolist()
        selected_cost = cost[selected].detach().cpu().tolist()
        return list(zip(selected_cost, selected_iou, selected_scale, selected_anchor, selected_x, selected_y))

    def forward(self, predictions, targets):
        config = self.config
        device = predictions[0].device
        total_loss = torch.zeros((), device=device)
        component_sums = {name: torch.zeros((), device=device) for name in ("box", "obj", "noobj", "cls")}
        batch_size = predictions[0].shape[0]
        prepared_scales = []
        box_decode_range = float(getattr(config, "BOX_DECODE_RANGE", 2.0))
        match_mode = self._resolve_anchor_match_mode()
        self._active_match_mode = match_mode
        use_simota = match_mode == "yolo7_simota"
        obj_pos_thresh = float(getattr(config, "SIMOTA_OBJ_POS_THRESH", 0.05)) if use_simota else 0.5
        ratio_thresh = float(getattr(config, "ANCHOR_MATCH_RATIO_THRESH", 3.5))
        hard_neg_ratio = int(getattr(config, "HARD_NEG_RATIO", 30))
        hard_neg_min = int(getattr(config, "HARD_NEG_MIN", 512))

        for i, pred in enumerate(predictions):
            _, _, grid_h, grid_w = pred.shape
            pred = pred.contiguous().permute(0, 2, 3, 1).contiguous().reshape(batch_size, grid_h, grid_w, 3, -1)
            prepared_scales.append(
                {
                    "pred_boxes": pred[..., :4],
                    "pred_obj": pred[..., 4],
                    "pred_cls": pred[..., 5:],
                    "target_boxes_abs": torch.zeros_like(pred[..., :4], dtype=torch.float32),
                    "target_obj": torch.zeros_like(pred[..., 4], dtype=torch.float32),
                    "target_cls": torch.zeros_like(pred[..., 5:], dtype=torch.float32),
                    "target_match_ratio": torch.full_like(pred[..., 4], -1.0 if use_simota else float("inf"), dtype=torch.float32),
                    "target_scale_weight": torch.ones_like(pred[..., 4], dtype=torch.float32),
                    "grid_h": grid_h,
                    "grid_w": grid_w,
                    "stride": self.strides[i],
                    "anchors": self.anchors[i].to(device),
                }
            )

        # ---------- 锚点匹配 ----------
        # 收集所有 GT box（一次批量操作，无 Python 循环）
        all_gt_list, all_cls_list, all_batch_idx = [], [], []
        for b in range(batch_size):
            t = targets[b].to(device)
            if len(t) == 0:
                continue
            boxes_abs = torch.stack([t[:, 1] * config.IMG_SIZE, t[:, 2] * config.IMG_SIZE,
                                     t[:, 3] * config.IMG_SIZE, t[:, 4] * config.IMG_SIZE], dim=1)
            all_gt_list.append(boxes_abs)
            all_cls_list.append(t[:, 0].long())
            all_batch_idx.append(torch.full((len(t),), b, device=device, dtype=torch.long))

        if all_gt_list:
            all_gt_boxes = torch.cat(all_gt_list)      # [total_gt, 4]
            all_gt_cls = torch.cat(all_cls_list)        # [total_gt]
            all_batch_idx_t = torch.cat(all_batch_idx)  # [total_gt]
        else:
            all_gt_boxes = torch.zeros((0, 4), device=device)
            all_gt_cls = torch.zeros((0,), device=device, dtype=torch.long)
            all_batch_idx_t = torch.zeros((0,), device=device, dtype=torch.long)

        if not use_simota and all_gt_boxes.numel() > 0:
            # 向量化 ratio 匹配（默认模式，全 GPU 批处理，无 Python 循环）
            self._vectorized_ratio_match(
                all_gt_boxes, all_gt_cls, all_batch_idx_t, batch_size, prepared_scales, ratio_thresh
            )
        elif use_simota and all_gt_boxes.numel() > 0:
            # SimOTA 仍用逐 box 循环（算法复杂度高，难以向量化）
            for gidx in range(len(all_gt_boxes)):
                bx = all_gt_boxes[gidx]
                cls_id = int(all_gt_cls[gidx].item())
                b = int(all_batch_idx_t[gidx].item())
                tx, ty, tw, th = bx[0] / config.IMG_SIZE, bx[1] / config.IMG_SIZE, bx[2], bx[3]
                size_weight = self._get_size_weight(tw.item(), th.item())
                candidates = self._match_anchors_yolo7_simota(gidx, tx, ty, tw, th, prepared_scales)
                for _, iou_value, scale_idx, anchor_idx, grid_x, grid_y in candidates:
                    sd = prepared_scales[scale_idx]
                    gw, gh = sd["grid_w"], sd["grid_h"]
                    if not (0 <= grid_x < gw and 0 <= grid_y < gh):
                        continue
                    if sd["target_match_ratio"][b, grid_y, grid_x, anchor_idx] >= iou_value:
                        continue
                    sd["target_boxes_abs"][b, grid_y, grid_x, anchor_idx] = bx
                    sd["target_obj"][b, grid_y, grid_x, anchor_idx] = max(iou_value, obj_pos_thresh)
                    sd["target_cls"][b, grid_y, grid_x, anchor_idx].zero_()
                    sd["target_cls"][b, grid_y, grid_x, anchor_idx, cls_id] = 1.0
                    sd["target_match_ratio"][b, grid_y, grid_x, anchor_idx] = iou_value
                    sd["target_scale_weight"][b, grid_y, grid_x, anchor_idx] = size_weight

        gt_boxes_abs_by_batch = [
            all_gt_boxes[all_batch_idx_t == b] if (all_batch_idx_t == b).any()
            else torch.zeros((0, 4), device=device, dtype=predictions[0].dtype)
            for b in range(batch_size)
        ]

        for scale_data in prepared_scales:
            pred_boxes = scale_data["pred_boxes"]
            pred_obj = scale_data["pred_obj"]
            pred_cls = scale_data["pred_cls"]
            target_obj = scale_data["target_obj"]
            target_cls = scale_data["target_cls"]
            target_boxes_abs = scale_data["target_boxes_abs"]
            target_scale_weight = scale_data["target_scale_weight"]

            pred_boxes_abs = decode_boxes_to_absolute(pred_boxes, scale_data["anchors"], scale_data["stride"], box_decode_range)
            obj_mask = target_obj >= obj_pos_thresh

            # Ignore mask: predictions with high IoU to any GT (not positive) are ignored
            ignore_mask = torch.zeros_like(target_obj, dtype=torch.bool)
            for b in range(batch_size):
                gt_boxes_abs = gt_boxes_abs_by_batch[b]
                if gt_boxes_abs.numel() == 0:
                    continue
                flat_pred_boxes = pred_boxes_abs[b].contiguous().reshape(-1, 4)
                max_iou = bbox_iou_matrix_xywh(flat_pred_boxes, gt_boxes_abs).max(dim=1).values
                ignore_mask[b] = max_iou.contiguous().view(scale_data["grid_h"], scale_data["grid_w"], 3) >= config.NOOBJ_IGNORE_IOU
            noobj_mask = (target_obj < obj_pos_thresh) & (~ignore_mask)

            # Positive losses
            if obj_mask.any():
                positive_weights = target_scale_weight[obj_mask]
                ciou = bbox_iou_xywh(pred_boxes_abs[obj_mask].to(torch.float32), target_boxes_abs[obj_mask], ciou=True)
                box_loss = weighted_mean(1.0 - ciou, positive_weights)
                obj_loss = self.focal_loss(pred_obj[obj_mask].to(torch.float32), target_obj[obj_mask], sample_weight=positive_weights)
                cls_loss = self.focal_loss(pred_cls[obj_mask].to(torch.float32), target_cls[obj_mask], sample_weight=positive_weights)
            else:
                box_loss = torch.zeros((), device=device)
                obj_loss = torch.zeros((), device=device)
                cls_loss = torch.zeros((), device=device)

            # Hard negative mining for noobj loss
            if noobj_mask.any():
                neg_losses = _focal_bce_unreduced(pred_obj[noobj_mask].to(torch.float32), target_obj[noobj_mask], self.focal_alpha, self.focal_gamma)
                pos_count = int(obj_mask.sum().item())
                k = max(hard_neg_min, hard_neg_ratio * max(pos_count, 1))
                k = min(k, neg_losses.numel())
                if k > 0:
                    neg_losses = torch.topk(neg_losses.flatten(), k=k, largest=True).values
                noobj_loss = neg_losses.mean()
            else:
                noobj_loss = torch.zeros((), device=device)

            if self.log_var_box is not None:
                prec_box = torch.exp(-self.log_var_box)
                prec_obj = torch.exp(-self.log_var_obj)
                prec_cls = torch.exp(-self.log_var_cls)
                scale_loss = (prec_box * box_loss + prec_obj * obj_loss + prec_cls * cls_loss
                              + self.noobj_weight * noobj_loss
                              + self.log_var_box + self.log_var_obj + self.log_var_cls)
            else:
                scale_loss = self.box_weight * box_loss + self.obj_weight * obj_loss + self.noobj_weight * noobj_loss + self.cls_weight * cls_loss
            if torch.isfinite(scale_loss):
                total_loss = total_loss + scale_loss
                component_sums["box"] = component_sums["box"] + box_loss.detach()
                component_sums["obj"] = component_sums["obj"] + obj_loss.detach()
                component_sums["noobj"] = component_sums["noobj"] + noobj_loss.detach()
                component_sums["cls"] = component_sums["cls"] + cls_loss.detach()

        stats = {name: float(value.item()) for name, value in component_sums.items()}
        stats["total"] = float(total_loss.detach().item())
        self.last_components = stats
        return total_loss, stats