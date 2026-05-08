import numpy as np
import torch

from models.geometry import apply_nms


def decode_detections_anchor_v8(config, preds, conf_thresh=None, nms_thresh=None, max_det=None, img_size=None):
    conf_thresh = config.CONF_THRESH if conf_thresh is None else conf_thresh
    nms_thresh = config.NMS_THRESH if nms_thresh is None else nms_thresh
    max_det = config.MAX_DET if max_det is None else max_det
    img_size = config.IMG_SIZE if img_size is None else img_size
    batch_size = preds[0].shape[0]
    detections = [[] for _ in range(batch_size)]

    for scale_idx, pred in enumerate(preds):
        grid_h, grid_w = pred.shape[2], pred.shape[3]
        stride = config.STRIDES[scale_idx]
        pred = pred.contiguous().permute(0, 2, 3, 1).contiguous().reshape(batch_size, grid_h, grid_w, 3, -1)
        obj_conf = torch.sigmoid(pred[..., 4])
        cls_conf = torch.sigmoid(pred[..., 5:])
        cls_score, cls_id = cls_conf.max(dim=-1)
        final_conf = obj_conf * cls_score
        valid_mask = (obj_conf >= conf_thresh) & (final_conf >= conf_thresh)
        if not valid_mask.any():
            continue

        device = pred.device
        dtype = pred.dtype
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h, device=device, dtype=dtype),
            torch.arange(grid_w, device=device, dtype=dtype),
            indexing="ij",
        )
        grid_x = grid_x.contiguous().view(1, grid_h, grid_w, 1)
        grid_y = grid_y.contiguous().view(1, grid_h, grid_w, 1)
        anchor_tensor = torch.tensor(config.ANCHORS[scale_idx], device=device, dtype=dtype).contiguous().view(1, 1, 1, 3, 2)

        x_center = (grid_x + torch.sigmoid(pred[..., 0])) * stride
        y_center = (grid_y + torch.sigmoid(pred[..., 1])) * stride
        w = anchor_tensor[..., 0] * torch.exp(torch.clamp(pred[..., 2], min=-8.0, max=8.0))
        h = anchor_tensor[..., 1] * torch.exp(torch.clamp(pred[..., 3], min=-8.0, max=8.0))
        x_center = x_center.clamp(0, img_size - 1)
        y_center = y_center.clamp(0, img_size - 1)
        w = w.clamp(1, img_size)
        h = h.clamp(1, img_size)

        for b in range(batch_size):
            sample_mask = valid_mask[b]
            if not sample_mask.any():
                continue
            detections[b].append(
                torch.stack(
                    [
                        x_center[b][sample_mask],
                        y_center[b][sample_mask],
                        w[b][sample_mask],
                        h[b][sample_mask],
                        final_conf[b][sample_mask],
                        cls_id[b][sample_mask].to(dtype),
                    ],
                    dim=1,
                )
            )

    for b in range(batch_size):
        detections[b] = apply_nms(config, torch.cat(detections[b], dim=0), nms_thresh, max_det) if detections[b] else np.zeros((0, 6), dtype=np.float32)
    return detections
