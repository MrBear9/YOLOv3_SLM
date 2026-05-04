# models layout

Shared modules live directly under `models/`:

- `teacher.py`: shared ConvTeacher.
- `dataset.py`: YOLO dataset and class-balanced sampler.
- `runtime.py`: device, logging, DataParallel, dataloader helpers.
- `geometry.py`: IoU, NMS, weighted mean.
- `losses.py`: shared loss primitives.
- `teacher_guidance.py`: teacher feature guidance loss and display helpers.
- `training_utils.py`: checkpoint, optimizer, training curve helpers.

YOLOv8-specific modules live under `models/yolov8/`:

- `config_v8.py`: config for `optical_teacher_yolov8_head.py`.
- `head_v8.py`: YOLOv8-style C2f/PAN head with legacy 3-anchor output.
- `loss_anchor_v8.py`: YOLOv3 anchor loss used with the YOLOv8-style head.
- `decode_anchor_v8.py`: anchor decode and NMS for the YOLOv8-style head.
- `metrics_anchor_v8.py`: validation metrics for the anchor-compatible version.
- `visualization_anchor_v8.py`: visualization for the anchor-compatible version.

The current `optical_teacher_yolov8_head.py` intentionally uses:

```text
YOLOv8-style head + legacy YOLOv3 anchor loss
```

It does not import or initialize `optical_teacher_yolo.py`, so running the YOLOv8-head experiment will only create logs under its own configured output directory.
