# models layout

Shared modules live directly under `models/`:

- `teacher.py`: shared ConvTeacher.
- `dataset.py`: YOLO dataset and class-balanced sampler.
- `runtime.py`: device, logging, DataParallel, dataloader helpers.
- `geometry.py`: IoU, NMS, weighted mean.
- `losses.py`: shared loss primitives.
- `teacher_guidance.py`: teacher feature guidance loss and display helpers.
- `training_utils.py`: checkpoint, optimizer, training curve helpers.

SLM student-specific modules live under `models/SLM/`:

- `config_slm.py`: config for `optical_slm_yolov8_head.py`.
- `optical_layers.py`: differentiable SLM modulation and ASM propagation layers.
- `dataset_slm.py`: SLM training/evaluation datasets.
- `losses_slm.py`: student feature, response, and SLM regularization losses.
- `utils_slm.py`: SLM checkpoint loading/saving, parameter grouping, and phase summaries.

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

`optical_slm_yolov8_head.py` trains the SLM optical student and a YOLOv8-style anchor detector from a checkpoint produced by `optical_teacher_yolov8_head.py`. It saves only:

- `optical_student_best.pth`: best SLM/student weights.
- `detector_best.pth`: best detector weights.

The SLM checkpoint payload intentionally avoids a top-level `"phase"` key, so phase extraction scripts can search tensor parameter names such as `phase_raw` without accidentally matching a string metadata field.

`Optical_yolo_detect/Optical_SLM_yolov8_head_model.py` loads those two SLM outputs for visualization/evaluation, and its local README explains the expected input/output files.
