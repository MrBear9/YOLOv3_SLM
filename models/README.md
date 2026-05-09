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

SLM phase training has a few hardware-facing safeguards:

- SLM phase parameters use `PHASE_WEIGHT_DECAY = 0.0`; ordinary weight decay would pull `phase_raw` toward zero and can collapse the phase map.
- `OpticalStudent` now supports staged normalization. By default `STUDENT_NORM_SCHEDULE = "late"` uses `STUDENT_NORM_EARLY_MODE = "none"` during the first `student_only` stage, then switches to the deployment normalization mode `STUDENT_NORM_MODE = "max"` for detector-only and joint training. This lets the phase layers first learn an unconstrained optical intensity pattern, then trains the detector on the normalized feature distribution used at inference.
- `losses_slm.py` includes a Pearson correlation feature term. This gives the SLM student a scale/offset-insensitive structural target, reducing the hard lower bound caused by absolute-value MSE terms when student and teacher feature scales differ.
- The phase diversity loss is now a lightweight regularizer instead of a dominant objective: `LOSS_PHASE_DIVERSITY_WEIGHT = 0.15`, with relaxed std/span/circular-std targets. It still rejects nearly flat phase maps, but leaves more room for feature matching.
- `losses_slm.py` penalizes low circular phase spread and excessive concentration near the `0 / 2*pi` wrap boundary.
- `optical_student_best.pth` is saved only when the SLM phase quality check passes, including centered std/span, circular std, and near-boundary ratio checks.
- `detector_best.pth` can still carry the paired student weights used by that detector, but it does not overwrite the standalone SLM extraction checkpoint unless the phase quality check passes.
- `optical_layers.py` supports vortex phase initialization for both SLM layers. This gives the phase optimizer a non-flat optical pattern instead of relying only on random phase.
- `ConfigSLM.SLM_INIT_MODE` controls initialization: `random`, `vortex`, `checkpoint`, or `vortex_checkpoint`. `checkpoint` and `vortex_checkpoint` load `ConfigSLM.SLM_INIT_CHECKPOINT` into the student but still start training from the first student-only stage.
- Teacher `abs` is intentionally unchanged for now. If the teacher is retrained later, removing `abs` may make the target feature range more compatible with physical optical intensity, but that is a separate experiment.

Useful runtime overrides:

```powershell
$env:OPTICAL_SLM_INIT_MODE="vortex"
$env:OPTICAL_SLM_STUDENT_NORM_SCHEDULE="late"
$env:OPTICAL_SLM_STUDENT_NORM_MODE="max"
python .\optical_slm_yolov8_head.py
```

For an ablation that keeps normalization enabled in every stage:

```powershell
$env:OPTICAL_SLM_STUDENT_NORM_SCHEDULE="always"
python .\optical_slm_yolov8_head.py
```

To continue from an earlier student checkpoint while still retraining from the first student stage:

```powershell
$env:OPTICAL_SLM_INIT_MODE="checkpoint"
$env:OPTICAL_SLM_INIT_CHECKPOINT="output\OpticalSLM_YOLOv8Head_student\optical_student_best.pth"
python .\optical_slm_yolov8_head.py
```

The SLM checkpoint payload intentionally avoids a top-level `"phase"` key, so phase extraction scripts can search tensor parameter names such as `phase_raw` without accidentally matching a string metadata field. It also stores wrapped `*_wrapped_slm_0_2pi` tensors for SLM loading.

`Optical_yolo_detect/Optical_SLM_yolov8_head_model.py` loads those two SLM outputs for visualization/evaluation, and its local README explains the expected input/output files.
