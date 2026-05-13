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

- `detector_best.pth`: recommended inference checkpoint, selected by validation `mAP50`.
- `optical_student_best.pth`: standalone mirror of the student weights paired with `detector_best.pth`, mainly for SLM phase extraction.

SLM phase training has a few hardware-facing safeguards:

- SLM phase parameters use `PHASE_WEIGHT_DECAY = 0.0`; ordinary weight decay would pull `phase_raw` toward zero and can collapse the phase map.
- `OpticalStudent` now defaults to `STUDENT_NORM_SCHEDULE = "late"` with `STUDENT_NORM_EARLY_MODE = "none"`, because direct `max` normalization during `student_only` can suppress useful optical feature learning in this setup.
- Training uses a short `student_adapt_max` stage after `student_only`. In this stage the detector stays frozen, `STUDENT_NORM_MODE = "max"` is enabled, and only the SLM student is optimized. Its purpose is to reduce the feature mismatch caused by switching from unnormalized optical intensity to the normalized feature distribution used by detector training and inference.
- Detector-only training is intentionally short by default (`DETECTOR_ONLY_EPOCHS = 40`) with a lower detector LR (`DETECTOR_LR = 2e-4`) because the detector quickly overfits the fixed SLM feature map after the first mAP plateau. It also supports mAP-based early stopping with `DETECTOR_EARLY_STOP_PATIENCE`.
- Joint training gives more weight to feature preservation and less to detection loss than before (`FEATURE_LOSS_WEIGHT_JOINT = 0.70`, `DETECTION_LOSS_WEIGHT_JOINT = 0.80`) so it is less likely to reduce train detection loss while damaging validation loss.
- `optical_slm_yolov8_head.py` supports per-stage `CosineAnnealingLR` with `ETA_MIN = 1e-6`, matching the teacher YOLOv8-head training style. A new scheduler is created whenever the stage optimizer is rebuilt.
- `losses_slm.py` includes a Pearson correlation feature term. This gives the SLM student a scale/offset-insensitive structural target, reducing the hard lower bound caused by absolute-value MSE terms when student and teacher feature scales differ.
- The phase diversity loss is now a lightweight regularizer instead of a dominant objective: `LOSS_PHASE_DIVERSITY_WEIGHT = 0.15`, with relaxed std/span/circular-std targets. It still rejects nearly flat phase maps, but leaves more room for feature matching.
- `losses_slm.py` penalizes low circular phase spread and excessive concentration near the `0 / 2*pi` wrap boundary.
- `student_adapt_max` only tracks normalized feature adaptation quality; it does not write `optical_student_best.pth`, because that stage is not selected by detector mAP.
- `detector_best.pth` carries both `detector_state_dict` and the paired `student_state_dict` from the same best-mAP epoch. This is the checkpoint to use for optical-student + detector inference, especially when mAP peaks around the detector-only window and later joint training does not improve.
- `optical_student_best.pth` is refreshed from the same student snapshot whenever `detector_best.pth` is updated, so phase extraction and detector evaluation stay aligned. Its metadata still records whether the SLM phase diversity check passed.
- `optical_layers.py` supports vortex phase initialization for both SLM layers. This gives the phase optimizer a non-flat optical pattern instead of relying only on random phase.
- `ConfigSLM.SLM_INIT_MODE` controls initialization: `random`, `vortex`, `checkpoint`, or `vortex_checkpoint`. `checkpoint` and `vortex_checkpoint` load `ConfigSLM.SLM_INIT_CHECKPOINT` into the student but still start training from the first student-only stage.
- Teacher `abs` is intentionally unchanged for now. If the teacher is retrained later, removing `abs` may make the target feature range more compatible with physical optical intensity, but that is a separate experiment.

Useful runtime overrides:

```powershell
$env:OPTICAL_SLM_INIT_MODE="vortex"
$env:OPTICAL_SLM_STUDENT_NORM_SCHEDULE="always"
$env:OPTICAL_SLM_STUDENT_NORM_MODE="max"
python .\optical_slm_yolov8_head.py
```

For the default delayed-normalization run:

```powershell
$env:OPTICAL_SLM_STUDENT_NORM_SCHEDULE="late"
$env:OPTICAL_SLM_STUDENT_NORM_EARLY_MODE="none"
$env:OPTICAL_SLM_STUDENT_ADAPT_MAX_EPOCHS="15"
python .\optical_slm_yolov8_head.py
```

Checkpoint selection notes:

- `detector_best.pth` is selected by validation `mAP50`, carries the paired `student_state_dict`, and is marked as the recommended inference checkpoint.
- `optical_student_best.pth` is no longer meant to be a final-train-loss or feature-loss snapshot. It mirrors the student paired with the best detector mAP, so standalone SLM extraction and detector evaluation stay aligned.

To continue from an earlier student checkpoint while still retraining from the first student stage:

```powershell
$env:OPTICAL_SLM_INIT_MODE="checkpoint"
$env:OPTICAL_SLM_INIT_CHECKPOINT="output\OpticalSLM_YOLOv8Head_student\optical_student_best.pth"
python .\optical_slm_yolov8_head.py
```

The SLM checkpoint payload intentionally avoids a top-level `"phase"` key, so phase extraction scripts can search tensor parameter names such as `phase_raw` without accidentally matching a string metadata field. It also stores wrapped `*_wrapped_slm_0_2pi` tensors for SLM loading.

`Optical_yolo_detect/Optical_SLM_yolov8_head_model.py` loads those two SLM outputs for visualization/evaluation, and its local README explains the expected input/output files.
