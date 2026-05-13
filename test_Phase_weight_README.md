# test_Phase_weight.py

`test_Phase_weight.py` extracts SLM phase layers from an optical student checkpoint and writes files that are easier to inspect before loading a spatial light modulator.

## Default Usage

```powershell
python .\test_Phase_weight.py
```

Default input:

```text
output/OpticalSLM_YOLOv8Head_student/optical_student_best.pth
```

Default output:

```text
output/OpticalSLM_YOLOv8Head_student/optical_phase_layers
```

Custom paths:

```powershell
python .\test_Phase_weight.py `
  --pth-file output\OpticalSLM_YOLOv8Head_student\optical_student_best.pth `
  --output-dir temple\optical_phase_layers
```

## Output Files

For each phase layer, the script writes:

- `*_raw_phase_raw.npy`: raw trainable parameter from the checkpoint.
- `*_wrapped_slm_0_2pi.npy`: SLM-loadable phase in `[0, 2*pi)`.
- `*_centered_minus_pi_pi.npy`: diagnostic phase in `[-pi, pi)`.
- `*_wrapped_slm_0_2pi.png`: 8-bit preview of the wrapped phase.
- `*_histogram.png`: histograms for raw, wrapped, and centered phase.
- `*_info.txt`: per-layer statistics.
- `optical_phase_layers_summary.txt`: summary for all extracted phase layers.

Use `*_wrapped_slm_0_2pi.npy` for SLM loading.

## How To Read The Statistics

`phase_raw` is unconstrained, so negative values are normal. During optical forward propagation, the model uses:

```python
phase_raw mod (2 * pi)
```

This wrapped phase is physically equivalent and is what the SLM should receive.

Do not judge a wrapped phase only by ordinary linear standard deviation. If most values are near `0` and `2*pi`, the linear standard deviation can look large even though the phase is almost flat. Prefer:

- `circular_std`: should not be near zero for a useful learned phase pattern.
- `near_0_or_2pi_ratio`: if this is close to `1.0`, most pixels are near the phase wrap boundary.
- `centered_phase std`: useful for judging whether the actual modulation is nearly zero.

If `circular_std < 0.1` or `near_0_or_2pi_ratio > 0.95`, the extracted phase is likely close to flat modulation and may not be useful for SLM deployment.

## Why Phase Can Collapse During Training

SLM phase is periodic, but the trainable tensor `phase_raw` is an ordinary unconstrained parameter. If normal optimizer weight decay is applied to this tensor, it continuously pulls the phase toward zero. That can produce a visually flat SLM phase even when detector training still appears to make progress.

The current SLM training code avoids this by:

- setting `PHASE_WEIGHT_DECAY = 0.0` for `phase_raw` parameters,
- supporting a short `student_adapt_max` stage when training starts with unnormalized optical intensity and later switches to `max` normalization,
- adding a Pearson correlation feature loss to focus part of the optimization on spatial structure instead of only absolute scale,
- adding circular phase diversity loss,
- lowering the phase diversity weight/targets so diversity prevents collapse without dominating feature learning,
- penalizing too many pixels near the `0 / 2*pi` wrap boundary,
- refusing to overwrite `optical_student_best.pth` unless the SLM phase quality check passes.
- supporting vortex phase initialization through `ConfigSLM.SLM_INIT_MODE = "vortex"`.
- supporting checkpoint initialization through `ConfigSLM.SLM_INIT_MODE = "checkpoint"` while still training from the first stage.

The teacher feature extractor still keeps its original `abs` operation. This was left unchanged intentionally, so any improvement after these changes comes from the SLM student side and the loss design rather than a retrained teacher target.

After training, rerun this extractor and check `optical_phase_layers_summary.txt`. A useful phase should not have near-zero `circular_std`, and `near_0_or_2pi_ratio` should not be close to `1.0`.
