"""Publication assets for the adaptive entry only; never select training updates.

Rank-zero, raw-module calls only: no distributed collectives or optimizer steps.
The fixed validation cohort is deliberately selected for illustration, not an
unbiased performance estimate. Before/after pairs freeze the digital detector.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from models.class_display import class_name_for_id
from models.detection_metrics_offline import _iou
from models.yolov8.detection_protocol import decode_detections
from models.yolov8.feature_adapter import prepare_slm_detector_feature


def sample_metrics(detections, targets, resolution):
    """Same-class, confidence-ordered unique matches at IoU >= .5.

    Input detections already filtered at confidence .25. Misses contribute zero
    to mean_gt_iou; F1 penalizes surplus predictions as well as missing targets.
    """
    h, w = resolution
    gt = [(int(t[0]), np.asarray(t[1:5]) * [w, h, w, h]) for t in targets
          if t[3] > 0 and t[4] > 0]
    used, scores = set(), []
    for d in sorted(detections, key=lambda x: float(x[4]), reverse=True):
        available = [i for i, (cls, _) in enumerate(gt) if cls == int(d[5]) and i not in used]
        best = max(available, key=lambda i: _iou(d, gt[i][1]), default=None)
        if best is not None and _iou(d, gt[best][1]) >= .5:
            used.add(best)
            scores.append(float(_iou(d, gt[best][1])))
    tp, n, m = len(scores), len(gt), len(detections)
    return dict(gt_count=n, prediction_count=m, matched=tp,
                precision=tp / max(m, 1), recall=tp / max(n, 1),
                f1=2 * tp / max(n + m, 1), mean_gt_iou=sum(scores) / max(n, 1))


@contextmanager
def observation(learner, detector):
    """Preserve random streams and mixed module training flags during export."""
    modules = list(learner.student.modules()) + list(detector.modules())
    modes = [m.training for m in modules]
    py, np_state = random.getstate(), np.random.get_state()
    device = learner.parameters[0].device
    devices = [device.index] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            learner.student.eval()
            detector.eval()
            yield
    finally:
        for module, mode in zip(modules, modes):
            module.training = mode
        random.setstate(py)
        np.random.set_state(np_state)


def array(tensor):
    return tensor.detach().squeeze().float().cpu().numpy().copy()


def circular_difference(after, before):
    difference = after - before
    return np.arctan2(np.sin(difference), np.cos(difference))


class AdaptiveSnapshots:
    def __init__(self, output, config, dataset, learner, detector, criterion,
                 *, samples=2, pool=32, seed=42):
        self.root = Path(output) / 'paper_snapshots'
        self.config, self.dataset = config, dataset
        self.learner, self.detector, self.criterion = learner, detector, criterion
        self.samples, self.pool, self.seed = samples, pool, seed
        self.device = learner.parameters[0].device
        self.selected = None

    def predict(self, feature):
        output = self.detector(prepare_slm_detector_feature(self.config, feature))
        return np.asarray(decode_detections(self.config, output, conf_thresh=.25,
            nms_thresh=getattr(self.config, 'VIS_NMS_THRESH', .5),
            max_det=getattr(self.config, 'METRIC_MAX_DET', 300))[0], dtype=np.float32).reshape(-1, 6)

    def select(self):
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / 'selection.json'
        if manifest.exists():
            saved = json.loads(manifest.read_text(encoding='utf-8'))
            self.selected = saved['selected']
            for entry in self.selected:
                actual = Path(self.dataset.entries[entry['index']]['image_path'])
                if hashlib.sha256(actual.read_bytes()).hexdigest() != entry['sha256']:
                    raise ValueError('Snapshot cohort changed; use the original validation dataset')
            return
        indices = np.random.default_rng(self.seed).choice(len(self.dataset),
            min(self.pool, len(self.dataset)), replace=False)
        candidates = []
        with torch.no_grad():
            for index in indices:
                sample = self.dataset[int(index)]
                gray = sample['gray_tensor'][None].to(self.device)
                prediction = self.predict(self.learner.student(gray))
                metrics = sample_metrics(prediction, sample['targets'].numpy(), self.config.RESOLUTION)
                candidates.append(dict(index=int(index), image_path=str(sample['image_path']), metrics=metrics))
        candidates.sort(key=lambda x: (x['metrics']['gt_count'] >= 2,
            x['metrics']['f1'], x['metrics']['mean_gt_iou']), reverse=True)
        self.selected = candidates[:self.samples]
        for entry in self.selected:
            entry['sha256'] = hashlib.sha256(Path(entry['image_path']).read_bytes()).hexdigest()
        manifest.write_text(json.dumps(dict(seed=self.seed, selected=self.selected,
            candidates=candidates, rule='Fixed seeded validation pool; multi-target first, then F1, then mean GT IoU. Selected once before the first observed phase cycle; reused on resume.',
            confidence=.25, matching_iou=.5,
            caveat='Illustrative selected examples, not dataset mAP or evidence of generalization.'),
            indent=2, ensure_ascii=False), encoding='utf-8')

    def capture(self, local_signals=False):
        with observation(self.learner, self.detector):
            if self.selected is None:
                self.select()
            phases = [array(layer.simulation_phase()) for layer in self.learner.layers]
            records = []
            for entry in self.selected:
                sample = self.dataset[entry['index']]
                gray = sample['gray_tensor'][None].to(self.device)
                incoming, fields = self.learner.forward_fields(gray)
                with torch.no_grad():
                    feature = self.learner.feature(fields[-1])
                    prediction = self.predict(feature)
                with torch.enable_grad():
                    terminal = fields[-1].detach().requires_grad_(local_signals)
                    output = self.detector(prepare_slm_detector_feature(self.config, self.learner.feature(terminal)))
                    loss = self.criterion(output, [sample['targets']])[0]
                    error = torch.autograd.grad(loss, terminal)[0].detach() if local_signals else None
                data = dict(input=array(gray), targets=sample['targets'].numpy(), predictions=prediction,
                            camera_intensity=array(self.learner.student.geometry.intensity(fields[-1])
                                if self.learner.student.geometry is not None else fields[-1].abs().square()))
                for i, field in enumerate(fields):
                    data[f'intensity_{i+1}'] = array(field.abs().square())
                if local_signals:
                    for i in reversed(range(len(fields))):
                        # Diagnostic only, no global training gradient/step or collectives.
                        data[f'error_magnitude_{i+1}'] = array(error.abs())
                        data[f'local_gradient_{i+1}'] = array(self.learner.local_gradient(i, incoming[i], error))
                        if i:
                            error = self.learner.beta * self.learner.feedback(i, error)
                records.append(dict(data=data, loss=float(loss.detach()),
                    metrics=sample_metrics(prediction, data['targets'], self.config.RESOLUTION)))
            return dict(phases=phases, records=records, updates=self.learner.updates,
                        calibration_count=self.learner.calibration_count)

    def save(self, cycle, before):
        with observation(self.learner, self.detector):
            self._save(cycle, before)

    def _save(self, cycle, before):
        after = self.capture()
        folder = self.root / f'cycle_{cycle:04d}'
        folder.mkdir(parents=True, exist_ok=True)
        phase_data = {}
        for i, (a, b) in enumerate(zip(before['phases'], after['phases']), 1):
            phase_data.update({f'before_{i}': a, f'after_{i}': b,
                               f'circular_delta_{i}': circular_difference(b, a)})
        np.savez_compressed(folder / 'phases.npz', **phase_data)
        history_path = self.root / 'history.json'
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        # Re-running a cycle from forward_last replaces, rather than duplicates, its assets.
        history = [r for r in history if r['cycle'] != cycle]
        for j, (left, right) in enumerate(zip(before['records'], after['records'])):
            assets = folder / f'sample_{j+1:02d}'
            assets.mkdir(exist_ok=True)
            raw = {f'before_{k}': v for k, v in left['data'].items()}
            raw.update({f'after_{k}': v for k, v in right['data'].items()})
            np.savez_compressed(assets / 'fields.npz', **raw)
            row = dict(cycle=cycle, sample=j+1, source=self.selected[j],
                before=left['metrics'], after=right['metrics'], loss_before=left['loss'], loss_after=right['loss'],
                updates_before=before['updates'], updates_after=after['updates'],
                calibration_before=before['calibration_count'], calibration_after=after['calibration_count'],
                phase_delta_rms=[float(np.sqrt(np.mean(v*v))) for k, v in phase_data.items() if k.startswith('circular_delta')],
                assets=str(assets.relative_to(self.root)), detector='Same frozen detector before and after phase block; before detector fine-tuning',
                scope='Simulated forward intensities and numerical local feedback; no hardware field measurement or full optical-chain training backpropagation')
            self.render(assets, left, right, phase_data, row)
            (assets / 'metrics.json').write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding='utf-8')
            history.append(row)
        history.sort(key=lambda r: (r['cycle'], r['sample']))
        history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding='utf-8')
        best = {}
        for r in history:
            key = str(r['sample'])
            score = lambda x: (x['after']['f1'], x['after']['mean_gt_iou'], -x['loss_after'])
            if key not in best or score(r) > score(best[key]):
                best[key] = r
        (self.root / 'best_examples.json').write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding='utf-8')

    def render(self, folder, left, right, phases, metadata):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        n = len(self.learner.layers)
        phase_offset = float(getattr(self.config, 'SLM_EXPORT_PHASE_OFFSET_RAD', np.pi))
        scales = {}
        style = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                 'font.size': 11, 'pdf.fonttype': 42}

        def show(ax, values, name, *, cmap='viridis', lo=0., hi=None):
            hi = max(float(np.max(values)) if hi is None else float(hi), lo + 1e-20)
            im = ax.imshow(values, cmap=cmap, norm=Normalize(lo, hi), interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            scales[name] = dict(min=lo, max=hi, cmap=cmap)
            # Standalone panel at original resolution, without labels or resizing.
            plt.imsave(folder / f'{name}.png', values, cmap=cmap, vmin=lo, vmax=hi)
            return im

        def boxes(ax, data):
            ax.imshow(data['input'], cmap='gray', vmin=0, vmax=1)
            h, w = data['input'].shape
            for t in data['targets']:
                _, x, y, bw, bh = t
                ax.add_patch(plt.Rectangle(((x-bw/2)*w, (y-bh/2)*h), bw*w, bh*h,
                    fill=False, edgecolor='#2b8c65', lw=1.2, linestyle='--'))
            for x, y, bw, bh, confidence, cls in data['predictions']:
                ax.add_patch(plt.Rectangle((x-bw/2, y-bh/2), bw, bh,
                    fill=False, edgecolor='#c83737', lw=1.1))
                label = class_name_for_id(getattr(self.config, 'CLASS_NAMES', None), int(cls))
                ax.text(max(0, x-bw/2), max(0, y-bh/2), f'{label} {confidence:.2f}',
                    fontsize=7, color='#a02020', bbox=dict(facecolor='white', alpha=.75, edgecolor='none', pad=.4))
            ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_xticks([]); ax.set_yticks([])

        with plt.rc_context(style):
            fig, axes = plt.subplots(3, n+3, figsize=(3.25*(n+3), 9), squeeze=False, layout='constrained')
            fig.suptitle(f'Adaptive phase update — cycle {metadata["cycle"]}\nForward optical path → camera → fixed detector', fontsize=16)
            for row, record in enumerate((left, right)):
                show(axes[row, 0], record['data']['input'], f'input_{row}', cmap='gray', hi=1)
                axes[row, 0].set_ylabel('Before' if row == 0 else 'After', fontsize=14)
                axes[row, 0].set_title('Incident intensity')
                for i in range(1, n+1):
                    a, b = left['data'][f'intensity_{i}'], right['data'][f'intensity_{i}']
                    scale = max(float(a.max()), float(b.max()), 1e-20)
                    # Common relative-log display makes before/after directly comparable.
                    values = np.log1p(record['data'][f'intensity_{i}'] / scale * 1000)
                    im = show(axes[row, i], values, f'propagation_{i}_{row}', hi=np.log1p(1000))
                    scales[f'propagation_{i}_{row}'].update(transform='log1p(1000*I/shared_max)', shared_max=scale)
                    axes[row, i].set_title(f'SLM {i} → ASM')
                    if row == 1:
                        fig.colorbar(im, ax=axes[:2, i], shrink=.6, label='Relative log intensity')
                camera_scale = max(float(left['data']['camera_intensity'].max()),
                                   float(right['data']['camera_intensity'].max()), 1e-20)
                camera = np.log1p(record['data']['camera_intensity'] / camera_scale * 1000)
                im = show(axes[row, n+1], camera, f'camera_{row}', hi=np.log1p(1000))
                scales[f'camera_{row}'].update(transform='log1p(1000*I/shared_max)', shared_max=camera_scale)
                axes[row, n+1].set_title('Camera-plane readout')
                if row == 1:
                    fig.colorbar(im, ax=axes[:2, n+1], shrink=.6, label='Relative log intensity')
                boxes(axes[row, -1], record['data'])
                h, w = record['data']['input'].shape
                panel = plt.figure(figsize=(w/100, h/100))
                boxes(panel.add_axes([0, 0, 1, 1]), record['data'])
                panel.savefig(folder / ('detection_before.png' if row == 0 else 'detection_after.png'), dpi=100)
                plt.close(panel)
                m = record['metrics']
                axes[row, -1].set_title(f'Detection: F1 {m["f1"]:.2f}, GT-IoU {m["mean_gt_iou"]:.2f}')
            axes[2, 0].axis('off')
            axes[2, 0].text(.02, .85, 'Same image / same detector\nGreen dashed: GT\nRed: predictions\n\nGT-IoU includes misses as 0\nSelected illustration; not mAP', va='top')
            for i in range(1, n+2):
                key = f'intensity_{i}' if i <= n else 'camera_intensity'
                delta = right['data'][key] - left['data'][key]
                limit = max(float(np.max(np.abs(delta))), 1e-20)
                im = show(axes[2, i], delta, f'intensity_delta_{i}', cmap='RdBu_r', lo=-limit, hi=limit)
                axes[2, i].set_title('Intensity change\n(after − before)')
                fig.colorbar(im, ax=axes[2, i], shrink=.65, format='%.1e')
            axes[2, -1].axis('off')
            axes[2, -1].text(.02, .85, f'Detection loss\n{left["loss"]:.4f} → {right["loss"]:.4f}\n\nPhase updates\n{metadata["updates_before"]} → {metadata["updates_after"]}\n\nDifference color scales\nare shown explicitly.', va='top')
            fig.savefig(folder / 'forward_comparison.png', dpi=180)
            fig.savefig(folder / 'forward_comparison.pdf')
            plt.close(fig)

            fig, axes = plt.subplots(4, n, figsize=(4*n, 12), squeeze=False, layout='constrained')
            fig.suptitle('Numerical local feedback → full-pixel phase update\nSignals before update; phase after update', fontsize=15)
            for i in range(1, n+1):
                displays = [
                    (left['data'][f'error_magnitude_{i}'], f'error_{i}', 'Local residual magnitude |e|', 'viridis', False),
                    (left['data'][f'local_gradient_{i}'], f'gradient_{i}', 'Local phase gradient\n(single-image diagnostic)', 'RdBu_r', True),
                    (phases[f'circular_delta_{i}'], f'phase_delta_{i}', 'Actual phase change (rad)\n(across training block)', 'RdBu_r', True),
                    (np.mod(phases[f'after_{i}'] + phase_offset, 2*np.pi), f'phase_after_{i}', 'Updated phase (rad)\n(export display convention)', 'gray', False)]
                for row, (values, name, title, cmap, signed) in enumerate(displays):
                    limit = 2*np.pi if row == 3 else max(float(np.max(np.abs(values))), 1e-20)
                    im = show(axes[row, i-1], values, name, cmap=cmap, lo=-limit if signed else 0, hi=limit)
                    axes[row, i-1].set_title(f'SLM {i}: {title}', fontsize=10)
                    fig.colorbar(im, ax=axes[row, i-1], shrink=.7, format='%.2g')
            fig.savefig(folder / 'local_learning.png', dpi=180)
            fig.savefig(folder / 'local_learning.pdf')
            plt.close(fig)
        metadata['display_scales'] = scales
        metadata['phase_display_offset_rad'] = phase_offset
        metadata['local_signal_note'] = 'Validation-image local signals are illustrative only; actual phase change is learned from training batches. Maps use native grids; adjacent propagation planes can have different pixel pitches/FOV.'
