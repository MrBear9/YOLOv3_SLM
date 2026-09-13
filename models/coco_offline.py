"""Official COCO bbox evaluation of cached YOLO detections and targets."""

import json
from importlib.metadata import version
from pathlib import Path

from PIL import Image


def require_coco():
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError(
            'Official COCO evaluation requires: python -m pip install pycocotools '
            '(or use --skip-coco for legacy evaluation only).'
        ) from exc
    return COCO, COCOeval


def source_box(box, source_size, resolution):
    """Invert the exact scalar scale/padding used by letterbox_image_targets."""
    sw, sh = source_size
    h, w = resolution
    scale = min(w / sw, h / sh)
    left = (w - max(1, round(sw * scale))) // 2
    top = (h - max(1, round(sh * scale))) // 2
    cx, cy, bw, bh = map(float, box)
    # Do not clip: preserve the detector's actual prediction for evaluation.
    return [(cx - bw / 2 - left) / scale,
            (cy - bh / 2 - top) / scale, bw / scale, bh / scale]


def evaluate_coco(detections, targets, source_paths, config, output_dir):
    COCO, COCOeval = require_coco()
    if not len(detections) == len(targets) == len(source_paths):
        raise ValueError('COCO export image counts do not match.')
    if not source_paths:
        raise ValueError('Cannot evaluate an empty image list.')
    import numpy as np

    dataset = {
        'info': {'description': 'YOLO labels converted to original-image COCO bbox coordinates'},
        'images': [], 'annotations': [],
        'categories': [{'id': i, 'name': config.CLASS_NAMES[i]} for i in range(config.NUM_CLASSES)],
    }
    results = []
    for image_id, (dets, labels, path) in enumerate(zip(detections, targets, source_paths), 1):
        with Image.open(path) as image:
            size = image.size
        dataset['images'].append({'id': image_id, 'file_name': str(Path(path).resolve()),
                                  'width': size[0], 'height': size[1]})
        for row in labels:
            row = np.asarray(row, dtype=float)
            if row.shape != (5,) or not np.isfinite(row).all():
                raise ValueError('Invalid YOLO target in COCO export.')
            cid = int(row[0])
            if row[0] != cid or not 0 <= cid < config.NUM_CLASSES:
                raise ValueError('Invalid target category.')
            if row[3] <= 0 or row[4] <= 0:
                continue
            h, w = config.RESOLUTION
            box = source_box(row[1:] * [w, h, w, h], size, config.RESOLUTION)
            dataset['annotations'].append({'id': len(dataset['annotations']) + 1,
                'image_id': image_id, 'category_id': cid, 'bbox': box,
                'area': box[2] * box[3], 'iscrowd': 0})
        for row in dets:
            row = np.asarray(row, dtype=float)
            if row.shape != (6,) or not np.isfinite(row).all():
                raise ValueError('Invalid prediction in COCO export.')
            cid = int(row[5])
            if row[5] != cid or not 0 <= cid < config.NUM_CLASSES or min(row[2:4]) < 0:
                raise ValueError('Invalid prediction category or box dimensions.')
            results.append({'image_id': image_id, 'category_id': cid,
                            'bbox': source_box(row[:4], size, config.RESOLUTION),
                            'score': float(row[4])})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in [('coco_annotations.json', dataset), ('coco_predictions.json', results)]:
        (output_dir / name).write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    gt = COCO()
    gt.dataset = dataset
    gt.createIndex()
    if results:
        predictions = gt.loadRes(results)
    else:
        # COCO.loadRes indexes the first result; build an empty result API explicitly.
        predictions = COCO()
        predictions.dataset = {'images': dataset['images'], 'categories': dataset['categories'],
                               'annotations': []}
        predictions.createIndex()
    evaluator = COCOeval(gt, predictions, 'bbox')
    evaluator.params.imgIds = [item['id'] for item in dataset['images']]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ('AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
             'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large')
    metrics = {name: float(value) if value >= 0 else None
               for name, value in zip(names, evaluator.stats)}
    precision = evaluator.eval['precision']
    per_class = {}
    for index, cid in enumerate(evaluator.params.catIds):
        values = precision[:, :, index, 0, -1]
        valid = values[values >= 0]
        per_class[int(cid)] = {'class_name': config.CLASS_NAMES[cid],
                              'AP': float(valid.mean()) if valid.size else None}
    metrics['per_class'] = per_class
    metrics['protocol'] = {
        'evaluator': 'pycocotools.COCOeval', 'version': version('pycocotools'),
        'iou_type': 'bbox', 'iou_thresholds': evaluator.params.iouThrs.tolist(),
        'max_dets': list(evaluator.params.maxDets), 'recall_points': len(evaluator.params.recThrs),
        'coordinates': 'original image pixels, inverse dataset letterbox; no clipping',
        'ground_truth': 'YOLO boxes; all iscrowd=0; area=box width*height',
        'decode_confidence': float(config.METRIC_CONF_THRESH),
        'decode_nms': float(config.METRIC_NMS_THRESH),
        'decode_max_det': int(config.METRIC_MAX_DET),
        'undefined_metric': 'null (official summary uses -1)',
    }
    (output_dir / 'coco_evaluation_report.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return metrics
