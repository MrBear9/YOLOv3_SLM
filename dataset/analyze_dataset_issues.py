import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = ROOT / "data" / "military" / "data.yaml"
DEFAULT_OUTPUT = ROOT / "output" / "anchor_clustering" / "dataset_issue_scan"


def windows_safe_path(path: Path) -> str:
    text = str(path)
    if os.name != "nt":
        return text
    if text.startswith("\\\\?\\"):
        return text
    try:
        resolved = path.resolve(strict=False)
    except Exception:
        resolved = Path(text)
    resolved_text = str(resolved)
    if resolved_text.startswith("\\\\?\\"):
        return resolved_text
    if resolved_text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved_text.lstrip("\\")
    return "\\\\?\\" + resolved_text


def write_text_compat(path: Path, data: str, encoding: str = "utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(windows_safe_path(path), "w", encoding=encoding, newline="") as f:
        f.write(data)


def xywh_to_xyxy(box):
    x, y, w, h = box
    return (
        x - w / 2.0,
        y - h / 2.0,
        x + w / 2.0,
        y + h / 2.0,
    )


def bbox_iou_xywh(box_a, box_b):
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(box_a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


@dataclass
class BoxRecord:
    split: str
    image_path: str
    label_path: str
    line_index: int
    class_id: int | None
    class_name: str
    x_center: float | None
    y_center: float | None
    width: float | None
    height: float | None
    area: float | None
    issue_type: str
    note: str = ""


@dataclass
class ImageRecord:
    split: str
    image_path: Path
    label_path: Path
    boxes: list[dict] = field(default_factory=list)
    class_counter: Counter = field(default_factory=Counter)
    image_issues: list[dict] = field(default_factory=list)
    line_entries: list[dict] = field(default_factory=list)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze YOLO dataset label quality without modifying labels.")
    parser.add_argument("--yaml", type=Path, default=DEFAULT_YAML, help="Path to dataset yaml file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory to write analysis outputs.")
    parser.add_argument("--mode", choices=("analyze_only", "review_only", "safe_remove", "resample_only", "all"), default="all", help="Which non-destructive outputs to generate.")
    parser.add_argument("--preview-count", type=int, default=40, help="Number of preview images to render.")
    parser.add_argument("--huge-side-thresh", type=float, default=0.9, help="Threshold for huge width/height.")
    parser.add_argument("--large-area-thresh", type=float, default=0.5, help="Threshold for large-area boxes.")
    parser.add_argument("--edge-margin", type=float, default=0.01, help="Margin used for edge-touch detection.")
    parser.add_argument("--tiny-area-thresh", type=float, default=0.0005, help="Threshold for tiny boxes.")
    parser.add_argument("--dense-mixed-small-area", type=float, default=0.02, help="Small-box threshold for mixed-scale image detection.")
    parser.add_argument("--duplicate-iou", type=float, default=0.85, help="IoU threshold for likely duplicate labels.")
    parser.add_argument("--many-objects-thresh", type=int, default=8, help="Threshold for crowded images.")
    parser.add_argument("--safe-remove-area-thresh", type=float, default=0.65, help="Conservative area threshold used for safe-remove proposals.")
    parser.add_argument("--safe-remove-side-thresh", type=float, default=0.95, help="Conservative side threshold used for safe-remove proposals.")
    parser.add_argument("--resample-max-repeat", type=int, default=4, help="Upper bound for suggested repeat factor in resample plan.")
    return parser.parse_args()


def load_dataset_config(yaml_path: Path):
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        ordered = [names[idx] for idx in sorted(names)]
    else:
        ordered = list(names)
    raw_dataset_root = Path(cfg.get("path", "."))
    dataset_root_candidates = []
    if raw_dataset_root.is_absolute():
        dataset_root_candidates.append(raw_dataset_root)
    else:
        dataset_root_candidates.append((yaml_path.parent / raw_dataset_root).resolve())
        dataset_root_candidates.append((ROOT / raw_dataset_root).resolve())
        dataset_root_candidates.append(yaml_path.parent.resolve())
    dataset_root = next((candidate for candidate in dataset_root_candidates if candidate.exists()), dataset_root_candidates[0])
    splits = {}
    for split in ("train", "val", "test"):
        images_rel = cfg.get(split)
        if images_rel is None:
            continue
        split_path = Path(images_rel)
        if split_path.is_absolute():
            images_dir = split_path
        else:
            images_dir = (dataset_root / split_path).resolve()
        labels_dir = images_dir.parent / "labels"
        splits[split] = {
            "images_dir": images_dir,
            "labels_dir": labels_dir,
        }
    return dataset_root, ordered, splits


def detect_box_issues(box, num_classes, class_names, args):
    issues = []
    class_id = box["class_id"]
    x = box["x_center"]
    y = box["y_center"]
    w = box["width"]
    h = box["height"]

    if class_id < 0 or class_id >= num_classes:
        issues.append(("invalid_box", f"class_id {class_id} out of range"))
    if w <= 0 or h <= 0:
        issues.append(("invalid_box", "non-positive width or height"))
    if x < 0 or x > 1 or y < 0 or y > 1 or w > 1 or h > 1:
        issues.append(("invalid_box", "coordinates out of normalized range"))

    if issues:
        return issues

    area = w * h
    if w > args.huge_side_thresh or h > args.huge_side_thresh:
        issues.append(("huge_box", "width or height exceeds huge-side threshold"))
    if area > args.large_area_thresh:
        issues.append(("large_area_box", "normalized box area exceeds large-area threshold"))
    if (x - w / 2.0) <= args.edge_margin or (y - h / 2.0) <= args.edge_margin or (x + w / 2.0) >= (1.0 - args.edge_margin) or (y + h / 2.0) >= (1.0 - args.edge_margin):
        issues.append(("edge_touch_box", "box is close to image boundary"))
    if area < args.tiny_area_thresh:
        issues.append(("tiny_box", "normalized box area is extremely small"))
    return issues


def render_previews(records, preview_dir: Path, top_n: int):
    preview_dir.mkdir(parents=True, exist_ok=True)
    palette = {
        "huge_box": "red",
        "large_area_box": "orange",
        "edge_touch_box": "yellow",
        "tiny_box": "cyan",
        "dense_mixed_image": "magenta",
        "duplicate_overlap_image": "blue",
        "many_objects_image": "white",
        "invalid_box": "red",
        "missing_label_file": "red",
        "missing_image_file": "red",
        "empty_label_file": "orange",
    }
    drawn = 0
    for record in records[:top_n]:
        image_path = record["image_path"]
        if not image_path.exists():
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue
        draw = ImageDraw.Draw(image)
        width_px, height_px = image.size

        for box in record["boxes"]:
            if box["width"] is None or box["height"] is None:
                continue
            x1, y1, x2, y2 = xywh_to_xyxy((box["x_center"], box["y_center"], box["width"], box["height"]))
            rect = (
                max(0, int(round(x1 * width_px))),
                max(0, int(round(y1 * height_px))),
                min(width_px - 1, int(round(x2 * width_px))),
                min(height_px - 1, int(round(y2 * height_px))),
            )
            issue_types = [issue["issue_type"] for issue in box["issues"]]
            color = palette.get(issue_types[0], "lime") if issue_types else "lime"
            draw.rectangle(rect, outline=color, width=3)
            label = f"{box['class_name']} | {','.join(issue_types[:2])}" if issue_types else box["class_name"]
            draw.text((rect[0] + 4, max(0, rect[1] - 14)), label, fill=color)

        for idx, issue in enumerate(record["image_issues"]):
            text = f"IMAGE: {issue['issue_type']}"
            draw.text((8, 8 + idx * 16), text, fill=palette.get(issue["issue_type"], "white"))

        output_path = preview_dir / f"{drawn:03d}_{record['split']}_{image_path.stem}.jpg"
        image.save(output_path, quality=90)
        drawn += 1


def should_generate(mode: str, target: str):
    if mode == "all":
        return True
    return mode == target


def should_safe_remove_entry(entry: dict, args):
    if entry.get("invalid", False):
        return True, "invalid_box"

    box = entry.get("box")
    if not box:
        return False, ""

    issue_types = {issue["issue_type"] for issue in box.get("issues", [])}
    area = box["area"]
    width = box["width"]
    height = box["height"]
    is_large_background_like = (
        "edge_touch_box" in issue_types and
        (
            area >= args.safe_remove_area_thresh or
            width >= args.safe_remove_side_thresh or
            height >= args.safe_remove_side_thresh
        ) and
        ("large_area_box" in issue_types or "huge_box" in issue_types)
    )
    if is_large_background_like:
        return True, "large_edge_background_box"
    return False, ""


def write_review_outputs(output_dir: Path, issue_rows: list[BoxRecord], high_risk_rows: list[dict], preview_candidates: list[dict], args):
    review_dir = output_dir / "review_only"
    review_dir.mkdir(parents=True, exist_ok=True)

    review_issue_types = {
        "huge_box",
        "large_area_box",
        "edge_touch_box",
        "tiny_box",
        "dense_mixed_image",
        "duplicate_overlap_image",
        "many_objects_image",
    }

    review_csv = review_dir / "review_candidates.csv"
    with review_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "image_path", "label_path", "line_index", "class_id", "class_name",
            "x_center", "y_center", "width", "height", "area", "issue_type", "note"
        ])
        for row in issue_rows:
            if row.issue_type in review_issue_types:
                writer.writerow([
                    row.split, row.image_path, row.label_path, row.line_index, row.class_id, row.class_name,
                    row.x_center, row.y_center, row.width, row.height, row.area, row.issue_type, row.note
                ])

    review_high_risk_csv = review_dir / "review_high_risk_images.csv"
    with review_high_risk_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image_path", "label_path", "risk_score", "issue_types", "box_count"])
        for row in high_risk_rows:
            if any(issue in row["issue_types"] for issue in review_issue_types):
                writer.writerow([row["split"], row["image_path"], row["label_path"], row["risk_score"], row["issue_types"], row["box_count"]])

    render_previews(preview_candidates, review_dir / "previews", min(args.preview_count, 30))


def write_safe_remove_outputs(output_dir: Path, all_image_records: list[ImageRecord], args):
    safe_dir = output_dir / "safe_remove"
    labels_dir = safe_dir / "proposed_labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = safe_dir / "safe_remove_manifest.csv"
    summary_json = safe_dir / "safe_remove_summary.json"
    summary_txt = safe_dir / "safe_remove_summary.txt"

    removed_rows = []
    kept_line_count = 0
    removed_line_count = 0
    touched_files = 0
    touched_files_by_split = Counter()

    for record in all_image_records:
        split_label_dir = labels_dir / record.split / "labels"
        split_label_dir.mkdir(parents=True, exist_ok=True)
        target_label = split_label_dir / record.label_path.name
        kept_lines = []
        file_removed = 0

        for entry in record.line_entries:
            should_remove, reason = should_safe_remove_entry(entry, args)
            if should_remove:
                removed_line_count += 1
                file_removed += 1
                removed_rows.append({
                    "split": record.split,
                    "image_path": str(record.image_path),
                    "label_path": str(record.label_path),
                    "line_index": entry["line_index"],
                    "reason": reason,
                    "raw_line": entry["raw_line"],
                })
            else:
                kept_line_count += 1
                kept_lines.append(entry["raw_line"])

        if file_removed > 0:
            touched_files += 1
            touched_files_by_split[record.split] += 1

        write_text_compat(target_label, "\n".join(kept_lines) + ("\n" if kept_lines else ""))

    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image_path", "label_path", "line_index", "reason", "raw_line"])
        for row in removed_rows:
            writer.writerow([row["split"], row["image_path"], row["label_path"], row["line_index"], row["reason"], row["raw_line"]])

    summary = {
        "mode": "safe_remove",
        "note": "This is a non-destructive proposal. Original labels were not modified.",
        "proposed_label_root": str(labels_dir.resolve()),
        "manifest_csv": str(manifest_csv.resolve()),
        "removed_line_count": removed_line_count,
        "kept_line_count": kept_line_count,
        "touched_files": touched_files,
        "touched_files_by_split": dict(touched_files_by_split),
        "rules": {
            "invalid_box": "remove invalid lines",
            "large_edge_background_box": f"remove boxes that are edge-touching and extremely large (area >= {args.safe_remove_area_thresh} or side >= {args.safe_remove_side_thresh})",
        },
    }
    write_text_compat(summary_json, json.dumps(summary, indent=2, ensure_ascii=False))
    write_text_compat(
        summary_txt,
        "\n".join([
            "Safe Remove Proposal Summary",
            "Original labels were not modified.",
            f"Proposed label root: {summary['proposed_label_root']}",
            f"Manifest csv: {summary['manifest_csv']}",
            f"Removed line count: {removed_line_count}",
            f"Kept line count: {kept_line_count}",
            f"Touched files: {touched_files}",
            *[f"- {split}: {count}" for split, count in touched_files_by_split.items()],
        ]),
    )


def write_resample_outputs(output_dir: Path, all_image_records: list[ImageRecord], class_box_counts: Counter, class_names: list[str], args):
    resample_dir = output_dir / "resample_only"
    resample_dir.mkdir(parents=True, exist_ok=True)

    resample_csv = resample_dir / "resample_plan.csv"
    upsample_txt = resample_dir / "upsample_candidates.txt"
    downsample_txt = resample_dir / "downsample_candidates.txt"
    summary_json = resample_dir / "resample_summary.json"

    if not class_box_counts:
        write_text_compat(summary_json, json.dumps({"mode": "resample_only", "note": "No class counts available."}, indent=2, ensure_ascii=False))
        return

    majority_class_id, majority_count = class_box_counts.most_common(1)[0]
    minority_counts = [count for cls_id, count in class_box_counts.items() if cls_id != majority_class_id]
    target_count = int(sorted(minority_counts)[len(minority_counts) // 2]) if minority_counts else majority_count
    majority_keep_probability = min(1.0, math.sqrt(target_count / max(majority_count, 1)))

    plan_rows = []
    upsample_paths = []
    downsample_paths = []

    for record in all_image_records:
        if not record.boxes:
            continue
        total_boxes = len(record.boxes)
        present_classes = sorted(record.class_counter.keys())
        dominant_class, dominant_count = record.class_counter.most_common(1)[0]
        class_labels = [class_names[class_id] for class_id in present_classes]
        minority_repeat_candidates = [
            max(1.0, target_count / max(class_box_counts[class_id], 1))
            for class_id in present_classes
            if class_id != majority_class_id
        ]
        sample_weight = max(minority_repeat_candidates) if minority_repeat_candidates else 1.0
        repeat_factor = min(args.resample_max_repeat, max(1, math.ceil(sample_weight)))

        recommendation = "keep"
        keep_probability = 1.0
        if minority_repeat_candidates:
            recommendation = "upsample"
            upsample_paths.append(str(record.image_path))
        elif dominant_class == majority_class_id and dominant_count / total_boxes >= 0.6:
            recommendation = "downsample_candidate"
            keep_probability = majority_keep_probability
            downsample_paths.append(str(record.image_path))

        plan_rows.append({
            "split": record.split,
            "image_path": str(record.image_path),
            "label_path": str(record.label_path),
            "present_classes": ",".join(class_labels),
            "dominant_class": class_names[dominant_class],
            "box_count": total_boxes,
            "sample_weight": round(sample_weight, 4),
            "repeat_factor": repeat_factor,
            "keep_probability": round(keep_probability, 4),
            "recommendation": recommendation,
        })

    with resample_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image_path", "label_path", "present_classes", "dominant_class", "box_count", "sample_weight", "repeat_factor", "keep_probability", "recommendation"])
        for row in plan_rows:
            writer.writerow([
                row["split"], row["image_path"], row["label_path"], row["present_classes"], row["dominant_class"],
                row["box_count"], row["sample_weight"], row["repeat_factor"], row["keep_probability"], row["recommendation"]
            ])

    write_text_compat(upsample_txt, "\n".join(sorted(set(upsample_paths))))
    write_text_compat(downsample_txt, "\n".join(sorted(set(downsample_paths))))

    summary = {
        "mode": "resample_only",
        "note": "Recommendation only. Original dataset was not modified.",
        "majority_class": class_names[majority_class_id],
        "majority_count": majority_count,
        "target_count": target_count,
        "majority_keep_probability": round(majority_keep_probability, 4),
        "resample_plan_csv": str(resample_csv.resolve()),
        "upsample_candidates_txt": str(upsample_txt.resolve()),
        "downsample_candidates_txt": str(downsample_txt.resolve()),
        "upsample_image_count": len(set(upsample_paths)),
        "downsample_candidate_count": len(set(downsample_paths)),
    }
    write_text_compat(summary_json, json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    args = parse_args()
    dataset_root, class_names, splits = load_dataset_config(args.yaml.resolve())
    num_classes = len(class_names)
    args.output.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output / "previews"

    issue_rows: list[BoxRecord] = []
    high_risk_rows: list[dict] = []
    preview_candidates: list[dict] = []
    all_image_records: list[ImageRecord] = []
    split_summary = {}
    class_box_counts = Counter()
    issue_counts = Counter()
    valid_boxes_total = 0

    images_with_class = defaultdict(set)
    per_image_class_counter = {}

    for split_name, split_cfg in splits.items():
        images_dir = split_cfg["images_dir"]
        labels_dir = split_cfg["labels_dir"]
        image_files = sorted([p for p in images_dir.iterdir() if p.is_file()])
        label_files = sorted([p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
        image_map = {p.stem: p for p in image_files}
        label_map = {p.stem: p for p in label_files}

        split_stats = {
            "images": len(image_files),
            "labels": len(label_files),
            "missing_label_files": 0,
            "missing_image_files": 0,
            "empty_label_files": 0,
            "boxes": 0,
            "images_with_boxes": 0,
        }

        for stem, image_path in image_map.items():
            if stem not in label_map:
                split_stats["missing_label_files"] += 1
                issue_counts["missing_label_file"] += 1

        for stem, label_path in label_map.items():
            image_path = image_map.get(stem)
            if image_path is None:
                split_stats["missing_image_files"] += 1
                issue_counts["missing_image_file"] += 1
                continue

            record = ImageRecord(split=split_name, image_path=image_path, label_path=label_path)
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                split_stats["empty_label_files"] += 1
                issue_counts["empty_label_file"] += 1
                high_risk_rows.append({
                    "split": split_name,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "risk_score": 1,
                    "issue_types": "empty_label_file",
                    "box_count": 0,
                })
                all_image_records.append(record)
                continue

            split_stats["images_with_boxes"] += 1
            lines = text.splitlines()
            for line_index, line in enumerate(lines, start=1):
                parts = line.strip().split()
                if len(parts) != 5:
                    record.line_entries.append({
                        "line_index": line_index,
                        "raw_line": line.strip(),
                        "invalid": True,
                        "box": None,
                    })
                    issue_rows.append(BoxRecord(
                        split=split_name,
                        image_path=str(image_path),
                        label_path=str(label_path),
                        line_index=line_index,
                        class_id=None,
                        class_name="invalid",
                        x_center=None,
                        y_center=None,
                        width=None,
                        height=None,
                        area=None,
                        issue_type="invalid_box",
                        note=f"expected 5 fields, got {len(parts)}",
                    ))
                    issue_counts["invalid_box"] += 1
                    continue

                try:
                    class_id = int(float(parts[0]))
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError:
                    record.line_entries.append({
                        "line_index": line_index,
                        "raw_line": line.strip(),
                        "invalid": True,
                        "box": None,
                    })
                    issue_rows.append(BoxRecord(
                        split=split_name,
                        image_path=str(image_path),
                        label_path=str(label_path),
                        line_index=line_index,
                        class_id=None,
                        class_name="invalid",
                        x_center=None,
                        y_center=None,
                        width=None,
                        height=None,
                        area=None,
                        issue_type="invalid_box",
                        note="failed to parse numeric values",
                    ))
                    issue_counts["invalid_box"] += 1
                    continue

                class_name = class_names[class_id] if 0 <= class_id < num_classes else f"class_{class_id}"
                box = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "area": width * height,
                    "line_index": line_index,
                    "issues": [],
                }
                issues = detect_box_issues(box, num_classes, class_names, args)
                for issue_type, note in issues:
                    issue_rows.append(BoxRecord(
                        split=split_name,
                        image_path=str(image_path),
                        label_path=str(label_path),
                        line_index=line_index,
                        class_id=class_id,
                        class_name=class_name,
                        x_center=x_center,
                        y_center=y_center,
                        width=width,
                        height=height,
                        area=width * height,
                        issue_type=issue_type,
                        note=note,
                    ))
                    issue_counts[issue_type] += 1
                    box["issues"].append({"issue_type": issue_type, "note": note})

                record.line_entries.append({
                    "line_index": line_index,
                    "raw_line": line.strip(),
                    "invalid": any(issue[0] == "invalid_box" for issue in issues),
                    "box": box,
                })

                if not any(issue[0] == "invalid_box" for issue in issues):
                    valid_boxes_total += 1
                    split_stats["boxes"] += 1
                    class_box_counts[class_id] += 1
                    images_with_class[class_id].add(str(image_path))
                    record.class_counter[class_id] += 1
                    record.boxes.append(box)

            if record.boxes:
                areas = [box["area"] for box in record.boxes]
                huge_count = sum(any(issue["issue_type"] in {"huge_box", "large_area_box"} for issue in box["issues"]) for box in record.boxes)
                small_count = sum(box["area"] < args.dense_mixed_small_area for box in record.boxes)
                risk_score = 0
                if huge_count > 0 and small_count > 0:
                    record.image_issues.append({"issue_type": "dense_mixed_image", "note": "contains both huge and small boxes"})
                    issue_counts["dense_mixed_image"] += 1
                    risk_score += 3
                if len(record.boxes) >= args.many_objects_thresh:
                    record.image_issues.append({"issue_type": "many_objects_image", "note": "image has many labeled objects"})
                    issue_counts["many_objects_image"] += 1
                    risk_score += 1

                duplicate_pairs = 0
                for idx_a, box_a in enumerate(record.boxes):
                    for idx_b in range(idx_a + 1, len(record.boxes)):
                        box_b = record.boxes[idx_b]
                        if box_a["class_id"] != box_b["class_id"]:
                            continue
                        iou = bbox_iou_xywh(
                            (box_a["x_center"], box_a["y_center"], box_a["width"], box_a["height"]),
                            (box_b["x_center"], box_b["y_center"], box_b["width"], box_b["height"]),
                        )
                        if iou >= args.duplicate_iou:
                            duplicate_pairs += 1
                if duplicate_pairs > 0:
                    record.image_issues.append({"issue_type": "duplicate_overlap_image", "note": f"{duplicate_pairs} same-class high-IoU pairs"})
                    issue_counts["duplicate_overlap_image"] += duplicate_pairs
                    risk_score += 2

                if record.class_counter:
                    per_image_class_counter[str(record.image_path)] = Counter(record.class_counter)

                if risk_score > 0 or any(box["issues"] for box in record.boxes):
                    issue_type_names = [issue["issue_type"] for issue in record.image_issues]
                    issue_type_names.extend(
                        sorted({
                            issue["issue_type"]
                            for box in record.boxes
                            for issue in box["issues"]
                        })
                    )
                    high_risk_rows.append({
                        "split": split_name,
                        "image_path": str(record.image_path),
                        "label_path": str(record.label_path),
                        "risk_score": risk_score + sum(len(box["issues"]) for box in record.boxes),
                        "issue_types": ",".join(issue_type_names),
                        "box_count": len(record.boxes),
                    })
                    preview_candidates.append({
                        "split": split_name,
                        "image_path": record.image_path,
                        "label_path": record.label_path,
                        "boxes": record.boxes,
                        "image_issues": record.image_issues,
                        "risk_score": risk_score + sum(len(box["issues"]) for box in record.boxes),
                    })

            all_image_records.append(record)
        split_summary[split_name] = split_stats

    preview_candidates.sort(key=lambda item: item["risk_score"], reverse=True)
    high_risk_rows.sort(key=lambda item: item["risk_score"], reverse=True)

    issue_csv = args.output / "issue_samples.csv"
    with issue_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "image_path", "label_path", "line_index", "class_id", "class_name",
            "x_center", "y_center", "width", "height", "area", "issue_type", "note"
        ])
        for row in issue_rows:
            writer.writerow([
                row.split, row.image_path, row.label_path, row.line_index, row.class_id, row.class_name,
                row.x_center, row.y_center, row.width, row.height, row.area, row.issue_type, row.note
            ])

    risk_csv = args.output / "high_risk_images.csv"
    with risk_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image_path", "label_path", "risk_score", "issue_types", "box_count"])
        for row in high_risk_rows:
            writer.writerow([row["split"], row["image_path"], row["label_path"], row["risk_score"], row["issue_types"], row["box_count"]])

    majority_class_images = set()
    minority_class_images = set()
    majority_class_id = class_box_counts.most_common(1)[0][0] if class_box_counts else None
    minority_class_ids = sorted([cls_id for cls_id in class_box_counts if cls_id != majority_class_id])

    if majority_class_id is not None:
        for image_path, class_counter in per_image_class_counter.items():
            dominant_class, dominant_count = class_counter.most_common(1)[0]
            if dominant_class == majority_class_id and dominant_count / sum(class_counter.values()) >= 0.6:
                majority_class_images.add(image_path)

    for cls_id in minority_class_ids:
        minority_class_images.update(images_with_class[cls_id])

    majority_txt = args.output / "majority_class_images.txt"
    minority_txt = args.output / "minority_class_images.txt"
    write_text_compat(majority_txt, "\n".join(sorted(majority_class_images)))
    write_text_compat(minority_txt, "\n".join(sorted(minority_class_images)))

    summary = {
        "dataset_yaml": str(args.yaml.resolve()),
        "dataset_root": str(dataset_root),
        "output_dir": str(args.output.resolve()),
        "mode": args.mode,
        "num_classes": num_classes,
        "class_names": {idx: name for idx, name in enumerate(class_names)},
        "split_summary": split_summary,
        "valid_boxes_total": valid_boxes_total,
        "class_box_counts": {class_names[idx]: class_box_counts[idx] for idx in sorted(class_box_counts)},
        "issue_counts": dict(issue_counts),
        "majority_class": class_names[majority_class_id] if majority_class_id is not None else None,
        "minority_classes": [class_names[idx] for idx in minority_class_ids],
        "preview_count": min(args.preview_count, len(preview_candidates)),
        "generated_files": {
            "issue_samples_csv": str(issue_csv.resolve()),
            "high_risk_images_csv": str(risk_csv.resolve()),
            "majority_class_images_txt": str(majority_txt.resolve()),
            "minority_class_images_txt": str(minority_txt.resolve()),
            "preview_dir": str(preview_dir.resolve()),
            "review_dir": str((args.output / "review_only").resolve()),
            "safe_remove_dir": str((args.output / "safe_remove").resolve()),
            "resample_dir": str((args.output / "resample_only").resolve()),
        },
    }

    summary_json = args.output / "summary.json"
    write_text_compat(summary_json, json.dumps(summary, indent=2, ensure_ascii=False))

    lines = [
        "Dataset Issue Scan Summary",
        f"Dataset YAML: {summary['dataset_yaml']}",
        f"Dataset root: {summary['dataset_root']}",
        f"Valid boxes total: {valid_boxes_total}",
        f"Majority class: {summary['majority_class']}",
        "",
        "Split summary:",
    ]
    for split_name, stats in split_summary.items():
        lines.append(f"- {split_name}: images={stats['images']}, labels={stats['labels']}, images_with_boxes={stats['images_with_boxes']}, boxes={stats['boxes']}, missing_label_files={stats['missing_label_files']}, missing_image_files={stats['missing_image_files']}, empty_label_files={stats['empty_label_files']}")
    lines.append("")
    lines.append("Class box counts:")
    for class_name, count in summary["class_box_counts"].items():
        lines.append(f"- {class_name}: {count}")
    lines.append("")
    lines.append("Issue counts:")
    for issue_name, count in sorted(issue_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {issue_name}: {count}")
    lines.append("")
    lines.append("Generated files:")
    for key, value in summary["generated_files"].items():
        lines.append(f"- {key}: {value}")

    summary_txt = args.output / "summary.txt"
    write_text_compat(summary_txt, "\n".join(lines))

    if should_generate(args.mode, "analyze_only"):
        render_previews(preview_candidates, preview_dir, args.preview_count)

    if should_generate(args.mode, "review_only"):
        write_review_outputs(args.output, issue_rows, high_risk_rows, preview_candidates, args)

    if should_generate(args.mode, "safe_remove"):
        write_safe_remove_outputs(args.output, all_image_records, args)

    if should_generate(args.mode, "resample_only"):
        write_resample_outputs(args.output, all_image_records, class_box_counts, class_names, args)

    print(f"Analysis complete. Summary written to: {summary_txt}")
    print(f"Issue samples written to: {issue_csv}")
    print(f"High-risk images written to: {risk_csv}")
    if should_generate(args.mode, "analyze_only"):
        print(f"Previews written to: {preview_dir}")
    if should_generate(args.mode, "review_only"):
        print(f"Review outputs written to: {args.output / 'review_only'}")
    if should_generate(args.mode, "safe_remove"):
        print(f"Safe-remove proposal written to: {args.output / 'safe_remove'}")
    if should_generate(args.mode, "resample_only"):
        print(f"Resample plan written to: {args.output / 'resample_only'}")


if __name__ == "__main__":
    main()
