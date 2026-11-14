"""复制选中的硬件导出样图，在目标 detection 图左下角标注白色 IoU。

用法：在一张图里挑好样图后，把它们的文件名（如 ``0010.png``）放进一个文件夹，
本脚本按同名的 0001-based ID 从导出目录复制 input / teacher_feature / slm1 / slm2
（可加 detection、json）到项目内的目标目录，目录结构与编号保持不变。

示例（项目根目录执行）：
    python src/copy_selected_hardware_samples.py
    python src/copy_selected_hardware_samples.py --experiment-dir output/Tv2_physical_1to1_2gpu --all-groups --write-manifest
    python src/copy_selected_hardware_samples.py --source output/experiment/hardware_slm --selection output/experiment/hardware_slm_option --dest output/experiment/hardware_slm_selected
    python src/copy_selected_hardware_samples.py --ids 0010,0021,2114
    python src/copy_selected_hardware_samples.py --groups input,teacher_feature,slm1,slm2,detection,json
    python src/copy_selected_hardware_samples.py --all-groups --write-manifest
    python src/copy_selected_hardware_samples.py --dry-run

不重新推理；源目录不会被修改，目标目录中已存在的同名文件会被覆盖。
默认复制 detection/json，并从每张图的 JSON 计算同类别一对一匹配。
IoU: xx.x = 100 × mean_matched_iou（只平均 TP，非 mAP，漏检见 JSON）；
无成功匹配时显示 N/A。已有旧 JSON 也可补算，复制后的 JSON 会写入匹配字段。
使用 --no-annotate-iou 可保持原始复制行为。
三个路径默认统一从 --experiment-dir 派生：hardware_slm 是源目录，
hardware_slm_option 是已挑选图片目录，hardware_slm_selected 是导出目录。
--source / --selection / --dest 可分别覆盖；相对路径统一相对于项目根目录。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hardware_detection_iou import add_detection_iou

DEFAULT_EXPERIMENT_DIR = Path("output/Tv2_scratch_control_2gpu_seed42")
DEFAULT_GROUPS = ("input", "teacher_feature", "slm1", "slm2", "detection", "json")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Copy selected hardware export samples (input/teacher_feature/SLM/...) into the project.",
    )
    parser.add_argument(
        "--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR,
        help=f"Common root for default source/selection/destination paths (default: {DEFAULT_EXPERIMENT_DIR}).",
    )
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Hardware export root (default: <experiment-dir>/hardware_slm).",
    )
    parser.add_argument(
        "--selection", type=Path, default=None,
        help="Folder with picked image filenames (default: <experiment-dir>/hardware_slm_option).",
    )
    parser.add_argument(
        "--ids", default="",
        help="Explicit comma separated sample IDs (e.g. 0010,0021); when given, --selection is not scanned.",
    )
    parser.add_argument(
        "--dest", type=Path, default=None,
        help="Destination root (default: <experiment-dir>/hardware_slm_selected).",
    )
    parser.add_argument(
        "--groups", default=",".join(DEFAULT_GROUPS),
        help=f"Comma separated export subfolders to copy (default: {','.join(DEFAULT_GROUPS)})",
    )
    parser.add_argument(
        "--all-groups", action="store_true",
        help="Copy every subfolder found under --source (input, teacher_feature, slm1..N, detection, json, ...).",
    )
    parser.add_argument(
        "--write-manifest", action="store_true",
        help="Also write selection_manifest.json in --dest listing the copied IDs and files.",
    )
    parser.add_argument(
        "--annotate-iou", action=argparse.BooleanOptionalAction, default=True,
        help="Add white bottom-left IoU (100 * mean TP IoU) to copied detection images and enrich copied JSON; --no-annotate-iou disables this.",
    )
    parser.add_argument(
        "--match-iou-threshold", type=float, default=None,
        help="Override JSON match IoU threshold; default: use saved threshold, or 0.5 for old JSON.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only report what would be copied.",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Exit with code 1 when any selected ID is missing in a group.",
    )
    return parser.parse_args(argv)


def resolve(path: Path) -> Path:
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def normalize_id(token: str) -> str:
    """Match the export convention: IDs are 4-digit zero padded (0001, 0010, ...).

    PowerShell turns an unquoted ``0010,0021`` into the integers 10 and 21, so pad
    short numeric tokens back to the export width.
    """
    token = token.strip()
    if token.isdigit() and len(token) < 4:
        return token.zfill(4)
    return token


def collect_ids(selection: Path, explicit: str = "") -> list[str]:
    if explicit.strip():
        return sorted({normalize_id(token) for token in re.split(r"[,;\s]+", explicit) if token.strip()})
    if not selection.is_dir():
        raise SystemExit(f"Selection folder not found: {selection}")
    ids = {
        normalize_id(item.stem)
        for item in selection.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    }
    if not ids:
        raise SystemExit(f"No image files found in selection folder: {selection}")
    return sorted(ids)


def resolve_groups(source: Path, args) -> list[str]:
    if args.all_groups:
        groups = sorted(
            item.name for item in source.iterdir()
            if item.is_dir() and not item.name.startswith(".")
        )
    else:
        groups = [name.strip() for name in args.groups.split(",") if name.strip()]
    unknown = [name for name in groups if not (source / name).is_dir()]
    if unknown:
        raise SystemExit(f"Group folder(s) missing under {source}: {', '.join(unknown)}")
    if not groups:
        raise SystemExit("No groups selected.")
    return groups


def annotate_detection_iou(path: Path, record: dict | None) -> None:
    """Annotate only the destination image; white text without an opaque box."""
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(path) as source_image:
        canvas = source_image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font_size = max(24, round(canvas.height * 0.14))
    font = None
    for font_name in ("arial.ttf", "DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    value = record["iou_metrics"]["mean_matched_iou"] if record is not None else None
    label = "IoU: N/A" if value is None else f"IoU: {100 * value:.1f}"
    margin = max(2, round(min(canvas.size) * 0.012))
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    draw.text((margin - left, canvas.height - margin - bottom), label, font=font, fill="white")
    canvas.save(path)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.match_iou_threshold is not None and not 0 < args.match_iou_threshold <= 1:
        raise SystemExit("--match-iou-threshold must be in (0, 1].")
    experiment_dir = resolve(args.experiment_dir)
    source = resolve(args.source) if args.source is not None else experiment_dir / "hardware_slm"
    selection = resolve(args.selection) if args.selection is not None else experiment_dir / "hardware_slm_option"
    dest = resolve(args.dest) if args.dest is not None else experiment_dir / "hardware_slm_selected"
    source, selection, dest = source.resolve(), selection.resolve(), dest.resolve()
    if not source.is_dir():
        raise SystemExit(f"Source export folder not found: {source}")
    if dest == source or source in dest.parents:
        raise SystemExit("--dest must not be the source folder or live inside it.")
    if dest == selection or selection in dest.parents:
        raise SystemExit("--dest must not be the selection folder or live inside it.")

    ids = collect_ids(selection, args.ids)
    groups = resolve_groups(source, args)
    if not ids:
        raise SystemExit("No sample IDs selected.")
    print(f"Source   : {source}")
    print(f"Selection: {selection if not args.ids.strip() else '--ids on command line'} ({len(ids)} IDs)")
    print(f"Dest     : {dest}")
    print(f"Groups   : {', '.join(groups)}")
    if args.dry_run:
        print("Mode     : dry-run (nothing is written)")

    iou_records = {}
    if args.annotate_iou and ("detection" in groups or "json" in groups):
        for sample_id in ids:
            record_path = source / "json" / f"{sample_id}.json"
            if not record_path.is_file():
                print(f"Warning: {sample_id}: no JSON; detection will show IoU: N/A.", file=sys.stderr)
                continue
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if "detections" not in record or "ground_truth" not in record:
                print(f"Warning: {sample_id}: JSON lacks detections/ground_truth; IoU: N/A.", file=sys.stderr)
                continue
            saved_threshold = record.get("iou_metrics", {}).get("match_iou_threshold", 0.5)
            threshold = args.match_iou_threshold if args.match_iou_threshold is not None else saved_threshold
            iou_records[sample_id] = add_detection_iou(record, threshold)
        print("IoU label: mean matched TP IoU x 100; unmatched cases: N/A; JSON includes TP/FP/FN.")

    copied = {group: [] for group in groups}
    missing = {group: [] for group in groups}
    for group in groups:
        group_dir = source / group
        out_dir = dest / group
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
        for sample_id in ids:
            matches = sorted(group_dir.glob(f"{sample_id}.*"))
            matches = [path for path in matches if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES | {".json"}]
            if not matches:
                missing[group].append(sample_id)
                continue
            for path in matches:
                target = out_dir / path.name
                if not args.dry_run:
                    shutil.copy2(path, target)
                    if args.annotate_iou and group == "detection" and path.suffix.lower() in IMAGE_SUFFIXES:
                        annotate_detection_iou(target, iou_records.get(sample_id))
                    elif args.annotate_iou and group == "json" and path.suffix.lower() == ".json" and sample_id in iou_records:
                        target.write_text(json.dumps(iou_records[sample_id], ensure_ascii=False, indent=2), encoding="utf-8")
                copied[group].append(target.name)

    for group in groups:
        print(f"[{group}] copied={len(copied[group])} missing={len(missing[group])}")
        if missing[group]:
            preview = ", ".join(missing[group][:12])
            more = "" if len(missing[group]) <= 12 else f" ... (+{len(missing[group]) - 12})"
            print(f"  missing IDs: {preview}{more}")

    total = sum(len(value) for value in copied.values())
    if not args.dry_run and total and args.write_manifest:
        manifest = {
            "experiment_dir": str(experiment_dir),
            "source": str(source),
            "selection": str(selection),
            "dest": str(dest),
            "groups": groups,
            "ids": ids,
            "files": {group: copied[group] for group in groups},
            "iou_annotation": {
                "enabled": args.annotate_iou,
                "metric": "100 * mean_matched_iou (TP matches only)",
                "position": "bottom_left", "color": "white", "no_match": "N/A",
                "match_iou_threshold_override": args.match_iou_threshold,
                "sample_thresholds": {key: record["iou_metrics"]["match_iou_threshold"] for key, record in iou_records.items()},
            },
        }
        (dest / "selection_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Manifest : {dest / 'selection_manifest.json'}")

    print(f"{'Would copy' if args.dry_run else 'Copied'} {total} files for {len(ids)} selected IDs.")
    if args.strict and any(missing.values()):
        print("Strict mode: some selected IDs are incomplete.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
