"""Export real precision--recall curves and per-class AP from TensorBoard.

The training code records PR data with ``SummaryWriter.add_pr_curve`` under
``PRCurve/<class>`` and AP50 scalars under one of the supported tag layouts.
This script reads those summaries directly; it never synthesizes PR curves
from AP values.

Example
-------
python src/export_tensorboard_pr_ap.py \
    --logdir output/Tv2_dmd640_scratch/tensorboard \
    --output output/Tv2_dmd640_scratch/tensorboard_exports
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.util import tensor_util


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.class_display import display_class_name


EVENT_GLOB = "events.out.tfevents.*"
DEFAULT_COLORS = (
    "#3B82C4",
    "#F28E2B",
    "#3FAE5A",
    "#8064A2",
    "#C44E52",
    "#4C9F9A",
    "#D4A72C",
    "#7A7A7A",
)


@dataclass(frozen=True)
class PRCurve:
    class_name: str
    recall: np.ndarray
    precision: np.ndarray
    ap50: float
    pr_step: int
    ap_step: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TensorBoard PR curves and per-class AP50 as two publication-ready figures."
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        required=True,
        help=(
            "TensorBoard run directory, or a parent directory containing runs. "
            "When multiple runs are found, the run with the newest event file is used."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory for exported figures.")
    parser.add_argument(
        "--step",
        default="best",
        help="TensorBoard step to export, 'best' (default, highest mAP50), or 'latest'.",
    )
    parser.add_argument("--pr-prefix", default="PRCurve/", help="PR tensor tag prefix.")
    parser.add_argument(
        "--ap-tag-template",
        default="auto",
        help=(
            "AP scalar tag template containing '{class}', for example "
            "'MetricsPerClass/{class}/ap50'. Default: auto."
        ),
    )
    parser.add_argument(
        "--map-tag",
        default="auto",
        help="Overall mAP scalar tag, or 'auto' (default).",
    )
    parser.add_argument(
        "--class-alias",
        action="append",
        default=[],
        metavar="TAG=LABEL",
        help="Override a displayed class name. May be repeated.",
    )
    parser.add_argument("--iou", default="0.5", help="IoU label used in titles. Default: 0.5.")
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI. Default: 300.")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png",),
        help="One or more output formats. Default: png.",
    )
    parser.add_argument(
        "--no-macro-curve",
        action="store_true",
        help="Do not draw the derived macro-average PR curve.",
    )
    parser.add_argument(
        "--curve-style",
        choices=("envelope", "raw"),
        default="envelope",
        help=(
            "PR rendering style. 'envelope' (default) merges repeated recall values "
            "and applies the standard monotonic precision envelope for publication; "
            "'raw' preserves every TensorBoard threshold point."
        ),
    )
    return parser.parse_args()


def find_event_dir(logdir: Path) -> Path:
    logdir = logdir.expanduser().resolve()
    if not logdir.exists():
        raise FileNotFoundError(f"TensorBoard path does not exist: {logdir}")

    direct = list(logdir.glob(EVENT_GLOB)) if logdir.is_dir() else []
    if direct:
        return logdir

    event_files = list(logdir.rglob(EVENT_GLOB)) if logdir.is_dir() else []
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {logdir}")
    newest = max(event_files, key=lambda path: path.stat().st_mtime)
    return newest.parent


def load_event_accumulator(event_dir: Path) -> event_accumulator.EventAccumulator:
    # The project logs many diagnostic histograms and images. Keep only one of
    # those heavy summaries while retaining all scalars and all PR tensors.
    size_guidance = {
        event_accumulator.SCALARS: 0,
        event_accumulator.TENSORS: 10000,
        event_accumulator.IMAGES: 1,
        event_accumulator.HISTOGRAMS: 1,
        event_accumulator.COMPRESSED_HISTOGRAMS: 1,
        event_accumulator.AUDIO: 1,
    }
    accumulator = event_accumulator.EventAccumulator(str(event_dir), size_guidance=size_guidance)
    accumulator.Reload()
    return accumulator


def parse_aliases(items: Sequence[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --class-alias {item!r}; expected TAG=LABEL.")
        key, label = item.split("=", 1)
        key, label = key.strip(), label.strip()
        if not key or not label:
            raise ValueError(f"Invalid --class-alias {item!r}; expected non-empty TAG=LABEL.")
        aliases[key] = label
    return aliases


def display_name(class_name: str, aliases: dict[str, str]) -> str:
    if class_name in aliases:
        return aliases[class_name]
    return display_class_name(class_name).replace("_", " ").strip().title()


def discover_pr_tags(
    accumulator: event_accumulator.EventAccumulator, prefix: str
) -> list[tuple[str, str]]:
    tensor_tags = accumulator.Tags().get(event_accumulator.TENSORS, [])
    result = [(tag.removeprefix(prefix), tag) for tag in tensor_tags if tag.startswith(prefix)]
    if not result:
        raise RuntimeError(
            f"No PR tensor tags matching {prefix!r} were found. "
            "This run did not log SummaryWriter.add_pr_curve data, so a real PR curve cannot be exported."
        )
    return result


def resolve_ap_tags(
    scalar_tags: set[str], class_names: Sequence[str], template: str
) -> tuple[str, dict[str, str]]:
    if template != "auto":
        if "{class}" not in template:
            raise ValueError("--ap-tag-template must contain '{class}'.")
        tags = {name: template.format_map({"class": name}) for name in class_names}
        missing = [tag for tag in tags.values() if tag not in scalar_tags]
        if missing:
            raise RuntimeError("Missing requested AP scalar tags: " + ", ".join(missing))
        return template, tags

    candidates = [
        "MetricsPerClass/{class}/ap50",
        "Metrics/All_Stages/AP/{class}",
    ]
    # Also support stage-specific SLM layouts, such as Metrics/joint_fit/AP/<class>.
    prefixes = sorted(
        {
            match.group(1)
            for tag in scalar_tags
            if (match := re.match(r"^(Metrics/[^/]+/AP/).+$", tag)) is not None
        }
    )
    candidates.extend(prefix + "{class}" for prefix in prefixes)

    for candidate in candidates:
        tags = {name: candidate.format_map({"class": name}) for name in class_names}
        if all(tag in scalar_tags for tag in tags.values()):
            return candidate, tags
    raise RuntimeError(
        "Could not find a complete set of per-class AP scalar tags. "
        "Pass --ap-tag-template with a template containing '{class}'."
    )


def resolve_map_tag(scalar_tags: set[str], requested: str, ap_template: str) -> str | None:
    if requested != "auto":
        if requested not in scalar_tags:
            raise RuntimeError(f"Requested mAP scalar tag was not found: {requested}")
        return requested

    candidates = ["Metrics/map50", "Metrics/All_Stages/map50"]
    stage_match = re.match(r"^(Metrics/[^/]+)/AP/\{class\}$", ap_template)
    if stage_match:
        candidates.insert(0, stage_match.group(1) + "/map50")
    return next((tag for tag in candidates if tag in scalar_tags), None)


def tensor_steps(accumulator: event_accumulator.EventAccumulator, tag: str) -> set[int]:
    return {int(event.step) for event in accumulator.Tensors(tag)}


def scalar_steps(accumulator: event_accumulator.EventAccumulator, tag: str) -> set[int]:
    return {int(event.step) for event in accumulator.Scalars(tag)}


def select_target_step(
    accumulator: event_accumulator.EventAccumulator,
    pr_tags: Iterable[str],
    ap_tags: Iterable[str],
    requested: str,
    map_tag: str | None,
) -> int:
    all_step_sets = [tensor_steps(accumulator, tag) for tag in pr_tags]
    all_step_sets.extend(scalar_steps(accumulator, tag) for tag in ap_tags)
    if any(not steps for steps in all_step_sets):
        raise RuntimeError("At least one required TensorBoard series is empty.")

    common_steps = set.intersection(*all_step_sets)
    if requested == "best":
        if map_tag is not None:
            candidates = [
                event for event in accumulator.Scalars(map_tag) if int(event.step) in common_steps
            ]
            if candidates:
                return int(max(candidates, key=lambda event: float(event.value)).step)
        # A missing overall mAP tag cannot define "best" without inventing a
        # selection rule, so fall back to the latest fully aligned step.
        if common_steps:
            return max(common_steps)
        return min(max(steps) for steps in all_step_sets)
    if requested == "latest":
        if common_steps:
            return max(common_steps)
        return min(max(steps) for steps in all_step_sets)

    try:
        requested_step = int(requested)
    except ValueError as exc:
        raise ValueError("--step must be an integer, 'best', or 'latest'.") from exc
    eligible_common = [step for step in common_steps if step <= requested_step]
    return max(eligible_common) if eligible_common else requested_step


def latest_at_or_before(events: Sequence, target_step: int, tag: str):
    eligible = [event for event in events if int(event.step) <= target_step]
    if not eligible:
        available = sorted(int(event.step) for event in events)
        raise RuntimeError(
            f"No event for {tag!r} at or before step {target_step}; available steps start at {available[0]}."
        )
    return max(eligible, key=lambda event: int(event.step))


def decode_pr_tensor(tensor_proto, tag: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tensor_util.make_ndarray(tensor_proto), dtype=np.float64)
    values = np.squeeze(values)
    if values.ndim == 1 and values.size % 6 == 0:
        values = values.reshape(6, -1)
    elif values.ndim == 2 and values.shape[1] == 6 and values.shape[0] != 6:
        values = values.T
    if values.ndim != 2 or values.shape[0] != 6:
        raise RuntimeError(
            f"Unexpected PR tensor shape for {tag!r}: {values.shape}; expected [6, thresholds]."
        )

    precision = values[4]
    recall = values[5]
    valid = np.isfinite(precision) & np.isfinite(recall)
    precision = np.clip(precision[valid], 0.0, 1.0)
    recall = np.clip(recall[valid], 0.0, 1.0)
    if precision.size < 2:
        raise RuntimeError(f"PR tensor {tag!r} contains fewer than two valid points.")
    order = np.argsort(recall, kind="stable")
    return recall[order], precision[order]


def read_curves(
    accumulator: event_accumulator.EventAccumulator,
    pr_entries: Sequence[tuple[str, str]],
    ap_tags: dict[str, str],
    target_step: int,
) -> list[PRCurve]:
    curves: list[PRCurve] = []
    for class_name, pr_tag in pr_entries:
        pr_event = latest_at_or_before(accumulator.Tensors(pr_tag), target_step, pr_tag)
        ap_tag = ap_tags[class_name]
        ap_event = latest_at_or_before(accumulator.Scalars(ap_tag), target_step, ap_tag)
        recall, precision = decode_pr_tensor(pr_event.tensor_proto, pr_tag)
        curves.append(
            PRCurve(
                class_name=class_name,
                recall=recall,
                precision=precision,
                ap50=float(ap_event.value),
                pr_step=int(pr_event.step),
                ap_step=int(ap_event.step),
            )
        )
    return curves


def read_map50(
    accumulator: event_accumulator.EventAccumulator,
    map_tag: str | None,
    target_step: int,
    curves: Sequence[PRCurve],
) -> tuple[float, int | None]:
    if map_tag is None:
        return float(np.mean([curve.ap50 for curve in curves])), None
    event = latest_at_or_before(accumulator.Scalars(map_tag), target_step, map_tag)
    return float(event.value), int(event.step)


def publication_curve(curve: PRCurve) -> tuple[np.ndarray, np.ndarray]:
    """Return a compact monotonic PR curve suitable for publication.

    TensorBoard stores one point per confidence threshold. Several adjacent
    thresholds may change only the false-positive count, leaving recall
    unchanged while precision changes. Connecting those duplicate-recall
    points produces vertical segments, most visibly at recall=0. For display,
    retain the best precision at each recall and apply the conventional
    right-to-left precision envelope. AP values are still read from the logged
    scalar summaries and are not recomputed from this display curve.
    """
    recalls = np.unique(curve.recall)
    precisions = np.array(
        [np.max(curve.precision[curve.recall == recall]) for recall in recalls], dtype=np.float64
    )
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    return recalls, precisions


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> list[Path]:
    paths: list[Path] = []
    for output_format in formats:
        path = output_dir / f"{stem}.{output_format}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_pr_curves(
    curves: Sequence[PRCurve],
    map50: float,
    aliases: dict[str, str],
    iou_label: str,
    show_macro: bool,
    curve_style: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    for index, curve in enumerate(curves):
        color = DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
        label = display_name(curve.class_name, aliases)
        if curve_style == "envelope":
            recall, precision = publication_curve(curve)
        else:
            recall, precision = curve.recall, curve.precision
        ax.plot(
            recall,
            precision,
            color=color,
            linewidth=2.0,
            label=f"{label} (AP50={curve.ap50:.3f})",
        )

    if show_macro:
        recall_grid = np.linspace(0.0, 1.0, 501)
        interpolated = []
        for curve in curves:
            recall, precision = publication_curve(curve)
            interpolated.append(np.interp(recall_grid, recall, precision, left=precision[0], right=0.0))
        macro_precision = np.mean(interpolated, axis=0)
        ax.plot(
            recall_grid,
            macro_precision,
            color="#202020",
            linewidth=2.6,
            label=f"All classes (mAP50={map50:.3f})",
        )

    ax.set_title(f"Precision–Recall Curves (IoU = {iou_label})", weight="semibold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#D9E0E6", linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#C9D1D9")
    return fig


def plot_per_class_ap(
    curves: Sequence[PRCurve], map50: float, aliases: dict[str, str], iou_label: str
) -> plt.Figure:
    labels = [display_name(curve.class_name, aliases) for curve in curves] + ["All classes"]
    values = [curve.ap50 for curve in curves] + [map50]
    colors = [DEFAULT_COLORS[index % len(DEFAULT_COLORS)] for index in range(len(curves))] + ["#202020"]

    fig_height = max(4.2, 0.58 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(6.2, fig_height), constrained_layout=True)
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, height=0.62, edgecolor="none")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Average Precision (AP)")
    ax.set_title(f"Per-class AP (IoU = {iou_label})", weight="semibold")
    ax.grid(axis="x", color="#D9E0E6", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for bar, value in zip(bars, values):
        inside = value >= 0.94
        ax.text(
            value - 0.015 if inside else value + 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="right" if inside else "left",
            color="white" if inside else "#202020",
            fontsize=9.5,
            weight="semibold",
        )
    return fig


def main() -> None:
    args = parse_args()
    aliases = parse_aliases(args.class_alias)
    event_dir = find_event_dir(args.logdir)
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TensorBoard run: {event_dir}")
    accumulator = load_event_accumulator(event_dir)
    pr_entries = discover_pr_tags(accumulator, args.pr_prefix)
    class_names = [class_name for class_name, _ in pr_entries]
    scalar_tags = set(accumulator.Tags().get(event_accumulator.SCALARS, []))
    ap_template, ap_tags = resolve_ap_tags(scalar_tags, class_names, args.ap_tag_template)
    map_tag = resolve_map_tag(scalar_tags, args.map_tag, ap_template)
    target_step = select_target_step(
        accumulator,
        (tag for _, tag in pr_entries),
        ap_tags.values(),
        args.step,
        map_tag,
    )
    curves = read_curves(accumulator, pr_entries, ap_tags, target_step)
    map50, map_step = read_map50(accumulator, map_tag, target_step, curves)

    configure_style()
    pr_figure = plot_pr_curves(
        curves,
        map50,
        aliases,
        args.iou,
        show_macro=not args.no_macro_curve,
        curve_style=args.curve_style,
    )
    ap_figure = plot_per_class_ap(curves, map50, aliases, args.iou)
    output_paths = []
    output_paths.extend(save_figure(pr_figure, output_dir, "pr_curves", args.formats, args.dpi))
    output_paths.extend(save_figure(ap_figure, output_dir, "per_class_ap", args.formats, args.dpi))

    print(f"Selected target step: {target_step}")
    for curve in curves:
        print(
            f"  {curve.class_name}: AP50={curve.ap50:.6f}, "
            f"PR step={curve.pr_step}, AP step={curve.ap_step}"
        )
    map_source = f"tag={map_tag}, step={map_step}" if map_tag else "mean of per-class AP50"
    print(f"  all classes: mAP50={map50:.6f} ({map_source})")
    for path in output_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
