"""Create a journal-style evaluation figure from ``evaluation_report.json``.

The report written by ``evaluate_teacher_v2_dataset.py`` contains per-class
confidence/TP labels, AP50 values, and scale-wise recall statistics.  This
script turns those real evaluation records into a compact four-panel figure:

1. per-class and macro precision-recall curves;
2. precision, recall, and F1 versus confidence threshold;
3. per-class AP50;
4. recall for small, medium, and large objects.

Example
-------
python src/plot_evaluation_report_figure.py \
    --report output/Tv2_dmd640_scratch/eval_results_val/evaluation_report.json \
    --output-stem paper/figures/military_validation_diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.class_display import display_class_name


DEFAULT_REPORT = Path(
    "output/Tv2_dmd640_scratch/eval_results_val/evaluation_report.json"
)
DEFAULT_OUTPUT_STEM = Path("paper/figures/evaluation_report_summary")

CLASS_COLORS = ["#3572B0", "#E6862A", "#36A35C", "#7B5AA6", "#A55E5E"]
METRIC_COLORS = {
    "Precision": "#3572B0",
    "Recall": "#E6862A",
    "F1": "#258B8C",
}
SIZE_COLORS = ["#6FA8DC", "#71B7A5", "#8064A2"]
GRID_COLOR = "#D7DADD"
TEXT_COLOR = "#222222"
EDGE_COLOR = "#303030"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot journal-style diagnostics from evaluation_report.json."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Evaluation report JSON produced by a dataset evaluation script.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_OUTPUT_STEM,
        help="Output path without extension; PNG and PDF are both generated.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution in dots per inch.",
    )
    parser.add_argument(
        "--panel-labels",
        action="store_true",
        help="Add panel letters a-d. They are hidden by default.",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation report not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("The report does not contain a valid 'metrics' object.")
    for required in ("per_class", "pr_data", "size_recall", "map50"):
        if required not in metrics:
            raise ValueError(f"The report metrics are missing '{required}'.")
    return report


def sorted_class_ids(metrics: Mapping[str, Any]) -> list[str]:
    class_ids = list(metrics["per_class"].keys())
    try:
        return sorted(class_ids, key=lambda value: int(value))
    except ValueError:
        return sorted(class_ids)


def pretty_class_names(report: Mapping[str, Any], class_ids: Sequence[str]) -> list[str]:
    raw_labels = list(report.get("labels_normalized_matrix", []))
    names: list[str] = []
    for position, class_id in enumerate(class_ids):
        numeric_id = int(class_id) if str(class_id).isdigit() else position
        raw_name = (
            str(raw_labels[numeric_id])
            if 0 <= numeric_id < len(raw_labels)
            else f"Class {class_id}"
        )
        names.append(raw_name)

    names = [display_class_name(name) for name in names]
    dataset_yaml = str(report.get("dataset_yaml", ""))
    dataset_prefix = Path(dataset_yaml).parent.name.lower() if dataset_yaml else ""
    if dataset_prefix:
        names = [name.removeprefix(f"{dataset_prefix}_") for name in names]

    prefixes = [name.split("_", maxsplit=1)[0] for name in names if "_" in name]
    common_prefix = prefixes[0] if len(prefixes) == len(names) and len(set(prefixes)) == 1 else ""
    if common_prefix:
        names = [name.removeprefix(f"{common_prefix}_") for name in names]

    return [name.replace("_", " ").title() for name in names]


def real_detection_records(
    metrics: Mapping[str, Any], class_id: str
) -> tuple[np.ndarray, np.ndarray]:
    pr_data = metrics["pr_data"].get(class_id)
    if pr_data is None:
        raise ValueError(f"Missing pr_data for class {class_id}.")

    confidence = np.asarray(pr_data.get("confidence", []), dtype=float)
    labels = np.asarray(pr_data.get("label", []), dtype=np.int8)
    if confidence.ndim != 1 or labels.ndim != 1 or confidence.size != labels.size:
        raise ValueError(
            f"Class {class_id} confidence and label arrays must be one-dimensional "
            "and have equal length."
        )
    if not np.isfinite(confidence).all():
        raise ValueError(f"Class {class_id} contains non-finite confidence values.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Class {class_id} labels must contain only 0 and 1.")

    # The evaluator appends (confidence=0, label=1) entries for false negatives
    # so TensorBoard receives the correct number of positives.  They belong in
    # the recall denominator, but are not actual detections to threshold.
    real_mask = confidence > 0.0
    return confidence[real_mask], labels[real_mask]


def interpolated_pr_curve(
    confidence: np.ndarray,
    labels: np.ndarray,
    gt_count: int,
    recall_grid: np.ndarray,
) -> np.ndarray:
    if gt_count <= 0 or confidence.size == 0:
        return np.zeros_like(recall_grid)

    order = np.argsort(-confidence, kind="stable")
    sorted_labels = labels[order].astype(float)
    true_positives = np.cumsum(sorted_labels)
    false_positives = np.cumsum(1.0 - sorted_labels)
    recall = true_positives / float(gt_count)
    precision = true_positives / np.maximum(true_positives + false_positives, 1.0)

    # Remove leading FP-only samples. Keeping several recall=0 samples creates
    # an artificial vertical line coincident with the y-axis in a PR plot.
    positive_recall = recall > 0.0
    if not positive_recall.any():
        return np.zeros_like(recall_grid)
    recall = recall[positive_recall]
    precision = precision[positive_recall]

    recall_points = np.concatenate(([0.0], recall))
    precision_points = np.concatenate(([precision[0]], precision))
    precision_points = np.maximum.accumulate(precision_points[::-1])[::-1]

    maximum_recall = float(recall_points[-1])
    if maximum_recall < 1.0:
        recall_points = np.concatenate(
            (recall_points, [np.nextafter(maximum_recall, 1.0), 1.0])
        )
        precision_points = np.concatenate((precision_points, [0.0, 0.0]))
    else:
        recall_points = np.concatenate((recall_points, [1.0]))
        precision_points = np.concatenate((precision_points, [0.0]))

    return np.interp(recall_grid, recall_points, precision_points)


def global_threshold_curves(
    records: Sequence[tuple[np.ndarray, np.ndarray]],
    total_gt: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    confidence = np.concatenate([record[0] for record in records])
    labels = np.concatenate([record[1] for record in records]).astype(float)
    thresholds = np.linspace(0.0, 1.0, 501)

    order = np.argsort(-confidence, kind="stable")
    confidence = confidence[order]
    labels = labels[order]
    cumulative_tp = np.cumsum(labels)
    cumulative_fp = np.cumsum(1.0 - labels)

    precision = np.full_like(thresholds, np.nan)
    recall = np.zeros_like(thresholds)
    f1 = np.zeros_like(thresholds)
    negative_confidence = -confidence

    for index, threshold in enumerate(thresholds):
        selected = int(
            np.searchsorted(negative_confidence, -threshold, side="right")
        )
        if selected == 0:
            continue
        tp = cumulative_tp[selected - 1]
        fp = cumulative_fp[selected - 1]
        precision[index] = tp / max(tp + fp, 1.0)
        recall[index] = tp / max(float(total_gt), 1.0)
        f1[index] = (
            2.0 * precision[index] * recall[index]
            / max(precision[index] + recall[index], 1e-12)
        )

    return thresholds, precision, recall, f1


def style_axis(axis: plt.Axes, *, both_grid_axes: bool = False) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(EDGE_COLOR)
    axis.spines["bottom"].set_color(EDGE_COLOR)
    axis.tick_params(direction="out", length=3.0, width=0.8, colors=TEXT_COLOR)
    axis.grid(
        axis="both" if both_grid_axes else "y",
        color=GRID_COLOR,
        linestyle=(0, (2.0, 2.0)),
        linewidth=0.65,
        zorder=0,
    )


def add_bar_labels(axis: plt.Axes, bars: Any, digits: int = 3) -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.annotate(
            f"{value:.{digits}f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color=TEXT_COLOR,
            clip_on=False,
        )


def add_panel_labels(axes: Sequence[plt.Axes]) -> None:
    for label, axis in zip("abcd", axes):
        axis.text(
            -0.14,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="top",
        )


def plot_report(
    report: Mapping[str, Any],
    output_stem: Path,
    *,
    dpi: int,
    panel_labels: bool,
) -> tuple[Path, Path]:
    metrics = report["metrics"]
    class_ids = sorted_class_ids(metrics)
    class_names = pretty_class_names(report, class_ids)
    class_colors = [CLASS_COLORS[index % len(CLASS_COLORS)] for index in range(len(class_ids))]

    per_class = metrics["per_class"]
    gt_counts = [int(per_class[class_id].get("gt_count", 0)) for class_id in class_ids]
    ap50_values = [float(per_class[class_id].get("ap50", 0.0)) for class_id in class_ids]
    records = [real_detection_records(metrics, class_id) for class_id in class_ids]

    output_stem = output_stem.with_suffix("")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")

    rc_params = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.4,
        "axes.linewidth": 0.8,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(rc_params):
        figure, axes_grid = plt.subplots(
            2,
            2,
            figsize=(7.15, 5.65),
            constrained_layout=True,
        )
        axis_pr, axis_threshold, axis_ap, axis_size = axes_grid.ravel()

        # (a) Precision-recall curves.
        recall_grid = np.linspace(0.0, 1.0, 501)
        class_precision_curves: list[np.ndarray] = []
        for class_name, color, ap50, gt_count, record in zip(
            class_names, class_colors, ap50_values, gt_counts, records
        ):
            precision_curve = interpolated_pr_curve(
                record[0], record[1], gt_count, recall_grid
            )
            class_precision_curves.append(precision_curve)
            axis_pr.plot(
                recall_grid,
                precision_curve,
                color=color,
                linewidth=1.5,
                label=f"{class_name} ({ap50:.3f})",
            )

        macro_precision = np.mean(class_precision_curves, axis=0)
        map50 = float(metrics.get("map50", np.mean(ap50_values)))
        axis_pr.plot(
            recall_grid,
            macro_precision,
            color="#1D1D1D",
            linewidth=2.2,
            label=f"Macro (mAP$_{{50}}$={map50:.3f})",
            zorder=5,
        )
        axis_pr.set_title("Precision–recall curves")
        axis_pr.set_xlabel("Recall")
        axis_pr.set_ylabel("Precision")
        axis_pr.set_xlim(0.0, 1.0)
        axis_pr.set_ylim(0.0, 1.02)
        axis_pr.legend(loc="lower left", frameon=True, framealpha=0.9)
        style_axis(axis_pr, both_grid_axes=True)

        # (b) Global precision/recall/F1 versus confidence threshold.
        thresholds, precision_curve, recall_curve, f1_curve = global_threshold_curves(
            records, sum(gt_counts)
        )
        best_index = int(np.nanargmax(f1_curve))
        best_threshold = float(thresholds[best_index])
        best_f1 = float(f1_curve[best_index])
        axis_threshold.plot(
            thresholds,
            precision_curve,
            color=METRIC_COLORS["Precision"],
            linewidth=1.7,
            label="Precision",
        )
        axis_threshold.plot(
            thresholds,
            recall_curve,
            color=METRIC_COLORS["Recall"],
            linewidth=1.7,
            label="Recall",
        )
        axis_threshold.plot(
            thresholds,
            f1_curve,
            color=METRIC_COLORS["F1"],
            linewidth=2.0,
            label=rf"F1 (max={best_f1:.3f} at $\tau$={best_threshold:.3f})",
        )
        operating_threshold = float(report.get("matrix_confidence_threshold", 0.35))
        axis_threshold.axvline(
            operating_threshold,
            color="#666666",
            linestyle="--",
            linewidth=1.0,
            label=rf"Operating $\tau$={operating_threshold:.2f}",
        )
        axis_threshold.scatter(
            [best_threshold],
            [best_f1],
            s=24,
            color=METRIC_COLORS["F1"],
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        axis_threshold.set_title("Confidence-threshold analysis")
        axis_threshold.set_xlabel("Confidence threshold")
        axis_threshold.set_ylabel("Metric value")
        axis_threshold.set_xlim(0.0, 1.0)
        axis_threshold.set_ylim(0.0, 1.02)
        axis_threshold.legend(loc="lower center", ncol=2, frameon=True, framealpha=0.9)
        style_axis(axis_threshold, both_grid_axes=True)

        # (c) Per-class AP50.
        x_class = np.arange(len(class_ids), dtype=float)
        ap_bars = axis_ap.bar(
            x_class,
            ap50_values,
            width=0.56,
            color=class_colors,
            edgecolor=EDGE_COLOR,
            linewidth=0.7,
            zorder=3,
        )
        axis_ap.axhline(
            map50,
            color="#333333",
            linestyle="--",
            linewidth=1.1,
        )
        add_bar_labels(axis_ap, ap_bars)
        axis_ap.set_title(
            rf"Per-class AP$_{{50}}$ (mAP$_{{50}}$={map50:.3f})"
        )
        axis_ap.set_ylabel("AP$_{50}$")
        axis_ap.set_xticks(x_class, class_names)
        axis_ap.set_ylim(0.0, 1.06)
        style_axis(axis_ap)

        # (d) Recall stratified by object size.
        size_order = ["small", "medium", "large"]
        size_recall = metrics["size_recall"]
        size_counts = metrics.get("size_gt_count", {})
        size_values = [float(size_recall.get(name, 0.0)) for name in size_order]
        size_labels = [
            f"{name.title()}\n(n={int(size_counts.get(name, 0))})"
            for name in size_order
        ]
        x_size = np.arange(len(size_order), dtype=float)
        size_bars = axis_size.bar(
            x_size,
            size_values,
            width=0.52,
            color=SIZE_COLORS,
            edgecolor=EDGE_COLOR,
            linewidth=0.7,
            zorder=3,
        )
        add_bar_labels(axis_size, size_bars)
        axis_size.set_title("Scale-wise recall")
        axis_size.set_ylabel("Recall")
        axis_size.set_xticks(x_size, size_labels)
        axis_size.set_ylim(0.0, 1.06)
        style_axis(axis_size)

        if panel_labels:
            add_panel_labels(list(axes_grid.ravel()))

        figure.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        plt.close(figure)

    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    report = load_report(args.report)
    png_path, pdf_path = plot_report(
        report,
        args.output_stem,
        dpi=args.dpi,
        panel_labels=args.panel_labels,
    )
    print(f"Saved PNG: {png_path.resolve()}")
    print(f"Saved PDF: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
