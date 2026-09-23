"""Plot a publication-style grouped mAP50 bar chart.

The default values reproduce the current validation-performance panel used in
the manuscript draft.  Edit ``DEFAULT_*`` below or pass values on the command
line when updated experimental results become available.

Example
-------
python src/plot_validation_performance_bars.py \
    --datasets Fashion Military \
    --digital 0.970 0.839 \
    --static-slm 0.929 0.537
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Editable default data
# ---------------------------------------------------------------------------
DEFAULT_DATASETS = ["Fashion", "Military"]
DEFAULT_DIGITAL_MAP50 = [0.970, 0.839]
DEFAULT_STATIC_SLM_MAP50 = [0.929, 0.631]


DIGITAL_COLOR = "#4A4A4A"
STATIC_SLM_COLOR = "#258B8C"
EDGE_COLOR = "#303030"
GRID_COLOR = "#D4D7D9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a journal-style grouped mAP50 bar chart."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset labels shown on the x-axis.",
    )
    parser.add_argument(
        "--digital",
        nargs="+",
        type=float,
        default=DEFAULT_DIGITAL_MAP50,
        help="mAP50 values for the digital-domain baseline.",
    )
    parser.add_argument(
        "--static-slm",
        nargs="+",
        type=float,
        default=DEFAULT_STATIC_SLM_MAP50,
        help="mAP50 values for the static dual-SLM branch.",
    )
    parser.add_argument(
        "--title",
        default="Current validation performance",
        help="Panel title. Pass an empty string to hide it.",
    )
    parser.add_argument(
        "--panel-label",
        default="",
        help="Optional panel label placed at the upper-left (hidden by default).",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("output/figures/current_validation_performance"),
        help="Output path without an extension; both PNG and PDF are written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution in dots per inch.",
    )
    return parser.parse_args()


def validate_data(
    datasets: Sequence[str],
    digital: Sequence[float],
    static_slm: Sequence[float],
) -> None:
    lengths = {len(datasets), len(digital), len(static_slm)}
    if len(lengths) != 1:
        raise ValueError(
            "--datasets, --digital, and --static-slm must contain the same "
            "number of entries."
        )
    if not datasets:
        raise ValueError("At least one dataset is required.")

    values = np.asarray([*digital, *static_slm], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("All mAP50 values must be finite numbers.")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("All mAP50 values must lie in the interval [0, 1].")


def add_value_labels(
    axis: plt.Axes,
    bars: Sequence[matplotlib.patches.Rectangle],
) -> None:
    for bar in bars:
        value = bar.get_height()
        axis.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11.5,
            color="#222222",
            clip_on=False,
        )


def plot_validation_performance(
    datasets: Sequence[str],
    digital: Sequence[float],
    static_slm: Sequence[float],
    output_stem: Path,
    *,
    title: str,
    panel_label: str,
    dpi: int,
) -> tuple[Path, Path]:
    validate_data(datasets, digital, static_slm)

    output_stem = output_stem.with_suffix("")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")

    # Set spacing relative to the full pair of bars, so inter-dataset whitespace
    # stays compact without collapsing the gap between the two series.
    width = 0.13
    intra_group_gap = 0.065
    inter_group_gap = 0.18
    group_width = 2 * width + intra_group_gap
    x = np.arange(len(datasets), dtype=float) * (group_width + inter_group_gap)
    bar_offset = (width + intra_group_gap) / 2.0

    rc_params = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 12,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(rc_params):
        # Approximately one journal column wide, with a taller plotting area
        # to keep the bars slender while retaining readable type at print size.
        figure, axis = plt.subplots(figsize=(3.45, 3.55))

        digital_bars = axis.bar(
            x - bar_offset,
            digital,
            width,
            label="Digital domain",
            color=DIGITAL_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.7,
            zorder=3,
        )
        static_bars = axis.bar(
            x + bar_offset,
            static_slm,
            width,
            label="Static dual-SLM",
            color=STATIC_SLM_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.7,
            zorder=3,
        )

        add_value_labels(axis, digital_bars)
        add_value_labels(axis, static_bars)

        axis.set_ylabel("mAP50")
        axis.set_xticks(x, datasets)
        side_margin = group_width / 2.0 + 0.08
        axis.set_xlim(x[0] - side_margin, x[-1] + side_margin)
        axis.set_ylim(0.0, 1.10)
        axis.set_yticks(np.linspace(0.0, 1.0, 6))
        axis.grid(
            axis="y",
            color=GRID_COLOR,
            linestyle=(0, (2.0, 2.0)),
            linewidth=0.7,
            zorder=0,
        )

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#333333")
        axis.spines["bottom"].set_color("#333333")
        axis.tick_params(direction="out", length=3.0, width=0.8)

        if title:
            axis.set_title(title, pad=10, fontweight="normal")
        if panel_label:
            axis.text(
                -0.17,
                1.075,
                panel_label,
                transform=axis.transAxes,
                fontsize=14,
                fontweight="bold",
                va="top",
                ha="left",
            )

        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=2,
            frameon=False,
            handlelength=1.05,
            handleheight=0.85,
            handletextpad=0.45,
            columnspacing=0.8,
            borderaxespad=0.0,
        )

        figure.subplots_adjust(left=0.19, right=0.98, top=0.86, bottom=0.23)
        figure.savefig(
            png_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        figure.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)

    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    png_path, pdf_path = plot_validation_performance(
        datasets=args.datasets,
        digital=args.digital,
        static_slm=args.static_slm,
        output_stem=args.output_stem,
        title=args.title,
        panel_label=args.panel_label,
        dpi=args.dpi,
    )
    print(f"Saved PNG: {png_path.resolve()}")
    print(f"Saved PDF: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
