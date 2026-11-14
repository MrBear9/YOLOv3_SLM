"""Plot layer count versus validation mAP50 and operating-point precision.

Run from any directory (requires matplotlib and numpy)::

    python src/plot_layer_performance.py
    python src/plot_layer_performance.py --series dynamic
    python src/plot_layer_performance.py --series both --no-placeholder
    python src/plot_layer_performance.py --fourth-map 0.50 --fourth-precision 0.78
    python src/plot_layer_performance.py --from-logs

Default: edit PLOT_DATA below, then run to plot those values directly.
The fourth layer is an explicitly labelled arbitrary layout placeholder, NOT
an estimate or experiment. Override paths with --runs DIR1 DIR2 DIR3 [DIR4]
(single series only). Supplying four runs replaces the placeholder with data.
Precision is TP/(TP+FP), not classification accuracy. With --from-logs,
both values are read from the SAME best-mAP50 epoch.
Logs are rounded; do not interpret these as full-precision offline COCO AP.
No error bars are generated without independent repeated experiments.
"""

from __future__ import annotations

# ======================== 绘图数据修改区 ========================
# 默认直接使用这里的数据，不会被日志覆盖。数值用 0~1 小数：0.588 = 58.8%。
# 每行依次为：(SLM层数, mAP50, Precision精确率, 是否为占位数据)。
# 第4层得到真实结果后，修改数值并将 True 改为 False，即恢复实线/实心点。
# static = 共享相位模型；dynamic = 动态相位模型。
# 初始1~3层数据来自各模型训练日志中最佳mAP50轮次，精确率取同一轮。
PLOT_DATA = {
    "static": [
        (1, 0.588, 0.802, False),
        (2, 0.540, 0.767, False),
        (3, 0.474, 0.763, False),
        (4, 0.500, 0.780, True),
    ],
    "dynamic": [
        (1, 0.815, 0.872, False),
        (2, 0.821, 0.892, False),
        (3, 0.809, 0.899, False),
        (4, 0.820, 0.900, True),
    ],
}
# Precision对应的置信度阈值；更改评估设置后请同步修改。
PRECISION_THRESHOLDS = {"static": 0.50, "dynamic": 0.35}
DATASET_LABEL = "Military validation"
# ==============================================================

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_SUFFIXES = ("1slm", "1to1", "3slm")
COLORS = ("#299D8F", "#8074AF")


def read_best(run: Path, layers: int) -> dict:
    """Read one unambiguous training log; never mix separate training runs."""
    files = [run] if run.is_file() else sorted((run / "logs").glob("training_log_*.txt"))
    if len(files) != 1:
        raise ValueError(f"Expected one log in {run}; found {len(files)}. Pass a log path explicitly.")
    log = files[0]
    rows = []
    for line_no, line in enumerate(log.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.split()
        if len(fields) < 9 or not fields[0].isdigit():
            continue
        try:
            map50 = float(fields[7])
        except ValueError:  # epochs with no validation
            continue
        operating = re.search(r"Prec@([0-9.]+)=([0-9.]+)", line)
        rows.append(dict(layers=layers, epoch=int(fields[0]), map50=map50,
                         precision=float(operating[2]) if operating else None,
                         confidence=float(operating[1]) if operating else None,
                         placeholder=False, source=str(log.resolve()), line=line_no))
    if not rows:
        raise ValueError(f"No validation rows in {log}")
    # max returns the first epoch on a rounded tie; preserve this rule in metadata.
    best = max(rows, key=lambda r: r["map50"])
    if best["precision"] is None:
        raise ValueError(f"Best epoch has no Prec@ threshold in {log}; no accuracy is invented.")
    return best


def unit_interval(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise argparse.ArgumentTypeError("Use a finite fraction in [0, 1], not a percentage.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--series", choices=("static", "dynamic", "both"), default="static")
    parser.add_argument("--from-logs", action="store_true", help="Read logs instead of PLOT_DATA")
    parser.add_argument("--runs", nargs="+", type=Path, help="Three/four ordered run directories or log files")
    parser.add_argument("--fourth-map", type=unit_interval, help="Arbitrary placeholder, fraction; static/dynamic defaults: 0.50/0.82")
    parser.add_argument("--fourth-precision", type=unit_interval, help="Arbitrary placeholder, fraction; static/dynamic defaults: 0.78/0.90")
    parser.add_argument("--dataset-label", default=DATASET_LABEL, help="Figure context; change when supplying another dataset")
    parser.add_argument("--no-placeholder", action="store_true", help="Only plot observed runs")
    parser.add_argument("--output-stem", type=Path, help="Output path without suffix")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--transparent", action="store_true")
    args = parser.parse_args()
    if args.runs and (args.series == "both" or len(args.runs) not in (3, 4)):
        parser.error("--runs requires one series and exactly three or four paths")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    groups = ("static", "dynamic") if args.series == "both" else (args.series,)
    records = {}
    from_logs = args.from_logs or bool(args.runs)
    for group in groups:
        if from_logs:
            prefix = "SLM_Tv2" if group == "static" else "Tv2"
            runs = args.runs or [ROOT / "output" / f"{prefix}_physical_{s}_2gpu" for s in RUN_SUFFIXES]
            rows = [read_best(run, i) for i, run in enumerate(runs, 1)]
            if len(rows) == 3 and not args.no_placeholder:
                fourth = next((r for r in PLOT_DATA[group] if r[0] == 4), None)
                if fourth is not None:
                    rows.append(dict(layers=4, epoch=None, map50=fourth[1], precision=fourth[2],
                                     confidence=rows[0]["confidence"], placeholder=True,
                                     source="Layout placeholder from PLOT_DATA; no fourth log supplied"))
        else:
            rows = [dict(layers=n, map50=m, precision=p, placeholder=placeholder,
                         confidence=PRECISION_THRESHOLDS[group], epoch=None,
                         source=f"{Path(__file__).resolve()} :: PLOT_DATA[{group!r}]")
                    for n, m, p, placeholder in PLOT_DATA[group]]
        if args.no_placeholder:
            rows = [r for r in rows if not r["placeholder"]]
        if not rows or not any(not r["placeholder"] for r in rows):
            raise ValueError("At least one non-placeholder row is required")
        layers = [r["layers"] for r in rows]
        if layers != sorted(set(layers)) or any(not isinstance(n, int) or n < 1 for n in layers):
            raise ValueError("Layer counts must be unique positive integers in ascending order")
        if any(r["placeholder"] for r in rows[:-1]):
            raise ValueError("Only the final row may be a placeholder")
        for row in rows:
            if row["layers"] == 4 and (args.fourth_map is not None or args.fourth_precision is not None):
                row.update(placeholder=True, source="Command-line layout placeholder")
                if args.fourth_map is not None:
                    row["map50"] = args.fourth_map
                if args.fourth_precision is not None:
                    row["precision"] = args.fourth_precision
        if args.no_placeholder:
            rows = [r for r in rows if not r["placeholder"]]
        if len({r["confidence"] for r in rows}) != 1:
            raise ValueError("Precision thresholds differ across layer counts; re-evaluate at a common threshold.")
        for row in rows:
            if not all(np.isfinite(row[k]) and 0 <= row[k] <= 1 for k in ("map50", "precision")):
                raise ValueError(f"Invalid metric in {row['source']}")
        records[group] = rows

    style = {"font.family": "Arial", "font.size": 12, "axes.labelsize": 14,
             "axes.labelweight": "bold", "axes.linewidth": 1.2,
             "xtick.labelsize": 12, "ytick.labelsize": 12, "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, len(groups), figsize=(4.7 * len(groups), 4.6), squeeze=False)
        fig.subplots_adjust(left=0.16 if len(groups) == 1 else 0.08, right=0.97,
                            top=0.88, bottom=0.23, wspace=0.35)
        for ax, group in zip(axes.flat, groups):
            rows = records[group]
            observed = [r for r in rows if not r["placeholder"]]
            threshold = rows[0]["confidence"]
            for key, label, color, marker, offset in zip(
                ("map50", "precision"), (r"mAP$_{50}$", f"Precision (conf. ≥ {threshold:g})"),
                COLORS, ("o", "s"), (-16, 8)
            ):
                x = [r["layers"] for r in observed]
                y = [100 * r[key] for r in observed]
                ax.plot(x, y, color=color, marker=marker, ms=6.5, lw=2, label=label)
                if rows[-1]["placeholder"]:
                    end = rows[-1]
                    ax.plot([x[-1], end["layers"]], [y[-1], 100 * end[key]], color=color, lw=1.6, ls="--")
                    ax.plot(end["layers"], 100 * end[key], marker=marker, ms=7, mfc="none", mec=color, mew=1.6)
                for row in rows:
                    ax.annotate(f"{100 * row[key]:.1f}" + ("*" if row["placeholder"] else ""),
                                (row["layers"], 100 * row[key]), xytext=(0, offset),
                                textcoords="offset points", ha="center", color=color, fontsize=10)
            ax.set(xlim=(rows[0]["layers"] - 0.3, rows[-1]["layers"] + 0.3), ylim=(0, 105),
                   xticks=[r["layers"] for r in rows], yticks=np.arange(0, 101, 20),
                   xlabel="Number of SLM layers", ylabel="Validation score (%)")
            ax.set_title("Shared-phase model" if group == "static" else "Dynamic phase model", fontsize=13, pad=12)
            ax.tick_params(direction="in", length=4, width=1.1, top=True, right=True)
            ax.legend(loc="lower left", frameon=False, fontsize=10.5, handlelength=2.5)
        note = args.dataset_label
        if from_logs:
            note += " · paired metrics at best mAP50 epoch"
        if any(r["placeholder"] for rows in records.values() for r in rows):
            note += "\n* Open markers: arbitrary placeholders (not measured)"
        fig.text(0.5, 0.035, note, ha="center", va="bottom", fontsize=9, color="#4A4A4A")
        stem = args.output_stem or ROOT / "output" / "figures" / f"layer_performance_{args.series}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            path = stem.with_suffix("." + ext)
            fig.savefig(path, dpi=args.dpi, transparent=args.transparent)
            print(path)
        plt.close(fig)
    metadata = {"selection": "First maximum printed mAP50 epoch; paired operating precision" if from_logs else "Editable PLOT_DATA in script",
                "precision_definition": "TP / (TP + FP), not classification accuracy",
                "uncertainty": "No repeated-run uncertainty available; no error bars",
                "source_precision": "Rounded training-log metrics" if from_logs else "User-editable values; initial observed values transcribed from rounded logs",
                "series": records}
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    for group, rows in records.items():
        print(group, [(r["layers"], r["map50"], r["precision"], r["epoch"]) for r in rows])


if __name__ == "__main__":
    main()
