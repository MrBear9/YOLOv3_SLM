"""Create paper-ready phase colormaps and spatial profile curves from SLM PNGs."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


TWO_PI = 2.0 * math.pi


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize SLM gray-drive PNGs produced by src/export_slm_phase.py."
    )
    parser.add_argument("--input", type=Path, required=True, help="PNG file or directory produced by export_slm_phase.py.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for paper figures only.")
    parser.add_argument("--lut", type=Path, default=None, help="Optional gray-to-phase LUT used during export.")
    parser.add_argument("--gray-inverted", action="store_true", help="Force inverted SLM gray drive.")
    parser.add_argument(
        "--profile-index", type=int, default=None,
        help="Row/column index for the spatial phase profiles (default: image center).",
    )
    return parser.parse_args()


def find_phase_pngs(source):
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Input path not found: {source}")
    paths = sorted(path for path in source.glob("*.png") if "phase" in path.stem.lower() or "slm" in path.stem.lower())
    if not paths:
        raise FileNotFoundError(f"No phase/SLM PNG files found in: {source}")
    return paths


def load_export_metadata(source):
    metadata_path = source / "phase_export_metadata.json" if source.is_dir() else source.parent / "phase_export_metadata.json"
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_lut(path):
    if path is None:
        return None, None
    if not path.is_file():
        raise FileNotFoundError(f"LUT not found: {path}")
    values = np.asarray(np.load(path), dtype=np.float32) if path.suffix.lower() == ".npy" else np.asarray(
        np.loadtxt(path, delimiter=","), dtype=np.float32
    )
    if values.ndim == 1:
        gray = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
        phase = values
    elif values.ndim == 2 and values.shape[1] == 2:
        gray, phase = values[:, 0], values[:, 1]
        gray = gray / (255.0 if gray.max() > 1.0 else 1.0)
    else:
        raise ValueError("--lut must contain phase values or (gray, phase) pairs.")
    order = np.argsort(gray)
    gray, phase = gray[order], phase[order]
    if gray.size < 2 or np.any(np.diff(gray) <= 0):
        raise ValueError("The LUT gray values must be strictly increasing.")
    return gray, phase


def gray_to_phase(gray, lut_gray, lut_phase, inverted):
    if inverted:
        gray = 1.0 - gray
    if lut_gray is None:
        return gray * TWO_PI
    return np.interp(gray, lut_gray, lut_phase).astype(np.float32)


def format_stats(phase):
    phase_range = float(phase.max() - phase.min())
    phase_std = float(phase.std())
    return f"range = {phase_range:.3f} rad, std = {phase_std:.3f} rad"


def save_colormap(phase, name, output_dir):
    fig, axis = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    image = axis.imshow(phase, cmap="viridis", vmin=0.0, vmax=TWO_PI, interpolation="nearest")
    axis.set_title(f"{name} phase map\n{format_stats(phase)}", fontweight="bold")
    axis.set_xlabel("x (pixel)")
    axis.set_ylabel("y (pixel)")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("phase (rad)")
    colorbar.set_ticks([0.0, math.pi, TWO_PI])
    colorbar.set_ticklabels(["0", "pi", "2pi"])
    path = output_dir / f"{name}_phase_colormap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_spatial_profiles(phase, name, output_dir, profile_index):
    height, width = phase.shape
    row = height // 2 if profile_index is None else profile_index
    col = width // 2 if profile_index is None else profile_index
    if not 0 <= row < height or not 0 <= col < width:
        raise ValueError(f"--profile-index must be in [0, {min(height, width) - 1}].")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "mathtext.fontset": "stix",
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0), constrained_layout=True, sharey=True)

    axes[0].plot(np.arange(width), phase[row, :], color="#2166ac", linewidth=0.5)
    axes[0].axhline(y=phase[row, col], color="gray", linestyle="--", linewidth=0.4, alpha=0.5)
    axes[0].set_xlabel("x (pixels)")
    axes[0].set_ylabel("Phase (rad)")
    axes[0].text(-0.20, 1.04, "(a)", transform=axes[0].transAxes, fontweight="bold", fontsize=11)

    axes[1].plot(np.arange(height), phase[:, col], color="#b2182b", linewidth=0.5)
    axes[1].axhline(y=phase[row, col], color="gray", linestyle="--", linewidth=0.4, alpha=0.5)
    axes[1].set_xlabel("y (pixels)")
    axes[1].text(-0.20, 1.04, "(b)", transform=axes[1].transAxes, fontweight="bold", fontsize=11)

    for axis in axes:
        axis.set_ylim(0.0, TWO_PI)
        axis.set_yticks([0.0, math.pi, TWO_PI])
        axis.set_yticklabels(["0", "$\\pi$", "$2\\pi$"])
        axis.tick_params(direction="in", top=True, right=True)
        axis.grid(alpha=0.3, linewidth=0.5, linestyle="--")

    path = output_dir / f"{name}_phase_profiles.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = parse_args()
    metadata = load_export_metadata(args.input)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lut_path = args.lut
    if lut_path is None and metadata.get("gray_to_phase_lut"):
        candidate = Path(metadata["gray_to_phase_lut"])
        if candidate.is_file():
            lut_path = candidate
    lut_gray, lut_phase = load_lut(lut_path)
    gray_inverted = bool(args.gray_inverted or metadata.get("gray_inverted", False))
    for path in find_phase_pngs(args.input):
        gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        phase = gray_to_phase(gray, lut_gray, lut_phase, gray_inverted)
        name = path.stem
        colormap_path = save_colormap(phase, name, output_dir)
        profile_path = save_spatial_profiles(phase, name, output_dir, args.profile_index)
        print(f"{path.name}: {format_stats(phase)}")
        print(f"  colormap: {colormap_path}")
        print(f"  profiles: {profile_path}")


if __name__ == "__main__":
    main()


"""
*_phase_colormap.png:twilight 循环伪彩图，统一范围 0~2pi,标题标注 phase range 和 std。
*_phase_profiles.png:相位空间横截面与纵截面曲线，默认取中心行与中心列，更符合光学论文展示相位起伏的形式

若 phase_export_metadata.json 存在，新脚本会自动继承导出时的 LUT 和灰度反转设置，确保可视化相位与实际 SLM 加载相位一致。也可用 --profile-index 320 指定剖面位置。
"""