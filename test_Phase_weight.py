import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


TWO_PI = 2 * np.pi


def torch_load_safe(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint, "direct"
    for key in ("student_state_dict", "model_state_dict", "state_dict", "teacher_state_dict", "detector_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value, key
    return checkpoint, "top_level"


def tensor_to_2d(tensor):
    array = tensor.detach().cpu().float().numpy()
    if array.ndim == 4:
        return array[0, 0] if array.shape[0] > 0 and array.shape[1] > 0 else array.squeeze()
    if array.ndim == 3:
        return array[0]
    if array.ndim == 2:
        return array
    if array.ndim == 1:
        size = int(np.sqrt(array.shape[0]))
        if size * size == array.shape[0]:
            return array.reshape(size, size)
    return None


def wrap_phase(raw_phase):
    return np.mod(raw_phase, TWO_PI).astype(np.float32)


def center_phase(wrapped_phase):
    return np.arctan2(np.sin(wrapped_phase), np.cos(wrapped_phase)).astype(np.float32)


def phase_to_uint8(wrapped_phase):
    normalized = np.clip(wrapped_phase, 0.0, TWO_PI) / TWO_PI
    return (normalized * 255).astype(np.uint8)


def circular_stats(wrapped_phase):
    vector = np.exp(1j * wrapped_phase.reshape(-1))
    resultant = float(np.abs(vector.mean()))
    circular_std = float(np.sqrt(max(-2.0 * np.log(max(resultant, 1e-12)), 0.0)))
    near_boundary = float(((wrapped_phase < 0.05) | (wrapped_phase > TWO_PI - 0.05)).mean())
    return resultant, circular_std, near_boundary


def safe_name(name):
    return name.replace(".", "_").replace(":", "_").replace("/", "_").replace("\\", "_")


def find_phase_layers(state_dict):
    phase_layers = {}
    keywords = ("phase_raw", "phase", "slm")
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        key_lower = key.lower()
        if any(keyword in key_lower for keyword in keywords):
            if value.ndim in (2, 3, 4) or (value.ndim == 1 and int(np.sqrt(value.numel())) ** 2 == value.numel()):
                phase_layers[key] = value
    if phase_layers:
        return phase_layers

    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and value.ndim == 4 and value.shape[0] == 1 and value.shape[1] == 1:
            phase_layers[key] = value
    return phase_layers


def save_histogram(raw_phase, wrapped_phase, centered_phase, layer_name, output_dir):
    path = output_dir / f"{safe_name(layer_name)}_histogram.png"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    items = [
        ("raw phase_raw", raw_phase, None),
        ("wrapped [0, 2pi)", wrapped_phase, (0.0, TWO_PI)),
        ("centered [-pi, pi)", centered_phase, (-np.pi, np.pi)),
    ]
    for ax, (title, values, value_range) in zip(axes, items):
        flat = values.reshape(-1)
        ax.hist(flat, bins=120, range=value_range, color="#4C78A8", alpha=0.85)
        ax.axvline(float(flat.mean()), color="#E45756", linewidth=1.4, label=f"mean={flat.mean():.4f}")
        ax.axvline(float(flat.mean() + flat.std()), color="#F58518", linewidth=1.0, linestyle="--", label=f"std={flat.std():.4f}")
        ax.axvline(float(flat.mean() - flat.std()), color="#F58518", linewidth=1.0, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("phase (rad)")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.suptitle(f"Phase distribution: {layer_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_phase_preview(wrapped_phase, layer_name, output_dir):
    image_path = output_dir / f"{safe_name(layer_name)}_wrapped_slm_0_2pi.png"
    Image.fromarray(phase_to_uint8(wrapped_phase), mode="L").save(image_path)
    return image_path


def write_layer_info(path, layer_name, original_shape, raw_phase, wrapped_phase, centered_phase, files):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Layer: {layer_name}\n")
        f.write(f"Original shape: {tuple(original_shape)}\n")
        f.write(f"2D shape: {raw_phase.shape}\n\n")
        f.write("Important note:\n")
        f.write("  phase_raw is an unconstrained trainable parameter and may contain negative values.\n")
        f.write("  The SLM-loadable phase is wrapped_phase = phase_raw mod (2*pi), in [0, 2*pi).\n\n")
        for label, data in (("raw_phase", raw_phase), ("wrapped_phase", wrapped_phase), ("centered_phase", centered_phase)):
            f.write(f"{label} stats:\n")
            f.write(f"  min: {float(data.min()):.8f}\n")
            f.write(f"  max: {float(data.max()):.8f}\n")
            f.write(f"  mean: {float(data.mean()):.8f}\n")
            f.write(f"  std: {float(data.std()):.8f}\n")
            f.write(f"  range: {float(data.max() - data.min()):.8f}\n\n")
        f.write("Files:\n")
        for label, file_path in files.items():
            f.write(f"  {label}: {file_path}\n")
        f.write("\nTop-left 10x10 wrapped phase values:\n")
        rows, cols = min(10, wrapped_phase.shape[0]), min(10, wrapped_phase.shape[1])
        for row in range(rows):
            f.write("  " + "  ".join(f"{wrapped_phase[row, col]:.6f}" for col in range(cols)) + "\n")


def save_combined_plot(saved_layers, output_dir):
    if len(saved_layers) < 2:
        return None
    path = output_dir / "combined_wrapped_phase_plot.png"
    layers = saved_layers[:2]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, item in zip(axes, layers):
        data = np.load(item["wrapped_npy_path"])
        im = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=TWO_PI)
        ax.set_title(f"{item['name']}\nstd={item['wrapped_std']:.4f}, range={item['wrapped_range']:.4f}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="phase (rad)")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_phase_layers(pth_file_path, output_dir="output/optical_phase_layers"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch_load_safe(pth_file_path)
    state_dict, source = extract_state_dict(checkpoint)

    print("=" * 70)
    print("Optical SLM phase extraction")
    print("=" * 70)
    print(f"Model file: {pth_file_path}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Checkpoint type: {type(checkpoint)}")
    print(f"State dict source: {source}")
    if isinstance(checkpoint, dict):
        print(f"Top-level keys: {list(checkpoint.keys())}")

    phase_layers = find_phase_layers(state_dict)
    if not phase_layers:
        print("No phase-like tensor layers found.")
        return []

    saved_layers = []
    for layer_name, tensor in phase_layers.items():
        raw_phase = tensor_to_2d(tensor)
        if raw_phase is None:
            print(f"Skip unsupported tensor: {layer_name}, shape={tuple(tensor.shape)}")
            continue
        raw_phase = raw_phase.astype(np.float32)
        wrapped_phase = wrap_phase(raw_phase)
        centered_phase = center_phase(wrapped_phase)
        base = safe_name(layer_name)

        resultant, circ_std, near_boundary = circular_stats(wrapped_phase)
        raw_npy = output_dir / f"{base}_raw_phase_raw.npy"
        wrapped_npy = output_dir / f"{base}_wrapped_slm_0_2pi.npy"
        centered_npy = output_dir / f"{base}_centered_minus_pi_pi.npy"
        np.save(raw_npy, raw_phase)
        np.save(wrapped_npy, wrapped_phase)
        np.save(centered_npy, centered_phase)
        image_path = save_phase_preview(wrapped_phase, layer_name, output_dir)
        hist_path = save_histogram(raw_phase, wrapped_phase, centered_phase, layer_name, output_dir)
        info_path = output_dir / f"{base}_info.txt"
        files = {
            "raw_npy": raw_npy,
            "wrapped_slm_npy": wrapped_npy,
            "centered_npy": centered_npy,
            "wrapped_png": image_path,
            "histogram_png": hist_path,
        }
        write_layer_info(info_path, layer_name, tensor.shape, raw_phase, wrapped_phase, centered_phase, files)

        item = {
            "name": layer_name,
            "shape": raw_phase.shape,
            "raw_min": float(raw_phase.min()),
            "raw_max": float(raw_phase.max()),
            "raw_mean": float(raw_phase.mean()),
            "raw_std": float(raw_phase.std()),
            "wrapped_min": float(wrapped_phase.min()),
            "wrapped_max": float(wrapped_phase.max()),
            "wrapped_mean": float(wrapped_phase.mean()),
            "wrapped_std": float(wrapped_phase.std()),
            "wrapped_range": float(wrapped_phase.max() - wrapped_phase.min()),
            "circular_resultant": resultant,
            "circular_std": circ_std,
            "near_boundary_ratio": near_boundary,
            "wrapped_npy_path": str(wrapped_npy),
            "img_path": str(image_path),
            "hist_path": str(hist_path),
            "info_path": str(info_path),
        }
        saved_layers.append(item)
        print(
            f"Saved {layer_name}: wrapped std={item['wrapped_std']:.6f}, "
            f"circular std={item['circular_std']:.6f}, near-boundary={item['near_boundary_ratio']:.4f}"
        )

    combined_path = save_combined_plot(saved_layers, output_dir)
    summary_path = output_dir / "optical_phase_layers_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Optical SLM phase layers summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Source file: {pth_file_path}\n")
        f.write(f"State dict source: {source}\n")
        f.write(f"Layer count: {len(saved_layers)}\n")
        f.write("Note: use *_wrapped_slm_0_2pi.npy for SLM loading.\n\n")
        for idx, item in enumerate(saved_layers, start=1):
            f.write(f"{idx}. {item['name']}\n")
            f.write(f"   shape: {item['shape']}\n")
            f.write(f"   raw min/max/mean/std: {item['raw_min']:.8f}, {item['raw_max']:.8f}, {item['raw_mean']:.8f}, {item['raw_std']:.8f}\n")
            f.write(f"   wrapped min/max/mean/std/range: {item['wrapped_min']:.8f}, {item['wrapped_max']:.8f}, {item['wrapped_mean']:.8f}, {item['wrapped_std']:.8f}, {item['wrapped_range']:.8f}\n")
            f.write(f"   circular resultant/std/near_0_or_2pi_ratio: {item['circular_resultant']:.8f}, {item['circular_std']:.8f}, {item['near_boundary_ratio']:.8f}\n")
            if item["circular_std"] < 0.1 or item["near_boundary_ratio"] > 0.95:
                f.write("   warning: circular stats indicate this phase map is close to flat modulation.\n")
            f.write(f"   wrapped_npy: {item['wrapped_npy_path']}\n")
            f.write(f"   image: {item['img_path']}\n")
            f.write(f"   histogram: {item['hist_path']}\n")
            f.write(f"   info: {item['info_path']}\n\n")
        if combined_path:
            f.write(f"Combined plot: {combined_path}\n")

    print("=" * 70)
    print(f"Extracted {len(saved_layers)} phase layer(s).")
    print(f"Summary: {summary_path}")
    print("Use *_wrapped_slm_0_2pi.npy for SLM loading.")
    print("=" * 70)
    return saved_layers


def parse_args():
    parser = argparse.ArgumentParser(description="Extract SLM phase layers and histograms.")
    parser.add_argument(
        "--pth-file",
        default="output/OpticalSLM_YOLOv8Head_student/optical_student_best.pth",
        help="Path to optical student checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/OpticalSLM_YOLOv8Head_student/optical_phase_layers",
        help="Directory for extracted phase files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_phase_layers(args.pth_file, args.output_dir)
