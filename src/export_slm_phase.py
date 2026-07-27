"""Export a trained optical student to native-resolution SLM gray-drive PNGs."""

import argparse
from pathlib import Path
import sys

import torch

# Allow ``python src/export_slm_phase.py`` from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.SLM.config_slm import ConfigSLM
from models.SLM.phase_io import export_student_phase_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Optical student checkpoint path.")
    parser.add_argument("--output", required=True, help="Output directory for grayscale PNGs.")
    parser.add_argument("--lut", default=None, help="Optional gray-to-phase LUT (.npy, .csv, or .txt).")
    parser.add_argument("--inverted", action="store_true", help="Use inverted SLM gray drive.")
    args = parser.parse_args()

    ConfigSLM.initialize()
    if args.lut is not None:
        ConfigSLM.SLM_GRAY_TO_PHASE_LUT = str(Path(args.lut).resolve())
    if args.inverted:
        ConfigSLM.SLM_GRAY_INVERTED = True
    paths, info = export_student_phase_images(ConfigSLM, args.checkpoint, args.output, device="cpu")
    print(f"Loaded {info['loaded']}/{info['total']} compatible student tensors.")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
