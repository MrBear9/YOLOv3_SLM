"""Static SLM distillation; retain V2 defaults or select a new CNN teacher."""
import argparse
from models.SLM import train
from models.SLM.config_slm import ConfigSLM as Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.teacher:
        if not args.output:
            parser.error("--teacher requires a separate --output directory")
        Config.TEACHER_DETECTOR_CHECKPOINT = args.teacher
    if args.output:
        Config.OUTPUT_DIR = args.output
    train()
