"""CNN physical teacher training (V2 default, V4 optional)."""
import argparse
from models import train
from models.yolov8.config_v8 import ConfigYOLOv8Anchor as Config


def configure(args):
    Config.TEACHER_INIT_MODE = "joint_checkpoint" if args.resume else "scratch"
    if args.resume:
        Config.TEACHER_INIT_CHECKPOINT = args.resume
    Config.TEACHER_ARCH = "physical_teacher_v4" if args.experiment == "dynamic-v4" else "convteacher_v2"
    if args.experiment == "dynamic-v2":
        Config.TEACHER_OUTPUT_DIR = "output/Tv2_scratch_control_2gpu_seed42"
        Config.PHASE1_TEACHER_LR = 4e-4
        Config.PHASE2_TEACHER_LR = 1.5e-4
    if args.output:
        Config.TEACHER_OUTPUT_DIR = args.output
    if args.data:
        Config.YAML_PATH = args.data
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    Config.TRAIN_SEED = args.seed
    Config.SAMPLER_SEED = args.seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("dynamic-v4", "dynamic-v2"), default="dynamic-v2")
    parser.add_argument("--output")
    parser.add_argument("--data")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        help="Resume teacher and detector from a joint teacher_detector_*.pth checkpoint",
    )
    configure(parser.parse_args())
    train()
