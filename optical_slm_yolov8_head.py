"""Static SLM distillation; retain V2 defaults or select a new CNN teacher."""
import argparse
from models.SLM.physical_defaults import add_optics_argument
from models.SLM import train
from models.SLM.config_slm import ConfigSLM as Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_optics_argument(parser)
    parser.add_argument("--teacher")
    parser.add_argument("--output")
    parser.add_argument("--data", help="YOLO dataset YAML used by student training")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-layers", type=int, choices=(1, 2, 3, 4),
                        help="Use the first N SLM layers in the optics YAML; must match the teacher.")
    parser.add_argument(
        "--init-mode",
        choices=("random", "teacher-static", "checkpoint"),
        help="SLM initialization; teacher-static imports V4's shared phase base",
    )
    parser.add_argument("--init-pair", help="paired SLM checkpoint for --init-mode checkpoint")
    args = parser.parse_args()
    if args.num_layers is not None:
        Config.configure_num_layers(args.num_layers)
    if args.teacher:
        if not args.output:
            parser.error("--teacher requires a separate --output directory")
        Config.TEACHER_DETECTOR_CHECKPOINT = args.teacher
    if args.output:
        Config.OUTPUT_DIR = args.output
    if args.data:
        Config.YAML_PATH = args.data
    if args.batch_size:
        if args.batch_size <= 0:
            parser.error("--batch-size must be positive")
        Config.BATCH_SIZE = args.batch_size
    if args.seed is not None:
        Config.TRAIN_SEED = args.seed
    if args.init_mode:
        Config.SLM_INIT_MODE = args.init_mode.replace("-", "_")
    if args.init_pair:
        Config.SLM_INIT_CHECKPOINT = args.init_pair
        if not args.init_mode:
            Config.SLM_INIT_MODE = "checkpoint"
    if Config.SLM_INIT_MODE == "checkpoint" and not Config.SLM_INIT_CHECKPOINT:
        parser.error("--init-mode checkpoint requires --init-pair")
    train()
