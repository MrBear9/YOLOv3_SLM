"""models - Optical teacher detector training framework.

When DETECTOR_HEAD_TYPE is "compact" or "center_detect", the teacher +
CompactOpticalDetector training loop is used (teacher_train_compact.py).
For all other head types the original anchor-based training loop is used
(teacher_train_loop.py).
"""

__version__ = "0.1.0"


def train():
    """Train with the appropriate loop based on DETECTOR_HEAD_TYPE config."""
    from models.yolov8.config_v8 import ConfigYOLOv8Anchor as Config

    head_type = str(getattr(Config, "DETECTOR_HEAD_TYPE", "light")).strip().lower()

    if head_type in ("compact", "center_detect"):
        from .teacher_train_compact import train as _train
    else:
        from .teacher_train_loop import train as _train

    return _train()


__all__ = ["train"]
