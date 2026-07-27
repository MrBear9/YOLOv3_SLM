"""models.SLM - SLM optical student training modules."""

__version__ = "0.1.0"


def train(*args, **kwargs):
    """Load the training loop only when training is explicitly requested.

    Physical teacher V2 imports propagation primitives from this package, while
    the training loop imports the teacher factory.  Delaying this import avoids
    that otherwise unavoidable circular dependency.
    """
    from .slm_train_loop import train as _train

    return _train(*args, **kwargs)


__all__ = ["train"]
