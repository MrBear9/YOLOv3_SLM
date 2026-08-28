"""Display-only class-name helpers.

Dataset class names remain untouched for checkpoints, reports, and metric
computation.  Human-facing figures and TensorBoard tags use the helpers here
to remove dataset-specific prefixes that otherwise make labels unnecessarily
wide.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


DISPLAY_PREFIXES = ("military_",)


def display_class_name(name: Any) -> str:
    """Return the compact human-facing form of a raw dataset class name."""
    text = str(name)
    folded = text.casefold()
    for prefix in DISPLAY_PREFIXES:
        if folded.startswith(prefix.casefold()) and len(text) > len(prefix):
            return text[len(prefix):]
    return text


def class_name_for_id(
    class_names: Mapping[int, Any] | Sequence[Any] | None,
    class_id: int,
    fallback: str | None = None,
) -> str:
    """Resolve and compact one class name from either a mapping or sequence."""
    fallback = fallback or f"class_{class_id}"
    if class_names is None:
        return fallback
    if isinstance(class_names, Mapping):
        raw_name = class_names.get(class_id, fallback)
    else:
        try:
            raw_name = class_names[class_id]
        except (IndexError, KeyError, TypeError):
            raw_name = fallback
    return display_class_name(raw_name)


def display_class_name_map(
    class_names: Mapping[int, Any] | Sequence[Any] | None,
    num_classes: int,
) -> dict[int, str]:
    """Build a compact display mapping without mutating ``class_names``."""
    return {
        class_id: class_name_for_id(class_names, class_id)
        for class_id in range(num_classes)
    }
