"""Independent still-image ALPR pipeline package."""

from typing import Any


def run_image(*args: Any, **kwargs: Any) -> Any:
    from .pipeline import run_image as implementation

    return implementation(*args, **kwargs)


__all__ = ["run_image"]
