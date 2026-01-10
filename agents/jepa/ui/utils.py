"""Utility functions for JEPA UI."""
from typing import Tuple


def get_inverse_color(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Get the inverse of an RGB color."""
    return (255 - rgb[0], 255 - rgb[1], 255 - rgb[2])


def lerp(start: float, end: float, t: float) -> float:
    """Linear interpolation between start and end."""
    return start + (end - start) * t
