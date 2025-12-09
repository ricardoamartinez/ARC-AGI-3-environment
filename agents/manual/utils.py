from typing import Tuple

def get_inverse_color(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (255 - rgb[0], 255 - rgb[1], 255 - rgb[2])

def lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t

