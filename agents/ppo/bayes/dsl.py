from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class ActionKind(str, Enum):
    """Abstract action kinds for the Bayes-adaptive simulator/planner."""

    NOOP = "noop"
    SET_CURSOR = "set_cursor"
    CLICK = "click"
    KEY = "key"


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    x: Optional[int] = None
    y: Optional[int] = None
    key: Optional[str] = None  # e.g., "UP", "DOWN", "LEFT", "RIGHT", "SPACE", "ENTER"

    @staticmethod
    def noop() -> "Action":
        return Action(kind=ActionKind.NOOP)

    @staticmethod
    def set_cursor(x: int, y: int) -> "Action":
        return Action(kind=ActionKind.SET_CURSOR, x=int(x), y=int(y))

    @staticmethod
    def click(x: Optional[int] = None, y: Optional[int] = None) -> "Action":
        return Action(kind=ActionKind.CLICK, x=None if x is None else int(x), y=None if y is None else int(y))

    @staticmethod
    def keypress(key: str) -> "Action":
        return Action(kind=ActionKind.KEY, key=str(key).upper())


@dataclass(frozen=True, slots=True)
class GridCursorState:
    """Minimal observable state used by the DSL simulator."""

    grid: np.ndarray  # (H,W) uint8
    cursor_y: int
    cursor_x: int

    @property
    def shape_hw(self) -> tuple[int, int]:
        h, w = self.grid.shape
        return int(h), int(w)



