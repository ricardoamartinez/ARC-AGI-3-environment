from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dsl import Action, ActionKind, GridCursorState


def _clamp_cursor(y: int, x: int, shape_hw: tuple[int, int]) -> tuple[int, int]:
    h, w = shape_hw
    cy = int(np.clip(int(y), 0, max(0, h - 1)))
    cx = int(np.clip(int(x), 0, max(0, w - 1)))
    return cy, cx


def apply_cursor_dynamics(state: GridCursorState, action: Action) -> GridCursorState:
    """Update cursor position according to abstract action semantics."""
    h, w = state.shape_hw
    cy, cx = state.cursor_y, state.cursor_x
    if action.kind == ActionKind.SET_CURSOR:
        if action.x is None or action.y is None:
            return state
        cy, cx = _clamp_cursor(action.y, action.x, (h, w))
    elif action.kind == ActionKind.CLICK:
        # Optional: click can optionally carry an explicit target coordinate (treated as instantaneous move).
        if action.x is not None and action.y is not None:
            cy, cx = _clamp_cursor(action.y, action.x, (h, w))
    return GridCursorState(grid=state.grid, cursor_y=cy, cursor_x=cx)


class RuleProgram(ABC):
    """A discrete hypothesis about the game's transition rules."""

    @abstractmethod
    def signature(self) -> str:  # stable string for debugging / logging
        raise NotImplementedError

    @abstractmethod
    def apply(self, state: GridCursorState, action: Action) -> GridCursorState:
        raise NotImplementedError

    def predict_grid(self, state: GridCursorState, action: Action) -> np.ndarray:
        return self.apply(state, action).grid


@dataclass(frozen=True, slots=True)
class IdentityRule(RuleProgram):
    """No grid dynamics; only cursor updates (handled externally)."""

    def signature(self) -> str:
        return "identity"

    def apply(self, state: GridCursorState, action: Action) -> GridCursorState:
        # Cursor update is separated so all rules share the same cursor semantics.
        st = apply_cursor_dynamics(state, action)
        return GridCursorState(grid=st.grid, cursor_y=st.cursor_y, cursor_x=st.cursor_x)


@dataclass(frozen=True, slots=True)
class PaintOnClickRule(RuleProgram):
    """On click: set the clicked cell to a fixed color."""

    color: int

    def signature(self) -> str:
        return f"paint_on_click(color={int(self.color)})"

    def apply(self, state: GridCursorState, action: Action) -> GridCursorState:
        st = apply_cursor_dynamics(state, action)
        if action.kind != ActionKind.CLICK:
            return st
        grid = st.grid.copy()
        cy, cx = _clamp_cursor(st.cursor_y, st.cursor_x, st.shape_hw)
        grid[cy, cx] = np.uint8(int(self.color) & 0xFF)
        return GridCursorState(grid=grid, cursor_y=cy, cursor_x=cx)


@dataclass(frozen=True, slots=True)
class ToggleOnClickRule(RuleProgram):
    """On click: toggle the clicked cell between two colors."""

    a: int
    b: int

    def signature(self) -> str:
        return f"toggle_on_click(a={int(self.a)},b={int(self.b)})"

    def apply(self, state: GridCursorState, action: Action) -> GridCursorState:
        st = apply_cursor_dynamics(state, action)
        if action.kind != ActionKind.CLICK:
            return st
        grid = st.grid.copy()
        cy, cx = _clamp_cursor(st.cursor_y, st.cursor_x, st.shape_hw)
        cur = int(grid[cy, cx])
        if cur == int(self.a):
            grid[cy, cx] = np.uint8(int(self.b) & 0xFF)
        elif cur == int(self.b):
            grid[cy, cx] = np.uint8(int(self.a) & 0xFF)
        else:
            # default behavior: set to b (acts like "turn on")
            grid[cy, cx] = np.uint8(int(self.b) & 0xFF)
        return GridCursorState(grid=grid, cursor_y=cy, cursor_x=cx)


@dataclass(frozen=True, slots=True)
class MoveColorOnKeyRule(RuleProgram):
    """On arrow-key: translate all pixels of a given color by 1 cell."""

    color: int
    background: int = 0

    def signature(self) -> str:
        return f"move_color_on_key(color={int(self.color)},bg={int(self.background)})"

    def _delta(self, key: str) -> Optional[tuple[int, int]]:
        k = key.upper()
        if k == "UP":
            return (-1, 0)
        if k == "DOWN":
            return (1, 0)
        if k == "LEFT":
            return (0, -1)
        if k == "RIGHT":
            return (0, 1)
        return None

    def apply(self, state: GridCursorState, action: Action) -> GridCursorState:
        st = apply_cursor_dynamics(state, action)
        if action.kind != ActionKind.KEY or not action.key:
            return st
        d = self._delta(action.key)
        if d is None:
            return st
        dy, dx = d
        grid = st.grid
        mask = grid == np.uint8(int(self.color) & 0xFF)
        if not bool(mask.any()):
            return st

        h, w = st.shape_hw
        ys, xs = np.where(mask)
        ny = ys + int(dy)
        nx = xs + int(dx)
        keep = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        ny = ny[keep]
        nx = nx[keep]

        out = grid.copy()
        out[mask] = np.uint8(int(self.background) & 0xFF)
        out[ny, nx] = np.uint8(int(self.color) & 0xFF)
        return GridCursorState(grid=out, cursor_y=st.cursor_y, cursor_x=st.cursor_x)



