from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .dsl import GridCursorState
from .objectifier import connected_components


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


@dataclass(frozen=True, slots=True)
class BoundaryMetrics:
    delta_frac: float
    palette_size: int
    non_bg_frac: float
    object_count: int
    reset_like: bool


def compute_boundary_metrics(
    prev: GridCursorState,
    cur: GridCursorState,
    *,
    background: int = 0,
    initial_grid: Optional[np.ndarray] = None,
) -> BoundaryMetrics:
    if prev.grid.shape != cur.grid.shape:
        raise ValueError("prev/cur grid shapes differ")

    diff = prev.grid != cur.grid
    delta_frac = float(np.mean(diff)) if diff.size else 0.0

    palette = set(int(x) for x in np.unique(cur.grid))
    palette_size = int(len(palette))

    non_bg_frac = float(np.mean(cur.grid != np.uint8(int(background) & 0xFF))) if cur.grid.size else 0.0

    objs = connected_components(cur.grid, background_colors={int(background)}, connectivity=4)
    object_count = int(len(objs))

    reset_like = False
    if initial_grid is not None and initial_grid.shape == cur.grid.shape:
        # Reset-like if we're very close to the initial grid.
        reset_like = float(np.mean(initial_grid != cur.grid)) < 0.02

    return BoundaryMetrics(
        delta_frac=delta_frac,
        palette_size=palette_size,
        non_bg_frac=non_bg_frac,
        object_count=object_count,
        reset_like=bool(reset_like),
    )


def heuristic_boundary_probability(
    prev: GridCursorState,
    cur: GridCursorState,
    *,
    initial_grid: Optional[np.ndarray],
    background: int = 0,
    delta_hi: float = 0.25,
    reset_bonus: float = 2.0,
    bias: float = 3.0,
) -> float:
    """Stateless version of the heuristic boundary probability for planning."""
    init = initial_grid if initial_grid is not None else prev.grid
    prev_reset_like = False
    if init.shape == prev.grid.shape:
        prev_reset_like = float(np.mean(init != prev.grid)) < 0.02
    m = compute_boundary_metrics(prev, cur, background=background, initial_grid=init)

    score = 0.0
    score += 6.0 * max(0.0, m.delta_frac - float(delta_hi))
    score += 1.0 * float(abs(m.non_bg_frac - float(np.mean(init != background))) > 0.15)
    score += 0.5 * float(m.palette_size > 8)
    score += 0.5 * float(m.object_count > 25)
    if (not prev_reset_like) and m.reset_like:
        score += float(reset_bonus)
    return float(np.clip(_sigmoid(score - float(bias)), 0.0, 1.0))


class HeuristicEventBoundaryDetector:
    """Visual-only event boundary detector (reward-free)."""

    def __init__(
        self,
        *,
        background: int = 0,
        delta_hi: float = 0.25,
        reset_bonus: float = 2.0,
        bias: float = 3.0,
    ) -> None:
        self.background = int(background)
        self.delta_hi = float(delta_hi)
        self.reset_bonus = float(reset_bonus)
        self.bias = float(bias)

        self._initial_grid: Optional[np.ndarray] = None
        self._prev: Optional[GridCursorState] = None

    def reset(self, initial: GridCursorState) -> None:
        self._initial_grid = initial.grid.copy()
        self._prev = None

    def observe(self, cur: GridCursorState) -> float:
        """Return P(boundary | current observation)."""
        if self._initial_grid is None:
            self._initial_grid = cur.grid.copy()

        if self._prev is None:
            self._prev = cur
            return 0.0

        prev_reset_like = False
        if self._initial_grid is not None and self._initial_grid.shape == self._prev.grid.shape:
            prev_reset_like = float(np.mean(self._initial_grid != self._prev.grid)) < 0.02

        m = compute_boundary_metrics(self._prev, cur, background=self.background, initial_grid=self._initial_grid)
        self._prev = cur

        # Simple bounded scoring: big instantaneous change + structural change + reset detection.
        score = 0.0
        score += 6.0 * max(0.0, m.delta_frac - self.delta_hi)  # only triggers for large deltas
        score += 1.0 * float(abs(m.non_bg_frac - float(np.mean(self._initial_grid != self.background))) > 0.15)
        score += 0.5 * float(m.palette_size > 8)
        score += 0.5 * float(m.object_count > 25)
        # Only count "reset-like" as a boundary if we were not already reset-like.
        if (not prev_reset_like) and m.reset_like:
            score += self.reset_bonus
        # Bias shifts the default probability downward so "no event" ≈ low probability.
        return float(np.clip(_sigmoid(score - self.bias), 0.0, 1.0))


class LearnedEventBoundaryModel(nn.Module):
    """Optional learned event boundary model.

    This is intentionally tiny and can be trained self-supervised (change-point consistency).
    """

    def __init__(self, in_dim: int = 5, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LearnedEventBoundaryDetector:
    """Wraps a `LearnedEventBoundaryModel` with the same observe() interface."""

    def __init__(self, model: LearnedEventBoundaryModel, *, background: int = 0, device: str | torch.device = "cpu"):
        self.model = model
        self.background = int(background)
        self.device = torch.device(device)
        self._initial_grid: Optional[np.ndarray] = None
        self._prev: Optional[GridCursorState] = None

    def reset(self, initial: GridCursorState) -> None:
        self._initial_grid = initial.grid.copy()
        self._prev = None

    def observe(self, cur: GridCursorState) -> float:
        if self._initial_grid is None:
            self._initial_grid = cur.grid.copy()
        if self._prev is None:
            self._prev = cur
            return 0.0

        m = compute_boundary_metrics(self._prev, cur, background=self.background, initial_grid=self._initial_grid)
        self._prev = cur
        x = np.array(
            [
                m.delta_frac,
                float(m.palette_size) / 32.0,
                m.non_bg_frac,
                float(m.object_count) / 64.0,
                1.0 if m.reset_like else 0.0,
            ],
            dtype=np.float32,
        )[None, :]
        xt = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            logit = self.model(xt)
            p = torch.sigmoid(logit).item()
        return float(np.clip(p, 0.0, 1.0))


