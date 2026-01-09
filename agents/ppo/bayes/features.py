from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .dsl import Action, ActionKind, GridCursorState


@dataclass(frozen=True, slots=True)
class TransitionFeatures:
    """A small, fixed feature vector for amortized rule proposals."""

    vec: np.ndarray  # (D,) float32


KEY_DIRS: Final[tuple[str, ...]] = ("UP", "DOWN", "LEFT", "RIGHT")


def _one_hot(name: str, choices: tuple[str, ...]) -> np.ndarray:
    v = np.zeros((len(choices),), dtype=np.float32)
    try:
        idx = choices.index(name)
    except ValueError:
        return v
    v[idx] = 1.0
    return v


def transition_features(
    prev: GridCursorState,
    action: Action,
    nxt: GridCursorState,
    *,
    near_radius: int = 2,
) -> TransitionFeatures:
    """Extract small numeric features from a single transition."""
    if prev.grid.shape != nxt.grid.shape:
        raise ValueError("prev/nxt grid shapes differ")

    diff = prev.grid != nxt.grid
    delta_frac = float(np.mean(diff)) if diff.size else 0.0

    cy, cx = int(prev.cursor_y), int(prev.cursor_x)
    h, w = prev.grid.shape
    r = int(max(0, near_radius))
    y0 = max(0, cy - r)
    y1 = min(h, cy + r + 1)
    x0 = max(0, cx - r)
    x1 = min(w, cx + r + 1)
    local = diff[y0:y1, x0:x1]
    delta_near = float(np.mean(local)) if local.size else 0.0

    kind_oh = _one_hot(str(action.kind.value), tuple(k.value for k in ActionKind))
    key_oh = np.zeros((len(KEY_DIRS),), dtype=np.float32)
    if action.kind == ActionKind.KEY and action.key:
        key_oh = _one_hot(action.key.upper(), KEY_DIRS)

    vec = np.concatenate(
        [
            np.array([delta_frac, delta_near], dtype=np.float32),
            kind_oh.astype(np.float32),
            key_oh.astype(np.float32),
        ],
        axis=0,
    )
    return TransitionFeatures(vec=vec.astype(np.float32))


def feature_dim() -> int:
    """Dimension of `transition_features().vec`."""
    return 2 + len(ActionKind) + len(KEY_DIRS)



