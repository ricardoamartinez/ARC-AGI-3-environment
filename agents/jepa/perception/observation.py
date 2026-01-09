from __future__ import annotations

from typing import Optional

import numpy as np
from gymnasium import spaces


class ObservationBuilder:
    """
    Minimal observation for online RL with human feedback.

    Reward is provided externally (manual dopamine/pain). We keep the observation
    simple and stable (always 10 channels) to avoid mixing in hand-crafted heuristics.

    Channels (uint8):
    - 0: current grid
    - 1: delta grid (0 or 255)
    - 2: cursor marker (0 or 255)
    - 3: goal marker (0 or 255)
    - 4: cursor_x (broadcast scalar 0..255)
    - 5: cursor_y (broadcast scalar 0..255)
    - 6: goal_x (broadcast scalar 0..255; 0 if no goal)
    - 7: goal_y (broadcast scalar 0..255; 0 if no goal)
    - 8: manual pain (scalar broadcast to grid, 0..255)
    - 9: manual dopamine (scalar broadcast to grid, 0..255)
    """

    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(grid_size, grid_size, 10), dtype=np.uint8
        )

    def build(
        self,
        current_grid: np.ndarray,
        last_grid: Optional[np.ndarray],
        cursor_x: float,
        cursor_y: float,
        manual_pain: float,
        manual_dopamine: float,
        goal: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        current = current_grid

        if last_grid is not None:
            delta = np.abs(current.astype(np.int16) - last_grid.astype(np.int16)).astype(np.uint8)
            delta[delta > 0] = 255
        else:
            delta = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        cursor = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        cx = int(max(0, min(self.grid_size - 1, cursor_x)))
        cy = int(max(0, min(self.grid_size - 1, cursor_y)))
        cursor[cy, cx] = 255

        goal_channel = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        goal_x_val = 0
        goal_y_val = 0
        if goal is not None:
            gx, gy = goal
            gx = int(max(0, min(self.grid_size - 1, gx)))
            gy = int(max(0, min(self.grid_size - 1, gy)))
            goal_channel[gy, gx] = 255
            goal_x_val = int(round((gx / max(1, self.grid_size - 1)) * 255.0))
            goal_y_val = int(round((gy / max(1, self.grid_size - 1)) * 255.0))

        pain_val = int(max(0, min(255, float(manual_pain) * 255.0)))
        dopamine_val = int(max(0, min(255, float(manual_dopamine) * 255.0)))
        pain_channel = np.full((self.grid_size, self.grid_size), pain_val, dtype=np.uint8)
        dopamine_channel = np.full((self.grid_size, self.grid_size), dopamine_val, dtype=np.uint8)

        cursor_x_val = int(round((cx / max(1, self.grid_size - 1)) * 255.0))
        cursor_y_val = int(round((cy / max(1, self.grid_size - 1)) * 255.0))

        cursor_x_channel = np.full((self.grid_size, self.grid_size), cursor_x_val, dtype=np.uint8)
        cursor_y_channel = np.full((self.grid_size, self.grid_size), cursor_y_val, dtype=np.uint8)
        goal_x_channel = np.full((self.grid_size, self.grid_size), goal_x_val, dtype=np.uint8)
        goal_y_channel = np.full((self.grid_size, self.grid_size), goal_y_val, dtype=np.uint8)

        return np.stack(
            [
                current,  # 0
                delta,  # 1
                cursor,  # 2
                goal_channel,  # 3
                cursor_x_channel,  # 4
                cursor_y_channel,  # 5
                goal_x_channel,  # 6
                goal_y_channel,  # 7
                pain_channel,  # 8
                dopamine_channel,  # 9
            ],
            axis=-1,
        )


