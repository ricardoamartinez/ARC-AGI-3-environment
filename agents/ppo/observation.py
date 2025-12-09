import hashlib
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from gymnasium import spaces

class ObservationBuilder:
    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        self._xv, self._yv = np.meshgrid(np.linspace(0, 6.28, grid_size), np.linspace(0, 6.28, grid_size))
        self._cached_goal_channel: Optional[np.ndarray] = None
        
        # 10 Channels
        # 0: Current
        # 1: Delta
        # 2: Focus
        # 3: Goal
        # 4: VelX
        # 5: VelY
        # 6: KB Bias
        # 7: Cursor Bias
        # 8: Pain
        # 9: Dopamine
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(grid_size, grid_size, 10), dtype=np.uint8
        )

    def precompute_goal_channel(self, goal_vector: np.ndarray):
        v = goal_vector
        goal_layer = np.sin(self._xv * v[0]) + np.cos(self._yv * v[1]) + np.sin(self._xv * self._yv * v[2]) + v[3]
        g_min, g_max = goal_layer.min(), goal_layer.max()
        if g_max > g_min:
            goal_layer = (goal_layer - g_min) / (g_max - g_min) * 255.0
        else:
            goal_layer[:] = 128.0
        self._cached_goal_channel = goal_layer.astype(np.uint8)

    def build(self, 
              current_grid: np.ndarray, 
              last_grid: Optional[np.ndarray], 
              focus_map: np.ndarray,
              detected_objects: List[List[Tuple[int, int]]],
              valuable_hashes: Dict[str, float],
              cursor_x: float, 
              cursor_y: float,
              vel_x: float,
              vel_y: float,
              locked_plan: Optional[Tuple[int, str]],
              value_map: Optional[np.ndarray] = None,
              modality_bias: float = 0.0,
              last_action_idx: int = -1,
              pain: Union[float, np.ndarray] = 0.0,
              dopamine: Union[float, np.ndarray] = 0.0) -> np.ndarray:
        
        # 1. State
        current = current_grid

        # 2. Delta
        if last_grid is not None:
             delta = np.abs(current.astype(np.int16) - last_grid.astype(np.int16)).astype(np.uint8)
             delta[delta > 0] = 255
        else:
             delta = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # 3. Focus Channel
        focus_channel = (focus_map * 255.0).astype(np.uint8)
        
        # Cursor Visualization (Crosshair)
        # Making it larger so CNN can track it easily
        cx = int(max(0, min(self.grid_size - 1, cursor_x)))
        cy = int(max(0, min(self.grid_size - 1, cursor_y)))
        
        # Draw 3x3 Crosshair
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                # Cross shape or Box? Box is more robust.
                if 0 <= cy+dy < self.grid_size and 0 <= cx+dx < self.grid_size:
                    focus_channel[cy+dy, cx+dx] = 255
        
        # --- RIPPLE EFFECT (Feedback for Click) ---
        if last_action_idx != -1 and last_action_idx <= 3:
             for r in range(cy - 3, cy + 4):
                 for c in range(cx - 3, cx + 4):
                     if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                         dist = np.sqrt((r-cy)**2 + (c-cx)**2)
                         if 2.0 <= dist <= 3.5: # Ring
                             focus_channel[r, c] = 255
                             
        # --- KEYBOARD FEEDBACK ---
        if last_action_idx >= 4:
             for i in range(1, 4):
                 if 0 <= cy - i < self.grid_size: focus_channel[cy - i, cx] = 255
                 if 0 <= cy + i < self.grid_size: focus_channel[cy + i, cx] = 255
                 if 0 <= cx - i < self.grid_size: focus_channel[cy, cx - i] = 255
                 if 0 <= cx + i < self.grid_size: focus_channel[cy, cx + i] = 255
        
        # Velocity Trail (helps agent see where it's going)
        vx, vy = vel_x * 5.0, vel_y * 5.0
        steps = int(max(abs(vx), abs(vy)))
        if steps > 0:
            for i in range(steps):
                px = int(cx - vx * (i/steps))
                py = int(cy - vy * (i/steps))
                if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                    # Don't overwrite the cursor crosshair
                    if focus_channel[py, px] < 200:
                        focus_channel[py, px] = 150 # Trail is dimmer

        # 4. Value / Goal Channel
        if value_map is not None:
            goal_channel = value_map
        else:
            goal_channel = self._cached_goal_channel if self._cached_goal_channel is not None else np.zeros_like(current)
        
        vx_norm = np.clip((vel_x + 10.0) / 20.0 * 255.0, 0, 255).astype(np.uint8)
        vy_norm = np.clip((vel_y + 10.0) / 20.0 * 255.0, 0, 255).astype(np.uint8)
        
        bias_offset = int(modality_bias * 50.0)
        vx_norm_int = vx_norm.astype(np.int16) + bias_offset
        vy_norm_int = vy_norm.astype(np.int16) + bias_offset
        
        vx_norm = np.clip(vx_norm_int, 0, 255).astype(np.uint8)
        vy_norm = np.clip(vy_norm_int, 0, 255).astype(np.uint8)
        
        vel_x_channel = np.full((self.grid_size, self.grid_size), vx_norm, dtype=np.uint8)
        vel_y_channel = np.full((self.grid_size, self.grid_size), vy_norm, dtype=np.uint8)
        
        kb_bias = max(0.0, -modality_bias)
        cursor_bias = max(0.0, modality_bias)
        
        kb_channel = np.full((self.grid_size, self.grid_size), int(kb_bias * 255), dtype=np.uint8)
        cursor_channel = np.full((self.grid_size, self.grid_size), int(cursor_bias * 255), dtype=np.uint8)
        
        # --- PAIN & DOPAMINE CHANNELS ---
        if isinstance(pain, np.ndarray) and pain.shape == (self.grid_size, self.grid_size):
            pain_channel = (np.clip(pain, 0, 1) * 255).astype(np.uint8)
        else:
            p_val = min(255, int(float(pain) * 25.0)) 
            pain_channel = np.full((self.grid_size, self.grid_size), p_val, dtype=np.uint8)
        
        # Dopamine Map support
        if isinstance(dopamine, np.ndarray) and dopamine.shape == (self.grid_size, self.grid_size):
            dopamine_channel = dopamine.astype(np.uint8)
        else:
            d_val = min(255, int(dopamine * 255.0))
            dopamine_channel = np.full((self.grid_size, self.grid_size), d_val, dtype=np.uint8)

        return np.stack([
            current,        # 0
            delta,          # 1
            focus_channel,  # 2
            goal_channel,   # 3
            vel_x_channel,  # 4
            vel_y_channel,  # 5
            kb_channel,     # 6 
            cursor_channel, # 7
            pain_channel,   # 8 
            dopamine_channel# 9
        ], axis=-1)
