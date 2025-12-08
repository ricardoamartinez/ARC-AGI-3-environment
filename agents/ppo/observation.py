import hashlib
import numpy as np
from typing import Optional, List, Dict, Tuple
from gymnasium import spaces

class ObservationBuilder:
    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        self._xv, self._yv = np.meshgrid(np.linspace(0, 6.28, grid_size), np.linspace(0, 6.28, grid_size))
        self._cached_goal_channel: Optional[np.ndarray] = None
        
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(grid_size, grid_size, 6), dtype=np.uint8
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
              last_action_idx: int = -1) -> np.ndarray:
        
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
        
        # Highlight objects
        for obj in detected_objects:
             # Calculate invariant hash for checking value
             if not obj: continue
             r0, c0 = obj[0]
             color = current[r0, c0]
             
             min_r = min(p[0] for p in obj)
             min_c = min(p[1] for p in obj)
             norm_pixels = sorted([(p[0]-min_r, p[1]-min_c) for p in obj])
             h_inv = hashlib.md5(f"{color}_{str(norm_pixels)}".encode()).hexdigest()
             
             is_valuable = valuable_hashes.get(h_inv, 0.0) > 1.0
             base_intensity = 150 if is_valuable else 50
             
             for r, c in obj:
                 if r < self.grid_size and c < self.grid_size:
                     if focus_channel[r, c] < base_intensity:
                         focus_channel[r, c] = base_intensity

             # Cheat Sheet: Highlight locked plan target
             if locked_plan:
                 _, target_hash = locked_plan
                 # Check strict or invariant
                 sorted_pixels = sorted(obj)
                 h_strict = hashlib.md5(f"{color}_{str(sorted_pixels)}".encode()).hexdigest()
                 
                 if h_strict == target_hash or h_inv == target_hash:
                     for r, c in obj:
                         focus_channel[r, c] = 255
        
        # Cursor
        cx = int(max(0, min(self.grid_size - 1, cursor_x)))
        cy = int(max(0, min(self.grid_size - 1, cursor_y)))
        focus_channel[cy, cx] = 255
        
        # --- RIPPLE EFFECT (Click Feedback) ---
        if last_action_idx != -1 and last_action_idx <= 3:
             # Draw Ripple for Click
             # 3px radius circle
             for r in range(cy - 3, cy + 4):
                 for c in range(cx - 3, cx + 4):
                     if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                         dist = np.sqrt((r-cy)**2 + (c-cx)**2)
                         if 2.0 <= dist <= 3.5: # Ring
                             focus_channel[r, c] = 255
                             
        # --- KEYBOARD FEEDBACK (Action Feedback) ---
        if last_action_idx >= 4:
            # Draw "Action Burst" (Cross pattern)
             for i in range(1, 4):
                 # Up/Down
                 if 0 <= cy - i < self.grid_size: focus_channel[cy - i, cx] = 255
                 if 0 <= cy + i < self.grid_size: focus_channel[cy + i, cx] = 255
                 # Left/Right
                 if 0 <= cx - i < self.grid_size: focus_channel[cy, cx - i] = 255
                 if 0 <= cx + i < self.grid_size: focus_channel[cy, cx + i] = 255
        
        # Velocity Trail
        vx, vy = vel_x * 5.0, vel_y * 5.0
        steps = int(max(abs(vx), abs(vy)))
        if steps > 0:
            for i in range(steps):
                px = int(cx - vx * (i/steps))
                py = int(cy - vy * (i/steps))
                if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                    if focus_channel[py, px] < 200:
                        focus_channel[py, px] = 200

        # 4. Goal Channel
        goal_channel = self._cached_goal_channel if self._cached_goal_channel is not None else np.zeros_like(current)
        
        # 5/6. Velocity Maps
        vx_norm = np.clip((vel_x + 10.0) / 20.0 * 255.0, 0, 255).astype(np.uint8)
        vy_norm = np.clip((vel_y + 10.0) / 20.0 * 255.0, 0, 255).astype(np.uint8)
        vel_x_channel = np.full((self.grid_size, self.grid_size), vx_norm, dtype=np.uint8)
        vel_y_channel = np.full((self.grid_size, self.grid_size), vy_norm, dtype=np.uint8)
        
        return np.stack([current, delta, focus_channel, goal_channel, vel_x_channel, vel_y_channel], axis=-1)

