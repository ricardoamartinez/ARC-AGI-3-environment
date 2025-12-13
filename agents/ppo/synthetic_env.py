import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Any, List

class SyntheticARCGymEnv(gym.Env):
    """
    A synthetic environment that mimics ARCGymEnv for testing PPO agents
    without requiring the ARC backend server.
    
    Supports basic modes based on game_id.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, agent, max_steps: int = 100):
        super().__init__()
        self.agent = agent
        self.max_steps = max_steps
        self.current_step = 0
        self.grid_size = 64
        self.game_id = getattr(agent, "game_id", "game_target_click")
        
        # State
        self.cursor_x = 32.0
        self.cursor_y = 32.0
        self.target_x = 10
        self.target_y = 10
        self.score = 0
        
        # Mimic observation space of ARCGymEnv
        # 10 Channels: Current, Delta, Focus, Goal, VelX, VelY, KB, Cursor, Pain, Dopamine
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.grid_size, self.grid_size, 10), dtype=np.uint8
        )
        
        # Mimic action space: (Continuous[3], Discrete[10])
        self.action_space = spaces.Tuple((
            spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            spaces.Discrete(10)
        ))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.score = 0
        
        # Randomize target
        self.target_x = np.random.randint(5, self.grid_size - 5)
        self.target_y = np.random.randint(5, self.grid_size - 5)
        self.cursor_x = self.grid_size // 2
        self.cursor_y = self.grid_size // 2
        
        return self._get_obs(), {}

    def step(self, action: Tuple[np.ndarray, int]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.current_step += 1
        
        # Parse Action
        cont_actions, disc_idx = action
        ax, ay, trigger = cont_actions
        
        reward = -0.01 # Time cost
        hit = False
        
        # GAME LOGIC
        if "nav" in self.game_id or "maze" in self.game_id:
            # KEYBOARD ONLY MODE
            # Ignore mouse velocity
            ax, ay = 0, 0
            
            # Map discrete actions to movement
            speed = 2.0
            if disc_idx == 4: self.cursor_y -= speed # UP
            elif disc_idx == 5: self.cursor_y += speed # DOWN
            elif disc_idx == 6: self.cursor_x -= speed # LEFT
            elif disc_idx == 7: self.cursor_x += speed # RIGHT
            
            # Maze walls?
            if "maze" in self.game_id:
                # Simple center block
                if 20 < self.cursor_x < 44 and 20 < self.cursor_y < 44:
                    reward -= 0.1 # Pain
                    # Bounce back (simplified)
                    if self.cursor_x > 32: self.cursor_x += 2
                    else: self.cursor_x -= 2
        
        else:
            # MOUSE MODE (Default)
            self.cursor_x += ax * 5.0
            self.cursor_y += ay * 5.0
        
        # Boundaries
        self.cursor_x = np.clip(self.cursor_x, 0, self.grid_size - 1)
        self.cursor_y = np.clip(self.cursor_y, 0, self.grid_size - 1)
        
        # Distance to target
        dist = np.sqrt((self.cursor_x - self.target_x)**2 + (self.cursor_y - self.target_y)**2)
        
        # Reward for being close
        if dist < 5.0:
            reward += 0.05
            
        # Target Interaction
        if "nav" in self.game_id or "maze" in self.game_id:
            # Just touching is enough for keyboard games
            if dist < 3.0:
                reward += 10.0
                hit = True
                self.target_x = np.random.randint(5, self.grid_size - 5)
                self.target_y = np.random.randint(5, self.grid_size - 5)
                self.score += 1
        else:
            # Must Click for mouse games
            if disc_idx <= 3 and dist < 2.0:
                reward += 10.0
                hit = True
                self.target_x = np.random.randint(5, self.grid_size - 5)
                self.target_y = np.random.randint(5, self.grid_size - 5)
                self.score += 1

        terminated = False
        truncated = self.current_step >= self.max_steps
        
        info = {
            "score": self.score,
            "dopamine": 1.0 if hit else 0.0,
            "pain": 0.0,
            "reward": reward
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        # Create dummy observation channels
        obs = np.zeros((self.grid_size, self.grid_size, 10), dtype=np.uint8)
        
        # Channel 0: Current Grid (Show Target)
        # Draw target as a pixel
        tx, ty = int(self.target_x), int(self.target_y)
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            obs[ty, tx, 0] = 255
            
        # Channel 7: Cursor
        cx, cy = int(self.cursor_x), int(self.cursor_y)
        if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
            obs[cy, cx, 7] = 255
            
        return obs
