import logging
import time
import json
import subprocess
import sys
import os
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState

logger = logging.getLogger()

class LiveVisualizerCallback(BaseCallback):
    """
    Callback for live visualization of the training process.
    """
    def __init__(self, gui_process, agent, verbose=0):
        super().__init__(verbose)
        self.gui_process = gui_process
        self.agent = agent

    def _on_step(self) -> bool:
        if self.gui_process and self.gui_process.poll() is None:
            try:
                latest_frame = self.agent.frames[-1] if self.agent.frames else None
                last_action = getattr(self.agent, "_last_action_viz", None)
                
                if latest_frame and latest_frame.frame:
                    msg = {
                        "grids": latest_frame.frame,
                        "game_id": self.agent.game_id,
                        "score": latest_frame.score,
                        "state": f"Step: {self.num_timesteps}",
                        "last_action": last_action,
                        "cursor": {"x": self.agent.cursor_x, "y": self.agent.cursor_y}
                    }
                    self.gui_process.stdin.write(json.dumps(msg) + "\n")
                    self.gui_process.stdin.flush()
            except Exception:
                pass # Ignore GUI errors to not stop training
        return True

class ARCGymEnv(gym.Env):
    """
    Gymnasium wrapper for ARC-AGI-3 games.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, agent: "PPOAgent", max_steps: int = 100):
        super().__init__()
        self.agent = agent
        self.max_steps = max_steps
        self.current_step = 0
        self.last_score = 0

        # Virtual Cursor State
        self.grid_size = 40
        self.cursor_x = self.grid_size // 2
        self.cursor_y = self.grid_size // 2

        # Observation: 30x30 grid (padded if smaller/larger)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.grid_size, self.grid_size, 1), dtype=np.uint8
        )

        # Actions: 
        # 0: Cursor Up
        # 1: Cursor Down
        # 2: Cursor Left
        # 3: Cursor Right
        # 4: Click (ACTION6 at Cursor)
        # 5: Game Up (ACTION1)
        # 6: Game Down (ACTION2)
        # 7: Game Left (ACTION3)
        # 8: Game Right (ACTION4)
        # 9: Space (ACTION5)
        # 10: Enter (ACTION7)
        
        self.action_space = spaces.Discrete(11)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.last_score = 0
        
        # Reset cursor to center
        self.agent.cursor_x = self.grid_size // 2
        self.agent.cursor_y = self.grid_size // 2
        
        # Trigger a reset in the actual game engine
        frame = self.agent.take_action(GameAction.RESET)
        if not frame:
            return np.zeros((self.grid_size, self.grid_size, 1), dtype=np.uint8), {}
            
        # Update agent state (important for GUID)
        self.agent.append_frame(frame)
            
        self.last_score = frame.score
        obs = self._process_frame(frame)
        return obs, {}

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.current_step += 1
        
        # Handle Cursor Movement locally
        if action_idx == 0: # C-UP
            self.agent.cursor_y = max(0, self.agent.cursor_y - 1)
        elif action_idx == 1: # C-DOWN
            self.agent.cursor_y = min(self.grid_size - 1, self.agent.cursor_y + 1)
        elif action_idx == 2: # C-LEFT
            self.agent.cursor_x = max(0, self.agent.cursor_x - 1)
        elif action_idx == 3: # C-RIGHT
            self.agent.cursor_x = min(self.grid_size - 1, self.agent.cursor_x + 1)
            
        # Determine Game Action
        game_action = None
        
        if action_idx == 4: # CLICK
            game_action = GameAction.ACTION6
            game_action.set_data({"x": self.agent.cursor_x, "y": self.agent.cursor_y, "game_id": self.agent.game_id})
        elif action_idx == 5: # G-UP
            game_action = GameAction.ACTION1
        elif action_idx == 6: # G-DOWN
            game_action = GameAction.ACTION2
        elif action_idx == 7: # G-LEFT
            game_action = GameAction.ACTION3
        elif action_idx == 8: # G-RIGHT
            game_action = GameAction.ACTION4
        elif action_idx == 9: # SPACE
            game_action = GameAction.ACTION5
        elif action_idx == 10: # ENTER
            game_action = GameAction.ACTION7
        
        frame = None
        if game_action:
            frame = self.agent.take_action(game_action)
        else:
            # No server action (just cursor move).
            if self.agent.frames:
                frame = self.agent.frames[-1]
            else:
                return np.zeros((self.grid_size, self.grid_size, 1), dtype=np.uint8), 0.0, True, False, {}

        if not frame:
             return np.zeros((self.grid_size, self.grid_size, 1), dtype=np.uint8), 0.0, True, False, {}

        if game_action:
            self.agent.append_frame(frame)
            self.agent._last_action_viz = {
                "id": game_action.value,
                "name": game_action.name,
                "data": game_action.action_data.model_dump()
            }
        else:
            # Optional: Update viz to show cursor movement
            self.agent._last_action_viz = {
                "name": "Cursor Move",
                "data": {"x": self.agent.cursor_x, "y": self.agent.cursor_y}
            }

        obs = self._process_frame(frame)
        
        # Reward logic
        reward = float(frame.score - self.last_score)
        self.last_score = frame.score
        
        if action_idx == 4:
            reward += 0.01 # Tiny reward to encourage clicking

        terminated = frame.state in [GameState.WIN, GameState.GAME_OVER]
        truncated = self.current_step >= self.max_steps
        
        if frame.state == GameState.WIN:
            reward += 100.0
        elif frame.state == GameState.GAME_OVER:
            reward -= 10.0

        return obs, reward, terminated, truncated, {"score": frame.score}

    def _process_frame(self, frame: FrameData) -> np.ndarray:
        # Extract grid
        if not frame.frame or not frame.frame[0]:
            return np.zeros((self.grid_size, self.grid_size, 1), dtype=np.uint8)
            
        grid = np.array(frame.frame[0], dtype=np.uint8)
        h, w = grid.shape
        
        processed = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        copy_h = min(h, self.grid_size)
        copy_w = min(w, self.grid_size)
        processed[:copy_h, :copy_w] = grid[:copy_h, :copy_w]
        
        # Draw Cursor on observation
        if 0 <= self.agent.cursor_y < self.grid_size and 0 <= self.agent.cursor_x < self.grid_size:
            processed[self.agent.cursor_y, self.agent.cursor_x] = 255
        
        return processed[:, :, np.newaxis]


class PPOAgent(Agent):
    """
    An agent that trains a PPO model to play the game.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.gui_process = None
        # Cursor state shared between Env and Visualizer
        self.cursor_x = 20 # Default center of 40x40
        self.cursor_y = 20
        self._start_gui()

    def _start_gui(self):
        try:
            # Re-use the manual_pygame.py script we built
            gui_script = os.path.join(os.path.dirname(__file__), "manual_pygame.py")
            self.gui_process = subprocess.Popen(
                [sys.executable, gui_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
                bufsize=1
            )
        except Exception as e:
            logger.error(f"Failed to start GUI: {e}")
            self.gui_process = None

    def cleanup(self, scorecard=None):
        super().cleanup(scorecard)
        if self.gui_process:
            try:
                self.gui_process.terminate()
            except:
                pass
            self.gui_process = None

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # PPO controls its own loop
        return True

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # Not used in training loop, but used if we run inference manually
        return GameAction.RESET

    def main(self) -> None:
        """
        Main entry point for the agent.
        """
        logger.info(f"Starting PPO Training for game {self.game_id}")
        
        # Create Environment
        # We pass 'self' so the environment can use our connection to the server
        env = ARCGymEnv(self, max_steps=100)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        # Setup Callback for Visualization
        callback = LiveVisualizerCallback(self.gui_process, self)
        
        # Create Model
        self.model = PPO(
            "CnnPolicy", 
            env, 
            verbose=1,
            learning_rate=0.0003,
            n_steps=256,
            batch_size=64,
            gamma=0.99,
            ent_coef=0.01,
        )
        
        # Train
        total_timesteps = 10000
        logger.info(f"Training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        
        # Save
        self.model.save(f"ppo_arc_{self.game_id}")
        logger.info("Training complete. Model saved.")
        
        # Final cleanup
        self.cleanup()

