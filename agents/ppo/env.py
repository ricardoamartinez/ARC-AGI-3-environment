import hashlib
import logging
import time
import gymnasium as gym
import numpy as np
from typing import Any, Optional, Set, List, Tuple, TYPE_CHECKING, Dict
from gymnasium import spaces

from ..structs import FrameData, GameAction, GameState
from .physics import PhysicsEngine
from .actions import ActionProcessor
from .tracker import ObjectTracker
from .motivation import IntrinsicMotivationSystem
from .observation import ObservationBuilder

logger = logging.getLogger()

if TYPE_CHECKING:
    from .agent import PPOAgent

class ARCGymEnv(gym.Env):
    """
    Gymnasium wrapper for ARC-AGI-3 games.
    Refactored to use component classes.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, agent: "PPOAgent", max_steps: int = 100):
        super().__init__()
        self.agent = agent
        self.max_steps = max_steps
        self.current_step = 0
        self.last_score = 0
        self.grid_size = 64
        
        # Components
        self.physics = PhysicsEngine(self.grid_size)
        self.action_processor = ActionProcessor()
        self.object_tracker = ObjectTracker()
        self.intrinsic_system = IntrinsicMotivationSystem(self.grid_size)
        self.obs_builder = ObservationBuilder(self.grid_size)
        
        self.last_grid: Optional[np.ndarray] = None
        
        # ES Population (kept in Env for now)
        self.es_population_size = 5
        self.es_population = [np.random.randn(4) for _ in range(self.es_population_size)]
        self.current_goal_vector = self.es_population[0]
        
        self.observation_space = self.obs_builder.observation_space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict[str, Any]]:
        logger.info("DEBUG: Env Reset Start")
        super().reset(seed=seed)
        
        self.current_step = 0
        self.last_score = 0
        self.last_grid = None
        
        # Reset Components
        self.physics.reset()
        self.action_processor.reset()
        self.object_tracker.reset()
        self.intrinsic_system.reset()
        
        # Reset Agent Cursor to center
        self.agent.cursor_x = self.grid_size // 2
        self.agent.cursor_y = self.grid_size // 2
        
        # Trigger Reset in Game Engine
        frame = self.agent.take_action(GameAction.RESET)
        
        if not frame:
            return np.zeros((self.grid_size, self.grid_size, 6), dtype=np.uint8), {}
            
        self.agent.append_frame(frame)
        self.last_score = frame.score
        
        # Sync Cursor if provided by engine
        if frame.frame and frame.frame[0]:
            grid = np.array(frame.frame[0], dtype=np.uint8)
            h, w = grid.shape
            self.agent.cursor_x = w / 2.0
            self.agent.cursor_y = h / 2.0
            self.last_grid = grid
            self.object_tracker.scan(grid)
            
            # Initial Hash
            s_hash = hashlib.md5(grid.tobytes()).hexdigest()
            self.intrinsic_system.visited_hashes.add(s_hash)
            self.intrinsic_system.state_visitation_counts[s_hash] = self.intrinsic_system.state_visitation_counts.get(s_hash, 0) + 1
            
        # Precompute Goal Channel
        self.obs_builder.precompute_goal_channel(self.current_goal_vector)
        
        current_grid = self.last_grid if self.last_grid is not None else np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        obs = self._get_obs(current_grid, None, -1)
        logger.info("DEBUG: Env Reset Done")
        return obs, {}

    def step(self, action_tensor: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.current_step += 1
        
        # 1. Physics & Movement
        curr_speed = self.physics.get_speed()
        final_action_idx, ax, ay, trigger, selection = self.action_processor.process(action_tensor, curr_speed)
        
        self.agent.cursor_x, self.agent.cursor_y = self.physics.update(
            ax, ay, self.agent.cursor_x, self.agent.cursor_y
        )
        
        # Sync Manual Dopamine
        if hasattr(self.agent, 'manual_dopamine'):
            self.intrinsic_system.manual_dopamine = self.agent.manual_dopamine
            
        # 2. Determine Game Action
        cx_int = int(round(self.agent.cursor_x))
        cy_int = int(round(self.agent.cursor_y))
        game_action = self.action_processor.get_game_action(final_action_idx, cx_int, cy_int, self.agent.game_id)
        
        # 3. Execute Action
        frame = None
        if game_action:
            frame = self.agent.take_action(game_action)
            self.agent.append_frame(frame)
            # Share data
            self.agent.latest_detected_objects = self.object_tracker.detected_objects
            
            # --- RESTORE VISUALIZATION METADATA ---
            action_name = game_action.name
            is_click = final_action_idx <= 3
            
            if final_action_idx == 4: action_name = "UP ↑"
            elif final_action_idx == 5: action_name = "DOWN ↓"
            elif final_action_idx == 6: action_name = "LEFT ←"
            elif final_action_idx == 7: action_name = "RIGHT →"
            elif final_action_idx == 8: action_name = "SPACE ␣"
            elif final_action_idx == 9: action_name = "ENTER ↵"
            elif is_click: action_name = "CLICK 🖱️"
            
            self.agent._last_action_viz = {
                "id": game_action.value,
                "name": action_name,
                "data": game_action.action_data.model_dump()
            }
        else:
            if self.agent.frames:
                 frame = self.agent.frames[-1]
            else:
                 return np.zeros((self.grid_size, self.grid_size, 6), dtype=np.uint8), 0.0, False, False, {}

            # --- RESTORE VISUALIZATION METADATA (Idle/Move) ---
            status = "Cursor Move"
            if self.action_processor.action_cooldown > 0:
                status = "Cooldown..."
            elif trigger > 0.2:
                # Approximate aiming status
                status = f"Aiming..."
            
            self.agent._last_action_viz = {
                "name": status,
                "data": {"x": self.agent.cursor_x, "y": self.agent.cursor_y}
            }
        
        if not frame:
             if self.agent.frames: frame = self.agent.frames[-1]
             else: return np.zeros((self.grid_size, self.grid_size, 6), dtype=np.uint8), 0.0, True, False, {}

        # 4. Process Outcome & Calculate Reward
        current_grid = np.array(frame.frame[0], dtype=np.uint8) if (frame.frame and frame.frame[0]) else None
        if current_grid is None:
             current_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # Base Reward (Time & Energy)
        reward = 0.0
        reward -= (abs(ax) + abs(ay)) * 0.01
        reward -= max(0.0, trigger) * 0.5
        reward -= 0.01 * self.current_step # Time penalty
        
        sparsity_multiplier = 1.0 / (1.0 + self.action_processor.consecutive_action_steps)
        
        if final_action_idx != -1:
            reward -= 2.0 # Action cost
            if self.action_processor.consecutive_action_steps > 1:
                reward -= 10.0 * self.action_processor.consecutive_action_steps
                self.intrinsic_system.dopamine_level = 0.0
        else:
             if trigger < 0.0:
                 reward += 0.1 + min(1.0, self.action_processor.idle_streak * 0.002)

        # Score Diff
        score_diff = float(frame.score - self.last_score)
        reward += score_diff * 10.0
        self.last_score = frame.score
        
        # Backpropagate Object Value if Score Improved
        if score_diff > 0:
            decay = 0.9
            current_val = score_diff * 5.0
            for step_hashes in reversed(self.object_tracker.episode_object_hashes):
                for h in step_hashes:
                    self.object_tracker.valuable_object_hashes[h] = self.object_tracker.valuable_object_hashes.get(h, 0) + current_val
                current_val *= decay
                if current_val < 0.01: break

        # Intrinsic System
        grid_changed = False
        if self.last_grid is not None and current_grid.shape == self.last_grid.shape:
             grid_changed = not np.array_equal(current_grid, self.last_grid)
        
        click_action = final_action_idx != -1 and final_action_idx <= 3
        
        # Update Object Tracker if grid changed
        if grid_changed:
            self.object_tracker.scan(current_grid)

        # Delegate to Intrinsic System
        reward = self.intrinsic_system.process_step(
            env=self,
            reward=reward,
            action_idx=final_action_idx,
            click_action=click_action,
            grid_changed=grid_changed,
            current_grid=current_grid,
            last_grid=self.last_grid,
            sparsity_multiplier=sparsity_multiplier
        )
        
        # Additional logic (Spam penalties, etc.) from original file
        if click_action and not grid_changed:
             reward -= 5.0
        elif final_action_idx in [4,5,6,7,8,9] and not grid_changed:
             reward += 1.8 # Refund sparsity
        
        # Check Win/Loss
        terminated = frame.state in [GameState.WIN, GameState.GAME_OVER]
        truncated = self.current_step >= self.max_steps
        
        if frame.state == GameState.WIN:
            reward += 100.0
        elif frame.state == GameState.GAME_OVER:
            reward -= 50.0

        prev_grid = self.last_grid
        self.last_grid = current_grid
        obs = self._get_obs(current_grid, prev_grid, final_action_idx)
        
        metrics = {
            "score": frame.score,
            "dopamine": self.intrinsic_system.predicted_dopamine,
            "manual_dopamine": self.intrinsic_system.manual_dopamine,
            "plan_confidence": self.intrinsic_system.plan_confidence,
            "reward": reward,
            "trigger": float(trigger)
        }
        
        return obs, reward, terminated, truncated, metrics

    def _get_obs(self, current_grid: np.ndarray, prev_grid: Optional[np.ndarray], last_action_idx: int = -1) -> np.ndarray:
         return self.obs_builder.build(
            current_grid=current_grid,
            last_grid=prev_grid,
            focus_map=self.intrinsic_system.focus_map,
            detected_objects=self.object_tracker.detected_objects,
            valuable_hashes=self.object_tracker.valuable_object_hashes,
            cursor_x=self.agent.cursor_x,
            cursor_y=self.agent.cursor_y,
            vel_x=self.physics.vel_x,
            vel_y=self.physics.vel_y,
            locked_plan=self.intrinsic_system.locked_plan,
            last_action_idx=last_action_idx
        )
