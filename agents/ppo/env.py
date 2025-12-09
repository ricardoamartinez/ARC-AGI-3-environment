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
            return np.zeros((self.grid_size, self.grid_size, 8), dtype=np.uint8), {}
            
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
        # Initial Pain/Dopamine is 0.0
        obs = self._get_obs(current_grid, None, -1, 0.0, 0.0)
        logger.info("DEBUG: Env Reset Done")
        return obs, {}

    def step(self, action_tensor: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.current_step += 1
        
        # Capture Previous State for Frustration Check
        prev_x = self.agent.cursor_x
        prev_y = self.agent.cursor_y
        
        # 1. Physics & Movement
        curr_speed = self.physics.get_speed()
        
        # Pass Modality Bias to enforce filtering (Deprecated: Now Soft Cost)
        # mod_bias = self.intrinsic_system.modality_bias
        final_action_idx, ax, ay, trigger, selection = self.action_processor.process(action_tensor, curr_speed)
        
        self.agent.cursor_x, self.agent.cursor_y = self.physics.update(
            ax, ay, self.agent.cursor_x, self.agent.cursor_y
        )
        
        # Sync Manual Dopamine & Pain
        if hasattr(self.agent, 'manual_dopamine'):
            self.intrinsic_system.manual_dopamine = self.agent.manual_dopamine
        if hasattr(self.agent, 'manual_pain'):
            self.intrinsic_system.manual_pain = self.agent.manual_pain
            
        # Check for Frustration (Intrinsic Pain)
        # If effort is high (ax, ay) but movement is zero, it means we are stuck.
        effort = np.sqrt(ax**2 + ay**2)
        actual_movement = np.sqrt((self.agent.cursor_x - prev_x)**2 + (self.agent.cursor_y - prev_y)**2)
        
        if effort > 0.5 and actual_movement < 0.01:
             # Pushing hard but not moving -> Frustration
             # Apply intrinsic pain to teach the model to stop pushing walls
             
             # CRITICAL UPDATE: Update the SPATIAL pain memory immediately!
             cx = int(round(self.agent.cursor_x))
             cy = int(round(self.agent.cursor_y))
             if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
                 # Strong negative imprint at the stuck location
                 self.intrinsic_system.pain_memory[cy, cx] = 1.0
                 # Spread
                 for dy in [-1, 0, 1]:
                     for dx in [-1, 0, 1]:
                         if 0 <= cy+dy < self.grid_size and 0 <= cx+dx < self.grid_size:
                             self.intrinsic_system.pain_memory[cy+dy, cx+dx] = max(self.intrinsic_system.pain_memory[cy+dy, cx+dx], 0.8)

             # Add to scalar manual pain
             self.intrinsic_system.manual_pain += 0.2 # Increased from 0.1 for stronger signal
             self.intrinsic_system.manual_pain = min(1.0, self.intrinsic_system.manual_pain)
             
             self.intrinsic_system.current_thought = "Frustration! Stuck at wall."
            
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
                 return np.zeros((self.grid_size, self.grid_size, 8), dtype=np.uint8), 0.0, False, False, {}

            # --- RESTORE VISUALIZATION METADATA (Idle/Move) ---
            status = "Cursor Move"
            # if self.action_processor.action_cooldown > 0:
            #    status = "Cooldown..."
            if trigger > 0.2:
                # Approximate aiming status
                status = f"Aiming..."
            
            self.agent._last_action_viz = {
                "name": status,
                "data": {"x": self.agent.cursor_x, "y": self.agent.cursor_y}
            }
        
        if not frame:
            if self.agent.frames: frame = self.agent.frames[-1]
            else: return np.zeros((self.grid_size, self.grid_size, 8), dtype=np.uint8), 0.0, True, False, {}

        # 4. Process Outcome & Calculate Reward
        current_grid = np.array(frame.frame[0], dtype=np.uint8) if (frame.frame and frame.frame[0]) else None
        if current_grid is None:
             current_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # Base Reward (Time & Energy)
        reward = 0.0
        
        # Calculate Energy Cost (Metabolic cost of moving/acting)
        # This is NOT "Pain" in the manual sense, but a cost to prevent spasms.
        energy_cost = 0.0
        energy_cost += (abs(ax) + abs(ay)) * 0.01
        energy_cost += max(0.0, trigger) * 0.5
        
        # Removed Time Penalty (0.01 * self.current_step) because it causes runaway negative reward in infinite episodes.
        
        sparsity_multiplier = 1.0 / (1.0 + self.action_processor.consecutive_action_steps)
        
        if final_action_idx != -1:
            energy_cost += 2.0 # Action cost
            if self.action_processor.consecutive_action_steps > 1:
                energy_cost += 10.0 * self.action_processor.consecutive_action_steps
                self.intrinsic_system.dopamine_level = 0.0
        else:
             if trigger < 0.0:
                 reward += 0.1 + min(1.0, self.action_processor.idle_streak * 0.002)

        # Apply Energy Cost to Reward
        reward -= energy_cost

        # Calculate Pain (Manual & Game Over)
        pain = 0.0
        
        # Manual Pain (Administered by User)
        if hasattr(self.agent, 'manual_pain') and self.agent.manual_pain > 0:
            # Strong penalty for manual pain
            pain += self.agent.manual_pain * 50.0

        # Apply Pain
        reward -= pain

        # --- STABILITY REWARD ---
        # Reward repeating the same action type (Momentum) to discourage jitter
        # Action types: 0-3 (Click), 4-7 (Move), 8-9 (Meta)
        current_action_type = -1
        if final_action_idx != -1:
            if final_action_idx <= 3: current_action_type = 0 # Click
            elif final_action_idx <= 7: current_action_type = 1 # Move
            else: current_action_type = 2 # Meta
        
        # Check against previous action (stored in action_processor or similar, but we can track here)
        if hasattr(self, 'last_action_type'):
            if self.last_action_type == current_action_type and current_action_type != -1:
                 # Bonus for continuity
                 reward += 0.5
            elif self.last_action_type != current_action_type and self.last_action_type != -1 and current_action_type != -1:
                 # Penalty for rapid switching
                 reward -= 0.5
        
        self.last_action_type = current_action_type

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
             # REDUCED penalty for keyboard actions to encourage exploration
             # Was +1.8 (refund), let's make it actually slightly positive or neutral 
             # so it doesn't fear pressing keys that don't immediately change the grid.
             # Keys like "Copy/Paste" or "Move" might not change grid immediately but change internal state.
             # We rely on Intrinsic System for novelty rewards.
             # Here we just avoid punishing it.
             reward += 2.0 # Refund + tiny bonus to offset "Action cost" (which is 2.0)
             # Net cost = 2.0 (action) - 2.0 (refund) = 0.0. Free to try.
        
        # Check Win/Loss
        terminated = frame.state in [GameState.WIN, GameState.GAME_OVER]
        
        # User requested "Never Reset"
        # We disable timeout truncation.
        # We also might want to disable termination on Win/Loss to allow continuous play,
        # but the game engine effectively ends the session.
        # We will set truncated to False always.
        truncated = False 
        
        if frame.state == GameState.WIN:
            reward += 100.0
            # If "Never Reset" means "Stay in this state", we can set terminated=False.
            # But the backend might not accept more actions.
            # Let's assume standard behavior for Win/Loss but no timeouts.
        elif frame.state == GameState.GAME_OVER:
            reward -= 50.0
            pain += 50.0 # Game Over is effectively max pain
            
        prev_grid = self.last_grid
        self.last_grid = current_grid
        
        metrics = {
            "score": frame.score,
            "dopamine": self.intrinsic_system.predicted_dopamine,
            "manual_dopamine": self.intrinsic_system.manual_dopamine,
            "pain": pain, # Now only Manual Pain + Game Over, not metabolic costs
            "manual_pain": getattr(self.agent, "manual_pain", 0.0),
            "plan_confidence": self.intrinsic_system.plan_confidence,
            "current_thought": getattr(self.intrinsic_system, "current_thought", ""),
            "reward": reward,
            "trigger": float(trigger),
            "maps": {
                "pain": self.intrinsic_system.pain_memory.tolist(),
                "visit": self.intrinsic_system.spatial_visitation_map.tolist(),
                # Re-compute value map for visualization (cheap)
                "value": self.intrinsic_system.get_value_map().tolist(),
                
                # --- NEW REAL OBSERVATION CHANNELS ---
                # To be populated after _get_obs call below
            }
        }
        
        # PASS SPATIAL PAIN MAP TO OBS
        obs = self._get_obs(current_grid, prev_grid, final_action_idx, self.intrinsic_system.pain_memory, metrics["dopamine"])
        
        # Extract channels from obs (H, W, C)
        # 0: Current (Raw colors)
        # 1: Delta
        # 2: Focus
        # 3: Goal
        # 4: VelX
        # 5: VelY
        # 6: KB Bias
        # 7: Cursor Bias
        # 8: Pain
        # 9: Dopamine
        
        # Add to metrics["maps"]
        # obs is uint8 0-255. Normalize to 0.0-1.0 for visualization consistency?
        # Pygame visualizer expects 0-1 usually, or can handle raw values if scaled.
        # Let's send raw 0-255 or normalized. 
        # Existing maps (pain, visit) are 0-1 float.
        # Let's normalize.
        
        obs_float = obs.astype(np.float32) / 255.0
        
        metrics["maps"]["obs_delta"] = obs_float[:,:,1].tolist()
        metrics["maps"]["obs_focus"] = obs_float[:,:,2].tolist()
        metrics["maps"]["obs_goal"] = obs_float[:,:,3].tolist()
        metrics["maps"]["obs_vel_x"] = obs_float[:,:,4].tolist()
        metrics["maps"]["obs_vel_y"] = obs_float[:,:,5].tolist()
        metrics["maps"]["obs_pain"] = obs_float[:,:,8].tolist()
        
        return obs, reward, terminated, truncated, metrics

    def _get_obs(self, current_grid: np.ndarray, prev_grid: Optional[np.ndarray], last_action_idx: int = -1, pain: Any = 0.0, dopamine: float = 0.0) -> np.ndarray:
         value_map = self.intrinsic_system.get_value_map()
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
            value_map=value_map,
            modality_bias=self.intrinsic_system.modality_bias,
            last_action_idx=last_action_idx,
            pain=pain,
            dopamine=dopamine
        )
