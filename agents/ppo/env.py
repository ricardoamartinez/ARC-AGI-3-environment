import hashlib
import gymnasium as gym
import numpy as np
from typing import Any, Optional, Set, List, Tuple, TYPE_CHECKING
from gymnasium import spaces

from ..structs import FrameData, GameAction, GameState
from .utils import find_connected_components

if TYPE_CHECKING:
    from .agent import PPOAgent

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
        
        # Heuristic Tracking
        self.visited_hashes: Set[str] = set()
        self.last_grid: Optional[np.ndarray] = None
        
        # Curiosity / Rare State Tracking
        self.state_visitation_counts: dict[str, int] = {} # Hash -> Count
        self.trajectory_buffer: List[str] = [] # Sequence of state hashes in current episode
        
        # Color Curiosity State
        self.interacted_color_actions: Set[Tuple[int, int]] = set() # (ActionIdx, Color)
        self.observed_color_transitions: Set[Tuple[int, int]] = set() # (OldColor, NewColor)
        
        # State-Action "Conditional" Exploration
        self.state_action_counts: dict[Tuple[str, int], int] = {} # (StateHash, ActionIdx) -> Count
        
        # "Object-Oriented Rule Learning" State
        # Maps object_hash -> accumulated value from successful trajectories
        self.valuable_object_hashes: dict[str, float] = {} 
        self.episode_object_hashes: List[List[str]] = [] # History of objects for current episode

        # Object Tracking State
        self.detected_objects: List[List[Tuple[int, int]]] = []
        self.current_object_idx = 0
        
        # Virtual Cursor State
        self.grid_size = 64
        self.cursor_x = self.grid_size // 2
        self.cursor_y = self.grid_size // 2

        # Observation: 64x64x3
        # Channel 0: Current Grid
        # Channel 1: Difference/Delta (Effect Visualization)
        # Channel 2: Object/Cursor Mask (Cause Candidates)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.grid_size, self.grid_size, 3), dtype=np.uint8
        )

        # Actions: 
        # 0: Jump to NEXT Object & Click
        # 1: Jump to PREVIOUS Object & Click
        # 2: Jitter Cursor & Click
        # 3: Jump to Center & Click
        # 4: Game Up (ACTION1)
        # 5: Game Down (ACTION2)
        # 6: Game Left (ACTION3)
        # 7: Game Right (ACTION4)
        # 8: Space (ACTION5)
        # 9: Enter (ACTION7)
        
        self.action_space = spaces.Discrete(10)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.last_score = 0
        # NOTE: We do NOT clear state_visitation_counts here, as curiosity is lifetime-based
        self.visited_hashes.clear() # Clear per-episode uniqueness, but keep lifetime counts separate
        self.trajectory_buffer = []
        self.recent_actions = [] # Reset action history
        self.interacted_color_actions.clear()
        self.observed_color_transitions.clear()
        self.episode_object_hashes = [] # Clear episode history
        
        if hasattr(self, 'interacted_objects'):
            self.interacted_objects.clear()
        self.last_grid = None
        self.detected_objects = []
        self.current_object_idx = 0
        
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
        
        # Initial Object Scan
        if frame.frame and frame.frame[0]:
             grid = np.array(frame.frame[0], dtype=np.uint8)
             self.last_grid = grid
             self.detected_objects = find_connected_components(grid)
             s_hash = hashlib.md5(grid.tobytes()).hexdigest()
             self.visited_hashes.add(s_hash)
             self.trajectory_buffer.append(s_hash)
             
             # Update lifetime counts
             self.state_visitation_counts[s_hash] = self.state_visitation_counts.get(s_hash, 0) + 1

        obs = self._process_frame(frame)
        return obs, {}

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.current_step += 1
        
        # --- SMART NAVIGATION & INTERACTION ---
        # Actions 0-3 are now coupled: Move AND Click
        
        click_action = False
        
        if action_idx == 0: # NEXT OBJECT & CLICK
            if self.detected_objects:
                self.current_object_idx = (self.current_object_idx + 1) % len(self.detected_objects)
                obj_pixels = self.detected_objects[self.current_object_idx]
                ys = [p[0] for p in obj_pixels]
                xs = [p[1] for p in obj_pixels]
                self.agent.cursor_y = int(np.mean(ys))
                self.agent.cursor_x = int(np.mean(xs))
            click_action = True
                
        elif action_idx == 1: # PREV OBJECT & CLICK
            if self.detected_objects:
                self.current_object_idx = (self.current_object_idx - 1) % len(self.detected_objects)
                obj_pixels = self.detected_objects[self.current_object_idx]
                ys = [p[0] for p in obj_pixels]
                xs = [p[1] for p in obj_pixels]
                self.agent.cursor_y = int(np.mean(ys))
                self.agent.cursor_x = int(np.mean(xs))
            click_action = True
                
        elif action_idx == 2: # JITTER & CLICK
            # Move randomly within a small 3x3 radius
            dx = np.random.randint(-2, 3)
            dy = np.random.randint(-2, 3)
            self.agent.cursor_x = max(0, min(self.grid_size - 1, self.agent.cursor_x + dx))
            self.agent.cursor_y = max(0, min(self.grid_size - 1, self.agent.cursor_y + dy))
            click_action = True
            
        elif action_idx == 3: # RESET & CLICK
            self.agent.cursor_x = self.grid_size // 2
            self.agent.cursor_y = self.grid_size // 2
            click_action = True
            
        # Determine Game Action
        game_action = None
        
        if click_action: # Actions 0-3
            game_action = GameAction.ACTION6
            game_action.set_data({"x": self.agent.cursor_x, "y": self.agent.cursor_y, "game_id": self.agent.game_id})
        elif action_idx == 4: # G-UP
            game_action = GameAction.ACTION1
        elif action_idx == 5: # G-DOWN
            game_action = GameAction.ACTION2
        elif action_idx == 6: # G-LEFT
            game_action = GameAction.ACTION3
        elif action_idx == 7: # G-RIGHT
            game_action = GameAction.ACTION4
        elif action_idx == 8: # SPACE
            game_action = GameAction.ACTION5
        elif action_idx == 9: # ENTER
            game_action = GameAction.ACTION7
        
        frame = None
        if game_action:
            frame = self.agent.take_action(game_action)
        else:
            # Should not be reached with current action space, but safe fallback
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
            self.agent._last_action_viz = {
                "name": "Cursor Move",
                "data": {"x": self.agent.cursor_x, "y": self.agent.cursor_y}
            }

        obs = self._process_frame(frame)
        
        # --- HEURISTIC REWARD CALCULATION ---
        reward = 0.0
        
        # 0. Time/Step Loss (Linear Accumulation)
        # "Per step loss accumulation should increase linearly"
        # We punish taking too long. The penalty grows as the episode progresses.
        # Step 1: -0.01, Step 50: -0.50, Step 100: -1.0
        time_penalty = -0.01 * self.current_step
        reward += time_penalty
        
        # 1. Extrinsic Reward: Score Change
        score_diff = float(frame.score - self.last_score)
        reward += score_diff * 10.0 
        self.last_score = frame.score
        
        # 2. Intrinsic Reward: Novelty & Interaction
        current_grid = np.array(frame.frame[0], dtype=np.uint8) if (frame.frame and frame.frame[0]) else None
        
        if current_grid is not None:
            state_hash = hashlib.md5(current_grid.tobytes()).hexdigest()
            
            # Check for Grid Change (Effectance)
            grid_changed = False
            if self.last_grid is not None and current_grid.shape == self.last_grid.shape:
                 if not np.array_equal(current_grid, self.last_grid):
                     grid_changed = True
                     
            # REWARD for State Change
            if grid_changed:
                 reward += 1.0
            # PUNISH for No State Change (if action was attempted)
            elif click_action or action_idx in [4, 5, 6, 7, 8, 9]:
                 reward -= 1.0 # Punish inefficiency
            
            # --- LIFETIME CURIOSITY (Rare State Seeking) ---
            # We track how many times we've EVER seen this state.
            # Reward is inversely proportional to visitation count.
            # R = 1 / sqrt(count) -> Diminishing returns for familiar states.
            
            visit_count = self.state_visitation_counts.get(state_hash, 0) + 1
            self.state_visitation_counts[state_hash] = visit_count
            
            curiosity_reward = 5.0 / np.sqrt(visit_count) 
            reward += curiosity_reward
            
            # Check for Grid Change (Effectance)
            # (Logic moved up for global reward/punishment)
            
            # Check for Novelty (New State in THIS episode)
            if state_hash not in self.visited_hashes:
                reward += 2.0 
                self.visited_hashes.add(state_hash)
            
            # 3. Heuristic: Object Interaction
            # REFRESH OBJECT LIST
            self.detected_objects = find_connected_components(current_grid)
            
            # --- OBJECT-ORIENTED RULE LEARNING (Sequence Bias) ---
            # 1. Hash current objects
            current_step_obj_hashes = []
            guidance_reward = 0.0
            
            for obj_pixels in self.detected_objects:
                if not obj_pixels: continue
                # Get color from first pixel
                r, c = obj_pixels[0]
                color = current_grid[r, c]
                # Hash: Color + Sorted Pixels (Absolute Position)
                # This identifies the specific state of this object
                sorted_pixels = sorted(obj_pixels)
                obj_hash = hashlib.md5(f"{color}_{str(sorted_pixels)}".encode()).hexdigest()
                current_step_obj_hashes.append(obj_hash)
                
                # Check if this object state has been part of a winning sequence before
                if obj_hash in self.valuable_object_hashes:
                    # Give a "Guidance Reward" for being in a state that led to points previously
                    guidance_reward += self.valuable_object_hashes[obj_hash]
            
            # Add to episode history
            self.episode_object_hashes.append(current_step_obj_hashes)
            
            # Apply Guidance Reward (Scaled down to avoid overwhelming)
            reward += guidance_reward * 0.1
            
            # 2. Backpropagate Value if Score Improved
            if score_diff > 0:
                # The actions/states leading up to this point were GOOD.
                # Backtrack through episode history and assign value to those object states.
                decay = 0.9
                current_val = score_diff * 5.0 # Base value of the achievement
                
                # Iterate backwards from current step
                for step_hashes in reversed(self.episode_object_hashes):
                    for h in step_hashes:
                        # Accumulate value (if multiple paths lead here, it gets more valuable)
                        self.valuable_object_hashes[h] = self.valuable_object_hashes.get(h, 0) + current_val
                    
                    current_val *= decay # Previous steps are slightly less directly responsible
                    if current_val < 0.01: break # Optimization
            
            # Track which objects we have interacted with in this episode
            if not hasattr(self, 'interacted_objects'):
                self.interacted_objects: Set[str] = set()
            
            # Action History Tracking (Anti-Collapse)
            if not hasattr(self, 'recent_actions'):
                self.recent_actions = []
            self.recent_actions.append(action_idx)
            if len(self.recent_actions) > 10:
                self.recent_actions.pop(0)
            
            # Check if cursor is on an object
            cursor_on_object = False
            current_object_hash = None
            
            for obj_pixels in self.detected_objects:
                if (self.agent.cursor_y, self.agent.cursor_x) in obj_pixels:
                    cursor_on_object = True
                    sorted_pixels = sorted(obj_pixels)
                    current_object_hash = hashlib.md5(str(sorted_pixels).encode()).hexdigest()
                    break
            
            if click_action: # Replaces logic for old action 4
                if cursor_on_object:
                    if current_object_hash and current_object_hash not in self.interacted_objects:
                        reward += 5.0 # HUGE Reward for clicking a NEW object
                        self.interacted_objects.add(current_object_hash)
                    else:
                        reward -= 2.0 # INCREASED Penalty for re-clicking
                else:
                    reward -= 1.0 # INCREASED Penalty for clicking empty space
            
            # REPETITION PENALTY (Anti-Collapse)
            # If the agent repeats the same action more than 3 times in the last 5 steps, penalize it.
            if len(self.recent_actions) >= 5:
                last_5 = self.recent_actions[-5:]
                if last_5.count(action_idx) >= 4:
                    reward -= 2.0 # Strong penalty for getting stuck in a loop
            
            if grid_changed and action_idx in [8, 9, 4, 5, 6, 7]: # Space, Enter, Arrows (Updated indices)
                # (Reward handled globally)
                pass
                
                # --- COLOR TRANSITION ANALYSIS (Effect Curiosity) ---
                # Identify what colors changed to what
                # We look at the diff between last_grid and current_grid
                diff_mask = current_grid != self.last_grid
                if np.any(diff_mask):
                    old_vals = self.last_grid[diff_mask]
                    new_vals = current_grid[diff_mask]
                    
                    # Find unique transitions in this step
                    step_transitions = set(zip(old_vals, new_vals))
                    
                    for t in step_transitions:
                        if t not in self.observed_color_transitions:
                            # Found a NEW way to change colors (e.g. turning Blue -> Red)
                            reward += 3.0 
                            self.observed_color_transitions.add(t)
            
            # NEW: Punishment for Ineffective Clicks
            # If the agent clicked (Actions 0-3) and the grid did NOT change, punish it.
            if click_action and not grid_changed:
                reward -= 2.0
            
            # --- STATE-ACTION CURIOSITY (Conditional Branching) ---
            # "Unlock novel paths": Try every action at least once from every state.
            # "Bias towards good attempts": If we are in a rare/good state, exploring from it is highly rewarded.
            
            sa_pair = (state_hash, action_idx)
            sa_count = self.state_action_counts.get(sa_pair, 0)
            self.state_action_counts[sa_pair] = sa_count + 1
            
            # Reward decreases as we repeat the same action in the same state.
            # R = 3.0 / (1 + count)
            # 1st time: +3.0
            # 2nd time: +1.5
            # 3rd time: +1.0
            # ...
            # This forces the agent to "branch out" (try other actions) before repeating.
            sa_reward = 3.0 / (1.0 + sa_count)
            
            # Bias: If this state itself is RARE (a "good attempt" or deep state), boost the exploration reward.
            # We want to explore thoroughly when we find something new.
            state_rarity_boost = 1.0
            if visit_count < 5: # We've been here fewer than 5 times
                state_rarity_boost = 2.0
            
            reward += sa_reward * state_rarity_boost

            # --- COLOR INTERACTION CURIOSITY (Action Curiosity) ---
            # Reward applying specific actions to specific colors for the first time
            # e.g. "Clicking on Blue", "Pressing Space on Red"
            
            # Get color under cursor
            cursor_color = 0
            if 0 <= self.agent.cursor_y < self.grid_size and 0 <= self.agent.cursor_x < self.grid_size:
                cursor_color = current_grid[self.agent.cursor_y, self.agent.cursor_x]
            
            # Only track for "active" actions (Click, Arrows, Space, Enter)
            # Indices: 0-3 (Click variations), 4-9 (Game Actions)
            # We simplify 0-3 to just "Click" (Action 4 conceptually) for color pairing
            effective_action = action_idx
            if action_idx <= 3: 
                effective_action = 99 # Special ID for "Any Click"
                
            action_color_pair = (effective_action, cursor_color)
            
            if action_color_pair not in self.interacted_color_actions:
                reward += 1.5 # Reward for trying this action on this color for the first time
                self.interacted_color_actions.add(action_color_pair)

            self.last_grid = current_grid

        terminated = frame.state in [GameState.WIN, GameState.GAME_OVER]
        truncated = self.current_step >= self.max_steps
        
        if frame.state == GameState.WIN:
            reward += 100.0
        elif frame.state == GameState.GAME_OVER:
            reward -= 50.0 # Heavy punishment for Game Over (losing)

        return obs, reward, terminated, truncated, {"score": frame.score}

    def _process_frame(self, frame: FrameData) -> np.ndarray:
        # 1. Extract Current Grid
        if not frame.frame or not frame.frame[0]:
            current = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        else:
            grid = np.array(frame.frame[0], dtype=np.uint8)
            h, w = grid.shape
            current = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
            copy_h = min(h, self.grid_size)
            copy_w = min(w, self.grid_size)
            current[:copy_h, :copy_w] = grid[:copy_h, :copy_w]
            
        # 2. Extract Previous Grid (for Effect/Delta)
        if self.last_grid is not None:
            prev = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
            h, w = self.last_grid.shape
            copy_h = min(h, self.grid_size)
            copy_w = min(w, self.grid_size)
            prev[:copy_h, :copy_w] = self.last_grid[:copy_h, :copy_w]
            
            # Delta: Absolute difference highlights changes
            delta = np.abs(current.astype(np.int16) - prev.astype(np.int16)).astype(np.uint8)
            # Amplify delta for visibility
            delta[delta > 0] = 255
        else:
            delta = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
            
        # 3. Construct Object/Cursor Mask (Cause Candidates)
        mask = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        # Draw detected objects faintly
        for obj in self.detected_objects:
            for r, c in obj:
                if r < self.grid_size and c < self.grid_size:
                    mask[r, c] = 100
        
        # Draw Cursor Brightly
        if 0 <= self.agent.cursor_y < self.grid_size and 0 <= self.agent.cursor_x < self.grid_size:
            mask[self.agent.cursor_y, self.agent.cursor_x] = 255
        
        # Stack Channels
        # Shape: (H, W, 3)
        return np.stack([current, delta, mask], axis=-1)
