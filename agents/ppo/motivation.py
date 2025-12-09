import hashlib
import time
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from .utils import find_connected_components

if TYPE_CHECKING:
    from .env import ARCGymEnv

class IntrinsicMotivationSystem:
    """
    Handles Dopamine, Curiosity, Rule Learning, and Plan Locking.
    Revised for Pavlovian Conditioning and Retroactive Credit Assignment.
    """
    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        
        # Dopamine & Focus
        self.dopamine_level = 0.0
        self.focus_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Learned Manifolds (Persistent Memory of Good/Bad Areas)
        self.learned_positive_manifold = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.learned_negative_manifold = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        self.manual_dopamine = 0.0
        self.manual_pain = 0.0
        self.pain_memory = np.zeros((grid_size, grid_size), dtype=np.float32) # Track painful locations (short term)
        self.current_thought = "Exploring environment." # Explicit thought for user
        
        # SPATIAL GOAL (Explicit User Flag)
        self.spatial_goal: Optional[Tuple[int, int]] = None
        
        # SPATIAL VISITATION MAP
        self.spatial_visitation_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Curiosity & State Tracking
        self.visited_hashes: Set[str] = set()
        self.state_visitation_counts: Dict[str, int] = {}
        self.state_action_counts: Dict[Tuple[str, int], int] = {}
        self.interacted_color_actions: Set[Tuple[int, int]] = set()
        
        # Conditioned Memory (Pavlovian)
        # Stores "Value" of invariant hashes based on past dopamine
        self.conditioned_stimuli: Dict[str, float] = {} 
        
        # Interaction History for Retroactive Credit Assignment
        self.interaction_history: List[dict] = []
        
        # Planning
        self.locked_plan: Optional[Tuple[int, str]] = None
        self.plan_confidence = 0.0
        
        # Modality Tracking (Cursor vs Keyboard)
        self.cursor_score = 0.0
        self.keyboard_score = 0.0
        self.modality_bias = 0.0 # -1.0 (Keyboard) to 1.0 (Cursor)
        self.modality_sensitivity = 0.5 # Slower adaptation to prevent jitter

    def reset(self):
        self.dopamine_level = 0.0
        self.manual_dopamine = 0.0
        self.manual_pain = 0.0
        self.focus_map.fill(0)
        self.pain_memory.fill(0)
        self.spatial_visitation_map.fill(0)
        self.current_thought = "Resetting."
        
        # We do NOT reset learned manifolds completely, as they represent long-term knowledge of "the task"
        # However, to prevent stuck states in new episodes, we decay them slightly.
        self.learned_positive_manifold *= 0.5
        self.learned_negative_manifold *= 0.5
        
        self.cursor_score = 0.0
        self.keyboard_score = 0.0
        self.modality_bias = 0.0
        
        self.visited_hashes.clear()
        self.interaction_history = []
        self.interacted_color_actions.clear()
        
        self.locked_plan = None
        self.plan_confidence = 0.0
        self.spatial_goal = None

    def update_dopamine(self, amount: float):
        self.dopamine_level = max(0.0, min(1.0, self.dopamine_level + amount))
    
    def decay(self):
        self.dopamine_level *= 0.95
        self.focus_map *= 0.9
        self.pain_memory *= 0.98
        self.spatial_visitation_map *= 0.995
        
        # Very slow decay for manifolds to allow "painting"
        self.learned_positive_manifold *= 0.999
        self.learned_negative_manifold *= 0.999

    def get_value_map(self) -> np.ndarray:
        """
        Returns a 64x64 map representing value (0-255).
        Combines Novelty + Locked Plan + Pain.
        Conditioned Stimuli are handled in get_dopamine_map.
        """
        # Base map based on Spatial Novelty
        novelty_map = 1.0 / (1.0 + self.spatial_visitation_map * 0.2)
        
        # Scale Novelty to Base Value (50 to 150)
        value_map = (novelty_map * 100.0 + 50.0).astype(np.float32)
        
        # Pain Avoidance
        pain_factor = np.clip(self.pain_memory * 20.0, 0, 128.0)
        
        # Add Negative Manifold
        pain_factor += np.clip(self.learned_negative_manifold * 20.0, 0, 128.0)
        
        value_map -= pain_factor
        
        # Focus
        focus_factor = self.focus_map * 50.0
        value_map += focus_factor
        
        # Locked Plan Highlight (Global Boost)
        if self.locked_plan:
             value_map += (self.dopamine_level * 20.0)
        
        # Explicit Spatial Goal (Flag)
        if self.spatial_goal:
            # Create a very strong gradient field towards the goal in the VALUE map too
            gx, gy = self.spatial_goal
            # Just highlight the specific point strongly
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                value_map[gy, gx] = 255.0
        
        return np.clip(value_map, 0, 255).astype(np.uint8)

    def get_dopamine_map(self, 
                        detected_objects: List[List[Tuple[int, int]]], 
                        grid: np.ndarray,
                        tracker: "ObjectTracker") -> np.ndarray:
        """
        Generates a spatial map of Expected Dopamine (Conditioned Stimuli).
        Highlights objects that have been associated with rewards.
        """
        d_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # 1. Global Dopamine (Smell) - decaying level
        d_map.fill(self.dopamine_level * 0.2)
        
        # 2. Object Specific Conditioning
        for obj_pixels in detected_objects:
            if not obj_pixels: continue
            
            # Get Invariant Hash
            r, c = obj_pixels[0]
            color = grid[r, c]
            inv_hash = tracker.get_invariant_hash(obj_pixels, color)
            
            # Check Value
            val = self.conditioned_stimuli.get(inv_hash, 0.0)
            
            # Also check tracker's valuable hashes (from score improvements)
            val += tracker.valuable_object_hashes.get(inv_hash, 0.0)
            
            if val > 0.1:
                intensity = min(1.0, val)
                for pr, pc in obj_pixels:
                    # Radius 1 expansion
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = pr+dr, pc+dc
                            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                                d_map[nr, nc] = max(d_map[nr, nc], intensity)
        
        # 3. Add Learned Positive Manifold (The "Painted" Boundary)
        d_map = np.maximum(d_map, self.learned_positive_manifold)
        
        # 4. Add Explicit Spatial Goal (The Flag)
        # This overrides everything. It's a beacon.
        if self.spatial_goal:
            gx, gy = self.spatial_goal
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                # Create a massive cone of dopamine centered on the goal
                # This ensures the gradient is visible from anywhere
                y_indices, x_indices = np.indices((self.grid_size, self.grid_size))
                dist = np.sqrt((x_indices - gx)**2 + (y_indices - gy)**2)
                
                # Inverted distance: 1.0 at source, falling off
                # Normalize distance to grid diagonal
                max_dist = self.grid_size * 1.414
                goal_gradient = 1.0 - (dist / max_dist)
                # Elevate it to be stronger than background
                goal_gradient = goal_gradient * 1.0 # Max 1.0 at source
                
                d_map = np.maximum(d_map, goal_gradient)

        # Manual Box Blur (Numpy only) to avoid scipy dependency and potential lag
        # A simple convolution with a kernel of ones
        kernel_size = 3
        pad = kernel_size // 2
        
        # Padded array
        padded = np.pad(d_map, pad, mode='constant', constant_values=0)
        
        # Vectorized convolution (summing shifted versions)
        # For a 3x3 box blur:
        blurred = np.zeros_like(d_map)
        for dy in range(-pad, pad+1):
            for dx in range(-pad, pad+1):
                # Shift and add
                blurred += padded[pad+dy : pad+dy+self.grid_size, pad+dx : pad+dx+self.grid_size]
        
        d_map = blurred / (kernel_size * kernel_size)
        
        # Second pass for more blur (approx Gaussian)
        padded = np.pad(d_map, pad, mode='constant', constant_values=0)
        blurred = np.zeros_like(d_map)
        for dy in range(-pad, pad+1):
            for dx in range(-pad, pad+1):
                blurred += padded[pad+dy : pad+dy+self.grid_size, pad+dx : pad+dx+self.grid_size]
        
        d_map = blurred / (kernel_size * kernel_size)

        # Rescale peak back to 1.0 (or original max) to prevent signal dilution
        d_max = d_map.max()
        if d_max > 0.01:
            # Maintain relative intensity but ensure visibility
            norm_factor = max(self.dopamine_level, 0.5) 
            # If explicit goal, ensure max intensity is high
            if self.spatial_goal: norm_factor = 1.0
            
            d_map = (d_map / d_max) * norm_factor
        
        return (np.clip(d_map, 0, 1) * 255).astype(np.uint8) # Allow values > 1.0 before clipping to uint8

    def process_step(self, 
                     env: "ARCGymEnv", 
                     reward: float, 
                     action_idx: int, 
                     click_action: bool, 
                     grid_changed: bool,
                     current_grid: np.ndarray,
                     last_grid: Optional[np.ndarray],
                     sparsity_multiplier: float) -> float:
        """
        Main logic for calculating intrinsic rewards and updating internal state.
        Returns modified reward.
        """
        self.decay()
        
        # --- SPATIAL COVERAGE ---
        cx = int(round(env.agent.cursor_x))
        cy = int(round(env.agent.cursor_y))
        
        if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
            self.spatial_visitation_map[cy, cx] += 1.0
            # Spread
            if cy > 0: self.spatial_visitation_map[cy-1, cx] += 0.2
            if cy < self.grid_size-1: self.spatial_visitation_map[cy+1, cx] += 0.2
            if cx > 0: self.spatial_visitation_map[cy, cx-1] += 0.2
            if cx < self.grid_size-1: self.spatial_visitation_map[cy, cx+1] += 0.2
            
            current_visits = self.spatial_visitation_map[cy, cx]
            exploration_reward = 0.2 / (current_visits + 0.5)
            reward += exploration_reward
            
            if exploration_reward > 0.1:
                self.current_thought = "Exploring new territory!"

        # --- MODALITY COST (Soft) ---
        if action_idx != -1:
            is_click = action_idx <= 3
            is_keyboard = action_idx >= 4
            
            # Soft bias application
            if self.modality_bias < -0.2 and is_click:
                reward -= 1.0 * abs(self.modality_bias) 
            elif self.modality_bias > 0.2 and is_keyboard:
                reward -= 1.0 * abs(self.modality_bias)

        # --- PAIN HANDLING ---
        if self.manual_pain > 0:
            self.dopamine_level = 0.0
            self.plan_confidence = 0.0
            self.locked_plan = None
            self.spatial_goal = None # Pain clears goal? Maybe not. But likely user wants to stop.
            
            # Update Negative Manifold
            if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
                # Add heavy penalty to the manifold
                self.learned_negative_manifold[cy, cx] = min(1.0, self.learned_negative_manifold[cy, cx] + self.manual_pain * 2.0)
                # Slight spread
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if 0 <= cy+dy < self.grid_size and 0 <= cx+dx < self.grid_size:
                             self.learned_negative_manifold[cy+dy, cx+dx] = max(self.learned_negative_manifold[cy+dy, cx+dx], self.learned_negative_manifold[cy, cx] * 0.5)
            
            # Pain Switch Modality
            if action_idx != -1:
                is_click = action_idx <= 3
                is_keyboard = action_idx >= 4
                if is_click:
                    self.cursor_score -= self.manual_pain * self.modality_sensitivity
                    self.keyboard_score += self.manual_pain * self.modality_sensitivity
                elif is_keyboard:
                    self.keyboard_score -= self.manual_pain * self.modality_sensitivity
                    self.cursor_score += self.manual_pain * self.modality_sensitivity
                self.modality_bias = np.clip(self.cursor_score - self.keyboard_score, -1.0, 1.0)
            
            # Pain Memory
            if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
                self.pain_memory[cy, cx] = 1.0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if 0 <= cy+dy < self.grid_size and 0 <= cx+dx < self.grid_size:
                            self.pain_memory[cy+dy, cx+dx] = max(self.pain_memory[cy+dy, cx+dx], 0.5)
            
            self.current_thought = "Pain detected! Marking as BAD."
        
        # --- INTERACTION HISTORY LOGGING ---
        # Identify object under cursor
        cursor_obj_invariant = "none"
        cursor_obj_hash = "none"
        
        target_grid = last_grid if (click_action and last_grid is not None) else current_grid
        
        # Get hash at current location
        obj_at_cursor_hash = env.object_tracker.get_object_hash_at(cy, cx, current_grid)
        
        # Let's find invariant hash
        obj_at_cursor_inv = "none"
        if obj_at_cursor_hash:
             # Find pixels
             for obj in env.object_tracker.detected_objects:
                 if (cy, cx) in obj:
                     color = current_grid[cy, cx]
                     obj_at_cursor_inv = env.object_tracker.get_invariant_hash(obj, color)
                     break
        
        self.interaction_history.append({
            "step": env.current_step,
            "action": action_idx,
            "invariant_hash": obj_at_cursor_inv,
            "grid_changed": grid_changed,
            "time": time.time(),
            "x": cx,
            "y": cy
        })
        if len(self.interaction_history) > 100: # Increase history size
            self.interaction_history.pop(0)

        # --- PAVLOVIAN CONDITIONING (Dopamine Handling) ---
        if self.manual_dopamine > 0.1:
            self.current_thought = "Good feedback! Conditioning..."
            reward += self.manual_dopamine * 0.05 # Immediate Reward (Drastically reduced)
            
            # Update Positive Manifold (Spatial Learning)
            if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
                # Add to positive manifold. 
                # Use a tighter Gaussian splat for precision if the user wants it.
                # Since we don't know if they want precision or area, we use a 3x3 splat with center bias.
                self.learned_positive_manifold[cy, cx] = min(1.0, self.learned_positive_manifold[cy, cx] + self.manual_dopamine * 0.05)
                
                # Spread
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0: continue
                        if 0 <= cy+dy < self.grid_size and 0 <= cx+dx < self.grid_size:
                            # 50% decay for neighbors -> tight-ish peak
                             self.learned_positive_manifold[cy+dy, cx+dx] = max(self.learned_positive_manifold[cy+dy, cx+dx], self.learned_positive_manifold[cy, cx] * 0.5)

            # Retroactive Credit Assignment
            best_match = None
            best_match_score = -1.0
            
            for i, event in enumerate(reversed(self.interaction_history)):
                steps_ago = env.current_step - event["step"]
                # Decay relevance
                relevance = 1.0 / (1.0 + steps_ago * 0.1)
                
                score = 0.0
                if event["grid_changed"]: score += 0.5
                elif event["invariant_hash"] != "none": score += 0.2
                else: score += 0.05
                
                final_score = score * relevance
                if final_score > best_match_score:
                    best_match_score = final_score
                    best_match = event
            
            if best_match and best_match["invariant_hash"] != "none":
                target_inv = best_match["invariant_hash"]
                target_act = best_match["action"]
                
                # 1. Update Conditioned Stimulus (The Object)
                # "This object is associated with reward"
                old_val = self.conditioned_stimuli.get(target_inv, 0.0)
                self.conditioned_stimuli[target_inv] = min(1.0, old_val + self.manual_dopamine * 0.05)
                
                # 2. Update Valuable Hashes in Tracker (Global knowledge)
                env.object_tracker.valuable_object_hashes[target_inv] = \
                    env.object_tracker.valuable_object_hashes.get(target_inv, 0.0) + self.manual_dopamine * 0.1
                
                self.current_thought = f"Conditioned: Object {target_inv[:6]} is GOOD."
                
                # 3. Lock Plan (Short term repetition)
                self.locked_plan = (target_act, target_inv)
                self.plan_confidence = 1.0
                self.dopamine_level = 1.0
            elif best_match:
                 # Condition Spatial Location if no object?
                 # Create a "virtual" object at that location in focus map
                 tx, ty = best_match["x"], best_match["y"]
                 # Radius 5 (Broader)
                 for dy in range(-5, 6):
                     for dx in range(-5, 6):
                         if 0 <= ty+dy < self.grid_size and 0 <= tx+dx < self.grid_size:
                             dist = np.sqrt(dx*dx + dy*dy)
                             if dist <= 5.0:
                                 self.focus_map[ty+dy, tx+dx] = max(self.focus_map[ty+dy, tx+dx], 1.0 - dist/5.0)
                 self.current_thought = "Conditioned: This AREA is GOOD."

        # --- PLAN FOLLOWING ---
        if self.locked_plan:
            t_action, t_inv = self.locked_plan
            
            # Am I interacting with the target?
            is_matching = False
            if obj_at_cursor_inv == t_inv:
                is_matching = True
            
            # If I am near it and doing the right action
            if is_matching and action_idx == t_action:
                reward += 2.0
                self.update_dopamine(0.1)
            elif is_matching:
                # Near object but wrong action
                reward += 0.5
            elif action_idx == t_action:
                 # Right action, wrong object
                 pass
            
            # Decay plan if failing
            if click_action and not is_matching and not grid_changed:
                self.plan_confidence -= 0.1
                if self.plan_confidence <= 0:
                    self.locked_plan = None
        
        # --- CURIOSITY (Reduced) ---
        state_hash = hashlib.md5(current_grid.tobytes()).hexdigest()
        if state_hash not in self.visited_hashes:
            reward += 0.1 * sparsity_multiplier
            self.visited_hashes.add(state_hash)
            
        return reward
