import hashlib
import logging
import gymnasium as gym
import numpy as np
from typing import Any, Optional, Set, List, Tuple, TYPE_CHECKING
from gymnasium import spaces

from ..structs import FrameData, GameAction, GameState
from .utils import find_connected_components

logger = logging.getLogger()

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
        self.valuable_object_hashes: dict[str, float] = {} 
        self.episode_object_hashes: List[List[str]] = [] # History of objects for current episode
        
        # Virtual Cursor State
        self.grid_size = 64
        # Cursor position is stored in agent for sharing with visualizer
        # Velocity is local to env for physics simulation
        self.vel_x = 0.0
        self.vel_y = 0.0
        
        # Physics constants for smooth, momentum-based cursor control
        # The model outputs acceleration; velocity accumulates over time
        self.acceleration = 1.5  # Force multiplier for model output (Increased for responsiveness)
        self.friction = 0.94     # Velocity retention per step (0.94 = smooth glide, less drag)
        self.max_velocity = 8.0  # Maximum cursor speed in pixels/step (Increased range)

        # --- DOPAMINE & FOCUS SYSTEM ---
        self.dopamine_level = 0.0 # 0.0 to 1.0
        self.manual_dopamine = 0.0 # From Human
        self.focus_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        # Store (ActionType, ObjectColor) -> EffectHash
        self.effect_memory: dict[Tuple[int, int], Set[str]] = {}
        
        # --- FAST DOPAMINE PREDICTOR (Online Learning) ---
        # Maps (Predicate_Hash) -> Predicted Dopamine
        # Simple tabular Q-learning for dopamine prediction
        # We use a learning rate of 0.5 for fast adaptation
        self.dopamine_predictor: dict[str, float] = {}
        self.predicted_dopamine = 0.0
        
        # --- PREDICATE & SEQUENCE MEMORY ---
        # Stores sequences of (Predicate_Hash, Action) -> Reward
        self.sequence_buffer: List[Tuple[str, int, float]] = []
        self.gold_standard_sequences: Set[str] = set() # Hashes of successful sequences
        self.knowledge_graph: dict[str, dict[int, str]] = {} # State -> Action -> Next_State
        
        # --- EVOLUTIONARY STRATEGY (Behavior Modeling) ---
        # Population of "Goal Vectors" (size 4) representing behavior style.
        # We evolve these to match the manual_dopamine signal.
        # Channel 3 will carry this Goal Vector as a spatial pattern.
        self.es_population_size = 5
        self.es_population = [np.random.randn(4) for _ in range(self.es_population_size)]
        self.es_fitness = [0.0] * self.es_population_size
        self.current_individual_idx = 0
        self.current_goal_vector = self.es_population[0]
        self.episode_dopamine_accum = 0.0
        
        # --- PICBREEDER STYLE BEHAVIOR LIBRARY ---
        # Dictionary: {Grid_Hash -> Set[Sequence_Hash]}
        # Maps a grid context to successful behaviors
        self.behavior_library: dict[str, Set[str]] = {}
        
        # Dictionary: {Sequence_Hash -> Sequence_Data}
        self.sequence_registry: dict[str, List[Tuple[str, int, float]]] = {}
        
        # Object Tracking State
        self.current_object_idx = 0

        # Observation: 64x64x6
        # Channel 0: Current Grid
        # Channel 1: Delta
        # Channel 2: Focus
        # Channel 3: Goal Pattern (ES Genome Spatialized)
        # Channel 4: Velocity X Map (128 = 0)
        # Channel 5: Velocity Y Map (128 = 0)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.grid_size, self.grid_size, 6), dtype=np.uint8
        )

        # Actions: 
        # We use a Dict space now to support continuous movement + discrete actions?
        # No, SB3 PPO needs flattening. 
        # Box(3) -> dx, dy, action_trigger
        # dx, dy in [-1, 1] -> mapped to cursor acceleration
        # action_trigger in [-1, 1] -> mapped to discrete actions
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Optimization: Precompute meshgrid for goal channel
        self._xv, self._yv = np.meshgrid(np.linspace(0, 6.28, self.grid_size), np.linspace(0, 6.28, self.grid_size))
        self._cached_goal_channel: Optional[np.ndarray] = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict[str, Any]]:
        logger.info("DEBUG: Env Reset Start")
        super().reset(seed=seed)
        # ... (Standard Reset) ...
        self.current_step = 0
        self.last_score = 0
        # NOTE: We do NOT clear state_visitation_counts here, as curiosity is lifetime-based
        self.visited_hashes.clear() # Clear per-episode uniqueness, but keep lifetime counts separate
        self.trajectory_buffer = []
        self.recent_actions = [] # Reset action history
        self.interacted_color_actions.clear()
        self.observed_color_transitions.clear()
        self.episode_object_hashes = [] # Clear episode history
        self.sequence_buffer = []
        
        # Keep libraries across resets
        # self.effect_memory.clear() 
        # self.behavior_library.clear()
        # self.dopamine_predictor.clear() # Keep prediction model!
        
        self.dopamine_level = 0.0
        self.manual_dopamine = 0.0 
        self.predicted_dopamine = 0.0
        
        self.focus_map.fill(0)
        self.locked_plan = None
        self.plan_confidence = 0.0
        
        if hasattr(self, 'interacted_objects'):
            self.interacted_objects.clear()
        self.last_grid = None
        self.detected_objects = []
        self.current_object_idx = 0
        
        # Reset cursor to center
        self.agent.cursor_x = self.grid_size // 2
        self.agent.cursor_y = self.grid_size // 2
        
        # Trigger a reset in the actual game engine
        logger.info("DEBUG: Sending RESET to Game Engine...")
        frame = self.agent.take_action(GameAction.RESET)
        logger.info("DEBUG: Game Engine Reset Complete")
        
        if not frame:
            return np.zeros((self.grid_size, self.grid_size, 6), dtype=np.uint8), {}
            
        # Update agent state (important for GUID)
        self.agent.append_frame(frame)
        self.agent.latest_detected_objects = self.detected_objects  # Share objects for visualization
        
        # Initialize Cursor to Center of the ACTUAL Grid
        if frame.frame and frame.frame[0]:
            grid = np.array(frame.frame[0])
            h, w = grid.shape
            self.agent.cursor_x = w / 2.0
            self.agent.cursor_y = h / 2.0
            # Also set velocity to 0
            self.vel_x = 0.0
            self.vel_y = 0.0
            
        self.last_score = frame.score
        
        # Reset Dopamine on Reset
        self.dopamine_level = 0.0
        self.focus_map.fill(0)
        # self.effect_memory.clear() # Don't clear knowledge!
        
        # Precompute Goal Channel for this episode
        v = self.current_goal_vector
        goal_layer = np.sin(self._xv * v[0]) + np.cos(self._yv * v[1]) + np.sin(self._xv * self._yv * v[2]) + v[3]
        g_min, g_max = goal_layer.min(), goal_layer.max()
        if g_max > g_min:
            goal_layer = (goal_layer - g_min) / (g_max - g_min) * 255.0
        else:
            goal_layer[:] = 128.0
        self._cached_goal_channel = goal_layer.astype(np.uint8)

        # Initial Object Scan
        if frame.frame and frame.frame[0]:
             grid = np.array(frame.frame[0], dtype=np.uint8)
             self.last_grid = grid
             logger.info("DEBUG: Finding Components...")
             self.detected_objects = find_connected_components(grid)
             logger.info("DEBUG: Components Found")
             s_hash = hashlib.md5(grid.tobytes()).hexdigest()
             self.visited_hashes.add(s_hash)
             self.trajectory_buffer.append(s_hash)
             
             # Update lifetime counts
             self.state_visitation_counts[s_hash] = self.state_visitation_counts.get(s_hash, 0) + 1
        
        logger.info("DEBUG: Processing Frame...")
        obs = self._process_frame(frame)
        logger.info("DEBUG: Env Reset Done")
        return obs, {}

    def step(self, action_tensor: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.current_step += 1
        
        # Parse Box Action
        # action_tensor is shape (3,)
        # dx, dy are now ACCELERATION (Force)
        ax = float(action_tensor[0])
        ay = float(action_tensor[1])
        trigger = float(action_tensor[2])
        
        # 1. Continuous Cursor Movement with Physics (Momentum)
        # The model outputs acceleration (force). Velocity accumulates.
        # This gives the cursor inertia and smooth motion.
        
        # Apply acceleration from model output
        self.vel_x += ax * self.acceleration
        self.vel_y += ay * self.acceleration
        
        # Apply friction (velocity decays toward zero when no input)
        self.vel_x *= self.friction
        self.vel_y *= self.friction
        
        # Clamp velocity to max speed
        self.vel_x = max(-self.max_velocity, min(self.max_velocity, self.vel_x))
        self.vel_y = max(-self.max_velocity, min(self.max_velocity, self.vel_y))

        # Update cursor position based on velocity
        self.agent.cursor_x += self.vel_x
        self.agent.cursor_y += self.vel_y
        
        # Bounce on walls (conserve some momentum)
        if self.agent.cursor_x < 0:
            self.agent.cursor_x = 0
            self.vel_x = -self.vel_x * 0.5
        elif self.agent.cursor_x > self.grid_size - 1:
            self.agent.cursor_x = self.grid_size - 1
            self.vel_x = -self.vel_x * 0.5
            
        if self.agent.cursor_y < 0:
            self.agent.cursor_y = 0
            self.vel_y = -self.vel_y * 0.5
        elif self.agent.cursor_y > self.grid_size - 1:
            self.agent.cursor_y = self.grid_size - 1
            self.vel_y = -self.vel_y * 0.5
        
        # Round for discrete logic
        # Use ROUND instead of FLOOR to align click with visual center
        cx_int = int(round(self.agent.cursor_x))
        cy_int = int(round(self.agent.cursor_y))
        
        # Sync Manual Dopamine from Agent
        if hasattr(self.agent, 'manual_dopamine'):
            self.manual_dopamine = self.agent.manual_dopamine
            
        # 2. Discrete Action Trigger
        # The agent controls the cursor. The click happens AT the cursor's location.
        # We no longer support "teleport clicks" (GameAction with coordinates).
        # The click action simply means "Click HERE".
        
        action_idx = -1 # No action
        if trigger > 0:
            # Map 0.0-1.0 to 0-9
            action_idx = int(trigger * 10.0)
            action_idx = min(9, max(0, action_idx))
            
        click_action = False
        
        # Mapping:
        # 0-3: CLICK (at current cursor position)
        # 4: UP (Legacy/Aux)
        # 5: DOWN
        # 6: LEFT
        # 7: RIGHT
        # 8: SPACE
        # 9: ENTER
        
        if action_idx in [0, 1, 2, 3]: # CLICK
            click_action = True
            
        # Determine Game Action
        game_action = None
        
        if click_action: 
            # Use the AGENT'S calculated integer coordinates for the click
            # The model must have steered the cursor here.
            # Enforce that the click MUST be at the cursor's location.
            # We strictly prevent the model from clicking anywhere else.
            
            # cx_int, cy_int are the ground truth cursor positions.
            game_action = GameAction.ACTION6
            game_action.set_data({"x": cx_int, "y": cy_int, "game_id": self.agent.game_id})
            
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
            # No game action, just movement
            if self.agent.frames:
                frame = self.agent.frames[-1]
            else:
                # Create dummy frame if none exist
                return np.zeros((self.grid_size, self.grid_size, 6), dtype=np.uint8), 0.0, False, False, {}

        if not frame:
             # Safe fallback
             if self.agent.frames:
                 frame = self.agent.frames[-1]
             else:
                 return np.zeros((self.grid_size, self.grid_size, 6), dtype=np.uint8), 0.0, True, False, {}

        if game_action:
            self.agent.append_frame(frame)
            self.agent.latest_detected_objects = self.detected_objects # Share objects
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
        
        # Continuous Movement Penalty (Energy Cost)
        # Small penalty for high velocity to encourage efficiency
        reward -= (abs(ax) + abs(ay)) * 0.01
        
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
        
        # 2. Intrinsic Reward: Structured Effectance & Dopamine
        current_grid = np.array(frame.frame[0], dtype=np.uint8) if (frame.frame and frame.frame[0]) else None
        
        # Decay Dopamine and Focus
        self.dopamine_level *= 0.95 # Slower decay
        self.focus_map *= 0.9 # Slower decay (persistence)
        
        if current_grid is not None:
            state_hash = hashlib.md5(current_grid.tobytes()).hexdigest()
            
            # A. Calculate Effect (What changed?)
            effect_reward = 0.0
            diff_mask = None
            grid_changed_flag = False # Initialize explicitly
            
            if self.last_grid is not None and current_grid.shape == self.last_grid.shape:
                if not np.array_equal(current_grid, self.last_grid):
                    grid_changed_flag = True
                    diff_mask = current_grid != self.last_grid
                    num_changed = np.sum(diff_mask)
                    
                    # Analyze Structure of Change
                    diff_int = diff_mask.astype(np.uint8)
                    changed_components = find_connected_components(diff_int)
                    num_components = len(changed_components)
                    
                    # STRICTER FILTER: Only reward if change is CONCISE
                    if num_components <= 2 and num_changed > 0:
                        # HIGH QUALITY CHANGE
                        effect_reward = 2.0
                        
                        # --- DOPAMINE SPIKE ---
                        self.dopamine_level = min(1.0, self.dopamine_level + 0.5)
                        
                        # Update Focus Map (Memory of Interest)
                        # ONLY light up the changed pixels (Focal)
                        for r in range(self.grid_size):
                            for c in range(self.grid_size):
                                if diff_mask[r, c]:
                                    self.focus_map[r, c] = 1.0 
                    else:
                        # NOISY CHANGE (Too many scattered pixels) -> BOREDOM
                        effect_reward = -0.5 # Punish creating noise
                        self.dopamine_level = max(0.0, self.dopamine_level - 0.2)
            
            # B. Apply Rewards
            # Include Manual Dopamine as Teacher Signal
            # Reward = Intrinsic + (Manual * Scale)
            reward += effect_reward
            if self.manual_dopamine > 0.1:
                reward += self.manual_dopamine * 100.0 # OVERWHELMING SIGNAL (was 5.0)
            
            # C. "Flow State" Bonus
            # If dopamine is high, reward actions near the Focus Map
            if self.dopamine_level > 0.3:
                # Check if cursor is in a high-focus area
                cx_int = int(max(0, min(self.grid_size - 1, self.agent.cursor_x)))
                cy_int = int(max(0, min(self.grid_size - 1, self.agent.cursor_y)))
                
                if 0 <= cy_int < self.grid_size and 0 <= cx_int < self.grid_size:
                    focus_val = self.focus_map[cy_int, cx_int]
                    if focus_val > 0.5: # Needs to be VERY focal
                        # "Sticking with it" Reward
                        reward += 2.0 * self.dopamine_level # Higher reward for focus
            
            # D. Penalty for Boring/Null Actions
            if click_action and effect_reward == 0.0:
                reward -= 1.0 # Wasting clicks
                self.dopamine_level = max(0.0, self.dopamine_level - 0.2) # Boredom
                
                # Plan Failure?
                if self.locked_plan:
                    # If we tried the locked plan and it failed, weaken confidence
                    self.plan_confidence -= 0.2
                    if self.plan_confidence <= 0:
                        self.locked_plan = None # Abandon plan
            
            elif action_idx in [4, 5, 6, 7, 8, 9] and effect_reward == 0.0:
                 # Moving cursor without clicking is neutral, but pressing buttons with no effect is bad
                 if action_idx >= 8: # Space/Enter
                     reward -= 0.5

            # E. Scientific Discovery (Rule Repetition)
            # If we did Action X on Object O and got Effect Z... and we do it AGAIN...
            # That's a huge signal that we found a mechanism.
            
            clicked_obj_hash = "none" # Default
            
            if diff_mask is not None and click_action and grid_changed_flag:
                # Identify Target (What was under cursor BEFORE change?)
                # We need self.last_grid for this
                cx_int = int(max(0, min(self.grid_size - 1, self.agent.cursor_x)))
                cy_int = int(max(0, min(self.grid_size - 1, self.agent.cursor_y)))
                
                if 0 <= cy_int < self.grid_size and 0 <= cx_int < self.grid_size:
                    
                    # Find which OBJECT we clicked on (if any)
                    clicked_obj_hash = "space"
                    current_obj_raw = None
                    
                    for obj in self.detected_objects: # Objects from PREVIOUS frame
                        if (cy_int, cx_int) in obj:
                             current_obj_raw = obj
                             sorted_pixels = sorted(obj)
                             r, c = obj[0]
                             color = self.last_grid[r, c]
                             clicked_obj_hash = hashlib.md5(f"{color}_{str(sorted_pixels)}".encode()).hexdigest()
                             break
                    
                    # Identify Effect Hash (Shape + New Color)
                    effect_vals = current_grid[diff_mask]
                    effect_hash = hashlib.md5(effect_vals.tobytes()).hexdigest()
                    
                    key = (action_idx, clicked_obj_hash)
                    
                    if key not in self.effect_memory:
                        self.effect_memory[key] = set()
                    
                    if effect_hash in self.effect_memory[key]:
                        # WE FOUND A REPEATABLE RULE!
                        reward += 5.0 # Eureka moment
                        self.dopamine_level = 1.0 # MAX DOPAMINE
                        
                        # LOCK PLAN
                        # We found something that works. Stick to it!
                        self.locked_plan = (action_idx, clicked_obj_hash)
                        self.plan_confidence = 1.0
                        
                    else:
                        self.effect_memory[key].add(effect_hash)
            
            # F. Plan Persistence Penalty
            # If we have a locked plan, and we do something else, punish heavily.
            if self.locked_plan:
                target_action, target_obj_hash = self.locked_plan
                
                # Check if we are following the plan
                is_following = False
                if action_idx == target_action:
                    # Check if we are targeting the right object type
                    # This is hard to check perfectly without full oracle, but we can check if we are clicking *an* object
                    if click_action:
                         # Check if current object matches hash?
                         # Hash changes if position changes. We need a looser "Object Type" hash (Color + Size + Shape, invariant to Pos)
                         pass 
                    is_following = True # Assume action match is enough for now
                
                if not is_following:
                    reward -= 2.0 # distracted!
            
            # G. Sequence & Trend Learning (Neuro-symbolic)
            # 1. Update Sequence Buffer
            current_predicate = clicked_obj_hash if click_action else "move"
            
            # --- FAST PREDICTOR UPDATE ---
            # Learn: Predicate -> Manual Dopamine
            # If we are doing 'current_predicate', what is the user feeling?
            lr = 0.2
            if current_predicate not in self.dopamine_predictor:
                self.dopamine_predictor[current_predicate] = 0.0
            
            # Update prediction based on current REAL manual dopamine
            self.episode_dopamine_accum += self.manual_dopamine # Accumulate for ES Fitness
            
            target = self.manual_dopamine
            old_val = self.dopamine_predictor.get(current_predicate, 0.0) 
            new_val = old_val + lr * (target - old_val)
            self.dopamine_predictor[current_predicate] = new_val
            
            # Set current prediction for visualization
            self.predicted_dopamine = new_val
            
            # Hash the current state predicate (simple obj hash for now)
            
            self.sequence_buffer.append((current_predicate, action_idx, reward))
            if len(self.sequence_buffer) > 5:
                self.sequence_buffer.pop(0)
            
            # 2. Check for Trends
            # Calculate "Dopamine Velocity"
            # If reward is increasing over the sequence, that's a trend.
            if len(self.sequence_buffer) >= 3:
                rewards = [s[2] for s in self.sequence_buffer]
                trend = np.polyfit(range(len(rewards)), rewards, 1)[0]
                
                if trend > 0.1:
                    # Positive Trend! 
                    # Boost Dopamine significantly ("You're getting warmer!")
                    self.dopamine_level = min(1.0, self.dopamine_level + 0.3)
                    
                    # OVERFIT INCENTIVE:
                    # If we are on a trend, lock the plan even harder
                    if self.locked_plan:
                        self.plan_confidence = min(1.0, self.plan_confidence + 0.2)
                        reward += 2.0 # Reward for maintaining the trend
            
            # 3. Sequence Consolidation
            # If this step was a "Scientific Discovery" (High Reward) OR High Manual Dopamine
            if reward > 4.0 or self.manual_dopamine > 0.8:
                # Create Sequence Seed
                seq_hash = hashlib.md5(str(self.sequence_buffer).encode()).hexdigest()
                self.gold_standard_sequences.add(seq_hash)
                
                # Save to Behavior Library (Picbreeder Logic)
                # Associate this sequence with the INITIAL State Hash of the sequence?
                # Simplification: Associate with current state hash
                if state_hash not in self.behavior_library:
                    self.behavior_library[state_hash] = set()
                self.behavior_library[state_hash].add(seq_hash)
                self.sequence_registry[seq_hash] = list(self.sequence_buffer)
            
            # 4. Recall Bonus (Behavior Matching)
            # If current sequence matches a gold standard, give anticipatory dopamine
            curr_seq_hash = hashlib.md5(str(self.sequence_buffer).encode()).hexdigest()
            
            # Check if this sequence is known for THIS context (Grid State)
            # This is "Context-Sensitive Recall"
            is_known_behavior = False
            if state_hash in self.behavior_library:
                if curr_seq_hash in self.behavior_library[state_hash]:
                    is_known_behavior = True
            
            if is_known_behavior:
                reward += 50.0 # "I know this trick works here!" (was 3.0)
                self.dopamine_level = min(1.0, self.dopamine_level + 0.4)
            elif curr_seq_hash in self.gold_standard_sequences:
                # Known sequence but maybe different context
                reward += 10.0 # "I remember this trick generally" (was 1.0)
                self.dopamine_level = min(1.0, self.dopamine_level + 0.2)

            # --- LIFETIME CURIOSITY (Rare State Seeking) ---
            # We track how many times we've EVER seen this state.
            # Reward is inversely proportional to visitation count.
            # R = 1 / sqrt(count) -> Diminishing returns for familiar states.
            
            visit_count = self.state_visitation_counts.get(state_hash, 0) + 1
            self.state_visitation_counts[state_hash] = visit_count
            
            curiosity_reward = 2.0 / np.sqrt(visit_count) # Reduced base curiosity in favor of Dopamine
            reward += curiosity_reward
            
            # Check for Novelty (New State in THIS episode)
            if state_hash not in self.visited_hashes:
                reward += 1.0 
                self.visited_hashes.add(state_hash)
            
            # 3. Heuristic: Object Interaction
            # REFRESH OBJECT LIST - OPTIMIZED
            # Only run connected components if the grid actually changed
            if grid_changed_flag:
                self.detected_objects = find_connected_components(current_grid)
            # Else reuse self.detected_objects from previous step
            
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
            
            grid_changed_flag = False
            if self.last_grid is not None and current_grid.shape == self.last_grid.shape:
                if not np.array_equal(current_grid, self.last_grid):
                    grid_changed_flag = True

            # ... (Rest of code)

            if grid_changed_flag and action_idx in [8, 9, 4, 5, 6, 7]: # Space, Enter, Arrows (Updated indices)
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
            if click_action and not grid_changed_flag:
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
            # Ensure Integer Indices for Array Access
            cx_int = int(max(0, min(self.grid_size - 1, self.agent.cursor_x)))
            cy_int = int(max(0, min(self.grid_size - 1, self.agent.cursor_y)))
            
            if 0 <= cy_int < self.grid_size and 0 <= cx_int < self.grid_size:
                cursor_color = current_grid[cy_int, cx_int]
            
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

        # Metrics for Visualization
        metrics = {
            "score": frame.score,
            "dopamine": self.predicted_dopamine, # Show LEARNED prediction, not heuristic
            "manual_dopamine": self.manual_dopamine,
            "plan_confidence": self.plan_confidence,
            "reward": reward
        }

        return obs, reward, terminated, truncated, metrics

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
            
        # 3. Construct Object/Cursor Mask -> NOW FOCUS MAP
        # Channel 2: Focus Map (Where interesting things happened + Objects)
        
        # We combine the "Focus Map" (Dopamine traces) with "Object Presence"
        # High value = Interesting area or Object
        
        # Scale Focus Map to 0-255
        focus_channel = (self.focus_map * 255.0).astype(np.uint8)
        
        # Add static object borders to Focus Channel so the agent still sees objects
        for obj in self.detected_objects:
            for r, c in obj:
                if r < self.grid_size and c < self.grid_size:
                    # If it's not already hot from dopamine, give it a baseline glow
                    if focus_channel[r, c] < 50:
                        focus_channel[r, c] = 50
        
        # Draw Cursor Brightly (Always Max Importance)
        # Ensure integer indices
        cx = int(max(0, min(self.grid_size - 1, self.agent.cursor_x)))
        cy = int(max(0, min(self.grid_size - 1, self.agent.cursor_y)))
        focus_channel[cy, cx] = 255
        
        # --- VELOCITY VISUALIZATION (Momentum Feedback) ---
        # Draw a "Tail" representing velocity so the CNN can sense motion
        vx = self.vel_x * 5.0 # Amplify for visibility
        vy = self.vel_y * 5.0
        
        # Draw line from (cx, cy) to (cx-vx, cy-vy)
        # Simple Bresenham or just a few points
        steps = int(max(abs(vx), abs(vy)))
        if steps > 0:
            for i in range(steps):
                alpha = (1.0 - i/steps) # Fade out
                px = int(cx - vx * (i/steps))
                py = int(cy - vy * (i/steps))
                if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                    # Use a distinct value? Or just mix into Focus?
                    # 200 intensity for trail
                    if focus_channel[py, px] < 200:
                        focus_channel[py, px] = 200
        
        # --- VISUAL CHEAT SHEET (Picbreeder Hint) ---
        # If we have a locked plan or known behavior, highlight the target!
        if self.locked_plan:
            _, target_obj_hash = self.locked_plan
            # Find the object corresponding to this hash in current detections
            # This is expensive but necessary for "Immediate Matching"
            # We need to scan current objects and hash them
            
            # Optimization: detected_objects is already computed if grid changed.
            # But we need 'current_grid' to compute color for hash.
            # 'current' channel 0 is the grid.
            
            for obj in self.detected_objects:
                if not obj: continue
                r0, c0 = obj[0]
                color = current[r0, c0]
                sorted_pixels = sorted(obj)
                # Hash must match exactly what we stored
                h = hashlib.md5(f"{color}_{str(sorted_pixels)}".encode()).hexdigest()
                
                if h == target_obj_hash:
                    # FOUND IT! Light it up!
                    for r, c in obj:
                        focus_channel[r, c] = 255 # Max Focus
                    break
            
        # --- ATTENTION GATING ---
        # If Dopamine is high, DIM everything that isn't in focus
        if self.dopamine_level > 0.5:
             # Create a mask where Focus < threshold
             dim_mask = focus_channel < 50
             
             # Dim the state channel (0)
             current[dim_mask] = (current[dim_mask] * 0.5).astype(np.uint8)
             
             # Dim the delta channel (1)
             delta[dim_mask] = (delta[dim_mask] * 0.5).astype(np.uint8)
        
        # Stack Channels
        # Shape: (H, W, 4)
        # Channel 0: State
        # Channel 1: Delta (Immediate Change)
        # Channel 2: Focus (Memory of Interest + Objects)
        # Channel 3: Goal Pattern (ES Genome)
        
        # Use cached goal channel
        if self._cached_goal_channel is None:
             # Fallback if reset wasn't called properly
             self._cached_goal_channel = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        
        goal_channel = self._cached_goal_channel
        
        # Channel 4, 5: Velocity Map (Spatially Broadcasted)
        # Map -10 to 10 -> 0 to 255 (Center 128)
        # v_norm = (v + 10) / 20 * 255
        vx_norm = np.clip((self.vel_x + 10.0) / 20.0 * 255.0, 0, 255).astype(np.uint8)
        vy_norm = np.clip((self.vel_y + 10.0) / 20.0 * 255.0, 0, 255).astype(np.uint8)
        
        vel_x_channel = np.full((self.grid_size, self.grid_size), vx_norm, dtype=np.uint8)
        vel_y_channel = np.full((self.grid_size, self.grid_size), vy_norm, dtype=np.uint8)
        
        return np.stack([current, delta, focus_channel, goal_channel, vel_x_channel, vel_y_channel], axis=-1)
