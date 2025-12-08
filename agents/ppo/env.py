import hashlib
import logging
import time
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
        self.acceleration = 0.2  # Force multiplier for model output (Decreased for 100fps)
        self.friction = 0.94     # Velocity retention per step (0.94 = smooth glide, less drag)
        self.max_velocity = 2.0  # Maximum cursor speed in pixels/step (Decreased for 100fps)

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
        
        # --- DELAYED CREDIT ASSIGNMENT BUFFER ---
        # Stores history of significant interactions for manual dopamine attribution.
        # Format: (Step, ActionIdx, ObjectHash, InvariantHash, GridChanged, Timestamp)
        self.interaction_history: List[dict] = []
        
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
        # Box(4) -> dx, dy, trigger, action_type
        # dx, dy: Cursor Acceleration [-1, 1]
        # trigger: Action Execution Strength [-1, 1] (Threshold at 0.5)
        # action_type: Continuous selection of WHICH action to take [-1, 1]
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # Optimization: Precompute meshgrid for goal channel
        self._xv, self._yv = np.meshgrid(np.linspace(0, 6.28, self.grid_size), np.linspace(0, 6.28, self.grid_size))
        self._cached_goal_channel: Optional[np.ndarray] = None
        
        # Anti-Spam State
        self.consecutive_action_steps = 0
        self.last_trigger_val = 0.0
        self.idle_streak = 0
        self.action_cooldown = 0
        self.cooldown_steps_after_action = 20 # ~200ms at 100fps

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
        self.interaction_history = [] # Clear interaction history
        
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
        self.idle_streak = 0
        self.action_cooldown = 0
        
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
        if self.action_cooldown > 0:
            self.action_cooldown -= 1
        
        # Parse Box Action (Shape 4)
        # action_tensor is now (4,)
        ax = float(action_tensor[0])
        ay = float(action_tensor[1])
        trigger = float(action_tensor[2])
        selection = float(action_tensor[3])
        
        # 1. Continuous Cursor Movement with Physics (Momentum)
        # Apply Deadzone to prevent "Drift Spam"
        if abs(ax) < 0.15: ax = 0.0
        if abs(ay) < 0.15: ay = 0.0
        
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
            
        # 2. Discrete Action Logic
        # DECOUPLED: Trigger vs Selection
        
        # A. Determine WHICH action is selected (continuous -> discrete bucket)
        # Map [-1, 1] to 0-9 uniform buckets
        # Size 2.0 / 10 = 0.2 per bucket
        # -1.0 to -0.8 -> 0
        # ...
        # 0.8 to 1.0 -> 9
        
        # Normalize to [0, 1]
        norm_selection = (selection + 1.0) / 2.0
        norm_selection = max(0.0, min(1.0, norm_selection))
        
        # 10 buckets
        action_idx = int(norm_selection * 10.0)
        action_idx = min(9, max(0, action_idx))
        
        # B. Check Trigger (Gate)
        # Threshold: 0.6 (Reduced to encourage initial exploration)
        ACT_THRESHOLD = 0.6
        
        should_act = False
        if trigger > ACT_THRESHOLD and self.action_cooldown == 0:
            should_act = True
            
        # C. Velocity Constraint (Safety)
        # Cannot click/act if moving too fast (Stabilize first)
        current_speed = np.sqrt(self.vel_x**2 + self.vel_y**2)
        if current_speed > 2.0:
            should_act = False # Override
        
        # D. Anti-Spam "Pulse" Logic
        # We want to penalize holding the button down.
        # Actions should be discrete presses.
        # CRITICAL FIX: If holding, force act to FALSE
        if (trigger > ACT_THRESHOLD) and (self.last_trigger_val > ACT_THRESHOLD):
            should_act = False # Cannot hold button to spam
            
        self.last_trigger_val = trigger
        
        # Reset effective action if not acting
        final_action_idx = action_idx if should_act else -1
        
        # Update consecutive counter
        if final_action_idx != -1:
            self.consecutive_action_steps += 1
            self.idle_streak = 0
            self.action_cooldown = self.cooldown_steps_after_action
        else:
            if self.action_cooldown == 0:
                self.consecutive_action_steps = 0
            self.idle_streak += 1
            
        # Mapping:
        # 0-3: CLICK (at current cursor position)
        # 4: UP
        # 5: DOWN
        # 6: LEFT
        # 7: RIGHT
        # 8: SPACE
        # 9: ENTER
        
        click_action = False
        if final_action_idx != -1:
             if final_action_idx <= 3:
                 click_action = True

        # Determine Game Action
        game_action = None
        
        if click_action: 
            # Use the AGENT'S calculated integer coordinates for the click
            game_action = GameAction.ACTION6
            game_action.set_data({"x": cx_int, "y": cy_int, "game_id": self.agent.game_id})
            
        elif final_action_idx == 4: # G-UP
            game_action = GameAction.ACTION1
        elif final_action_idx == 5: # G-DOWN
            game_action = GameAction.ACTION2
        elif final_action_idx == 6: # G-LEFT
            game_action = GameAction.ACTION3
        elif final_action_idx == 7: # G-RIGHT
            game_action = GameAction.ACTION4
        elif final_action_idx == 8: # SPACE
            game_action = GameAction.ACTION5
        elif final_action_idx == 9: # ENTER
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

        # --- HEURISTIC REWARD CALCULATION ---
        reward = 0.0
        
        if game_action:
            self.agent.append_frame(frame)
            self.agent.latest_detected_objects = self.detected_objects # Share objects
            
            # Add symbols for visualization
            action_name = game_action.name
            if final_action_idx == 4: action_name = "UP ↑"
            elif final_action_idx == 5: action_name = "DOWN ↓"
            elif final_action_idx == 6: action_name = "LEFT ←"
            elif final_action_idx == 7: action_name = "RIGHT →"
            elif final_action_idx == 8: action_name = "SPACE ␣"
            elif final_action_idx == 9: action_name = "ENTER ↵"
            elif click_action: action_name = "CLICK 🖱️"
            
            self.agent._last_action_viz = {
                "id": game_action.value,
                "name": action_name,
                "data": game_action.action_data.model_dump()
            }
        else:
            status = "Cursor Move"
            if self.action_cooldown > 0:
                status = "Cooldown..."
                # Add penalty for spamming while in cooldown
                if trigger > ACT_THRESHOLD:
                    reward -= 1.0 # Use accumulated reward (init at 0.0 above)
                    self.action_cooldown += 2 # Add more delay to punish spamming
            
            elif trigger > 0.2: 
                status = f"Aiming ({action_idx})..."
            
            self.agent._last_action_viz = {
                "name": status,
                "data": {"x": self.agent.cursor_x, "y": self.agent.cursor_y}
            }

        obs = self._process_frame(frame)
        
        sparsity_multiplier = 1.0 / (1.0 + self.consecutive_action_steps)
        
        # Continuous Movement Penalty (Energy Cost)
        reward -= (abs(ax) + abs(ay)) * 0.01
        
        # TENSION / URGE PENALTY (Bias towards Idle)
        # Even if not acting, high trigger value is "stressful" and costs energy.
        # This forces the agent to keep trigger negative (relaxed) when not using it.
        reward -= max(0.0, trigger) * 0.5
        
        # if cooldown_trigger_tension:
        #    reward -= 2.0
        #    self.dopamine_level = max(0.0, self.dopamine_level - 0.2)
        
        # ACTION COSTS & PENALTIES
        if final_action_idx != -1:
             # Base Sparsity Cost: "It costs energy to act."
             # Reduced from 5.0 to 2.0 to make acting less prohibitive
             reward -= 2.0
             
             # Consecutive Action Penalty (Anti-Spam/Machine Gun)
             # We want discrete taps, not holding the button.
             if self.consecutive_action_steps > 1:
                 reward -= 10.0 * self.consecutive_action_steps # Scaling punishment
                 # Kill dopamine if spamming
                 self.dopamine_level = 0.0
        else:
            # Idle Reward (Small relaxation bonus)
            # If we are effectively idling (trigger low), give a small drip of reward.
            # This makes "doing nothing" better than "doing something useless".
            if trigger < 0.0:
                idle_bonus = 0.1 + min(1.0, self.idle_streak * 0.002)
                reward += idle_bonus
        
        # 0. Time/Step Loss (Linear Accumulation)
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
            
            # --- IDENTIFY CLICKED OBJECT (Moved Early) ---
            clicked_obj_hash = "none"
            clicked_obj_invariant = "none"
            
            if click_action:
                # Find which OBJECT we clicked on (if any)
                # We use self.detected_objects from the PREVIOUS step (what was there before click)
                cx_int = int(max(0, min(self.grid_size - 1, self.agent.cursor_x)))
                cy_int = int(max(0, min(self.grid_size - 1, self.agent.cursor_y)))
                
                if 0 <= cy_int < self.grid_size and 0 <= cx_int < self.grid_size:
                    for obj in self.detected_objects:
                        if (cy_int, cx_int) in obj:
                             sorted_pixels = sorted(obj)
                             # We need the color from the LAST grid (before change)
                             if self.last_grid is not None:
                                 r, c = obj[0]
                                 color = self.last_grid[r, c]
                                 # 1. Instance Hash (Exact position)
                                 clicked_obj_hash = hashlib.md5(f"{color}_{str(sorted_pixels)}".encode()).hexdigest()
                                 
                                 # 2. Invariant Hash (Generalization: Color + Normalized Shape)
                                 # Normalize coordinates to top-left of object
                                 min_r = min(p[0] for p in obj)
                                 min_c = min(p[1] for p in obj)
                                 norm_pixels = sorted([(p[0]-min_r, p[1]-min_c) for p in obj])
                                 clicked_obj_invariant = hashlib.md5(f"{color}_{str(norm_pixels)}".encode()).hexdigest()
                             break

            # --- UPDATE INTERACTION HISTORY ---
            # Record any significant action (Click or Game Action)
            # We want to be able to look back and say "Ah, THAT was the good action"
            if click_action or action_idx >= 4:
                self.interaction_history.append({
                    "step": self.current_step,
                    "action": action_idx,
                    "object_hash": clicked_obj_hash,
                    "invariant_hash": clicked_obj_invariant,
                    "grid_changed": grid_changed_flag,
                    "time": time.time()
                })
                # Keep buffer manageable
                if len(self.interaction_history) > 20:
                    self.interaction_history.pop(0)

            # REFACTOR: If we get manual dopamine, LOCK IN on this behavior.
            # The human likes this!
            # Initialize target variables to None before checking
            target_action = None
            target_obj = None
            target_invariant = None
            
            if self.manual_dopamine > 0.1:
                # HUGE reward for satisfying the human
                reward += self.manual_dopamine * 50.0 
                
                # --- DELAYED CREDIT ASSIGNMENT (SMARTER) ---
                # Search backwards for the "Cause" of this dopamine.
                # We use an eligibility trace with time decay.
                
                best_match = None
                best_match_score = -1.0
                
                # We look back up to 20 steps (approx 2-5 seconds)
                for i, event in enumerate(reversed(self.interaction_history)):
                    # Decay factor based on how long ago it happened
                    # (Recent events are more likely the cause, but reaction time implies a slight delay)
                    steps_ago = self.current_step - event["step"]
                    
                    # Human reaction time heuristic: Peak relevance is ~0.5s - 1.0s ago (approx 2-5 steps)
                    # We use a skewed bell curve or just simple exponential decay from a delayed peak?
                    # Let's use simple decay for now, but prioritize "Change" events.
                    
                    relevance = 1.0 / (1.0 + steps_ago * 0.2)
                    
                    score = 0.0
                    if event["grid_changed"]:
                        score += 5.0 # Highest priority: It actually DID something
                    elif event["object_hash"] != "none":
                        score += 2.0 # Medium priority: It touched something
                    else:
                        score += 0.5 # Low priority: Just an action
                        
                    final_score = score * relevance
                    
                    if final_score > best_match_score:
                        best_match_score = final_score
                        best_match = event
                
                # If no specific event found, use current (fallback)
                if best_match:
                    target_action = best_match["action"]
                    target_obj = best_match["object_hash"]
                    target_invariant = best_match.get("invariant_hash", "none")
                elif click_action and clicked_obj_hash != "none":
                    target_action = action_idx
                    target_obj = clicked_obj_hash
                    target_invariant = clicked_obj_invariant
                elif final_action_idx == -1:
                    # User rewarded SILENCE/IDLE.
                    target_action = -1
                    target_obj = "none"
                    target_invariant = "none"
                
                if target_action is not None:
                     # Use Invariant Hash if available for generalization
                     # This effectively says "Clicking RED SQUARES is good", not just "Clicking THIS pixel"
                     lock_target = target_invariant if target_invariant != "none" else target_obj
                     
                     # Allow locking to "none" for Idle
                     if lock_target != "none" or target_action == -1:
                         # INTRINSICALLY CONDITIONED REWARD
                         # If we are locked in, and we repeat the action, give INTERNAL reward.
                         
                         self.locked_plan = (target_action, lock_target)
                         self.plan_confidence = 1.0
                         # Boost dopamine level to sustain focus
                         self.dopamine_level = 1.0
                         
                         # REINFORCE THE PAST EVENT
                         # Store value on INVARIANT hash to allow generalization
                         if lock_target in self.valuable_object_hashes:
                             self.valuable_object_hashes[lock_target] += 5.0 # Mark object TYPE as VERY valuable
                         else:
                             self.valuable_object_hashes[lock_target] = 5.0

            
            # C. "Flow State" Bonus
            # If dopamine is high, reward actions near the Focus Map
            if self.dopamine_level > 0.3:
                # Check if cursor is in a high-focus area
                cx_int = int(max(0, min(self.grid_size - 1, self.agent.cursor_x)))
                cy_int = int(max(0, min(self.grid_size - 1, self.agent.cursor_y)))
                
                # REFACTOR: "Deliberate Action" Reward
                # If the agent is in a high dopamine state (excited/focused) AND it takes a valid action...
                # ...we reward it for "Following Through".
                # But we PUNISH it heavily for "Breaking Flow" (clicking elsewhere).
                
                if 0 <= cy_int < self.grid_size and 0 <= cx_int < self.grid_size:
                    focus_val = self.focus_map[cy_int, cx_int]
                    
                    if click_action:
                        if focus_val > 0.5: 
                            # "You clicked where you were looking!" -> GOOD
                            reward += 5.0 
                        else:
                            # "You clicked random noise while excited about something else!" -> BAD
                            reward -= 5.0 # Stop being distracted!
                            self.dopamine_level *= 0.5 # Lose focus immediately
                    elif focus_val > 0.5:
                         # Just hovering over interesting stuff is good
                         reward += 0.5 * self.dopamine_level
            
            # D. Penalty for Boring/Null Actions
            # INTRINSIC PENALTY FOR SPAM / INEFFECTIVE ACTIONS
            # "Energy Cost" for action vs "Value" of outcome.
            
            if click_action:
                if effect_reward <= 0.0:
                    # Action had NO positive effect.
                    # Was it just a "miss" or "spam"?
                    
                    # 1. Dynamic Spam Penalty (Ratio based)
                    # Instead of hardcoded "3 in a row", we track local effectiveness.
                    if len(self.recent_actions) > 5:
                        recent_clicks = [a for a in self.recent_actions[-10:] if a <= 3] # Count clicks
                        if len(recent_clicks) > 5:
                             # If we are clicking a lot...
                             # We need to know if we are achieving anything.
                             # This requires tracking recent *successes* which we don't have easily in this block.
                             # Fallback to local repetition check but slightly looser.
                             
                             if self.recent_actions[-1] == action_idx and self.recent_actions[-2] == action_idx:
                                  # STILL PUNISH MANIC REPETITION (Even 2 in a row if ineffective is bad)
                                  reward -= 5.0 
                                  self.dopamine_level = 0.0
                    
                    reward -= 2.0 # Standard "Waste of Energy" penalty
                
                # Plan Failure?
                if self.locked_plan and effect_reward <= 0.0:
                     # If we tried the locked plan and it failed, weaken confidence
                    self.plan_confidence -= 0.1 # Slower decay
                    if self.plan_confidence <= 0:
                        self.locked_plan = None # Abandon plan

            elif action_idx in [4, 5, 6, 7, 8, 9] and effect_reward == 0.0:
                 # Moving cursor without clicking is neutral-ish (energy cost handled above),
                 # but pressing buttons (Space/Enter) with no effect is bad.
                 if action_idx >= 8: # Space/Enter
                     reward -= 2.0
            
            # REFACTOR: Remove hardcoded repetition check below in favor of above logic?
            # Keeping the "loop" check as a fail-safe.
            
            # E. Scientific Discovery (Rule Repetition)
            # If we did Action X on Object O and got Effect Z... and we do it AGAIN...
            # That's a huge signal that we found a mechanism.
            
            # clicked_obj_hash was calculated early above
            
            if diff_mask is not None and click_action and grid_changed_flag:
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
                target_action, target_invariant_hash = self.locked_plan
                
                # Check if we are following the plan
                is_following = False
                
                # Allow minor variations in action type (e.g. Action 0 vs 1 if they are both clicks)
                is_click_plan = target_action >= 0 and target_action <= 3
                is_click_action = action_idx >= 0 and action_idx <= 3
                
                action_match = False
                if target_action == -1:
                    # Plan is to IDLE
                    if final_action_idx == -1:
                        action_match = True
                else:
                    # Plan is an ACTION
                    # We only match if we ACTUALLY ACTED (final_action_idx != -1)
                    if final_action_idx != -1:
                         # Check if indices match (or generalized click match)
                         if action_idx == target_action:
                             action_match = True
                         elif is_click_plan and is_click_action:
                             action_match = True
                
                if action_match:
                    if is_click_action and target_action != -1:
                         # Check object match using INVARIANT hash
                         # We calculated clicked_obj_invariant earlier
                         # If the plan was locked to "none" (empty space), check that too
                         if clicked_obj_invariant == target_invariant_hash:
                             is_following = True
                         elif clicked_obj_hash == target_invariant_hash: # Fallback to instance hash
                             is_following = True
                    else:
                        is_following = True
                
                if is_following:
                     # REWARD FOR REPETITION (The "Quick Learner" Reward)
                     # If we are following the locked plan, give a small internal boost
                     # This effectively makes the agent "want" to repeat the successful behavior
                     reward += 2.0 
                     self.dopamine_level = min(1.0, self.dopamine_level + 0.1) # Sustain focus
                else:
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
            
            curiosity_reward = 0.5 / np.sqrt(visit_count) # Reduced base curiosity in favor of Dopamine/Sparsity
            reward += curiosity_reward * sparsity_multiplier
            
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
                reward -= 5.0 # Increased penalty for clicking nothing (Total -7.0 with sparsity)
            
            # Reduce penalty for non-click actions to encourage exploration
            elif final_action_idx in [4, 5, 6, 7, 8, 9] and not grid_changed_flag:
                 # Huge refund on sparsity cost for trying new buttons
                 # We want the agent to try these at least once!
                 # Sparsity is -2.0. Refund +1.8 = Net -0.2 (almost free)
                 reward += 1.8 
            
            # --- STATE-ACTION CURIOSITY (Conditional Branching) ---
            # "Unlock novel paths": Try every action at least once from every state.
            # "Bias towards good attempts": If we are in a rare/good state, exploring from it is highly rewarded.
            
            # Use final_action_idx (only reward executed actions)
            effective_sa_idx = final_action_idx
            if final_action_idx <= 3 and final_action_idx != -1:
                effective_sa_idx = 99 # Unify clicks
            
            sa_pair = (state_hash, effective_sa_idx)
            sa_count = self.state_action_counts.get(sa_pair, 0)
            self.state_action_counts[sa_pair] = sa_count + 1
            
            # Reward decreases as we repeat the same action in the same state.
            # R = 5.0 / (1 + count) (Balanced: 5.0 - 2.0 = +3.0 initial profit)
            # 1st time: +5.0
            # 2nd time: +2.5 (Net +0.5)
            # 3rd time: +1.6 (Net -0.4) -> Stops repeating here
            # ...
            # This forces the agent to "branch out" (try other actions) before repeating.
            sa_reward = 5.0 / (1.0 + sa_count)
            
            # Bias: If this state itself is RARE (a "good attempt" or deep state), boost the exploration reward.
            # We want to explore thoroughly when we find something new.
            state_rarity_boost = 1.0
            if visit_count < 5: # We've been here fewer than 5 times
                state_rarity_boost = 2.0
            
            # Only give curiosity reward if we actually DID something (or if idle is rare)
            # If we are just idling, we don't need a huge boost, but it's okay to reward staying still if it's new.
            reward += sa_reward * state_rarity_boost * sparsity_multiplier

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
            effective_action = final_action_idx
            if final_action_idx <= 3 and final_action_idx != -1: 
                effective_action = 99 # Special ID for "Any Click"
                
            action_color_pair = (effective_action, cursor_color)
            
            if final_action_idx != -1 and action_color_pair not in self.interacted_color_actions:
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
            "reward": reward,
            "trigger": float(trigger) # Show the raw urge/tension
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
             # Calculate Invariant Hash for this object
            r0, c0 = obj[0]
            color = current[r0, c0]
            min_r = min(p[0] for p in obj)
            min_c = min(p[1] for p in obj)
            norm_pixels = sorted([(p[0]-min_r, p[1]-min_c) for p in obj])
            h_inv = hashlib.md5(f"{color}_{str(norm_pixels)}".encode()).hexdigest()
            
            is_valuable = False
            if h_inv in self.valuable_object_hashes:
                 # If value > threshold, treat as high focus
                 if self.valuable_object_hashes[h_inv] > 1.0:
                     is_valuable = True
            
            base_intensity = 150 if is_valuable else 50
            
            for r, c in obj:
                if r < self.grid_size and c < self.grid_size:
                    # If it's not already hot from dopamine, give it a baseline glow
                    # If valuable, give it a STRONGER glow
                    if focus_channel[r, c] < base_intensity:
                        focus_channel[r, c] = base_intensity
        
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
                _, target_hash = self.locked_plan
                
                # Check current objects for match
                # Use INVARIANT hash matching if possible
                
                # We need to re-scan current objects for invariance
                # Optimization: do this as part of the focus update logic?
                # For now, let's do it here.
                
                # `current` is defined above as current_grid's shorter name, but `current_grid` variable name is not present in scope.
                # `current` holds the extracted grid data.
                if current is not None:
                     # Re-find components for current frame if not already (process_frame called after update)
                     # But detected_objects is updated in step().
                     
                     for obj in self.detected_objects:
                        if not obj: continue
                        r0, c0 = obj[0]
                        color = current[r0, c0]
                        
                        # 1. Strict Hash
                        sorted_pixels = sorted(obj)
                        h_strict = hashlib.md5(f"{color}_{str(sorted_pixels)}".encode()).hexdigest()
                        
                        # 2. Invariant Hash
                        min_r = min(p[0] for p in obj)
                        min_c = min(p[1] for p in obj)
                        norm_pixels = sorted([(p[0]-min_r, p[1]-min_c) for p in obj])
                        h_inv = hashlib.md5(f"{color}_{str(norm_pixels)}".encode()).hexdigest()
                        
                        # Match EITHER
                        if h_strict == target_hash or h_inv == target_hash:
                             # FOUND IT! Light it up!
                            for r, c in obj:
                                focus_channel[r, c] = 255 # Max Focus

            
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
