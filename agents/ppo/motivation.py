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
    """
    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        
        # Dopamine & Focus
        self.dopamine_level = 0.0
        self.focus_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.manual_dopamine = 0.0
        
        # Curiosity & State Tracking
        self.visited_hashes: Set[str] = set()
        self.state_visitation_counts: Dict[str, int] = {}
        self.state_action_counts: Dict[Tuple[str, int], int] = {}
        self.interacted_color_actions: Set[Tuple[int, int]] = set()
        self.observed_color_transitions: Set[Tuple[int, int]] = set()
        
        # Rule Learning & Planning
        self.effect_memory: Dict[Tuple[int, int], Set[str]] = {}
        self.sequence_buffer: List[Tuple[str, int, float]] = []
        self.gold_standard_sequences: Set[str] = set()
        self.behavior_library: Dict[str, Set[str]] = {}
        self.sequence_registry: Dict[str, List[Tuple[str, int, float]]] = {}
        self.interaction_history: List[dict] = []
        
        # Prediction
        self.dopamine_predictor: Dict[str, float] = {}
        self.predicted_dopamine = 0.0
        self.episode_dopamine_accum = 0.0
        
        # Planning
        self.locked_plan: Optional[Tuple[int, str]] = None
        self.plan_confidence = 0.0
        self.recent_actions: List[int] = []

    def reset(self):
        self.dopamine_level = 0.0
        self.manual_dopamine = 0.0
        self.predicted_dopamine = 0.0
        self.focus_map.fill(0)
        
        self.visited_hashes.clear()
        self.sequence_buffer = []
        self.interaction_history = []
        self.recent_actions = []
        self.interacted_color_actions.clear()
        self.observed_color_transitions.clear()
        
        self.locked_plan = None
        self.plan_confidence = 0.0
        self.episode_dopamine_accum = 0.0

    def update_dopamine(self, amount: float):
        self.dopamine_level = max(0.0, min(1.0, self.dopamine_level + amount))
    
    def decay(self):
        self.dopamine_level *= 0.95
        self.focus_map *= 0.9

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
        
        # --- 1. Extrinsic (Score) handled in Env ---
        
        # --- 2. Intrinsic: Effectance & Dopamine ---
        state_hash = hashlib.md5(current_grid.tobytes()).hexdigest()
        
        effect_reward = 0.0
        diff_mask = None
        
        if last_grid is not None and current_grid.shape == last_grid.shape:
            if grid_changed:
                diff_mask = current_grid != last_grid
                num_changed = np.sum(diff_mask)
                
                # Analyze structure of change
                diff_int = diff_mask.astype(np.uint8)
                changed_components = find_connected_components(diff_int)
                num_components = len(changed_components)
                
                # Concise change reward
                if num_components <= 2 and num_changed > 0:
                    effect_reward = 2.0
                    self.update_dopamine(0.5)
                    # Update Focus Map
                    for r in range(self.grid_size):
                        for c in range(self.grid_size):
                            if diff_mask[r, c]:
                                self.focus_map[r, c] = 1.0
                else:
                    effect_reward = -0.5
                    self.update_dopamine(-0.2)

        # --- 3. Identify Clicked Object ---
        clicked_obj_hash = "none"
        clicked_obj_invariant = "none"
        
        cx_int = int(round(env.agent.cursor_x))
        cy_int = int(round(env.agent.cursor_y))
        cx_int = max(0, min(self.grid_size - 1, cx_int))
        cy_int = max(0, min(self.grid_size - 1, cy_int))
        
        if click_action and last_grid is not None:
             # Look at PREVIOUS detected objects (before change)
             for obj in env.object_tracker.detected_objects:
                 if (cy_int, cx_int) in obj:
                     r, c = obj[0]
                     color = last_grid[r, c]
                     sorted_pixels = sorted(obj)
                     clicked_obj_hash = hashlib.md5(f"{color}_{str(sorted_pixels)}".encode()).hexdigest()
                     
                     clicked_obj_invariant = env.object_tracker.get_invariant_hash(obj, color)
                     break
        
        # --- 4. Update Interaction History ---
        if click_action or action_idx >= 4:
            self.interaction_history.append({
                "step": env.current_step,
                "action": action_idx,
                "object_hash": clicked_obj_hash,
                "invariant_hash": clicked_obj_invariant,
                "grid_changed": grid_changed,
                "time": time.time()
            })
            if len(self.interaction_history) > 20:
                self.interaction_history.pop(0)

        # --- 5. Manual Dopamine & Plan Locking ---
        if self.manual_dopamine > 0.1:
            reward += self.manual_dopamine * 50.0
            
            # Delayed Credit Assignment
            best_match = None
            best_match_score = -1.0
            
            for i, event in enumerate(reversed(self.interaction_history)):
                steps_ago = env.current_step - event["step"]
                relevance = 1.0 / (1.0 + steps_ago * 0.2)
                
                score = 0.0
                if event["grid_changed"]: score += 5.0
                elif event["object_hash"] != "none": score += 2.0
                else: score += 0.5
                
                final_score = score * relevance
                if final_score > best_match_score:
                    best_match_score = final_score
                    best_match = event
            
            target_action = None
            target_obj = None
            target_invariant = None
            
            if best_match:
                target_action = best_match["action"]
                target_obj = best_match["object_hash"]
                target_invariant = best_match.get("invariant_hash", "none")
            elif click_action and clicked_obj_hash != "none":
                target_action = action_idx
                target_obj = clicked_obj_hash
                target_invariant = clicked_obj_invariant
            elif action_idx == -1:
                target_action = -1
                target_obj = "none"
                target_invariant = "none"
                
            if target_action is not None:
                lock_target = target_invariant if target_invariant != "none" else target_obj
                
                if lock_target != "none" or target_action == -1:
                    self.locked_plan = (target_action, lock_target)
                    self.plan_confidence = 1.0
                    self.dopamine_level = 1.0
                    
                    if lock_target in env.object_tracker.valuable_object_hashes:
                        env.object_tracker.valuable_object_hashes[lock_target] += 5.0
                    else:
                        env.object_tracker.valuable_object_hashes[lock_target] = 5.0

        # --- 6. Flow State / Plan Persistence ---
        if self.locked_plan:
            t_action, t_inv = self.locked_plan
            is_following = False
            
            is_click_plan = t_action >= 0 and t_action <= 3
            is_click_action = action_idx >= 0 and action_idx <= 3
            
            action_match = False
            if t_action == -1:
                if action_idx == -1: action_match = True
            elif action_idx != -1:
                if action_idx == t_action: action_match = True
                elif is_click_plan and is_click_action: action_match = True
            
            if action_match:
                if is_click_action and t_action != -1:
                    if clicked_obj_invariant == t_inv or clicked_obj_hash == t_inv:
                        is_following = True
                else:
                    is_following = True
            
            if is_following:
                reward += 2.0
                self.update_dopamine(0.1)
            else:
                reward -= 2.0

        # --- 7. Sequence Learning & Prediction ---
        current_predicate = clicked_obj_hash if click_action else "move"
        
        # Fast Predictor
        lr = 0.2
        if current_predicate not in self.dopamine_predictor:
            self.dopamine_predictor[current_predicate] = 0.0
        
        self.episode_dopamine_accum += self.manual_dopamine
        old_val = self.dopamine_predictor.get(current_predicate, 0.0)
        new_val = old_val + lr * (self.manual_dopamine - old_val)
        self.dopamine_predictor[current_predicate] = new_val
        self.predicted_dopamine = new_val
        
        self.sequence_buffer.append((current_predicate, action_idx, reward))
        if len(self.sequence_buffer) > 5:
            self.sequence_buffer.pop(0)
            
        # --- 8. Lifetime Curiosity ---
        visit_count = self.state_visitation_counts.get(state_hash, 0) + 1
        self.state_visitation_counts[state_hash] = visit_count
        curiosity_reward = 0.5 / np.sqrt(visit_count)
        reward += curiosity_reward * sparsity_multiplier
        
        if state_hash not in self.visited_hashes:
            reward += 1.0
            self.visited_hashes.add(state_hash)
            
        # --- 9. Object Guidance ---
        guidance_reward = 0.0
        current_step_obj_hashes = env.object_tracker.get_object_hashes(current_grid)
        for h in current_step_obj_hashes:
            if h in env.object_tracker.valuable_object_hashes:
                guidance_reward += env.object_tracker.valuable_object_hashes[h]
        
        env.object_tracker.episode_object_hashes.append(current_step_obj_hashes)
        reward += guidance_reward * 0.1
        
        # --- 10. State-Action Curiosity ---
        effective_sa_idx = action_idx
        if click_action and action_idx != -1: effective_sa_idx = 99
        
        sa_pair = (state_hash, effective_sa_idx)
        sa_count = self.state_action_counts.get(sa_pair, 0)
        self.state_action_counts[sa_pair] = sa_count + 1
        
        sa_reward = 5.0 / (1.0 + sa_count)
        state_rarity_boost = 2.0 if visit_count < 5 else 1.0
        reward += sa_reward * state_rarity_boost * sparsity_multiplier
        
        # --- 11. Color Interaction Curiosity ---
        if action_idx != -1:
            cursor_color = 0
            if 0 <= cy_int < self.grid_size and 0 <= cx_int < self.grid_size:
                cursor_color = current_grid[cy_int, cx_int]
            
            effective_action = 99 if click_action else action_idx
            action_color_pair = (effective_action, cursor_color)
            
            if action_color_pair not in self.interacted_color_actions:
                reward += 1.5
                self.interacted_color_actions.add(action_color_pair)

        # Plan Failure check
        if self.locked_plan and click_action and effect_reward <= 0.0:
            self.plan_confidence -= 0.1
            if self.plan_confidence <= 0:
                self.locked_plan = None
        
        return reward

