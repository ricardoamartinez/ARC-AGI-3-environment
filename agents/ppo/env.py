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
        
        # ES Population
        self.es_population_size = 5
        self.es_population = [np.random.randn(4) for _ in range(self.es_population_size)]
        self.current_goal_vector = self.es_population[0]
        
        self.observation_space = self.obs_builder.observation_space
        self.action_space = spaces.Tuple((
            spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            spaces.Discrete(10)
        ))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict[str, Any]]:
        logger.info("DEBUG: Env Reset Start")
        super().reset(seed=seed)
        
        self.current_step = 0
        self.last_score = 0
        self.last_grid = None
        
        self.physics.reset()
        self.action_processor.reset()
        self.object_tracker.reset()
        self.intrinsic_system.reset()
        
        self.agent.cursor_x = self.grid_size // 2
        self.agent.cursor_y = self.grid_size // 2
        
        # Ensure we actually start the game.
        # If the game was just selected, it might be in 'NOT_STARTED' state.
        # Sending RESET should fix this.
        frame = None
        for attempt in range(5):
            frame = self.agent.take_action(GameAction.RESET)
            if frame:
                break
            logger.warning(f"Reset attempt {attempt+1} failed, retrying in 2s...")
            time.sleep(2)
        
        if not frame:
            logger.error("Failed to reset game after multiple attempts.")
            raise RuntimeError("Failed to reset game - API Error or Network Issue")
            
        self.agent.append_frame(frame)
        self.last_score = frame.score
        
        if frame.frame and frame.frame[0]:
            grid = np.array(frame.frame[0], dtype=np.uint8)
            h, w = grid.shape
            self.agent.cursor_x = w / 2.0
            self.agent.cursor_y = h / 2.0
            self.last_grid = grid
            self.object_tracker.scan(grid)
            
            s_hash = hashlib.md5(grid.tobytes()).hexdigest()
            self.intrinsic_system.visited_hashes.add(s_hash)
            
        self.obs_builder.precompute_goal_channel(self.current_goal_vector)
        
        current_grid = self.last_grid if self.last_grid is not None else np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        
        obs = self._get_obs(current_grid, None, -1, 0.0)
        logger.info("DEBUG: Env Reset Done")
        return obs, {}

    def step(self, action: Tuple[np.ndarray, int]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.current_step += 1
        
        cont_actions, disc_idx = action
        
        prev_x = self.agent.cursor_x
        prev_y = self.agent.cursor_y
        
        curr_speed = self.physics.get_speed()
        final_action_idx, ax, ay, trigger, _ = self.action_processor.process(cont_actions, disc_idx, curr_speed)
        
        self.agent.cursor_x, self.agent.cursor_y, hit_wall = self.physics.update(
            ax, ay, self.agent.cursor_x, self.agent.cursor_y
        )
        
        reward = 0.0
        
        if hasattr(self.agent, 'manual_dopamine'):
            self.intrinsic_system.manual_dopamine = self.agent.manual_dopamine
        if hasattr(self.agent, 'manual_pain'):
            self.intrinsic_system.manual_pain = self.agent.manual_pain
            
        # Frustration & Wall Impact
        effort = np.sqrt(ax**2 + ay**2)
        
        # Immediate Wall Penalty
        if hit_wall:
            reward -= 0.05 # Immediate "Ouch" for hitting the wall (Drastically reduced)
            
            # If trying to move into it, that's frustration.
            if effort > 0.1: # Lowered threshold to catch even subtle pushes
                 cx = int(round(self.agent.cursor_x))
                 cy = int(round(self.agent.cursor_y))
                 if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
                     self.intrinsic_system.pain_memory[cy, cx] = 1.0
                     # Spread pain to neighbors to create a "hot zone" at the wall
                     for dy in [-1, 0, 1]:
                         for dx in [-1, 0, 1]:
                             if 0 <= cy+dy < self.grid_size and 0 <= cx+dx < self.grid_size:
                                 self.intrinsic_system.pain_memory[cy+dy, cx+dx] = max(self.intrinsic_system.pain_memory[cy+dy, cx+dx], 0.8)

                 # Ramp up internal pain state (Slowed down frustration buildup)
                 self.intrinsic_system.manual_pain += 0.05
                 self.intrinsic_system.manual_pain = min(1.0, self.intrinsic_system.manual_pain)
                 self.intrinsic_system.current_thought = "Frustration! Stuck at wall."
        
        # Pain Repulsion Reward (Active Avoidance)
        # If moving AWAY from high pain, give reward?
        # Or punish being IN pain.
        cx_int = int(round(self.agent.cursor_x))
        cy_int = int(round(self.agent.cursor_y))
        
        current_pain_val = self.intrinsic_system.pain_memory[cy_int, cx_int] if (0 <= cx_int < self.grid_size and 0 <= cy_int < self.grid_size) else 0.0
        
        # Scale pain by manual intensity if active
        if hasattr(self.agent, 'manual_pain') and self.agent.manual_pain > 0:
             current_pain_val = max(current_pain_val, self.agent.manual_pain)

        if current_pain_val > 0.1:
            reward -= current_pain_val * 0.1 # Stronger punishment for sitting in pain (Reduced from 0.5)
            
        # Repulsion Check: Did we move away from higher pain?
        prev_cx_int = int(round(prev_x))
        prev_cy_int = int(round(prev_y))
        if 0 <= prev_cx_int < self.grid_size and 0 <= prev_cy_int < self.grid_size:
            prev_pain = self.intrinsic_system.pain_memory[prev_cy_int, prev_cx_int]
            if current_pain_val < prev_pain:
                reward += 0.05 # Reward escaping pain (Reduced from 0.2)
                self.intrinsic_system.current_thought = "Escaping pain!"

        cx_int = int(round(self.agent.cursor_x))
        cy_int = int(round(self.agent.cursor_y))
        game_action = self.action_processor.get_game_action(final_action_idx, cx_int, cy_int, self.agent.game_id)
        
        frame = None
        if game_action:
            frame = self.agent.take_action(game_action)
            if frame:
                self.agent.append_frame(frame)
                self.agent.latest_detected_objects = self.object_tracker.detected_objects
                
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
                logger.warning("Action failed (returned None).")
        else:
            if self.agent.frames:
                 frame = self.agent.frames[-1]
            else:
                 return np.zeros((self.grid_size, self.grid_size, 8), dtype=np.uint8), 0.0, False, False, {}

            status = "Cursor Move"
            if trigger > 0.2:
                status = f"Aiming..."
            
            self.agent._last_action_viz = {
                "name": status,
                "data": {"x": self.agent.cursor_x, "y": self.agent.cursor_y}
            }
        
        if not frame:
            if self.agent.frames: frame = self.agent.frames[-1]
            else: return np.zeros((self.grid_size, self.grid_size, 8), dtype=np.uint8), 0.0, True, False, {}

        current_grid = np.array(frame.frame[0], dtype=np.uint8) if (frame.frame and frame.frame[0]) else None
        if current_grid is None:
             current_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # REMOVED ENERGY COST to prevent "Lazy Agent" syndrome
        # Only punish extreme jitter if needed, but for now let it move freely.
        
        # Methodical Motion Cost (Req 10): Small penalty for high speed to encourage efficiency/precision
        # but not enough to discourage exploration.
        velocity_cost = (ax**2 + ay**2) * 0.002
        reward -= velocity_cost
        
        sparsity_multiplier = 1.0 / (1.0 + self.action_processor.consecutive_action_steps)
        
        # Wall Penalty (Bounce)
        # If the cursor hit a wall this step, physics engine would have clamped it.
        # We can check if it is at the edge and trying to move further?
        # Simpler: If velocity was high but movement was zero.
        # This is already handled by Frustration check above.
        
        if final_action_idx != -1:
            # Small action cost to prefer doing nothing over random spamming
            reward -= 0.01 
        else:
             if trigger < 0.0:
                 reward += 0.001 # Tiny existence reward

        # Handling Spatial Goal (Flag)
        # We introduce explicit distance-based feedback to create a smooth potential field.
        # Further away = Pain (Negative Reward)
        # Closer = Dopamine (Positive Reward)
        dist_dopamine = 0.0
        dist_pain = 0.0
        
        # SYNCHRONIZE SPATIAL GOAL from Agent to Intrinsic System
        if hasattr(self.agent, 'spatial_goal'):
            self.intrinsic_system.spatial_goal = self.agent.spatial_goal

        if self.intrinsic_system.spatial_goal:
            gx, gy = self.intrinsic_system.spatial_goal
            
            # Distance
            dist = np.sqrt((self.agent.cursor_x - gx)**2 + (self.agent.cursor_y - gy)**2)
            max_dist = self.grid_size * 1.414
            
            # Normalized distance (0.0 to 1.0)
            dn = dist / max_dist
            
            # 1. Distance-based Pain (The further, the more painful)
            # Linearly increasing pain as you get further
            # Reduced scale to ensure it doesn't overpower everything else excessively
            dist_pain = dn * 0.05
            
            # 2. Distance-based Dopamine (The closer, the better)
            # Linearly increasing dopamine as you get closer
            dist_dopamine = (1.0 - dn) * 0.1
            
            # Apply to Reward
            # Net effect: Gradient of +5.0 (at goal) to -5.0 (at far corner)
            # This overrides or adds to other rewards
            reward += dist_dopamine - dist_pain
            
            # Also add alignment bonus for Flag (D-Term)
            if dist > 1.0:
                # Vector to target
                dx = gx - self.agent.cursor_x
                dy = gy - self.agent.cursor_y
                # Reuse normalization logic... handled implicitly by PPO learning the gradient
                # But we can boost it:
                pass

        pain = 0.0
        # Use Intrinsic System's manual_pain, which includes both User Input AND Frustration
        if self.intrinsic_system.manual_pain > 0:
            pain += self.intrinsic_system.manual_pain * 0.5
        reward -= pain

        # Shape Reward for Approaching Conditioned Objects (Dopamine Magnetism)
        # Check dopamine map at cursor
        dopamine_map = self.intrinsic_system.get_dopamine_map(
            self.object_tracker.detected_objects,
            current_grid,
            self.object_tracker
        )
        
        # Merge Focus Map and LEARNED POSITIVE MANIFOLD into Dopamine Map for Gradient Following
        # If user clicked "Good" in empty space, it's in learned_positive_manifold.
        # We want the agent to gravitate to that strongly.
        
        # Combine maps: Object Conditioning (dopamine_map) + Explicit Spatial Manifold (learned_positive) + Short-term Focus
        combined_dopamine_map = np.maximum(dopamine_map, self.intrinsic_system.learned_positive_manifold * 2.5)
        combined_dopamine_map = np.maximum(combined_dopamine_map, self.intrinsic_system.focus_map * 2.5 * 0.5)

        cursor_dopamine = combined_dopamine_map[cy_int, cx_int] / 2.5 # 0.0 to 1.0
        
        # Check Negative Manifold (Explicit Pain Areas)
        cursor_pain = self.intrinsic_system.learned_negative_manifold[cy_int, cx_int]
        if cursor_pain > 0.1:
            reward -= cursor_pain * 0.2 # Extremely strong repulsion
            self.intrinsic_system.current_thought = "Bad area!"

        if cursor_dopamine > 0.1:
            # "Effective Conditioning": Strong reward for staying on target.
            # Scaling up to ensure it dominates exploration noise.
            reward += cursor_dopamine * 0.2 
            self.intrinsic_system.current_thought = "I like this spot!"

        # --- DOPAMINE GRADIENT REWARD (The "Scent" Trail) ---
        # Reward moving TOWARDS the center of dopamine mass
        # Threshold lowered to 10 to catch even faint signals in the manifold
        d_indices = np.where(combined_dopamine_map > 10) 
        if len(d_indices[0]) > 0:
            # Centroid of dopamine - simple and robust
            # Ideally we'd use the peak (argmax) for precision if there's a strong single point
            
            # Check if there is a "strong" peak (> 200) -> Go for max
            # Else go for mean
            max_val = np.max(combined_dopamine_map)
            if max_val > 150: # Lowered threshold to latch onto peaks sooner
                # Find closest peak to cursor
                peaks = np.argwhere(combined_dopamine_map >= max_val * 0.9)
                # Just pick the first one or closest one?
                # Distances
                dists = (peaks[:, 0] - self.agent.cursor_y)**2 + (peaks[:, 1] - self.agent.cursor_x)**2
                closest_idx = np.argmin(dists)
                target_y, target_x = peaks[closest_idx]
            else:
                target_y = np.mean(d_indices[0])
                target_x = np.mean(d_indices[1])
            
            # Vector to target
            dx = target_x - self.agent.cursor_x
            dy = target_y - self.agent.cursor_y
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist > 1.0:
                # Normalize direction
                dir_x = dx / dist
                dir_y = dy / dist
                
                # Normalize velocity
                speed = np.sqrt(ax*ax + ay*ay)
                if speed > 0.1:
                    vx = ax / speed
                    vy = ay / speed
                    
                    # Dot product: Alignment between Velocity and Target Direction
                    alignment = (vx * dir_x + vy * dir_y)
                    
                    if alignment > 0:
                        reward += alignment * 0.1 # Super strong reward for moving in right direction (was 2.0)
                        # This effectively overrides the policy to be a gradient climber
                    else:
                        reward -= 0.05 # Stronger penalty for moving away
                        
                    self.intrinsic_system.current_thought = "Following the scent..."
        
        current_action_type = -1
        if final_action_idx != -1:
            if final_action_idx <= 3: current_action_type = 0 
            elif final_action_idx <= 7: current_action_type = 1 
            else: current_action_type = 2 
        
        if hasattr(self, 'last_action_type'):
            if self.last_action_type == current_action_type and current_action_type != -1:
                 reward += 0.01
            elif self.last_action_type != current_action_type and self.last_action_type != -1 and current_action_type != -1:
                 reward -= 0.01
        
        self.last_action_type = current_action_type

        score_diff = float(frame.score - self.last_score)
        reward += score_diff * 1.0
        self.last_score = frame.score
        
        if score_diff > 0:
            decay = 0.9
            current_val = score_diff * 0.5
            for step_hashes in reversed(self.object_tracker.episode_object_hashes):
                for h in step_hashes:
                    self.object_tracker.valuable_object_hashes[h] = self.object_tracker.valuable_object_hashes.get(h, 0) + current_val
                current_val *= decay
                if current_val < 0.01: break
        
        grid_changed = False
        if self.last_grid is not None and current_grid.shape == self.last_grid.shape:
             grid_changed = not np.array_equal(current_grid, self.last_grid)
        
        click_action = final_action_idx != -1 and final_action_idx <= 3
        
        if grid_changed:
            self.object_tracker.scan(current_grid)

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
        
        if click_action and not grid_changed:
             reward -= 0.2 # Reduced penalty
        elif final_action_idx in [4,5,6,7,8,9] and not grid_changed:
             reward += 0.1 # Slight refund
        
        terminated = False # CONTINUAL LEARNING: Never terminate. Agent must learn to reset.
        truncated = False 
        
        if frame.state == GameState.WIN:
            reward += 10.0
            # Optional: Auto-reset on win to save time? Or let agent bask in glory?
            # User said "NEVER reset".
            # BUT: We MUST trigger a reset if the game is over/won to actually play the next level or replay.
            # If we don't, the grid stays static forever in 'WIN' or 'GAME_OVER' state and no actions (except reset) do anything.
            # So "Continual Learning" implies the Agent presses Reset, OR the Env auto-resets after a delay.
            # To enable flow, let's auto-reset after a Win/Loss but treat it as a continuation of the same 'life' (no done=True)
            # OR we force the agent to learn to press RESET (Action 0).
            # The error 'GAME_NOT_STARTED_ERROR' suggests the game is waiting for a reset.
            # So if we are in this state, we should probably force a reset if the agent doesn't do it?
            # Let's trust the agent to learn it, but give a hint?
            # Actually, the error `GAME_NOT_STARTED_ERROR` happened because we selected a game but didn't START it?
            # The select_game_interactively fetches a thumbnail (which does Open -> Reset -> Close).
            # Then we select it.
            # Then we call env.reset() -> agent.take_action(RESET).
            # This should start the game.
            pass
        elif frame.state == GameState.GAME_OVER:
            reward -= 5.0
            pain += 5.0
            # Game Over screen is just another state. Agent must press RESET.
            
        prev_grid = self.last_grid
        self.last_grid = current_grid
        
        # METRICS Construction
        # Using cursor values to show instantaneous experience
        metrics = {
            "score": frame.score,
            # Dopamine: Show max of map-based (visual) or distance-based (analytical)
            # This ensures the graph reflects the strong signal even if the map is blurry
            "dopamine": max(cursor_dopamine, dist_dopamine if 'dist_dopamine' in locals() else 0.0), 
            "manual_dopamine": self.intrinsic_system.manual_dopamine,
            
            # Pain: Use total pain experienced at cursor (Manual + Location based + Distance based)
            "pain": max(pain, cursor_pain * 50.0 if 'cursor_pain' in locals() else 0.0, dist_pain if 'dist_pain' in locals() else 0.0),
            # Show intrinsic pain (includes frustration) rather than just user input
            "manual_pain": self.intrinsic_system.manual_pain,
            
            "plan_confidence": self.intrinsic_system.plan_confidence,
            "current_thought": getattr(self.intrinsic_system, "current_thought", ""),
            "reward": reward,
            "trigger": float(trigger),
            "maps": {
                "pain": self.intrinsic_system.pain_memory.tolist(),
                "visit": self.intrinsic_system.spatial_visitation_map.tolist(),
                "value": self.intrinsic_system.get_value_map().tolist(),
            }
        }
        
        obs = self._get_obs(current_grid, prev_grid, final_action_idx, self.intrinsic_system.pain_memory, dopamine_map)
        
        obs_float = obs.astype(np.float32) / 255.0
        
        metrics["maps"]["obs_delta"] = obs_float[:,:,1].tolist()
        metrics["maps"]["obs_focus"] = obs_float[:,:,2].tolist()
        metrics["maps"]["obs_goal"] = obs_float[:,:,3].tolist()
        metrics["maps"]["obs_vel_x"] = obs_float[:,:,4].tolist()
        metrics["maps"]["obs_vel_y"] = obs_float[:,:,5].tolist()
        metrics["maps"]["obs_pain"] = obs_float[:,:,8].tolist()
        metrics["maps"]["obs_dopamine"] = obs_float[:,:,9].tolist()
        
        # FINAL SAFETY: Clamp reward and check for NaNs
        if np.isnan(reward) or np.isinf(reward):
             reward = 0.0
        reward = np.clip(reward, -10.0, 10.0)
        
        # Ensure obs is clean
        obs = np.nan_to_num(obs, nan=0, posinf=255, neginf=0).astype(np.uint8)
        
        return obs, reward, terminated, truncated, metrics

    def _get_obs(self, current_grid: np.ndarray, prev_grid: Optional[np.ndarray], last_action_idx: int = -1, pain: Any = 0.0, dopamine_map: Any = 0.0) -> np.ndarray:
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
            dopamine=dopamine_map # Passing the Map!
        )
