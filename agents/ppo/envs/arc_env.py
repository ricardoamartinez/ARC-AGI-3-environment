import logging
import os
import time
import gymnasium as gym
import numpy as np
from typing import Any, Optional, Tuple, TYPE_CHECKING
from gymnasium import spaces

from ...structs import GameAction, GameState
from ..control.physics import PhysicsEngine
from ..control.actions import ActionProcessor
from ..perception.observation import ObservationBuilder

logger = logging.getLogger()

if TYPE_CHECKING:
    from ..agent import PPOAgent


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
        self.last_state = GameState.NOT_PLAYED
        self.grid_size = 64
        # Action mode:
        # - "target_cell": policy selects an (x,y) cell directly (precise pointing)
        # - "target_pos": policy selects a continuous target position; physics uses PD to move smoothly & stop
        # - "delta": policy outputs (ax,ay,trigger)+discrete index (legacy)
        # Default depends on trainer:
        # - SAC requires target_pos (continuous 2D Box)
        # - RTAC legacy defaults to delta
        if "PPO_ACTION_MODE" in os.environ:
            self.action_mode = os.environ.get("PPO_ACTION_MODE", "delta").strip().lower()
        else:
            trainer = os.environ.get("PPO_TRAINER", "sac").strip().lower()
            self.action_mode = "target_pos" if trainer == "sac" else "delta"

        # === CursorGoalEnv-style world dynamics (reference implementation) ===
        # World is a square [-bounds, bounds]^2; default bounds=1.0.
        self.world_bounds = float(os.environ.get("PPO_BOUNDS", "1.0"))
        self.world_bounds = max(1e-6, self.world_bounds)

        # In vel-mode, action is a velocity *command* in [-1,1]; it is scaled by max_speed (world units).
        # Reference default max_speed=0.06.
        self.vel_max_speed = float(os.environ.get("PPO_MAX_SPEED", "0.06"))
        self.vel_max_speed = max(0.0, self.vel_max_speed)

        # Optional response smoothing (reference has vel_response parameter):
        # vel <- (1-alpha)*vel + alpha*(action*max_speed)
        # alpha=1.0 means no smoothing (instant response).
        self.vel_response = float(os.environ.get("PPO_VEL_RESPONSE", "1.0"))
        self.vel_response = float(np.clip(self.vel_response, 0.0, 1.0))

        # Reward mode:
        # - "progress": (prev_dist - dist)*reward_scale - action_l2_penalty*||a||^2 (+goal_bonus on reach)
        # - "dense":   -dist^2*reward_scale - action_penalty - vel_penalty - jerk_penalty (+arrival_bonus on entry)
        self.reward_mode = os.environ.get("PPO_REWARD_MODE", "dense").strip().lower()

        # Sparse reward mode (game solving): ignore all shaping and only reward on WIN.
        # This is intentionally off by default; cursor-to-goal trainers rely on dense shaping.
        self.sparse_reward_enabled = os.environ.get("PPO_SPARSE_REWARD", "0") == "1"
        self.success_bonus = float(os.environ.get("PPO_SUCCESS_BONUS", "1.0"))
        self.success_bonus = max(0.0, self.success_bonus)

        self.reward_scale = float(os.environ.get("PPO_REWARD_SCALE", "2.0"))
        self.reward_scale = max(0.0, self.reward_scale)
        self.action_l2_penalty = float(os.environ.get("PPO_ACTION_L2_PENALTY", "0.001"))
        self.action_l2_penalty = max(0.0, self.action_l2_penalty)

        # Dense-mode penalties (reference defaults)
        self.vel_l2_penalty = float(os.environ.get("PPO_VEL_L2_PENALTY", "0.08"))
        self.vel_l2_penalty = max(0.0, self.vel_l2_penalty)
        self.jerk_l2_penalty = float(os.environ.get("PPO_JERK_L2_PENALTY", "0.001"))
        self.jerk_l2_penalty = max(0.0, self.jerk_l2_penalty)
        self.arrival_bonus = float(os.environ.get("PPO_ARRIVAL_BONUS", "2.0"))
        self.arrival_bonus = max(0.0, self.arrival_bonus)

        # Progress-mode extras
        self.goal_bonus = float(os.environ.get("PPO_GOAL_BONUS", "1.0"))
        self.goal_bonus = max(0.0, self.goal_bonus)
        self.goal_change_interval = int(os.environ.get("PPO_GOAL_CHANGE_INTERVAL", "0"))
        self.goal_change_interval = max(0, self.goal_change_interval)

        # Goal semantics
        self.goal_radius = float(os.environ.get("PPO_GOAL_RADIUS", "0.06"))  # world units
        self.goal_radius = max(0.0, self.goal_radius)
        self.auto_new_goal_on_reach = os.environ.get("PPO_AUTO_NEW_GOAL_ON_REACH", "0") == "1"

        # World state for vel-mode
        self._w_pos = np.zeros((2,), dtype=np.float32)
        self._w_vel = np.zeros((2,), dtype=np.float32)
        self._w_prev_action = np.zeros((2,), dtype=np.float32)
        self._in_goal = False
        self._vel_steps = 0

        # Energy / force penalty (encourages stopping and smooth control)
        self.energy_penalty_enabled = os.environ.get("PPO_ENERGY_PENALTY", "0") == "1"
        self.energy_penalty_coef = float(os.environ.get("PPO_ENERGY_COEF", "0.02"))
        self.energy_penalty_mode = os.environ.get("PPO_ENERGY_PENALTY_MODE", "near_goal").strip().lower()
        self.energy_near_goal_radius = float(os.environ.get("PPO_ENERGY_NEAR_GOAL_RADIUS", "6.0"))
        self.energy_near_goal_radius = max(1e-6, self.energy_near_goal_radius)

        # If false, we never send API game actions; we only learn cursor navigation + goal reaching.
        # This avoids resets and prevents GAME_NOT_STARTED / GAME_OVER issues from crashing training.
        # Default ON so actions actually execute in-game.
        self.allow_game_actions = os.environ.get("PPO_ALLOW_GAME_ACTIONS", "1") == "1"
        # Bootstrapping: server may require RESET once to get the first frame. Disable only if you
        # guarantee frames already exist (otherwise training can't start).
        self.disable_reset_bootstrap = os.environ.get("PPO_DISABLE_RESET_BOOTSTRAP", "0") == "1"

        # Target command smoothing (reduces jitter from action noise / exploration).
        # Higher alpha = more responsive to new targets (less smoothing)
        self.target_ema_alpha = float(os.environ.get("PPO_TARGET_EMA_ALPHA", "0.6"))
        self.target_ema_alpha = max(0.0, min(1.0, self.target_ema_alpha))
        self._ema_tx: float | None = None
        self._ema_ty: float | None = None
        self._ema_speed_scale: float | None = None

        # Target-pos semantics:
        # Absolute targets map [-1,1] directly to [0..63], which makes early exploration look like
        # "teleport to corners" and can cause the policy to get stuck on walls.
        # For online TD learning (RTAC), a *relative* target (local delta) is much more stable.
        trainer = os.environ.get("PPO_TRAINER", "sac").strip().lower()
        if "PPO_TARGET_RELATIVE" in os.environ:
            self.target_pos_relative = os.environ.get("PPO_TARGET_RELATIVE", "0") == "1"
        else:
            # Default: RTAC (and other non-SAC trainers) use relative targets.
            self.target_pos_relative = trainer != "sac"
        self.target_delta_max = float(os.environ.get("PPO_TARGET_DELTA_MAX", "16.0"))  # Increased for faster cursor
        self.target_delta_max = max(0.0, min(float(self.grid_size - 1), self.target_delta_max))

        # Mild penalty for hitting walls (helps prevent edge/corner stickiness).
        self.wall_penalty = float(os.environ.get("PPO_WALL_PENALTY", "0.02"))
        self.wall_penalty = max(0.0, self.wall_penalty)
        
        # Penalty for game actions that have no effect (encourages learning valid actions)
        self.ineffective_action_penalty = float(os.environ.get("PPO_INEFFECTIVE_ACTION_PENALTY", "0.5"))
        self.ineffective_action_penalty = max(0.0, self.ineffective_action_penalty)
        
        # Components (delta-mode only)
        self.physics = PhysicsEngine(self.grid_size)
        self.action_processor = ActionProcessor()
        self.obs_builder = ObservationBuilder(self.grid_size)
        
        self.last_grid: Optional[np.ndarray] = None
        
        self.observation_space = self.obs_builder.observation_space
        if self.action_mode in ("vel", "velocity", "cursor_vel"):
            # Hybrid action space: 3D continuous (ax, ay, trigger) + discrete action index.
            # This allows both velocity control AND discrete game actions (UP/DOWN/LEFT/RIGHT/SPACE/CLICK/ENTER).
            self.action_space = spaces.Tuple((
                spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                spaces.Discrete(10)
            ))
        elif self.action_mode == "target_cell":
            self.action_space = spaces.MultiDiscrete([self.grid_size, self.grid_size])
        elif self.action_mode == "target_pos":
            # 3D continuous: target_x, target_y, speed_scale in [-1,1].
            # speed_scale is mapped to [0,1] and scales max speed so the policy can "brake" and settle.
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        else:
            self.action_space = spaces.Tuple((
                spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                spaces.Discrete(10)
            ))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict[str, Any]]:
        logger.info("DEBUG: Env Reset Start")
        super().reset(seed=seed)
        
        self.current_step = 0
        self.last_grid = None
        self.last_score = 0
        self.last_state = GameState.NOT_PLAYED
        
        if self.action_mode != "target_cell":
            self.physics.reset()
            self.action_processor.reset()
            self._ema_tx = None
            self._ema_ty = None
            self._ema_speed_scale = None
            self._w_pos[...] = 0.0
            self._w_vel[...] = 0.0
            self._w_prev_action[...] = 0.0
            self._in_goal = False
            self._vel_steps = 0
        
        self.agent.cursor_x = self.grid_size // 2
        self.agent.cursor_y = self.grid_size // 2
        # Initialize vel-mode world pos to center.
        self._w_pos[...] = 0.0
        self._w_vel[...] = 0.0
        self._w_prev_action[...] = 0.0
        self._in_goal = False
        self._vel_steps = 0
        
        self.last_state = GameState.NOT_PLAYED
        
        frame = None
        if self.disable_reset_bootstrap:
            # No reset allowed: rely on any pre-existing frame (rare) or fail fast with a clear message.
            frame = self.agent.frames[-1] if self.agent.frames else None
            if frame is None:
                # Cursor-only / goal-learning mode without server bootstrapping: start from a blank grid.
                blank = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
                self.last_grid = blank
                # Provide something for the manual UI to render even without any server frames.
                self.agent._latest_grid_for_ui = [blank.tolist()]
                obs = self._get_obs(blank, None)
                logger.info("DEBUG: Env Reset Done (blank bootstrap; PPO_DISABLE_RESET_BOOTSTRAP=1)")
                return obs, {}
        else:
            # Bootstrap the very first frame. Server may require RESET to start a play session.
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
        
        if frame.frame and frame.frame[0]:
            grid = np.array(frame.frame[0], dtype=np.uint8)
            h, w = grid.shape
            self.agent.cursor_x = w / 2.0
            self.agent.cursor_y = h / 2.0
            self.last_grid = grid
            self.last_score = int(getattr(frame, "score", 0))
            self.last_state = frame.state
        
        current_grid = self.last_grid if self.last_grid is not None else np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        # Keep a copy for the UI in case server frames stop flowing.
        self.agent._latest_grid_for_ui = [current_grid.tolist()]
        
        obs = self._get_obs(current_grid, None)
        logger.info("DEBUG: Env Reset Done")
        return obs, {}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.current_step += 1

        prev_x = self.agent.cursor_x
        prev_y = self.agent.cursor_y
        goal = getattr(self.agent, "spatial_goal", None)
        hit_wall = False

        trigger = -1.0
        final_action_idx = -1
        action_ax = 0.0
        action_ay = 0.0

        if self.action_mode in ("vel", "velocity", "cursor_vel"):
            # Hybrid action: (continuous[3], discrete) -> velocity control + discrete game actions
            # Unpack action tuple
            if isinstance(action, tuple) and len(action) == 2:
                cont_actions, disc_idx = action
            else:
                # Fallback: pure continuous (legacy compatibility)
                cont_actions = action if isinstance(action, np.ndarray) else np.array(action, dtype=np.float32)
                disc_idx = 0
            
            if isinstance(cont_actions, np.ndarray):
                ax = float(cont_actions[0]) if cont_actions.shape[0] >= 1 else 0.0
                ay = float(cont_actions[1]) if cont_actions.shape[0] >= 2 else 0.0
                trigger = float(cont_actions[2]) if cont_actions.shape[0] >= 3 else -1.0
            else:
                ax = float(cont_actions[0]) if len(cont_actions) >= 1 else 0.0
                ay = float(cont_actions[1]) if len(cont_actions) >= 2 else 0.0
                trigger = float(cont_actions[2]) if len(cont_actions) >= 3 else -1.0
            
            a = np.array([float(np.clip(ax, -1.0, 1.0)), float(np.clip(ay, -1.0, 1.0))], dtype=np.float32)
            action_ax, action_ay = float(a[0]), float(a[1])
            
            # Process discrete action if trigger exceeds threshold
            curr_speed = float(np.linalg.norm(self._w_vel))
            final_action_idx, _, _, trigger = self.action_processor.process(
                np.array([action_ax, action_ay, trigger], dtype=np.float32),
                int(disc_idx),
                curr_speed
            )

            require_goal = os.environ.get("PPO_REQUIRE_GOAL", "1") == "1"
            if goal is None and require_goal:
                # Idle; do not integrate dynamics.
                self._w_vel[...] = 0.0
                self._w_prev_action[...] = 0.0
                self.agent._last_action_viz = {"name": "Idle (no goal)", "data": {"x": float(self.agent.cursor_x), "y": float(self.agent.cursor_y)}}
            else:
                # Optional periodic goal moves (reference feature). Only used when enabled.
                self._vel_steps += 1
                if self.goal_change_interval > 0 and (self._vel_steps % self.goal_change_interval) == 0:
                    gx = float(np.random.uniform(-self.world_bounds, self.world_bounds))
                    gy = float(np.random.uniform(-self.world_bounds, self.world_bounds))
                    # Store as grid goal so UI stays consistent.
                    gx_cell = int(round((gx + self.world_bounds) / (2.0 * self.world_bounds) * float(self.grid_size - 1)))
                    gy_cell = int(round((self.world_bounds - gy) / (2.0 * self.world_bounds) * float(self.grid_size - 1)))
                    gx_cell = int(np.clip(gx_cell, 0, self.grid_size - 1))
                    gy_cell = int(np.clip(gy_cell, 0, self.grid_size - 1))
                    self.agent.spatial_goal = (gx_cell, gy_cell)
                    self.agent.goal_version = int(getattr(self.agent, "goal_version", 0)) + 1
                    goal = self.agent.spatial_goal

                # Convert grid goal to world goal.
                if goal is not None:
                    gx_cell, gy_cell = goal
                    gxw = (float(gx_cell) / float(max(1, self.grid_size - 1))) * (2.0 * self.world_bounds) - self.world_bounds
                    gyw = self.world_bounds - (float(gy_cell) / float(max(1, self.grid_size - 1))) * (2.0 * self.world_bounds)
                    g_w = np.array([gxw, gyw], dtype=np.float32)
                else:
                    g_w = None

                # World dynamics:
                # desired_vel = action * max_speed
                desired_vel = a * float(self.vel_max_speed)
                alpha = float(self.vel_response)
                self._w_vel = (1.0 - alpha) * self._w_vel + alpha * desired_vel
                new_pos = self._w_pos + self._w_vel

                # Bounds clamp
                b = float(self.world_bounds)
                clipped = np.clip(new_pos, -b, b)
                hit_wall = bool(np.any(clipped != new_pos))
                self._w_pos = clipped.astype(np.float32)

                # Map world -> grid cursor for UI
                cx = (float(self._w_pos[0]) + b) / (2.0 * b) * float(self.grid_size - 1)
                cy = (b - float(self._w_pos[1])) / (2.0 * b) * float(self.grid_size - 1)
                self.agent.cursor_x = float(np.clip(cx, 0.0, float(self.grid_size - 1)))
                self.agent.cursor_y = float(np.clip(cy, 0.0, float(self.grid_size - 1)))

                self.agent._last_action_viz = {"name": f"Vel ({action_ax:+.2f},{action_ay:+.2f})", "data": {"ax": action_ax, "ay": action_ay}}
        elif self.action_mode == "target_cell":
            # Policy selects a grid cell directly.
            # action can be np.ndarray([x,y]) or tuple/list.
            if isinstance(action, np.ndarray):
                tx = int(action[0])
                ty = int(action[1])
            else:
                tx = int(action[0])
                ty = int(action[1])
            tx = int(max(0, min(self.grid_size - 1, tx)))
            ty = int(max(0, min(self.grid_size - 1, ty)))
            self.agent.cursor_x = float(tx)
            self.agent.cursor_y = float(ty)
            self.agent._last_action_viz = {"name": f"Target ({tx},{ty})", "data": {"x": tx, "y": ty}}
        elif self.action_mode == "target_pos":
            if isinstance(action, np.ndarray):
                ax = float(action[0])
                ay = float(action[1])
                sraw = float(action[2]) if action.shape[0] >= 3 else 1.0
            else:
                ax = float(action[0])
                ay = float(action[1])
                sraw = float(action[2]) if len(action) >= 3 else 1.0
            ax = float(np.clip(ax, -1.0, 1.0))
            ay = float(np.clip(ay, -1.0, 1.0))
            sraw = float(np.clip(sraw, -1.0, 1.0))
            action_ax, action_ay = ax, ay

            # Intrinsic UX/safety fix: when no goal is set, don't let a stochastic policy
            # wander into corners. Idle in-place until the user sets a goal.
            require_goal = os.environ.get("PPO_REQUIRE_GOAL", "1") == "1"
            if goal is None and require_goal:
                # Hard stop (no drift), and hold target at current position.
                self.physics.reset()
                self._ema_tx = float(self.agent.cursor_x)
                self._ema_ty = float(self.agent.cursor_y)
                self._ema_speed_scale = 0.0
                tx = float(self.agent.cursor_x)
                ty = float(self.agent.cursor_y)
                speed_scale = 0.0
                self.agent.cursor_x, self.agent.cursor_y, _hit_wall = self.physics.update_to_target(
                    tx, ty, self.agent.cursor_x, self.agent.cursor_y, speed_scale=speed_scale
                )
                hit_wall = bool(_hit_wall)
                self.agent._last_action_viz = {
                    "name": "Idle (no goal)",
                    "data": {"x": float(self.agent.cursor_x), "y": float(self.agent.cursor_y)},
                }
            else:
                # Relative target (stable default for RTAC): local delta in cells.
                if bool(getattr(self, "target_pos_relative", False)):
                    d = float(getattr(self, "target_delta_max", 8.0))
                    tx = float(self.agent.cursor_x) + float(ax) * d
                    ty = float(self.agent.cursor_y) + float(ay) * d
                else:
                    # Absolute target: map [-1,1] -> [0, grid-1]
                    tx = (ax + 1.0) * 0.5 * float(self.grid_size - 1)
                    ty = (ay + 1.0) * 0.5 * float(self.grid_size - 1)
                tx = float(np.clip(tx, 0.0, float(self.grid_size - 1)))
                ty = float(np.clip(ty, 0.0, float(self.grid_size - 1)))

                # Map [-1,1] -> [0, PPO_SPEED_SCALE_MAX]. Values >1 allow fast travel when far.
                speed_scale_max = float(os.environ.get("PPO_SPEED_SCALE_MAX", "2.0"))
                speed_scale_max = max(0.0, speed_scale_max)
                speed_scale = float(np.clip((sraw + 1.0) * 0.5 * speed_scale_max, 0.0, speed_scale_max))

                # Smooth the command to reduce twitchiness (acts like human motor smoothing).
                a = self.target_ema_alpha
                if a > 0.0:
                    if self._ema_tx is None:
                        self._ema_tx, self._ema_ty, self._ema_speed_scale = tx, ty, speed_scale
                    else:
                        self._ema_tx = (1.0 - a) * self._ema_tx + a * tx
                        self._ema_ty = (1.0 - a) * self._ema_ty + a * ty
                        self._ema_speed_scale = (1.0 - a) * float(self._ema_speed_scale) + a * speed_scale
                    tx = float(self._ema_tx)
                    ty = float(self._ema_ty)
                    speed_scale = float(self._ema_speed_scale)

                self.agent.cursor_x, self.agent.cursor_y, _hit_wall = self.physics.update_to_target(
                    tx, ty, self.agent.cursor_x, self.agent.cursor_y, speed_scale=speed_scale
                )
                hit_wall = bool(_hit_wall)
                self.agent._last_action_viz = {
                    "name": f"TargetPos ({tx:.1f},{ty:.1f}) v={speed_scale:.2f}",
                    "data": {"x": tx, "y": ty, "speed_scale": speed_scale},
                }
        else:
            cont_actions, disc_idx = action
            curr_speed = self.physics.get_speed()
            final_action_idx, ax, ay, trigger = self.action_processor.process(cont_actions, disc_idx, curr_speed)
            action_ax, action_ay = ax, ay
            self.agent.cursor_x, self.agent.cursor_y, _hit_wall = self.physics.update(
                ax, ay, self.agent.cursor_x, self.agent.cursor_y
            )
            hit_wall = bool(_hit_wall)
        
        # Base reward: human feedback
        manual_dopamine = float(getattr(self.agent, "manual_dopamine", 0.0))
        manual_pain = float(getattr(self.agent, "manual_pain", 0.0))
        reward = manual_dopamine - manual_pain

        # Reward shaping for vel-mode (reference formulas).
        entered_goal = False
        reached = False
        dist_world = None
        prev_dist_world = None
        shaped = None
        penalty = None

        if self.action_mode in ("vel", "velocity", "cursor_vel") and goal is not None:
            gx_cell, gy_cell = goal
            b = float(self.world_bounds)
            gxw = (float(gx_cell) / float(max(1, self.grid_size - 1))) * (2.0 * b) - b
            gyw = b - (float(gy_cell) / float(max(1, self.grid_size - 1))) * (2.0 * b)
            g = np.array([gxw, gyw], dtype=np.float32)

            prev_dist_world = float(np.linalg.norm(g - self._w_pos))

            # NOTE: reward uses the *executed* action command in [-1,1].
            a = np.array([action_ax, action_ay], dtype=np.float32)

            # Current dist after integration
            dist_world = float(np.linalg.norm(g - self._w_pos))

            if self.reward_mode == "progress":
                shaped = (prev_dist_world - dist_world) * float(self.reward_scale)
                penalty = float(self.action_l2_penalty) * float(np.sum(a * a))
                reward += float(shaped) - float(penalty)
                reached = bool(dist_world <= float(self.goal_radius))
                if reached:
                    reward += float(self.goal_bonus)
            else:
                # dense
                dist_cost = (dist_world * dist_world) * float(self.reward_scale)
                action_penalty = float(self.action_l2_penalty) * float(np.sum(a * a))
                vel_penalty = float(self.vel_l2_penalty) * float(np.sum(self._w_vel * self._w_vel))
                jerk = a - self._w_prev_action
                jerk_penalty = float(self.jerk_l2_penalty) * float(np.sum(jerk * jerk))

                reward += -dist_cost - action_penalty - vel_penalty - jerk_penalty

                reached = bool(dist_world <= float(self.goal_radius))
                entered_goal = (not bool(self._in_goal)) and reached
                if entered_goal:
                    reward += float(self.arrival_bonus)

            # Goal auto-teleport on reach (optional; default OFF for click-to-set-goal).
            if reached and self.auto_new_goal_on_reach:
                gx = float(np.random.uniform(-b, b))
                gy = float(np.random.uniform(-b, b))
                gx_cell2 = int(round((gx + b) / (2.0 * b) * float(self.grid_size - 1)))
                gy_cell2 = int(round((b - gy) / (2.0 * b) * float(self.grid_size - 1)))
                gx_cell2 = int(np.clip(gx_cell2, 0, self.grid_size - 1))
                gy_cell2 = int(np.clip(gy_cell2, 0, self.grid_size - 1))
                self.agent.spatial_goal = (gx_cell2, gy_cell2)
                self.agent.goal_version = int(getattr(self.agent, "goal_version", 0)) + 1
                goal = self.agent.spatial_goal

            self._in_goal = reached
            self._w_prev_action = np.array([action_ax, action_ay], dtype=np.float32)

        # Energy penalty (optional): discourages motion/jitter so it settles on the goal.
        # IMPORTANT: if we penalize speed far from the goal, the optimal policy is to crawl ("inching").
        # So default behavior is to penalize speed only near the goal.
        if self.action_mode == "target_pos":
            s = float(self.physics.get_speed())
            action_energy = float(s * s)
        else:
            action_energy = float(action_ax * action_ax + action_ay * action_ay)
        if self.energy_penalty_enabled and self.energy_penalty_coef > 0.0:
            w = 1.0
            if self.energy_penalty_mode == "near_goal":
                goal = getattr(self.agent, "spatial_goal", None)
                if goal is not None:
                    gx, gy = goal
                    dist_now = float(np.sqrt((self.agent.cursor_x - gx) ** 2 + (self.agent.cursor_y - gy) ** 2))
                    # Weight ramps from 0 (far) -> 1 (at/near goal)
                    w = float(np.clip(1.0 - (dist_now / self.energy_near_goal_radius), 0.0, 1.0))
                else:
                    w = 0.0
            reward -= self.energy_penalty_coef * w * action_energy

        # Wall penalty (helps escape corner/edge lock-in).
        if hit_wall and float(getattr(self, "wall_penalty", 0.0)) > 0.0:
            reward -= float(getattr(self, "wall_penalty", 0.0))

        # Optional goal shaping (debug): reward progress toward clicked goal.
        # Default is OFF; when OFF we only log goal distance/progress.
        goal_dist = None
        goal_progress = 0.0
        if goal is not None:
            gx, gy = goal
            prev_dist = float(np.sqrt((prev_x - gx) ** 2 + (prev_y - gy) ** 2))
            goal_dist = float(np.sqrt((self.agent.cursor_x - gx) ** 2 + (self.agent.cursor_y - gy) ** 2))
            goal_progress = prev_dist - goal_dist
            if bool(getattr(self.agent, "goal_shaping_enabled", False)):
                # Goal shaping modes (for headless convergence testing).
                # - potential: dense potential only (smooth/low-variance)
                # - potential_progress: potential + clipped progress + hit bonus (default)
                shaping_mode = os.environ.get("PPO_GOAL_SHAPING_MODE", "potential_progress").strip().lower()
                max_dist = float(self.grid_size * 1.41421356237)
                reward += -goal_dist / max_dist
                if shaping_mode != "potential":
                    # Progress term can be spiky; clip to keep learning stable.
                    clipped_progress = float(np.clip(goal_progress, -1.0, 1.0))
                    reward += 1.0 * clipped_progress
                    if goal_dist <= 1.0:
                        reward += 1.0
        goal_hit = bool(goal_dist is not None and goal_dist <= 1.0)

        cx_int = int(round(self.agent.cursor_x))
        cy_int = int(round(self.agent.cursor_y))
        
        game_action = None
        action_data = None
        if self.allow_game_actions and self.action_mode != "target_cell":
            result = self.action_processor.get_game_action(final_action_idx, cx_int, cy_int, self.agent.game_id)
            if result is not None:
                game_action, action_data = result
        
        frame = None
        action_had_effect = False
        if game_action:
            # Check if game needs reset (GAME_OVER state)
            if self.last_state == GameState.GAME_OVER:
                logger.info("Game is GAME_OVER, attempting auto-reset...")
                reset_frame = self.agent.take_action(GameAction.RESET, {"game_id": self.agent.game_id})
                if reset_frame:
                    self.agent.append_frame(reset_frame)
                    self.last_state = reset_frame.state
                    logger.info("Auto-reset successful, state=%s", reset_frame.state)
            
            # Capture grid state BEFORE the action to detect if it actually changed anything
            grid_before = None
            score_before = self.last_score
            if self.last_grid is not None:
                grid_before = self.last_grid.copy()
            
            # Pass action_data explicitly to avoid race conditions with mutable enum state
            frame = self.agent.take_action(game_action, action_data)
            if frame:
                self.agent.append_frame(frame)
                
                # Check if the action actually changed the game state
                grid_after = np.array(frame.frame[0], dtype=np.uint8) if (frame.frame and frame.frame[0]) else None
                score_after = int(getattr(frame, "score", 0))
                
                # Detect if action had any effect: grid changed OR score changed OR game state changed
                if grid_before is not None and grid_after is not None:
                    if not np.array_equal(grid_before, grid_after):
                        action_had_effect = True
                if score_after != score_before:
                    action_had_effect = True
                if frame.state != self.last_state:
                    action_had_effect = True
                
                # ALWAYS show raw model output - what action it chose
                action_name = game_action.name
                if final_action_idx == 4: action_name = "UP ↑"
                elif final_action_idx == 5: action_name = "DOWN ↓"
                elif final_action_idx == 6: action_name = "LEFT ←"
                elif final_action_idx == 7: action_name = "RIGHT →"
                elif final_action_idx == 8: action_name = "SPACE ␣"
                elif final_action_idx == 9: action_name = "ENTER ↵"
                elif final_action_idx <= 3: action_name = "CLICK"
                
                self.agent._last_action_viz = {
                    "id": game_action.value,
                    "name": action_name,
                    "data": {"x": cx_int, "y": cy_int, "ax": action_ax, "ay": action_ay},
                    "had_effect": action_had_effect
                }
                
                if action_had_effect:
                    logger.debug("Game action HAD EFFECT: %s (id=%s)", action_name, game_action.value)
                else:
                    logger.debug("Game action NO EFFECT: %s (id=%s)", action_name, game_action.value)
            else:
                # API refused action - try reset if game might be over
                logger.warning("Game action FAILED: %s", game_action.name)
                reset_frame = self.agent.take_action(GameAction.RESET, {"game_id": self.agent.game_id})
                if reset_frame:
                    self.agent.append_frame(reset_frame)
                    frame = reset_frame
                else:
                    frame = self.agent.frames[-1] if self.agent.frames else None
        else:
            if self.agent.frames:
                 frame = self.agent.frames[-1]
            else:
                 raise RuntimeError("No frames available to build observation.")

            if self.action_mode != "target_cell":
                status = "Cursor Move"
                if trigger > 0.2:
                    status = f"Aiming..."
                # Include ax/ay for keyboard direction highlighting
                self.agent._last_action_viz = {
                    "name": status,
                    "data": {
                        "x": self.agent.cursor_x, 
                        "y": self.agent.cursor_y,
                        "ax": action_ax,
                        "ay": action_ay,
                    }
                }
        
        if not frame:
            if self.agent.frames:
                frame = self.agent.frames[-1]
            else:
                raise RuntimeError("No frame returned and no previous frames available.")

        # Ineffective action penalty: penalize game actions that don't change the game state
        # This teaches the model to only press buttons when they will have an effect
        ineffective_penalty_applied = 0.0
        if game_action is not None and not action_had_effect:
            ineffective_penalty_applied = float(self.ineffective_action_penalty)
            reward -= ineffective_penalty_applied

        current_grid = np.array(frame.frame[0], dtype=np.uint8) if (frame.frame and frame.frame[0]) else None
        if current_grid is None:
             raise RuntimeError("Frame contains no grid data.")

        # Keep a copy for the UI in case server frames stop flowing or game actions are disabled.
        self.agent._latest_grid_for_ui = [current_grid.tolist()]

        # Continual learning: do not terminate or truncate episodes here.
        terminated = False
        truncated = False
        
        # Store state for next step
        self.last_state = frame.state
            
        prev_grid = self.last_grid
        self.last_grid = current_grid
        
        metrics = {
            "score": frame.score,
            "reward": reward,
            "trigger": float(trigger),
            "final_action_idx": int(final_action_idx),
            "manual_dopamine": manual_dopamine,
            "manual_pain": manual_pain,
            "goal": {"x": goal[0], "y": goal[1]} if goal is not None else None,
            "goal_version": int(getattr(self.agent, "goal_version", 0)),
            "goal_dist": goal_dist,
            "goal_progress": goal_progress,
            "goal_hit": goal_hit,
            "entered_goal": bool(entered_goal),
            "goal_dist_world": dist_world,
            "prev_goal_dist_world": prev_dist_world,
            "shaped": shaped,
            "penalty": penalty,
            "goal_shaping_enabled": bool(getattr(self.agent, "goal_shaping_enabled", False)),
            # Diagnostics: verify what palette/range the game is using
            "grid_min": int(current_grid.min()) if current_grid.size else 0,
            "grid_max": int(current_grid.max()) if current_grid.size else 0,
            "action_mode": self.action_mode,
            "cursor_speed": float(self.physics.get_speed()),
            "hit_wall": bool(hit_wall),
            "action_energy": action_energy,
            "energy_penalty_coef": float(self.energy_penalty_coef) if self.energy_penalty_enabled else 0.0,
            "action_had_effect": action_had_effect,
            "ineffective_action_penalty": ineffective_penalty_applied,
        }

        # Game terminal signals (used by sparse reward training / world-model training).
        # Note: this env is configured as a continuing task by default (terminated/truncated stay False),
        # but we still expose game outcome in info for downstream trainers.
        win = bool(frame.state == GameState.WIN)
        game_over = bool(frame.state == GameState.GAME_OVER)
        metrics["game_state"] = str(frame.state)
        metrics["game_success"] = win
        metrics["game_over"] = game_over
        metrics["score_delta"] = int(frame.score) - int(self.last_score)

        # Optional sparse reward override: only reward on WIN.
        if bool(getattr(self, "sparse_reward_enabled", False)):
            reward = float(self.success_bonus) if win else 0.0
            metrics["reward"] = reward

        # Update last_* for delta calculations next step.
        self.last_score = int(frame.score)
        self.last_state = frame.state
        
        obs = self._get_obs(current_grid, prev_grid)
        return obs, reward, terminated, truncated, metrics

    def vector_obs(self) -> np.ndarray:
        """
        Reference-style vector observation.

        progress mode: [rel_x, rel_y, vel_x, vel_y, dist]
        dense mode:    [rel_x, rel_y, vel_x, vel_y, prev_a_x, prev_a_y, dist]

        All values are in world units (bounds/max_speed space), matching the pasted reference code.
        """
        goal = getattr(self.agent, "spatial_goal", None)
        if goal is None:
            # Keep shapes stable based on reward_mode.
            return np.zeros((5 if self.reward_mode == "progress" else 7,), dtype=np.float32)

        gx_cell, gy_cell = goal
        b = float(self.world_bounds)
        gxw = (float(gx_cell) / float(max(1, self.grid_size - 1))) * (2.0 * b) - b
        gyw = b - (float(gy_cell) / float(max(1, self.grid_size - 1))) * (2.0 * b)
        g = np.array([gxw, gyw], dtype=np.float32)

        rel = (g - self._w_pos).astype(np.float32)
        dist = float(np.linalg.norm(rel))
        vel = self._w_vel.astype(np.float32)

        if self.reward_mode == "progress":
            return np.array([rel[0], rel[1], vel[0], vel[1], dist], dtype=np.float32)
        else:
            pa = self._w_prev_action.astype(np.float32)
            return np.array([rel[0], rel[1], vel[0], vel[1], pa[0], pa[1], dist], dtype=np.float32)

    def _get_obs(self, current_grid: np.ndarray, prev_grid: Optional[np.ndarray]) -> np.ndarray:
         # Pad/Resize current_grid to 64x64 if needed
         h, w = current_grid.shape
         target_size = self.grid_size
         
         padded_grid = np.zeros((target_size, target_size), dtype=np.uint8)
         # Center crop or Center pad
         pad_y = (target_size - h) // 2
         pad_x = (target_size - w) // 2
         
         # Safe copy with bounds check
         # Case 1: Grid smaller than Target (Pad)
         # Case 2: Grid larger than Target (Crop)
         
         src_y_start = max(0, -pad_y)
         src_x_start = max(0, -pad_x)
         src_y_end = min(h, h - (h - target_size) - src_y_start) if h > target_size else h
         src_x_end = min(w, w - (w - target_size) - src_x_start) if w > target_size else w
         
         dst_y_start = max(0, pad_y)
         dst_x_start = max(0, pad_x)
         
         # Calculate dimensions to copy
         copy_h = min(src_y_end - src_y_start, target_size - dst_y_start)
         copy_w = min(src_x_end - src_x_start, target_size - dst_x_start)
         
         if copy_h > 0 and copy_w > 0:
             padded_grid[dst_y_start:dst_y_start+copy_h, dst_x_start:dst_x_start+copy_w] = \
                 current_grid[src_y_start:src_y_start+copy_h, src_x_start:src_x_start+copy_w]
         
         # Also pad/resize prev_grid if it exists
         padded_prev = None
         if prev_grid is not None:
             padded_prev = np.zeros((target_size, target_size), dtype=np.uint8)
             if copy_h > 0 and copy_w > 0:
                 padded_prev[dst_y_start:dst_y_start+copy_h, dst_x_start:dst_x_start+copy_w] = \
                     prev_grid[src_y_start:src_y_start+copy_h, src_x_start:src_x_start+copy_w]

         return self.obs_builder.build(
            current_grid=padded_grid,
            last_grid=padded_prev,
            cursor_x=self.agent.cursor_x,
            cursor_y=self.agent.cursor_y,
            manual_pain=float(getattr(self.agent, "manual_pain", 0.0)),
            manual_dopamine=float(getattr(self.agent, "manual_dopamine", 0.0)),
            goal=getattr(self.agent, "spatial_goal", None),
        )


