"""
World Model Trainer - V-JEPA style training for ARC-AGI-3.

Uses world model for planning + sparse reward learning.
"""

import logging
import os
import time
from typing import Optional

import numpy as np
import torch

from ..envs.arc_env import ARCGymEnv
from ..viz.live_visualizer import LiveVisualizerCallback
from ..viz.jepa_visualizer import JEPAVisualizer
from ..world_model import WorldModel
from ..intrinsic import StateCounter

logger = logging.getLogger()


def run_world_model_training(agent) -> None:
    """
    World model based training with sparse rewards.

    1. Learns dynamics: (state, action) -> next_state
    2. Learns win predictor: state -> P(win)
    3. Uses CEM planning to find winning action sequences
    4. Curiosity bonus for exploration until first win
    """
    logger.info("Starting World Model training for game %s", agent.game_id)

    device = os.environ.get("JEPA_RTAC_DEVICE", "cuda")
    device = device if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Config
    latent_dim = int(os.environ.get("WM_LATENT_DIM", "256"))
    planning_horizon = int(os.environ.get("WM_PLANNING_HORIZON", "10"))
    use_planner = os.environ.get("WM_USE_PLANNER", "1") == "1"
    use_curiosity = os.environ.get("WM_USE_CURIOSITY", "1") == "1"
    curiosity_scale = float(os.environ.get("WM_CURIOSITY_SCALE", "0.1"))
    train_every = int(os.environ.get("WM_TRAIN_EVERY", "4"))
    log_every = int(os.environ.get("JEPA_LOG_EVERY", "50"))
    exploration_steps = int(os.environ.get("WM_EXPLORATION_STEPS", "1000"))
    buffer_size = int(os.environ.get("WM_BUFFER_SIZE", "50000"))
    batch_size = int(os.environ.get("WM_BATCH_SIZE", "64"))
    lr = float(os.environ.get("WM_LR", "1e-4"))
    mask_ratio = float(os.environ.get("WM_MASK_RATIO", "0.0"))
    mask_ratio = float(np.clip(mask_ratio, 0.0, 0.99))
    mask_loss_coef = float(os.environ.get("WM_MASK_COEF", "1.0"))
    variance_coef = float(os.environ.get("WM_VAR_COEF", "0.0"))
    variance_target = float(os.environ.get("WM_VAR_TARGET", "1.0"))
    # Exploration: in delta-mode, discrete actions only execute when trigger > threshold.
    trigger_prob = float(os.environ.get("WM_TRIGGER_PROB", "0.25"))
    trigger_prob = float(np.clip(trigger_prob, 0.0, 1.0))
    # Optionally "render" the cursor into the grid for the world model.
    # This helps model click-like actions where outcomes depend on (x,y) interaction.
    mark_cursor = os.environ.get("WM_MARK_CURSOR", "1") == "1"
    cursor_token = int(os.environ.get("WM_CURSOR_TOKEN", "255"))
    cursor_token = int(np.clip(cursor_token, 0, 255))
    # Optional: also mark the *previous* cursor position with a second token.
    # This approximates a short temporal context (velocity) without changing the encoder API.
    mark_prev_cursor = os.environ.get("WM_MARK_PREV_CURSOR", "1") == "1"
    prev_cursor_token = int(os.environ.get("WM_PREV_CURSOR_TOKEN", "254"))
    prev_cursor_token = int(np.clip(prev_cursor_token, 0, 255))

    # Initialize environment
    env = ARCGymEnv(agent, max_steps=1_000_000)
    callback = LiveVisualizerCallback(agent.gui_process, agent)
    callback._quit_event = agent._quit_event

    # Determine action encoding for the world model.
    # - Box actions: continuous only; discrete token is always 0 (NONE).
    # - Tuple(Box(3), Discrete(10)) "delta" mode: store continuous (ax, ay) and
    #   store executed discrete action as `disc_token = final_action_idx + 1` in [0..10].
    from gymnasium import spaces

    wm_cont_dim = 2
    wm_num_discrete = 1
    env_is_delta = False
    if isinstance(env.action_space, spaces.Tuple):
        env_is_delta = True
        wm_cont_dim = 2  # model ignores trigger; discrete token captures executed action/no-op
        wm_num_discrete = 11  # NONE + 10 executed indices (-1..9) -> (0..10)
    elif isinstance(env.action_space, spaces.Box) and env.action_space.shape is not None:
        wm_cont_dim = int(env.action_space.shape[0])
        wm_num_discrete = 1

    # Rollout loss settings (V-JEPA 2 style)
    rollout_steps = int(os.environ.get("WM_ROLLOUT_STEPS", "2"))
    rollout_loss_coef = float(os.environ.get("WM_ROLLOUT_COEF", "1.0"))
    use_patch_tokens = os.environ.get("WM_USE_PATCH_TOKENS", "1") == "1"
    predictor_layers = int(os.environ.get("WM_PREDICTOR_LAYERS", "6"))
    predictor_heads = int(os.environ.get("WM_PREDICTOR_HEADS", "8"))

    # Initialize world model with V-JEPA 2 style architecture
    # Temporal and memory settings
    use_temporal_encoder = os.environ.get("WM_USE_TEMPORAL", "1") == "1"
    num_frames = int(os.environ.get("WM_NUM_FRAMES", "4"))
    use_ewc = os.environ.get("WM_USE_EWC", "1") == "1"
    ewc_lambda = float(os.environ.get("WM_EWC_LAMBDA", "100.0"))
    ewc_consolidate_every = int(os.environ.get("WM_EWC_CONSOLIDATE_EVERY", "500"))
    consolidation_buffer_size = int(os.environ.get("WM_CONSOLIDATION_SIZE", "500"))
    
    world_model = WorldModel(
        grid_size=env.grid_size,
        latent_dim=latent_dim,
        continuous_action_dim=wm_cont_dim,
        num_discrete_actions=wm_num_discrete,
        device=device,
        planning_horizon=planning_horizon,
        use_planner=use_planner,
        lr=lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        mask_ratio=mask_ratio,
        mask_loss_coef=mask_loss_coef,
        variance_coef=variance_coef,
        variance_target=variance_target,
        # V-JEPA 2 style settings
        rollout_steps=rollout_steps,
        rollout_loss_coef=rollout_loss_coef,
        use_patch_tokens=use_patch_tokens,
        predictor_layers=predictor_layers,
        predictor_heads=predictor_heads,
        # Temporal encoding for motion awareness
        use_temporal_encoder=use_temporal_encoder,
        num_frames=num_frames,
        # EWC for catastrophic forgetting prevention
        use_ewc=use_ewc,
        ewc_lambda=ewc_lambda,
        ewc_consolidate_every=ewc_consolidate_every,
        consolidation_buffer_size=consolidation_buffer_size,
    )

    # Curiosity module for exploration
    curiosity = StateCounter(reward_scale=curiosity_scale) if use_curiosity else None

    # 3D PCA Latent Visualizer (like RTAC)
    show_viz = os.environ.get("JEPA_SHOW_VIZ", "1") == "1"
    visualizer: Optional[JEPAVisualizer] = None
    if show_viz:
        try:
            visualizer = JEPAVisualizer(
                width=1000,
                height=700,
                title=f"World Model Latent Space - {agent.game_id}",
            )
            visualizer.start()
            logger.info("Started 3D PCA Latent Visualizer - click points to see decoded grids")
        except Exception as e:
            logger.warning(f"Could not start visualizer: {e}")
            visualizer = None

    # Bootstrap
    env.reset()

    logger.info(
        "World Model config: latent_dim=%s planning_horizon=%s use_planner=%s use_curiosity=%s",
        latent_dim, planning_horizon, use_planner, use_curiosity,
    )

    step = 0
    total_wins = 0
    last_train_stats = {}  # Persist last training stats for logging

    prev_cx: int | None = None
    prev_cy: int | None = None

    def _grid_with_cursors(g: np.ndarray, *, cur_cx: int, cur_cy: int, prev_xy: tuple[int, int] | None) -> np.ndarray:
        if not mark_cursor:
            return g
        out = g.copy()
        try:
            if mark_prev_cursor and prev_xy is not None:
                px, py = prev_xy
                px = int(np.clip(px, 0, env.grid_size - 1))
                py = int(np.clip(py, 0, env.grid_size - 1))
                out[py, px] = np.uint8(prev_cursor_token)
            out[int(cur_cy), int(cur_cx)] = np.uint8(cursor_token)
        except Exception:
            return g
        return out

    # Track imagination mode state
    in_imagination = False
    imagination_initial_grid = None
    
    try:
        while not agent._quit_event.is_set():
            step += 1
            
            # ==================== IMAGINATION MODE ====================
            # Handle imagination mode: let user play in world model's predictions
            if agent.imagination_mode:
                if not in_imagination:
                    # Just entered imagination mode - initialize
                    current_grid = env.last_grid
                    if current_grid is not None:
                        imagination_initial_grid = current_grid.copy()
                        imagined_grid = world_model.enter_imagination(current_grid)
                        agent.imagination_grid = imagined_grid.tolist()
                        agent.imagination_step_count = 0
                        in_imagination = True
                        logger.info("Entered imagination mode - playing in world model's mind")
                
                # Process imagination action if pending
                if in_imagination and agent.imagination_action is not None:
                    cont_action = np.array(agent.imagination_action["cont_action"], dtype=np.float32)
                    disc_action = int(agent.imagination_action["disc_action"])
                    agent.imagination_action = None  # Clear the pending action
                    
                    # Take a step in imagination
                    try:
                        predicted_grid, pred_reward, win_prob = world_model.imagine_step(
                            cont_action, disc_action
                        )
                        agent.imagination_grid = predicted_grid.tolist()
                        agent.imagination_reward = pred_reward
                        agent.imagination_win_prob = win_prob
                        agent.imagination_step_count += 1
                        logger.debug(f"Imagination step {agent.imagination_step_count}: reward={pred_reward:.2f}, win_prob={win_prob:.2f}")
                    except Exception as e:
                        logger.warning(f"Imagination step error: {e}")
                
                # Always send UI update during imagination mode
                if callback:
                    callback.on_step({
                        "imagination_mode": True,
                        "imagination_reward": agent.imagination_reward,
                        "imagination_win_prob": agent.imagination_win_prob,
                    })
                
                # In imagination mode, skip regular training loop
                time.sleep(0.02)  # Small delay to not hog CPU
                continue
            else:
                if in_imagination:
                    # Just exited imagination mode
                    world_model.exit_imagination()
                    in_imagination = False
                    agent.imagination_grid = None
                    logger.info("Exited imagination mode - back to reality")

            # Get current grid
            current_grid = env.last_grid
            if current_grid is None:
                continue
            # Cursor positions for temporal cursor marking
            cx0 = int(np.clip(int(round(float(agent.cursor_x))), 0, env.grid_size - 1))
            cy0 = int(np.clip(int(round(float(agent.cursor_y))), 0, env.grid_size - 1))
            if prev_cx is None or prev_cy is None:
                prev_cx, prev_cy = cx0, cy0
            state_for_model = _grid_with_cursors(current_grid, cur_cx=cx0, cur_cy=cy0, prev_xy=(prev_cx, prev_cy))

            # Initialize rollout state for honest visualization (first step only)
            if step == 0 or world_model._rollout_z is None:
                world_model.init_rollout_state(state_for_model)

            # Choose action
            if step < exploration_steps or not use_planner or world_model.wins_seen == 0:
                # Pure exploration: random actions + curiosity
                if env_is_delta:
                    # Sample cursor deltas; choose whether to execute a discrete action this step.
                    ax, ay = np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
                    do_trigger = np.random.random() < trigger_prob
                    trigger = 1.0 if do_trigger else -1.0
                    disc_idx = int(np.random.randint(0, 10))
                    cont3 = np.array([ax, ay, trigger], dtype=np.float32)
                    action = (cont3, disc_idx)
                    cont_for_model = np.array([ax, ay], dtype=np.float32)
                    disc_token_for_model = 0  # will be overwritten from env info after step
                else:
                    # Continuous-only env
                    action = np.random.uniform(-1.0, 1.0, size=(wm_cont_dim,)).astype(np.float32)
                    cont_for_model = action.astype(np.float32)
                    disc_token_for_model = 0
            else:
                # Plan using world model
                cont_plan, disc_token, expected_reward = world_model.plan_action(state_for_model)
                if env_is_delta:
                    # Convert disc token (0..10) back to env delta action (cont3 + disc_idx)
                    if disc_token > 0:
                        trigger = 1.0
                        disc_idx = int(disc_token - 1)
                    else:
                        trigger = -1.0
                        disc_idx = 0
                    cont3 = np.array([float(cont_plan[0]), float(cont_plan[1]), float(trigger)], dtype=np.float32)
                    action = (cont3, disc_idx)
                    cont_for_model = np.array([float(cont_plan[0]), float(cont_plan[1])], dtype=np.float32)
                    disc_token_for_model = int(disc_token)
                else:
                    action = cont_plan.astype(np.float32)
                    cont_for_model = cont_plan.astype(np.float32)
                    disc_token_for_model = 0

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            info = info or {}

            # Check for win
            from ...structs import GameState
            win = bool(info.get("game_success", False)) or (
                hasattr(agent, "frames") and agent.frames and agent.frames[-1].state == GameState.WIN
            )
            game_over = bool(info.get("game_over", False)) or (
                hasattr(agent, "frames") and agent.frames and agent.frames[-1].state == GameState.GAME_OVER
            )

            if win:
                total_wins += 1
                logger.info("WIN detected! Total wins: %d", total_wins)

            # Get next grid
            next_grid = env.last_grid
            if next_grid is None:
                continue
            cx1 = int(np.clip(int(round(float(agent.cursor_x))), 0, env.grid_size - 1))
            cy1 = int(np.clip(int(round(float(agent.cursor_y))), 0, env.grid_size - 1))
            # For next state, previous cursor is current state's cursor.
            next_for_model = _grid_with_cursors(next_grid, cur_cx=cx1, cur_cy=cy1, prev_xy=(cx0, cy0))
            # Update prev cursor for next loop (t-1 cursor at the next state's time).
            prev_cx, prev_cy = cx0, cy0

            # Add curiosity bonus
            curiosity_reward = 0.0
            if curiosity is not None:
                curiosity_reward = curiosity.get_bonus(next_grid)
                curiosity.update(next_grid)

            # If env reports the executed discrete action, prefer that for training.
            if env_is_delta:
                try:
                    final_action_idx = int(info.get("final_action_idx", -1))
                except Exception:
                    final_action_idx = -1
                disc_token_for_model = int(final_action_idx + 1)  # -1..9 -> 0..10

            # Store transition
            world_model.add_transition(
                state=state_for_model,
                cont_action=cont_for_model,
                disc_action=int(disc_token_for_model),
                next_state=next_for_model,
                reward=reward + curiosity_reward,
                done=bool(terminated or truncated or win or game_over),
                win=win,
            )

            # Feed latent embeddings to 3D visualizer (like RTAC)
            if visualizer is not None and step % 5 == 0:  # Every 5 steps to reduce overhead
                try:
                    with torch.no_grad():
                        # Encode current state to latent (visual embedding)
                        grid_t = torch.from_numpy(state_for_model).unsqueeze(0).to(device)
                        z = world_model.encoder(grid_t)  # (1, latent_dim) or (1, num_patches+1, latent_dim)
                        if z.dim() == 3:
                            visual_emb = z[:, 0]  # Use CLS token
                        else:
                            visual_emb = z
                        
                        # Get action embedding and combine with visual (joint embedding)
                        action_idx = disc_token_for_model if disc_token_for_model > 0 else 0
                        action_t = torch.tensor([action_idx], device=device)
                        action_emb = world_model.predictor.discrete_embed(action_t)  # (1, 64)
                        
                        # Combine: pad action_emb to latent_dim or project
                        if action_emb.shape[-1] < visual_emb.shape[-1]:
                            # Pad action embedding to match visual dim
                            padded_action = torch.zeros_like(visual_emb)
                            padded_action[:, :action_emb.shape[-1]] = action_emb
                            joint_emb = visual_emb + padded_action
                        else:
                            joint_emb = visual_emb + action_emb[:, :visual_emb.shape[-1]]
                        
                        embedding = joint_emb.squeeze(0).cpu().numpy()
                        
                        # Predict next grid using HONEST multi-step rollout
                        # This shows true model quality (like imagination mode)
                        # Re-anchors every 10 steps to show "fresh restart" quality
                        predicted_grid = world_model.predict_next_grid_from_rollout(
                            cont_for_model,
                            action_idx,
                            real_grid=state_for_model,  # For re-anchoring
                            reanchor_every=10,  # Reset every 10 steps to show both
                        )
                        
                        # Determine if action had effect (for green/red coloring)
                        # Use the environment's "action_had_effect" field which indicates
                        # if the discrete action actually changed the game state
                        # GREEN = action changed game state (effective)
                        # RED = action had no effect (wasted/ineffective)
                        env_had_effect = bool(info.get("action_had_effect", False)) if info else False
                        had_effect = (
                            env_had_effect or  # Action changed the game state
                            win or  # Win is always good
                            reward > 0  # Any positive reward
                        )
                        
                        visualizer.add_embedding(
                            embedding=embedding,
                            action_idx=action_idx,
                            had_effect=had_effect,
                            grid=state_for_model,
                            cursor=(float(cx0), float(cy0)),
                            predicted_grid=predicted_grid,
                            next_grid=next_for_model,
                        )
                        # Debug: log first few had_effect values
                        if step < 100 and step % 20 == 0:
                            logger.debug(f"VIZ: step={step} reward={reward:.2f} env_had_effect={env_had_effect} had_effect={had_effect}")
                except Exception as e:
                    if step < 50:  # Log early errors for debugging
                        logger.warning(f"Visualizer error at step {step}: {e}")

            # Train world model
            if step % train_every == 0:
                train_stats = world_model.train_step()
                if train_stats:
                    last_train_stats = train_stats  # Persist for logging
                    # Update visualizer with training metrics
                    if visualizer is not None:
                        try:
                            visualizer.add_metrics(
                                loss=train_stats.get("predictor_loss", 0),
                                accuracy=1.0 - min(1.0, train_stats.get("predictor_loss", 1.0)),  # Approx accuracy
                                latent_loss=train_stats.get("reward_loss", 0),
                            )
                        except Exception:
                            pass

            # Update callback
            info["reward"] = float(reward)
            info["curiosity_reward"] = float(curiosity_reward)
            info["total_wins"] = total_wins
            info["wins_seen"] = world_model.wins_seen
            if step >= exploration_steps and use_planner and world_model.wins_seen > 0:
                info["wm_expected_reward"] = float(expected_reward) if "expected_reward" in locals() else None
            callback.on_step(info)

            # In full game-solving mode, restart quickly after WIN/GAME_OVER so exploration continues.
            if win or game_over:
                try:
                    if world_model.planner is not None:
                        world_model.planner.reset()
                except Exception:
                    pass
                prev_cx = None
                prev_cy = None
                try:
                    env.reset()
                    # Re-init rollout state after reset for honest visualization
                    world_model._rollout_z = None
                except Exception:
                    # If reset fails (e.g., transient API issue), keep going.
                    pass

            # Optional throttle
            # Speed slider: 1.0 = fastest (no sleep), 0.0 = slowest (1 sec sleep)
            if agent.training_speed < 1.0:
                sleep_time = (1.0 - agent.training_speed)  # Up to 1.0s sleep
                time.sleep(sleep_time)

            # Logging
            if log_every > 0 and step % log_every == 0:
                stats_str = ""
                if last_train_stats:
                    stats_str = f" pred_loss={last_train_stats.get('predictor_loss', 0):.4f}"
                    stats_str += f" win_loss={last_train_stats.get('win_loss', 0):.4f}"
                    stats_str += f" reward_loss={last_train_stats.get('reward_loss', 0):.4f}"
                    stats_str += f" dec_loss={last_train_stats.get('decoder_loss', 0):.4f}"

                logger.info(
                    "step=%s reward=%.3f curiosity=%.4f wins=%d buffer=%d%s",
                    step, float(reward), curiosity_reward, total_wins,
                    len(world_model.buffer), stats_str,
                )

                if curiosity:
                    cstats = curiosity.get_stats()
                    logger.info("  curiosity: unique_states=%d", cstats["unique_states"])

    except KeyboardInterrupt:
        logger.info("Training interrupted")
    finally:
        # Stop visualizer
        if visualizer is not None:
            try:
                visualizer.stop()
                logger.info("Stopped 3D PCA Visualizer")
            except Exception:
                pass
        
        agent.cleanup()
        # Save world model
        try:
            world_model.save(f"world_model_{agent.game_id}.pth")
        except Exception as e:
            logger.error(f"Failed to save world model: {e}")

        logger.info("Training complete: %s steps, %s wins", step, total_wins)
