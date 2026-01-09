from __future__ import annotations

"""
Clean online TD(0) actor-critic trainer (continuing task, no terminals, no resets).

This replaces the previous RTAC implementation with a minimal, stable algorithm:
  - vector observation (rel goal, velocity, prev action, dist, goal_present)
  - tanh-squashed Gaussian policy (reparameterized)
  - per-step TD(0) critic update
  - per-step policy-gradient actor update with entropy bonus

Now enhanced with V-JEPA 2 inspired joint embedding for visual-action learning:
  - Learns which actions will be effective in each visual state
  - Biases action selection away from ineffective actions
  - Fast adaptation to avoid penalized actions

Designed for the ARC-AGI-3 cursor-to-goal training loop used by PPOAgent + manual UI.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..envs.arc_env import ARCGymEnv
from ..viz.live_visualizer import LiveVisualizerCallback
from ..viz.jepa_visualizer import JEPAVisualizer
from ..world_model.action_jepa import ActionJEPA

logger = logging.getLogger()


@dataclass
class TrainCfg:
    gamma: float
    actor_lr: float
    critic_lr: float
    entropy_coef: float
    grad_clip_norm: float
    log_every: int
    warmup_steps: int


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

        # Match the reference snippet init: orthogonal weights (gain=1), zero biases.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def tanh_normal_rsample_and_logprob(
    mu: torch.Tensor,
    log_std: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample u ~ Normal(mu, std) with rsample, then a=tanh(u).
    Returns: (a, log_prob(a), entropy_u)
    """
    # Reference clamp
    log_std = torch.clamp(log_std, -5.0, 2.0)
    std = torch.exp(log_std)
    normal = torch.distributions.Normal(mu, std)
    u = normal.rsample()
    a = torch.tanh(u)
    log_p_u = normal.log_prob(u).sum(dim=-1, keepdim=True)
    log_det = torch.log(1.0 - a.pow(2) + eps).sum(dim=-1, keepdim=True)
    log_p_a = log_p_u - log_det
    entropy = normal.entropy().sum(dim=-1, keepdim=True)
    return a, log_p_a, entropy


def _pick_device(name: str) -> torch.device:
    name = (name or "").strip().lower()
    if name in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_rtac_training(agent) -> None:
    """
    Entry point used by PPOAgent.main() when JEPA_TRAINER=rtac.
    """
    logger.info("Starting RTAC (online TD actor-critic) for game %s", agent.game_id)

    device = _pick_device(os.environ.get("JEPA_RTAC_DEVICE", "auto"))
    logger.info("Using device: %s", device)

    # Force a stable control mode for this trainer.
    if os.environ.get("JEPA_ACTION_MODE", "").strip().lower() not in ("vel", "velocity", "cursor_vel"):
        logger.warning(
            "RTAC trainer expects JEPA_ACTION_MODE=vel. Current JEPA_ACTION_MODE=%r. "
            "Set [trainer].action_mode='vel' in JEPA_config.toml.",
            os.environ.get("JEPA_ACTION_MODE"),
        )

    cfg = TrainCfg(
        gamma=float(os.environ.get("JEPA_GAMMA", "0.97")),
        actor_lr=float(os.environ.get("JEPA_ACTOR_LR", os.environ.get("JEPA_LR", "3e-4"))),
        critic_lr=float(os.environ.get("JEPA_CRITIC_LR", "8e-4")),
        entropy_coef=float(os.environ.get("JEPA_ENTROPY_COEF", "1e-3")),
        grad_clip_norm=float(os.environ.get("JEPA_GRAD_CLIP_NORM", os.environ.get("JEPA_MAX_GRAD_NORM", "5.0"))),
        log_every=int(os.environ.get("JEPA_LOG_EVERY", "50")),
        warmup_steps=int(os.environ.get("JEPA_WARMUP_STEPS", "0")),
    )

    env = ARCGymEnv(agent, max_steps=1_000_000)
    callback = LiveVisualizerCallback(agent.gui_process, agent)
    callback._quit_event = agent._quit_event

    # Bootstrap server frame once (env.reset handles it based on JEPA_DISABLE_RESET_BOOTSTRAP).
    env.reset()

    obs_dim = int(env.vector_obs().shape[0])
    # Hybrid action space: 3 continuous (ax, ay, trigger) + 10 discrete actions
    cont_act_dim = 3
    disc_act_dim = 10

    actor_cont = MLP(obs_dim, cont_act_dim, hidden=64).to(device)
    actor_disc = MLP(obs_dim, disc_act_dim, hidden=64).to(device)
    critic = MLP(obs_dim, 1, hidden=64).to(device)
    log_std = nn.Parameter(torch.full((cont_act_dim,), -0.5, device=device))

    actor_opt = torch.optim.Adam(list(actor_cont.parameters()) + list(actor_disc.parameters()) + [log_std], lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    
    # For compatibility
    actor = actor_cont

    agent.model = actor  # for saving/debug UI compatibility

    # === V-JEPA 2 Action Learning ===
    use_action_jepa = os.environ.get("JEPA_USE_ACTION_JEPA", "1") == "1"
    action_jepa_bias = float(os.environ.get("JEPA_JEPA_ACTION_BIAS", "2.0"))  # How much to bias toward effective actions
    jepa_train_every = int(os.environ.get("JEPA_JEPA_TRAIN_EVERY", "4"))
    jepa_warmup = int(os.environ.get("JEPA_JEPA_WARMUP", "100"))
    
    action_jepa: Optional[ActionJEPA] = None
    jepa_visualizer: Optional[JEPAVisualizer] = None
    show_jepa_viz = os.environ.get("JEPA_SHOW_JEPA_VIZ", "1") == "1"
    
    if use_action_jepa:
        action_jepa = ActionJEPA(
            grid_size=env.grid_size,
            embed_dim=128,
            num_discrete_actions=disc_act_dim,
            encoder_depth=4,
            encoder_heads=4,
            patch_size=8,
            device=str(device),
            ema_decay=0.996,
            learning_rate=1e-4,
        )
        logger.info("ActionJEPA initialized for visual-action learning (bias=%.2f)", action_jepa_bias)
        
        # Start the JEPA visualizer if enabled
        if show_jepa_viz:
            jepa_visualizer = JEPAVisualizer(
                width=800,
                height=600,
                title="ActionJEPA Embedding Visualizer",
            )
            jepa_visualizer.start()
            logger.info("JEPA Visualizer started - 3D PCA + click-to-decode")

    logger.info(
        "RTAC online-AC config: action_mode=%s reward_mode=%s obs_dim=%s cont_act_dim=%s disc_act_dim=%s "
        "gamma=%s actor_lr=%s critic_lr=%s entropy_coef=%s warmup_steps=%s",
        getattr(env, "action_mode", None),
        getattr(env, "reward_mode", None),
        obs_dim,
        cont_act_dim,
        disc_act_dim,
        cfg.gamma,
        cfg.actor_lr,
        cfg.critic_lr,
        cfg.entropy_coef,
        cfg.warmup_steps,
    )

    step = 0
    last_log_ts = time.time()
    prev_grid = None  # For ActionJEPA

    try:
        while not agent._quit_event.is_set():
            step += 1

            obs = env.vector_obs()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Get current grid for ActionJEPA
            current_grid = env.last_grid

            if step <= cfg.warmup_steps:
                cont_action = np.random.uniform(-1.0, 1.0, size=(cont_act_dim,)).astype(np.float32)
                disc_action = np.random.randint(0, disc_act_dim)
                logp_t = None
                ent_t = None
            else:
                # Continuous action
                mu = actor_cont(obs_t)
                a_t, logp_t, ent_t = tanh_normal_rsample_and_logprob(mu, log_std)
                cont_action = a_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
                
                # Discrete action (sample from softmax)
                disc_logits = actor_disc(obs_t)
                
                # === ActionJEPA: Bias action selection toward effective actions ===
                if action_jepa is not None and step > jepa_warmup and current_grid is not None:
                    try:
                        # Get predicted effectiveness for each action
                        action_probs = action_jepa.predict_action_effectiveness(
                            current_grid,
                            float(agent.cursor_x),
                            float(agent.cursor_y),
                        )
                        # Convert to log-space bias
                        effectiveness_bias = torch.tensor(
                            action_probs * action_jepa_bias,
                            dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        # Add bias to logits (encourage effective actions)
                        disc_logits = disc_logits + effectiveness_bias
                    except Exception as e:
                        pass  # Fallback to unbiased if JEPA fails
                
                disc_probs = F.softmax(disc_logits, dim=-1)
                disc_dist = torch.distributions.Categorical(probs=disc_probs)
                disc_action = int(disc_dist.sample().item())
                # Add discrete log prob to total
                disc_logp = disc_dist.log_prob(torch.tensor(disc_action, device=device))
                if logp_t is not None:
                    logp_t = logp_t + disc_logp.unsqueeze(0).unsqueeze(0)

            # Step environment with hybrid action (continuing task: terminated/truncated ignored)
            action = (cont_action, disc_action)
            _obs_img, reward, _terminated, _truncated, info = env.step(action)
            info = info or {}
            info["reward"] = float(reward)
            callback.on_step(info)

            # === ActionJEPA: Collect experience and train ===
            if action_jepa is not None and current_grid is not None:
                next_grid = env.last_grid
                if next_grid is not None:
                    # Get whether action had effect from info
                    had_effect = bool(info.get("action_had_effect", False))
                    final_action_idx = int(info.get("final_action_idx", -1))
                    
                    # Only add if a discrete action was taken
                    if final_action_idx >= 0:
                        action_jepa.add_experience(
                            state=current_grid,
                            action_idx=final_action_idx,
                            cursor_x=float(agent.cursor_x),
                            cursor_y=float(agent.cursor_y),
                            had_effect=had_effect,
                            next_state=next_grid,
                        )
                    
                    # Train ActionJEPA periodically
                    if step % jepa_train_every == 0:
                        jepa_stats = action_jepa.train_step(batch_size=32)
                        if jepa_stats:
                            info["jepa_loss"] = jepa_stats.get("total_loss", 0)
                            info["jepa_acc"] = jepa_stats.get("ema_acc", 0)
                            
                            # Send metrics to visualizer
                            if jepa_visualizer is not None:
                                jepa_visualizer.add_metrics(
                                    loss=jepa_stats.get("total_loss", 0),
                                    accuracy=jepa_stats.get("accuracy", 0),
                                    latent_loss=jepa_stats.get("latent_loss", 0),
                                )
                    
                    # Send embedding to visualizer for 3D PCA
                    if jepa_visualizer is not None and final_action_idx >= 0:
                        try:
                            # Get visual embedding for this state
                            with torch.no_grad():
                                state_t = torch.tensor(current_grid, dtype=torch.long, device=action_jepa.device).unsqueeze(0)
                                visual_emb = action_jepa.visual_encoder(state_t, return_all_tokens=False)
                                action_emb = action_jepa.action_embed(
                                    torch.tensor([final_action_idx], device=action_jepa.device),
                                    torch.tensor([[float(agent.cursor_x) / env.grid_size * 2 - 1,
                                                   float(agent.cursor_y) / env.grid_size * 2 - 1]], 
                                                  dtype=torch.float32, device=action_jepa.device)
                                )
                                # Combine visual + action embeddings for joint space
                                joint_emb = (visual_emb + action_emb).squeeze(0).cpu().numpy()
                                
                                # Decode grid from latent for visualization
                                predicted_grid = action_jepa.encode_and_decode(current_grid)
                                
                            jepa_visualizer.add_embedding(
                                embedding=joint_emb,
                                action_idx=final_action_idx,
                                had_effect=had_effect,
                                grid=current_grid,
                                cursor=(float(agent.cursor_x), float(agent.cursor_y)),
                                predicted_grid=predicted_grid,
                                next_grid=next_grid,
                            )
                        except Exception as e:
                            pass  # Don't crash training for viz errors
            
            prev_grid = current_grid

            # Critic TD(0) update
            next_obs = env.vector_obs()
            next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            r_t = torch.as_tensor([[float(reward)]], dtype=torch.float32, device=device)

            v = critic(obs_t)
            with torch.no_grad():
                v_next = critic(next_obs_t)
                target = r_t + cfg.gamma * v_next

            td_error = target - v
            critic_loss = 0.5 * td_error.pow(2).mean()

            critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg.grad_clip_norm)
            critic_opt.step()

            # Actor update (online policy gradient with baseline)
            if logp_t is not None:
                adv = td_error.detach()
                actor_loss = -(logp_t * adv).mean()
                if ent_t is not None and cfg.entropy_coef > 0.0:
                    actor_loss = actor_loss - cfg.entropy_coef * ent_t.mean()
                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(list(actor_cont.parameters()) + list(actor_disc.parameters()) + [log_std], cfg.grad_clip_norm)
                actor_opt.step()

            # Optional UI-driven throttle
            # Speed slider: 1.0 = fastest (no sleep), 0.0 = slowest (1 sec sleep)
            if agent.training_speed < 1.0:
                sleep_time = (1.0 - agent.training_speed)  # Up to 1.0s sleep
                time.sleep(sleep_time)

            if cfg.log_every > 0 and step % cfg.log_every == 0:
                with torch.no_grad():
                    std_now = torch.exp(torch.clamp(log_std, -5.0, 2.0)).mean().item()
                
                # Base logging
                log_msg = f"step={step} reward={float(reward):.3f} goal_dist={info.get('goal_dist')} hit_wall={info.get('hit_wall')} std~{float(std_now):.3f}"
                
                # Add ActionJEPA stats if available
                if action_jepa is not None:
                    log_msg += f" jepa_acc={action_jepa.effective_acc:.2%} jepa_buf={len(action_jepa.buffer_states)}"
                
                # Add ineffective penalty info
                ineff_penalty = info.get("ineffective_action_penalty", 0)
                if ineff_penalty > 0:
                    log_msg += f" ineff_penalty={ineff_penalty:.2f}"
                
                logger.info(log_msg)

    except KeyboardInterrupt:
        logger.info("Training interrupted")
    finally:
        # Stop visualizer
        if jepa_visualizer is not None:
            jepa_visualizer.stop()
            logger.info("JEPA Visualizer stopped")
            
        agent.cleanup()
        # Save actor + critic + ActionJEPA (best-effort)
        try:
            torch.save(actor_cont.state_dict(), f"rtac_actor_{agent.game_id}.pth")
            torch.save(actor_disc.state_dict(), f"rtac_actor_disc_{agent.game_id}.pth")
            torch.save(critic.state_dict(), f"rtac_critic_{agent.game_id}.pth")
            torch.save({"log_std": log_std.detach().cpu()}, f"rtac_logstd_{agent.game_id}.pth")
            
            # Save ActionJEPA
            if action_jepa is not None:
                action_jepa.save(f"action_jepa_{agent.game_id}.pth")
                logger.info(
                    "ActionJEPA saved: train_steps=%d buffer=%d acc=%.2f%%",
                    action_jepa.train_steps,
                    len(action_jepa.buffer_states),
                    action_jepa.effective_acc * 100,
                )
        except Exception as e:
            logger.warning(f"Failed to save models: {e}")


