from __future__ import annotations

"""
Clean online TD(0) actor-critic trainer (continuing task, no terminals, no resets).

This replaces the previous RTAC implementation with a minimal, stable algorithm:
  - vector observation (rel goal, velocity, prev action, dist, goal_present)
  - tanh-squashed Gaussian policy (reparameterized)
  - per-step TD(0) critic update
  - per-step policy-gradient actor update with entropy bonus

Designed for the ARC-AGI-3 cursor-to-goal training loop used by PPOAgent + manual UI.
"""

import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..envs.arc_env import ARCGymEnv
from ..viz.live_visualizer import LiveVisualizerCallback

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
    Entry point used by PPOAgent.main() when PPO_TRAINER=rtac.
    """
    logger.info("Starting RTAC (online TD actor-critic) for game %s", agent.game_id)

    device = _pick_device(os.environ.get("PPO_RTAC_DEVICE", "auto"))
    logger.info("Using device: %s", device)

    # Force a stable control mode for this trainer.
    if os.environ.get("PPO_ACTION_MODE", "").strip().lower() not in ("vel", "velocity", "cursor_vel"):
        logger.warning(
            "RTAC trainer expects PPO_ACTION_MODE=vel. Current PPO_ACTION_MODE=%r. "
            "Set [trainer].action_mode='vel' in ppo_config.toml.",
            os.environ.get("PPO_ACTION_MODE"),
        )

    cfg = TrainCfg(
        gamma=float(os.environ.get("PPO_GAMMA", "0.97")),
        actor_lr=float(os.environ.get("PPO_ACTOR_LR", os.environ.get("PPO_LR", "3e-4"))),
        critic_lr=float(os.environ.get("PPO_CRITIC_LR", "8e-4")),
        entropy_coef=float(os.environ.get("PPO_ENTROPY_COEF", "1e-3")),
        grad_clip_norm=float(os.environ.get("PPO_GRAD_CLIP_NORM", os.environ.get("PPO_MAX_GRAD_NORM", "5.0"))),
        log_every=int(os.environ.get("PPO_LOG_EVERY", "50")),
        warmup_steps=int(os.environ.get("PPO_WARMUP_STEPS", "0")),
    )

    env = ARCGymEnv(agent, max_steps=1_000_000)
    callback = LiveVisualizerCallback(agent.gui_process, agent)
    callback._quit_event = agent._quit_event

    # Bootstrap server frame once (env.reset handles it based on PPO_DISABLE_RESET_BOOTSTRAP).
    env.reset()

    obs_dim = int(env.vector_obs().shape[0])
    act_dim = int(getattr(env.action_space, "shape", (2,))[0])

    actor = MLP(obs_dim, act_dim, hidden=64).to(device)
    critic = MLP(obs_dim, 1, hidden=64).to(device)
    log_std = nn.Parameter(torch.full((act_dim,), -0.5, device=device))

    actor_opt = torch.optim.Adam(list(actor.parameters()) + [log_std], lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    agent.model = actor  # for saving/debug UI compatibility

    logger.info(
        "RTAC online-AC config: action_mode=%s reward_mode=%s obs_dim=%s act_dim=%s "
        "gamma=%s actor_lr=%s critic_lr=%s entropy_coef=%s warmup_steps=%s",
        getattr(env, "action_mode", None),
        getattr(env, "reward_mode", None),
        obs_dim,
        act_dim,
        cfg.gamma,
        cfg.actor_lr,
        cfg.critic_lr,
        cfg.entropy_coef,
        cfg.warmup_steps,
    )

    step = 0
    last_log_ts = time.time()

    try:
        while not agent._quit_event.is_set():
            step += 1

            obs = env.vector_obs()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            if step <= cfg.warmup_steps:
                action = np.random.uniform(-1.0, 1.0, size=(act_dim,)).astype(np.float32)
                logp_t = None
                ent_t = None
            else:
                mu = actor(obs_t)
                a_t, logp_t, ent_t = tanh_normal_rsample_and_logprob(mu, log_std)
                action = a_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

            # Step environment (continuing task: terminated/truncated ignored)
            _obs_img, reward, _terminated, _truncated, info = env.step(action)
            info = info or {}
            info["reward"] = float(reward)
            callback.on_step(info)

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
                nn.utils.clip_grad_norm_(list(actor.parameters()) + [log_std], cfg.grad_clip_norm)
                actor_opt.step()

            # Optional UI-driven throttle
            if agent.training_speed > 0:
                time.sleep(agent.training_speed * 0.1)

            if cfg.log_every > 0 and step % cfg.log_every == 0:
                with torch.no_grad():
                    std_now = torch.exp(torch.clamp(log_std, -5.0, 2.0)).mean().item()
                logger.info(
                    "step=%s reward=%.3f dist_norm=%s goal_dist=%s hit_wall=%s std~%.3f",
                    step,
                    float(reward),
                    info.get("goal_dist_norm"),
                    info.get("goal_dist"),
                    info.get("hit_wall"),
                    float(std_now),
                )

    except KeyboardInterrupt:
        logger.info("Training interrupted")
    finally:
        agent.cleanup()
        # Save actor + critic (best-effort)
        try:
            torch.save(actor.state_dict(), f"rtac_actor_{agent.game_id}.pth")
            torch.save(critic.state_dict(), f"rtac_critic_{agent.game_id}.pth")
            torch.save({"log_std": log_std.detach().cpu()}, f"rtac_logstd_{agent.game_id}.pth")
        except Exception:
            pass


