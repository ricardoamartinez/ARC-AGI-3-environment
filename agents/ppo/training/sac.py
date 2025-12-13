from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..envs.arc_env import ARCGymEnv
from ..policy.sac_models import SACActor, SACCritic
from ..viz.live_visualizer import LiveVisualizerCallback

logger = logging.getLogger()

if TYPE_CHECKING:
    from ..agent import PPOAgent


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    act: torch.Tensor
    rew: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], action_dim: int = 2):
        self.capacity = int(capacity)
        self.action_dim = int(action_dim)
        self.obs = np.zeros((self.capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=np.uint8)
        self.act = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rew = np.zeros((self.capacity,), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: float) -> None:
        self.obs[self.idx] = obs
        self.next_obs[self.idx] = next_obs
        self.act[self.idx] = act
        self.rew[self.idx] = float(rew)
        self.done[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def clear(self) -> None:
        self.idx = 0
        self.size = 0

    def sample(self, batch_size: int, device: str) -> ReplayBatch:
        idxs = np.random.randint(0, self.size, size=(batch_size,))

        # Stored as uint8 HWC; models expect float32 BCHW in [0,1]
        obs = torch.as_tensor(self.obs[idxs], device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        next_obs = (
            torch.as_tensor(self.next_obs[idxs], device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        )
        act = torch.as_tensor(self.act[idxs], device=device, dtype=torch.float32)
        rew = torch.as_tensor(self.rew[idxs], device=device, dtype=torch.float32)
        done = torch.as_tensor(self.done[idxs], device=device, dtype=torch.float32)
        return ReplayBatch(obs=obs, act=act, rew=rew, next_obs=next_obs, done=done)


def _soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def run_sac_training(agent: "PPOAgent") -> None:
    """
    Soft Actor-Critic (SAC) trainer for fast, stable continuous control.

    IMPORTANT: This trainer expects a continuous 2D action space. Use:
      PPO_ACTION_MODE=target_pos
    so the env accepts a Box(2,) action and maps it to a smooth cursor trajectory.
    """

    logger.info(f"Starting SAC Training for game {agent.game_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Make UI responsive: optional async updates (learner thread) + lightweight inference device.
    async_updates = os.environ.get("PPO_SAC_ASYNC_UPDATES", "1") == "1"
    infer_device = os.environ.get("PPO_SAC_INFER_DEVICE", "cpu").strip().lower()
    if infer_device not in ("cpu", "cuda"):
        infer_device = "cpu"
    if infer_device == "cuda" and not torch.cuda.is_available():
        infer_device = "cpu"
    sync_every_updates = int(os.environ.get("PPO_SAC_SYNC_EVERY_UPDATES", "50"))
    sync_every_updates = max(1, sync_every_updates)

    env = ARCGymEnv(agent, max_steps=1_000_000)
    # SAC here is implemented for continuous actions (target_pos mode).
    # In delta mode the env uses a Tuple action space and .shape is None, so check via isinstance.
    from gymnasium import spaces

    if not (isinstance(env.action_space, spaces.Box) and env.action_space.shape is not None):
        raise RuntimeError(
            "SAC requires PPO_ACTION_MODE=target_pos so env.action_space is a Box. "
            f"Got action_mode={getattr(env, 'action_mode', None)} action_space={env.action_space}."
        )
    action_dim = int(env.action_space.shape[0])

    # Hyperparams
    lr = float(os.environ.get("PPO_SAC_LR", os.environ.get("PPO_LR", "3e-4")))
    gamma = float(os.environ.get("PPO_SAC_GAMMA", "0.99"))
    tau = float(os.environ.get("PPO_SAC_TAU", "0.005"))
    batch_size = int(os.environ.get("PPO_SAC_BATCH_SIZE", "256"))
    # Note: replay buffer stores full uint8 observations (64x64xC). Large values can consume huge RAM.
    # Keep a conservative default; override via PPO_SAC_BUFFER_SIZE as needed.
    buffer_size = int(os.environ.get("PPO_SAC_BUFFER_SIZE", "10000"))
    warmup = int(os.environ.get("PPO_SAC_WARMUP", "2000"))
    updates_per_step = int(os.environ.get("PPO_SAC_UPDATES_PER_STEP", "1"))
    deterministic_actions = os.environ.get("PPO_DETERMINISTIC_ACTIONS", "0") == "1"
    deterministic_after = int(os.environ.get("PPO_SAC_DETERMINISTIC_AFTER_STEPS", "0"))

    # Entropy temperature
    auto_alpha = os.environ.get("PPO_SAC_AUTO_ALPHA", "1") == "1"
    target_entropy = float(os.environ.get("PPO_SAC_TARGET_ENTROPY", str(-2.0)))
    init_alpha = float(os.environ.get("PPO_SAC_ALPHA", "0.2"))

    # Replay buffer stores uint8 HWC to save RAM
    obs0, _ = env.reset()
    obs_shape = obs0.shape  # (H,W,C)
    rb = ReplayBuffer(capacity=buffer_size, obs_shape=obs_shape, action_dim=action_dim)
    rb_lock = threading.Lock()

    actor = SACActor(env.observation_space, action_dim=action_dim).to(device)
    q1 = SACCritic(env.observation_space, action_dim=action_dim).to(device)
    q2 = SACCritic(env.observation_space, action_dim=action_dim).to(device)
    q1_t = SACCritic(env.observation_space, action_dim=action_dim).to(device)
    q2_t = SACCritic(env.observation_space, action_dim=action_dim).to(device)
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=lr, eps=1e-5)
    q_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr, eps=1e-5)

    log_alpha = torch.tensor(np.log(init_alpha), device=device, dtype=torch.float32, requires_grad=True)
    alpha_opt = optim.Adam([log_alpha], lr=lr, eps=1e-5) if auto_alpha else None

    # Inference-only actor (optionally on CPU to avoid any GPU contention/lag)
    actor_infer = SACActor(env.observation_space, action_dim=action_dim).to(infer_device)
    actor_infer.load_state_dict({k: v.detach().to(infer_device) for k, v in actor.state_dict().items()})
    actor_infer.eval()

    callback = LiveVisualizerCallback(agent.gui_process, agent)
    callback._quit_event = agent._quit_event

    log_every = int(os.environ.get("PPO_LOG_EVERY", "50"))
    logger.info(
        "SAC config: action_mode=%s lr=%s gamma=%s tau=%s batch=%s buf=%s warmup=%s upd/step=%s auto_alpha=%s target_entropy=%s",
        getattr(env, "action_mode", "unknown"),
        lr,
        gamma,
        tau,
        batch_size,
        buffer_size,
        warmup,
        updates_per_step,
        auto_alpha,
        target_entropy,
    )

    obs = obs0
    step_idx = 0
    ema_goal_dist: float | None = None
    ema_beta = float(os.environ.get("PPO_GOAL_EMA_BETA", "0.98"))
    hit_window = int(os.environ.get("PPO_GOAL_HIT_WINDOW", "500"))
    recent_hits: list[int] = []
    last_goal_version: int | None = None
    clear_on_goal_change = os.environ.get("PPO_SAC_CLEAR_BUFFER_ON_GOAL_CHANGE", "1") == "1"
    goal_change_burst_updates = int(os.environ.get("PPO_SAC_GOAL_CHANGE_BURST_UPDATES", "200"))
    burst_remaining = 0
    learner_updates = 0

    def _sync_actor_to_infer() -> None:
        nonlocal learner_updates
        if infer_device == device:
            actor_infer.load_state_dict(actor.state_dict())
        else:
            actor_infer.load_state_dict({k: v.detach().to(infer_device) for k, v in actor.state_dict().items()})
        actor_infer.eval()
        learner_updates = 0

    def _one_update() -> None:
        nonlocal learner_updates
        with rb_lock:
            if rb.size < warmup:
                return
            b = rb.sample(batch_size=batch_size, device=device)
        alpha = float(torch.exp(log_alpha).detach().cpu().item())

        with torch.no_grad():
            next_act_out = actor.sample(b.next_obs, deterministic=False)
            next_a = next_act_out.action
            next_logp = next_act_out.log_prob
            q1_next = q1_t(b.next_obs, next_a)
            q2_next = q2_t(b.next_obs, next_a)
            q_next = torch.minimum(q1_next, q2_next) - alpha * next_logp
            target_q = b.rew + (1.0 - b.done) * gamma * q_next

        q1_pred = q1(b.obs, b.act)
        q2_pred = q2(b.obs, b.act)
        q_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), max_norm=1.0)
        q_opt.step()

        act_out2 = actor.sample(b.obs, deterministic=False)
        a2 = act_out2.action
        logp2 = act_out2.log_prob
        q1_pi = q1(b.obs, a2)
        q2_pi = q2(b.obs, a2)
        q_pi = torch.minimum(q1_pi, q2_pi)
        actor_loss = (alpha * logp2 - q_pi).mean()

        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        actor_opt.step()

        if auto_alpha and alpha_opt is not None:
            alpha_loss = -(log_alpha * (logp2.detach() + target_entropy)).mean()
            alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            alpha_opt.step()

        _soft_update(q1_t, q1, tau=tau)
        _soft_update(q2_t, q2, tau=tau)

        learner_updates += 1
        if learner_updates >= sync_every_updates:
            _sync_actor_to_infer()

    def _learner_loop() -> None:
        # Background learner thread: updates never block UI/render loop.
        nonlocal burst_remaining
        try:
            while not agent._quit_event.is_set():
                # Handle burst budget (goal change)
                if burst_remaining > 0:
                    n = int(min(burst_remaining, 2000))
                    for _ in range(n):
                        _one_update()
                    burst_remaining = max(0, burst_remaining - n)
                else:
                    # steady-state updates
                    for _ in range(max(0, int(updates_per_step))):
                        _one_update()

                # Yield CPU a bit so the UI stays snappy even on slower machines.
                time.sleep(0.001)
        except Exception as e:
            logger.exception("Async SAC learner crashed: %s", e)

    try:
        learner_thread = None
        if async_updates:
            learner_thread = threading.Thread(target=_learner_loop, daemon=True)
            learner_thread.start()
        while not agent._quit_event.is_set():
            step_idx += 1

            # Action selection
            # Env returns HWC uint8; model expects BCHW float in [0,1]
            obs_t = (
                torch.as_tensor(np.transpose(obs, (2, 0, 1)), device=infer_device, dtype=torch.float32)
                .unsqueeze(0)
                / 255.0
            )
            det_now = deterministic_actions or (deterministic_after > 0 and step_idx >= deterministic_after)
            with torch.no_grad():
                act_out = actor_infer.sample(obs_t, deterministic=det_now)
            act = act_out.action.squeeze(0).cpu().numpy().astype(np.float32)

            next_obs, reward, terminated, truncated, info = env.step(act)
            callback.on_step(info)

            # Goal-change adaptation:
            # When the user clicks a new goal, we want learning to adapt immediately.
            # Optionally clear replay so old-goal data doesn't dominate gradients.
            gv = None
            if info is not None:
                try:
                    gv = int(info.get("goal_version", 0))
                except Exception:
                    gv = None
            if gv is not None:
                if last_goal_version is None:
                    last_goal_version = gv
                elif gv != last_goal_version:
                    logger.info(
                        "Goal changed (goal_version %s -> %s). %s",
                        last_goal_version,
                        gv,
                        "Clearing replay buffer for fast adaptation." if clear_on_goal_change else "Keeping replay buffer.",
                    )
                    last_goal_version = gv
                    ema_goal_dist = None
                    recent_hits.clear()
                    if clear_on_goal_change:
                        with rb_lock:
                            rb.clear()
                    # Do an immediate burst of updates on new-goal data (after we have at least a few samples).
                    # This is the closest thing to "just-in-time" adaptation in off-policy RL.
                    if goal_change_burst_updates > 0:
                        # We'll trigger the burst by temporarily inflating updates_per_step below.
                        burst_remaining = goal_change_burst_updates
            # Count down a burst budget as we perform extra updates.

            # Convergence diagnostics (works in both GUI and headless runs)
            gd = None
            ghit = 0
            if info:
                gd = info.get("goal_dist", None)
                ghit = 1 if bool(info.get("goal_hit", False)) else 0
            if gd is not None:
                gd_f = float(gd)
                ema_goal_dist = gd_f if ema_goal_dist is None else (ema_beta * ema_goal_dist + (1.0 - ema_beta) * gd_f)
                recent_hits.append(ghit)
                if len(recent_hits) > hit_window:
                    recent_hits.pop(0)

            # Env is continual; we treat done as 0 always (unless user changes it later)
            done = 0.0
            with rb_lock:
                rb.add(obs, act, float(reward), next_obs, done)
            obs = next_obs

            # Updates happen in the learner thread if enabled.

            if log_every > 0 and step_idx % log_every == 0:
                # Best-effort extra diagnostics (won't crash if missing)
                alpha = float(torch.exp(log_alpha).detach().cpu().item())
                hit_rate = (float(sum(recent_hits)) / float(len(recent_hits))) if recent_hits else 0.0
                action_str = ",".join([f"{float(x):.2f}" for x in act[: min(3, len(act))]])
                logger.info(
                    "step=%s reward=%.3f goal_dist=%s goal_dist_ema=%s hit_rate=%.3f speed=%.3f action=(%s) alpha=%.3f rb=%s",
                    step_idx,
                    float(reward),
                    info.get("goal_dist") if info else None,
                    (None if ema_goal_dist is None else float(ema_goal_dist)),
                    hit_rate,
                    float(info.get("cursor_speed", 0.0) if info else 0.0),
                    action_str,
                    alpha,
                    rb.size,
                )

            if agent.training_speed > 0:
                time.sleep(agent.training_speed * 0.1)

    except KeyboardInterrupt:
        logger.info("Training interrupted")
    finally:
        agent.cleanup()
        # Save actor + critics
        torch.save(actor.state_dict(), f"sac_actor_{agent.game_id}.pth")
        torch.save(q1.state_dict(), f"sac_q1_{agent.game_id}.pth")
        torch.save(q2.state_dict(), f"sac_q2_{agent.game_id}.pth")


