"""
WorldModel - V-JEPA(-2)-inspired latent dynamics + goal-conditioned planning for ARC-AGI-3.

This module is intentionally "video-JEPA shaped", but adapted to ARC's discrete grid frames:

- **Encoder / Target Encoder**: Patchified ViT-like encoder with an EMA target (stop-grad teacher),
  similar to JEPA/V-JEPA's collapse prevention.
- **Action-conditioned Predictor**: Predicts next latent given current latent + action.
- **(Optional) Win predictor**: Learns P(win|z) from sparse win labels.
- **Goal-latent planning**: After observing at least one win, store the *winning latent(s)* and plan by
  minimizing the latent distance to a stored win goal (V-JEPA-2-AC "image-goal" style planning).

Notes:
- ARC grids may use palettes larger than 16. We default to 256 embeddings to be safe.
- The environment's "delta" action mode can yield an executed discrete action of -1..9.
  We represent this in the model as 0..10 via `disc_token = final_action_idx + 1`.
"""

import logging
from typing import Deque, Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import GridEncoder
from .predictor import ActionConditionedPredictor
from .win_predictor import WinPredictor, ValuePredictor
from .planner import CEMPlanner
from .mask_predictor import MaskedTokenPredictor

logger = logging.getLogger()


class TransitionBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        *,
        state: torch.Tensor,
        cont_action: torch.Tensor,
        disc_action: torch.Tensor,
        next_state: torch.Tensor,
        reward: float,
        done: bool,
        win: bool,
    ) -> None:
        self.buffer.append({
            "state": state, "cont_action": cont_action, "disc_action": disc_action, "next_state": next_state,
            "reward": reward, "done": done, "win": win,
        })

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in idxs]
        return {
            "state": torch.stack([b["state"] for b in batch]),
            "cont_action": torch.stack([b["cont_action"] for b in batch]),
            "disc_action": torch.stack([b["disc_action"] for b in batch]),
            "next_state": torch.stack([b["next_state"] for b in batch]),
            "reward": torch.tensor([float(b["reward"]) for b in batch], dtype=torch.float32),
            "done": torch.tensor([float(b["done"]) for b in batch], dtype=torch.float32),
            "win": torch.tensor([float(b["win"]) for b in batch], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.buffer)


class WorldModel(nn.Module):
    def __init__(
        self,
        grid_size: int = 64,
        patch_size: int = 8,
        num_colors: int = 256,
        latent_dim: int = 256,
        # Action representation: continuous + discrete-token.
        # - Continuous: cursor control (default 2D: ax, ay)
        # - Discrete token: executed discrete action (default 11: NONE + 10 indices from env "delta" mode)
        continuous_action_dim: int = 2,
        num_discrete_actions: int = 1,
        encoder_depth: int = 6,
        predictor_layers: int = 4,
        device: str = "cuda",
        lr: float = 1e-4,
        buffer_size: int = 10000,
        batch_size: int = 32,
        planning_horizon: int = 10,
        use_planner: bool = True,
        planner_num_samples: int = 100,
        planner_num_elites: int = 10,
        planner_iterations: int = 5,
        ema_momentum: float = 0.99,
        freeze_encoder_after: int = 0,
        win_goal_maxlen: int = 32,
        # Optional V-JEPA-style masked denoising loss on patch tokens.
        mask_ratio: float = 0.0,
        mask_loss_coef: float = 1.0,
        # Optional JEPA-style anti-collapse regularizer (batch variance floor).
        # When enabled, penalizes low variance in the latent across the batch.
        variance_coef: float = 0.0,
        variance_target: float = 1.0,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.continuous_action_dim = int(continuous_action_dim)
        self.num_discrete_actions = int(num_discrete_actions)
        self.batch_size = batch_size
        self.use_planner = use_planner
        self.freeze_encoder_after = int(freeze_encoder_after)
        self._encoder_frozen = False
        self.mask_ratio = float(mask_ratio)
        self.mask_loss_coef = float(mask_loss_coef)
        self.variance_coef = float(variance_coef)
        self.variance_target = float(variance_target)

        self.encoder = GridEncoder(grid_size, patch_size, num_colors, latent_dim, encoder_depth).to(self.device)
        self.target_encoder = GridEncoder(grid_size, patch_size, num_colors, latent_dim, encoder_depth).to(self.device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.mask_predictor: MaskedTokenPredictor | None = None
        if self.mask_ratio > 0.0:
            self.mask_predictor = MaskedTokenPredictor(dim=latent_dim, hidden_dim=latent_dim * 2).to(self.device)

        self.predictor = ActionConditionedPredictor(
            latent_dim=latent_dim,
            continuous_dim=self.continuous_action_dim,
            num_discrete_actions=self.num_discrete_actions,
            hidden_dim=latent_dim * 2,
            num_layers=predictor_layers,
        ).to(self.device)
        self.win_predictor = WinPredictor(latent_dim, latent_dim).to(self.device)
        self.value_predictor = ValuePredictor(latent_dim, latent_dim).to(self.device)

        if use_planner:
            self.planner = CEMPlanner(
                continuous_dim=self.continuous_action_dim,
                num_discrete_actions=self.num_discrete_actions,
                horizon=planning_horizon,
                num_samples=planner_num_samples,
                num_elites=planner_num_elites,
                num_iterations=planner_iterations,
            )
        else:
            self.planner = None

        self.buffer = TransitionBuffer(buffer_size)
        enc_params = list(self.encoder.parameters())
        if self.mask_predictor is not None:
            enc_params += list(self.mask_predictor.parameters())
        self.encoder_opt = torch.optim.Adam(enc_params, lr=lr)
        self.predictor_opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.win_opt = torch.optim.Adam(self.win_predictor.parameters(), lr=lr)
        self.ema_momentum = float(ema_momentum)
        self.train_steps = 0
        self.wins_seen = 0
        self.win_goal_latents: Deque[torch.Tensor] = deque(maxlen=int(win_goal_maxlen))
        logger.info(f"WorldModel initialized: latent_dim={latent_dim}, device={self.device}")

    @torch.no_grad()
    def encode(self, grid):
        if isinstance(grid, np.ndarray):
            grid = torch.from_numpy(grid).to(self.device)
        if grid.dim() == 2:
            grid = grid.unsqueeze(0)
        return self.encoder.encode(grid)

    @torch.no_grad()
    def predict_next(self, z: torch.Tensor, cont_action: torch.Tensor, disc_action: Optional[torch.Tensor] = None):
        if z.dim() == 1: z = z.unsqueeze(0)
        if cont_action.dim() == 1: cont_action = cont_action.unsqueeze(0)
        if disc_action is not None and disc_action.dim() == 0:
            disc_action = disc_action.unsqueeze(0)
        return self.predictor(z, cont_action, disc_action).squeeze(0)

    @torch.no_grad()
    def predict_win(self, z):
        if z.dim() == 1: z = z.unsqueeze(0)
        return float(self.win_predictor(z).item())

    def add_transition(
        self,
        *,
        state: np.ndarray,
        cont_action: np.ndarray,
        disc_action: int = 0,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        win: bool,
    ) -> None:
        # Store on CPU; move to device on sampling.
        st = torch.as_tensor(state, dtype=torch.uint8)
        nst = torch.as_tensor(next_state, dtype=torch.uint8)
        ca = torch.as_tensor(cont_action, dtype=torch.float32)
        da = torch.as_tensor(int(disc_action), dtype=torch.long)
        self.buffer.add(
            state=st,
            cont_action=ca,
            disc_action=da,
            next_state=nst,
            reward=float(reward),
            done=bool(done),
            win=bool(win),
        )

        if win:
            self.wins_seen += 1
            # Store a goal latent from the *winning* next_state (V-JEPA-2-AC image-goal planning style).
            try:
                with torch.no_grad():
                    zg = self.target_encoder.encode(nst.unsqueeze(0).to(self.device))
                    self.win_goal_latents.append(zg.squeeze(0).detach().cpu())
            except Exception:
                # Never let goal caching crash online training.
                pass

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return {}
        batch = self.buffer.sample(self.batch_size)
        states = batch["state"].to(self.device)
        cont_actions = batch["cont_action"].to(self.device)
        disc_actions = batch["disc_action"].to(self.device)
        next_states = batch["next_state"].to(self.device)
        wins = batch["win"].to(self.device)

        # Optional stage-wise freezing (V-JEPA2 -> V-JEPA2-AC style).
        if (not self._encoder_frozen) and self.freeze_encoder_after > 0 and self.train_steps >= self.freeze_encoder_after:
            self._encoder_frozen = True
            logger.info("WorldModel: freezing encoder after %d train steps", self.train_steps)

        if self._encoder_frozen:
            with torch.no_grad():
                z = self.encoder.encode(states)
        else:
            z = self.encoder.encode(states)
        with torch.no_grad():
            z_next_target = self.target_encoder.encode(next_states)

        z_next_pred = self.predictor(z, cont_actions, disc_actions)
        predictor_loss = F.l1_loss(z_next_pred, z_next_target)

        # Optional V-JEPA-style masked denoising in token space (representation learning).
        mask_loss = None
        if (not self._encoder_frozen) and self.mask_predictor is not None and self.mask_ratio > 0.0:
            B = states.shape[0]
            num_patches = int(getattr(self.encoder, "num_patches", 0))
            if num_patches > 0:
                patch_mask = (torch.rand((B, num_patches), device=self.device) < float(self.mask_ratio))
                # Ensure at least one masked patch per sample.
                rows_no_mask = (patch_mask.sum(dim=1) == 0)
                if bool(rows_no_mask.any()):
                    idx = torch.randint(0, num_patches, (B,), device=self.device)
                    patch_mask[torch.arange(B, device=self.device), idx] = True

                tokens_ctx = self.encoder(states, return_all_tokens=True, patch_mask=patch_mask)
                tokens_pred = self.mask_predictor(tokens_ctx)
                with torch.no_grad():
                    tokens_tgt = self.target_encoder(states, return_all_tokens=True)

                # Loss only on masked patch tokens (exclude CLS token at index 0).
                pred_p = tokens_pred[:, 1:, :]
                tgt_p = tokens_tgt[:, 1:, :]
                l1 = (pred_p - tgt_p).abs().mean(dim=-1)  # (B, num_patches)
                mask_f = patch_mask.float()
                mask_loss = (l1 * mask_f).sum() / (mask_f.sum() + 1e-6)

        total_pred_loss = predictor_loss
        if mask_loss is not None:
            total_pred_loss = total_pred_loss + float(self.mask_loss_coef) * mask_loss

        # Optional JEPA-style variance regularization to avoid representation collapse.
        var_loss = None
        mean_var = None
        if (not self._encoder_frozen) and float(self.variance_coef) > 0.0:
            # Mean variance across embedding dims.
            mean_var = z.var(dim=0, unbiased=False).mean()
            var_loss = torch.relu(float(self.variance_target) - mean_var)
            total_pred_loss = total_pred_loss + float(self.variance_coef) * var_loss

        if not self._encoder_frozen:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.predictor_opt.zero_grad()
        total_pred_loss.backward()
        if not self._encoder_frozen:
            self.encoder_opt.step()
        self.predictor_opt.step()

        with torch.no_grad():
            z_detached = self.encoder.encode(states)
        win_logits = self.win_predictor.forward_logits(z_detached)
        win_loss = F.binary_cross_entropy_with_logits(win_logits.squeeze(-1), wins)

        self.win_opt.zero_grad(set_to_none=True)
        win_loss.backward()
        self.win_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.encoder.parameters(), self.target_encoder.parameters(), strict=True):
                pt.data = self.ema_momentum * pt.data + (1 - self.ema_momentum) * p.data

        self.train_steps += 1
        out = {
            "predictor_loss": float(predictor_loss.item()),
            "win_loss": float(win_loss.item()),
            "buffer_size": len(self.buffer),
            "wins_seen": self.wins_seen,
        }
        if mask_loss is not None:
            out["mask_loss"] = float(mask_loss.item())
        if mean_var is not None:
            out["latent_var_mean"] = float(mean_var.item())
        if var_loss is not None:
            out["var_loss"] = float(var_loss.item())
        return out

    @torch.no_grad()
    def plan_action(self, grid):
        if not self.use_planner:
            cont = np.random.uniform(-1, 1, self.continuous_action_dim).astype(np.float32)
            return cont, 0, 0.0
        if self.planner is None:
            cont = np.random.uniform(-1, 1, self.continuous_action_dim).astype(np.float32)
            return cont, 0, 0.0

        z = self.encode(grid).squeeze(0)

        # If we have seen a win at least once, plan toward a stored winning latent (V-JEPA2-AC goal planning).
        goal_latent: torch.Tensor | None = None
        if len(self.win_goal_latents) > 0:
            # Sample a goal to avoid overfitting to a single win state.
            goal_latent = self.win_goal_latents[np.random.randint(0, len(self.win_goal_latents))].to(self.device)

        def reward_fn(z_traj: torch.Tensor) -> float:
            # z_traj: (H+1, latent_dim)
            z_final = z_traj[-1]
            if goal_latent is not None:
                # Maximize negative distance = minimize L1 distance.
                return -float(F.l1_loss(z_final, goal_latent, reduction="mean").item())
            # Fallback: maximize predicted win probability.
            return float(self.win_predictor(z_final.unsqueeze(0)).item())

        best_cont, best_disc, best_reward = self.planner.plan(z, self.predictor, reward_fn, self.device)
        return best_cont.detach().cpu().numpy().astype(np.float32), int(best_disc), float(best_reward)

    def save(self, path):
        torch.save({"encoder": self.encoder.state_dict(), "target_encoder": self.target_encoder.state_dict(),
                   "predictor": self.predictor.state_dict(), "win_predictor": self.win_predictor.state_dict(),
                   "train_steps": self.train_steps, "wins_seen": self.wins_seen}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.target_encoder.load_state_dict(ckpt["target_encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])
        self.win_predictor.load_state_dict(ckpt["win_predictor"])
        self.train_steps = ckpt.get("train_steps", 0)
        self.wins_seen = ckpt.get("wins_seen", 0)
