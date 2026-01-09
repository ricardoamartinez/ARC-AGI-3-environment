"""
WorldModel - V-JEPA 2 inspired latent dynamics + goal-conditioned planning for ARC-AGI-3.

Online RL adaptation of V-JEPA 2 principles:

- **Encoder / Target Encoder**: Patchified ViT-like encoder with 3D-RoPE and EMA target for stable learning.
  All components train together online (no separate pretraining phase).
- **Action-conditioned Predictor**: Transformer with block-causal attention, predicts next latent 
  given current latent + action using L1 loss.
- **Teacher-forcing + Rollout Loss**: Per V-JEPA 2 paper Eq. 2-4, uses both single-step teacher-forcing
  and multi-step rollout loss for better dynamics learning.
- **(Optional) Win predictor**: Learns P(win|z) from sparse win labels.
- **Goal-latent planning**: Store winning latent(s) and plan by minimizing L1 distance to goal
  (CEM planner, inspired by V-JEPA 2-AC planning).

Key V-JEPA 2 principles retained:
- L1 loss for all latent predictions (not MSE or cosine)
- EMA target encoder for stable targets
- 3D-RoPE for positional encoding
- Block-causal attention in predictor
- Joint embedding space for visual-action learning

Notes:
- ARC grids may use palettes larger than 16. We default to 256 embeddings to be safe.
- The environment's "delta" action mode can yield an executed discrete action of -1..9.
  We represent this in the model as 0..10 via `disc_token = final_action_idx + 1`.
"""

import logging
from typing import Deque, Optional, List, Dict, Any
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
    """Buffer for single transitions (state, action, next_state)."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

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

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
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


class SequenceBuffer:
    """
    Buffer for sequences of transitions for rollout loss training.
    
    Per V-JEPA 2 paper (Section 3.1), we need sequences for the rollout loss:
    L_rollout = ||P(a_1:T, s_1, z_1) - z_{T+1}||_1
    """
    
    def __init__(self, capacity: int = 1000, sequence_length: int = 3):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer: deque = deque(maxlen=capacity)
        self.current_episode: List[Dict[str, Any]] = []
    
    def add_step(
        self,
        *,
        state: torch.Tensor,
        cont_action: torch.Tensor,
        disc_action: torch.Tensor,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Add a step to current episode."""
        self.current_episode.append({
            "state": state,
            "cont_action": cont_action,
            "disc_action": disc_action,
            "next_state": next_state,
        })
        
        # Extract sequences when we have enough steps
        if len(self.current_episode) >= self.sequence_length:
            seq = self.current_episode[-self.sequence_length:]
            self.buffer.append({
                "states": [s["state"] for s in seq],
                "cont_actions": [s["cont_action"] for s in seq],
                "disc_actions": [s["disc_action"] for s in seq],
                "final_state": seq[-1]["next_state"],
            })
        
        # Reset on episode end
        if done:
            self.current_episode = []
    
    def sample(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Sample a batch of sequences."""
        if len(self.buffer) < batch_size:
            return None
        
        idxs = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in idxs]
        
        T = self.sequence_length
        return {
            "states": torch.stack([torch.stack(b["states"]) for b in batch]),  # (B, T, H, W)
            "cont_actions": torch.stack([torch.stack(b["cont_actions"]) for b in batch]),  # (B, T, cont_dim)
            "disc_actions": torch.stack([torch.stack(b["disc_actions"]) for b in batch]),  # (B, T)
            "final_state": torch.stack([b["final_state"] for b in batch]),  # (B, H, W)
        }
    
    def __len__(self):
        return len(self.buffer)


class WorldModel(nn.Module):
    """
    V-JEPA 2 style World Model with:
    - 3D-RoPE encoder for stable representation learning
    - Block-causal Transformer predictor for dynamics
    - Teacher-forcing + Rollout loss (per paper Eq. 2-4)
    - Patch-level processing for rich representations
    """
    
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
        predictor_layers: int = 6,
        predictor_heads: int = 8,
        device: str = "cuda",
        lr: float = 1e-4,
        buffer_size: int = 10000,
        sequence_buffer_size: int = 1000,
        batch_size: int = 32,
        planning_horizon: int = 10,
        use_planner: bool = True,
        planner_num_samples: int = 100,
        planner_num_elites: int = 10,
        planner_iterations: int = 5,
        ema_momentum: float = 0.99,
        win_goal_maxlen: int = 32,
        # Optional V-JEPA-style masked denoising loss on patch tokens.
        mask_ratio: float = 0.0,
        mask_loss_coef: float = 1.0,
        # Optional JEPA-style anti-collapse regularizer (batch variance floor).
        # When enabled, penalizes low variance in the latent across the batch.
        variance_coef: float = 0.0,
        variance_target: float = 1.0,
        # Rollout loss settings (per V-JEPA 2 paper Eq. 3-4)
        rollout_steps: int = 2,
        rollout_loss_coef: float = 1.0,
        # Whether to use patch tokens or just CLS for dynamics
        use_patch_tokens: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.continuous_action_dim = int(continuous_action_dim)
        self.num_discrete_actions = int(num_discrete_actions)
        self.batch_size = batch_size
        self.use_planner = use_planner
        self.mask_ratio = float(mask_ratio)
        self.mask_loss_coef = float(mask_loss_coef)
        self.variance_coef = float(variance_coef)
        self.variance_target = float(variance_target)
        self.rollout_steps = int(rollout_steps)
        self.rollout_loss_coef = float(rollout_loss_coef)
        self.use_patch_tokens = use_patch_tokens
        
        # Number of patches for predictor
        num_patches = (grid_size // patch_size) ** 2

        # Encoder with 3D-RoPE (V-JEPA 2 style)
        self.encoder = GridEncoder(
            grid_size, patch_size, num_colors, latent_dim, encoder_depth,
            use_rope=True  # Enable 3D-RoPE
        ).to(self.device)
        
        # Target encoder (EMA)
        self.target_encoder = GridEncoder(
            grid_size, patch_size, num_colors, latent_dim, encoder_depth,
            use_rope=True
        ).to(self.device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.mask_predictor: MaskedTokenPredictor | None = None
        if self.mask_ratio > 0.0:
            self.mask_predictor = MaskedTokenPredictor(dim=latent_dim, hidden_dim=latent_dim * 2).to(self.device)

        # Action-conditioned predictor with Transformer + block-causal attention
        self.predictor = ActionConditionedPredictor(
            latent_dim=latent_dim,
            continuous_dim=self.continuous_action_dim,
            num_discrete_actions=self.num_discrete_actions,
            hidden_dim=latent_dim * 2,
            num_layers=predictor_layers,
            num_heads=predictor_heads,
            num_patches=num_patches + 1,  # +1 for CLS token
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

        # Buffers
        self.buffer = TransitionBuffer(buffer_size)
        self.sequence_buffer = SequenceBuffer(
            capacity=sequence_buffer_size,
            sequence_length=self.rollout_steps + 1  # Need T+1 states for T-step rollout
        )
        
        # Optimizers
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
        
        logger.info(
            f"WorldModel initialized: latent_dim={latent_dim}, device={self.device}, "
            f"use_rope=True, use_patch_tokens={use_patch_tokens}, rollout_steps={rollout_steps}"
        )

    @torch.no_grad()
    def encode(self, grid, return_all_tokens: bool = False):
        """
        Encode grid to latent representation.
        
        Args:
            grid: (H, W) or (B, H, W) grid
            return_all_tokens: if True, return all patch tokens; else return CLS token
        """
        if isinstance(grid, np.ndarray):
            grid = torch.from_numpy(grid).to(self.device)
        if grid.dim() == 2:
            grid = grid.unsqueeze(0)
        return self.encoder(grid, return_all_tokens=return_all_tokens)

    @torch.no_grad()
    def predict_next(
        self, 
        z: torch.Tensor, 
        cont_action: torch.Tensor, 
        disc_action: Optional[torch.Tensor] = None
    ):
        """Predict next latent given current latent and action."""
        if z.dim() == 1: 
            z = z.unsqueeze(0)
        if cont_action.dim() == 1: 
            cont_action = cont_action.unsqueeze(0)
        if disc_action is not None and disc_action.dim() == 0:
            disc_action = disc_action.unsqueeze(0)
        return self.predictor(z, cont_action, disc_action).squeeze(0)

    @torch.no_grad()
    def predict_win(self, z):
        """Predict win probability from latent."""
        if z.dim() == 1: 
            z = z.unsqueeze(0)
        elif z.dim() == 3:
            # If patch tokens provided, use CLS token
            z = z[:, 0]
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
        """Add transition to both single-step and sequence buffers."""
        # Store on CPU; move to device on sampling.
        st = torch.as_tensor(state, dtype=torch.uint8)
        nst = torch.as_tensor(next_state, dtype=torch.uint8)
        ca = torch.as_tensor(cont_action, dtype=torch.float32)
        da = torch.as_tensor(int(disc_action), dtype=torch.long)
        
        # Single-step buffer
        self.buffer.add(
            state=st,
            cont_action=ca,
            disc_action=da,
            next_state=nst,
            reward=float(reward),
            done=bool(done),
            win=bool(win),
        )
        
        # Sequence buffer for rollout loss
        self.sequence_buffer.add_step(
            state=st,
            cont_action=ca,
            disc_action=da,
            next_state=nst,
            done=bool(done),
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

    def train_step(self) -> Dict[str, float]:
        """
        Training step with V-JEPA 2 style losses:
        - Teacher-forcing loss (Eq. 2): single-step prediction
        - Rollout loss (Eq. 3): multi-step autoregressive prediction
        - Total loss (Eq. 4): L = L_teacher-forcing + L_rollout
        """
        if len(self.buffer) < self.batch_size:
            return {}
        
        batch = self.buffer.sample(self.batch_size)
        states = batch["state"].to(self.device)
        cont_actions = batch["cont_action"].to(self.device)
        disc_actions = batch["disc_action"].to(self.device)
        next_states = batch["next_state"].to(self.device)
        wins = batch["win"].to(self.device)
        B = states.shape[0]

        # === Encode with patch tokens if enabled ===
        if self.use_patch_tokens:
            # Get all tokens including CLS
            z = self.encoder(states, return_all_tokens=True)  # (B, num_patches+1, latent_dim)
            with torch.no_grad():
                z_next_target = self.target_encoder(next_states, return_all_tokens=True)
        else:
            # CLS token only
            z = self.encoder.encode(states)  # (B, latent_dim)
            with torch.no_grad():
                z_next_target = self.target_encoder.encode(next_states)

        # === Teacher-forcing loss (Eq. 2) ===
        z_next_pred = self.predictor(z, cont_actions, disc_actions)
        teacher_forcing_loss = F.l1_loss(z_next_pred, z_next_target)

        # === Rollout loss (Eq. 3) ===
        rollout_loss = None
        if self.rollout_loss_coef > 0.0 and len(self.sequence_buffer) >= self.batch_size:
            seq_batch = self.sequence_buffer.sample(self.batch_size)
            if seq_batch is not None:
                seq_states = seq_batch["states"].to(self.device)  # (B, T, H, W)
                seq_cont = seq_batch["cont_actions"].to(self.device)  # (B, T, cont_dim)
                seq_disc = seq_batch["disc_actions"].to(self.device)  # (B, T)
                final_state = seq_batch["final_state"].to(self.device)  # (B, H, W)
                
                T = seq_states.shape[1]
                B_seq = seq_states.shape[0]
                
                # Encode initial state
                if self.use_patch_tokens:
                    z_init = self.encoder(seq_states[:, 0], return_all_tokens=True)
                else:
                    z_init = self.encoder.encode(seq_states[:, 0])
                
                # Autoregressive rollout
                z_current = z_init
                for t in range(T):
                    z_current = self.predictor(z_current, seq_cont[:, t], seq_disc[:, t])
                
                # Target: encode final state
                with torch.no_grad():
                    if self.use_patch_tokens:
                        z_final_target = self.target_encoder(final_state, return_all_tokens=True)
                    else:
                        z_final_target = self.target_encoder.encode(final_state)
                
                rollout_loss = F.l1_loss(z_current, z_final_target)

        # === Combine losses (Eq. 4) ===
        predictor_loss = teacher_forcing_loss
        if rollout_loss is not None:
            predictor_loss = predictor_loss + self.rollout_loss_coef * rollout_loss

        # === Optional V-JEPA-style masked denoising ===
        mask_loss = None
        if self.mask_predictor is not None and self.mask_ratio > 0.0:
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

        # === Optional variance regularization ===
        var_loss = None
        mean_var = None
        if float(self.variance_coef) > 0.0:
            # Get CLS token for variance computation
            z_cls = z[:, 0] if self.use_patch_tokens else z
            mean_var = z_cls.var(dim=0, unbiased=False).mean()
            var_loss = torch.relu(float(self.variance_target) - mean_var)
            total_pred_loss = total_pred_loss + float(self.variance_coef) * var_loss

        # === Backward pass ===
        self.encoder_opt.zero_grad(set_to_none=True)
        self.predictor_opt.zero_grad()
        total_pred_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        
        self.encoder_opt.step()
        self.predictor_opt.step()

        # === Win predictor (uses CLS token) ===
        with torch.no_grad():
            z_detached = self.encoder.encode(states)
        win_logits = self.win_predictor.forward_logits(z_detached)
        win_loss = F.binary_cross_entropy_with_logits(win_logits.squeeze(-1), wins)

        self.win_opt.zero_grad(set_to_none=True)
        win_loss.backward()
        self.win_opt.step()

        # === EMA update ===
        with torch.no_grad():
            for p, pt in zip(self.encoder.parameters(), self.target_encoder.parameters(), strict=True):
                pt.data = self.ema_momentum * pt.data + (1 - self.ema_momentum) * p.data

        self.train_steps += 1
        
        # === Logging ===
        out: Dict[str, float] = {
            "teacher_forcing_loss": float(teacher_forcing_loss.item()),
            "predictor_loss": float(predictor_loss.item()),
            "win_loss": float(win_loss.item()),
            "buffer_size": len(self.buffer),
            "sequence_buffer_size": len(self.sequence_buffer),
            "wins_seen": self.wins_seen,
        }
        if rollout_loss is not None:
            out["rollout_loss"] = float(rollout_loss.item())
        if mask_loss is not None:
            out["mask_loss"] = float(mask_loss.item())
        if mean_var is not None:
            out["latent_var_mean"] = float(mean_var.item())
        if var_loss is not None:
            out["var_loss"] = float(var_loss.item())
        return out

    @torch.no_grad()
    def plan_action(self, grid):
        """
        Plan action using CEM with goal-conditioned L1 distance (V-JEPA 2-AC style).
        
        Uses CLS token for planning efficiency, even when training uses patch tokens.
        """
        if not self.use_planner:
            cont = np.random.uniform(-1, 1, self.continuous_action_dim).astype(np.float32)
            return cont, 0, 0.0
        if self.planner is None:
            cont = np.random.uniform(-1, 1, self.continuous_action_dim).astype(np.float32)
            return cont, 0, 0.0

        # Use CLS token for planning (more efficient than full patch tokens)
        z = self.encode(grid, return_all_tokens=False).squeeze(0)

        # If we have seen a win at least once, plan toward a stored winning latent (V-JEPA2-AC goal planning).
        goal_latent: torch.Tensor | None = None
        if len(self.win_goal_latents) > 0:
            # Sample a goal to avoid overfitting to a single win state.
            goal_latent = self.win_goal_latents[np.random.randint(0, len(self.win_goal_latents))].to(self.device)

        def reward_fn(z_traj: torch.Tensor) -> float:
            # z_traj: (H+1, latent_dim) or (H+1, num_patches+1, latent_dim)
            z_final = z_traj[-1]
            # Handle patch tokens case
            if z_final.dim() == 2:
                z_final = z_final[0]  # Use CLS token
            if goal_latent is not None:
                # Maximize negative distance = minimize L1 distance (V-JEPA 2-AC Eq. 5)
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
