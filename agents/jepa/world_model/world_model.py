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
from typing import Deque, Optional, List, Dict, Any, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import GridEncoder, GridDecoder, TemporalGridEncoder
from .predictor import ActionConditionedPredictor
from .win_predictor import WinPredictor, ValuePredictor, RewardPredictor
from .planner import CEMPlanner
from .mask_predictor import MaskedTokenPredictor
from .patch_decoder import PatchDecoder

logger = logging.getLogger()


class TransitionBuffer:
    """
    Buffer for single transitions with prioritized replay and consolidation.
    
    Features:
    - Main buffer: FIFO with capacity, older samples evicted
    - Consolidation buffer: Important samples never evicted (catastrophic forgetting prevention)
    - Priority sampling: Higher prediction error = higher sampling probability
    """
    
    def __init__(self, capacity: int = 10000, consolidation_size: int = 500):
        self.capacity = capacity
        self.consolidation_size = consolidation_size
        self.buffer: deque = deque(maxlen=capacity)
        
        # Consolidation buffer for important/unique experiences (never evicted)
        self.consolidation_buffer: List[Dict] = []
        self.consolidation_priorities: List[float] = []  # Prediction error when added
        
        # Priority scores for main buffer (updated during training)
        self.priorities: deque = deque(maxlen=capacity)
        
        # Track unique states seen
        self._state_hashes: set = set()

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
        prediction_error: float = 1.0,  # Default high priority for new samples
    ) -> None:
        transition = {
            "state": state, "cont_action": cont_action, "disc_action": disc_action, 
            "next_state": next_state, "reward": reward, "done": done, "win": win,
        }
        self.buffer.append(transition)
        self.priorities.append(prediction_error + 0.1)  # Small epsilon for stability
        
        # Check if this is a unique/important state worth consolidating
        state_hash = hash(state.numpy().tobytes())
        is_unique = state_hash not in self._state_hashes
        is_important = win or abs(reward) > 50 or prediction_error > 0.5
        
        if is_unique or is_important:
            self._state_hashes.add(state_hash)
            self._maybe_consolidate(transition, prediction_error)
    
    def _maybe_consolidate(self, transition: Dict, priority: float):
        """Add to consolidation buffer if important enough."""
        if len(self.consolidation_buffer) < self.consolidation_size:
            # Buffer not full, just add
            self.consolidation_buffer.append(transition)
            self.consolidation_priorities.append(priority)
        else:
            # Replace lowest priority if this is higher
            min_idx = np.argmin(self.consolidation_priorities)
            if priority > self.consolidation_priorities[min_idx]:
                self.consolidation_buffer[min_idx] = transition
                self.consolidation_priorities[min_idx] = priority
    
    def update_priorities(self, indices: List[int], new_priorities: List[float]):
        """Update priorities for sampled transitions (for TD-error style updates)."""
        for idx, prio in zip(indices, new_priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = prio + 0.1

    def sample(self, batch_size: int, consolidation_ratio: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Sample with mix of main buffer + consolidation buffer.
        OPTIMIZED: Uses uniform sampling for O(1) time, not O(n) prioritized sampling.
        
        Args:
            batch_size: Total samples to return
            consolidation_ratio: Fraction from consolidation buffer (default 10%)
        """
        # Determine split
        n_consolidation = min(
            int(batch_size * consolidation_ratio),
            len(self.consolidation_buffer)
        )
        n_main = batch_size - n_consolidation
        
        batch = []
        sampled_indices = []
        
        # FAST uniform sampling from main buffer - O(1) not O(n)
        if n_main > 0 and len(self.buffer) > 0:
            buf_len = len(self.buffer)
            # Use replacement for speed (no collision checking)
            idxs = np.random.randint(0, buf_len, size=min(n_main, buf_len))
            batch.extend([self.buffer[i] for i in idxs])
            sampled_indices.extend(idxs.tolist())
        
        # Sample from consolidation buffer (uniform - all important)
        if n_consolidation > 0:
            cons_idxs = np.random.choice(
                len(self.consolidation_buffer),
                n_consolidation,
                replace=False
            )
            batch.extend([self.consolidation_buffer[i] for i in cons_idxs])
        
        if not batch:
            return None
        
        return {
            "state": torch.stack([b["state"] for b in batch]),
            "cont_action": torch.stack([b["cont_action"] for b in batch]),
            "disc_action": torch.stack([b["disc_action"] for b in batch]),
            "next_state": torch.stack([b["next_state"] for b in batch]),
            "reward": torch.tensor([float(b["reward"]) for b in batch], dtype=torch.float32),
            "done": torch.tensor([float(b["done"]) for b in batch], dtype=torch.float32),
            "win": torch.tensor([float(b["win"]) for b in batch], dtype=torch.float32),
            "_sampled_indices": sampled_indices,  # For priority updates
        }

    def __len__(self):
        return len(self.buffer) + len(self.consolidation_buffer)


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
        """Sample a batch of sequences - OPTIMIZED for constant time."""
        buf_len = len(self.buffer)
        if buf_len < batch_size:
            return None
        
        # Fast random sampling with replacement for speed
        idxs = torch.randint(0, buf_len, (batch_size,))
        
        # Pre-allocate tensors for speed
        T = self.sequence_length
        sample_0 = self.buffer[0]
        H, W = sample_0["states"][0].shape
        cont_dim = sample_0["cont_actions"][0].shape[0]
        
        states = torch.empty(batch_size, T, H, W, dtype=torch.uint8)
        cont_actions = torch.empty(batch_size, T, cont_dim, dtype=torch.float32)
        disc_actions = torch.empty(batch_size, T, dtype=torch.long)
        final_states = torch.empty(batch_size, H, W, dtype=torch.uint8)
        
        for i, idx in enumerate(idxs):
            b = self.buffer[idx.item()]
            for t in range(T):
                states[i, t] = b["states"][t]
                cont_actions[i, t] = b["cont_actions"][t]
                disc_actions[i, t] = b["disc_actions"][t]
            final_states[i] = b["final_state"]
        
        return {
            "states": states,
            "cont_actions": cont_actions,
            "disc_actions": disc_actions,
            "final_state": final_states,
        }
    
    def __len__(self):
        return len(self.buffer)


class EWCRegularizer:
    """
    Elastic Weight Consolidation (EWC) - Prevents catastrophic forgetting.
    
    Stores "important" weights after successful tasks and penalizes
    changes to those weights during future training.
    
    Simplified version: Uses squared gradient magnitude as importance estimate.
    """
    
    def __init__(self, model: nn.Module, ewc_lambda: float = 1000.0):
        self.model = model
        self.ewc_lambda = ewc_lambda
        
        # Storage for consolidated parameters and their importance
        self.consolidated_params: Dict[str, torch.Tensor] = {}
        self.importance: Dict[str, torch.Tensor] = {}
        self.num_consolidations = 0
    
    @torch.no_grad()
    def consolidate(self, dataloader_or_batch: Any = None):
        """
        Consolidate current weights as important.
        
        Call this after the model has learned something valuable
        (e.g., after a win, after reaching a new state, periodically).
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Store current parameters
                if name not in self.consolidated_params:
                    self.consolidated_params[name] = param.clone().detach()
                    self.importance[name] = torch.zeros_like(param)
                else:
                    # Running average of consolidated params
                    alpha = 0.1
                    self.consolidated_params[name] = (
                        (1 - alpha) * self.consolidated_params[name] + 
                        alpha * param.clone().detach()
                    )
                
                # Estimate importance from gradient magnitude (if available)
                if param.grad is not None:
                    # Accumulate importance (online Fisher approximation)
                    self.importance[name] = (
                        0.9 * self.importance[name] + 
                        0.1 * param.grad.detach() ** 2
                    )
        
        self.num_consolidations += 1
    
    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty loss.
        
        Returns:
            Scalar penalty to add to training loss
        """
        if not self.consolidated_params:
            return torch.tensor(0.0)
        
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.consolidated_params:
                # Penalize deviation from consolidated weights, weighted by importance
                diff = param - self.consolidated_params[name].to(param.device)
                importance = self.importance[name].to(param.device)
                loss = loss + (importance * diff ** 2).sum()
        
        return self.ewc_lambda * loss


class WorldModel(nn.Module):
    """
    V-JEPA 2 style World Model with:
    - 3D-RoPE encoder for stable representation learning
    - Block-causal Transformer predictor for dynamics
    - Teacher-forcing + Rollout loss (per paper Eq. 2-4)
    - Patch-level processing for rich representations
    - Frame stacking for temporal/motion awareness
    - EWC regularization for catastrophic forgetting prevention
    - Consolidation buffer for important experiences
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
        # Temporal encoding for motion awareness
        use_temporal_encoder: bool = True,
        num_frames: int = 4,  # Number of frames to stack
        # EWC for catastrophic forgetting prevention
        use_ewc: bool = True,
        ewc_lambda: float = 100.0,
        ewc_consolidate_every: int = 500,  # Consolidate every N steps
        # Consolidation buffer size
        consolidation_buffer_size: int = 500,
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
        self.use_temporal_encoder = use_temporal_encoder
        self.num_frames = num_frames
        self.use_ewc = use_ewc
        self.ewc_consolidate_every = ewc_consolidate_every
        self.consolidation_buffer_size = consolidation_buffer_size
        
        # Number of patches for predictor
        num_patches = (grid_size // patch_size) ** 2

        # Choose encoder based on temporal setting
        if use_temporal_encoder:
            # Temporal encoder with frame stacking + velocity channels
            self.encoder = TemporalGridEncoder(
                grid_size, patch_size, num_colors, latent_dim, encoder_depth,
                num_frames=num_frames,
            ).to(self.device)
            
            self.target_encoder = TemporalGridEncoder(
                grid_size, patch_size, num_colors, latent_dim, encoder_depth,
                num_frames=num_frames,
            ).to(self.device)
            logger.info(f"Using TemporalGridEncoder with {num_frames} frame stacking")
        else:
            # Standard encoder with 3D-RoPE
            self.encoder = GridEncoder(
                grid_size, patch_size, num_colors, latent_dim, encoder_depth,
                use_rope=True
            ).to(self.device)
            
            self.target_encoder = GridEncoder(
                grid_size, patch_size, num_colors, latent_dim, encoder_depth,
                use_rope=True
            ).to(self.device)
        
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        
        # Patch-based decoder for high-fidelity reconstruction
        # Uses ALL patch tokens (not just CLS) for better spatial detail
        self.decoder = PatchDecoder(
            latent_dim=latent_dim,
            grid_size=grid_size,
            patch_size=patch_size,
            num_colors=num_colors,
            hidden_dim=latent_dim * 4,  # Larger for better capacity
        ).to(self.device)
        # Higher learning rate for decoder to learn faster
        self.decoder_opt = torch.optim.Adam(self.decoder.parameters(), lr=lr * 3)
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
        
        # Reward predictor: R(z, a, z') -> expected reward for this transition
        # This is CRITICAL for learning to avoid penalties!
        self.reward_predictor = RewardPredictor(
            latent_dim=latent_dim,
            action_dim=self.continuous_action_dim + 64,  # cont + discrete embedding
            hidden_dim=latent_dim,
        ).to(self.device)
        self.reward_pred_opt = torch.optim.Adam(self.reward_predictor.parameters(), lr=lr)

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

        # Buffers with consolidation for catastrophic forgetting prevention
        self.buffer = TransitionBuffer(
            capacity=buffer_size,
            consolidation_size=consolidation_buffer_size
        )
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
        
        # EWC regularizer for catastrophic forgetting prevention
        self.ewc: Optional[EWCRegularizer] = None
        if use_ewc:
            self.ewc = EWCRegularizer(self, ewc_lambda=ewc_lambda)
            logger.info(f"EWC enabled: lambda={ewc_lambda}, consolidate_every={ewc_consolidate_every}")
        
        self.ema_momentum = float(ema_momentum)
        self.train_steps = 0
        self.wins_seen = 0
        self.win_goal_latents: Deque[torch.Tensor] = deque(maxlen=int(win_goal_maxlen))
        
        # Track last prediction error for priority updates
        self._last_pred_error: float = 1.0
        
        # Imagination mode state
        self.imagination_mode = False
        self._imagination_state: Optional[torch.Tensor] = None
        self._imagination_step_count = 0
        self._imagination_reanchor_every = 3  # Re-encode every N steps to prevent drift
        
        # Honest rollout state for visualization (shows true multi-step quality)
        self._rollout_z: Optional[torch.Tensor] = None
        self._rollout_steps = 0
        
        logger.info(
            f"WorldModel initialized: latent_dim={latent_dim}, device={self.device}, "
            f"temporal={use_temporal_encoder}, frames={num_frames}, "
            f"ewc={use_ewc}, rollout_steps={rollout_steps}"
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

    @torch.no_grad()
    def decode_grid(self, z: torch.Tensor) -> np.ndarray:
        """
        Decode latent embedding back to grid for visualization.
        
        Args:
            z: (latent_dim,), (1, latent_dim) or (1, num_patches+1, latent_dim)
            
        Returns:
            (grid_size, grid_size) decoded grid as numpy uint8
        """
        if self.decoder is None:
            return np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).to(self.device)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        # PatchDecoder handles both 2D and 3D inputs
        
        grid = self.decoder.decode(z)  # (1, H, W)
        return grid.squeeze(0).cpu().numpy().astype(np.uint8)

    @torch.no_grad()
    def predict_next_grid(
        self,
        grid: np.ndarray,
        cont_action: np.ndarray,
        disc_action: int = 0,
    ) -> np.ndarray:
        """
        Predict next grid given current grid and action.
        
        NOTE: This is single-step prediction from a real grid.
        For honest multi-step prediction like imagination mode, use
        predict_next_grid_from_rollout() instead.
        
        Args:
            grid: (H, W) current grid
            cont_action: (cont_dim,) continuous action
            disc_action: discrete action index
            
        Returns:
            (H, W) predicted next grid
        """
        # Encode current grid with FULL patch tokens for better prediction
        z = self.encode(grid, return_all_tokens=True)  # (1, num_patches+1, latent_dim)
        
        # Prepare actions
        cont_t = torch.from_numpy(cont_action).float().unsqueeze(0).to(self.device)
        disc_t = torch.tensor([disc_action], device=self.device)
        
        # Predict next latent (predictor handles full patch tokens)
        z_next = self.predictor(z, cont_t, disc_t)  # (1, num_patches+1, latent_dim)
        
        # Decode to grid using full patch tokens
        return self.decode_grid(z_next)
    
    # ==================== HONEST MULTI-STEP PREDICTION ====================
    # These methods maintain rollout state to show true model quality
    
    def init_rollout_state(self, grid: np.ndarray) -> None:
        """Initialize rollout state from a real grid (call once at start)."""
        self._rollout_z = self.encode(grid, return_all_tokens=True)
        self._rollout_steps = 0
    
    @torch.no_grad()
    def predict_next_grid_from_rollout(
        self,
        cont_action: np.ndarray,
        disc_action: int = 0,
        real_grid: Optional[np.ndarray] = None,
        reanchor_every: int = 10,
    ) -> np.ndarray:
        """
        Predict next grid using ACCUMULATED rollout state (honest prediction).
        
        This is what imagination mode does - it predicts from previous predictions,
        not from freshly encoded real grids. Shows true model quality.
        
        Args:
            cont_action: Continuous action
            disc_action: Discrete action
            real_grid: If provided and steps >= reanchor_every, reset to real grid
            reanchor_every: How often to re-ground to real grid (0 = never)
            
        Returns:
            Predicted next grid
        """
        if self._rollout_z is None:
            if real_grid is not None:
                self.init_rollout_state(real_grid)
            else:
                raise RuntimeError("Call init_rollout_state first")
        
        # Prepare actions
        cont_t = torch.from_numpy(cont_action).float().unsqueeze(0).to(self.device)
        disc_t = torch.tensor([disc_action], device=self.device)
        
        # Predict from accumulated state (NOT from fresh encoding!)
        z_next = self.predictor(self._rollout_z, cont_t, disc_t)
        
        # Update rollout state
        self._rollout_z = z_next
        self._rollout_steps += 1
        
        # Optional: re-anchor periodically to show "reset" quality
        if reanchor_every > 0 and real_grid is not None and self._rollout_steps >= reanchor_every:
            self._rollout_z = self.encode(real_grid, return_all_tokens=True)
            self._rollout_steps = 0
        
        return self.decode_grid(z_next)

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
        
        # Single-step buffer with priority based on last prediction error
        self.buffer.add(
            state=st,
            cont_action=ca,
            disc_action=da,
            next_state=nst,
            reward=float(reward),
            done=bool(done),
            win=bool(win),
            prediction_error=self._last_pred_error,
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
        
        # === ACTION-CONTRASTIVE LOSS ===
        # Force model to produce DIFFERENT predictions for DIFFERENT actions
        # If the model ignores actions, this loss will be high
        action_contrastive_loss = None
        if B >= 2:
            # Generate counterfactual: what if we used a DIFFERENT action?
            # Shuffle actions within batch
            perm = torch.randperm(B, device=self.device)
            shuffled_disc = disc_actions[perm]
            shuffled_cont = cont_actions[perm]
            
            # Find samples where action actually changed
            action_changed = (shuffled_disc != disc_actions)
            
            if action_changed.any():
                # Predict with shuffled actions
                z_counterfactual = self.predictor(z, shuffled_cont, shuffled_disc)
                
                # The predictions SHOULD be different where actions differ
                # Compute L2 distance between predictions
                pred_diff = (z_next_pred - z_counterfactual).pow(2).mean(dim=-1)
                if z_next_pred.dim() == 3:
                    pred_diff = pred_diff.mean(dim=-1)  # Average over patches
                
                # We WANT predictions to be different when actions differ
                # Loss = -log(pred_diff) encourages larger differences
                # Use margin: predictions should differ by at least margin
                margin = 0.1
                action_contrastive_loss = F.relu(margin - pred_diff[action_changed]).mean()

        # === Rollout loss (Eq. 3) with OPTIMIZED trajectory training ===
        # Fast batched operations for constant-time training
        rollout_loss = None
        rollout_decoder_loss = None
        cycle_consistency_loss = None
        
        if self.rollout_loss_coef > 0.0 and len(self.sequence_buffer) >= self.batch_size:
            seq_batch = self.sequence_buffer.sample(self.batch_size)
            if seq_batch is not None:
                seq_states = seq_batch["states"].to(self.device)  # (B, T, H, W)
                seq_cont = seq_batch["cont_actions"].to(self.device)  # (B, T, cont_dim)
                seq_disc = seq_batch["disc_actions"].to(self.device)  # (B, T)
                final_state = seq_batch["final_state"].to(self.device)  # (B, H, W)
                
                T = seq_states.shape[1]
                B_seq = seq_states.shape[0]
                
                # BATCHED target encoding - encode ALL states in ONE forward pass
                # Stack all states: (B, T, H, W) + final -> (B*(T+1), H, W)
                all_states = torch.cat([
                    seq_states.view(B_seq * T, self.grid_size, self.grid_size),
                    final_state
                ], dim=0)  # (B*T + B, H, W)
                
                with torch.no_grad():
                    if self.use_patch_tokens:
                        all_z_targets = self.target_encoder(all_states, return_all_tokens=True)
                    else:
                        all_z_targets = self.target_encoder.encode(all_states)
                
                # Encode initial state for rollout
                if self.use_patch_tokens:
                    z_init = self.encoder(seq_states[:, 0], return_all_tokens=True)
                else:
                    z_init = self.encoder.encode(seq_states[:, 0])
                
                # Autoregressive rollout - only keep first and last for efficiency
                z_current = z_init
                z_first_pred = None
                for t in range(T):
                    z_current = self.predictor(z_current, seq_cont[:, t], seq_disc[:, t])
                    if t == 0:
                        z_first_pred = z_current
                z_final_pred = z_current
                
                # Extract targets for first step and final step
                # all_z_targets is (B*T + B, ...), split it
                if self.use_patch_tokens:
                    z_target_dim = all_z_targets.shape[1:]  # (num_patches+1, latent_dim)
                    z_targets_seq = all_z_targets[:B_seq * T].view(B_seq, T, *z_target_dim)
                    z_target_final = all_z_targets[B_seq * T:]  # (B, ...)
                else:
                    z_targets_seq = all_z_targets[:B_seq * T].view(B_seq, T, -1)
                    z_target_final = all_z_targets[B_seq * T:]
                
                # Rollout loss: first prediction + final prediction (skip intermediate for speed)
                loss_first = F.l1_loss(z_first_pred, z_targets_seq[:, 0] if T > 0 else z_target_final)
                loss_final = F.l1_loss(z_final_pred, z_target_final)
                rollout_loss = 0.3 * loss_first + 0.7 * loss_final  # Emphasize final
                
                # === DECODER ON FINAL ROLLOUT PREDICTION ONLY (fast) ===
                if self.decoder is not None:
                    z_final_detached = z_final_pred.detach()
                    grid_logits_final = self.decoder(z_final_detached)
                    rollout_decoder_loss = F.cross_entropy(grid_logits_final, final_state.long())
                
                # === CYCLE CONSISTENCY (every 4th step for speed) ===
                if self.decoder is not None and self.train_steps % 4 == 0:
                    with torch.no_grad():
                        decoded_logits = self.decoder(z_final_detached)
                        decoded_grid = decoded_logits.argmax(dim=1)
                    if self.use_patch_tokens:
                        z_reencoded = self.encoder(decoded_grid, return_all_tokens=True)
                    else:
                        z_reencoded = self.encoder.encode(decoded_grid)
                    cycle_consistency_loss = F.l1_loss(z_reencoded, z_target_final)

        # === Combine losses (Eq. 4) ===
        predictor_loss = teacher_forcing_loss
        if action_contrastive_loss is not None:
            predictor_loss = predictor_loss + 1.0 * action_contrastive_loss  # Strong weight!
        if rollout_loss is not None:
            predictor_loss = predictor_loss + self.rollout_loss_coef * rollout_loss
        if cycle_consistency_loss is not None:
            predictor_loss = predictor_loss + 0.5 * cycle_consistency_loss

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

        # === EWC penalty for catastrophic forgetting prevention ===
        ewc_loss = None
        if self.ewc is not None:
            ewc_loss = self.ewc.penalty()
            if ewc_loss.item() > 0:
                total_pred_loss = total_pred_loss + ewc_loss
        
        # === Backward pass ===
        self.encoder_opt.zero_grad(set_to_none=True)
        self.predictor_opt.zero_grad()
        total_pred_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        
        self.encoder_opt.step()
        self.predictor_opt.step()
        
        # Track prediction error for priority updates
        self._last_pred_error = float(teacher_forcing_loss.detach().item())
        
        # Periodic EWC consolidation
        if self.ewc is not None and self.train_steps % self.ewc_consolidate_every == 0:
            self.ewc.consolidate()

        # === Win predictor (uses CLS token) ===
        with torch.no_grad():
            z_detached = self.encoder.encode(states)
            z_next_detached = self.encoder.encode(next_states)
        win_logits = self.win_predictor.forward_logits(z_detached)
        win_loss = F.binary_cross_entropy_with_logits(win_logits.squeeze(-1), wins)

        self.win_opt.zero_grad(set_to_none=True)
        win_loss.backward()
        self.win_opt.step()
        
        # === Reward predictor: Learn R(z, a, z') from actual rewards ===
        # This is CRITICAL for penalty avoidance!
        rewards_batch = batch["reward"].to(self.device)
        
        # Embed discrete action for reward predictor
        disc_embed = self.predictor.discrete_embed(disc_actions)  # (B, 64)
        action_combined = torch.cat([cont_actions, disc_embed], dim=-1)  # (B, cont_dim + 64)
        
        pred_rewards = self.reward_predictor(z_detached, action_combined, z_next_detached)
        reward_loss = F.mse_loss(pred_rewards.squeeze(-1), rewards_batch)
        
        self.reward_pred_opt.zero_grad(set_to_none=True)
        reward_loss.backward()
        self.reward_pred_opt.step()
        
        # === Decoder training: Reconstruct grid from latent ===
        # Uses FULL patch tokens for high-fidelity spatial reconstruction
        decoder_loss = None
        if self.decoder is not None:
            # Encode states with FULL patch tokens (not just CLS)
            z_for_decode = self.encoder(states, return_all_tokens=True).detach()
            
            # Decode to grid logits using all patch tokens
            grid_logits = self.decoder(z_for_decode)  # (B, num_colors, H, W)
            states_long = states.long()
            decoder_loss_current = F.cross_entropy(grid_logits, states_long)
            
            # Part 2: Decode PREDICTED next states (critical for moving objects!)
            # Use full patch tokens from predictor
            z_next_pred_for_decode = z_next_pred.detach()
            grid_logits_pred = self.decoder(z_next_pred_for_decode)
            next_states_long = next_states.long()
            decoder_loss_pred = F.cross_entropy(grid_logits_pred, next_states_long)
            
            # Combined loss with emphasis on predicted states (for dynamics)
            decoder_loss = 0.2 * decoder_loss_current + 0.3 * decoder_loss_pred
            
            # Add rollout decoder loss - CRITICAL for multi-step imagination
            if rollout_decoder_loss is not None:
                decoder_loss = decoder_loss + 0.5 * rollout_decoder_loss
            
            self.decoder_opt.zero_grad(set_to_none=True)
            decoder_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
            self.decoder_opt.step()

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
            "reward_loss": float(reward_loss.item()),
            "buffer_size": len(self.buffer),
            "sequence_buffer_size": len(self.sequence_buffer),
            "wins_seen": self.wins_seen,
        }
        if action_contrastive_loss is not None:
            out["action_contrast_loss"] = float(action_contrastive_loss.item())
        if rollout_loss is not None:
            out["rollout_loss"] = float(rollout_loss.item())
        if rollout_decoder_loss is not None:
            out["rollout_dec_loss"] = float(rollout_decoder_loss.item())
        if cycle_consistency_loss is not None:
            out["cycle_loss"] = float(cycle_consistency_loss.item())
        if mask_loss is not None:
            out["mask_loss"] = float(mask_loss.item())
        if mean_var is not None:
            out["latent_var_mean"] = float(mean_var.item())
        if var_loss is not None:
            out["var_loss"] = float(var_loss.item())
        if decoder_loss is not None:
            out["decoder_loss"] = float(decoder_loss.item())
        return out

    @torch.no_grad()
    def plan_action(self, grid):
        """
        Plan action using CEM with goal-conditioned L1 distance (V-JEPA 2-AC style)
        PLUS learned reward prediction (critical for penalty avoidance).
        
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

        # Coefficients for reward components
        goal_dist_coef = 1.0
        reward_pred_coef = 10.0  # Amplify reward signal for penalty avoidance
        win_prob_coef = 1.0

        def reward_fn(z_traj: torch.Tensor, actions_traj: tuple = None) -> float:
            """
            Compute reward for a trajectory. Combines:
            1. Goal distance (V-JEPA 2-AC style)
            2. Predicted step rewards (for penalty avoidance!)
            3. Win probability
            
            actions_traj: (cont_actions, disc_actions) tuple if provided by planner
            """
            total_reward = 0.0
            
            # Handle different trajectory formats
            if z_traj.dim() == 3:
                # (H+1, num_patches+1, latent_dim) -> use CLS tokens
                z_sequence = z_traj[:, 0, :]  # (H+1, latent_dim)
            else:
                z_sequence = z_traj  # (H+1, latent_dim)
            
            z_final = z_sequence[-1]
            
            # 1. Goal distance term (minimize L1 to goal)
            if goal_latent is not None:
                goal_dist = float(F.l1_loss(z_final, goal_latent, reduction="mean").item())
                total_reward -= goal_dist_coef * goal_dist
            else:
                # Fallback: maximize win probability
                total_reward += win_prob_coef * float(self.win_predictor(z_final.unsqueeze(0)).item())
            
            # 2. Sum of predicted step rewards (CRITICAL for penalty avoidance!)
            # This makes the planner avoid actions that lead to penalties
            if actions_traj is not None and len(z_sequence) > 1:
                cont_actions, disc_actions = actions_traj
                for t in range(len(z_sequence) - 1):
                    z_t = z_sequence[t]
                    z_next = z_sequence[t + 1]
                    
                    # Get action embedding
                    disc_embed = self.predictor.discrete_embed(disc_actions[t:t+1])  # (1, 64)
                    cont_t = cont_actions[t:t+1]  # (1, cont_dim)
                    action_combined = torch.cat([cont_t, disc_embed], dim=-1)  # (1, cont_dim + 64)
                    
                    # Predict reward for this step
                    pred_r = self.reward_predictor(
                        z_t.unsqueeze(0),
                        action_combined,
                        z_next.unsqueeze(0)
                    )
                    total_reward += reward_pred_coef * float(pred_r.item())
            
            return total_reward

        best_cont, best_disc, best_reward = self.planner.plan(z, self.predictor, reward_fn, self.device)
        return best_cont.detach().cpu().numpy().astype(np.float32), int(best_disc), float(best_reward)

    @torch.no_grad()
    def multi_step_rollout(
        self,
        z: torch.Tensor,
        continuous_actions: torch.Tensor,
        discrete_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Wrapper for predictor's multi-step rollout (used by CEM planner).
        
        Args:
            z: (batch, latent_dim) initial CLS latent
            continuous_actions: (batch, horizon, continuous_dim)
            discrete_actions: (batch, horizon)
        Returns:
            z_trajectory: (batch, horizon+1, latent_dim)
        """
        return self.predictor.multi_step_rollout(z, continuous_actions, discrete_actions)
    
    def save(self, path):
        torch.save({"encoder": self.encoder.state_dict(), "target_encoder": self.target_encoder.state_dict(),
                   "predictor": self.predictor.state_dict(), "win_predictor": self.win_predictor.state_dict(),
                   "reward_predictor": self.reward_predictor.state_dict(),
                   "train_steps": self.train_steps, "wins_seen": self.wins_seen}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.target_encoder.load_state_dict(ckpt["target_encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])
        self.win_predictor.load_state_dict(ckpt["win_predictor"])
        if "reward_predictor" in ckpt:
            self.reward_predictor.load_state_dict(ckpt["reward_predictor"])
        self.train_steps = ckpt.get("train_steps", 0)
        self.wins_seen = ckpt.get("wins_seen", 0)
    
    # ==================== IMAGINATION MODE ====================
    # Play inside the world model's imagination to see what it predicts
    
    def enter_imagination(self, initial_grid: np.ndarray) -> np.ndarray:
        """
        Enter imagination mode starting from a given state.
        
        The world model will now predict future states based on actions,
        simulating what it thinks will happen without the real environment.
        
        Args:
            initial_grid: Starting state to imagine from
            
        Returns:
            The initial imagined state (decoded from latent)
        """
        self.imagination_mode = True
        self._imagination_step_count = 0
        self._imagination_reanchor_every = 3  # Re-encode every N steps to prevent drift
        
        # Reset temporal encoder history first so we start fresh
        if self.use_temporal_encoder and hasattr(self.encoder, 'reset_history'):
            self.encoder.reset_history()
        
        # Encode initial state - MUST use return_all_tokens=True for patch decoder!
        z = self.encode(initial_grid, return_all_tokens=True)
        self._imagination_state = z
        
        # Return decoded initial state
        return self.decode_grid(z)
    
    def exit_imagination(self):
        """Exit imagination mode and return to reality."""
        self.imagination_mode = False
        self._imagination_state = None
        
        # Reset temporal encoder history
        if self.use_temporal_encoder and hasattr(self.encoder, 'reset_history'):
            self.encoder.reset_history()
    
    @torch.no_grad()
    def imagine_step(
        self,
        cont_action: np.ndarray,
        disc_action: int = 0,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Take a step in imagination mode.
        
        Uses the world model to predict what would happen if the given
        action were taken in the current imagined state.
        
        Args:
            cont_action: Continuous action (cursor movement)
            disc_action: Discrete action (button press)
            
        Returns:
            Tuple of:
            - predicted_grid: The imagined next state decoded to grid
            - predicted_reward: What the model thinks the reward would be
            - win_prob: Probability of winning from this state
        """
        if not self.imagination_mode or self._imagination_state is None:
            raise RuntimeError("Not in imagination mode. Call enter_imagination first.")
        
        # Prepare actions
        cont_t = torch.from_numpy(cont_action).float().unsqueeze(0).to(self.device)
        disc_t = torch.tensor([disc_action], device=self.device)
        
        z_current = self._imagination_state
        
        # Predict next latent state (preserves patch tokens)
        z_next = self.predictor(z_current, cont_t, disc_t)
        
        # Decode to grid using full patch tokens
        predicted_grid = self.decode_grid(z_next)
        
        # Track step count (no re-anchoring needed with proper cycle-consistency training)
        self._imagination_step_count += 1
        
        # Extract CLS tokens for scalar predictors (reward, win)
        # Full patch tokens: (B, num_patches+1, latent_dim), CLS is at index 0
        if z_current.dim() == 3:
            z_current_cls = z_current[:, 0]  # (B, latent_dim)
            z_next_cls = z_next[:, 0]  # (B, latent_dim)
        else:
            z_current_cls = z_current
            z_next_cls = z_next
        
        # Predict reward for this transition
        disc_embed = self.predictor.discrete_embed(disc_t)
        action_combined = torch.cat([cont_t, disc_embed], dim=-1)
        pred_reward = float(self.reward_predictor(z_current_cls, action_combined, z_next_cls).item())
        
        # Predict win probability
        win_prob = float(self.win_predictor(z_next_cls).item())
        
        # Update imagination state (keep full patch tokens for next prediction)
        self._imagination_state = z_next
        
        return predicted_grid, pred_reward, win_prob
    
    @torch.no_grad()
    def imagine_trajectory(
        self,
        cont_actions: np.ndarray,
        disc_actions: np.ndarray,
        initial_grid: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        Imagine a full trajectory of actions.
        
        Args:
            cont_actions: (T, cont_dim) sequence of continuous actions
            disc_actions: (T,) sequence of discrete actions
            initial_grid: Optional starting state (uses current if not provided)
            
        Returns:
            List of (predicted_grid, predicted_reward, win_prob) for each step
        """
        if initial_grid is not None:
            self.enter_imagination(initial_grid)
        elif not self.imagination_mode:
            raise RuntimeError("Either provide initial_grid or call enter_imagination first")
        
        results = []
        T = len(cont_actions)
        
        for t in range(T):
            grid, reward, win_p = self.imagine_step(cont_actions[t], int(disc_actions[t]))
            results.append((grid, reward, win_p))
        
        return results
    
    def get_imagination_state(self) -> Optional[torch.Tensor]:
        """Get current latent state in imagination mode."""
        return self._imagination_state
    
    def set_imagination_state(self, z: torch.Tensor):
        """Set imagination state directly (for branching)."""
        self._imagination_state = z.to(self.device)
        self.imagination_mode = True