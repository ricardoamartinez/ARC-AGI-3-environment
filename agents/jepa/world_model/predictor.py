"""
Action-Conditioned Predictor - V-JEPA 2-AC style dynamics model.

Per V-JEPA 2 paper (Section 3.1):
- ~300M parameter transformer with block-causal attention
- Processes interleaved (action, state, patch_tokens) sequences
- Autoregressively predicts next frame representation
- Uses L1 loss with teacher-forcing + rollout loss

Supports hybrid action space: continuous cursor control + a discrete action *token*.

The discrete token is intentionally generic:
- token 0 is typically reserved for "NO-OP" (no discrete game action executed)
- tokens 1..N-1 map to environment-specific executed actions (e.g., key-presses, click types)

In ARC's default `ARCGymEnv` "delta" mode, the env produces an executed `final_action_idx` in -1..9
(-1 means no discrete action executed). In the world model we represent this as:

  disc_token = final_action_idx + 1    # -> 0..10
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockCausalAttention(nn.Module):
    """
    Block-Causal Attention for V-JEPA 2-AC.
    
    Per the paper (Section 3.1):
    "We use a block-causal attention pattern in the predictor so that each 
    patch feature at a given time step can attend to the action, end-effector 
    state, and other patch features from the same timestep, as well as those 
    from previous time steps."
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            causal_mask: (seq_len, seq_len) boolean mask where True = masked (cannot attend)
        """
        B, N, D = x.shape
        
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if causal_mask is not None:
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        
        return out


class BlockCausalTransformerLayer(nn.Module):
    """Transformer layer with block-causal attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BlockCausalAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ActionConditionedPredictor(nn.Module):
    """
    V-JEPA 2-AC style action-conditioned world model predictor.
    
    Per paper (Section 3.1):
    - Transformer with block-causal attention
    - Processes all patch tokens (not just CLS)
    - Autoregressively predicts next frame representation
    
    Architecture scaled down from paper's 300M params for ARC domain:
    - Paper: 24 layers, 16 heads, 1024 hidden dim
    - Here: configurable, defaults to smaller for efficiency
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        continuous_dim: int = 2,
        num_discrete_actions: int = 11,  # default: NO-OP + 10 env discrete indices
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_patches: int = 64,  # (grid_size // patch_size) ** 2
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.continuous_dim = continuous_dim
        self.num_discrete_actions = num_discrete_actions
        self.num_patches = num_patches
        self.num_layers = num_layers
        
        # Action embedding dimension
        action_embed_dim = 64
        
        # Discrete action embedding
        self.discrete_embed = nn.Embedding(num_discrete_actions, action_embed_dim)
        
        # Continuous action projection
        self.continuous_proj = nn.Linear(continuous_dim, action_embed_dim)
        
        # Combined action projection to hidden dim
        self.action_proj = nn.Linear(action_embed_dim * 2, hidden_dim)
        
        # Patch token projection (from encoder latent_dim to hidden_dim)
        self.token_proj = nn.Linear(latent_dim, hidden_dim)
        
        # === FiLM CONDITIONING: Force action to modulate features ===
        # Actions generate scale and shift parameters that MUST affect the output
        self.film_generator = nn.Sequential(
            nn.Linear(action_embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # gamma and beta
        )
        
        # Learnable temporal position embedding for each timestep's tokens
        # (action token, then patch tokens at each timestep)
        self.time_embed = nn.Embedding(64, hidden_dim)  # max 64 timesteps
        
        # Block-causal transformer layers
        self.layers = nn.ModuleList([
            BlockCausalTransformerLayer(hidden_dim, num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection back to latent_dim for each patch
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        # Final norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Residual weight for smoother predictions
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
    
    def _create_block_causal_mask(
        self,
        num_timesteps: int,
        tokens_per_timestep: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create block-causal mask where tokens at time t can attend to 
        all tokens at times <= t.
        
        Returns:
            mask: (total_seq_len, total_seq_len) boolean mask, True = cannot attend
        """
        total_len = num_timesteps * tokens_per_timestep
        
        # Create timestep index for each position
        timesteps = torch.arange(num_timesteps, device=device).repeat_interleave(tokens_per_timestep)
        
        # Position i can attend to position j if timesteps[j] <= timesteps[i]
        mask = timesteps.unsqueeze(0) > timesteps.unsqueeze(1)  # True = cannot attend
        
        return mask
    
    def forward(
        self,
        z: torch.Tensor,
        continuous_action: torch.Tensor,
        discrete_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single-step prediction: predict next latent from current latent + action.
        
        Args:
            z: (batch, latent_dim) current CLS token OR (batch, num_patches+1, latent_dim) all tokens
            continuous_action: (batch, continuous_dim) cursor action
            discrete_action: (batch,) discrete action index
        Returns:
            z_next: (batch, latent_dim) predicted next CLS token
        """
        B = z.shape[0]
        
        # Handle both CLS-only and full-token inputs
        if z.dim() == 2:
            # CLS-only mode: (B, D) -> treat as single "patch"
            z = z.unsqueeze(1)  # (B, 1, D)
            num_patches = 1
        else:
            num_patches = z.shape[1]
        
        # Default discrete action = 0 (NONE/no-op)
        if discrete_action is None:
            discrete_action = torch.zeros(B, dtype=torch.long, device=z.device)
        
        # Embed action
        discrete_emb = self.discrete_embed(discrete_action)  # (B, 64)
        continuous_emb = self.continuous_proj(continuous_action)  # (B, 64)
        action_combined = torch.cat([discrete_emb, continuous_emb], dim=-1)  # (B, 128)
        action_token = self.action_proj(action_combined).unsqueeze(1)  # (B, 1, hidden_dim)
        
        # === FiLM: Generate action-dependent scale and shift ===
        # This FORCES the action to influence every feature
        film_params = self.film_generator(action_combined)  # (B, hidden_dim * 2)
        gamma = film_params[:, :self.hidden_dim].unsqueeze(1)  # (B, 1, hidden_dim) - scale
        beta = film_params[:, self.hidden_dim:].unsqueeze(1)   # (B, 1, hidden_dim) - shift
        
        # Project patch tokens
        patch_tokens = self.token_proj(z)  # (B, num_patches, hidden_dim)
        
        # Add temporal embedding (time=0 for single step)
        time_emb = self.time_embed(torch.zeros(1, dtype=torch.long, device=z.device))
        action_token = action_token + time_emb
        patch_tokens = patch_tokens + time_emb
        
        # Interleave: [action, patch1, patch2, ...] for this timestep
        x = torch.cat([action_token, patch_tokens], dim=1)  # (B, 1+num_patches, hidden_dim)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask=None)
        
        x = self.norm(x)
        
        # === Apply FiLM modulation AFTER transformer ===
        # This ensures action directly affects the output, can't be ignored
        x = gamma * x + beta
        
        # Output: project patch tokens back to latent_dim
        # Skip the action token (first token)
        patch_out = x[:, 1:, :]  # (B, num_patches, hidden_dim)
        patch_out = self.output_proj(patch_out)  # (B, num_patches, latent_dim)
        
        # Direct delta prediction - NO learned residual weight
        # Model MUST predict the change, can't learn to ignore it
        z_next = z + patch_out  # z_next = z + delta, model must learn delta
        
        # Return CLS token if that's what was input, else return all
        if num_patches == 1:
            return z_next.squeeze(1)  # (B, latent_dim)
        else:
            return z_next  # (B, num_patches, latent_dim)
    
    def forward_sequence(
        self,
        z_sequence: torch.Tensor,
        continuous_actions: torch.Tensor,
        discrete_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process a sequence of frames with actions for teacher-forcing training.
        
        Per paper (Section 3.1): "The sequence of feature maps, end-effector states, 
        and actions are temporally interleaved as (aₖ, sₖ, zₖ)ₖ∈[15]"
        
        Args:
            z_sequence: (batch, T, num_patches+1, latent_dim) encoded frames
            continuous_actions: (batch, T, continuous_dim) actions
            discrete_actions: (batch, T) discrete action indices
        Returns:
            z_pred: (batch, T, num_patches+1, latent_dim) predicted next frame tokens
        """
        B, T, N, D = z_sequence.shape
        
        if discrete_actions is None:
            discrete_actions = torch.zeros(B, T, dtype=torch.long, device=z_sequence.device)
        
        # Build interleaved sequence
        tokens_per_timestep = 1 + N  # action token + patch tokens
        total_seq_len = T * tokens_per_timestep
        
        sequence = []
        for t in range(T):
            # Action token for this timestep
            discrete_emb = self.discrete_embed(discrete_actions[:, t])
            continuous_emb = self.continuous_proj(continuous_actions[:, t])
            action_combined = torch.cat([discrete_emb, continuous_emb], dim=-1)
            action_token = self.action_proj(action_combined).unsqueeze(1)  # (B, 1, hidden_dim)
            
            # Patch tokens for this timestep
            patch_tokens = self.token_proj(z_sequence[:, t])  # (B, N, hidden_dim)
            
            # Add temporal embedding
            time_emb = self.time_embed(torch.tensor([t], device=z_sequence.device))
            action_token = action_token + time_emb
            patch_tokens = patch_tokens + time_emb
            
            sequence.append(action_token)
            sequence.append(patch_tokens)
        
        x = torch.cat(sequence, dim=1)  # (B, total_seq_len, hidden_dim)
        
        # Create block-causal mask
        causal_mask = self._create_block_causal_mask(T, tokens_per_timestep, x.device)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask=causal_mask)
        
        x = self.norm(x)
        
        # Extract patch predictions for each timestep
        predictions = []
        for t in range(T):
            start_idx = t * tokens_per_timestep + 1  # Skip action token
            end_idx = start_idx + N
            patch_out = x[:, start_idx:end_idx, :]  # (B, N, hidden_dim)
            patch_out = self.output_proj(patch_out)  # (B, N, latent_dim)
            
            # Residual
            alpha = torch.sigmoid(self.residual_weight)
            z_pred_t = z_sequence[:, t] + alpha * patch_out
            predictions.append(z_pred_t)
        
        return torch.stack(predictions, dim=1)  # (B, T, N, latent_dim)
    
    def multi_step_rollout(
        self,
        z: torch.Tensor,
        continuous_actions: torch.Tensor,
        discrete_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Roll out multiple steps autoregressively.
        
        Args:
            z: (batch, latent_dim) initial CLS latent OR (batch, num_patches+1, latent_dim)
            continuous_actions: (batch, horizon, continuous_dim)
            discrete_actions: (batch, horizon) or None
        Returns:
            z_trajectory: (batch, horizon+1, latent_dim) if CLS input
                         or (batch, horizon+1, num_patches+1, latent_dim) if full tokens
        """
        B, H, _ = continuous_actions.shape
        
        trajectory = [z]
        z_current = z
        
        for t in range(H):
            cont_act = continuous_actions[:, t]
            disc_act = discrete_actions[:, t] if discrete_actions is not None else None
            z_next = self.forward(z_current, cont_act, disc_act)
            trajectory.append(z_next)
            z_current = z_next
        
        return torch.stack(trajectory, dim=1)
