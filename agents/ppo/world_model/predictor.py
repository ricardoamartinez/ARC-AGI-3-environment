"""
Action-Conditioned Predictor - V-JEPA style dynamics model.

Supports hybrid action space: continuous cursor control + a discrete action *token*.

The discrete token is intentionally generic:
- token 0 is typically reserved for "NO-OP" (no discrete game action executed)
- tokens 1..N-1 map to environment-specific executed actions (e.g., key-presses, click types)

In ARC's default `ARCGymEnv` "delta" mode, the env produces an executed `final_action_idx` in -1..9
(-1 means no discrete action executed). In the world model we represent this as:

  disc_token = final_action_idx + 1    # -> 0..10
"""

import torch
import torch.nn as nn


class ActionConditionedPredictor(nn.Module):
    """
    Predicts next latent given current latent + hybrid action.
    
    Hybrid action = (continuous_cursor, discrete_game_action)
    - continuous: 2D cursor velocity/position
    - discrete: environment-specific token (0 = NO-OP)
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        continuous_dim: int = 2,
        num_discrete_actions: int = 11,  # default: NO-OP + 10 env discrete indices
        hidden_dim: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.continuous_dim = continuous_dim
        self.num_discrete_actions = num_discrete_actions
        
        # Discrete action embedding
        self.action_embed = nn.Embedding(num_discrete_actions, 64)
        
        # Total action input: continuous + discrete embedding
        action_input_dim = continuous_dim + 64
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim + action_input_dim, hidden_dim)
        
        # Predictor MLP
        layers = []
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.predictor = nn.Sequential(*layers)
        
        # Residual weight
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(
        self,
        z: torch.Tensor,
        continuous_action: torch.Tensor,
        discrete_action: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim) current latent
            continuous_action: (batch, continuous_dim) cursor action
            discrete_action: (batch,) discrete action index (0=NONE, 1-7=ACTION1-7, 8=RESET)
        Returns:
            z_next: (batch, latent_dim) predicted next latent
        """
        B = z.shape[0]
        
        # Default discrete action = 0 (NONE/no-op)
        if discrete_action is None:
            discrete_action = torch.zeros(B, dtype=torch.long, device=z.device)
        
        # Embed discrete action
        action_emb = self.action_embed(discrete_action)  # (B, 64)
        
        # Concatenate continuous + discrete
        action_repr = torch.cat([continuous_action, action_emb], dim=-1)
        
        # Concatenate latent and action
        x = torch.cat([z, action_repr], dim=-1)
        
        # Project and predict
        x = self.input_proj(x)
        delta_z = self.predictor(x)
        
        # Residual connection
        alpha = torch.sigmoid(self.residual_weight)
        z_next = z + alpha * delta_z
        
        return z_next
    
    def multi_step_rollout(
        self,
        z: torch.Tensor,
        continuous_actions: torch.Tensor,
        discrete_actions: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Roll out multiple steps.
        
        Args:
            z: (batch, latent_dim) initial latent
            continuous_actions: (batch, horizon, continuous_dim)
            discrete_actions: (batch, horizon) or None
        Returns:
            z_trajectory: (batch, horizon+1, latent_dim)
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
