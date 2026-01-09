"""
Win Predictor - Predicts probability of winning from latent state.
"""

import torch
import torch.nn as nn


class WinPredictor(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for i in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
    def forward(self, z):
        return torch.sigmoid(self.net(z))
    
    def forward_logits(self, z):
        return self.net(z)


class ValuePredictor(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for i in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.net(z)


class RewardPredictor(nn.Module):
    def __init__(self, latent_dim: int = 256, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2 + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, z, action, z_next):
        x = torch.cat([z, action, z_next], dim=-1)
        return self.net(x)
