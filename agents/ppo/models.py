import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

class SpatialAttention(nn.Module):
    """
    Self-Attention mechanism to learn spatial focalization.
    Takes feature map (B, C, H, W) and returns spatial weights (B, 1, H, W) and attended features (B, C).
    """
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        
        # Flatten H, W
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1) # (B, N, C')
        proj_key   = self.key(x).view(B, -1, H * W) # (B, C', N)
        
        # Attention Map
        energy = torch.bmm(proj_query, proj_key) # (B, N, N)
        
        # Stable Softmax
        energy_max = energy.max(dim=-1, keepdim=True)[0]
        attention = F.softmax(energy - energy_max, dim=-1) # (B, N, N)
        
        proj_value = self.value(x).view(B, -1, H * W) # (B, C, N)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # (B, C, N)
        out = out.view(B, C, H, W)
        
        out = self.gamma * out + x
        return out, attention

class ArcViTFeatureExtractor(nn.Module):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__()
        
        self.grid_size = 64
        self.features_dim = features_dim
        
        # Differentiable Color Embedding
        self.color_embedding = nn.Embedding(10, 10)
        
        # Visual Stream (CNN)
        self.visual_channels = 17
        
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(self.visual_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Spatial Attention
        self.attention = SpatialAttention(64)
        
        # Downsample
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # 8x8
            nn.ReLU(),
        )
        
        # 256 * 8 * 8 = 16384
        self.visual_flatten_dim = 256 * 8 * 8
        
        # Somatic Stream
        self.somatic_dim = 4
        self.somatic_mlp = nn.Sequential(
            nn.Linear(self.somatic_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Fusion
        self.fusion_dim = self.visual_flatten_dim + 32
        self.final_proj = nn.Linear(self.fusion_dim, features_dim)
        
        # Hook for visualization
        self.latest_feature_map = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x_255 = (observations * 255.0 + 0.5)
        
        # 1. Extract Visual Inputs & Differentiable Embedding
        colors_long = x_255[:, 0, :, :].long()
        colors_long = torch.clamp(colors_long, 0, 9)
        colors_emb = self.color_embedding(colors_long).permute(0, 3, 1, 2)
        
        deltas = observations[:, 1:2, :, :]
        focus  = observations[:, 2:3, :, :]
        goals  = observations[:, 3:4, :, :]
        
        vel_x = (observations[:, 4:5, :, :] * 255.0 - 128.0) / 128.0
        vel_y = (observations[:, 5:6, :, :] * 255.0 - 128.0) / 128.0
        
        # Add Spatial Pain and Dopamine
        pain_map = observations[:, 8:9, :, :]
        dopamine_map = observations[:, 9:10, :, :]
        
        visual_x = torch.cat([colors_emb, deltas, focus, goals, vel_x, vel_y, pain_map, dopamine_map], dim=1)
        
        # 2. Extract Somatic Inputs
        somatic_map = observations[:, 6:10, :, :]
        somatic_vec = somatic_map.mean(dim=[2, 3])
        
        # 3. Process Visual
        feat_map = self.visual_cnn(visual_x)
        attended_feat, attn_map = self.attention(feat_map)
        
        if not self.training:
             with torch.no_grad():
                 self.latest_feature_map = attended_feat.mean(dim=1).detach()
        else:
             self.latest_feature_map = attended_feat.mean(dim=1).detach()
        
        down = self.downsample(attended_feat)
        visual_flat = down.flatten(start_dim=1)
        
        # 4. Process Somatic
        somatic_emb = self.somatic_mlp(somatic_vec)
        
        # 5. Fuse
        fused = torch.cat([visual_flat, somatic_emb], dim=1)
        return self.final_proj(fused)

class OnlineActorCritic(nn.Module):
    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__()
        self.features_extractor = ArcViTFeatureExtractor(observation_space, features_dim)
        
        # Recurrent Layer (LSTM)
        # Input: features_dim, Hidden: features_dim
        self.lstm = nn.LSTMCell(features_dim, features_dim)
        
        # --- SPLIT OUTPUT HEADS ---
        # 1. Continuous Head
        self.actor_continuous = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3), # [vx, vy, trigger]
            nn.Tanh()
        )
        
        self.log_std = nn.Parameter(torch.zeros(3) - 0.5)
        
        # 2. Discrete Head
        self.actor_discrete = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        # Value Network
        self.critic = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.LSTMCell):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif 'bias' in name:
                    param.data.fill_(0.0)
    
    def forward(self, obs: torch.Tensor, hidden_states: tuple) -> tuple:
        """
        Forward pass with recurrence.
        Returns: (mean_cont, std_cont, logits_disc, value, (next_hx, next_cx))
        """
        features = self.features_extractor(obs)
        
        # LSTM Step
        hx, cx = hidden_states
        next_hx, next_cx = self.lstm(features, (hx, cx))
        
        # Heads use the new hidden state
        # Actor Heads
        mean_continuous = self.actor_continuous(next_hx)
        
        clamped_log_std = torch.clamp(self.log_std, -20.0, 2.0)
        std_continuous = torch.exp(clamped_log_std)
        
        logits_discrete = self.actor_discrete(next_hx)
        
        # Critic
        value = self.critic(next_hx)
        
        return mean_continuous, std_continuous, logits_discrete, value, (next_hx, next_cx)
