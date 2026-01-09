import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os


class SpatialAttention(nn.Module):
    """
    Self-Attention mechanism to learn spatial focalization.
    Takes feature map (B, C, H, W) and returns spatial weights (B, 1, H, W) and attended features (B, C).
    """

    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()

        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, N, C')
        proj_key = self.key(x).view(B, -1, H * W)  # (B, C', N)

        energy = torch.bmm(proj_query, proj_key)  # (B, N, N)

        energy_max = energy.max(dim=-1, keepdim=True)[0]
        attention = F.softmax(energy - energy_max, dim=-1)  # (B, N, N)

        proj_value = self.value(x).view(B, -1, H * W)  # (B, C, N)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B, C, N)
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out, attention


class SomaticOnlyFeatureExtractor(nn.Module):
    """
    Ultra-lightweight features for cursor-to-goal control.

    Uses only channels 4..9 (cursor_x, cursor_y, goal_x, goal_y, pain, dopamine),
    plus an explicit goal_present flag (from channel 3 goal marker),
    spatially reduced to (B,7) then passed through a small MLP.
    """

    def __init__(self, features_dim: int = 64):
        super().__init__()
        self.features_dim = int(features_dim)
        self.mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, self.features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations is BCHW in [0,1]
        # Channels 4-9: cursor_x, cursor_y, goal_x, goal_y, pain, dopamine
        somatic_map = observations[:, 4:10, :, :]
        somatic_vec = somatic_map.mean(dim=(2, 3))  # (B,6)
        # Channel 3: goal marker (0 or 1). Use max to get a clean present/absent flag.
        goal_present = observations[:, 3:4, :, :].amax(dim=(2, 3))  # (B,1)
        x = torch.cat([somatic_vec, goal_present], dim=-1)  # (B,7)
        return self.mlp(x)


class ArcViTFeatureExtractor(nn.Module):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__()

        self.grid_size = 64
        self.features_dim = features_dim

        # Do not assume a fixed palette size. Some ARC-like variants may use > 16 symbols/colors.
        # Observations are normalized to [0,1] then scaled back to [0,255] here.
        self.color_embedding = nn.Embedding(256, 16)

        # observation_space shape is (64, 64, 10).
        # Channel 0 is Color (Indices). We embed it -> 16 channels.
        # Channels 1-9 are scalar maps.
        # Total visual channels = 16 + 9 = 25.
        self.visual_channels = 25

        self.visual_cnn = nn.Sequential(
            nn.Conv2d(self.visual_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.ReLU(),
        )

        # Apply attention after downsampling to avoid OOM in batched SAC updates.
        # 8x8 => N=64 (cheap) vs 64x64 => N=4096 (explodes with batch sizes like 256).
        self.attention = SpatialAttention(256)

        self.visual_flatten_dim = 256 * 8 * 8

        # Somatic Stream (cursor/goal coords + pain/dopamine)
        self.somatic_dim = 6
        self.somatic_mlp = nn.Sequential(
            nn.Linear(self.somatic_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.fusion_dim = self.visual_flatten_dim + 32
        self.final_proj = nn.Linear(self.fusion_dim, features_dim)

        self.latest_feature_map = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x_255 = observations * 255.0 + 0.5

        colors_long = x_255[:, 0, :, :].long()
        colors_long = torch.clamp(colors_long, 0, 255)
        colors_emb = self.color_embedding(colors_long).permute(0, 3, 1, 2)

        other_channels = observations[:, 1:, :, :]
        visual_x = torch.cat([colors_emb, other_channels], dim=1)

        # Channels 4-9: cursor_x, cursor_y, goal_x, goal_y, pain, dopamine
        somatic_map = observations[:, 4:10, :, :]
        somatic_vec = somatic_map.mean(dim=[2, 3])

        feat_map = self.visual_cnn(visual_x)
        down = self.downsample(feat_map)
        attended_down, _attn = self.attention(down)

        if not self.training:
            with torch.no_grad():
                self.latest_feature_map = attended_down.mean(dim=1).detach()
        else:
            self.latest_feature_map = attended_down.mean(dim=1).detach()

        visual_flat = attended_down.flatten(start_dim=1)

        somatic_emb = self.somatic_mlp(somatic_vec)

        fused = torch.cat([visual_flat, somatic_emb], dim=1)
        return self.final_proj(fused)


class OnlineActorCritic(nn.Module):
    def __init__(
        self, observation_space: gym.spaces.Box, action_space: gym.Space, features_dim: int = 256
    ):
        super().__init__()

        feat_mode = os.environ.get("JEPA_FEATURES", "full").strip().lower()
        if feat_mode in ("somatic", "somatic_only", "coords"):
            self.features_extractor = SomaticOnlyFeatureExtractor(features_dim=features_dim)
        else:
            self.features_extractor = ArcViTFeatureExtractor(observation_space, features_dim)

        self.lstm = nn.LSTMCell(features_dim, features_dim)

        self.action_mode = "delta"
        if isinstance(action_space, spaces.MultiDiscrete) and len(action_space.nvec) == 2:
            if int(action_space.nvec[0]) == 64 and int(action_space.nvec[1]) == 64:
                self.action_mode = "target_cell"

        # Delta-mode (legacy): (ax,ay,trigger) + discrete (10)
        self.actor_continuous = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        self.log_std = nn.Parameter(torch.zeros(3) - 0.5)
        self.actor_discrete = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        # Target-cell mode: logits for x and y (each 64-way)
        self.actor_target = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.actor_target_x = nn.Linear(128, 64)
        self.actor_target_y = nn.Linear(128, 64)

        self.critic = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.apply(self._init_weights)

        # Critical stability tweak for online RL:
        # Initialize policy/value heads with smaller output scale so the initial policy does not
        # saturate to extreme tanh actions (which map to edges/corners and get "stuck").
        try:
            # Actor heads: small gain => near-zero mean initially
            if isinstance(self.actor_continuous[-1], nn.Linear):
                nn.init.orthogonal_(self.actor_continuous[-1].weight, gain=0.01)
                if self.actor_continuous[-1].bias is not None:
                    self.actor_continuous[-1].bias.data.zero_()
            if isinstance(self.actor_discrete[-1], nn.Linear):
                nn.init.orthogonal_(self.actor_discrete[-1].weight, gain=0.01)
                if self.actor_discrete[-1].bias is not None:
                    self.actor_discrete[-1].bias.data.zero_()
            if isinstance(self.actor_target_x, nn.Linear):
                nn.init.orthogonal_(self.actor_target_x.weight, gain=0.01)
                if self.actor_target_x.bias is not None:
                    self.actor_target_x.bias.data.zero_()
            if isinstance(self.actor_target_y, nn.Linear):
                nn.init.orthogonal_(self.actor_target_y.weight, gain=0.01)
                if self.actor_target_y.bias is not None:
                    self.actor_target_y.bias.data.zero_()
            # Critic head: standard gain
            if isinstance(self.critic[-1], nn.Linear):
                nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
                if self.critic[-1].bias is not None:
                    self.critic[-1].bias.data.zero_()
        except Exception:
            pass

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.LSTMCell):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif "bias" in name:
                    param.data.fill_(0.0)

    def forward(self, obs: torch.Tensor, hidden_states: tuple) -> tuple:
        features = self.features_extractor(obs)

        hx, cx = hidden_states
        if os.environ.get("JEPA_DISABLE_LSTM", "0") == "1":
            # Feedforward mode (much cheaper; good for low-lag UI). Keep shape compatible.
            next_hx, next_cx = features, cx.detach() * 0.0
        else:
            next_hx, next_cx = self.lstm(features, (hx, cx))

        mean_continuous = self.actor_continuous(next_hx)
        # Mean clip prevents base Normal mean from exploding, which would make tanh(action) saturate.
        mean_clip = float(os.environ.get("JEPA_MEAN_CLIP", "3.0"))
        if mean_clip > 0:
            mean_continuous = torch.clamp(mean_continuous, -mean_clip, mean_clip)

        # Prevent variance collapse (which can lock the policy into bad corner targets).
        # Default floor keeps some exploration while still allowing near-deterministic control.
        log_std_min = float(os.environ.get("JEPA_LOG_STD_MIN", "-2.0"))
        log_std_max = float(os.environ.get("JEPA_LOG_STD_MAX", "0.0"))
        if log_std_min > log_std_max:
            log_std_min, log_std_max = log_std_max, log_std_min
        clamped_log_std = torch.clamp(self.log_std, log_std_min, log_std_max)
        std_continuous = torch.exp(clamped_log_std)

        if self.action_mode == "target_cell":
            trunk = self.actor_target(next_hx)
            logits_x = self.actor_target_x(trunk)
            logits_y = self.actor_target_y(trunk)
            logits_discrete = torch.cat([logits_x, logits_y], dim=-1)  # (B, 128)
        else:
            logits_discrete = self.actor_discrete(next_hx)

        value = self.critic(next_hx)

        return mean_continuous, std_continuous, logits_discrete, value, (next_hx, next_cx)


