from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import ArcViTFeatureExtractor


@dataclass
class SquashedGaussianOutput:
    action: torch.Tensor  # (B, A) in [-1,1]
    log_prob: torch.Tensor  # (B,)
    mean_action: torch.Tensor  # (B, A) in [-1,1]


class SomaticFeatureExtractor(nn.Module):
    """
    Lightweight features for cursor-to-goal control.

    Uses only the explicit coordinate channels + pain/dopamine (channels 4..9),
    plus an explicit goal_present flag (from channel 3 goal marker),
    averaged/reduced over space to a 7D vector, then an MLP.
    """

    def __init__(self, features_dim: int = 128):
        super().__init__()
        self.features_dim = int(features_dim)
        self.mlp = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, self.features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs is BCHW in [0,1]. Channels 4-9: cursor_x, cursor_y, goal_x, goal_y, pain, dopamine
        somatic_map = obs[:, 4:10, :, :]
        somatic_vec = somatic_map.mean(dim=(2, 3))  # (B,6)
        goal_present = obs[:, 3:4, :, :].amax(dim=(2, 3))  # (B,1)
        x = torch.cat([somatic_vec, goal_present], dim=-1)  # (B,7)
        return self.mlp(x)


class SACActor(nn.Module):
    """
    Squashed-Gaussian actor for continuous control (SAC).
    """

    def __init__(self, observation_space, action_dim: int = 2, features_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim

        feat_mode = os.environ.get("JEPA_SAC_FEATURES", "full").strip().lower()
        if feat_mode in ("somatic", "somatic_only", "coords"):
            self.features = SomaticFeatureExtractor(features_dim=features_dim)
        else:
            self.features = ArcViTFeatureExtractor(observation_space, features_dim)

        self.trunk = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

        # SAC standard bounds for numerical stability
        self.min_log_std = -20.0
        self.max_log_std = 2.0

        self.apply(self._init_weights)

        # Make "go fast by default" easy to discover:
        # If using (x,y,speed_scale) action, bias the last dim toward +2 => tanh ~ 0.96.
        if self.action_dim >= 3 and self.mean.bias is not None:
            with torch.no_grad():
                self.mean.bias[-1].fill_(2.0)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.features(obs)
        h = self.trunk(z)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), self.min_log_std, self.max_log_std)
        return mean, log_std

    def sample(self, obs: torch.Tensor, deterministic: bool) -> SquashedGaussianOutput:
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        if deterministic:
            u = mean
        else:
            u = mean + std * torch.randn_like(mean)

        a = torch.tanh(u)
        mean_a = torch.tanh(mean)

        # Log prob with tanh correction
        # log N(u | mean, std) - sum(log(1 - tanh(u)^2))
        normal_log_prob = (
            -0.5 * (((u - mean) / (std + 1e-8)) ** 2 + 2.0 * log_std + torch.log(torch.tensor(2.0 * torch.pi)))
        ).sum(dim=-1)
        correction = torch.log(torch.clamp(1.0 - a.pow(2), min=1e-6)).sum(dim=-1)
        log_prob = normal_log_prob - correction

        return SquashedGaussianOutput(action=a, log_prob=log_prob, mean_action=mean_a)


class SACCritic(nn.Module):
    """
    Q(s,a) critic for SAC.
    """

    def __init__(self, observation_space, action_dim: int = 2, features_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim

        feat_mode = os.environ.get("JEPA_SAC_FEATURES", "full").strip().lower()
        if feat_mode in ("somatic", "somatic_only", "coords"):
            self.features = SomaticFeatureExtractor(features_dim=features_dim)
        else:
            self.features = ArcViTFeatureExtractor(observation_space, features_dim)

        self.q = nn.Sequential(
            nn.Linear(features_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = self.features(obs)
        x = torch.cat([z, action], dim=-1)
        return self.q(x).squeeze(-1)


