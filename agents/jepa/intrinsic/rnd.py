"""
Random Network Distillation (RND) for curiosity-driven exploration.

Reference: Burda et al. "Exploration by Random Network Distillation" (2018)

In sparse reward settings, RND provides intrinsic motivation by rewarding the agent
for visiting states that are "surprising" - i.e., states where the predictor network
cannot accurately predict the output of a fixed random target network.

Also includes StateCounter for simple count-based exploration bonuses.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger()


class RNDNetwork(nn.Module):
    """
    Small CNN for processing grid observations.
    Used for both target (fixed) and predictor (trained) networks.
    """

    def __init__(self, grid_size: int = 64, embedding_dim: int = 128, num_colors: int = 16):
        super().__init__()
        self.grid_size = grid_size
        self.embedding_dim = embedding_dim
        self.num_colors = num_colors

        # Embed each color as a vector (like a vocabulary)
        self.color_embed = nn.Embedding(num_colors, 8)

        # Simple CNN to process the grid
        # Input: (batch, 8, grid_size, grid_size) after embedding
        self.conv = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=4, stride=2, padding=1),  # -> 32x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # -> 8x8
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )

        # Calculate flattened size
        conv_out_size = 64 * 8 * 8  # 4096

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, embedding_dim),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: (batch, grid_size, grid_size) tensor of integer color indices [0, num_colors)

        Returns:
            (batch, embedding_dim) tensor of embeddings
        """
        # grid: (B, H, W) -> embedded: (B, H, W, 8) -> (B, 8, H, W)
        embedded = self.color_embed(grid.long())
        embedded = embedded.permute(0, 3, 1, 2).contiguous()

        conv_out = self.conv(embedded)
        return self.fc(conv_out)


class RNDModule:
    """
    Random Network Distillation module for computing intrinsic rewards.

    Usage:
        rnd = RNDModule(device="cuda")
        intrinsic_reward = rnd.compute_intrinsic_reward(grid_obs)
        rnd.update(grid_obs)  # Train predictor
    """

    def __init__(
        self,
        grid_size: int = 64,
        embedding_dim: int = 128,
        device: str = "cuda",
        lr: float = 1e-4,
        update_proportion: float = 0.25,
        reward_scale: float = 1.0,
        normalize_rewards: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.grid_size = grid_size
        self.embedding_dim = embedding_dim
        self.update_proportion = update_proportion
        self.reward_scale = reward_scale
        self.normalize_rewards = normalize_rewards

        # Target network (fixed random weights - never trained)
        self.target = RNDNetwork(grid_size, embedding_dim).to(self.device)
        for param in self.target.parameters():
            param.requires_grad = False
        self.target.eval()

        # Predictor network (trained to predict target's output)
        self.predictor = RNDNetwork(grid_size, embedding_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        # Running statistics for reward normalization
        self.reward_rms_mean = 0.0
        self.reward_rms_var = 1.0
        self.reward_rms_count = 1e-4

        logger.info(
            "RND initialized: grid_size=%d embed_dim=%d device=%s lr=%s scale=%s",
            grid_size,
            embedding_dim,
            self.device,
            lr,
            reward_scale,
        )

    def _preprocess(self, grid: np.ndarray) -> torch.Tensor:
        """Convert numpy grid to tensor and add batch dimension if needed."""
        if isinstance(grid, torch.Tensor):
            t = grid
        else:
            t = torch.from_numpy(grid)

        if t.dim() == 2:
            t = t.unsqueeze(0)  # Add batch dimension

        return t.to(self.device)

    def compute_intrinsic_reward(self, grid: np.ndarray) -> float:
        """
        Compute intrinsic reward for a single grid observation.

        The reward is the prediction error (MSE) between predictor and target.
        High error = novel state = high reward.
        """
        with torch.no_grad():
            grid_t = self._preprocess(grid)
            target_feat = self.target(grid_t)
            pred_feat = self.predictor(grid_t)

            # MSE between predictor and target
            error = F.mse_loss(pred_feat, target_feat, reduction="none").mean(dim=1)
            reward = float(error.item())

            # Normalize reward using running statistics
            if self.normalize_rewards:
                self._update_reward_rms(reward)
                reward = reward / max(np.sqrt(self.reward_rms_var), 1e-8)

            return reward * self.reward_scale

    def compute_intrinsic_reward_batch(self, grids: np.ndarray) -> np.ndarray:
        """Compute intrinsic rewards for a batch of grid observations."""
        with torch.no_grad():
            grids_t = self._preprocess(grids)
            target_feat = self.target(grids_t)
            pred_feat = self.predictor(grids_t)

            errors = F.mse_loss(pred_feat, target_feat, reduction="none").mean(dim=1)
            rewards = errors.cpu().numpy()

            if self.normalize_rewards:
                for i, r in enumerate(rewards):
                    self._update_reward_rms(r)
                rewards = rewards / max(np.sqrt(self.reward_rms_var), 1e-8)

            return rewards * self.reward_scale

    def _update_reward_rms(self, reward: float) -> None:
        """Update running mean/variance for reward normalization."""
        self.reward_rms_count += 1
        delta = reward - self.reward_rms_mean
        self.reward_rms_mean += delta / self.reward_rms_count
        delta2 = reward - self.reward_rms_mean
        self.reward_rms_var += (delta * delta2 - self.reward_rms_var) / self.reward_rms_count

    def update(self, grid: np.ndarray) -> float:
        """
        Update the predictor network on the given observation.
        Only updates with probability `update_proportion` to avoid overfitting.

        Returns the loss value.
        """
        if np.random.random() > self.update_proportion:
            return 0.0

        grid_t = self._preprocess(grid)

        with torch.no_grad():
            target_feat = self.target(grid_t)

        pred_feat = self.predictor(grid_t)
        loss = F.mse_loss(pred_feat, target_feat)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def update_batch(self, grids: np.ndarray) -> float:
        """Update predictor on a batch of observations."""
        grids_t = self._preprocess(grids)

        with torch.no_grad():
            target_feat = self.target(grids_t)

        pred_feat = self.predictor(grids_t)
        loss = F.mse_loss(pred_feat, target_feat)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())


class StateCounter:
    """
    Simple count-based exploration bonus.

    Tracks how often each state (grid configuration) has been visited
    and provides a bonus reward inversely proportional to the visit count.

    For large/continuous state spaces, uses a hash of the observation.
    """

    def __init__(
        self,
        reward_scale: float = 0.1,
        decay: float = 0.999,
        hash_obs: bool = True,
    ):
        self.reward_scale = reward_scale
        self.decay = decay
        self.hash_obs = hash_obs
        self.counts: dict[str, float] = defaultdict(float)
        self.total_visits = 0

        logger.info(
            "StateCounter initialized: scale=%s decay=%s hash=%s",
            reward_scale,
            decay,
            hash_obs,
        )

    def _get_key(self, obs: np.ndarray) -> str:
        """Get a hashable key for the observation."""
        if self.hash_obs:
            # Hash the flattened observation
            obs_bytes = obs.tobytes()
            return hashlib.md5(obs_bytes).hexdigest()
        else:
            # Use tuple representation (slower but exact)
            return str(tuple(obs.flatten().tolist()))

    def get_bonus(self, obs: np.ndarray) -> float:
        """
        Get exploration bonus for the observation.
        Bonus is scaled by 1/sqrt(count + 1).
        """
        key = self._get_key(obs)
        count = self.counts[key]
        bonus = self.reward_scale / np.sqrt(count + 1.0)
        return float(bonus)

    def update(self, obs: np.ndarray) -> None:
        """Record a visit to this state."""
        key = self._get_key(obs)
        self.counts[key] += 1.0
        self.total_visits += 1

        # Periodic decay to forget old states (helps in non-stationary envs)
        if self.decay < 1.0 and self.total_visits % 10000 == 0:
            for k in self.counts:
                self.counts[k] *= self.decay

    def get_stats(self) -> dict:
        """Get statistics about state visitation."""
        if not self.counts:
            return {"unique_states": 0, "total_visits": 0, "avg_count": 0}

        counts = list(self.counts.values())
        return {
            "unique_states": len(self.counts),
            "total_visits": self.total_visits,
            "avg_count": np.mean(counts),
            "max_count": np.max(counts),
            "min_count": np.min(counts),
        }


class CombinedIntrinsicReward:
    """
    Combines multiple intrinsic reward sources.

    Useful for using RND + count-based bonuses together.
    """

    def __init__(
        self,
        use_rnd: bool = True,
        use_counts: bool = True,
        rnd_scale: float = 1.0,
        count_scale: float = 0.1,
        device: str = "cuda",
        grid_size: int = 64,
    ):
        self.use_rnd = use_rnd
        self.use_counts = use_counts

        self.rnd: Optional[RNDModule] = None
        self.counter: Optional[StateCounter] = None

        if use_rnd:
            self.rnd = RNDModule(
                grid_size=grid_size,
                device=device,
                reward_scale=rnd_scale,
            )

        if use_counts:
            self.counter = StateCounter(reward_scale=count_scale)

        logger.info("CombinedIntrinsicReward: rnd=%s counts=%s", use_rnd, use_counts)

    def compute_reward(self, grid: np.ndarray) -> float:
        """Compute combined intrinsic reward."""
        reward = 0.0

        if self.rnd is not None:
            reward += self.rnd.compute_intrinsic_reward(grid)

        if self.counter is not None:
            reward += self.counter.get_bonus(grid)

        return reward

    def update(self, grid: np.ndarray) -> dict:
        """Update all intrinsic modules and return stats."""
        stats = {}

        if self.rnd is not None:
            rnd_loss = self.rnd.update(grid)
            stats["rnd_loss"] = rnd_loss

        if self.counter is not None:
            self.counter.update(grid)
            stats.update(self.counter.get_stats())

        return stats
