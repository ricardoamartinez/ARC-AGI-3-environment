import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class ArcViTFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # ARC-AGI-3 Config
        self.grid_size = 64
        # We switch to CNN for speed and 1:1 feature mapping
        # But keep class name for compatibility
        
        # Input: 
        # 0: Color (OneHot 10)
        # 1: Delta (1)
        # 2: Mask (1)
        # 3: Goal (1)
        # 4: VelX (1)
        # 5: VelY (1)
        # Total: 10 + 1 + 1 + 1 + 1 + 1 = 15 channels
        self.input_channels = 15
        
        # CNN Architecture (maintains resolution 64x64 to capture fine details)
        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Feature Map Hook
        self.latest_feature_map = None
        
        # Flatten: 64 * 64 * 64 = 262,144 (Too big for Linear)
        # So we use Global Average Pooling or Strided Convs?
        # User wants 1:1 understanding.
        # Let's add a few strided layers to downsample safely.
        
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # 8x8
            nn.ReLU(),
        )
        
        # 256 * 8 * 8 = 16384
        self.flatten_dim = 256 * 8 * 8
        self.final_proj = nn.Linear(self.flatten_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Optimization: Remove .max() check to avoid CPU-GPU sync. 
        # Assume observations are already normalized to [0, 1] by SB3 (standard for uint8 Box).
        # We need 0-255 scale for colors and some logic.
        
        # Safe rounding to recover integers
        x_255 = (observations * 255.0 + 0.5)
        
        # One-Hot Encoding
        colors_long = x_255[:, 0, :, :].long() # (B, 64, 64)
        colors_long = torch.clamp(colors_long, 0, 9)
        colors_one_hot = torch.nn.functional.one_hot(colors_long, num_classes=10).float()
        # Permute One-Hot to Channel First: (B, H, W, C) -> (B, C, H, W)
        colors_one_hot = colors_one_hot.permute(0, 3, 1, 2)

        # Other channels are simpler if we use the 0-1 inputs directly where appropriate
        # Deltas, Masks, Goals are originally 0 or 255 in ObservationBuilder, so 0.0 or 1.0 here.
        # But wait, goals can be 0-255. So we want them 0-1.
        # ObservationBuilder: delta[delta > 0] = 255. So it's binary. 0 or 1.
        # Goals: normalized to 0-255. So here 0-1.
        
        deltas = observations[:, 1:2, :, :] # Keep dim (B, 1, 64, 64)
        masks  = observations[:, 2:3, :, :]
        goals  = observations[:, 3:4, :, :]
        
        # Velocity was normalized to 0-255 in Env.
        # We want to map it back to approx [-1, 1].
        # Env: norm = (v + 10) / 20 * 255
        # Here: obs = norm / 255 = (v + 10) / 20
        # obs * 20 - 10 = v.
        # We want v_norm = v / (something). 
        # Previous logic: (x_255 - 128) / 128.
        # x_255 = obs * 255.
        # (obs * 255 - 128) / 128 = obs * 1.99 - 1.0.
        
        vel_x = (observations[:, 4:5, :, :] * 255.0 - 128.0) / 128.0
        vel_y = (observations[:, 5:6, :, :] * 255.0 - 128.0) / 128.0
        
        # (B, 15, 64, 64)
        # colors_one_hot is (B, 10, 64, 64)
        x = torch.cat([colors_one_hot, deltas, masks, goals, vel_x, vel_y], dim=1)
        
        # Run CNN
        feat_map = self.cnn(x)
        
        # Store Feature Map for Visualization (Detached)
        # Average across channels to get spatial intensity: (B, 64, 64, 64) -> (B, 64, 64)
        if not self.training: 
             with torch.no_grad():
                 self.latest_feature_map = feat_map.mean(dim=1).detach() 
        else:
             self.latest_feature_map = feat_map.mean(dim=1).detach()
        
        x = self.downsample(feat_map)
        
        x = x.flatten(start_dim=1)
        return self.final_proj(x)
