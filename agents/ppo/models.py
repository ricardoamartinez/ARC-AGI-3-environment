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
        # Ensure 0-255 range
        if observations.max() <= 1.0:
            observations = observations * 255.0
            
        x = observations
        
        # One-Hot Encoding to allow Gradients
        colors_long = x[:, 0, :, :].long() # (B, 64, 64)
        deltas = x[:, 1, :, :].float() / 255.0
        masks  = x[:, 2, :, :].float() / 255.0
        goals  = x[:, 3, :, :].float() / 255.0
        vel_x  = (x[:, 4, :, :].float() - 128.0) / 128.0 # Normalize to ~[-1, 1]
        vel_y  = (x[:, 5, :, :].float() - 128.0) / 128.0
        
        colors_long = torch.clamp(colors_long, 0, 9)
        colors_one_hot = torch.nn.functional.one_hot(colors_long, num_classes=10).float()
        # Permute One-Hot to Channel First: (B, H, W, C) -> (B, C, H, W)
        colors_one_hot = colors_one_hot.permute(0, 3, 1, 2)
        
        deltas = deltas.unsqueeze(1) # (B, 1, 64, 64)
        masks = masks.unsqueeze(1)
        goals = goals.unsqueeze(1)
        vel_x = vel_x.unsqueeze(1)
        vel_y = vel_y.unsqueeze(1)
        
        # (B, 15, 64, 64)
        x = torch.cat([colors_one_hot, deltas, masks, goals, vel_x, vel_y], dim=1)
        
        # Run CNN
        feat_map = self.cnn(x)
        
        # Store Feature Map for Visualization (Detached)
        # Average across channels to get spatial intensity: (B, 64, 64, 64) -> (B, 64, 64)
        # Take the last item in batch
        if not self.training: # Optional optimization? No, we need it during training viz
             with torch.no_grad():
                 self.latest_feature_map = feat_map.mean(dim=1).detach() # (B, 64, 64)
        else:
             # Detach to avoid leak
             self.latest_feature_map = feat_map.mean(dim=1).detach()
        
        x = self.downsample(feat_map)
        
        x = x.flatten(start_dim=1)
        return self.final_proj(x)
