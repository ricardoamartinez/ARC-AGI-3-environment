import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class ArcViTFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # ARC-AGI-3 Config
        self.grid_size = 64
        self.patch_size = 8 # 64x64 grid / 8x8 patches = 8x8 patches = 64 tokens
        self.num_patches = (self.grid_size // self.patch_size) ** 2
        self.embed_dim = 128
        self.n_heads = 4
        self.n_layers = 4
        
        # 1. Embedding Layers
        # Channel 0: Color (Integers 0-255). We use an Embedding layer.
        # We allow up to 256 possible values (uint8).
        self.color_embedding = nn.Embedding(num_embeddings=256, embedding_dim=16)
        
        # Channels 1 & 2: Delta & Mask.
        # These are continuous-ish (0-255), but mask is sparse. 
        # We will project them linearly.
        # Input to patch projection:
        # Per pixel: ColorEmbed(16) + Delta(1) + Mask(1) = 18 dims
        # Patch size 8x8 = 64 pixels.
        # Patch input dim = 64 * 18 = 1152
        
        self.pixel_dim = 16 + 2 # 16 for color, 1 for delta, 1 for mask
        self.patch_input_dim = (self.patch_size ** 2) * self.pixel_dim
        
        self.patch_projection = nn.Linear(self.patch_input_dim, self.embed_dim)
        
        # 2. Positional Embeddings (Learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.n_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        
        # 4. Final Projection to features_dim
        self.final_proj = nn.Linear(self.embed_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (Batch, 3, 64, 64)
        # SB3 might have normalized this to [0,1]. We need to be careful.
        
        # Ensure we have 0-255 range for embeddings
        if observations.max() <= 1.0:
            observations = observations * 255.0
            
        x = observations # (B, 3, 64, 64)
        
        # Split channels
        colors = x[:, 0, :, :].long() # (B, 64, 64) - Ints for Embedding
        deltas = x[:, 1, :, :].float() / 255.0 # (B, 64, 64) - Float 0-1
        masks  = x[:, 2, :, :].float() / 255.0 # (B, 64, 64) - Float 0-1
        
        # Embed Colors: (B, 64, 64, 16)
        colors_emb = self.color_embedding(colors)
        
        # Concatenate meta-data: (B, 64, 64, 18)
        # Add singleton dim to deltas/masks for concat: (B, 64, 64, 1)
        deltas = deltas.unsqueeze(-1)
        masks = masks.unsqueeze(-1)
        full_grid = torch.cat([colors_emb, deltas, masks], dim=-1)
        
        # Patchify
        # Reshape to (B, N_Patches, Patch_Pixels, Channels)
        # (B, 8, 8, 8, 8, 18) -> (B, 8, 8, 8, 8, 18)
        B = x.shape[0]
        GS = self.grid_size # 64
        PS = self.patch_size # 8
        
        # Unfold creates patches
        # Input: (B, GS, GS, C)
        # Output: (B, GS/PS, GS/PS, PS, PS, C)
        # We need to handle the permutes carefully.
        
        # Easier way: Reshape
        # (B, H, W, C) -> (B, H/P, P, W/P, P, C) -> (B, H/P, W/P, P, P, C)
        x_reshaped = full_grid.view(B, GS // PS, PS, GS // PS, PS, self.pixel_dim)
        x_reshaped = x_reshaped.permute(0, 1, 3, 2, 4, 5) # (B, 8, 8, 8, 8, 18)
        
        # Flatten patches: (B, 64, 8*8*18) = (B, 64, 1152)
        patches = x_reshaped.contiguous().view(B, -1, PS * PS * self.pixel_dim)
        
        # Project patches to embeddings
        x = self.patch_projection(patches) # (B, 64, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, 65, embed_dim)
        
        # Add Positional Embedding
        x = x + self.pos_embedding
        
        # Run Transformer
        x = self.transformer(x)
        
        # Take CLS token result
        output = x[:, 0]
        
        return self.final_proj(output)
