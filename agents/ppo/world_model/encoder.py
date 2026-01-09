"""
Grid Encoder - V-JEPA style Vision Transformer for ARC grids.

Encodes 2D game grids into latent embeddings using patch-based ViT.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Convert grid into patch embeddings."""
    
    def __init__(
        self,
        grid_size: int = 64,
        patch_size: int = 8,
        num_colors: int = 16,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.num_patches = (grid_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Color embedding (like vocabulary)
        self.color_embed = nn.Embedding(num_colors, 16)
        
        # Patch projection: (patch_size * patch_size * color_dim) -> embed_dim
        patch_dim = patch_size * patch_size * 16
        self.proj = nn.Linear(patch_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, height, width) grid of color indices
        Returns:
            (batch, num_patches, embed_dim) patch embeddings
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, H, W = x.shape
        
        # Embed colors: (B, H, W) -> (B, H, W, 16)
        x = self.color_embed(x.long())
        
        # Reshape to patches: (B, H, W, 16) -> (B, num_patches, patch_dim)
        x = x.unfold(1, self.patch_size, self.patch_size)  # (B, n_h, W, 16, p)
        x = x.unfold(2, self.patch_size, self.patch_size)  # (B, n_h, n_w, 16, p, p)
        x = x.permute(0, 1, 2, 4, 5, 3).contiguous()  # (B, n_h, n_w, p, p, 16)
        x = x.view(B, -1, self.patch_size * self.patch_size * 16)  # (B, num_patches, patch_dim)
        
        # Project to embed_dim
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class GridEncoder(nn.Module):
    """
    V-JEPA style encoder for game grids.
    
    Converts 2D grid -> sequence of patch embeddings -> latent representation.
    """
    
    def __init__(
        self,
        grid_size: int = 64,
        patch_size: int = 8,
        num_colors: int = 256,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (grid_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(grid_size, patch_size, num_colors, embed_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Mask token used for V-JEPA-style masked denoising objectives.
        # When provided a `patch_mask`, masked patches are replaced by this token.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = False,
        patch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, height, width) grid of color indices
            return_all_tokens: if True, return all patch tokens; else return CLS token
            patch_mask: optional boolean mask of shape (batch, num_patches) where True indicates a masked patch
        Returns:
            if return_all_tokens: (batch, num_patches+1, embed_dim)
            else: (batch, embed_dim) - CLS token only
        """
        B = x.shape[0]
        
        # Patch embedding + positional embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed

        # Optional patch masking (V-JEPA style): replace masked patch embeddings with a learned mask token.
        if patch_mask is not None:
            # Accept (num_patches,) and broadcast to batch.
            if patch_mask.dim() == 1:
                patch_mask = patch_mask.unsqueeze(0).expand(B, -1)
            # Ensure boolean type on the right device.
            patch_mask = patch_mask.to(device=x.device, dtype=torch.bool)
            mask_tok = self.mask_token.expand(B, self.num_patches, -1)
            x = torch.where(patch_mask.unsqueeze(-1), mask_tok, x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        if return_all_tokens:
            return x
        else:
            return x[:, 0]  # CLS token
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward with CLS token output."""
        return self.forward(x, return_all_tokens=False)
