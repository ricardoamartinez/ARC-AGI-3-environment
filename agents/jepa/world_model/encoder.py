"""
Grid Encoder - V-JEPA 2 style Vision Transformer for ARC grids.

Encodes 2D game grids into latent embeddings using patch-based ViT.
Uses 3D Rotary Position Embedding (3D-RoPE) per V-JEPA 2 paper Section 2.1.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for transformers.
    
    Applies rotation to query/key vectors based on position, enabling
    better relative position awareness.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Precompute cos/sin for all positions
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)  # (seq_len, dim/2)
        # Create [cos, cos, ...], [sin, sin, ...] pattern
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply rotary embedding to input tensor.
        
        Args:
            x: (batch, seq_len, dim) or (batch, heads, seq_len, head_dim)
            positions: optional position indices
        """
        seq_len = x.shape[-2] if x.dim() == 4 else x.shape[1]
        
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        return self._apply_rotary(x, cos, sin)
    
    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding using rotation formula."""
        # Split into pairs and rotate
        x1, x2 = x[..., ::2], x[..., 1::2]
        # Interleave cos/sin properly
        cos = cos[..., ::2]
        sin = sin[..., ::2]
        
        # Rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1).flatten(-2)
        
        return rotated


class RotaryPositionalEmbedding3D(nn.Module):
    """
    3D Rotary Position Embedding for spatiotemporal patches.
    
    Per V-JEPA 2 paper (Section 2.1):
    "We use a 3D extension of traditional 1D-RoPE by partitioning the 
    feature dimension into three approximately equal segments (for the 
    temporal, height, and width axes) and applying the 1D rotations 
    separately to the segment for each axis."
    
    For 2D grids (no temporal), we use 2D-RoPE with H and W segments.
    """
    
    def __init__(
        self, 
        dim: int, 
        max_h: int = 64, 
        max_w: int = 64, 
        max_t: int = 1,
        base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_h = max_h
        self.max_w = max_w
        self.max_t = max_t
        
        # Partition dim into segments for T, H, W
        # For 2D (max_t=1), we split between H and W only
        if max_t > 1:
            self.dim_t = dim // 3
            self.dim_h = dim // 3
            self.dim_w = dim - 2 * (dim // 3)
        else:
            # 2D case: split between H and W
            self.dim_t = 0
            self.dim_h = dim // 2
            self.dim_w = dim - dim // 2
        
        # Compute inverse frequencies for each axis
        if self.dim_t > 0:
            inv_freq_t = 1.0 / (base ** (torch.arange(0, self.dim_t, 2).float() / self.dim_t))
            self.register_buffer('inv_freq_t', inv_freq_t, persistent=False)
        
        inv_freq_h = 1.0 / (base ** (torch.arange(0, self.dim_h, 2).float() / self.dim_h))
        inv_freq_w = 1.0 / (base ** (torch.arange(0, self.dim_w, 2).float() / self.dim_w))
        self.register_buffer('inv_freq_h', inv_freq_h, persistent=False)
        self.register_buffer('inv_freq_w', inv_freq_w, persistent=False)
        
        # Build position cache
        self._build_cache()
    
    def _build_cache(self):
        """Precompute cos/sin for all positions."""
        # Height positions
        pos_h = torch.arange(self.max_h, dtype=torch.float32)
        freqs_h = torch.outer(pos_h, self.inv_freq_h)
        emb_h = torch.cat([freqs_h, freqs_h], dim=-1)
        self.register_buffer('cos_h', emb_h.cos(), persistent=False)
        self.register_buffer('sin_h', emb_h.sin(), persistent=False)
        
        # Width positions
        pos_w = torch.arange(self.max_w, dtype=torch.float32)
        freqs_w = torch.outer(pos_w, self.inv_freq_w)
        emb_w = torch.cat([freqs_w, freqs_w], dim=-1)
        self.register_buffer('cos_w', emb_w.cos(), persistent=False)
        self.register_buffer('sin_w', emb_w.sin(), persistent=False)
        
        # Temporal positions (if 3D)
        if self.dim_t > 0:
            pos_t = torch.arange(self.max_t, dtype=torch.float32)
            freqs_t = torch.outer(pos_t, self.inv_freq_t)
            emb_t = torch.cat([freqs_t, freqs_t], dim=-1)
            self.register_buffer('cos_t', emb_t.cos(), persistent=False)
            self.register_buffer('sin_t', emb_t.sin(), persistent=False)
    
    def get_rotary_embedding(
        self, 
        num_patches_h: int, 
        num_patches_w: int, 
        num_frames: int = 1,
        device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos/sin embeddings for a grid of patches.
        
        Returns:
            cos, sin: (num_patches, dim) tensors
        """
        num_patches = num_patches_h * num_patches_w * num_frames
        
        cos_list = []
        sin_list = []
        
        for t in range(num_frames):
            for h in range(num_patches_h):
                for w in range(num_patches_w):
                    # Concatenate embeddings for each dimension
                    if self.dim_t > 0:
                        cos_patch = torch.cat([
                            self.cos_t[t],
                            self.cos_h[h],
                            self.cos_w[w]
                        ])
                        sin_patch = torch.cat([
                            self.sin_t[t],
                            self.sin_h[h],
                            self.sin_w[w]
                        ])
                    else:
                        cos_patch = torch.cat([self.cos_h[h], self.cos_w[w]])
                        sin_patch = torch.cat([self.sin_h[h], self.sin_w[w]])
                    
                    cos_list.append(cos_patch)
                    sin_list.append(sin_patch)
        
        cos = torch.stack(cos_list, dim=0)  # (num_patches, dim)
        sin = torch.stack(sin_list, dim=0)
        
        if device is not None:
            cos = cos.to(device)
            sin = sin.to(device)
        
        return cos, sin
    
    def apply_rotary_to_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embedding to query and key tensors.
        
        Args:
            q, k: (batch, heads, seq_len, head_dim) or (batch, seq_len, dim)
            cos, sin: (seq_len, dim)
        """
        # Handle different input shapes
        if q.dim() == 3:
            # (B, N, D) format
            cos = cos.unsqueeze(0)  # (1, N, D)
            sin = sin.unsqueeze(0)
        else:
            # (B, H, N, D) format - need to broadcast
            cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)
            sin = sin.unsqueeze(0).unsqueeze(0)
        
        q_rot = self._rotate(q, cos, sin)
        k_rot = self._rotate(k, cos, sin)
        
        return q_rot, k_rot
    
    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotation to tensor."""
        # Split and rotate pairs
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos_half = cos[..., ::2]
        sin_half = sin[..., ::2]
        
        rotated = torch.stack([
            x1 * cos_half - x2 * sin_half,
            x1 * sin_half + x2 * cos_half,
        ], dim=-1).flatten(-2)
        
        return rotated


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
    """
    Transformer block with pre-norm and optional RoPE.
    
    Supports 3D-RoPE per V-JEPA 2 paper for stable training at scale.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_rope: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        
        self.norm1 = nn.LayerNorm(dim)
        
        # Separate Q, K, V projections for RoPE application
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self, 
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            rope_cos, rope_sin: (seq_len, dim) rotary embeddings
        """
        B, N, D = x.shape
        
        # Pre-norm
        x_norm = self.norm1(x)
        
        # Project to Q, K, V
        q = self.q_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K (not V)
        if self.use_rope and rope_cos is not None and rope_sin is not None:
            # Reshape for per-head application
            cos = rope_cos.view(N, self.num_heads, self.head_dim).transpose(0, 1)  # (H, N, head_dim)
            sin = rope_sin.view(N, self.num_heads, self.head_dim).transpose(0, 1)
            
            # Apply rotation
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)
        attn_out = self.out_proj(attn_out)
        
        # Residual
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x
    
    def _apply_rope(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary embedding to tensor. x: (B, H, N, head_dim)"""
        # Split into pairs
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos_half = cos[..., ::2].unsqueeze(0)  # (1, H, N, head_dim/2)
        sin_half = sin[..., ::2].unsqueeze(0)
        
        # Rotate
        rotated = torch.stack([
            x1 * cos_half - x2 * sin_half,
            x1 * sin_half + x2 * cos_half,
        ], dim=-1).flatten(-2)
        
        return rotated


class GridEncoder(nn.Module):
    """
    V-JEPA 2 style encoder for game grids.
    
    Converts 2D grid -> sequence of patch embeddings -> latent representation.
    
    Uses 3D-RoPE (Rotary Position Embedding) per V-JEPA 2 paper Section 2.1
    instead of absolute positional embeddings for better stability at scale.
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
        use_rope: bool = True,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches_per_side = grid_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.use_rope = use_rope
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(grid_size, patch_size, num_colors, embed_dim)
        
        # 3D-RoPE for positional encoding (V-JEPA 2 style)
        # For 2D grids, we use 2D variant (no temporal dimension)
        if use_rope:
            self.rope = RotaryPositionalEmbedding3D(
                dim=embed_dim,
                max_h=self.num_patches_per_side,
                max_w=self.num_patches_per_side,
                max_t=1,  # 2D grid, no temporal
            )
            # No learnable pos_embed when using RoPE
            self.pos_embed = None
        else:
            # Fallback to learnable absolute positional embeddings
            self.rope = None
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Mask token used for V-JEPA-style masked denoising objectives.
        # When provided a `patch_mask`, masked patches are replaced by this token.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks with RoPE support
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, use_rope=use_rope)
            for _ in range(depth)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Cache for RoPE embeddings
        self._rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        # Initialize positional embedding (if not using RoPE)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _get_rope_embeddings(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get or compute RoPE embeddings for patches."""
        if self._rope_cache is not None:
            cos, sin = self._rope_cache
            if cos.device == device:
                return cos, sin
        
        cos, sin = self.rope.get_rotary_embedding(
            self.num_patches_per_side,
            self.num_patches_per_side,
            num_frames=1,
            device=device,
        )
        self._rope_cache = (cos, sin)
        return cos, sin
    
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
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional embedding (if not using RoPE)
        if self.pos_embed is not None:
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
        
        # Get RoPE embeddings for patch tokens (excluding CLS)
        rope_cos, rope_sin = None, None
        if self.use_rope and self.rope is not None:
            patch_cos, patch_sin = self._get_rope_embeddings(x.device)
            # Add zeros for CLS token position (no rotation for CLS)
            cls_zeros = torch.zeros(1, self.embed_dim, device=x.device)
            rope_cos = torch.cat([cls_zeros, patch_cos], dim=0)
            rope_sin = torch.cat([cls_zeros, patch_sin], dim=0)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, rope_cos=rope_cos, rope_sin=rope_sin)
        
        x = self.norm(x)
        
        if return_all_tokens:
            return x
        else:
            return x[:, 0]  # CLS token
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward with CLS token output."""
        return self.forward(x, return_all_tokens=False)
