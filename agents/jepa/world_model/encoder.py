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
        # RoPE uses pairs of dimensions, so each segment must be EVEN
        # For 2D (max_t=1), we split between H and W only
        if max_t > 1:
            # Split frequency pairs evenly across T, H, W
            total_pairs = dim // 2  # e.g., 32 pairs for dim=64
            pairs_t = total_pairs // 3  # e.g., 10 pairs
            pairs_h = total_pairs // 3  # e.g., 10 pairs
            pairs_w = total_pairs - 2 * pairs_t  # e.g., 12 pairs (remainder)
            self.dim_t = pairs_t * 2  # e.g., 20
            self.dim_h = pairs_h * 2  # e.g., 20
            self.dim_w = pairs_w * 2  # e.g., 24
            # Total: 20 + 20 + 24 = 64 ✓
        else:
            # 2D case: split between H and W (both even)
            self.dim_t = 0
            total_pairs = dim // 2
            pairs_h = total_pairs // 2
            pairs_w = total_pairs - pairs_h
            self.dim_h = pairs_h * 2
            self.dim_w = pairs_w * 2
        
        # Compute inverse frequencies for each axis
        # Each inv_freq has dim/2 elements, which get doubled via cat([freqs,freqs])
        if self.dim_t > 0:
            num_freqs_t = self.dim_t // 2
            inv_freq_t = 1.0 / (base ** (torch.arange(0, num_freqs_t).float() / num_freqs_t))
            self.register_buffer('inv_freq_t', inv_freq_t, persistent=False)
        
        num_freqs_h = self.dim_h // 2
        num_freqs_w = self.dim_w // 2
        inv_freq_h = 1.0 / (base ** (torch.arange(0, num_freqs_h).float() / num_freqs_h))
        inv_freq_w = 1.0 / (base ** (torch.arange(0, num_freqs_w).float() / num_freqs_w))
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


class TemporalGridEncoder(nn.Module):
    """
    Temporal-aware Grid Encoder that processes stacked frames for motion understanding.
    
    Key features:
    - Frame stacking: Encodes N consecutive frames together
    - Velocity channels: Explicit frame differences for motion detection
    - True 3D-RoPE: Uses temporal dimension for position encoding
    - Motion-aware representations for better moving object prediction
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
        num_frames: int = 4,  # Number of frames to stack
    ):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.num_patches_per_side = grid_size // patch_size
        self.num_patches_per_frame = self.num_patches_per_side ** 2
        self.total_patches = self.num_patches_per_frame * num_frames
        
        # Color embedding for grid values
        self.color_embed = nn.Embedding(num_colors, 16)
        
        # Patch projection with velocity channel
        # Each patch contains: color embeddings + velocity (frame diff)
        patch_dim = patch_size * patch_size * 16
        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        
        # Velocity encoder: encodes frame differences
        self.velocity_proj = nn.Linear(patch_size * patch_size, embed_dim // 4)
        self.combine_proj = nn.Linear(embed_dim + embed_dim // 4, embed_dim)
        
        # 3D-RoPE with temporal dimension enabled
        self.rope = RotaryPositionalEmbedding3D(
            dim=embed_dim,
            max_h=self.num_patches_per_side,
            max_w=self.num_patches_per_side,
            max_t=num_frames,  # Enable temporal dimension
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Temporal aggregation tokens (one per frame, learns to summarize each frame)
        self.temporal_tokens = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, use_rope=True)
            for _ in range(depth)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Frame history buffer
        self.frame_history: Optional[torch.Tensor] = None
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.temporal_tokens, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _compute_velocity_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity (frame difference) features.
        
        Args:
            frames: (B, T, H, W) stacked frames
            
        Returns:
            (B, T, num_patches, velocity_dim) velocity features
        """
        B, T, H, W = frames.shape
        
        # Compute frame differences (velocity)
        # Pad first frame with zeros (no previous frame)
        frame_diffs = torch.zeros_like(frames, dtype=torch.float32)
        frame_diffs[:, 1:] = (frames[:, 1:].float() - frames[:, :-1].float())
        
        # Normalize differences to [-1, 1]
        frame_diffs = frame_diffs / 16.0  # Assuming max color diff is ~16
        frame_diffs = frame_diffs.clamp(-1, 1)
        
        # Reshape to patches
        frame_diffs = frame_diffs.unfold(2, self.patch_size, self.patch_size)
        frame_diffs = frame_diffs.unfold(3, self.patch_size, self.patch_size)
        # (B, T, n_h, n_w, patch_size, patch_size)
        frame_diffs = frame_diffs.reshape(B, T, -1, self.patch_size * self.patch_size)
        # (B, T, num_patches, patch_size^2)
        
        return frame_diffs
    
    def _patchify_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Convert stacked frames to patch embeddings.
        
        Args:
            frames: (B, T, H, W) stacked frames
            
        Returns:
            (B, T*num_patches, embed_dim) patch embeddings
        """
        B, T, H, W = frames.shape
        
        # Embed colors for all frames
        color_emb = self.color_embed(frames.long())  # (B, T, H, W, 16)
        
        # Reshape to patches
        color_emb = color_emb.unfold(2, self.patch_size, self.patch_size)
        color_emb = color_emb.unfold(3, self.patch_size, self.patch_size)
        # (B, T, n_h, n_w, 16, patch_size, patch_size)
        color_emb = color_emb.permute(0, 1, 2, 3, 5, 6, 4).contiguous()
        # (B, T, n_h, n_w, patch_size, patch_size, 16)
        color_emb = color_emb.reshape(B, T, -1, self.patch_size * self.patch_size * 16)
        # (B, T, num_patches, patch_dim)
        
        # Project to embed_dim
        patch_emb = self.patch_proj(color_emb)  # (B, T, num_patches, embed_dim)
        
        # Compute velocity features
        velocity_feats = self._compute_velocity_features(frames)  # (B, T, num_patches, patch^2)
        velocity_emb = self.velocity_proj(velocity_feats)  # (B, T, num_patches, embed_dim//4)
        
        # Combine patch + velocity
        combined = torch.cat([patch_emb, velocity_emb], dim=-1)  # (B, T, num_patches, embed_dim + embed_dim//4)
        combined = self.combine_proj(combined)  # (B, T, num_patches, embed_dim)
        
        # Flatten temporal and spatial
        combined = combined.reshape(B, T * self.num_patches_per_frame, self.embed_dim)
        
        return combined
    
    def update_frame_history(self, frame: torch.Tensor):
        """
        Add a new frame to history buffer.
        
        Args:
            frame: (B, H, W) or (H, W) single frame
        """
        if frame.dim() == 2:
            frame = frame.unsqueeze(0)
        B = frame.shape[0]
        
        if self.frame_history is None or self.frame_history.shape[0] != B:
            # Initialize with repeated current frame
            self.frame_history = frame.unsqueeze(1).repeat(1, self.num_frames, 1, 1)
        else:
            # Shift and add new frame
            self.frame_history = torch.cat([
                self.frame_history[:, 1:],
                frame.unsqueeze(1)
            ], dim=1)
    
    def reset_history(self):
        """Reset frame history buffer."""
        self.frame_history = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = False,
        use_history: bool = True,
    ) -> torch.Tensor:
        """
        Encode frames to latent representation.
        
        Args:
            x: (B, H, W) single frame or (B, T, H, W) stacked frames
            return_all_tokens: Return all tokens or just CLS
            use_history: If True and single frame, use frame history
            
        Returns:
            Latent representation
        """
        # Handle single frame input
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        if x.dim() == 3:
            # Single frame - use history if available
            if use_history:
                self.update_frame_history(x)
                x = self.frame_history
            else:
                # Repeat single frame
                x = x.unsqueeze(1).repeat(1, self.num_frames, 1, 1)
        
        B, T, H, W = x.shape
        assert T == self.num_frames, f"Expected {self.num_frames} frames, got {T}"
        
        # Get patch embeddings with velocity
        patches = self._patchify_frames(x)  # (B, T*num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)  # (B, 1 + T*num_patches, embed_dim)
        
        # Get 3D-RoPE embeddings
        rope_cos, rope_sin = self.rope.get_rotary_embedding(
            self.num_patches_per_side,
            self.num_patches_per_side,
            num_frames=T,
            device=x.device,
        )
        # Add zeros for CLS token
        cls_zeros = torch.zeros(1, self.embed_dim, device=x.device)
        rope_cos = torch.cat([cls_zeros, rope_cos], dim=0)
        rope_sin = torch.cat([cls_zeros, rope_sin], dim=0)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, rope_cos=rope_cos, rope_sin=rope_sin)
        
        x = self.norm(x)
        
        if return_all_tokens:
            return x
        else:
            return x[:, 0]  # CLS token
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with CLS token output."""
        return self.forward(x, return_all_tokens=False)


class GridDecoder(nn.Module):
    """
    Decoder that converts latent embeddings back to grid predictions.
    
    Used for visualization and understanding what the world model "sees".
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        grid_size: int = 64,
        num_colors: int = 11,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.num_colors = num_colors
        
        # Project latent to spatial features
        self.project = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Upscale to grid
        # Start from 8x8 and upscale to grid_size
        self.init_size = 8
        self.project_spatial = nn.Linear(hidden_dim, self.init_size * self.init_size * 64)
        
        # Transposed convolutions to upscale
        # 8x8 -> 16x16 -> 32x32 -> 64x64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 8->16
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16->32
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 32->64
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, num_colors, kernel_size=3, padding=1),  # Final prediction
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to grid logits.
        
        Args:
            z: (batch, latent_dim) latent embedding
            
        Returns:
            (batch, num_colors, grid_size, grid_size) grid logits
        """
        B = z.shape[0]
        
        # Project to hidden
        h = self.project(z)  # (B, hidden_dim)
        
        # Project to spatial
        h = self.project_spatial(h)  # (B, init_size * init_size * 64)
        h = h.view(B, 64, self.init_size, self.init_size)
        
        # Decode to grid
        logits = self.decoder(h)  # (B, num_colors, grid_size, grid_size)
        
        return logits
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to discrete grid.
        
        Args:
            z: (batch, latent_dim) latent embedding
            
        Returns:
            (batch, grid_size, grid_size) decoded grid (uint8)
        """
        logits = self.forward(z)
        return logits.argmax(dim=1)
