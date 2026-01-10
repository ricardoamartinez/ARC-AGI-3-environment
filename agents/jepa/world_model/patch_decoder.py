"""
Patch-based Decoder - Uses ALL patch tokens for high-fidelity reconstruction.

The key insight: CLS token alone (64 dims) cannot capture all spatial details.
Using all 64 patch tokens (64 * 64 = 4096 dims) preserves spatial structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchDecoder(nn.Module):
    """
    Decode from ALL patch tokens back to grid.
    
    This is much more powerful than CLS-only decoding because:
    - Each patch token encodes a specific spatial region
    - We can directly project each patch back to its grid region
    - Spatial information is preserved through the entire pipeline
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        grid_size: int = 64,
        patch_size: int = 8,
        num_colors: int = 11,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.num_colors = num_colors
        self.num_patches_per_side = grid_size // patch_size  # 8
        self.num_patches = self.num_patches_per_side ** 2  # 64
        
        # Per-patch decoder: latent_dim -> patch_size * patch_size * num_colors
        self.patch_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_size * patch_size * num_colors),
        )
        
        # Optional: Global context from CLS token
        self.cls_context = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # Refinement conv for smoothing patch boundaries
        self.refine = nn.Sequential(
            nn.Conv2d(num_colors, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, num_colors, kernel_size=3, padding=1),
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode patch tokens to grid logits.
        
        Args:
            z: (batch, num_patches+1, latent_dim) - includes CLS token at position 0
               OR (batch, T*num_patches+1, latent_dim) - temporal encoder output
               OR (batch, latent_dim) - CLS token only (fallback mode)
               
        Returns:
            (batch, num_colors, grid_size, grid_size) grid logits
        """
        if z.dim() == 2:
            # CLS-only mode: use CLS to decode via learned upsampling
            B = z.shape[0]
            cls_token = z
            cls_context = self.cls_context(cls_token)  # (B, latent_dim)
            # Expand CLS to all patches
            patch_tokens = cls_context.unsqueeze(1).expand(-1, self.num_patches, -1)
        else:
            B = z.shape[0]
            num_tokens = z.shape[1]
            
            # Extract CLS token (always first)
            cls_token = z[:, 0]  # (B, latent_dim)
            all_patch_tokens = z[:, 1:]  # (B, num_tokens-1, latent_dim)
            
            # Handle temporal encoder: may have T*num_patches tokens
            # We need exactly num_patches tokens for spatial reconstruction
            if all_patch_tokens.shape[1] > self.num_patches:
                # Temporal encoder output: take LAST frame's patches (most recent)
                # Or average across frames for smoother output
                num_frames = all_patch_tokens.shape[1] // self.num_patches
                # Reshape to (B, T, num_patches, D) and take last frame
                temporal_patches = all_patch_tokens.view(B, num_frames, self.num_patches, -1)
                patch_tokens = temporal_patches[:, -1]  # Last frame
            elif all_patch_tokens.shape[1] < self.num_patches:
                # Fewer patches than expected - pad with CLS-derived context
                cls_context = self.cls_context(cls_token)
                pad_size = self.num_patches - all_patch_tokens.shape[1]
                padding = cls_context.unsqueeze(1).expand(-1, pad_size, -1)
                patch_tokens = torch.cat([all_patch_tokens, padding], dim=1)
            else:
                patch_tokens = all_patch_tokens
            
            # Add global context from CLS
            cls_context = self.cls_context(cls_token)  # (B, latent_dim)
            patch_tokens = patch_tokens + cls_context.unsqueeze(1)
        
        # Decode each patch
        # (B, num_patches, latent_dim) -> (B, num_patches, P*P*C)
        patch_logits = self.patch_decoder(patch_tokens)
        
        # Reshape to spatial grid
        # (B, num_patches, P*P*C) -> (B, n_h, n_w, P, P, C)
        patch_logits = patch_logits.view(
            B, 
            self.num_patches_per_side, 
            self.num_patches_per_side,
            self.patch_size,
            self.patch_size,
            self.num_colors
        )
        
        # Rearrange to (B, C, H, W)
        patch_logits = patch_logits.permute(0, 5, 1, 3, 2, 4).contiguous()
        patch_logits = patch_logits.view(B, self.num_colors, self.grid_size, self.grid_size)
        
        # Refine to smooth patch boundaries
        logits = patch_logits + self.refine(patch_logits)
        
        return logits
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to discrete grid.
        
        Args:
            z: (batch, num_patches+1, latent_dim) or (batch, latent_dim)
            
        Returns:
            (batch, grid_size, grid_size) decoded grid
        """
        logits = self.forward(z)
        return logits.argmax(dim=1)


class ActionConditionedPatchDecoder(nn.Module):
    """
    Patch decoder with explicit action conditioning.
    
    This helps the decoder understand which action was taken,
    improving predictions for action-dependent state changes.
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        grid_size: int = 64,
        patch_size: int = 8,
        num_colors: int = 11,
        hidden_dim: int = 256,
        num_discrete_actions: int = 11,
        continuous_dim: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.num_colors = num_colors
        self.num_patches_per_side = grid_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        
        # Action embedding
        self.discrete_embed = nn.Embedding(num_discrete_actions, 32)
        self.continuous_proj = nn.Linear(continuous_dim, 32)
        self.action_proj = nn.Linear(64, latent_dim)
        
        # FiLM conditioning layers (scale and shift based on action)
        self.film_gamma = nn.Linear(latent_dim, latent_dim)
        self.film_beta = nn.Linear(latent_dim, latent_dim)
        
        # Per-patch decoder with action conditioning
        self.patch_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_size * patch_size * num_colors),
        )
        
        # Refinement
        self.refine = nn.Sequential(
            nn.Conv2d(num_colors, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, num_colors, kernel_size=3, padding=1),
        )
        
    def forward(
        self, 
        z: torch.Tensor,
        cont_action: torch.Tensor = None,
        disc_action: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Decode with action conditioning.
        
        Args:
            z: (batch, num_patches+1, latent_dim)
            cont_action: (batch, 2) continuous action
            disc_action: (batch,) discrete action index
        """
        if z.dim() == 2:
            B = z.shape[0]
            z = z.unsqueeze(1).expand(-1, self.num_patches + 1, -1)
        
        B = z.shape[0]
        patch_tokens = z[:, 1:]  # Skip CLS
        
        # Apply action conditioning via FiLM
        if cont_action is not None and disc_action is not None:
            disc_emb = self.discrete_embed(disc_action)
            cont_emb = self.continuous_proj(cont_action)
            action_emb = self.action_proj(torch.cat([disc_emb, cont_emb], dim=-1))
            
            # FiLM: gamma * x + beta
            gamma = self.film_gamma(action_emb).unsqueeze(1)  # (B, 1, D)
            beta = self.film_beta(action_emb).unsqueeze(1)
            patch_tokens = gamma * patch_tokens + beta
        
        # Decode patches
        patch_logits = self.patch_decoder(patch_tokens)
        
        # Reshape to grid
        patch_logits = patch_logits.view(
            B, 
            self.num_patches_per_side, 
            self.num_patches_per_side,
            self.patch_size,
            self.patch_size,
            self.num_colors
        )
        patch_logits = patch_logits.permute(0, 5, 1, 3, 2, 4).contiguous()
        patch_logits = patch_logits.view(B, self.num_colors, self.grid_size, self.grid_size)
        
        # Refine
        logits = patch_logits + self.refine(patch_logits)
        
        return logits
    
    def decode(self, z: torch.Tensor, cont_action=None, disc_action=None) -> torch.Tensor:
        logits = self.forward(z, cont_action, disc_action)
        return logits.argmax(dim=1)
