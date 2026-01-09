"""
V-JEPA 2 Inspired Joint Embedding for Visual-Action Learning.

Key ideas from V-JEPA 2:
1. Learn representations in joint embedding space (no pixel reconstruction)
2. Predict in latent space, not pixel space
3. Use EMA target encoder for stable targets
4. Contrastive alignment between visual states and action outcomes

This module learns:
- Which actions will have effects in the current visual state
- Visual-action coordination for mouse and keyboard
- Fast adaptation to avoid penalized (ineffective) actions
"""

import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .encoder import GridEncoder


class ActionEmbedding(nn.Module):
    """
    Learnable embeddings for discrete actions in joint space.
    Also handles continuous cursor position encoding.
    """
    
    def __init__(
        self,
        num_discrete_actions: int = 10,
        embed_dim: int = 256,
        cursor_dim: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_discrete_actions = num_discrete_actions
        
        # Discrete action embeddings (0-9: click variants, UP, DOWN, LEFT, RIGHT, SPACE, ENTER)
        self.action_embed = nn.Embedding(num_discrete_actions, embed_dim)
        
        # Cursor position encoding (for click/spatial actions)
        self.cursor_encoder = nn.Sequential(
            nn.Linear(cursor_dim, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )
        
        # Combine discrete action + cursor position
        self.combiner = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.action_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        action_idx: torch.Tensor,
        cursor_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            action_idx: (batch,) discrete action indices
            cursor_pos: (batch, 2) normalized cursor positions in [-1, 1]
        Returns:
            (batch, embed_dim) action embeddings
        """
        # Get discrete action embedding
        action_emb = self.action_embed(action_idx)  # (B, embed_dim)
        
        if cursor_pos is not None:
            # Encode cursor position
            cursor_emb = self.cursor_encoder(cursor_pos)  # (B, embed_dim)
            # Combine action + cursor
            combined = torch.cat([action_emb, cursor_emb], dim=-1)  # (B, 2*embed_dim)
            return self.combiner(combined)
        else:
            return action_emb


class ActionEffectivenessPredictor(nn.Module):
    """
    Predicts whether an action will be effective given the visual state.
    
    Uses cross-attention between visual features and action embeddings
    to predict action effectiveness (will it change the game state?).
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_discrete_actions: int = 10,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_discrete_actions = num_discrete_actions
        
        # Cross-attention: action queries visual state
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True
        )
        
        # Prediction head: visual + action -> effectiveness probability
        self.pred_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        action_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: (batch, num_patches+1, embed_dim) from GridEncoder
            action_embed: (batch, embed_dim) action embedding
        Returns:
            (batch, 1) effectiveness logits (sigmoid to get probability)
        """
        # Action as query, visual as key/value
        action_query = action_embed.unsqueeze(1)  # (B, 1, embed_dim)
        
        attn_out, _ = self.cross_attn(
            action_query, visual_features, visual_features
        )  # (B, 1, embed_dim)
        
        # Predict effectiveness
        logits = self.pred_head(attn_out.squeeze(1))  # (B, 1)
        return logits


class ActionJEPA(nn.Module):
    """
    V-JEPA 2 Inspired Joint Embedding Predictive Architecture for Actions.
    
    Components:
    1. Visual encoder (GridEncoder) - encodes game state to latent
    2. Action embeddings - encodes actions in same space
    3. Effectiveness predictor - predicts if action will work
    4. EMA target encoder - stable targets for contrastive learning
    
    Training objectives:
    1. Action effectiveness prediction (binary classification)
    2. Contrastive alignment (effective actions closer to matching states)
    3. Latent prediction (predict next state embedding from current + action)
    """
    
    def __init__(
        self,
        grid_size: int = 64,
        embed_dim: int = 128,
        num_discrete_actions: int = 10,
        encoder_depth: int = 4,
        encoder_heads: int = 4,
        patch_size: int = 8,
        device: str = "cuda",
        ema_decay: float = 0.996,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.num_discrete_actions = num_discrete_actions
        self.ema_decay = ema_decay
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Visual encoder (online)
        self.visual_encoder = GridEncoder(
            grid_size=grid_size,
            patch_size=patch_size,
            num_colors=256,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
        ).to(self.device)
        
        # Visual encoder (EMA target - not trained directly)
        self.target_encoder = GridEncoder(
            grid_size=grid_size,
            patch_size=patch_size,
            num_colors=256,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
        ).to(self.device)
        # Initialize target as copy of online
        self.target_encoder.load_state_dict(self.visual_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        
        # Action embeddings
        self.action_embed = ActionEmbedding(
            num_discrete_actions=num_discrete_actions,
            embed_dim=embed_dim,
        ).to(self.device)
        
        # Effectiveness predictor
        self.effectiveness_pred = ActionEffectivenessPredictor(
            embed_dim=embed_dim,
            num_heads=encoder_heads,
            num_discrete_actions=num_discrete_actions,
        ).to(self.device)
        
        # Latent predictor: predict next state embedding from current + action
        self.latent_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        ).to(self.device)
        
        # All action effectiveness predictor: given visual, predict which actions will work
        self.all_actions_pred = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_discrete_actions),
        ).to(self.device)
        
        # Grid decoder: reconstruct grid from latent embedding
        # Simple MLP decoder for visualization (not for training loss)
        num_patches = (grid_size // patch_size) ** 2
        self.grid_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, grid_size * grid_size * 10),  # 10 possible colors
        ).to(self.device)
        self.decoder_grid_size = grid_size
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        # Experience buffer for training
        self.buffer_states: list = []
        self.buffer_actions: list = []
        self.buffer_cursors: list = []
        self.buffer_had_effect: list = []
        self.buffer_next_states: list = []
        self.buffer_size = 10000
        
        # Training stats
        self.train_steps = 0
        self.effective_acc = 0.0
        self.pred_loss = 0.0
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update of target encoder."""
        for p_online, p_target in zip(
            self.visual_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            p_target.data = self.ema_decay * p_target.data + (1 - self.ema_decay) * p_online.data
    
    def add_experience(
        self,
        state: np.ndarray,
        action_idx: int,
        cursor_x: float,
        cursor_y: float,
        had_effect: bool,
        next_state: np.ndarray,
    ):
        """Add experience to buffer for training."""
        self.buffer_states.append(state.copy())
        self.buffer_actions.append(action_idx)
        # Normalize cursor to [-1, 1]
        norm_x = (cursor_x / (self.grid_size - 1)) * 2 - 1
        norm_y = (cursor_y / (self.grid_size - 1)) * 2 - 1
        self.buffer_cursors.append([norm_x, norm_y])
        self.buffer_had_effect.append(had_effect)
        self.buffer_next_states.append(next_state.copy())
        
        # Trim buffer if needed
        if len(self.buffer_states) > self.buffer_size:
            self.buffer_states = self.buffer_states[-self.buffer_size:]
            self.buffer_actions = self.buffer_actions[-self.buffer_size:]
            self.buffer_cursors = self.buffer_cursors[-self.buffer_size:]
            self.buffer_had_effect = self.buffer_had_effect[-self.buffer_size:]
            self.buffer_next_states = self.buffer_next_states[-self.buffer_size:]
    
    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Train the joint embedding model on a batch of experiences.
        
        Returns training statistics.
        """
        if len(self.buffer_states) < batch_size:
            return {}
        
        self.train()
        
        # Sample batch
        indices = np.random.choice(len(self.buffer_states), batch_size, replace=False)
        
        states = torch.tensor(
            np.stack([self.buffer_states[i] for i in indices]),
            dtype=torch.long, device=self.device
        )
        actions = torch.tensor(
            [self.buffer_actions[i] for i in indices],
            dtype=torch.long, device=self.device
        )
        cursors = torch.tensor(
            [self.buffer_cursors[i] for i in indices],
            dtype=torch.float32, device=self.device
        )
        had_effect = torch.tensor(
            [self.buffer_had_effect[i] for i in indices],
            dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_states = torch.tensor(
            np.stack([self.buffer_next_states[i] for i in indices]),
            dtype=torch.long, device=self.device
        )
        
        # Forward pass
        # 1. Encode visual state (return all tokens for cross-attention)
        visual_feats = self.visual_encoder(states, return_all_tokens=True)  # (B, P+1, D)
        visual_cls = visual_feats[:, 0]  # (B, D) - CLS token
        
        # 2. Get action embeddings
        action_emb = self.action_embed(actions, cursors)  # (B, D)
        
        # 3. Predict effectiveness of the taken action
        effectiveness_logits = self.effectiveness_pred(visual_feats, action_emb)  # (B, 1)
        
        # 4. Predict all action effectiveness from visual CLS
        all_action_logits = self.all_actions_pred(visual_cls)  # (B, num_actions)
        
        # 5. Latent prediction: predict next state embedding
        combined = torch.cat([visual_cls, action_emb], dim=-1)  # (B, 2D)
        pred_next_emb = self.latent_predictor(combined)  # (B, D)
        
        # 6. Target: encode next state with EMA encoder
        with torch.no_grad():
            target_next_emb = self.target_encoder(next_states, return_all_tokens=False)  # (B, D)
        
        # === Losses ===
        # Loss 1: Effectiveness prediction (binary cross-entropy)
        effectiveness_loss = F.binary_cross_entropy_with_logits(
            effectiveness_logits, had_effect
        )
        
        # Loss 2: All-action effectiveness (multi-label)
        # Create target: 1.0 for the taken action if it had effect, else 0.0
        all_action_targets = torch.zeros_like(all_action_logits)
        for i, (act_idx, eff) in enumerate(zip(actions, had_effect)):
            if eff > 0.5:
                all_action_targets[i, act_idx] = 1.0
        all_action_loss = F.binary_cross_entropy_with_logits(
            all_action_logits, all_action_targets
        )
        
        # Loss 3: Latent prediction (cosine similarity with EMA target)
        # Only for actions that had effect (those that changed state)
        mask = had_effect.squeeze(1) > 0.5
        if mask.sum() > 0:
            pred_norm = F.normalize(pred_next_emb[mask], dim=-1)
            target_norm = F.normalize(target_next_emb[mask], dim=-1)
            latent_loss = (1 - (pred_norm * target_norm).sum(dim=-1)).mean()
        else:
            latent_loss = torch.tensor(0.0, device=self.device)
        
        # Loss 4: Grid reconstruction (for visualization decoder training)
        decoded_logits = self.grid_decoder(visual_cls)  # (B, grid_size * grid_size * 10)
        decoded_logits = decoded_logits.view(states.shape[0], self.decoder_grid_size, self.decoder_grid_size, 10)
        decoded_logits = decoded_logits.permute(0, 3, 1, 2)  # (B, 10, H, W)
        # Clamp grid values to valid class range (0-9), treating cursor markers as background
        states_clamped = states.long().clamp(0, 9)
        reconstruction_loss = F.cross_entropy(decoded_logits, states_clamped)
        
        # Total loss
        total_loss = effectiveness_loss + 0.5 * all_action_loss + 0.5 * latent_loss + 0.1 * reconstruction_loss
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        # Update EMA target
        self.update_target_encoder()
        
        # Compute accuracy
        with torch.no_grad():
            pred_effective = (torch.sigmoid(effectiveness_logits) > 0.5).float()
            accuracy = (pred_effective == had_effect).float().mean().item()
        
        self.train_steps += 1
        self.effective_acc = 0.9 * self.effective_acc + 0.1 * accuracy
        self.pred_loss = 0.9 * self.pred_loss + 0.1 * total_loss.item()
        
        return {
            "effectiveness_loss": effectiveness_loss.item(),
            "all_action_loss": all_action_loss.item(),
            "latent_loss": latent_loss.item() if isinstance(latent_loss, torch.Tensor) else latent_loss,
            "reconstruction_loss": reconstruction_loss.item(),
            "total_loss": total_loss.item(),
            "accuracy": accuracy,
            "ema_acc": self.effective_acc,
            "buffer_size": len(self.buffer_states),
        }
    
    @torch.no_grad()
    def predict_action_effectiveness(
        self,
        state: np.ndarray,
        cursor_x: float,
        cursor_y: float,
    ) -> np.ndarray:
        """
        Predict effectiveness of all actions for the current state.
        
        Returns:
            (num_discrete_actions,) array of effectiveness probabilities
        """
        self.eval()
        
        state_t = torch.tensor(state, dtype=torch.long, device=self.device).unsqueeze(0)
        visual_cls = self.visual_encoder(state_t, return_all_tokens=False)  # (1, D)
        
        all_action_logits = self.all_actions_pred(visual_cls)  # (1, num_actions)
        probs = torch.sigmoid(all_action_logits).squeeze(0).cpu().numpy()
        
        return probs
    
    @torch.no_grad()
    def get_action_mask(
        self,
        state: np.ndarray,
        cursor_x: float,
        cursor_y: float,
        threshold: float = 0.3,
    ) -> np.ndarray:
        """
        Get a mask of actions likely to be effective.
        
        Returns:
            (num_discrete_actions,) boolean array
        """
        probs = self.predict_action_effectiveness(state, cursor_x, cursor_y)
        return probs > threshold
    
    @torch.no_grad()
    def decode_grid(self, embedding: np.ndarray) -> np.ndarray:
        """
        Decode a latent embedding back to a grid.
        
        Args:
            embedding: (embed_dim,) latent embedding
            
        Returns:
            (grid_size, grid_size) decoded grid
        """
        self.eval()
        
        emb_t = torch.tensor(embedding, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.grid_decoder(emb_t)  # (1, grid_size * grid_size * 10)
        logits = logits.view(1, self.decoder_grid_size, self.decoder_grid_size, 10)
        
        # Get argmax for each cell
        grid = logits.argmax(dim=-1).squeeze(0).cpu().numpy().astype(np.uint8)
        return grid
    
    @torch.no_grad()
    def encode_and_decode(self, grid: np.ndarray) -> np.ndarray:
        """
        Encode a grid to latent and decode it back (for testing reconstruction).
        
        Args:
            grid: (grid_size, grid_size) input grid
            
        Returns:
            (grid_size, grid_size) reconstructed grid
        """
        self.eval()
        
        grid_t = torch.tensor(grid, dtype=torch.long, device=self.device).unsqueeze(0)
        emb = self.visual_encoder(grid_t, return_all_tokens=False)  # (1, embed_dim)
        
        logits = self.grid_decoder(emb)  # (1, grid_size * grid_size * 10)
        logits = logits.view(1, self.decoder_grid_size, self.decoder_grid_size, 10)
        
        reconstructed = logits.argmax(dim=-1).squeeze(0).cpu().numpy().astype(np.uint8)
        return reconstructed
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            "visual_encoder": self.visual_encoder.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "action_embed": self.action_embed.state_dict(),
            "effectiveness_pred": self.effectiveness_pred.state_dict(),
            "latent_predictor": self.latent_predictor.state_dict(),
            "all_actions_pred": self.all_actions_pred.state_dict(),
            "grid_decoder": self.grid_decoder.state_dict(),
            "train_steps": self.train_steps,
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.visual_encoder.load_state_dict(checkpoint["visual_encoder"])
        self.target_encoder.load_state_dict(checkpoint["target_encoder"])
        self.action_embed.load_state_dict(checkpoint["action_embed"])
        self.effectiveness_pred.load_state_dict(checkpoint["effectiveness_pred"])
        self.latent_predictor.load_state_dict(checkpoint["latent_predictor"])
        self.all_actions_pred.load_state_dict(checkpoint["all_actions_pred"])
        self.train_steps = checkpoint.get("train_steps", 0)
