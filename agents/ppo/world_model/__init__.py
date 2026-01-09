"""
V-JEPA style World Model for ARC-AGI-3.

Components:
- GridEncoder: Encodes game grids to latent embeddings
- ActionConditionedPredictor: Predicts next latent given current + (continuous_action, discrete_token)
- WinPredictor: Predicts probability of winning from latent (optional; sparse labels)
- CEMPlanner: Cross-Entropy Method for action sequence search
"""

from .encoder import GridEncoder
from .predictor import ActionConditionedPredictor
from .win_predictor import WinPredictor
from .planner import CEMPlanner
from .world_model import WorldModel

__all__ = [
    "GridEncoder",
    "ActionConditionedPredictor",
    "WinPredictor",
    "CEMPlanner",
    "WorldModel",
]
