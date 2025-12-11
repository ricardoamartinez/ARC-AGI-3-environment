import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class Hypothesis:
    def __init__(self, name: str):
        self.name = name
        self.confidence = 0.5
        self.successes = 0
        self.failures = 0

    def update(self, feedback: float, context: Dict):
        """
        Update confidence based on feedback (-1.0 to 1.0) and context.
        Context contains: grid, x, y, object_color, etc.
        """
        match = self.matches(context)
        if match:
            # Prediction matched context
            if feedback > 0:
                self.successes += 1
                self.confidence += 0.1 * feedback
            elif feedback < 0:
                self.failures += 1
                self.confidence += 0.1 * feedback # Reduce confidence
        
        self.confidence = max(0.01, min(0.99, self.confidence))

    def matches(self, context: Dict) -> bool:
        return False

    def generate_map(self, grid: np.ndarray) -> np.ndarray:
        return np.zeros_like(grid, dtype=np.float32)

class ColorHypothesis(Hypothesis):
    def __init__(self, color: int):
        super().__init__(f"Target Color {color}")
        self.color = color

    def matches(self, context: Dict) -> bool:
        # Check if the interaction was with this color
        return context.get("color") == self.color

    def generate_map(self, grid: np.ndarray) -> np.ndarray:
        # Return map where pixels == color are 1.0
        return (grid == self.color).astype(np.float32)

class RegionHypothesis(Hypothesis):
    def __init__(self, region_name: str, x_range: Tuple[float, float], y_range: Tuple[float, float]):
        super().__init__(f"Target Region {region_name}")
        self.x_range = x_range
        self.y_range = y_range

    def matches(self, context: Dict) -> bool:
        # Context x, y are normalized 0-1 (if passed correctly) or integers
        # Let's assume context passes both or we normalize
        h, w = context.get("grid_shape", (64, 64))
        nx = context["x"] / w
        ny = context["y"] / h
        return (self.x_range[0] <= nx <= self.x_range[1] and 
                self.y_range[0] <= ny <= self.y_range[1])

    def generate_map(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        map_ = np.zeros((h, w), dtype=np.float32)
        
        x0 = int(self.x_range[0] * w)
        x1 = int(self.x_range[1] * w)
        y0 = int(self.y_range[0] * h)
        y1 = int(self.y_range[1] * h)
        
        map_[y0:y1, x0:x1] = 1.0
        return map_

class HypothesisEngine:
    def __init__(self):
        self.hypotheses: List[Hypothesis] = []
        self._init_hypotheses()
        self.current_hypothesis: Optional[Hypothesis] = None
        
    def _init_hypotheses(self):
        # Colors 0-9
        for c in range(10):
            self.hypotheses.append(ColorHypothesis(c))
            
        # Regions
        self.hypotheses.append(RegionHypothesis("Top", (0.0, 1.0), (0.0, 0.5)))
        self.hypotheses.append(RegionHypothesis("Bottom", (0.0, 1.0), (0.5, 1.0)))
        self.hypotheses.append(RegionHypothesis("Left", (0.0, 0.5), (0.0, 1.0)))
        self.hypotheses.append(RegionHypothesis("Right", (0.5, 1.0), (0.0, 1.0)))
        self.hypotheses.append(RegionHypothesis("Center", (0.25, 0.75), (0.25, 0.75)))
        
    def update(self, feedback: float, context: Dict):
        """
        Feedback: +1.0 (Dopamine) or -1.0 (Pain)
        """
        for h in self.hypotheses:
            h.update(feedback, context)
            
        # Sort by confidence
        self.hypotheses.sort(key=lambda x: x.confidence, reverse=True)
        
        # Pick top
        if self.hypotheses[0].confidence > 0.6:
            if self.current_hypothesis != self.hypotheses[0]:
                self.current_hypothesis = self.hypotheses[0]
                logger.info(f"New Best Hypothesis: {self.current_hypothesis.name} ({self.current_hypothesis.confidence:.2f})")

    def get_reward_map(self, grid: np.ndarray) -> Optional[np.ndarray]:
        if self.current_hypothesis:
            return self.current_hypothesis.generate_map(grid)
        return None

