import hashlib
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from .utils import find_connected_components

class ObjectTracker:
    """
    Manages object detection, hashing, and value tracking.
    """
    def __init__(self):
        self.detected_objects: List[List[Tuple[int, int]]] = []
        self.valuable_object_hashes: Dict[str, float] = {}
        self.episode_object_hashes: List[List[str]] = []
        self.interacted_objects: Set[str] = set()

    def reset(self):
        self.detected_objects = []
        self.episode_object_hashes = []
        self.interacted_objects.clear()
        # valuable_object_hashes persists across episodes (Lifetime learning)

    def scan(self, grid: np.ndarray):
        """Scans grid for connected components."""
        self.detected_objects = find_connected_components(grid)

    def get_object_hash_at(self, r: int, c: int, grid: np.ndarray) -> Optional[str]:
        """Finds object at (r, c) and returns its hash."""
        for obj_pixels in self.detected_objects:
            if (r, c) in obj_pixels:
                sorted_pixels = sorted(obj_pixels)
                # Hash based on sorted pixels (Shape + Position)
                # Note: This is an Instance Hash.
                return hashlib.md5(str(sorted_pixels).encode()).hexdigest()
        return None

    def get_object_hashes(self, grid: np.ndarray) -> List[str]:
        """Returns hashes for all current objects (Color + Shape/Pos)."""
        hashes = []
        for obj_pixels in self.detected_objects:
            if not obj_pixels: continue
            r, c = obj_pixels[0]
            color = grid[r, c]
            sorted_pixels = sorted(obj_pixels)
            h = hashlib.md5(f"{color}_{str(sorted_pixels)}".encode()).hexdigest()
            hashes.append(h)
        return hashes
    
    def get_invariant_hash(self, obj_pixels: List[Tuple[int, int]], color: int) -> str:
        """Returns translation-invariant hash (Color + Shape)."""
        min_r = min(p[0] for p in obj_pixels)
        min_c = min(p[1] for p in obj_pixels)
        norm_pixels = sorted([(p[0]-min_r, p[1]-min_c) for p in obj_pixels])
        return hashlib.md5(f"{color}_{str(norm_pixels)}".encode()).hexdigest()

