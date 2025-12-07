import numpy as np
from typing import List, Tuple

def find_connected_components(grid: np.ndarray) -> List[List[Tuple[int, int]]]:
    """
    Simple BFS to find connected components (objects) in the grid.
    Returns a list of objects, where each object is a list of (r, c) coordinates.
    Background (0) is ignored.
    """
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    objects = []
    
    def get_neighbors(r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0 and not visited[r, c]:
                # Start new component
                component = []
                color = grid[r, c]
                queue = [(r, c)]
                visited[r, c] = True
                
                while queue:
                    curr_r, curr_c = queue.pop(0)
                    component.append((curr_r, curr_c))
                    
                    for nr, nc in get_neighbors(curr_r, curr_c):
                        if not visited[nr, nc] and grid[nr, nc] == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                objects.append(component)
    return objects

