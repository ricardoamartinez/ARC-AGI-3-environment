import numpy as np
from typing import List, Tuple
from scipy.ndimage import label

def find_connected_components(grid: np.ndarray) -> List[List[Tuple[int, int]]]:
    """
    Fast connected components using scipy.ndimage.label.
    Returns a list of objects, where each object is a list of (r, c) coordinates.
    Background (0) is ignored.
    Objects must be of the same color to be connected.
    """
    rows, cols = grid.shape
    objects = []
    
    # Scipy label finds connected components of non-zero elements.
    # However, ARC objects are defined by COLOR + connectivity.
    # A grid might have a blue block touching a red block.
    # scipy.ndimage.label treats all non-zero as "structure" usually, unless we separate by color.
    
    # Fast path: Iterate unique colors
    unique_colors = np.unique(grid)
    # Skip background 0
    unique_colors = unique_colors[unique_colors != 0]
    
    structure = np.array([[0,1,0], [1,1,1], [0,1,0]]) # 4-connectivity
    
    for color in unique_colors:
        # Create binary mask for this color
        mask = (grid == color)
        # Label components
        labeled_array, num_features = label(mask, structure=structure)
        
        if num_features == 0:
            continue
            
        # Extract coordinates
        # This can be vectorized?
        # Find indices for each label
        
        # Optimization: ndimage.find_objects might be faster but returns slices.
        # We need pixel lists.
        
        # Fast approach: Flatten and group? 
        # Or just iterate 1..num_features
        
        # To avoid Python loops over pixels, we can use np.argwhere per label? 
        # But looping 'num_features' times is better than looping pixels.
        
        for i in range(1, num_features + 1):
            # Get boolean mask for this object
            # obj_mask = (labeled_array == i)
            # coords = np.argwhere(obj_mask)
            # objects.append([tuple(p) for p in coords])
            
            # Even faster:
            coords = np.argwhere(labeled_array == i)
            # Convert to list of tuples (int, int)
            # We need standard python ints for compatibility with existing code serialization
            objects.append(coords.tolist())

    return objects

