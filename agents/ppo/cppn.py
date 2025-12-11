import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CPPN(nn.Module):
    """
    Compositional Pattern Producing Network (CPPN).
    Maps spatial coordinates (x, y, r) to output patterns.
    Used for generating Intrinsic Motivation (Dopamine) maps.
    """
    def __init__(self, output_dim=1, hidden_dim=32, num_layers=3):
        super().__init__()
        
        # Inputs: x, y, radius (distance from center)
        self.input_dim = 3 
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        
        # Initial layer
        self.layers.append(nn.Linear(self.input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        # Output layer
        self.final_layer = nn.Linear(hidden_dim, output_dim)
        
        # Activation functions available for "composition"
        # We will randomly assign an activation function to each neuron/layer roughly
        # Or simplistic version: Alternate activations per layer or sum them
        # For a true CPPN, each node has its own activation. 
        # Here we approximate by having parallel paths or switching activations.
        
    def forward(self, x, y, r):
        """
        x, y, r: Tensors of shape (Batch, Height, Width) or (Batch, N)
        """
        # Flatten inputs for Linear layers: (Batch*H*W, 1)
        # Assuming inputs are already flattened or shaped correctly
        
        # Stack inputs: (N, 3)
        inp = torch.stack([x, y, r], dim=-1)
        
        out = inp
        
        # Simple "Composition" approximation:
        # Pass through linear weights, then apply a MIX of activations
        for i, layer in enumerate(self.layers):
            out = layer(out)
            
            # Mix activations (Picbreeder style: Sin, Cos, Gaussian, Sigmoid)
            # We split the hidden units into groups and apply different functions
            chunks = torch.chunk(out, 4, dim=-1)
            
            c0 = torch.tanh(chunks[0])         # Tanh (Sigmoid-like)
            c1 = torch.sin(chunks[1] * 5.0)    # Sine (Repetition)
            c2 = torch.exp(-chunks[2]**2)      # Gaussian (Symmetry/Locality)
            c3 = F.relu(chunks[3])             # ReLU (Linearity)
            
            out = torch.cat([c0, c1, c2, c3], dim=-1)
            
        # Final projection
        out = self.final_layer(out)
        return torch.sigmoid(out) # Output 0.0 to 1.0 (Dopamine probability)

    def mutate(self, magnitude=0.1):
        """
        Evolve the network weights slightly.
        """
        with torch.no_grad():
            for param in self.parameters():
                noise = torch.randn_like(param) * magnitude
                param.add_(noise)

def generate_cppn_map(cppn, width=64, height=64, device='cpu'):
    """
    Generates a 2D grid from the CPPN.
    """
    # Create coordinate grid
    xs = torch.linspace(-1, 1, width, device=device)
    ys = torch.linspace(-1, 1, height, device=device)
    
    # Meshgrid
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    
    # Radius (distance from center)
    grid_r = torch.sqrt(grid_x**2 + grid_y**2)
    
    # Flatten
    flat_x = grid_x.reshape(-1)
    flat_y = grid_y.reshape(-1)
    flat_r = grid_r.reshape(-1)
    
    # Inference
    with torch.no_grad():
        output = cppn(flat_x, flat_y, flat_r)
        
    # Reshape back to image
    img = output.reshape(height, width).cpu().numpy()
    return img

