import numpy as np
from typing import Tuple

class PhysicsEngine:
    """
    Handles cursor physics.
    Remastered for Direct Velocity Control (Stability & Continuous Usability).
    Removes 'ice physics' (inertia/friction) in favor of precise response.
    """
    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        self.vel_x = 0.0
        self.vel_y = 0.0
        
        # Physics constants
        self.max_velocity = 5.0  # Increased for faster target acquisition
        self.deadzone = 0.05     # Slightly larger deadzone to eliminate micro-jitter

    def reset(self):
        self.vel_x = 0.0
        self.vel_y = 0.0

    def update(self, input_x: float, input_y: float, current_x: float, current_y: float) -> Tuple[float, float, bool]:
        """
        Updates position based on direct velocity inputs.
        input_x, input_y: [-1.0, 1.0] from Agent
        Returns new (x, y) coordinates and hit_wall boolean.
        """
        # Apply Deadzone
        if abs(input_x) < self.deadzone: input_x = 0.0
        if abs(input_y) < self.deadzone: input_y = 0.0
        
        # Direct Velocity Mapping (No Acceleration/Momentum)
        # This gives the agent "FPS-like" control.
        self.vel_x = input_x * self.max_velocity
        self.vel_y = input_y * self.max_velocity
        
        # Update position
        new_x = current_x + self.vel_x
        new_y = current_y + self.vel_y

        # Hard Constraint (Wall Bounce/Stop)
        # We perform a "hard stop" which is robust.
        # But we ensure velocity is zeroed so it doesn't "build up" virtual speed.
        
        hit_wall = False
        
        if new_x < 0:
            new_x = 0
            self.vel_x = 0
            hit_wall = True
        elif new_x > self.grid_size - 1:
            new_x = self.grid_size - 1
            self.vel_x = 0
            hit_wall = True

        if new_y < 0:
            new_y = 0
            self.vel_y = 0
            hit_wall = True
        elif new_y > self.grid_size - 1:
            new_y = self.grid_size - 1
            self.vel_y = 0
            hit_wall = True

        return new_x, new_y, hit_wall

    def get_velocity(self) -> Tuple[float, float]:
        return self.vel_x, self.vel_y
    
    def get_speed(self) -> float:
        return np.sqrt(self.vel_x**2 + self.vel_y**2)
