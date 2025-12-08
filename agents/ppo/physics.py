import numpy as np
from typing import Tuple

class PhysicsEngine:
    """
    Handles cursor physics: acceleration, velocity, friction, and boundary collisions.
    """
    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        self.vel_x = 0.0
        self.vel_y = 0.0
        
        # Physics constants
        self.acceleration = 0.2  # Force multiplier
        self.friction = 0.94     # Velocity retention per step
        self.max_velocity = 2.0  # Max cursor speed pixels/step

    def reset(self):
        self.vel_x = 0.0
        self.vel_y = 0.0

    def update(self, ax: float, ay: float, current_x: float, current_y: float) -> Tuple[float, float]:
        """
        Updates velocity and position based on acceleration inputs.
        Returns new (x, y) coordinates.
        """
        # Deadzone
        if abs(ax) < 0.15: ax = 0.0
        if abs(ay) < 0.15: ay = 0.0

        # Apply acceleration
        self.vel_x += ax * self.acceleration
        self.vel_y += ay * self.acceleration

        # Apply friction
        self.vel_x *= self.friction
        self.vel_y *= self.friction

        # Clamp velocity
        self.vel_x = max(-self.max_velocity, min(self.max_velocity, self.vel_x))
        self.vel_y = max(-self.max_velocity, min(self.max_velocity, self.vel_y))

        # Update position
        new_x = current_x + self.vel_x
        new_y = current_y + self.vel_y

        # Bounce on walls
        if new_x < 0:
            new_x = 0
            self.vel_x = -self.vel_x * 0.5
        elif new_x > self.grid_size - 1:
            new_x = self.grid_size - 1
            self.vel_x = -self.vel_x * 0.5

        if new_y < 0:
            new_y = 0
            self.vel_y = -self.vel_y * 0.5
        elif new_y > self.grid_size - 1:
            new_y = self.grid_size - 1
            self.vel_y = -self.vel_y * 0.5

        return new_x, new_y

    def get_velocity(self) -> Tuple[float, float]:
        return self.vel_x, self.vel_y
    
    def get_speed(self) -> float:
        return np.sqrt(self.vel_x**2 + self.vel_y**2)

