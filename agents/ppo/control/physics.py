from __future__ import annotations

import os
from typing import Tuple

import numpy as np


class PhysicsEngine:
    """
    Cursor motion model.

    Current design: direct velocity control (no inertia), with deadzone + hard wall clamp.
    """

    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        self.vel_x = 0.0
        self.vel_y = 0.0
        # Smooth, inertial cursor dynamics (for non-jumpy trajectories).
        # Policy outputs are treated as desired velocity in [-1,1].
        # Increased defaults for more responsive cursor movement.
        self.max_speed = float(os.environ.get("PPO_MAX_SPEED", "6.0"))  # cells/step - higher for faster movement
        # Blend factor for reaching desired velocity (0..1). Higher = more responsive.
        self.inertia_alpha = float(os.environ.get("PPO_INERTIA_ALPHA", "0.5"))
        self.inertia_alpha = max(0.0, min(1.0, self.inertia_alpha))
        # Additional damping each step (0..1). Lower = less friction.
        self.damping = float(os.environ.get("PPO_DAMPING", "0.02"))
        self.damping = max(0.0, min(1.0, self.damping))

        # Spring-damper (PD) control gains for target-position mode
        # Higher kp = faster acceleration toward target
        self.kp = float(os.environ.get("PPO_PD_KP", "0.6"))
        self.kd = float(os.environ.get("PPO_PD_KD", "0.2"))
        self.kp = max(0.0, self.kp)
        self.kd = max(0.0, self.kd)

        # Stop deadband to eliminate jitter: if we're close enough and already slow,
        # we "stick" and zero velocity (static friction).
        # Reduced stop_radius so cursor moves more freely
        self.stop_radius = float(os.environ.get("PPO_STOP_RADIUS", "0.3"))  # cells
        self.stop_speed = float(os.environ.get("PPO_STOP_SPEED", "0.1"))  # cells/step
        self.stop_radius = max(0.0, self.stop_radius)
        self.stop_speed = max(0.0, self.stop_speed)

        # Allow speed_scale > 1.0 to move faster when far, while still allowing the agent to brake near the goal.
        self.speed_scale_max = float(os.environ.get("PPO_SPEED_SCALE_MAX", "2.0"))
        self.speed_scale_max = max(0.0, self.speed_scale_max)

    def reset(self):
        self.vel_x = 0.0
        self.vel_y = 0.0

    def update(
        self, input_x: float, input_y: float, current_x: float, current_y: float
    ) -> Tuple[float, float, bool]:
        desired_vx = float(input_x) * self.max_speed
        desired_vy = float(input_y) * self.max_speed

        # Inertial smoothing toward desired velocity
        a = self.inertia_alpha
        self.vel_x = (1.0 - a) * self.vel_x + a * desired_vx
        self.vel_y = (1.0 - a) * self.vel_y + a * desired_vy

        # Damping / friction
        self.vel_x *= (1.0 - self.damping)
        self.vel_y *= (1.0 - self.damping)

        new_x = current_x + self.vel_x
        new_y = current_y + self.vel_y

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

    def update_to_target(
        self,
        target_x: float,
        target_y: float,
        current_x: float,
        current_y: float,
        speed_scale: float = 1.0,
    ) -> Tuple[float, float, bool]:
        """
        Smooth second-order cursor dynamics:
        - policy chooses a target position (continuous)
        - we apply spring-damper to move toward it and naturally come to rest
        """
        # If we're basically at the target and already slow, stop completely (prevents jitter).
        ex = float(target_x) - float(current_x)
        ey = float(target_y) - float(current_y)
        err = float(np.sqrt(ex * ex + ey * ey))
        speed0 = float(np.sqrt(self.vel_x**2 + self.vel_y**2))
        if err <= self.stop_radius and speed0 <= self.stop_speed:
            self.vel_x = 0.0
            self.vel_y = 0.0
            # Snap to target (still clamped by walls below)
            new_x = float(target_x)
            new_y = float(target_y)
            hit_wall = False
            if new_x < 0:
                new_x = 0
                hit_wall = True
            elif new_x > self.grid_size - 1:
                new_x = self.grid_size - 1
                hit_wall = True
            if new_y < 0:
                new_y = 0
                hit_wall = True
            elif new_y > self.grid_size - 1:
                new_y = self.grid_size - 1
                hit_wall = True
            return new_x, new_y, hit_wall

        # PD acceleration
        ax = self.kp * ex - self.kd * self.vel_x
        ay = self.kp * ey - self.kd * self.vel_y

        # Integrate velocity, clamp max speed (optionally scaled by policy)
        self.vel_x += ax
        self.vel_y += ay

        speed = float(np.sqrt(self.vel_x**2 + self.vel_y**2))
        smax = float(self.max_speed) * float(np.clip(speed_scale, 0.0, self.speed_scale_max))
        if speed > smax and speed > 1e-6:
            s = smax / speed
            self.vel_x *= s
            self.vel_y *= s

        # Additional damping
        self.vel_x *= (1.0 - self.damping)
        self.vel_y *= (1.0 - self.damping)

        new_x = current_x + self.vel_x
        new_y = current_y + self.vel_y

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


