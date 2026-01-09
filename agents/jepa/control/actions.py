from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from ...structs import GameAction


class ActionProcessor:
    """
    Converts raw policy outputs into game actions.

    Note: This project uses a hybrid action space:
    - continuous_actions: [ax, ay, trigger]
    - discrete_action_idx: int [0-9]
    """

    def __init__(self):
        # Minimal mapping: trigger > threshold enables the discrete action index
        # Default 0.5 so there's a 50% chance of action firing when trigger is random.
        # The policy learns to increase/decrease trigger based on rewards.
        self.act_threshold = float(os.environ.get("JEPA_TRIGGER_THRESHOLD", "0.5"))

    def reset(self):
        pass

    def process(
        self,
        continuous_actions: np.ndarray,
        discrete_action_idx: int,
        current_speed: float,
    ) -> Tuple[int, float, float, float]:
        raw_ax = float(continuous_actions[0])
        raw_ay = float(continuous_actions[1])
        trigger = float(continuous_actions[2])
        ax = raw_ax
        ay = raw_ay
        final_action_idx = discrete_action_idx if trigger > self.act_threshold else -1
        return final_action_idx, ax, ay, trigger

    def get_game_action(
        self, final_action_idx: int, cursor_x: int, cursor_y: int, game_id: str
    ) -> Optional[Tuple[GameAction, dict]]:
        """
        Returns a tuple of (GameAction, action_data) to avoid race conditions
        with the mutable state on GameAction enum members.
        
        Returns None if no action should be taken.
        """
        if final_action_idx == -1:
            return None

        click_action = final_action_idx <= 3

        if click_action:
            return (GameAction.ACTION6, {"x": cursor_x, "y": cursor_y, "game_id": game_id})
        elif final_action_idx == 4:
            return (GameAction.ACTION1, {"game_id": game_id})  # UP
        elif final_action_idx == 5:
            return (GameAction.ACTION2, {"game_id": game_id})  # DOWN
        elif final_action_idx == 6:
            return (GameAction.ACTION3, {"game_id": game_id})  # LEFT
        elif final_action_idx == 7:
            return (GameAction.ACTION4, {"game_id": game_id})  # RIGHT
        elif final_action_idx == 8:
            return (GameAction.ACTION5, {"game_id": game_id})  # SPACE
        elif final_action_idx == 9:
            return (GameAction.ACTION7, {"game_id": game_id})  # ENTER

        return None


