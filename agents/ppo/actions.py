import numpy as np
from typing import Optional, Tuple
from ..structs import GameAction

class ActionProcessor:
    """
    Handles processing of raw action tensors into game actions.
    Now supports hybrid actions (continuous + discrete).
    """
    def __init__(self):
        self.consecutive_action_steps = 0
        self.last_trigger_val = 0.0
        self.idle_streak = 0
        self.act_threshold = 0.0
        self.is_holding = False
        
        # Smoothing (EMA)
        self.smooth_x = 0.0
        self.smooth_y = 0.0
        # Increased alpha for snappier response (was 0.2)
        self.smoothing_alpha = 0.6 

    def reset(self):
        self.consecutive_action_steps = 0
        self.last_trigger_val = 0.0
        self.idle_streak = 0
        self.is_holding = False
        self.smooth_x = 0.0
        self.smooth_y = 0.0

    def process(self, 
                continuous_actions: np.ndarray, 
                discrete_action_idx: int,
                current_speed: float) -> Tuple[int, float, float, float, float]:
        """
        Parses action inputs.
        continuous_actions: [ax, ay, trigger]
        discrete_action_idx: int [0-9]
        """
        raw_ax = float(continuous_actions[0])
        raw_ay = float(continuous_actions[1])
        trigger = float(continuous_actions[2])
        
        # Apply EMA Smoothing
        self.smooth_x = self.smoothing_alpha * raw_ax + (1 - self.smoothing_alpha) * self.smooth_x
        self.smooth_y = self.smoothing_alpha * raw_ay + (1 - self.smoothing_alpha) * self.smooth_y
        
        ax = self.smooth_x
        ay = self.smooth_y
        
        # Trigger Logic
        is_trigger_active = trigger > self.act_threshold
        should_act = False

        if is_trigger_active:
            # Semi-Automatic: Only act on rising edge (press) or if holding appropriate tool?
            # For now, stick to press logic to avoid machine-gunning.
            if not self.is_holding:
                should_act = True
            self.is_holding = True
        else:
            self.is_holding = False
        
        self.last_trigger_val = trigger

        final_action_idx = discrete_action_idx if should_act else -1
        
        # Update counters
        if final_action_idx != -1:
            self.consecutive_action_steps += 1
            self.idle_streak = 0
        else:
            self.consecutive_action_steps = 0
            self.idle_streak += 1

        # Return dummy selection (0.0) for backwards compat if needed
        return final_action_idx, ax, ay, trigger, 0.0

    def get_game_action(self, final_action_idx: int, cursor_x: int, cursor_y: int, game_id: str) -> Optional[GameAction]:
        if final_action_idx == -1:
            return None
            
        click_action = (final_action_idx <= 3)
        
        if click_action:
            game_action = GameAction.ACTION6
            game_action.set_data({"x": cursor_x, "y": cursor_y, "game_id": game_id})
            return game_action
        elif final_action_idx == 4: return GameAction.ACTION1 # UP
        elif final_action_idx == 5: return GameAction.ACTION2 # DOWN
        elif final_action_idx == 6: return GameAction.ACTION3 # LEFT
        elif final_action_idx == 7: return GameAction.ACTION4 # RIGHT
        elif final_action_idx == 8: return GameAction.ACTION5 # SPACE
        elif final_action_idx == 9: return GameAction.ACTION7 # ENTER
        
        return None
