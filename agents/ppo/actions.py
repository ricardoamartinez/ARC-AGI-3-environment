import numpy as np
from typing import Optional, Tuple
from ..structs import GameAction

class ActionProcessor:
    """
    Handles processing of raw action tensors into game actions and discrete selections.
    Manages cooldowns and anti-spam logic.
    """
    def __init__(self):
        self.consecutive_action_steps = 0
        self.last_trigger_val = 0.0
        self.idle_streak = 0
        self.action_cooldown = 0
        self.cooldown_steps_after_action = 15 # Balanced cooldown
        self.act_threshold = 0.7 # Higher threshold to reduce accidental clicks
        self.is_holding = False # Explicit holding state

    def reset(self):
        self.consecutive_action_steps = 0
        self.last_trigger_val = 0.0
        self.idle_streak = 0
        self.action_cooldown = 0
        self.is_holding = False

    def process(self, action_tensor: np.ndarray, current_speed: float) -> Tuple[int, float, float, float, float]:
        """
        Parses action tensor.
        Returns: (final_action_idx, ax, ay, trigger, selection)
        """
        if self.action_cooldown > 0:
            self.action_cooldown -= 1

        ax = float(action_tensor[0])
        ay = float(action_tensor[1])
        trigger = float(action_tensor[2])
        selection = float(action_tensor[3])

        # Map selection to discrete buckets [0-9]
        norm_selection = max(0.0, min(1.0, (selection + 1.0) / 2.0))
        action_idx = int(norm_selection * 10.0)
        action_idx = min(9, max(0, action_idx))
        
        # Trigger Logic
        is_trigger_active = trigger > self.act_threshold
        should_act = False

        if is_trigger_active:
            # Only act on rising edge AND if cooldown is 0
            if not self.is_holding and self.action_cooldown == 0:
                # Velocity Constraint
                if current_speed <= 2.0:
                    should_act = True
            
            self.is_holding = True
        else:
            self.is_holding = False
        
        self.last_trigger_val = trigger

        final_action_idx = action_idx if should_act else -1
        
        # Update counters
        if final_action_idx != -1:
            self.consecutive_action_steps += 1
            self.idle_streak = 0
            self.action_cooldown = self.cooldown_steps_after_action
        else:
            if self.action_cooldown == 0:
                self.consecutive_action_steps = 0
            self.idle_streak += 1

        return final_action_idx, ax, ay, trigger, selection

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

