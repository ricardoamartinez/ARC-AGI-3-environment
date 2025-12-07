import json
import logging
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger()

class LiveVisualizerCallback(BaseCallback):
    """
    Callback for live visualization of the training process.
    """
    def __init__(self, gui_process, agent, verbose=0):
        super().__init__(verbose)
        self.gui_process = gui_process
        self.agent = agent
        self._quit_event = None  # Will be set by agent

    def _on_step(self) -> bool:
        # Check for quit signal
        if self._quit_event and self._quit_event.is_set():
            logger.info("Quit signal detected in callback, stopping training...")
            return False  # Return False to stop training
        
        if self.gui_process and self.gui_process.poll() is None:
            try:
                latest_frame = self.agent.frames[-1] if self.agent.frames else None
                last_action = getattr(self.agent, "_last_action_viz", None)
                
                if latest_frame and latest_frame.frame:
                    msg = {
                        "grids": latest_frame.frame,
                        "game_id": self.agent.game_id,
                        "score": latest_frame.score,
                        "state": f"Step: {self.num_timesteps}",
                        "last_action": last_action,
                        "cursor": {"x": self.agent.cursor_x, "y": self.agent.cursor_y}
                    }
                    self.gui_process.stdin.write(json.dumps(msg) + "\n")
                    self.gui_process.stdin.flush()
            except Exception:
                pass # Ignore GUI errors to not stop training
        return True

