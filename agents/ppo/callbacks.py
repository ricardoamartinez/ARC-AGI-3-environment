import json
import logging
import torch
import numpy as np
import time
import queue
import threading
from scipy.ndimage import gaussian_filter

logger = logging.getLogger()

class LiveVisualizerCallback:
    """
    Callback for live visualization of the training process.
    Refactored for RTAC (No SB3 dependency).
    """
    def __init__(self, gui_process, agent):
        self.gui_process = gui_process
        self.agent = agent
        self._quit_event = None  # Will be set by agent
        self.sleep_time = 0.0 # Delay in seconds
        self.step_count = 0
        
        # Write Queue
        self.msg_queue = queue.Queue(maxsize=100)
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._writer_thread.start()

    def _write_loop(self):
        """Thread that writes JSON messages to the GUI process stdin."""
        while True:
            try:
                msg = self.msg_queue.get()
                if msg is None: break # sentinel
                
                if self.gui_process and self.gui_process.poll() is None:
                    try:
                        self.gui_process.stdin.write(json.dumps(msg) + "\n")
                        self.gui_process.stdin.flush()
                    except (BrokenPipeError, IOError):
                        pass
            except Exception:
                pass

    def on_step(self, info: dict = None) -> bool:
        self.step_count += 1
        
        # Check for quit signal
        if self._quit_event and self._quit_event.is_set():
            logger.info("Quit signal detected in callback, stopping training...")
            return False
        
        # --- SPEED CONTROL ---
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)
        
        if self.gui_process and self.gui_process.poll() is None:
            try:
                # Sync speed
                if hasattr(self.agent, "training_speed"):
                    self.sleep_time = self.agent.training_speed
                
                latest_frame = self.agent.frames[-1] if self.agent.frames else None
                last_action = getattr(self.agent, "_last_action_viz", None)
                
                # Extract Saliency / Heatmap
                heatmap_data = None
                maps_data = {}
                
                # Extract maps from info
                if info and "maps" in info:
                    maps_data = info["maps"]
                
                # Use latest_feature_map from ArcViTFeatureExtractor if available
                # Agent has .model which is OnlineActorCritic
                # .model.features_extractor.latest_feature_map
                
                if self.step_count % 4 == 0 and self.agent.model:
                     try:
                         feat_map = self.agent.model.features_extractor.latest_feature_map
                         if feat_map is not None:
                             # feat_map is (B, 64, 64)
                             # Take first batch
                             saliency = feat_map[0] # (64, 64)
                             
                             # Normalize
                             s_min, s_max = saliency.min(), saliency.max()
                             range_val = s_max - s_min
                             if range_val > 1e-6:
                                 saliency = (saliency - s_min) / range_val
                             else:
                                 # If uniform, check if active or dead
                                 if s_max > 0.1: # Threshold for "Active"
                                     saliency.fill_(1.0)
                                 else:
                                     saliency.fill_(0.0)
                                 
                             # Convert to list
                             heatmap_data = saliency.cpu().numpy().tolist()
                             maps_data["attention"] = heatmap_data
                     except Exception:
                         pass

                # Get Objects & Metrics
                objects = getattr(self.agent, "latest_detected_objects", [])
                dopamine = 0.0
                plan_confidence = 0.0
                manual_dopamine = 0.0
                pain = 0.0
                manual_pain = 0.0
                current_thought = ""
                
                if info:
                    dopamine = float(info.get("dopamine", 0.0))
                    plan_confidence = float(info.get("plan_confidence", 0.0))
                    manual_dopamine = float(info.get("manual_dopamine", 0.0))
                    pain = float(info.get("pain", 0.0))
                    manual_pain = float(info.get("manual_pain", 0.0))
                    current_thought = str(info.get("current_thought", ""))

                if latest_frame and latest_frame.frame:
                    msg = {
                        "grids": latest_frame.frame,
                        "game_id": self.agent.game_id,
                        "score": latest_frame.score,
                        "state": f"Step: {self.step_count}",
                        "last_action": last_action,
                        "cursor": {"x": self.agent.cursor_x, "y": self.agent.cursor_y},
                        # Only update heatmap if computed
                        **({"attention": heatmap_data} if heatmap_data is not None else {}),
                        "maps": maps_data,
                        "objects": objects,
                        "metrics": {
                            "dopamine": dopamine,
                            "plan_confidence": plan_confidence,
                            "reward_mean": 0.0, # Removed buffer dependence
                            "manual_dopamine": manual_dopamine,
                            "pain": pain,
                            "manual_pain": manual_pain,
                            "current_thought": current_thought
                        }
                    }
                    
                    # Non-blocking write via queue
                    if not self.msg_queue.full():
                        self.msg_queue.put(msg)
                        
            except Exception:
                pass 
        return True
