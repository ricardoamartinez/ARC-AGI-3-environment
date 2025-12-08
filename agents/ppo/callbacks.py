import json
import logging
import torch
import numpy as np
import time
import queue
import threading
from stable_baselines3.common.callbacks import BaseCallback
from scipy.ndimage import gaussian_filter

logger = logging.getLogger()

class LiveVisualizerCallback(BaseCallback):
    """
    Callback for live visualization of the training process.
    Uses a separate thread to write to the GUI process to avoid blocking training.
    """
    def __init__(self, gui_process, agent, verbose=0):
        super().__init__(verbose)
        self.gui_process = gui_process
        self.agent = agent
        self._quit_event = None  # Will be set by agent
        self.sleep_time = 0.0 # Delay in seconds
        
        # Write Queue
        self.msg_queue = queue.Queue(maxsize=2) # Keep it small to drop old frames
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

    def _on_step(self) -> bool:
        # Check for quit signal
        if self._quit_event and self._quit_event.is_set():
            logger.info("Quit signal detected in callback, stopping training...")
            return False  # Return False to stop training
        
        # --- SPEED CONTROL ---
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)
        
        # --- OPTIMIZATION REMOVED: Smoothness Priority ---
        # We update GUI every step for smooth cursor visualization.
        # The CNN model is fast enough to handle this.
        # if self.num_timesteps % 4 != 0 and self.sleep_time == 0:
        #    return True

        if self.gui_process and self.gui_process.poll() is None:
            try:
                # Sync speed
                if hasattr(self.agent, "training_speed"):
                    self.sleep_time = self.agent.training_speed
                
                latest_frame = self.agent.frames[-1] if self.agent.frames else None
                last_action = getattr(self.agent, "_last_action_viz", None)
                
                # Extract Saliency Map (Gradient of Choice w.r.t Input)
                heatmap_data = None
                
                # OPTIMIZATION: Only compute heavy Saliency Map every 4 steps to keep framerate high for smooth cursor
                # Only enable if on GPU to avoid CPU lag
                is_gpu = self.model.device.type == "cuda"
                if is_gpu and self.num_timesteps % 4 == 0:
                    try:
                        # Get current observation from locals
                        obs = self.locals.get("new_obs")
                        
                        if obs is not None and self.model:
                            # Convert to tensor
                            device = self.model.device
                            obs_tensor = torch.as_tensor(obs).to(device)
                            if obs_tensor.dtype != torch.float32:
                                obs_tensor = obs_tensor.float()
                                
                            # Enable Gradients
                            obs_tensor.requires_grad = True
                            
                            # Forward Pass
                            dist = self.model.policy.get_distribution(obs_tensor)
                            
                            if hasattr(dist, 'distribution') and hasattr(dist.distribution, 'logits'):
                                 logits = dist.distribution.logits
                                 chosen_idx = logits[0].argmax()
                                 score = logits[0, chosen_idx]
                                 
                                 self.model.policy.zero_grad()
                                 score.backward()
                                 
                                 if obs_tensor.grad is not None:
                                     grads = obs_tensor.grad[0]
                                     saliency, _ = grads.abs().max(dim=0)
                                     
                                     # Normalize
                                     s_min, s_max = saliency.min(), saliency.max()
                                     if s_max > s_min:
                                         saliency = (saliency - s_min) / (s_max - s_min)
                                     
                                     # Smooth
                                     sal_np = saliency.detach().cpu().numpy()
                                     sal_np = gaussian_filter(sal_np, sigma=1.0)
                                     
                                     # Re-normalize strictly 0-1 for GUI visibility
                                     s_min, s_max = sal_np.min(), sal_np.max()
                                     if s_max > s_min:
                                         sal_np = (sal_np - s_min) / (s_max - s_min)
                                     else:
                                         sal_np[:] = 0.0
                                         
                                     heatmap_data = sal_np.tolist()
                                     
                                     # If Saliency is too weak (model ignoring continuous inputs), fallback to Attention
                                     if s_max < 1e-6:
                                         heatmap_data = None # Trigger fallback

                            # FALLBACK: Attention Weights (Reliable visualization for discrete inputs)
                            if heatmap_data is None:
                                 if hasattr(self.model.policy.features_extractor, 'latest_attention_weights'):
                                     w = self.model.policy.features_extractor.latest_attention_weights
                                     if w is not None and w.shape[-1] == 65:
                                         # Last batch item, CLS token (0) attention to patches (1:)
                                         patch_attn = w[-1, 0, 1:] # (64,)
                                         
                                         # Handle dynamic patch size
                                         num_tokens = patch_attn.shape[0]
                                         grid_tokens = int(np.sqrt(num_tokens)) # e.g. 64
                                         
                                         patch_attn = patch_attn.reshape(grid_tokens, grid_tokens).detach().cpu().numpy()
                                         
                                         # Upscale if needed (Patch > 1)
                                         scale = 64 // grid_tokens
                                         if scale > 1:
                                             patch_attn = np.kron(patch_attn, np.ones((scale, scale)))
                                         
                                         # Normalize
                                         p_min, p_max = patch_attn.min(), patch_attn.max()
                                         if p_max > p_min:
                                             patch_attn = (patch_attn - p_min) / (p_max - p_min)
                                         
                                         heatmap_data = patch_attn.tolist()

                    except Exception as e:
                        pass

                # Get Objects & Metrics
                objects = getattr(self.agent, "latest_detected_objects", [])
                dopamine = 0.0
                plan_confidence = 0.0
                
                try:
                    infos = self.locals.get("infos", [])
                    if infos and len(infos) > 0:
                        info = infos[0]
                        dopamine = float(info.get("dopamine", 0.0))
                        plan_confidence = float(info.get("plan_confidence", 0.0))
                        manual_dopamine = float(info.get("manual_dopamine", 0.0))
                except:
                    pass

                if latest_frame and latest_frame.frame:
                    # Calculate reward mean safely
                    rew_mean = 0.0
                    if len(self.model.ep_info_buffer) > 0:
                        val = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                        if not np.isnan(val) and not np.isinf(val):
                            rew_mean = float(val)

                    msg = {
                        "grids": latest_frame.frame,
                        "game_id": self.agent.game_id,
                        "score": latest_frame.score,
                        "state": f"Step: {self.num_timesteps}",
                        "last_action": last_action,
                        "cursor": {"x": self.agent.cursor_x, "y": self.agent.cursor_y},
                        # Only update heatmap if computed
                        **({"attention": heatmap_data} if heatmap_data is not None else {}),
                        "objects": objects,
                        "metrics": {
                            "dopamine": dopamine,
                            "plan_confidence": plan_confidence,
                            "reward_mean": rew_mean,
                            "manual_dopamine": manual_dopamine
                        }
                    }
                    
                    # Non-blocking write via queue
                    if not self.msg_queue.full():
                        self.msg_queue.put(msg)
                        
            except Exception:
                pass 
        return True

