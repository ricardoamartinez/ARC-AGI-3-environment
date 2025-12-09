import logging
import time
import json
import subprocess
import sys
import os
import threading
from typing import Any, Optional, Set, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState

from .env import ARCGymEnv
from .callbacks import LiveVisualizerCallback
from .models import OnlineActorCritic

logger = logging.getLogger()

class PPOAgent(Agent):
    """
    Real-Time Actor-Critic (RTAC) Agent with Eligibility Traces.
    Revised for Differentiability, Correct Optimization, and Recurrence.
    """
    
    @staticmethod
    def fetch_game_thumbnail(root_url: str, headers: dict, game_id: str) -> Optional[list]:
        """Fetch the initial game state as a thumbnail."""
        import requests
        try:
            r = requests.post(
                f"{root_url}/api/scorecard/open",
                json={"tags": ["thumbnail"]},
                headers=headers,
                timeout=5
            )
            if not r.ok:
                return None
            card_id = r.json().get("card_id")
            if not card_id:
                return None
            
            reset_data = {
                "card_id": card_id,
                "game_id": game_id
            }
            r = requests.post(
                f"{root_url}/api/cmd/RESET",
                json=reset_data,
                headers=headers,
                timeout=5
            )
            if not r.ok:
                requests.post(f"{root_url}/api/scorecard/close", json={"card_id": card_id}, headers=headers, timeout=2)
                return None
            
            frame_data = r.json()
            if "frame" in frame_data and frame_data["frame"]:
                thumbnail = frame_data["frame"][0] if frame_data["frame"] else None
            else:
                thumbnail = None
            
            requests.post(f"{root_url}/api/scorecard/close", json={"card_id": card_id}, headers=headers, timeout=2)
            return thumbnail
        except Exception as e:
            logger.debug(f"Error fetching thumbnail for {game_id}: {e}")
            return None
    
    @staticmethod
    def select_game_interactively(root_url: str, headers: dict) -> Optional[str]:
        import requests
        import concurrent.futures
        
        try:
            logger.info("Fetching game list...")
            r = requests.get(f"{root_url}/api/games", headers=headers, timeout=10)
            if not r.ok:
                logger.error(f"Failed to fetch games: {r.status_code}")
                return None
            games = r.json()
        except Exception as e:
            logger.error(f"Error fetching games: {e}")
            return None
            
        if not games:
            logger.error("No games available.")
            return None
        
        logger.info(f"Fetching thumbnails for {len(games)} games...")
        games_with_thumbnails = []
        
        def fetch_thumb(g):
            thumb = PPOAgent.fetch_game_thumbnail(root_url, headers, g["game_id"])
            return {**g, "thumbnail": thumb}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_thumb, g): g for g in games}
            for future in concurrent.futures.as_completed(futures):
                try:
                    game_with_thumb = future.result()
                    games_with_thumbnails.append(game_with_thumb)
                except Exception as e:
                    logger.debug(f"Error fetching thumbnail: {e}")
                    games_with_thumbnails.append(futures[future])
        
        game_dict = {g["game_id"]: g for g in games_with_thumbnails}
        games_with_thumbnails = [game_dict[g["game_id"]] for g in games if g["game_id"] in game_dict]
        
        logger.info(f"Fetched {sum(1 for g in games_with_thumbnails if g.get('thumbnail'))} thumbnails")
            
        try:
            logger.info("Launching Game Selection GUI...")
            proc = subprocess.Popen(
                [sys.executable, "-m", "agents.manual"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
                bufsize=1
            )
            
            msg = {"action": "SHOW_GAME_SELECTOR", "games": games_with_thumbnails}
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
            
            selected_game_id = None
            while proc.poll() is None:
                line = proc.stdout.readline()
                if not line:
                    break
                try:
                    data = json.loads(line.strip())
                    if data.get("action") == "GAME_SELECTED":
                        selected_game_id = data.get("game_id")
                        logger.info(f"User selected game: {selected_game_id}")
                        break
                    elif data.get("action") == "QUIT":
                        logger.info("User quit selection.")
                        break
                except json.JSONDecodeError:
                    pass
            
            proc.terminate()
            return selected_game_id
            
        except Exception as e:
            logger.error(f"Error in GUI selector: {e}")
            return None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.target_model = None
        self.gui_process = None
        self.cursor_x = 32
        self.cursor_y = 32
        self._quit_event = threading.Event()
        self._gui_reader_thread = None
        self.training_speed = 0.0
        self.manual_dopamine = 0.0
        self.manual_pain = 0.0
        self.spatial_goal = None # (x, y)
        self._start_gui()

    def _start_gui(self):
        try:
            print(f"DEBUG: Launching Pygame GUI module agents.manual")
            
            self.gui_process = subprocess.Popen(
                [sys.executable, "-m", "agents.manual"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
                bufsize=1
            )
            
            def read_gui_output():
                if self.gui_process and self.gui_process.stdout:
                    while self.gui_process.poll() is None:
                        try:
                            line = self.gui_process.stdout.readline()
                            if not line:
                                break
                            try:
                                data = json.loads(line.strip())
                                if data.get("action") == "QUIT":
                                    logger.info("Quit signal received from GUI")
                                    self._quit_event.set()
                                    break
                                elif data.get("action") == "SET_SPEED":
                                    speed = float(data.get("value", 0.0))
                                    self.training_speed = max(0.0, min(1.0, speed))
                                elif data.get("action") == "SET_MANUAL_DOPAMINE":
                                    self.manual_dopamine = min(1.0, float(data.get("value", 0.0)))
                                elif data.get("action") == "SET_MANUAL_PAIN":
                                    self.manual_pain = min(1.0, float(data.get("value", 0.0)))
                                elif data.get("action") == "SET_SPATIAL_GOAL":
                                    x = int(data.get("x", 0))
                                    y = int(data.get("y", 0))
                                    self.spatial_goal = (x, y)
                                elif data.get("action") == "CLEAR_SPATIAL_GOAL":
                                    self.spatial_goal = None
                            except json.JSONDecodeError:
                                pass
                        except Exception as e:
                            logger.debug(f"Error reading GUI output: {e}")
                            break
            
            self._gui_reader_thread = threading.Thread(target=read_gui_output, daemon=True)
            self._gui_reader_thread.start()
        except Exception as e:
            logger.error(f"Failed to start GUI: {e}")
            self.gui_process = None

    def cleanup(self, scorecard=None):
        super().cleanup(scorecard)
        if self.gui_process:
            try:
                self.gui_process.terminate()
            except:
                pass
            self.gui_process = None

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return True

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        return GameAction.RESET

    def main(self) -> None:
        """
        Main entry point for Real-Time Actor-Critic (RTAC) Training.
        """
        logger.info(f"Starting RTAC Training for game {self.game_id}")
        
        # Hyperparameters
        lr = 3e-4
        gamma = 0.99
        lam = 0.95 # Eligibility Trace decay
        tau = 0.005 # Target network soft update
        entropy_coef = 0.02
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Env
        env = ARCGymEnv(self, max_steps=1_000_000)
        
        # Models
        self.model = OnlineActorCritic(env.observation_space, env.action_space).to(device)
        self.target_model = OnlineActorCritic(env.observation_space, env.action_space).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # FIX: Use SGD for manual gradient/trace updates
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        
        # Initialize Traces
        traces = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        
        # Visualizer Callback wrapper
        callback = LiveVisualizerCallback(self.gui_process, self)
        callback._quit_event = self._quit_event
        
        obs, _ = env.reset()
        obs = np.transpose(obs, (2, 0, 1)) / 255.0
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Initialize Recurrent State
        # Hidden dim is 256 (same as features_dim in models.py)
        hidden_dim = 256
        hx = torch.zeros(1, hidden_dim, device=device)
        cx = torch.zeros(1, hidden_dim, device=device)
        
        step_count = 0
        
        try:
            while not self._quit_event.is_set():
                step_count += 1
                
                # 1. Forward Pass with Recurrence
                # Returns: mean_cont, std_cont, logits_disc, value, (next_hx, next_cx)
                mean_cont, std_cont, logits_disc, value, (next_hx, next_cx) = self.model(obs_tensor, (hx, cx))
                
                # NaN Check
                if torch.isnan(mean_cont).any() or torch.isnan(std_cont).any():
                     logger.error("NaN detected in model output! Resetting model parameters...")
                     self.model.apply(self.model._init_weights)
                     self.target_model.load_state_dict(self.model.state_dict())
                     traces = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
                     hx = torch.zeros(1, hidden_dim, device=device)
                     cx = torch.zeros(1, hidden_dim, device=device)
                     mean_cont, std_cont, logits_disc, value, (next_hx, next_cx) = self.model(obs_tensor, (hx, cx))

                # Sample Actions
                # Dynamic Entropy / Noise Control
                # If we have strong external feedback (manual dopamine OR spatial goal OR pain), we want to "exploit" the signal immediately.
                # Reduce noise to focus on the learned gradient.
                # High noise during high pain causes thrashing, which leads to hitting walls again.
                if self.manual_dopamine > 0.1 or self.spatial_goal is not None or self.manual_pain > 0.1:
                     current_std = torch.clamp(std_cont * 0.1, 0.01, 1.0) # Sharpen focus
                else:
                     current_std = std_cont

                dist_cont = torch.distributions.Normal(mean_cont, current_std)
                action_cont = dist_cont.sample()
                action_cont_clipped = torch.clamp(action_cont, -1.0, 1.0)
                
                dist_disc = torch.distributions.Categorical(logits=logits_disc)
                action_disc = dist_disc.sample()
                
                # 2. Step Env
                action_cont_np = action_cont_clipped.cpu().detach().numpy()[0]
                action_disc_int = action_disc.cpu().item()
                step_action = (action_cont_np, action_disc_int)
                
                next_obs, reward, terminated, truncated, info = env.step(step_action)
                
                # Preprocess Next Obs
                next_obs = np.transpose(next_obs, (2, 0, 1)) / 255.0
                callback.on_step(info) 
                
                # 3. Next State Value (Bootstrap)
                # Pass the NEW hidden state to the target model for one-step lookahead
                next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    # We only care about the value output
                    _, _, _, next_value_target, _ = self.target_model(next_obs_tensor, (next_hx, next_cx))
                
                if terminated or truncated:
                    delta = reward - value.item()
                    next_value_scalar = 0.0
                else:
                    delta = reward + gamma * next_value_target.item() - value.item()
                    next_value_scalar = next_value_target.item()
                
                # 4. Compute Gradients for Traces
                log_prob_cont = dist_cont.log_prob(action_cont).sum(dim=-1)
                log_prob_disc = dist_disc.log_prob(action_disc)
                log_prob_total = log_prob_cont + log_prob_disc
                
                entropy_cont = dist_cont.entropy().sum(dim=-1)
                entropy_disc = dist_disc.entropy()
                entropy_total = entropy_cont + entropy_disc
                
                loss_for_grads = log_prob_total + entropy_coef * entropy_total + value
                
                self.model.zero_grad()
                loss_for_grads.backward()
                
                # 5. Update Traces and Parameters
                with torch.no_grad():
                    if np.isnan(delta) or np.isinf(delta): delta = 0.0
                    d = torch.clamp(torch.tensor(delta, device=device), -10.0, 10.0)
                    
                    # Boost learning if significant external feedback
                    is_urgent = (self.manual_dopamine > 0.0) or (self.manual_pain > 0.0) or (self.spatial_goal is not None)
                    update_steps = 5 if is_urgent else 1
                    
                    # Dynamic Learning Rate for urgency (Reduced from 5.0 to 2.0 to prevent instability/overshoot)
                    current_lr = lr * 2.0 if is_urgent else lr

                    for _ in range(update_steps):
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                clipped_grad = torch.clamp(param.grad, -1.0, 1.0)
                                traces[name] = gamma * lam * traces[name] + clipped_grad
                                traces[name] = torch.clamp(traces[name], -10.0, 10.0)
                                param.grad = -d * traces[name]
                            else:
                                traces[name] = gamma * lam * traces[name]
                        
                        # Apply gradients with dynamic LR
                        for group in optimizer.param_groups:
                            group['lr'] = current_lr
                            
                        optimizer.step()
                        
                        # Restore base LR
                        for group in optimizer.param_groups:
                            group['lr'] = lr
                        
                        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                            target_param.data.mul_(1.0 - tau).add_(param.data * tau)

                # Loop Updates
                obs = next_obs
                obs_tensor = next_obs_tensor
                value = next_value_scalar
                
                # Update Hidden State (Detach for Truncated BPTT / Eligibility Traces)
                # We do NOT backprop through time steps explicitly (traces handle it)
                hx = next_hx.detach()
                cx = next_cx.detach()
                
                if terminated or truncated:
                    # NEVER reset the environment state during continual learning.
                    # Just reset the episode-specific markers if needed, but keep the agent's memory.
                    # For infinite horizon, we treat termination as just another state transition with 0 value lookahead.
                    
                    # Reset traces on "death" but keep memory?
                    # If we die, we should probably clear short-term memory to avoid confusion
                    # but keep long-term weights.
                    traces = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
                    
                    # We DO NOT reset the env here. The env handles its own lifecycle.
                    # If the env sends "terminated", it means the game is over and it has likely auto-reset internally
                    # or is waiting for a manual reset action.
                    # But for "continual learning", we want the agent to learn to press reset.
                    
                    # If the env auto-resets, next_obs is already the new state.
                    # If the env does NOT auto-reset, next_obs is the game over screen.
                    
                    # Let's assume Env auto-resets on termination for now to keep the loop going,
                    # BUT we preserve the hidden state to simulate "reincarnation" or persistent self.
                    
                    # Actually, user requested "NEVER reset". 
                    # So we should modify the Env to NOT send terminated=True unless strictly necessary.
                    pass
                
                if self.training_speed > 0:
                     time.sleep(self.training_speed * 0.1)

        except KeyboardInterrupt:
            logger.info("Training interrupted")
        finally:
            self.cleanup()
            if self.model:
                torch.save(self.model.state_dict(), f"rtac_model_{self.game_id}.pth")
