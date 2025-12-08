import logging
import time
import json
import subprocess
import sys
import os
import threading
from typing import Any, Optional, Set, List, Tuple

import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState

from .env import ARCGymEnv
from .callbacks import LiveVisualizerCallback
from .models import ArcViTFeatureExtractor

logger = logging.getLogger()

class PPOAgent(Agent):
    """
    An agent that trains a PPO model to play the game.
    """
    
    @staticmethod
    def fetch_game_thumbnail(root_url: str, headers: dict, game_id: str) -> Optional[list]:
        """Fetch the initial game state as a thumbnail."""
        import requests
        try:
            # Open a temporary scorecard
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
            
            # Reset the game to get initial state
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
                # Close scorecard before returning
                requests.post(f"{root_url}/api/scorecard/close", json={"card_id": card_id}, headers=headers, timeout=2)
                return None
            
            frame_data = r.json()
            if "frame" in frame_data and frame_data["frame"]:
                thumbnail = frame_data["frame"][0] if frame_data["frame"] else None
            else:
                thumbnail = None
            
            # Close the temporary scorecard
            requests.post(f"{root_url}/api/scorecard/close", json={"card_id": card_id}, headers=headers, timeout=2)
            
            return thumbnail
        except Exception as e:
            logger.debug(f"Error fetching thumbnail for {game_id}: {e}")
            return None
    
    @staticmethod
    def select_game_interactively(root_url: str, headers: dict) -> Optional[str]:
        import requests
        import concurrent.futures
        
        # 1. Fetch Games
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
        
        # 2. Fetch Thumbnails in parallel
        logger.info(f"Fetching thumbnails for {len(games)} games...")
        games_with_thumbnails = []
        
        def fetch_thumb(g):
            thumb = PPOAgent.fetch_game_thumbnail(root_url, headers, g["game_id"])
            return {**g, "thumbnail": thumb}
        
        # Use ThreadPoolExecutor to fetch thumbnails in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_thumb, g): g for g in games}
            for future in concurrent.futures.as_completed(futures):
                try:
                    game_with_thumb = future.result()
                    games_with_thumbnails.append(game_with_thumb)
                except Exception as e:
                    logger.debug(f"Error fetching thumbnail: {e}")
                    # Add game without thumbnail
                    games_with_thumbnails.append(futures[future])
        
        # Sort back to original order
        game_dict = {g["game_id"]: g for g in games_with_thumbnails}
        games_with_thumbnails = [game_dict[g["game_id"]] for g in games if g["game_id"] in game_dict]
        
        logger.info(f"Fetched {sum(1 for g in games_with_thumbnails if g.get('thumbnail'))} thumbnails")
            
        # 2. Launch GUI
        gui_script = os.path.join(os.path.dirname(__file__), "..", "templates", "manual_pygame.py")
        gui_script = os.path.abspath(gui_script)
        
        if not os.path.exists(gui_script):
             logger.error(f"GUI script not found at {gui_script}")
             return None

        try:
            logger.info("Launching Game Selection GUI...")
            proc = subprocess.Popen(
                [sys.executable, gui_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
                bufsize=1
            )
            
            # 3. Send Games List with Thumbnails
            msg = {"action": "SHOW_GAME_SELECTOR", "games": games_with_thumbnails}
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
            
            # 4. Wait for Selection
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
            
            # Cleanup
            proc.terminate()
            return selected_game_id
            
        except Exception as e:
            logger.error(f"Error in GUI selector: {e}")
            return None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.gui_process = None
        # Cursor state shared between Env and Visualizer
        self.cursor_x = 32 # Default center of 64x64
        self.cursor_y = 32
        self._quit_event = threading.Event()
        self._gui_reader_thread = None
        self.training_speed = 0.0 # 0.0 (Fast) to 1.0 (Slow)
        self.manual_dopamine = 0.0 # Shared state
        self._start_gui()

    def _start_gui(self):
        try:
            # We assume manual_pygame.py is in ../templates/ relative to this file
            # or we can look for it in the agents directory.
            # Current file: agents/ppo/agent.py
            # Target: agents/templates/manual_pygame.py
            
            gui_script = os.path.join(os.path.dirname(__file__), "..", "templates", "manual_pygame.py")
            gui_script = os.path.abspath(gui_script)
            
            if not os.path.exists(gui_script):
                 logger.error(f"GUI script not found at {gui_script}")
                 return

            self.gui_process = subprocess.Popen(
                [sys.executable, gui_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
                bufsize=1
            )
            
            # Start thread to read GUI output for quit signals
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
                                    self.manual_dopamine = float(data.get("value", 0.0))
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
        # PPO controls its own loop
        return True

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # Not used in training loop, but used if we run inference manually
        return GameAction.RESET

    def main(self) -> None:
        """
        Main entry point for the agent.
        """
        logger.info(f"Starting PPO Training for game {self.game_id}")
        
        # Create Environment
        # We pass 'self' so the environment can use our connection to the server
        env = ARCGymEnv(self, max_steps=100)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        # Setup Callback for Visualization
        callback = LiveVisualizerCallback(self.gui_process, self)
        
        # Define Policy Keywords to swap the architecture to ViT
        policy_kwargs = dict(
            features_extractor_class=ArcViTFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[128, 128], vf=[128, 128]), # MLP heads (Decoder) for Policy and Value
            log_std_init=-0.5 # Increased variance (std=0.6) to allow exploring different action buckets
        )
        
        # Check for CUDA
        device = "auto"
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            device = "cuda"
        else:
            logger.warning("CUDA is NOT available. Training will be slow on CPU.")
            logger.warning("To enable GPU acceleration, please install a CUDA-compatible PyTorch version.")
            device = "cpu"

        # Create Model
        self.model = PPO(
            "CnnPolicy", 
            env, 
            verbose=1,
            learning_rate=0.0003,
            n_steps=512, # Increased from 128 for better stability and longer horizon
            batch_size=64, # Increased from 8 to stabilize gradients (CNN is efficient enough)
            gamma=0.99,
            ent_coef=0.005, # Further reduced entropy to discourage random spamming
            policy_kwargs=policy_kwargs,
            device=device
        )
        
        # Train
        total_timesteps = 10000
        logger.info(f"Training for {total_timesteps} timesteps...")
        logger.info("Press 'q' in the GUI window to quit training early.")
        
        try:
            # Check for quit signal periodically during training
            # We'll modify the callback to check for quit events
            callback._quit_event = self._quit_event
            self.model.learn(total_timesteps=total_timesteps, callback=callback)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        finally:
            if self._quit_event.is_set():
                logger.info("Training stopped early due to quit signal")
            
            # Save model if training was interrupted
            if self.model:
                try:
                    self.model.save(f"ppo_arc_{self.game_id}")
                    logger.info("Model saved.")
                except Exception as e:
                    logger.warning(f"Failed to save model: {e}")
            
            # Final cleanup
            self.cleanup()

