import logging
import time
import json
import subprocess
import sys
import os
import random
import threading
from typing import Any, Optional, Set, List, Tuple

import numpy as np

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState

from .training import run_rtac_training, run_sac_training, run_world_model_training

logger = logging.getLogger()

class JEPAAgent(Agent):
    """
    V-JEPA 2 RL Agent for ARC-AGI-3.
    
    Uses Video Joint Embedding Predictive Architecture for:
    - Visual-action coordination in joint embedding space
    - Predicting action effectiveness from visual state
    - Fast adaptation to avoid ineffective actions
    - Real-time latent space visualization
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
        import time
        
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
            thumb = JEPAAgent.fetch_game_thumbnail(root_url, headers, g["game_id"])
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
            start_time = time.time()
            while proc.poll() is None:
                line = proc.stdout.readline()
                if not line:
                    break
                try:
                    data = json.loads(line.strip())
                    if data.get("action"):
                        logger.debug(f"[selector] recv action={data.get('action')} keys={list(data.keys())}")
                    if data.get("action") == "GAME_SELECTED":
                        selected_game_id = data.get("game_id")
                        logger.info(f"User selected game: {selected_game_id}")
                        break
                    elif data.get("action") == "QUIT":
                        logger.info("User quit selection.")
                        break
                except json.JSONDecodeError:
                    # Pygame and other libs may print non-JSON to stdout; keep it visible in DEBUG.
                    s = line.strip()
                    if s:
                        logger.debug(f"[selector] non-json stdout: {s[:200]}")

                # Safety timeout so we don't hang forever if the child UI stops sending
                if time.time() - start_time > 120:
                    logger.error("Timed out waiting for game selection (120s).")
                    break
            
            proc.terminate()
            return selected_game_id
            
        except Exception as e:
            logger.error(f"Error in GUI selector: {e}")
            return None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.gui_process = None
        self.cursor_x = 32
        self.cursor_y = 32
        self._quit_event = threading.Event()
        self._gui_reader_thread = None
        self.training_speed = 1.0  # 1.0 = fastest (no sleep), 0.0 = slowest
        self.manual_dopamine = 0.0
        self.manual_pain = 0.0
        self.spatial_goal: Optional[Tuple[int, int]] = None
        # Incremented whenever the goal changes so the trainer can reset traces/LSTM cleanly.
        self.goal_version: int = 0
        # Goal shaping is a debug/training aid for cursor navigation.
        # Default ON (so it learns quickly). Set JEPA_GOAL_SHAPING=0 to force OFF.
        self.goal_shaping_enabled: bool = os.environ.get("JEPA_GOAL_SHAPING", "1") != "0"

        # Optional: allow headless training (no pygame subprocess). Useful for non-interactive runs.
        no_gui = os.environ.get("JEPA_NO_GUI", "0") == "1"
        if not no_gui:
            self._start_gui()

        # Optional: set a fixed spatial goal from env (format: "x,y").
        # Default: only applied in headless runs, so UI runs still rely on clicking a goal.
        allow_goal_env = no_gui or os.environ.get("JEPA_GOAL_FROM_ENV_IN_UI", "0") == "1"
        fixed_goal = os.environ.get("JEPA_FIXED_GOAL", "").strip()
        if allow_goal_env and fixed_goal:
            try:
                sx, sy = fixed_goal.split(",", 1)
                self.spatial_goal = (int(sx), int(sy))
                self.goal_version += 1
                logger.info(f"Fixed goal from JEPA_FIXED_GOAL: {self.spatial_goal}")
            except Exception:
                logger.warning(f"Invalid JEPA_FIXED_GOAL='{fixed_goal}' (expected 'x,y'). Ignoring.")

        # Optional: pick a random goal at startup (useful for headless convergence tests).
        # If JEPA_FIXED_GOAL is set, it wins.
        if allow_goal_env and self.spatial_goal is None and os.environ.get("JEPA_RANDOM_GOAL", "0") == "1":
            # ARC env uses a 64x64 grid for cursor learning.
            gx = random.randint(0, 63)
            gy = random.randint(0, 63)
            self.spatial_goal = (gx, gy)
            self.goal_version += 1
            logger.info(f"Random goal from JEPA_RANDOM_GOAL: {self.spatial_goal}")

        # Always start with a default goal so training begins immediately.
        if self.spatial_goal is None:
            self.spatial_goal = (32, 32)
            self.goal_version += 1
            logger.info(f"Default goal set: {self.spatial_goal}")

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
                                    # Map slider 0.0-1.0 to a useful delay range.
                                    # 0.0 -> 1.0s delay (1 step/sec)
                                    # 1.0 -> 0.0s delay (max speed)
                                    # Use a non-linear mapping for better control at low speeds?
                                    # Actually, user requested "1 step per second to max speed".
                                    # Max speed = 0 delay.
                                    # Min speed (slider=0) = 1.0s delay.
                                    # Let's use simple linear interpolation of delay.
                                    # Delay = (1.0 - speed) * 1.0
                                    
                                    # The original logic used self.training_speed in some way?
                                    # Let's check how training_speed is used.
                                    self.training_speed = max(0.0, min(1.0, speed))
                                elif data.get("action") == "SET_MANUAL_DOPAMINE":
                                    self.manual_dopamine = min(1.0, float(data.get("value", 0.0)))
                                elif data.get("action") == "SET_MANUAL_PAIN":
                                    self.manual_pain = min(1.0, float(data.get("value", 0.0)))
                                elif data.get("action") == "SET_SPATIAL_GOAL":
                                    x = int(data.get("x", 0))
                                    y = int(data.get("y", 0))
                                    self.spatial_goal = (x, y)
                                    self.goal_version += 1
                                    logger.info(f"Set goal: {self.spatial_goal}")
                                elif data.get("action") == "CLEAR_SPATIAL_GOAL":
                                    self.spatial_goal = None
                                    self.goal_version += 1
                                    logger.info("Cleared goal")
                                elif data.get("action") == "TOGGLE_GOAL_SHAPING":
                                    self.goal_shaping_enabled = not self.goal_shaping_enabled
                                    logger.info(f"Goal shaping enabled: {self.goal_shaping_enabled}")
                                # Handle game actions from user (keyboard input)
                                elif data.get("action") in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7", "RESET"):
                                    action_name = data.get("action")
                                    try:
                                        game_action = GameAction[action_name]
                                        frame = self.take_action(game_action)
                                        if frame:
                                            self.append_frame(frame)
                                            logger.debug(f"Human action: {action_name}")
                                    except Exception as e:
                                        logger.debug(f"Error executing human action {action_name}: {e}")
                                    
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
            except Exception as e:
                logger.debug(f"Failed to terminate GUI process: {e}")
            self.gui_process = None

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return True

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        return GameAction.RESET

    def main(self) -> None:
        """
        Main entry point for PPOAgent training.
        """
        # PPOAgent overrides Agent.main(), so we must initialize the base timer here
        # (otherwise Agent.cleanup() will print epoch seconds).
        self.timer = time.time()
        trainer = os.environ.get("JEPA_TRAINER", "sac").strip().lower()
        if trainer == "rtac":
            run_rtac_training(self)
        elif trainer in ("world_model", "wm", "jepa"):
            run_world_model_training(self)
        else:
            # SAC is the default: stable continuous control with replay + target critics
            run_sac_training(self)
