import logging
import json
import subprocess
import sys
import os
from typing import Optional

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState

logger = logging.getLogger()

class Manual(Agent):
    """
    A simple interactive agent that lets a human pick each action.
    Uses a separate GUI process for visualization and input.
    """

    MAX_ACTIONS = 1_000_000

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._quit = False
        self.gui_process = None
        self._start_gui()

    def _start_gui(self):
        try:
            print(f"DEBUG: Launching Pygame GUI module agents.manual")
            print(f"DEBUG: Using python: {sys.executable}")
            
            # We assume the package is named 'agents.manual' (after rename)
            self.gui_process = subprocess.Popen(
                [sys.executable, "-m", "agents.manual"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr, # Forward stderr to console
                text=True,
                bufsize=1 # Line buffered
            )
            print(f"DEBUG: GUI process started with PID {self.gui_process.pid}")
        except Exception as e:
            print(f"Failed to start GUI: {e}")
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
        if self._quit or latest_frame.state in {
            GameState.WIN,
            GameState.GAME_OVER,
        }:
            if self.gui_process:
                try:
                    self.gui_process.terminate()
                except:
                    pass
            return True
        return False

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        self._print_status(latest_frame)
        self._send_frame_to_gui(latest_frame)

        # Wait for action from GUI or Fallback to terminal
        action = None
        if self.gui_process and self.gui_process.poll() is None:
            try:
                # Read line from GUI stdout
                line = self.gui_process.stdout.readline()
                if line:
                    action = self._parse_gui_action(line.strip())
            except Exception as e:
                print(f"Error reading from GUI: {e}")
        
        # If GUI didn't return a valid action (or isn't running), ask terminal
        if not action and not self._quit:
             if self.gui_process and self.gui_process.poll() is None:
                 pass
             else:
                return self._ask_terminal(latest_frame)

        if not action:
             return GameAction.RESET

        return action

    def _send_frame_to_gui(self, frame: FrameData):
        if not self.gui_process or self.gui_process.poll() is not None:
            return
            
        try:
            if frame.frame:
                msg = {
                    "grids": frame.frame,
                    "game_id": self.game_id,
                    "score": frame.score
                }
                self.gui_process.stdin.write(json.dumps(msg) + "\n")
                self.gui_process.stdin.flush()
        except Exception as e:
            print(f"Failed to send frame to GUI: {e}")

    def _parse_gui_action(self, line: str) -> Optional[GameAction]:
        try:
            data = json.loads(line)
            cmd = data.get("action", "")
            
            if cmd == "QUIT":
                self._quit = True
                return GameAction.RESET
            
            if cmd == "RESET":
                return GameAction.RESET
                
            if cmd.startswith("ACTION"):
                if cmd == "ACTION6":
                    # Manual Agent behaves like PPO Agent.
                    # We ignore any coordinates sent from GUI and just use ACTION6 generic.
                    # The env/agent must handle the cursor position.
                    action = GameAction.ACTION6
                    return action
                else:
                    return GameAction.from_name(cmd)
                    
        except Exception as e:
            print(f"Failed to parse GUI action '{line}': {e}")
        return None

    def _ask_terminal(self, latest_frame: FrameData) -> GameAction:
        available = latest_frame.available_actions or list(GameAction)
        action: Optional[GameAction] = None

        while not action:
            raw = input(
                "Enter action (name/id), 'reset', or 'quit' "
                "[ex: action1, 1, action6 3 4]: "
            ).strip()

            if not raw:
                continue

            parts = raw.lower().split()
            cmd = parts[0]

            if cmd in {"q", "quit", "exit"}:
                self._quit = True
                return GameAction.RESET

            if cmd in {"r", "reset"}:
                action = GameAction.RESET
            else:
                try:
                    action = (
                        GameAction.from_id(int(cmd))
                        if cmd.isdigit()
                        else GameAction.from_name(cmd)
                    )
                except ValueError as e:
                    print(f"Unknown action '{cmd}': {e}")
                    action = None
                    continue

            if available and action not in available:
                print(
                    f"Action {action.name} not allowed. Allowed: "
                    f"{', '.join(a.name for a in available)}"
                )
                action = None
                continue

            data: dict[str, int] = {"game_id": self.game_id}

            if action.is_complex():
                x, y = self._parse_coords(parts)
                data["x"], data["y"] = x, y

            action.set_data(data)
        
        return action

    def _parse_coords(self, parts: list[str]) -> tuple[int, int]:
        if len(parts) >= 3:
            try:
                return int(parts[1]), int(parts[2])
            except ValueError:
                print("Coordinates must be integers in range 0-63.")

        while True:
            coords = input("Enter x y (0-63 0-63): ").strip().split()
            if len(coords) != 2:
                print("Please provide two numbers, e.g., '3 4'.")
                continue
            try:
                x, y = int(coords[0]), int(coords[1])
                if 0 <= x <= 63 and 0 <= y <= 63:
                    return x, y
                print("Coordinates must be between 0 and 63.")
            except ValueError:
                print("Coordinates must be integers.")

    def _print_status(self, frame: FrameData) -> None:
        print(
            f"\nGame: {self.game_id} | Score: {frame.score} | "
            f"State: {frame.state} | Actions taken: {self.action_counter}"
        )
        if frame.frame:
            height = len(frame.frame)
            width = len(frame.frame[0]) if frame.frame else 0
            channels = len(frame.frame[0][0]) if frame.frame and frame.frame[0] else 0
            print(f"Frame size: {width}x{height}x{channels}")
            
        if frame.available_actions:
            names = ", ".join(a.name for a in frame.available_actions)
            print(f"Available actions: {names}")
        else:
            print("Available actions: RESET, ACTION1-7")

