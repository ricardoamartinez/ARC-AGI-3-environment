import logging
from typing import Optional
import base64
import json
import subprocess
import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# from .langgraph_thinking.vision import render_frame 
# Replacing import with local custom renderer

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState

logger = logging.getLogger()

# Enhanced Palette including Standard ARC colors
CUSTOM_PALETTE = {
    0: (0, 0, 0),       # Black
    1: (0, 116, 217),   # Blue
    2: (255, 65, 54),   # Red
    3: (46, 204, 64),   # Green
    4: (255, 220, 0),   # Yellow
    5: (170, 170, 170), # Gray
    6: (240, 18, 190),  # Fuchsia
    7: (255, 133, 27),  # Orange
    8: (127, 219, 255), # Teal
    9: (135, 12, 37),   # Maroon
    10: (255, 255, 255), # White
    11: (128, 128, 128), # Gray
    12: (200, 200, 255), # Player/Special
}

SCALE_FACTOR = 30 # Initial scale

def custom_render_frame(array_3d: list[list[list[int]]], description: str, max_size: int = 800) -> str:
    """
    Renders a game frame to a PNG image with corrected colors.
    Scales dynamically to fit within max_size x max_size.
    """
    # Convert the 3D array to a NumPy array (take first channel)
    if not array_3d:
        return ""
        
    np_array = np.array(array_3d[0], dtype=np.uint8)
    orig_height, orig_width = np_array.shape

    # Calculate dynamic scale factor
    scale_x = max_size // orig_width
    scale_y = max_size // orig_height
    dynamic_scale = min(scale_x, scale_y, 30) # Cap at 30x scaling
    dynamic_scale = max(dynamic_scale, 10) # Ensure at least 10x scaling

    # Create an empty RGB image with scaled dimensions
    scaled_width = (orig_width) * dynamic_scale
    scaled_height = (orig_height) * dynamic_scale

    img = Image.new("RGB", (scaled_width, scaled_height), (0, 0, 0))
    pixels = img.load()

    # Fill the image with colors from the palette
    for y in range(orig_height):
        for x in range(orig_width):
            color_num = np_array[y, x]
            # Default to dark gray for unknown
            color = CUSTOM_PALETTE.get(color_num, (50, 50, 50)) 
            
            # Draw scaled pixel
            for i in range(dynamic_scale):
                for j in range(dynamic_scale):
                    pixels[
                        x * dynamic_scale + j,
                        y * dynamic_scale + i,
                    ] = color
                    
    # Optional: Draw minimal grid lines
    draw = ImageDraw.Draw(img)
    grid_color = (30, 30, 30)
    for x in range(0, scaled_width, dynamic_scale):
        draw.line([(x, 0), (x, scaled_height)], fill=grid_color)
    for y in range(0, scaled_height, dynamic_scale):
        draw.line([(0, y), (scaled_width, y)], fill=grid_color)

    # Convert image to base64
    from io import BytesIO
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str, dynamic_scale


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
            gui_script = os.path.join(os.path.dirname(__file__), "manual_pygame.py")
            print(f"DEBUG: Launching Pygame GUI script at {gui_script}")
            print(f"DEBUG: Using python: {sys.executable}")
            
            self.gui_process = subprocess.Popen(
                [sys.executable, gui_script],
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
             # Only ask terminal if GUI is dead, otherwise we loop?
             # Actually, if GUI is running, readline() should block until user inputs.
             # So if we are here, either readline returned empty (process died) or we got an action.
             
             if self.gui_process and self.gui_process.poll() is None:
                 # GUI is alive but maybe we want to allow terminal input too?
                 # It's hard to poll both. Let's assume GUI is primary if active.
                 pass
             else:
                # GUI is dead, fallback
                return self._ask_terminal(latest_frame)

        if not action:
             # If we still have no action (e.g. GUI closed), quit
             return GameAction.RESET

        return action

    def _send_frame_to_gui(self, frame: FrameData):
        if not self.gui_process or self.gui_process.poll() is not None:
            return
            
        try:
            if frame.frame:
                # frame.frame is a 3D array [Frames/Layers, H, W]
                # We send the whole list of frames to the GUI for animation
                
                # Sanity check dimensions
                # If it is [Time, H, W], we send it as is.
                
                msg = {
                    "grids": frame.frame, # Send all frames
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
                    # REFACTOR: Manual Agent now behaves like PPO Agent.
                    # We ignore any coordinates sent from GUI and just use ACTION6 generic.
                    # The env/agent must handle the cursor position.
                    
                    # BUT: This is the Agent class. It receives "ACTION6" from GUI.
                    # It needs to return a GameAction.
                    
                    # If we strip coordinates here, the 'take_action' later needs to know where to click?
                    # NO: The environment (if wrapping this) handles cursor. 
                    # If we are running ManualAgent directly against the Game Server, we DO need coords.
                    
                    # Wait, ManualAgent is usually wrapped or used directly?
                    # If used directly, we are the "Environment".
                    
                    # If we want to force "Cursor Logic" on ManualAgent, we need to track cursor state HERE.
                    # However, ManualAgent is usually just a pass-through.
                    
                    # The user asked to "remove the logic entirely to rather use the actual cursor object instead".
                    # This implies we should NOT parse x/y from the GUI message.
                    
                    action = GameAction.ACTION6
                    # We explicitly do NOT set data with x/y.
                    # action.set_data(action_data) <- REMOVED
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
