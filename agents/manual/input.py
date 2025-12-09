import pygame
import sys
from .state import GameState
from .network import NetworkHandler
from .constants import *

class InputProcessor:
    def __init__(self, state: GameState, network: NetworkHandler):
        self.state = state
        self.network = network

    def process_events(self, events: list, scale_factor: int):
        for event in events:
            if event.type == pygame.QUIT:
                self.network.send_action("QUIT")
                self.state.running = False
            
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)
            
            elif event.type == pygame.KEYUP:
                self._handle_keyup(event)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mousedown(event, scale_factor)
                
            elif event.type == pygame.MOUSEBUTTONUP:
                self._handle_mouseup(event)
            
            elif event.type == pygame.MOUSEMOTION:
                self._handle_mousemotion(event)

    def _handle_keydown(self, event):
        self.state.waiting_for_server = True
        
        # Mapping pygame keys to actions
        key_map = {
            pygame.K_UP: "ACTION1",
            pygame.K_DOWN: "ACTION2",
            pygame.K_LEFT: "ACTION3",
            pygame.K_RIGHT: "ACTION4",
            pygame.K_SPACE: "ACTION5",
            pygame.K_RETURN: "ACTION7",
            pygame.K_r: "RESET",
            pygame.K_q: "QUIT"
        }
        
        if event.key in key_map:
            action = key_map[event.key]
            self.network.send_action(action)
            if action == "QUIT":
                self.state.running = False
            return

        # Specific Logic
        if event.key == pygame.K_h:
            self.state.show_heatmap = not self.state.show_heatmap
        
        elif pygame.K_0 <= event.key <= pygame.K_9:
            idx = event.key - pygame.K_0
            if idx == 0: idx = 10
            if 1 <= idx <= len(HEATMAP_MODES):
                self.state.selected_heatmap_mode = HEATMAP_MODES[idx-1]
                self.state.show_heatmap = True
        
        elif event.key == pygame.K_d:
            self.state.holding_d_key = True
        elif event.key == pygame.K_p:
            self.state.holding_p_key = True

    def _handle_keyup(self, event):
        if event.key == pygame.K_d:
            self.state.holding_d_key = False
        elif event.key == pygame.K_p:
            self.state.holding_p_key = False

    def _handle_mousedown(self, event, scale_factor):
        x, y = event.pos
        
        if self.state.game_select_mode:
            # Game selection logic is handled in the renderer/app logic mainly because it needs layout info
            # But we can try to handle it here if we knew the layout.
            # For simplicity, the original code handled it inline.
            # Ideally, we pass the click to the UI manager.
            # We'll assume the main app loop handles UI interaction for Game Selection to keep this clean
            # OR we emit a signal.
            pass
            
        elif event.button == 1: # Left Click
            # Grid Interaction
            grid_w = self._get_current_grid_width(scale_factor)
            if x < grid_w:
                gx = int(x / scale_factor)
                gy = int(y / scale_factor)
                self.state.spatial_goal_pos = (gx, gy)
                self.network.send_action("SET_SPATIAL_GOAL", x=gx, y=gy)
            
            # UI Interaction is handled by UI components usually
            # But for simplicity here, we can set flags that the App checks
            # or we rely on the UI components `handle_event` method called from App.
            pass
            
        elif event.button == 3: # Right Click
            grid_w = self._get_current_grid_width(scale_factor)
            if x < grid_w and self.state.spatial_goal_pos:
                self.state.spatial_goal_pos = None
                self.network.send_action("CLEAR_SPATIAL_GOAL")

    def _handle_mouseup(self, event):
        if event.button == 1:
            self.state.dragging_slider = False
            # Key releases are handled in keyup, but if button release affects keys:
            if self.state.holding_d_key: self.state.holding_d_key = False # Logic from original (maybe just UI button release?)
            if self.state.holding_p_key: self.state.holding_p_key = False

    def _handle_mousemotion(self, event):
        # Slider dragging handled in UI components
        pass

    def _get_current_grid_width(self, scale_factor):
        # We need to know the grid dimensions.
        if self.state.current_grids:
            last_grid = self.state.current_grids[-1]
            return len(last_grid[0]) * scale_factor
        return GRID_WIDTH_DEFAULT # Fallback

