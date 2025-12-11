import sys
import pygame
import collections

from .constants import *
from .state import GameState
from .network import NetworkHandler
from .input import InputProcessor
from .ui.renderer import GameRenderer
from .ui.components import Button, Slider
from .utils import lerp

class ManualGame:
    def __init__(self):
        try:
            pygame.init()
        except ImportError:
            sys.exit(0)

        # Setup Window
        self.window_width = WINDOW_WIDTH_DEFAULT
        self.window_height = WINDOW_HEIGHT_DEFAULT
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        pygame.display.set_caption("ARC-AGI-3 Agent (Refactored)")

        # Resources
        self.clock = pygame.time.Clock()
        self.fonts = {
            "normal": pygame.font.SysFont(*FONT_CONFIG["normal"]),
            "title": pygame.font.SysFont(*FONT_CONFIG["title"]),
            "overlay": pygame.font.SysFont(*FONT_CONFIG["overlay"]),
            "small": pygame.font.SysFont(*FONT_CONFIG["small"])
        }

        # Subsystems
        self.state = GameState()
        self.network = NetworkHandler()
        self.renderer = GameRenderer(self.screen, self.fonts)
        self.input_processor = InputProcessor(self.state, self.network)

        # Dynamic Scaling
        self.scale_factor = SCALE_FACTOR_DEFAULT
        self.grid_width = GRID_WIDTH_DEFAULT

        # UI Components (Initialized later/dynamically)
        self.slider = None
        self.d_btn = None
        self.p_btn = None
        self.heatmap_btns = []

    def run(self):
        while self.state.running:
            self._handle_network_messages()
            self._update_layout_and_controls()
            
            # Event Processing
            events = pygame.event.get()
            self.input_processor.process_events(events, self.scale_factor)
            
            # UI Event Handling
            if not self.state.game_select_mode:
                self._handle_ui_events(events)
            else:
                self._handle_game_select_events(events)
            
            self._update_logic()
            self._draw()
            
            self.clock.tick(60)

        self.network.stop()
        pygame.quit()

    def _handle_network_messages(self):
        messages = self.network.get_messages()
        for data in messages:
            self._process_message(data)

    def _process_message(self, data):
        # Update State based on data
        if "last_action" in data:
            self.state.last_action_info = data["last_action"]
            if self.state.last_action_info and self.state.last_action_info.get("id") == 6:
                adata = self.state.last_action_info.get("data", {})
                self.state.last_click_pos = (adata.get("x", 0), adata.get("y", 0))
                self.state.last_click_time = pygame.time.get_ticks()

        if "game_id" in data: self.state.game_id = data["game_id"]
        if "score" in data: self.state.score = data["score"]
        if "state" in data: self.state.state = data["state"]
        
        if "cursor" in data:
            target = (data["cursor"]["x"], data["cursor"]["y"])
            self.state.cursor_pos = target
            if self.state.visual_cursor_pos is None:
                self.state.visual_cursor_pos = list(target)

        if "attention" in data:
            self.state.current_attention_map = data["attention"]
            self.state.current_maps["attention"] = data["attention"]
        
        if "maps" in data:
            maps = data["maps"]
            if isinstance(maps, dict):
                for k, v in maps.items():
                    if isinstance(v, list):
                        self.state.current_maps[k] = v

        if "objects" in data:
            self.state.current_objects = data["objects"]

        if "metrics" in data:
            m = data["metrics"]
            for k in ["reward", "dopamine", "confidence", "manual_dopamine", "pain", "trigger"]:
                key_map = {"reward": "reward_mean", "confidence": "plan_confidence"}
                val_key = key_map.get(k, k)
                if val_key in m:
                    self.state.metrics_history[k].append(m[val_key])
            if "current_thought" in m:
                self.state.current_thought = m["current_thought"]
            
            # Trim history
            for k in self.state.metrics_history:
                if len(self.state.metrics_history[k]) > MAX_HISTORY:
                    self.state.metrics_history[k] = self.state.metrics_history[k][-MAX_HISTORY:]

        if "grids" in data:
            grids = data["grids"]
            self.state.waiting_for_server = False
            if grids:
                self.state.current_grids = grids
                self.state.current_grid_idx = 0
                self.state.animation_timer = pygame.time.get_ticks()
                self._update_scaling(grids[-1])
                self.renderer.update_palette(grids[-1])
        
        elif "grid" in data:
             self.state.current_grids = [data["grid"]]
             self.state.current_grid_idx = 0
             self.state.waiting_for_server = False
        
        elif "action" in data and data["action"] == "SHOW_GAME_SELECTOR":
            self.state.available_games = data.get("games", [])
            self.state.game_select_mode = True

    def _update_scaling(self, grid):
        if self.scale_factor == 1: # Only update if not set (or create a flag for initial set)
            grid_h = len(grid)
            grid_w = len(grid[0])
            scale_x = 800 // grid_w
            scale_y = 800 // grid_h
            self.scale_factor = min(scale_x, scale_y, 40)
            self.scale_factor = max(self.scale_factor, 10)
            
            self.grid_width = grid_w * self.scale_factor
            target_h = max(grid_h * self.scale_factor, 600)
            target_w = self.grid_width + SIDEBAR_WIDTH
            
            if target_w != self.window_width or target_h != self.window_height:
                self.window_width = target_w
                self.window_height = target_h
                self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)

    def _update_layout_and_controls(self):
        # Recreate/Update UI controls based on current dimensions
        # Constants from original code
        SIDEBAR_X = self.grid_width
        SIDEBAR_PAD = 20
        CONTROLS_Y_START = self.window_height - 200
        
        slider_x = SIDEBAR_X + SIDEBAR_PAD
        slider_w = SIDEBAR_WIDTH - (SIDEBAR_PAD * 2)
        
        # Speed Slider
        self.slider = Slider(
            pygame.Rect(slider_x, CONTROLS_Y_START + 25, slider_w, 20),
            0.0, 1.0, self.state.speed_val,
            callback=lambda v: self._set_speed(v)
        )
        self.slider.dragging = self.state.dragging_slider # Sync state

        # Dopamine Button
        d_btn_y = CONTROLS_Y_START + 75
        self.d_btn = Button(
            pygame.Rect(slider_x, d_btn_y, slider_w, 40),
            "DOPAMINE (D)", self.fonts["normal"],
            (0, 100, 0), (255, 255, 255),
            active_color=(50, 255, 50),
            callback=lambda: self._set_holding_d(True)
        )
        self.d_btn.is_active = self.state.holding_d_key

        # Pain Button
        p_btn_y = d_btn_y + 50
        self.p_btn = Button(
            pygame.Rect(slider_x, p_btn_y, slider_w, 40),
            "PAIN (P)", self.fonts["normal"],
            (100, 0, 0), (255, 255, 255),
            active_color=(255, 50, 50),
            callback=lambda: self._set_holding_p(True)
        )
        self.p_btn.is_active = self.state.holding_p_key

        # Heatmap Buttons
        self.heatmap_btns = []
        hm_y_start = p_btn_y + 50
        cols = 4
        hm_btn_w = (slider_w - 15) // cols
        hm_btn_h = 25
        gap = 5
        
        for i, mode in enumerate(HEATMAP_MODES):
            row = i // cols
            col = i % cols
            hm_x = slider_x + col * (hm_btn_w + gap)
            hm_y = hm_y_start + row * (hm_btn_h + gap)
            
            btn = Button(
                pygame.Rect(hm_x, hm_y, hm_btn_w, hm_btn_h),
                mode[:4].capitalize(), # Short label
                self.fonts["small"],
                HEATMAP_COLORS.get(mode, (100, 100, 100)),
                (255, 255, 255),
                active_color=HEATMAP_COLORS.get(mode, (200, 200, 200)), # Simplified
                callback=lambda m=mode: self._set_heatmap(m)
            )
            # Active check logic needs to be checked in draw or passed
            # We'll just rely on state
            self.heatmap_btns.append(btn)

    def _set_speed(self, val):
        self.state.speed_val = val
        self.network.send_action("SET_SPEED", value=val)

    def _set_holding_d(self, val):
        self.state.holding_d_key = val
        
    def _set_holding_p(self, val):
        self.state.holding_p_key = val

    def _set_heatmap(self, mode):
        self.state.selected_heatmap_mode = mode
        self.state.show_heatmap = True

    def _handle_ui_events(self, events):
        for event in events:
            # Pass to slider
            if self.slider and self.slider.handle_event(event):
                self.state.dragging_slider = self.slider.dragging
            
            # Pass to buttons
            if self.d_btn: self.d_btn.handle_event(event)
            if self.p_btn: self.p_btn.handle_event(event)
            for btn in self.heatmap_btns:
                btn.handle_event(event)
            
            # Mouse Up specific logic for releasing holds
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    # Release buttons if they were held by click
                    # The buttons handle click callbacks, but release logic is global here
                    if self.state.holding_d_key: self.state.holding_d_key = False
                    if self.state.holding_p_key: self.state.holding_p_key = False

            # KeyDown events for shortcuts
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_b:
                    self.network.send_action("BREED_MOTIVATION")


    def _handle_game_select_events(self, events):
         for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                
                # Check grid
                cols = 4
                box_w = 150
                box_h = 150
                gap = 20
                start_x = 50
                start_y = 70
                
                for i, g in enumerate(self.state.available_games):
                    row = i // cols
                    col = i % cols
                    x = start_x + col * (box_w + gap)
                    y = start_y + row * (box_h + gap)
                    
                    if pygame.Rect(x, y, box_w, box_h).collidepoint(mx, my):
                        self.network.send_action("GAME_SELECTED", game_id=g["game_id"])
                        self.state.game_select_mode = False
                        pygame.event.clear()
                        return

    def _update_logic(self):
        # Animation
        now = pygame.time.get_ticks()
        if not self.state.game_select_mode and self.state.current_grids and len(self.state.current_grids) > 1:
            if now - self.state.animation_timer > ANIMATION_SPEED:
                self.state.animation_timer = now
                if self.state.current_grid_idx < len(self.state.current_grids) - 1:
                    self.state.current_grid_idx += 1
        
        # Cursor Interpolation
        if self.state.visual_cursor_pos and self.state.cursor_pos:
            vx, vy = self.state.visual_cursor_pos
            tx, ty = self.state.cursor_pos
            lerp_factor = 0.2
            vx += (tx - vx) * lerp_factor
            vy += (ty - vy) * lerp_factor
            if abs(tx - vx) < 0.01: vx = tx
            if abs(ty - vy) < 0.01: vy = ty
            self.state.visual_cursor_pos = [vx, vy]

        # Dopamine/Pain Logic
        if self.state.holding_d_key:
            self.state.manual_dopamine_val = min(1.0, self.state.manual_dopamine_val + 0.02)
        else:
            self.state.manual_dopamine_val = max(0.0, self.state.manual_dopamine_val - 0.01)

        if self.state.holding_p_key:
            self.state.manual_pain_val = min(1.0, self.state.manual_pain_val + 0.05)
        else:
            self.state.manual_pain_val = max(0.0, self.state.manual_pain_val - 0.02)
            
        # Send updates if changed
        # We need to store last sent values to avoid spamming
        if not hasattr(self, 'last_sent_dopamine'): self.last_sent_dopamine = -1
        if abs(self.state.manual_dopamine_val - self.last_sent_dopamine) > 0.01:
            self.network.send_action("SET_MANUAL_DOPAMINE", value=self.state.manual_dopamine_val)
            self.last_sent_dopamine = self.state.manual_dopamine_val
            
        if not hasattr(self, 'last_sent_pain'): self.last_sent_pain = -1
        if abs(self.state.manual_pain_val - self.last_sent_pain) > 0.01:
            self.network.send_action("SET_MANUAL_PAIN", value=self.state.manual_pain_val)
            self.last_sent_pain = self.state.manual_pain_val

    def _draw(self):
        if self.state.game_select_mode:
            self.renderer.draw_game_selector(self.state)
        else:
            self.renderer.draw_main_interface(self.state, self.scale_factor, self.grid_width)
            
            # Draw UI overlays (Buttons/Slider) on top of sidebar
            # Note: renderer draws "controls placeholder", but we should actually draw the objects here
            # or pass them to renderer.
            # Since we have the objects:
            if self.slider: self.slider.draw(self.screen)
            if self.d_btn: self.d_btn.draw(self.screen)
            if self.p_btn: self.p_btn.draw(self.screen)
            for btn in self.heatmap_btns:
                # Highlight if active
                # The generic button class has is_active, but we need to update it for heatmap toggles
                if btn.text.lower().startswith(self.state.selected_heatmap_mode[:3].lower()) and self.state.show_heatmap:
                    # Rough check matching
                    pass 
                    # Actually we passed callback.
                    
                btn.draw(self.screen)
        
        pygame.display.flip()

