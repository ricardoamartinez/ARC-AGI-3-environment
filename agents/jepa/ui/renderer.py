"""Rendering for JEPA UI."""
import pygame
import collections
from typing import Dict, Tuple, List, Optional

from .state import UIState
from .constants import *
from .components import Button, Slider, Graph
from .utils import get_inverse_color


class GameRenderer:
    """Renders the game grid and UI elements."""
    
    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.palette = BASE_PALETTE.copy()
        
    def update_palette(self, grid: List[List[int]]):
        """Dynamic background logic - swap dominant color with black."""
        try:
            flat_grid = [int(c) for row in grid for c in row]
            if flat_grid:
                counter = collections.Counter(flat_grid)
                bg_val, _ = counter.most_common(1)[0]
                
                new_palette = BASE_PALETTE.copy()
                
                if bg_val != 0:
                    original_bg_color = new_palette.get(bg_val, (0, 0, 0))
                    new_palette[bg_val] = (0, 0, 0)
                    new_palette[0] = original_bg_color
                
                self.palette = new_palette
        except Exception:
            # Fallback to base palette on any error
            self.palette = BASE_PALETTE.copy()

    def draw_game_selector(self, state: UIState):
        self.screen.fill((20, 20, 20))
        
        title_surf = self.fonts["title"].render("SELECT A GAME", True, (255, 255, 255))
        self.screen.blit(title_surf, (self.screen.get_width() // 2 - title_surf.get_width() // 2, 20))
        
        if not state.available_games:
            loading_surf = self.fonts["normal"].render("Loading games...", True, (150, 150, 150))
            self.screen.blit(loading_surf, (self.screen.get_width() // 2 - loading_surf.get_width() // 2, 100))
            return

        cols = 4
        box_w = 150
        box_h = 150
        gap = 20
        start_x = 50
        start_y = 70
        
        mx, my = pygame.mouse.get_pos()
        
        for i, g in enumerate(state.available_games):
            row = i // cols
            col = i % cols
            
            x = start_x + col * (box_w + gap)
            y = start_y + row * (box_h + gap)
            
            rect = pygame.Rect(x, y, box_w, box_h)
            is_hovered = rect.collidepoint(mx, my)
            
            color = (50, 50, 50) if not is_hovered else (80, 80, 80)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (100, 100, 100), rect, 2)
            
            self._draw_thumbnail(g.get("thumbnail"), x, y, box_w, box_h)
            
            g_title = g.get("title", g["game_id"][:4].upper())
            text_surf = self.fonts["title"].render(g_title, True, (0, 0, 0))
            text_rect = text_surf.get_rect(center=(x + box_w // 2, y + box_h - 15))
            
            label_padding = 5
            label_bg_rect = text_rect.inflate(label_padding * 2, label_padding * 2)
            pygame.draw.rect(self.screen, (255, 105, 180), label_bg_rect)
            self.screen.blit(text_surf, text_rect)

    def _draw_thumbnail(self, thumbnail, x, y, box_w, box_h):
        if not thumbnail: return
        
        thumb_area_h = box_h - 30
        thumb_area_w = box_w - 4
        thumb_x = x + 2
        thumb_y = y + 2
        
        grid_rows = len(thumbnail)
        grid_cols = len(thumbnail[0]) if grid_rows > 0 else 0
        
        if grid_rows > 0 and grid_cols > 0:
            scale_x = thumb_area_w / grid_cols
            scale_y = thumb_area_h / grid_rows
            thumb_scale = min(scale_x, scale_y, 8)
            
            flat_grid = [c for row in thumbnail for c in row]
            thumb_palette = BASE_PALETTE.copy()
            if flat_grid:
                counter = collections.Counter(flat_grid)
                bg_val, _ = counter.most_common(1)[0]
                if bg_val != 0:
                    original_bg_color = thumb_palette.get(bg_val, (0, 0, 0))
                    thumb_palette[bg_val] = (0, 0, 0)
                    thumb_palette[0] = original_bg_color
            
            total_thumb_w = grid_cols * thumb_scale
            total_thumb_h = grid_rows * thumb_scale
            thumb_offset_x = (thumb_area_w - total_thumb_w) // 2
            thumb_offset_y = (thumb_area_h - total_thumb_h) // 2
            
            for r in range(grid_rows):
                for c in range(grid_cols):
                    val = thumbnail[r][c]
                    color_val = thumb_palette.get(val, (50, 50, 50))
                    
                    cell_x = int(thumb_x + thumb_offset_x + c * thumb_scale)
                    cell_y = int(thumb_y + thumb_offset_y + r * thumb_scale)
                    cell_w = int(thumb_x + thumb_offset_x + (c + 1) * thumb_scale) - cell_x
                    cell_h = int(thumb_y + thumb_offset_y + (r + 1) * thumb_scale) - cell_y
                    
                    if cell_x < x + box_w - 2 and cell_y < y + box_h - 30:
                         pygame.draw.rect(self.screen, color_val, (cell_x, cell_y, cell_w, cell_h))

    def draw_main_interface(self, state: UIState, scale_factor: int, grid_width: int):
        self.screen.fill(BG_COLOR)
        
        self._draw_sidebar_bg(grid_width)
        self._draw_info_panel(state, grid_width)
        self._draw_graphs(state, grid_width)
        self._draw_grid(state, scale_factor, grid_width)
        self._draw_overlays(state, scale_factor)
        self._draw_keyboard(state)
        
        # Draw imagination mode overlay if active
        if state.imagination_mode:
            self._draw_imagination_overlay(state, grid_width)

    def _draw_sidebar_bg(self, x_start):
        pygame.draw.rect(self.screen, SIDEBAR_BG, (x_start, 0, SIDEBAR_WIDTH, self.screen.get_height()))
        pygame.draw.line(self.screen, BORDER_COLOR, (x_start, 0), (x_start, self.screen.get_height()), 2)

    def _draw_info_panel(self, state: UIState, x_start):
        pad = 20
        curr_y = 20
        x = x_start + pad
        
        def draw_line(text, font_key="normal", color=TEXT_COLOR):
            nonlocal curr_y
            surf = self.fonts[font_key].render(text, True, color)
            self.screen.blit(surf, (x, curr_y))
            curr_y += surf.get_height() + 5

        draw_line(f"Game: {state.game_id}", "title")
        draw_line(f"Score: {state.score}")
        draw_line(f"State: {state.state}")

        if state.current_thought:
            draw_line(f"Thought: {state.current_thought}", "normal", (0, 255, 255))
        
        status_color = (255, 255, 0) if state.waiting_for_server else (0, 255, 0)
        status_text = "Status: Processing..." if state.waiting_for_server else "Status: Ready"
        draw_line(status_text, "normal", status_color)
        
        curr_y += 10
        
        if state.last_action_info:
            aid = state.last_action_info.get("id")
            aname = state.last_action_info.get("name")
            if not aname: aname = ACTION_NAMES.get(aid, str(aid))
            
            if aid == 6:
                adata = state.last_action_info.get("data", {})
                if adata and "x" in adata:
                    if "(" not in aname: aname += f" ({adata['x']}, {adata['y']})"
            
            draw_line("Last Action:", "normal", (200, 200, 200))
            draw_line(aname, "title", (0, 200, 255))
            
            overlay_surf = self.fonts["overlay"].render(aname, True, (0, 255, 255))
            shadow_surf = self.fonts["overlay"].render(aname, True, (0, 0, 0))
            self.screen.blit(shadow_surf, (22, self.screen.get_height() - 48))
            self.screen.blit(overlay_surf, (20, self.screen.get_height() - 50))

        curr_y += 10
        draw_line("CONTROLS:", "title")
        draw_line("ARROWS: Move  SPACE: Use", "normal", (150, 150, 150))
        draw_line("CLICK: Set Goal  R-CLICK: Clear", "normal", (0, 255, 0))
        draw_line("0-9: Select Channel  I: Imagine", "normal", (150, 150, 150))
        draw_line("D: Dopamine  P: Pain  Q: Quit", "normal", (150, 150, 150))

    def _draw_graphs(self, state: UIState, x_start):
        GRAPHS_Y_START = 350
        CONTROLS_Y_START = self.screen.get_height() - 200
        
        graph_titles = [
            ("Reward", "reward", (0, 255, 0), None, None),
            ("Dopamine (Human)", "manual_dopamine", (255, 120, 120), 0.0, 1.0),
            ("Pain (Human)", "manual_pain", (255, 0, 0), 0.0, 1.0),
            ("Action Urge (Trigger)", "trigger", (255, 255, 0), -1.0, 1.0),
            ("Cursor Speed", "cursor_speed", (200, 200, 200), 0.0, None),
            ("Action Energy", "action_energy", (180, 180, 255), 0.0, None),
            ("Goal Distance", "goal_dist", (0, 255, 255), 0.0, None),
            ("Goal Progress", "goal_progress", (0, 180, 255), None, None),
        ]

        avail_h = CONTROLS_Y_START - GRAPHS_Y_START - 20
        graph_slot_h = avail_h / len(graph_titles)
        actual_graph_h = min(60, graph_slot_h - 25)
        
        graph_w = SIDEBAR_WIDTH - 40
        graph_x = x_start + 20

        for i, (title, key, color, min_v, max_v) in enumerate(graph_titles):
            gy = GRAPHS_Y_START + (i * graph_slot_h)
            rect = pygame.Rect(graph_x, gy + 20, graph_w, actual_graph_h)
            
            g = Graph(rect, title, key, color, min_v, max_v)
            g.draw(self.screen, state.metrics_history.get(key, []), self.fonts["normal"])

    def _draw_grid(self, state: UIState, scale: int, grid_width: int):
        if not state.current_grids:
            return

        # In imagination mode, use the imagined grid (decoded from world model latent)
        if state.imagination_mode and state.imagination_grid is not None:
            current_grid = state.imagination_grid
        else:
            current_grid = state.current_grids[state.current_grid_idx]
        rows = len(current_grid)
        cols = len(current_grid[0])
        
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, grid_width, self.screen.get_height()))

        if not state.show_heatmap:
            # Safety check: ensure palette is a dict
            palette = self.palette if isinstance(self.palette, dict) else BASE_PALETTE
            for r in range(rows):
                for c in range(cols):
                    val = current_grid[r][c]
                    color = palette.get(int(val), (50, 50, 50))
                    rect = (c * scale, r * scale, scale, scale)
                    pygame.draw.rect(self.screen, color, rect)
        else:
            self._draw_heatmap(state, scale, rows, cols)

    def _draw_heatmap(self, state: UIState, scale: int, rows: int, cols: int):
        mode = state.selected_heatmap_mode
        if mode not in state.current_maps:
            return
            
        active_map = state.current_maps[mode]
        hm_color = HEATMAP_COLORS.get(mode, (255, 255, 255))
        
        if active_map:
            map_rows = len(active_map)
            map_cols = len(active_map[0]) if map_rows > 0 else 0
            
            for r in range(map_rows):
                row_data = active_map[r]
                for c in range(map_cols):
                    val = row_data[c]
                    if mode == "visit": val = min(1.0, val / 5.0)
                    elif mode == "value": val = max(0.0, min(1.0, val / 255.0))
                    
                    intensity = max(0.0, min(1.0, val))
                    color = (int(hm_color[0] * intensity), 
                             int(hm_color[1] * intensity), 
                             int(hm_color[2] * intensity))
                    
                    rect = (c * scale, r * scale, scale, scale)
                    pygame.draw.rect(self.screen, color, rect)
        
        title = f"CHANNEL: {mode.upper()}"
        t_surf = self.fonts["overlay"].render(title, True, hm_color)
        t_shad = self.fonts["overlay"].render(title, True, (0, 0, 0))
        self.screen.blit(t_shad, (22, 22))
        self.screen.blit(t_surf, (20, 20))

    def _draw_overlays(self, state: UIState, scale: int):
        # Goal Flag
        if state.spatial_goal_pos:
            gx, gy = state.spatial_goal_pos
            fx = gx * scale + scale // 2
            fy = gy * scale + scale // 2

            pygame.draw.line(self.screen, (0, 255, 0), (fx, fy), (fx, fy - 20), 3)
            pygame.draw.polygon(
                self.screen,
                (0, 255, 0),
                [(fx, fy - 20), (fx + 15, fy - 15), (fx, fy - 10)],
            )
            pygame.draw.circle(self.screen, (0, 255, 0), (fx, fy), 5)
            radius = (pygame.time.get_ticks() // 50) % 20 + 5
            pygame.draw.circle(self.screen, (0, 255, 0), (fx, fy), radius, 1)

        # Click Ripple
        if state.last_click_pos:
            lx, ly = state.last_click_pos
            time_diff = pygame.time.get_ticks() - state.last_click_time
            if time_diff < CLICK_VIS_DURATION:
                center_x = int(round(lx)) * scale + (scale // 2)
                center_y = int(round(ly)) * scale + (scale // 2)
                
                progress = time_diff / CLICK_VIS_DURATION
                radius = int(scale * progress * 1.5)
                alpha = int(255 * (1.0 - progress))
                
                if radius > 0:
                    s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(s, (255, 255, 255, alpha), (radius, radius), radius, 2)
                    self.screen.blit(s, (center_x - radius, center_y - radius))

        # Virtual Cursor
        if state.visual_cursor_pos:
            cx, cy = state.visual_cursor_pos
            
            active_cx, active_cy = int(round(cx)), int(round(cy))
            if state.current_grids:
                grid = state.current_grids[0]
                if 0 <= active_cx < len(grid[0]) and 0 <= active_cy < len(grid):
                    rect = (active_cx * scale, active_cy * scale, scale, scale)
                    s = pygame.Surface((scale, scale), pygame.SRCALPHA)
                    s.fill((0, 100, 255, 50))
                    self.screen.blit(s, rect)
                    pygame.draw.rect(self.screen, (0, 100, 255), rect, 2)
            
            tip_x = cx * scale + (scale / 2)
            tip_y = cy * scale + (scale / 2)
            self._draw_cursor_sprite(tip_x, tip_y)

    def _draw_cursor_sprite(self, x, y):
        c_size = 24
        arrow_pts = [
            (0, 0), (0, c_size), (c_size * 0.25, c_size * 0.75),
            (c_size * 0.5, c_size * 1.2), (c_size * 0.7, c_size * 1.1),
            (c_size * 0.45, c_size * 0.65), (c_size * 0.75, c_size * 0.65)
        ]
        screen_pts = [(x + p[0], y + p[1]) for p in arrow_pts]
        shadow_pts = [(p[0]+1, p[1]+1) for p in screen_pts]
        
        pygame.draw.polygon(self.screen, (0, 0, 0), shadow_pts)
        pygame.draw.polygon(self.screen, (255, 255, 255), screen_pts)
        pygame.draw.polygon(self.screen, (0, 0, 0), screen_pts, 1)

    def _draw_keyboard(self, state: UIState):
        kb_base_x = 30
        kb_base_y = self.screen.get_height() - 150
        
        # Keys: (state_lookup_key, display_label, x, y, w, h)
        keys = [
            ("UP", "\u2191", 50, 0, 40, 40),       # ↑
            ("LEFT", "\u2190", 5, 45, 40, 40),     # ←
            ("DOWN", "\u2193", 50, 45, 40, 40),    # ↓
            ("RIGHT", "\u2192", 95, 45, 40, 40),   # →
            ("ENTER", "ENT", 145, 45, 60, 40),
            ("SPACE", "", 5, 90, 130, 30)          # Empty label for space
        ]
        
        for k_sym, k_label, kx, ky, kw, kh in keys:
            last_act = state.key_activations.get(k_sym, 0)
            was_penalized = state.key_penalized.get(k_sym, False)
            now = pygame.time.get_ticks()
            time_diff = now - last_act
            
            brightness = 50
            is_active = False
            
            if time_diff < 50:
                brightness = 255
                is_active = True
            elif time_diff < KEY_FADE_DURATION:
                ratio = 1.0 - (time_diff / KEY_FADE_DURATION)
                brightness = int(50 + ratio * 200)
            
            rect = pygame.Rect(kb_base_x + kx, kb_base_y + ky, kw, kh)
            
            if is_active or time_diff < KEY_FADE_DURATION:
                if was_penalized:
                    bg_color = (brightness, int(brightness * 0.3), int(brightness * 0.3))
                else:
                    bg_color = (brightness, brightness, brightness)
            else:
                bg_color = (brightness, brightness, brightness)
            pygame.draw.rect(self.screen, bg_color, rect, border_radius=5)
            
            if is_active:
                border_col = (255, 50, 50) if was_penalized else (0, 255, 0)
            else:
                border_col = (100, 100, 100)
            pygame.draw.rect(self.screen, border_col, rect, 2, border_radius=5)
            
            if k_label:
                text_col = (0, 0, 0) if brightness > 150 else (255, 255, 255)
                label_surf = self.fonts["title"].render(k_label, True, text_col)
                label_rect = label_surf.get_rect(center=rect.center)
                self.screen.blit(label_surf, label_rect)
    
    def _draw_imagination_overlay(self, state: UIState, grid_width: int):
        """Draw dream-like overlay when in imagination mode."""
        # Semi-transparent purple overlay for dream effect
        overlay = pygame.Surface((grid_width, self.screen.get_height()), pygame.SRCALPHA)
        overlay.fill((100, 50, 150, 40))  # Purple tint
        self.screen.blit(overlay, (0, 0))
        
        # Draw "IMAGINATION MODE" banner at top
        banner_height = 35
        banner = pygame.Surface((grid_width, banner_height), pygame.SRCALPHA)
        banner.fill((80, 40, 120, 200))  # Dark purple
        self.screen.blit(banner, (0, 0))
        
        # Title text
        title_text = "IMAGINATION MODE - Playing in World Model's Mind"
        title_surf = self.fonts["title"].render(title_text, True, (200, 150, 255))
        title_rect = title_surf.get_rect(center=(grid_width // 2, banner_height // 2))
        self.screen.blit(title_surf, title_rect)
        
        # Draw info panel at bottom
        info_y = self.screen.get_height() - 70
        info_panel = pygame.Surface((grid_width, 70), pygame.SRCALPHA)
        info_panel.fill((40, 20, 60, 180))
        self.screen.blit(info_panel, (0, info_y))
        
        # Stats
        stats_x = 10
        stats_y = info_y + 10
        
        step_text = f"Steps: {state.imagination_step_count}"
        step_surf = self.fonts["normal"].render(step_text, True, (200, 200, 255))
        self.screen.blit(step_surf, (stats_x, stats_y))
        
        reward_text = f"Predicted Reward: {state.imagination_reward:.2f}"
        reward_color = (100, 255, 100) if state.imagination_reward >= 0 else (255, 100, 100)
        reward_surf = self.fonts["normal"].render(reward_text, True, reward_color)
        self.screen.blit(reward_surf, (stats_x, stats_y + 20))
        
        win_text = f"Win Probability: {state.imagination_win_prob * 100:.1f}%"
        win_surf = self.fonts["normal"].render(win_text, True, (255, 220, 100))
        self.screen.blit(win_surf, (stats_x, stats_y + 40))
        
        # Exit hint
        hint_text = "Press [I] to exit imagination mode"
        hint_surf = self.fonts["normal"].render(hint_text, True, (150, 150, 200))
        hint_rect = hint_surf.get_rect(right=grid_width - 10, bottom=self.screen.get_height() - 10)
        self.screen.blit(hint_surf, hint_rect)
        
        # Draw pulsing border for dream effect
        import time
        pulse = int(128 + 127 * (0.5 + 0.5 * ((time.time() * 2) % 1)))
        border_color = (pulse // 2, 50, pulse)
        pygame.draw.rect(self.screen, border_color, (0, 0, grid_width, self.screen.get_height()), 4)
