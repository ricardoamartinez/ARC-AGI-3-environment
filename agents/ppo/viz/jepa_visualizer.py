"""
Real-time V-JEPA Action Embedding Visualizer.

Features:
1. 3D PCA visualization of joint visual-action embeddings
2. Click-to-decode: click any point to see decoded grid
3. Live loss/accuracy metrics graphs
4. Color coding by action effectiveness
"""

import threading
import queue
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

import pygame
from pygame import Surface
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for thread safety
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Colors for different action types
ACTION_COLORS = {
    0: (255, 100, 100),   # CLICK (red)
    1: (255, 100, 100),   # CLICK
    2: (255, 100, 100),   # CLICK
    3: (255, 100, 100),   # CLICK
    4: (100, 255, 100),   # UP (green)
    5: (100, 255, 100),   # DOWN (green)
    6: (100, 255, 100),   # LEFT (green)
    7: (100, 255, 100),   # RIGHT (green)
    8: (100, 100, 255),   # SPACE (blue)
    9: (255, 255, 100),   # ENTER (yellow)
}

ACTION_NAMES = {
    0: "CLICK", 1: "CLICK", 2: "CLICK", 3: "CLICK",
    4: "UP", 5: "DOWN", 6: "LEFT", 7: "RIGHT",
    8: "SPACE", 9: "ENTER"
}


class JEPAVisualizer:
    """
    Real-time visualization window for ActionJEPA embeddings.
    
    Shows:
    - 3D PCA of visual-action joint embeddings
    - Click any point to decode back to grid
    - Live training metrics (loss, accuracy)
    - Color coding by action type and effectiveness
    """
    
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        title: str = "ActionJEPA Embedding Visualizer",
    ):
        self.width = width
        self.height = height
        self.title = title
        self.running = False
        
        # Data storage
        self.embeddings: List[np.ndarray] = []
        self.action_indices: List[int] = []
        self.had_effects: List[bool] = []
        self.grids: List[np.ndarray] = []  # Ground truth grids
        self.predicted_grids: List[Optional[np.ndarray]] = []  # Predicted/decoded grids
        self.cursors: List[Tuple[float, float]] = []
        
        # PCA components
        self.pca = PCA(n_components=3)
        self.pca_embeddings: Optional[np.ndarray] = None
        
        # Metrics history
        self.loss_history: List[float] = []
        self.acc_history: List[float] = []
        self.latent_loss_history: List[float] = []
        
        # Selected point for decoding
        self.selected_idx: Optional[int] = None
        self.pinned: bool = False  # True = user clicked a point, stays until click outside
        self.decoded_grid: Optional[np.ndarray] = None
        self.decoded_predicted_grid: Optional[np.ndarray] = None
        
        # 3D rotation and zoom
        self.rotation_x = 30
        self.rotation_y = 45
        self.zoom = 1.0
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        
        # Thread-safe update queue
        self.update_queue = queue.Queue()
        
        # Max points to visualize (for performance)
        self.max_points = 500
        
    def start(self):
        """Start the visualization in a separate thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the visualization."""
        self.running = False
        
    def add_embedding(
        self,
        embedding: np.ndarray,
        action_idx: int,
        had_effect: bool,
        grid: np.ndarray,
        cursor: Tuple[float, float],
        predicted_grid: Optional[np.ndarray] = None,
    ):
        """Add a new embedding point (thread-safe)."""
        self.update_queue.put({
            "type": "embedding",
            "embedding": embedding.copy(),
            "action_idx": action_idx,
            "had_effect": had_effect,
            "grid": grid.copy(),
            "predicted_grid": predicted_grid.copy() if predicted_grid is not None else None,
            "cursor": cursor,
        })
        
    def add_metrics(
        self,
        loss: float,
        accuracy: float,
        latent_loss: float = 0.0,
    ):
        """Add training metrics (thread-safe)."""
        self.update_queue.put({
            "type": "metrics",
            "loss": loss,
            "accuracy": accuracy,
            "latent_loss": latent_loss,
        })
        
    def _process_updates(self):
        """Process pending updates from the queue."""
        while not self.update_queue.empty():
            try:
                update = self.update_queue.get_nowait()
                
                if update["type"] == "embedding":
                    self.embeddings.append(update["embedding"])
                    self.action_indices.append(update["action_idx"])
                    self.had_effects.append(update["had_effect"])
                    self.grids.append(update["grid"])
                    self.predicted_grids.append(update.get("predicted_grid"))
                    self.cursors.append(update["cursor"])
                    
                    # Trim to max points
                    if len(self.embeddings) > self.max_points:
                        self.embeddings = self.embeddings[-self.max_points:]
                        self.action_indices = self.action_indices[-self.max_points:]
                        self.had_effects = self.had_effects[-self.max_points:]
                        self.grids = self.grids[-self.max_points:]
                        self.predicted_grids = self.predicted_grids[-self.max_points:]
                        self.cursors = self.cursors[-self.max_points:]
                        
                        # Adjust selected_idx if we're pinned and items were trimmed
                        if self.pinned and self.selected_idx is not None:
                            self.selected_idx = max(0, self.selected_idx - 1)
                    
                    # Update PCA if we have enough points
                    if len(self.embeddings) >= 10:
                        try:
                            emb_array = np.stack(self.embeddings)
                            self.pca_embeddings = self.pca.fit_transform(emb_array)
                        except Exception:
                            pass
                    
                    # If not pinned, automatically update to show latest point
                    if not self.pinned:
                        self._update_to_latest_point()
                            
                elif update["type"] == "metrics":
                    self.loss_history.append(update["loss"])
                    self.acc_history.append(update["accuracy"])
                    self.latent_loss_history.append(update["latent_loss"])
                    
                    # Trim history
                    max_history = 200
                    if len(self.loss_history) > max_history:
                        self.loss_history = self.loss_history[-max_history:]
                        self.acc_history = self.acc_history[-max_history:]
                        self.latent_loss_history = self.latent_loss_history[-max_history:]
                        
            except queue.Empty:
                break
                
    def _run(self):
        """Main visualization loop."""
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)
        clock = pygame.time.Clock()
        
        # Fonts
        try:
            font = pygame.font.SysFont("consolas", 12)
            title_font = pygame.font.SysFont("consolas", 16, bold=True)
        except:
            font = pygame.font.Font(None, 14)
            title_font = pygame.font.Font(None, 18)
        
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.dragging = True
                        self.last_mouse_pos = event.pos
                        # Check if clicking on a point
                        self._handle_click(event.pos)
                    elif event.button == 4:  # Scroll up - zoom in
                        self.zoom = min(5.0, self.zoom * 1.2)
                    elif event.button == 5:  # Scroll down - zoom out
                        self.zoom = max(0.2, self.zoom / 1.2)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        dx = event.pos[0] - self.last_mouse_pos[0]
                        dy = event.pos[1] - self.last_mouse_pos[1]
                        self.rotation_y += dx * 0.5
                        self.rotation_x += dy * 0.5
                        self.last_mouse_pos = event.pos
                elif event.type == pygame.MOUSEWHEEL:
                    # Alternative scroll handling
                    if event.y > 0:  # Scroll up
                        self.zoom = min(5.0, self.zoom * 1.2)
                    elif event.y < 0:  # Scroll down
                        self.zoom = max(0.2, self.zoom / 1.2)
            
            # Process pending updates
            self._process_updates()
            
            # Clear screen
            screen.fill((0, 0, 0))  # OLED black
            
            # Draw sections
            self._draw_3d_scatter(screen, font, title_font)
            self._draw_metrics(screen, font, title_font)
            self._draw_decoded_grid(screen, font, title_font)
            self._draw_legend(screen, font)
            
            pygame.display.flip()
            clock.tick(30)  # 30 FPS
            
        pygame.quit()
        
    def _handle_click(self, pos: Tuple[int, int]):
        """Handle click to select/pin a point or unpin."""
        # Check if click is in the 3D scatter area (left half)
        if pos[0] > self.width // 2:
            return
        
        if self.pca_embeddings is None or len(self.pca_embeddings) == 0:
            # No embeddings yet, nothing to select
            self.pinned = False
            return
            
        # Find nearest point
        scatter_center = (self.width // 4, self.height // 2 - 50)
        scatter_scale = 80 * self.zoom
        
        min_dist = float('inf')
        nearest_idx = None
        
        # Pre-calculate rotation matrices
        cos_y = np.cos(np.radians(self.rotation_y))
        sin_y = np.sin(np.radians(self.rotation_y))
        cos_x = np.cos(np.radians(self.rotation_x))
        sin_x = np.sin(np.radians(self.rotation_x))
        
        for i, pca_point in enumerate(self.pca_embeddings):
            # Project 3D to 2D
            x, y, z = pca_point
            
            # Rotate around Y axis
            x_rot = x * cos_y - z * sin_y
            z_rot = x * sin_y + z * cos_y
            
            # Rotate around X axis
            y_rot = y * cos_x - z_rot * sin_x
            
            # Project to 2D
            screen_x = int(scatter_center[0] + x_rot * scatter_scale)
            screen_y = int(scatter_center[1] - y_rot * scatter_scale)
            
            dist = np.sqrt((pos[0] - screen_x)**2 + (pos[1] - screen_y)**2)
            if dist < min_dist and dist < 20:  # 20 pixel threshold
                min_dist = dist
                nearest_idx = i
                
        if nearest_idx is not None:
            # Clicked on a point - pin it
            self.selected_idx = nearest_idx
            self.pinned = True
            self._update_decoded_grids_for_selected()
        else:
            # Clicked in 3D area but not on a point - unpin, go back to live
            self.pinned = False
            self._update_to_latest_point()
            
    def _update_decoded_grids_for_selected(self):
        """Update the decoded grids to show the selected point's data."""
        if self.selected_idx is not None and self.selected_idx < len(self.grids):
            self.decoded_grid = self.grids[self.selected_idx]
            if self.selected_idx < len(self.predicted_grids):
                self.decoded_predicted_grid = self.predicted_grids[self.selected_idx]
            else:
                self.decoded_predicted_grid = None
                
    def _update_to_latest_point(self):
        """Update to show the latest/most recent point."""
        if len(self.grids) > 0:
            self.selected_idx = len(self.grids) - 1
            self._update_decoded_grids_for_selected()
            
    def _draw_3d_scatter(self, screen: Surface, font, title_font):
        """Draw the 3D PCA scatter plot."""
        # Title
        title = title_font.render("Joint Embedding Space (PCA 3D)", True, (255, 255, 255))
        screen.blit(title, (10, 10))
        
        if self.pca_embeddings is None or len(self.pca_embeddings) < 3:
            text = font.render("Collecting embeddings...", True, (150, 150, 150))
            screen.blit(text, (self.width // 4 - 50, self.height // 2))
            return
            
        # Draw 3D scatter
        scatter_center = (self.width // 4, self.height // 2 - 50)
        scatter_scale = 80 * self.zoom
        
        # Sort by Z for proper depth rendering
        cos_y = np.cos(np.radians(self.rotation_y))
        sin_y = np.sin(np.radians(self.rotation_y))
        cos_x = np.cos(np.radians(self.rotation_x))
        sin_x = np.sin(np.radians(self.rotation_x))
        
        # Calculate screen positions and Z-depths
        screen_points = []
        for i, pca_point in enumerate(self.pca_embeddings):
            x, y, z = pca_point
            
            # Rotate around Y axis
            x_rot = x * cos_y - z * sin_y
            z_rot = x * sin_y + z * cos_y
            
            # Rotate around X axis
            y_rot = y * cos_x - z_rot * sin_x
            z_final = y * sin_x + z_rot * cos_x
            
            screen_x = int(scatter_center[0] + x_rot * scatter_scale)
            screen_y = int(scatter_center[1] - y_rot * scatter_scale)
            
            screen_points.append((i, screen_x, screen_y, z_final))
            
        # Sort by depth (back to front)
        screen_points.sort(key=lambda p: p[3])
        
        # Draw points
        for i, sx, sy, _ in screen_points:
            action_idx = self.action_indices[i]
            had_effect = self.had_effects[i]
            
            # Base color from action type
            color = ACTION_COLORS.get(action_idx, (200, 200, 200))
            
            # Dim if action had no effect
            if not had_effect:
                color = tuple(c // 3 for c in color)
                
            # Highlight selected point
            is_selected = i == self.selected_idx
            radius = 8 if is_selected else 4
            
            pygame.draw.circle(screen, color, (sx, sy), radius)
            
            # Draw highlight ring for selected point
            if is_selected:
                # Pinned = orange ring, Live = green ring
                ring_color = (255, 200, 100) if self.pinned else (100, 255, 100)
                pygame.draw.circle(screen, ring_color, (sx, sy), radius + 3, 2)
                
        # Draw axes
        axis_len = 60
        origin = scatter_center
        
        # X axis (red)
        x_end = (int(origin[0] + axis_len * cos_y), int(origin[1]))
        pygame.draw.line(screen, (255, 100, 100), origin, x_end, 2)
        
        # Y axis (green)
        y_end = (int(origin[0]), int(origin[1] - axis_len * cos_x))
        pygame.draw.line(screen, (100, 255, 100), origin, y_end, 2)
        
        # Z axis (blue)
        z_end = (int(origin[0] - axis_len * sin_y), int(origin[1] + axis_len * sin_x))
        pygame.draw.line(screen, (100, 100, 255), origin, z_end, 2)
        
        # Stats
        explained_var = self.pca.explained_variance_ratio_ if hasattr(self.pca, 'explained_variance_ratio_') else [0, 0, 0]
        if len(explained_var) >= 3:
            var_text = f"PCA Variance: {sum(explained_var[:3])*100:.1f}% (PC1:{explained_var[0]*100:.1f}% PC2:{explained_var[1]*100:.1f}% PC3:{explained_var[2]*100:.1f}%)"
            var_surf = font.render(var_text, True, (150, 150, 150))
            screen.blit(var_surf, (10, self.height - 80))
            
        points_text = f"Points: {len(self.embeddings)} | Drag: rotate | Scroll: zoom ({self.zoom:.1f}x) | Click: decode"
        points_surf = font.render(points_text, True, (150, 150, 150))
        screen.blit(points_surf, (10, self.height - 60))
        
    def _draw_metrics(self, screen: Surface, font, title_font):
        """Draw training metrics graphs."""
        # Metrics panel on right side
        panel_x = self.width // 2 + 10
        panel_y = 10
        panel_w = self.width // 2 - 20
        panel_h = 180
        
        # Title
        title = title_font.render("Training Metrics", True, (255, 255, 255))
        screen.blit(title, (panel_x, panel_y))
        
        # Draw loss graph
        graph_y = panel_y + 25
        graph_h = 70
        
        if len(self.loss_history) > 1:
            # Loss graph
            pygame.draw.rect(screen, (15, 15, 20), (panel_x, graph_y, panel_w, graph_h))
            
            max_loss = max(self.loss_history) if self.loss_history else 1
            min_loss = min(self.loss_history) if self.loss_history else 0
            loss_range = max(max_loss - min_loss, 0.001)
            
            points = []
            for i, loss in enumerate(self.loss_history):
                x = panel_x + int(i / len(self.loss_history) * panel_w)
                y = graph_y + graph_h - int((loss - min_loss) / loss_range * graph_h)
                points.append((x, y))
                
            if len(points) > 1:
                pygame.draw.lines(screen, (255, 100, 100), False, points, 2)
                
            loss_label = font.render(f"Loss: {self.loss_history[-1]:.4f}", True, (255, 100, 100))
            screen.blit(loss_label, (panel_x, graph_y - 15))
            
        # Draw accuracy graph
        graph_y = panel_y + 110
        
        if len(self.acc_history) > 1:
            pygame.draw.rect(screen, (15, 15, 20), (panel_x, graph_y, panel_w, graph_h))
            
            points = []
            for i, acc in enumerate(self.acc_history):
                x = panel_x + int(i / len(self.acc_history) * panel_w)
                y = graph_y + graph_h - int(acc * graph_h)
                points.append((x, y))
                
            if len(points) > 1:
                pygame.draw.lines(screen, (100, 255, 100), False, points, 2)
                
            acc_label = font.render(f"Accuracy: {self.acc_history[-1]*100:.1f}%", True, (100, 255, 100))
            screen.blit(acc_label, (panel_x, graph_y - 15))
            
    def _get_dynamic_palette(self, grid: np.ndarray) -> dict:
        """
        Get a dynamically adjusted palette based on grid background.
        Matches the logic in agents/manual/ui/renderer.py
        """
        # Base palette - must match BASE_PALETTE in agents/manual/constants.py
        base_palette = {
            0: (255, 255, 255),  # #FFFFFF - white
            1: (204, 204, 204),  # #CCCCCC - light gray
            2: (153, 153, 153),  # #999999 - gray
            3: (102, 102, 102),  # #666666 - dark gray
            4: (51, 51, 51),     # #333333 - darker gray
            5: (0, 0, 0),        # #000000 - black
            6: (229, 58, 163),   # #E53AA3 - pink/magenta
            7: (255, 123, 204),  # #FF7BCC - light pink
            8: (249, 60, 49),    # #F93C31 - red
            9: (30, 147, 255),   # #1E93FF - blue
            10: (136, 216, 241), # #88D8F1 - cyan
            11: (255, 220, 0),   # #FFDC00 - yellow
            12: (255, 133, 27),  # #FF851B - orange
            13: (146, 18, 49),   # #921231 - maroon
            14: (79, 204, 48),   # #4FCC30 - green
            15: (163, 86, 214),  # #A356D6 - purple
            # Special markers for cursor
            254: (255, 0, 255),  # Cursor outline - magenta
            255: (0, 255, 0),    # Cursor center - green
        }
        
        # Dynamic background logic - find most common value and swap with black
        flat_grid = grid.flatten()
        # Filter out cursor markers for background detection
        valid_vals = flat_grid[(flat_grid < 254)]
        if len(valid_vals) > 0:
            # Count occurrences
            unique, counts = np.unique(valid_vals, return_counts=True)
            bg_val = unique[np.argmax(counts)]
            
            if bg_val != 0:
                # Create a copy and swap colors
                palette = base_palette.copy()
                original_bg_color = palette.get(int(bg_val), (0, 0, 0))
                # Set dominant to Black
                palette[int(bg_val)] = (0, 0, 0)
                # Set 0 to the original color of the dominant (Swap)
                palette[0] = original_bg_color
                return palette
        
        return base_palette
    
    def _draw_decoded_grid(self, screen: Surface, font, title_font):
        """Draw both GT and predicted grids for selected point."""
        panel_x = self.width // 2 + 10
        panel_y = 210
        panel_w = self.width // 2 - 20
        panel_h = self.height - panel_y - 80
        
        # Title - indicate if pinned or live
        if self.pinned:
            title_text = f"Pinned Point #{self.selected_idx} (click away to unpin)"
            title_color = (255, 200, 100)  # Orange for pinned
        else:
            title_text = "Live View (click a point to pin)"
            title_color = (100, 255, 100)  # Green for live
            
        title = title_font.render(title_text, True, title_color)
        screen.blit(title, (panel_x, panel_y))
        
        if self.decoded_grid is None:
            text = font.render("Waiting for embeddings...", True, (150, 150, 150))
            screen.blit(text, (panel_x, panel_y + 30))
            return
        
        # Get dynamic palette based on ground truth grid background
        arc_colors = self._get_dynamic_palette(self.decoded_grid)
        
        # Calculate grid sizes - show two grids side by side
        grid_y = panel_y + 40
        half_width = (panel_w - 20) // 2
        grid_display_size = min(half_width, panel_h - 80)
        cell_size = max(1, grid_display_size // self.decoded_grid.shape[0])
        
        # Draw Ground Truth grid (left)
        gt_label = font.render("Ground Truth", True, (100, 255, 100))
        screen.blit(gt_label, (panel_x + half_width // 2 - 40, grid_y - 18))
        
        for y in range(self.decoded_grid.shape[0]):
            for x in range(self.decoded_grid.shape[1]):
                val = int(self.decoded_grid[y, x])
                # Look up color, fallback to dark gray for unknown values
                color = arc_colors.get(val, arc_colors.get(val % 16, (50, 50, 50)))
                
                rect = pygame.Rect(
                    panel_x + x * cell_size,
                    grid_y + y * cell_size,
                    cell_size,  # No gap - no gridlines
                    cell_size,
                )
                pygame.draw.rect(screen, color, rect)
        
        # Draw Predicted grid (right)
        pred_x = panel_x + half_width + 10
        pred_label = font.render("Predicted (Decoded)", True, (255, 200, 100))
        screen.blit(pred_label, (pred_x + half_width // 2 - 55, grid_y - 18))
        
        if self.decoded_predicted_grid is not None:
            for y in range(self.decoded_predicted_grid.shape[0]):
                for x in range(self.decoded_predicted_grid.shape[1]):
                    val = int(self.decoded_predicted_grid[y, x])
                    # Look up color, fallback to dark gray for unknown values
                    color = arc_colors.get(val, arc_colors.get(val % 16, (50, 50, 50)))
                    
                    rect = pygame.Rect(
                        pred_x + x * cell_size,
                        grid_y + y * cell_size,
                        cell_size,  # No gap - no gridlines
                        cell_size,
                    )
                    pygame.draw.rect(screen, color, rect)
        else:
            # No predicted grid available
            no_pred = font.render("(No prediction available)", True, (100, 100, 100))
            screen.blit(no_pred, (pred_x + 10, grid_y + grid_display_size // 2))
                
        # Show selected point info
        if self.selected_idx is not None:
            action_idx = self.action_indices[self.selected_idx]
            had_effect = self.had_effects[self.selected_idx]
            cursor = self.cursors[self.selected_idx]
            
            info_text = f"Action: {ACTION_NAMES.get(action_idx, '?')} | Effect: {'Yes' if had_effect else 'No'} | Cursor: ({cursor[0]:.1f}, {cursor[1]:.1f})"
            info_surf = font.render(info_text, True, (255, 255, 255))
            screen.blit(info_surf, (panel_x, self.height - 70))
            
            # Show difference count if both grids available
            if self.decoded_predicted_grid is not None:
                try:
                    diff_count = int(np.sum(self.decoded_grid != self.decoded_predicted_grid))
                    total_cells = self.decoded_grid.shape[0] * self.decoded_grid.shape[1]
                    acc = (total_cells - diff_count) / total_cells * 100
                    diff_text = f"Reconstruction: {acc:.1f}% ({diff_count} cells differ)"
                    diff_surf = font.render(diff_text, True, (200, 200, 100))
                    screen.blit(diff_surf, (panel_x, self.height - 55))
                except:
                    pass
            
    def _draw_legend(self, screen: Surface, font):
        """Draw action type legend."""
        legend_x = 10
        legend_y = self.height - 40
        
        # Effective vs ineffective
        pygame.draw.circle(screen, (200, 200, 200), (legend_x + 5, legend_y + 5), 5)
        text = font.render("Effective", True, (200, 200, 200))
        screen.blit(text, (legend_x + 15, legend_y))
        
        pygame.draw.circle(screen, (60, 60, 60), (legend_x + 100, legend_y + 5), 5)
        text = font.render("Ineffective (dim)", True, (100, 100, 100))
        screen.blit(text, (legend_x + 110, legend_y))
