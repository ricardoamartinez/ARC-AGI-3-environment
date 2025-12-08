import sys
import json
import base64
import os
import io
import collections

try:
    import pygame
except ImportError:
    sys.exit(0)

def main():
    pygame.init()
    
    # Constants
    SCALE_FACTOR = 1 # Initial scale, will update dynamically
    SIDEBAR_WIDTH = 300
    WINDOW_HEIGHT = 1000 # Increased height
    GRID_WIDTH = 800 # Default start
    WINDOW_WIDTH = GRID_WIDTH + SIDEBAR_WIDTH
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("ARC-AGI-3 Agent")
    
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)
    title_font = pygame.font.SysFont("Arial", 20, bold=True)
    overlay_font = pygame.font.SysFont("Arial", 32, bold=True)
    
    # State
    current_grids = [] # List of grids for animation
    current_grid_idx = 0
    animation_timer = 0
    ANIMATION_SPEED = 100 # ms per frame
    
    game_id = "Waiting..."
    score = 0
    state = "Starting..."
    waiting_for_server = False
    
    # Selection Mode State
    game_select_mode = False
    available_games = [] # List of {"game_id": str, "title": str}
    
    # Heatmap State
    current_attention_map = None # 64x64 list of lists
    current_objects = [] # List of list of (r, c)
    show_heatmap = False
    
    last_action_info = None
    last_click_pos = None
    last_click_time = 0
    CLICK_VIS_DURATION = 500 # ms
    
    # Keyboard Symbol Persistence
    key_activations = {} # symbol -> timestamp
    KEY_FADE_DURATION = 300 # Fast fade for responsiveness
    
    cursor_pos = None # (x, y) target
    visual_cursor_pos = None # (x, y) current interpolated
    
    # Colors
    BG_COLOR = (0, 0, 0)
    TEXT_COLOR = (255, 255, 255)
    SIDEBAR_BG = (0, 0, 0)
    BORDER_COLOR = (50, 50, 50)
    GRID_LINE_COLOR = (30, 30, 30)
    
    # ARC Palette (Verified Vibrant)
    # 0 is typically background (Black). 10 is typically White.
    # We will dynamically swap these based on which is dominant to ensure BG is always Black.
    BASE_PALETTE = {
        0: (0, 0, 0),       # Black
        1: (30, 144, 255),  # Blue
        2: (255, 69, 0),    # Red
        3: (50, 205, 50),   # Green
        4: (255, 215, 0),   # Yellow
        5: (169, 169, 169), # Gray
        6: (255, 20, 147),  # Fuchsia
        7: (255, 140, 0),   # Orange
        8: (0, 255, 255),   # Cyan
        9: (128, 0, 128),   # Maroon
        10: (255, 255, 255), # White
        11: (105, 105, 105), # Gray
        12: (255, 255, 255), # Player
    }
    
    # Active palette to be updated dynamically
    PALETTE = BASE_PALETTE.copy()

    ACTION_NAMES = {
        1: "Move UP",
        2: "Move DOWN",
        3: "Move LEFT",
        4: "Move RIGHT",
        5: "USE (Space)",
        6: "CLICK",
        7: "CONFIRM (Enter)",
        0: "RESET"
    }

    # Metrics History
    metrics_history = {
        "reward": [],
        "dopamine": [],
        "confidence": [],
        "manual_dopamine": [],
        "trigger": []
    }
    MAX_HISTORY = 200
    
    # Speed Slider & Manual Dopamine
    speed_val = 0.0 
    manual_dopamine_val = 0.0 # 0.0 to 1.0, controlled by 'D' key
    dragging_slider = False
    holding_d_key = False
    
    running = True
    
    def send_action(action_name, **kwargs):
        msg = {"action": action_name}
        msg.update(kwargs)
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    # Input reading thread
    import threading
    import queue
    input_queue = queue.Queue()

    def read_stdin():
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                input_queue.put(line)
            except ValueError:
                break
    
    t = threading.Thread(target=read_stdin, daemon=True)
    t.start()

    def get_inverse_color(rgb):
        return (255 - rgb[0], 255 - rgb[1], 255 - rgb[2])

    while running:
        # Handle Input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                send_action("QUIT")
                running = False
            
            elif event.type == pygame.KEYDOWN:
                # Set status immediately
                waiting_for_server = True
                
                if event.key == pygame.K_UP:
                    send_action("ACTION1")
                elif event.key == pygame.K_DOWN:
                    send_action("ACTION2")
                elif event.key == pygame.K_LEFT:
                    send_action("ACTION3")
                elif event.key == pygame.K_RIGHT:
                    send_action("ACTION4")
                elif event.key == pygame.K_SPACE:
                    send_action("ACTION5")
                elif event.key == pygame.K_RETURN:
                    send_action("ACTION7")
                elif event.key == pygame.K_r:
                    send_action("RESET")
                elif event.key == pygame.K_h:
                    show_heatmap = not show_heatmap
                elif event.key == pygame.K_d:
                    holding_d_key = True
                elif event.key == pygame.K_q:
                    send_action("QUIT")
                    running = False
            
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_d:
                    holding_d_key = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    x, y = event.pos
                    
                    # Calculate dynamic positions again for hit testing
                    graph_start_y = y_offset + 20
                    min_slider_y = graph_start_y + 360
                    slider_y = min(WINDOW_HEIGHT - 60, max(WINDOW_HEIGHT - 80, min_slider_y))
                    
                    slider_x = GRID_WIDTH + 20
                    slider_w = SIDEBAR_WIDTH - 40
                    slider_h = 20
                    
                    # Check Slider
                    slider_rect = pygame.Rect(slider_x, slider_y, slider_w, slider_h)
                    if slider_rect.inflate(10, 10).collidepoint(x, y):
                        dragging_slider = True
                        ratio = (x - slider_x) / slider_w
                        speed_val = max(0.0, min(1.0, ratio))
                        send_action("SET_SPEED", value=speed_val)
                    
                    # Check Dopamine Button
                    btn_y = slider_y - 50
                    btn_h = 40
                    btn_rect = pygame.Rect(slider_x, btn_y, slider_w, btn_h)
                    if btn_rect.collidepoint(x, y):
                        holding_d_key = True
                    
                    # Check if click is in grid area
                    elif x < GRID_WIDTH:
                        # During training, the MODEL controls the cursor and clicks.
                        # Human clicks on the grid are ignored - this is just a visualizer.
                        # The model steers the cursor via acceleration output and triggers clicks.
                        pass

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging_slider = False
                    if holding_d_key: # Release button
                        holding_d_key = False
            
            elif event.type == pygame.MOUSEMOTION:
                if dragging_slider:
                    x, y = event.pos
                    slider_x = GRID_WIDTH + 20
                    slider_w = SIDEBAR_WIDTH - 40
                    
                    ratio = (x - slider_x) / slider_w
                    speed_val = max(0.0, min(1.0, ratio))
                    send_action("SET_SPEED", value=speed_val)

        # Manual Dopamine Logic
        # Gradual release: Slower rise/fall
        if holding_d_key:
            manual_dopamine_val = min(1.0, manual_dopamine_val + 0.02) # Double the rise rate (was 0.01)
        else:
            manual_dopamine_val = max(0.0, manual_dopamine_val - 0.01) # Slower decay
            
        if abs(manual_dopamine_val) > 0.001 or holding_d_key:
             # Send update if value is significant or changing
             # We piggyback on any action or just send dedicated update?
             # Better to send dedicated update, but don't flood.
             # Actually, we can send it alongside other messages or sparsely.
             # Let's send it every frame? No, too heavy.
             # Send only on change?
             pass
        
        # To keep it simple, we send it via SET_MANUAL_DOPAMINE if changed
        # We need a 'last_sent_dopamine' check
        if 'last_sent_dopamine' not in locals(): last_sent_dopamine = -1
        if abs(manual_dopamine_val - last_sent_dopamine) > 0.01:
            send_action("SET_MANUAL_DOPAMINE", value=manual_dopamine_val)
            last_sent_dopamine = manual_dopamine_val

        # Process incoming messages
        # Consume ALL messages in queue to catch up to latest state
        while not input_queue.empty():
            line = input_queue.get()
            try:
                data = json.loads(line)
                # Always update state if available
                if "last_action" in data:
                    last_action_info = data["last_action"]
                    if last_action_info:
                        # If it's a click (ACTION6)
                        aid = last_action_info.get("id")
                        adata = last_action_info.get("data", {})
                        
                        if aid == 6:
                            last_click_pos = (adata.get("x", 0), adata.get("y", 0))
                            last_click_time = pygame.time.get_ticks()
                            
                            # Do NOT snap visual cursor. 
                            # The cursor should continue its smooth trajectory.
                            # The click visualization will appear at last_click_pos.
                        else:
                            # Reset click if a move action happens? Or keep it?
                            # Keep it to show history briefly
                            pass

                if "game_id" in data:
                    game_id = data["game_id"]
                if "score" in data:
                    score = data["score"]
                if "state" in data:
                    state = data["state"]
                if "cursor" in data:
                    target = (data["cursor"]["x"], data["cursor"]["y"])
                    cursor_pos = target
                    if visual_cursor_pos is None:
                        visual_cursor_pos = list(target)

                if "attention" in data:
                    current_attention_map = data["attention"]
                if "objects" in data:
                    current_objects = data["objects"]

                if "metrics" in data:
                    m = data["metrics"]
                    # Update history
                    if "reward_mean" in m: metrics_history["reward"].append(m["reward_mean"])
                    if "dopamine" in m: metrics_history["dopamine"].append(m["dopamine"])
                    if "plan_confidence" in m: metrics_history["confidence"].append(m["plan_confidence"])
                    if "manual_dopamine" in m: metrics_history["manual_dopamine"].append(m["manual_dopamine"])
                    if "trigger" in m: metrics_history["trigger"].append(m["trigger"])
                    
                    # Trim
                    for k in metrics_history:
                        if len(metrics_history[k]) > MAX_HISTORY:
                            metrics_history[k] = metrics_history[k][-MAX_HISTORY:]

                if "grids" in data:
                    grids = data["grids"]
                    waiting_for_server = False
                    
                    if grids:
                        current_grids = grids
                        current_grid_idx = 0
                        animation_timer = pygame.time.get_ticks()
                        
                        # Dynamic Background Logic
                        # Determine dominant color in the last grid
                        last_grid = grids[-1]
                        # Flatten
                        flat_grid = [c for row in last_grid for c in row]
                        if flat_grid:
                            # Find most common value
                            counter = collections.Counter(flat_grid)
                            bg_val, _ = counter.most_common(1)[0]
                            
                            # Update PALETTE to ensure bg_val maps to Black (0,0,0)
                            # And swap 0 to take bg_val's original color if needed
                            
                            # Reset to base first
                            PALETTE = BASE_PALETTE.copy()
                            
                            if bg_val != 0:
                                # The background is NOT 0 (e.g. it is 10 White)
                                original_bg_color = PALETTE.get(bg_val, (0,0,0))
                                original_zero_color = PALETTE.get(0, (0,0,0))
                                
                                # Set dominant to Black
                                PALETTE[bg_val] = (0, 0, 0)
                                
                                # Set 0 to the original color of the dominant (Swap)
                                # e.g. If dominant was White, 0 becomes White.
                                PALETTE[0] = original_bg_color
                            
                            # If bg_val is 0, PALETTE[0] is already Black, so we are good.

                        # Update scaling only if needed (e.g. first frame) to avoid jitter
                        if SCALE_FACTOR == 1:
                            last_grid = grids[-1]
                            grid_h = len(last_grid)
                            grid_w = len(last_grid[0])
                            scale_x = 800 // grid_w
                            scale_y = 800 // grid_h
                            SCALE_FACTOR = min(scale_x, scale_y, 40)
                            SCALE_FACTOR = max(SCALE_FACTOR, 10)
                            GRID_WIDTH = grid_w * SCALE_FACTOR
                            min_height = 600
                            target_h = max(grid_h * SCALE_FACTOR, min_height)
                            target_w = GRID_WIDTH + SIDEBAR_WIDTH
                            if target_w != WINDOW_WIDTH or target_h != WINDOW_HEIGHT:
                                WINDOW_WIDTH = target_w
                                WINDOW_HEIGHT = target_h
                                screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
                            
                elif "grid" in data:
                    # Fallback
                    current_grids = [data["grid"]]
                    current_grid_idx = 0
                    waiting_for_server = False
                
                elif "action" in data:
                    if data["action"] == "SHOW_GAME_SELECTOR":
                        available_games = data.get("games", [])
                        game_select_mode = True
                        
            except json.JSONDecodeError:
                pass
            except Exception as e:
                sys.stderr.write(f"GUI Error: {e}\n")

        # Animation Logic
        now = pygame.time.get_ticks()
        if not game_select_mode and current_grids and len(current_grids) > 1:
            if now - animation_timer > ANIMATION_SPEED:
                animation_timer = now
                if current_grid_idx < len(current_grids) - 1:
                    current_grid_idx += 1
                # Stop at last frame, don't loop
        
        # Interpolate Cursor
        # Smoothly move visual_cursor_pos towards cursor_pos
        if visual_cursor_pos and cursor_pos:
            vx, vy = visual_cursor_pos
            tx, ty = cursor_pos
            
            # Simple Lerp
            # 0.2 factor at 60FPS = Fast convergence but smooth
            lerp_factor = 0.2 
            
            vx += (tx - vx) * lerp_factor
            vy += (ty - vy) * lerp_factor
            
            # Snap if close
            if abs(tx - vx) < 0.01: vx = tx
            if abs(ty - vy) < 0.01: vy = ty
            
            visual_cursor_pos = [vx, vy]

        # Draw
        screen.fill(BG_COLOR)
        
        if game_select_mode:
            # Draw Game Selection Grid
            screen.fill((20, 20, 20)) # Dark grey bg
            
            title_surf = title_font.render("SELECT A GAME", True, (255, 255, 255))
            screen.blit(title_surf, (WINDOW_WIDTH // 2 - title_surf.get_width() // 2, 20))
            
            # Show loading if no games yet
            if not available_games:
                loading_surf = font.render("Loading games...", True, (150, 150, 150))
                screen.blit(loading_surf, (WINDOW_WIDTH // 2 - loading_surf.get_width() // 2, 100))
                pygame.display.flip()
                clock.tick(60)
                continue
            
            # Grid Layout
            cols = 4
            box_w = 150
            box_h = 150
            gap = 20
            start_x = 50
            start_y = 70
            
            mx, my = pygame.mouse.get_pos()
            
            for i, g in enumerate(available_games):
                row = i // cols
                col = i % cols
                
                x = start_x + col * (box_w + gap)
                y = start_y + row * (box_h + gap)
                
                # Hover detection
                rect = pygame.Rect(x, y, box_w, box_h)
                is_hovered = rect.collidepoint(mx, my)
                
                # Handle Click
                if is_hovered and pygame.mouse.get_pressed()[0]:
                    # Send selection and exit mode
                    send_action("GAME_SELECTED", game_id=g["game_id"])
                    game_select_mode = False
                    # Clear events to prevent double click issues
                    pygame.event.clear()
                    
                # Draw Box
                color = (50, 50, 50) if not is_hovered else (80, 80, 80)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (100, 100, 100), rect, 2) # Border
                
                # Draw Thumbnail if available
                thumbnail = g.get("thumbnail")
                if thumbnail:
                    # Calculate thumbnail area (leave room for title at bottom)
                    thumb_area_h = box_h - 30  # Reserve bottom 30px for title
                    thumb_area_w = box_w - 4   # Small padding
                    thumb_x = x + 2
                    thumb_y = y + 2
                    
                    # Get grid dimensions
                    grid_rows = len(thumbnail)
                    grid_cols = len(thumbnail[0]) if grid_rows > 0 else 0
                    
                    if grid_rows > 0 and grid_cols > 0:
                        # Calculate scale to fit
                        scale_x = thumb_area_w / grid_cols
                        scale_y = thumb_area_h / grid_rows
                        thumb_scale = min(scale_x, scale_y, 8)  # Max 8px per cell
                        
                        # Determine background color for this thumbnail
                        flat_grid = [c for row in thumbnail for c in row]
                        if flat_grid:
                            counter = collections.Counter(flat_grid)
                            bg_val, _ = counter.most_common(1)[0]
                            # Use base palette for thumbnails
                            thumb_palette = BASE_PALETTE.copy()
                            if bg_val != 0:
                                original_bg_color = thumb_palette.get(bg_val, (0,0,0))
                                thumb_palette[bg_val] = (0, 0, 0)
                                thumb_palette[0] = original_bg_color
                        else:
                            thumb_palette = BASE_PALETTE.copy()
                        
                        # Draw thumbnail grid (no gridlines, solid seamless fill)
                        # Calculate total thumbnail size
                        total_thumb_w = grid_cols * thumb_scale
                        total_thumb_h = grid_rows * thumb_scale
                        
                        # Center thumbnail in available space
                        thumb_offset_x = (thumb_area_w - total_thumb_w) // 2
                        thumb_offset_y = (thumb_area_h - total_thumb_h) // 2
                        
                        for r in range(grid_rows):
                            for c in range(grid_cols):
                                val = thumbnail[r][c]
                                color_val = thumb_palette.get(val, (50, 50, 50))
                                
                                # Calculate seamless cell boundaries
                                cell_x = int(thumb_x + thumb_offset_x + c * thumb_scale)
                                cell_y = int(thumb_y + thumb_offset_y + r * thumb_scale)
                                
                                # Calculate width/height to next cell to ensure no gaps
                                next_cell_x = int(thumb_x + thumb_offset_x + (c + 1) * thumb_scale)
                                next_cell_y = int(thumb_y + thumb_offset_y + (r + 1) * thumb_scale)
                                
                                cell_w = next_cell_x - cell_x
                                cell_h = next_cell_y - cell_y
                                
                                # Only draw if within bounds
                                if cell_x < x + box_w - 2 and cell_y < y + box_h - 30:
                                    # Draw filled rectangle (no border, seamless)
                                    pygame.draw.rect(screen, color_val, (cell_x, cell_y, cell_w, cell_h))
                
                # Draw Title at bottom
                # Pink background for text like in the image
                g_title = g.get("title", g["game_id"][:4].upper())
                
                text_surf = title_font.render(g_title, True, (0, 0, 0)) # Black text
                text_rect = text_surf.get_rect(center=(x + box_w // 2, y + box_h - 15))
                
                # Pink label bg
                label_padding = 5
                label_bg_rect = text_rect.inflate(label_padding * 2, label_padding * 2)
                pygame.draw.rect(screen, (255, 105, 180), label_bg_rect) # Hot Pink
                screen.blit(text_surf, text_rect)

        else:
            # Standard Agent View
            
            # Draw Sidebar
            # Draw separator line
            pygame.draw.line(screen, BORDER_COLOR, (GRID_WIDTH, 0), (GRID_WIDTH, WINDOW_HEIGHT), 2)
            
            # Draw Text Info
        y_offset = 20
        def draw_text(text, font_obj, color=TEXT_COLOR):
            nonlocal y_offset
            surf = font_obj.render(text, True, color)
            screen.blit(surf, (GRID_WIDTH + 20, y_offset))
            y_offset += 30

        draw_text(f"Game: {game_id}", title_font)
        draw_text(f"Score: {score}", font)
        draw_text(f"State: {state}", font) # Added back state/steps display
        
        if waiting_for_server:
            draw_text("Status: Processing...", font, (255, 255, 0))
        else:
            draw_text("Status: Ready", font, (0, 255, 0))
            
        if last_action_info:
            aid = last_action_info.get("id")
            # Prioritize name sent by environment (contains symbols)
            aname = last_action_info.get("name")
            if not aname:
                aname = ACTION_NAMES.get(aid, str(aid))
            
            # Add details for clicks
            if aid == 6:
                adata = last_action_info.get("data", {})
                if adata and "x" in adata and "y" in adata:
                    if "(" not in aname:
                        aname += f" ({adata['x']}, {adata['y']})"
            
            draw_text(f"Last Action: {aname}", font, (0, 200, 255))
            
            # Draw Bottom-Left Overlay
            overlay_surf = overlay_font.render(aname, True, (0, 255, 255))
            # Shadow
            shadow_surf = overlay_font.render(aname, True, (0, 0, 0))
            screen.blit(shadow_surf, (22, WINDOW_HEIGHT - 48))
            screen.blit(overlay_surf, (20, WINDOW_HEIGHT - 50))
            
        y_offset += 20
        
        draw_text("CONTROLS:", title_font)
        draw_text("Arrows: Move (Action 1-4)", font)
        draw_text("Space: Use (Action 5)", font)
        draw_text("Click: Place/Interact (Action 6)", font)
        draw_text("Enter: Confirm (Action 7)", font)
        draw_text("R: Reset Level", font)
        draw_text("H: Toggle Heatmap", font)
        draw_text("Q: Quit", font)
        
        # --- DRAW GRAPHS ---
        # Draw at bottom of sidebar
        graph_start_y = y_offset + 20
        graph_h = 60
        graph_w = SIDEBAR_WIDTH - 40
        graph_x = GRID_WIDTH + 20
        
        # Move graphs up if they go off screen
        # 3 graphs * (60 + 20) = 240px
        # Slider = 40px
        # Text = ~200px
        # Total = 480px. Should fit in 800px.
        
        def draw_graph(title, data, y_pos, color, y_min=None, y_max=None):
            if not data: return
            
            # Title
            t_surf = font.render(title, True, color)
            screen.blit(t_surf, (graph_x, y_pos))
            
            # Box
            rect = pygame.Rect(graph_x, y_pos + 20, graph_w, graph_h)
            pygame.draw.rect(screen, (30, 30, 30), rect)
            pygame.draw.rect(screen, (100, 100, 100), rect, 1)
            
            if len(data) < 2: return
            
            # Scale
            vals = data
            min_v = min(vals) if y_min is None else y_min
            max_v = max(vals) if y_max is None else y_max
            
            if max_v == min_v: max_v += 1e-6
            
            points = []
            for i, v in enumerate(vals):
                px = graph_x + (i / (MAX_HISTORY - 1)) * graph_w 
                # Invert Y (pygame 0 is top)
                norm_v = (v - min_v) / (max_v - min_v)
                py = (y_pos + 20 + graph_h) - (norm_v * graph_h)
                points.append((px, py))
            
            if len(points) > 1:
                pygame.draw.lines(screen, color, False, points, 2)
                
            # Draw current value text
            curr = vals[-1]
            v_surf = font.render(f"{curr:.2f}", True, color)
            screen.blit(v_surf, (graph_x + graph_w - v_surf.get_width(), y_pos))

        draw_graph("Dopamine Level (AI)", metrics_history["dopamine"], graph_start_y, (0, 255, 255), y_min=0.0, y_max=1.0)
        draw_graph("Dopamine Level (Human)", metrics_history["manual_dopamine"], graph_start_y + 90, (255, 100, 100), y_min=0.0, y_max=1.0)
        draw_graph("Action Urge (Trigger)", metrics_history["trigger"], graph_start_y + 180, (255, 255, 0), y_min=-1.0, y_max=1.0)
        draw_graph("Plan Confidence", metrics_history["confidence"], graph_start_y + 270, (255, 0, 255), y_min=0.0, y_max=1.0)
        draw_graph("Avg Reward", metrics_history["reward"], graph_start_y + 360, (0, 255, 0))
        
        # --- DRAW SLIDER (Fixed Position) ---
        # Ensure it's below the graphs but ON SCREEN
        min_slider_y = graph_start_y + 450
        # Clamp to ensure visibility even if overlapping
        slider_y = min(WINDOW_HEIGHT - 60, max(WINDOW_HEIGHT - 80, min_slider_y))
        
        slider_x = GRID_WIDTH + 20
        slider_w = SIDEBAR_WIDTH - 40
        slider_h = 20
        
        # Label
        s_text = font.render(f"Speed Delay: {speed_val:.2f}s", True, (200, 200, 200))
        screen.blit(s_text, (slider_x, slider_y - 25))
        
        # Bar
        pygame.draw.rect(screen, (50, 50, 50), (slider_x, slider_y, slider_w, slider_h))
        
        # Handle
        handle_x = slider_x + int(speed_val * slider_w)
        handle_rect = pygame.Rect(handle_x - 5, slider_y - 5, 10, slider_h + 10)
        pygame.draw.rect(screen, (200, 200, 200), handle_rect)
        
        # --- DRAW DOPAMINE BUTTON ---
        btn_x = slider_x
        btn_y = slider_y - 50
        btn_w = slider_w
        btn_h = 40
        
        btn_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        # Color changes if active
        btn_color = (255, 50, 50) if holding_d_key else (100, 0, 0)
        pygame.draw.rect(screen, btn_color, btn_rect)
        pygame.draw.rect(screen, (255, 255, 255), btn_rect, 2)
        
        btn_text = font.render("HOLD FOR DOPAMINE (D)", True, (255, 255, 255))
        text_rect = btn_text.get_rect(center=btn_rect.center)
        screen.blit(btn_text, text_rect)

        # Draw Game Grid
        if current_grids and len(current_grids) > 0:
            current_grid = current_grids[current_grid_idx]
            rows = len(current_grid)
            cols = len(current_grid[0])
            
            # Draw black background for grid area to clear any artifacts
            pygame.draw.rect(screen, (0, 0, 0), (0, 0, GRID_WIDTH, WINDOW_HEIGHT))
            
            for r in range(rows):
                for c in range(cols):
                    val = current_grid[r][c]
                    color = PALETTE.get(val, (50, 50, 50))
                    
                    rect = (c * SCALE_FACTOR, r * SCALE_FACTOR, SCALE_FACTOR, SCALE_FACTOR)
                    pygame.draw.rect(screen, color, rect)
                    # Optional grid lines
                    # pygame.draw.rect(screen, GRID_LINE_COLOR, rect, 1)

            # Draw Heatmap Overlay
            if show_heatmap and current_attention_map:
                # 1. Determine dimensions
                hm_rows = len(current_attention_map)
                hm_cols = len(current_attention_map[0]) if hm_rows > 0 else 0
                
                if hm_rows > 0 and hm_cols > 0:
                    # Create Surface
                    hm_surf = pygame.Surface((hm_cols, hm_rows), flags=pygame.SRCALPHA)
                    
                    # Fill Surface
                    for r in range(hm_rows):
                        row_data = current_attention_map[r]
                        for c in range(hm_cols):
                            val = row_data[c] # 0.0 to 1.0
                            
                            # Visualization: Heatmap
                            # Remove cutoff to debug visibility
                            # Boost alpha
                            if val < 0.05:
                                alpha = 0
                            else:
                                # Non-linear alpha for visibility
                                # 0.1 -> 50
                                # 1.0 -> 200
                                alpha = int(50 + val * 150)
                                alpha = min(255, alpha)
                            
                            # Use Bright Red/Orange
                            hm_surf.set_at((c, r), (255, 50, 0, alpha))
                    
                    # Scale logic ...
                    target_w = hm_cols * SCALE_FACTOR
                    target_h = hm_rows * SCALE_FACTOR
                    
                    full_hm = pygame.transform.scale(hm_surf, (target_w, target_h))
                    
                    visible_w = cols * SCALE_FACTOR
                    visible_h = rows * SCALE_FACTOR
                    
                    blit_w = min(target_w, visible_w)
                    blit_h = min(target_h, visible_h)
                    
                    screen.blit(full_hm, (0, 0), (0, 0, blit_w, blit_h))
            
            # Draw Object Interest Overlay (Yellow Outlines)
            if show_heatmap and current_objects and current_attention_map:
                # Calculate aggregated score for each object
                obj_scores = []
                for obj in current_objects:
                    score_sum = 0
                    count = 0
                    for r, c in obj:
                        if r < len(current_attention_map) and c < len(current_attention_map[0]):
                            score_sum += current_attention_map[r][c]
                            count += 1
                    
                    avg_score = score_sum / count if count > 0 else 0
                    obj_scores.append((avg_score, obj))
                
                # Sort by score
                obj_scores.sort(key=lambda x: x[0], reverse=True)
                
                # Draw top 3 objects
                for i, (score, obj) in enumerate(obj_scores[:3]):
                    if score < 0.1: continue # Ignore low interest
                    
                    # Draw Outline
                    # Naive approach: draw rect for each pixel, or convex hull
                    # Let's draw rects for each pixel for simplicity
                    
                    # Color based on rank
                    if i == 0: color = (0, 255, 0) # Green (Top)
                    elif i == 1: color = (255, 255, 0) # Yellow
                    else: color = (255, 165, 0) # Orange
                    
                    for r, c in obj:
                        rect = (c * SCALE_FACTOR, r * SCALE_FACTOR, SCALE_FACTOR, SCALE_FACTOR)
                        pygame.draw.rect(screen, color, rect, 2)
                    
                    # Label the object with its score
                    if obj:
                         # Find center
                        rs = [p[0] for p in obj]
                        cs = [p[1] for p in obj]
                        cr = sum(rs) / len(rs)
                        cc = sum(cs) / len(cs)
                        
                        txt = font.render(f"{score:.2f}", True, (255, 255, 255))
                        screen.blit(txt, (cc * SCALE_FACTOR, cr * SCALE_FACTOR))

            # Highlight mouse hover (REMOVED - duplicate visual)
            # mx, my = pygame.mouse.get_pos()
            # if mx < GRID_WIDTH: ...
            
            # Visualize last click with subtle ripple
            if last_click_pos:
                lx, ly = last_click_pos
                time_diff = pygame.time.get_ticks() - last_click_time
                if time_diff < CLICK_VIS_DURATION:
                    # Calculate center in screen coords
                    center_x = int(round(lx)) * SCALE_FACTOR + (SCALE_FACTOR // 2)
                    center_y = int(round(ly)) * SCALE_FACTOR + (SCALE_FACTOR // 2)
                    
                    # Growing ripple effect
                    # Radius grows from 0 to SCALE_FACTOR
                    progress = time_diff / CLICK_VIS_DURATION
                    radius = int(SCALE_FACTOR * progress * 1.5)
                    
                    # Fade out alpha
                    alpha = int(255 * (1.0 - progress))
                    
                    # Draw Circle
                    if radius > 0:
                        s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                        # Draw circle on surface
                        pygame.draw.circle(s, (255, 255, 255, alpha), (radius, radius), radius, 2)
                        # Blit centered
                        screen.blit(s, (center_x - radius, center_y - radius))

            # Draw Virtual Cursor
            if visual_cursor_pos:
                cx, cy = visual_cursor_pos
                
                # Visual Highlight of the "Active" cell (Rounded)
                active_cx = int(round(cx))
                active_cy = int(round(cy))
                
                # Draw Active Cell Highlight (Blue tint)
                if current_grids:
                    grid = current_grids[0]
                    rows = len(grid)
                    cols = len(grid[0])
                    if 0 <= active_cx < cols and 0 <= active_cy < rows:
                         rect = (active_cx * SCALE_FACTOR, active_cy * SCALE_FACTOR, SCALE_FACTOR, SCALE_FACTOR)
                         s = pygame.Surface((SCALE_FACTOR, SCALE_FACTOR), pygame.SRCALPHA)
                         s.fill((0, 100, 255, 50)) # Transparent blue
                         screen.blit(s, rect)
                         pygame.draw.rect(screen, (0, 100, 255), rect, 2)
                
                # Draw Actual Cursor (Proper Mouse Sprite)
                # Position is the floating point location
                # We draw the tip of the cursor at the center of the 'virtual' position
                # Or should the center of the crosshair be the tip?
                # Let's align the TIP of the arrow to (px, py) to be precise.
                
                cursor_tip_x = cx * SCALE_FACTOR + (SCALE_FACTOR / 2)
                cursor_tip_y = cy * SCALE_FACTOR + (SCALE_FACTOR / 2)
                
                # Draw Arrow Cursor
                c_size = 24 # Slightly larger constant size
                
                # Classic Arrow Shape
                arrow_pts = [
                    (0, 0),
                    (0, c_size),
                    (c_size * 0.25, c_size * 0.75),
                    (c_size * 0.5, c_size * 1.2),
                    (c_size * 0.7, c_size * 1.1),
                    (c_size * 0.45, c_size * 0.65),
                    (c_size * 0.75, c_size * 0.65)
                ]
                
                # Transform to screen space
                screen_pts = [(cursor_tip_x + p[0], cursor_tip_y + p[1]) for p in arrow_pts]
                
                # Shadow
                shadow_pts = [(p[0]+1, p[1]+1) for p in screen_pts]
                pygame.draw.polygon(screen, (0, 0, 0), shadow_pts)
                
                # Fill White
                pygame.draw.polygon(screen, (255, 255, 255), screen_pts)
                
                # Border Black
                pygame.draw.polygon(screen, (0, 0, 0), screen_pts, 1)
                
            # Update Keyboard Symbol State
            if last_action_info:
                aname = last_action_info.get("name", "")
                symbol = ""
                if "↑" in aname: symbol = "↑"
                elif "↓" in aname: symbol = "↓"
                elif "←" in aname: symbol = "←"
                elif "→" in aname: symbol = "→"
                elif "␣" in aname: symbol = "␣"
                elif "↵" in aname: symbol = "↵"
                
                if symbol:
                    key_activations[symbol] = pygame.time.get_ticks()

            # Draw Visual Keyboard (Bottom Left)
            kb_base_x = 30
            kb_base_y = WINDOW_HEIGHT - 150
            
            # Layout: (Symbol, Label, rel_x, rel_y, w, h)
            keys = [
                ("↑", "↑", 50, 0, 40, 40),
                ("←", "←", 5, 45, 40, 40),
                ("↓", "↓", 50, 45, 40, 40),
                ("→", "→", 95, 45, 40, 40),
                ("↵", "ENT", 145, 45, 60, 40),
                ("␣", "", 5, 90, 130, 30) # Space bar
            ]
            
            for k_sym, k_label, kx, ky, kw, kh in keys:
                # Calculate Brightness based on activation
                last_act = key_activations.get(k_sym, 0)
                time_diff = pygame.time.get_ticks() - last_act
                
                brightness = 50 # Default dark grey
                text_color = (150, 150, 150) # Dim text
                
                if time_diff < KEY_FADE_DURATION:
                    ratio = 1.0 - (time_diff / KEY_FADE_DURATION) # 1.0 -> 0.0
                    brightness = int(50 + ratio * 205) # 50 -> 255
                    text_color = (255, 255, 255) if ratio > 0.5 else (200, 200, 200)
                
                # Draw Key Rect
                rect = pygame.Rect(kb_base_x + kx, kb_base_y + ky, kw, kh)
                pygame.draw.rect(screen, (brightness, brightness, brightness), rect, border_radius=5)
                pygame.draw.rect(screen, (100, 100, 100), rect, 2, border_radius=5) # Border
                
                # Draw Label
                if k_label:
                    # Use title_font for arrows/ENT
                    label_surf = title_font.render(k_label, True, (0, 0, 0) if brightness > 150 else (255, 255, 255))
                    label_rect = label_surf.get_rect(center=rect.center)
                    screen.blit(label_surf, label_rect)
                
        else:
            # Placeholder text
            text = font.render("Waiting for game state...", True, (100, 100, 100))
            screen.blit(text, (GRID_WIDTH // 2 - 100, WINDOW_HEIGHT // 2))

        pygame.display.flip()
        clock.tick(60) # 60 FPS for smoother feedback

    pygame.quit()

if __name__ == "__main__":
    main()
