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
    WINDOW_HEIGHT = 800
    GRID_WIDTH = 800 # Default start
    WINDOW_WIDTH = GRID_WIDTH + SIDEBAR_WIDTH
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("ARC-AGI-3 Agent")
    
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)
    title_font = pygame.font.SysFont("Arial", 20, bold=True)
    
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
    
    last_action_info = None
    last_click_pos = None
    last_click_time = 0
    CLICK_VIS_DURATION = 500 # ms
    
    cursor_pos = None # (x, y)
    
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
                elif event.key == pygame.K_q:
                    send_action("QUIT")
                    running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    x, y = event.pos
                    # Check if click is in grid area
                    if x < GRID_WIDTH:
                        grid_x = x // SCALE_FACTOR
                        grid_y = y // SCALE_FACTOR
                        # Assume max 63 for safety, but grid bounds are better
                        max_x = 63
                        max_y = 63
                        if current_grids and len(current_grids) > 0:
                             # Use current frame dimensions
                            curr = current_grids[current_grid_idx]
                            max_y = len(curr)
                            max_x = len(curr[0])
                            
                        if 0 <= grid_x < max_x and 0 <= grid_y < max_y:
                            waiting_for_server = True
                            send_action("ACTION6", data={"game_id": game_id, "x": grid_x, "y": grid_y})

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
                        # If it's a click (ACTION6 or has x/y)
                        aid = last_action_info.get("id")
                        adata = last_action_info.get("data", {})
                        
                        if aid == 6 or (adata and "x" in adata and "y" in adata):
                            last_click_pos = (adata.get("x", 0), adata.get("y", 0))
                            last_click_time = pygame.time.get_ticks()
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
                    cursor_pos = (data["cursor"]["x"], data["cursor"]["y"])

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
            aname = ACTION_NAMES.get(aid, last_action_info.get("name", str(aid)))
            
            # Add details for clicks
            if aid == 6:
                adata = last_action_info.get("data", {})
                if adata and "x" in adata and "y" in adata:
                    aname += f" ({adata['x']}, {adata['y']})"
            
            draw_text(f"Last Action: {aname}", font, (0, 200, 255))
            
        y_offset += 20
        
        draw_text("CONTROLS:", title_font)
        draw_text("Arrows: Move (Action 1-4)", font)
        draw_text("Space: Use (Action 5)", font)
        draw_text("Click: Place/Interact (Action 6)", font)
        draw_text("Enter: Confirm (Action 7)", font)
        draw_text("R: Reset Level", font)
        draw_text("Q: Quit", font)

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
                    
            # Highlight mouse hover
            mx, my = pygame.mouse.get_pos()
            if mx < GRID_WIDTH:
                hx = mx // SCALE_FACTOR
                hy = my // SCALE_FACTOR
                if 0 <= hx < cols and 0 <= hy < rows:
                    h_rect = (hx * SCALE_FACTOR, hy * SCALE_FACTOR, SCALE_FACTOR, SCALE_FACTOR)
                    pygame.draw.rect(screen, (255, 255, 255), h_rect, 2)
            
            # Visualize last click with mouse cursor sprite
            if last_click_pos:
                lx, ly = last_click_pos
                time_diff = pygame.time.get_ticks() - last_click_time
                if time_diff < CLICK_VIS_DURATION:
                    # Calculate position (cursor hot spot at click location)
                    hot_x = int(lx * SCALE_FACTOR + SCALE_FACTOR // 2)
                    hot_y = int(ly * SCALE_FACTOR + SCALE_FACTOR // 2)
                    
                    # Cursor size (standard is 16x16, scale appropriately)
                    cursor_size = max(10, min(int(SCALE_FACTOR * 1.5), 24))
                    
                    # Standard Windows-style mouse cursor (arrow pointing up-left)
                    # Hot spot is at the tip (0, 0 relative)
                    # Define points for a simple arrow cursor
                    points = [
                        (0, 0),           # Tip
                        (0, cursor_size), # Bottom-left
                        (cursor_size * 0.3, cursor_size * 0.7), # Inner corner
                        (cursor_size * 0.5, cursor_size * 0.9), # Tail bottom-left
                        (cursor_size * 0.7, cursor_size * 0.7), # Tail bottom-right
                        (cursor_size * 0.5, cursor_size * 0.5), # Tail inner
                        (cursor_size * 0.7, cursor_size * 0.5), # Right edge
                    ]
                    
                    # Offset and rotate slightly to look natural (tilted left)
                    # Actually, simpler: just hardcode the tilted shape
                    tilted_points = [
                        (0, 0),
                        (0, cursor_size),
                        (cursor_size * 0.25, cursor_size * 0.75),
                        (cursor_size * 0.45, cursor_size * 1.2), # Tail extended
                        (cursor_size * 0.65, cursor_size * 1.1), # Tail width
                        (cursor_size * 0.45, cursor_size * 0.65),
                        (cursor_size * 0.75, cursor_size * 0.65)
                    ]
                    
                    # Adjust to screen coords
                    screen_points = [(hot_x + p[0], hot_y + p[1]) for p in tilted_points]
                    
                    # Draw black outline/shadow first
                    shadow_points = [(p[0] + 1, p[1] + 1) for p in screen_points]
                    pygame.draw.polygon(screen, (0, 0, 0), shadow_points)
                    
                    # Draw white fill
                    pygame.draw.polygon(screen, (255, 255, 255), screen_points)
                    
                    # Draw black border
                    pygame.draw.polygon(screen, (0, 0, 0), screen_points, 1)

            # Draw Virtual Cursor
            if cursor_pos:
                cx, cy = cursor_pos
                # Draw a hollow white box at cursor position
                rect = (cx * SCALE_FACTOR, cy * SCALE_FACTOR, SCALE_FACTOR, SCALE_FACTOR)
                # Outer white
                pygame.draw.rect(screen, (255, 255, 255), rect, 2)
                # Inner black for contrast
                inner_rect = (cx * SCALE_FACTOR + 2, cy * SCALE_FACTOR + 2, SCALE_FACTOR - 4, SCALE_FACTOR - 4)
                pygame.draw.rect(screen, (0, 0, 0), inner_rect, 1)
                    
        else:
            # Placeholder text
            text = font.render("Waiting for game state...", True, (100, 100, 100))
            screen.blit(text, (GRID_WIDTH // 2 - 100, WINDOW_HEIGHT // 2))

        pygame.display.flip()
        clock.tick(60) # 60 FPS for smoother feedback

    pygame.quit()

if __name__ == "__main__":
    main()
