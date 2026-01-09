import pygame

# Initialize pygame fonts safely (will need to be called after pygame.init)
# We will define font configurations here, but actual font objects need initialization
FONT_CONFIG = {
    "normal": ("Arial", 16),
    "title": ("Arial", 20, True),
    "overlay": ("Arial", 32, True),
    "small": ("Arial", 12)
}

# Dimensions
SCALE_FACTOR_DEFAULT = 1
SIDEBAR_WIDTH = 300
WINDOW_HEIGHT_DEFAULT = 1000
GRID_WIDTH_DEFAULT = 800
WINDOW_WIDTH_DEFAULT = GRID_WIDTH_DEFAULT + SIDEBAR_WIDTH

# Animation
ANIMATION_SPEED = 100 # ms per frame
CLICK_VIS_DURATION = 500 # ms
KEY_FADE_DURATION = 300

# Colors
BG_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
SIDEBAR_BG = (0, 0, 0)
BORDER_COLOR = (50, 50, 50)
GRID_LINE_COLOR = (30, 30, 30)

# ARC Palette (Original)
# BASE_PALETTE = {
#     0: (0, 0, 0),       # Black
#     1: (30, 144, 255),  # Blue
#     2: (255, 69, 0),    # Red
#     3: (50, 205, 50),   # Green
#     4: (255, 215, 0),   # Yellow
#     5: (169, 169, 169), # Gray
#     6: (255, 20, 147),  # Fuchsia
#     7: (255, 140, 0),   # Orange
#     8: (0, 255, 255),   # Cyan
#     9: (128, 0, 128),   # Maroon
#     10: (255, 255, 255), # White
#     11: (105, 105, 105), # Gray
#     12: (255, 255, 255), # Player
# }

# New Palette from Reasoning Agent
BASE_PALETTE = {
    0: (255, 255, 255),  # #FFFFFF
    1: (204, 204, 204),  # #CCCCCC
    2: (153, 153, 153),  # #999999
    3: (102, 102, 102),  # #666666
    4: (51, 51, 51),     # #333333
    5: (0, 0, 0),        # #000000
    6: (229, 58, 163),   # #E53AA3
    7: (255, 123, 204),  # #FF7BCC
    8: (249, 60, 49),    # #F93C31
    9: (30, 147, 255),   # #1E93FF
    10: (136, 216, 241), # #88D8F1
    11: (255, 220, 0),   # #FFDC00
    12: (255, 133, 27),  # #FF851B
    13: (146, 18, 49),   # #921231
    14: (79, 204, 48),   # #4FCC30
    15: (163, 86, 214),  # #A356D6
}

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

HEATMAP_COLORS = {
    "attention": (255, 100, 0),   # Orange/Red
    "pain": (255, 0, 0),          # Pure Red
    "visit": (0, 255, 255),       # Cyan
    "value": (0, 255, 0),         # Green
    "controllability": (0, 255, 120),  # Green-ish
    "surprise": (255, 180, 0),         # Orange
    "disagreement": (255, 0, 255),     # Magenta
    "obs_delta": (255, 255, 0),   # Yellow
    "obs_focus": (255, 0, 255),   # Magenta
    "obs_goal": (0, 100, 255),    # Blue
    "obs_vel_x": (150, 150, 255), # Light Blue
    "obs_vel_y": (150, 150, 255), # Light Blue
    "obs_pain": (200, 0, 0)       # Dark Red
}

HEATMAP_MODES = [
    "attention", "pain", "visit", "value",
    "controllability", "surprise", "disagreement",
    "obs_delta", "obs_focus", "obs_goal", 
    "obs_vel_x", "obs_vel_y", "obs_pain"
]

MAX_HISTORY = 200
