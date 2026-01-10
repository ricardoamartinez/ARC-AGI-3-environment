"""UI Constants for JEPA Agent visualization."""
import pygame

# Font configurations (actual font objects created after pygame.init)
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
ANIMATION_SPEED = 100  # ms per frame
CLICK_VIS_DURATION = 500  # ms
KEY_FADE_DURATION = 300

# Colors
BG_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
SIDEBAR_BG = (0, 0, 0)
BORDER_COLOR = (50, 50, 50)
GRID_LINE_COLOR = (30, 30, 30)

# ARC Palette
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
    "attention": (255, 100, 0),
    "pain": (255, 0, 0),
    "visit": (0, 255, 255),
    "value": (0, 255, 0),
    "controllability": (0, 255, 120),
    "surprise": (255, 180, 0),
    "disagreement": (255, 0, 255),
    "obs_delta": (255, 255, 0),
    "obs_focus": (255, 0, 255),
    "obs_goal": (0, 100, 255),
    "obs_vel_x": (150, 150, 255),
    "obs_vel_y": (150, 150, 255),
    "obs_pain": (200, 0, 0)
}

HEATMAP_MODES = [
    "attention", "pain", "visit", "value",
    "controllability", "surprise", "disagreement",
    "obs_delta", "obs_focus", "obs_goal", 
    "obs_vel_x", "obs_vel_y", "obs_pain"
]

MAX_HISTORY = 200
