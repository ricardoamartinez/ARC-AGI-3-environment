from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

@dataclass
class GameState:
    # Game Data
    game_id: str = "Waiting..."
    score: int = 0
    state: str = "Starting..."
    current_thought: str = ""
    waiting_for_server: bool = False
    
    # Grids & Animation
    current_grids: List[List[List[int]]] = field(default_factory=list)
    current_grid_idx: int = 0
    animation_timer: int = 0
    
    # Selection Mode
    game_select_mode: bool = False
    available_games: List[Dict[str, Any]] = field(default_factory=list)
    
    # Heatmaps
    current_attention_map: Optional[List[List[float]]] = None
    current_maps: Dict[str, List[List[float]]] = field(default_factory=dict)
    current_objects: List[List[Tuple[int, int]]] = field(default_factory=list)
    show_heatmap: bool = False
    selected_heatmap_mode: str = "attention"
    
    # Interaction State
    last_action_info: Optional[Dict[str, Any]] = None
    last_click_pos: Optional[Tuple[int, int]] = None
    last_click_time: int = 0
    
    # Cursor
    cursor_pos: Optional[Tuple[int, int]] = None
    visual_cursor_pos: Optional[List[float]] = None
    spatial_goal_pos: Optional[Tuple[int, int]] = None
    
    # Metrics
    metrics_history: Dict[str, List[float]] = field(default_factory=lambda: {
        "reward": [],
        "dopamine": [],
        "confidence": [],
        "manual_dopamine": [],
        "pain": [],
        "trigger": []
    })
    
    # Controls
    speed_val: float = 0.0
    manual_dopamine_val: float = 0.0
    manual_pain_val: float = 0.0
    dragging_slider: bool = False
    holding_d_key: bool = False
    holding_p_key: bool = False
    
    # Keyboard Visuals
    key_activations: Dict[str, int] = field(default_factory=dict)

    running: bool = True

