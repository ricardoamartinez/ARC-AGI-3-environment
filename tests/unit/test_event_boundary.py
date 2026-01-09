import numpy as np
import pytest

from agents.ppo.bayes.dsl import GridCursorState
from agents.ppo.bayes.event_boundary import HeuristicEventBoundaryDetector


@pytest.mark.unit
class TestEventBoundary:
    def test_stable_state_low_boundary_prob(self):
        grid = np.zeros((8, 8), dtype=np.uint8)
        st = GridCursorState(grid=grid, cursor_y=0, cursor_x=0)
        det = HeuristicEventBoundaryDetector(background=0)
        det.reset(st)
        assert det.observe(st) == 0.0  # first observation after reset
        p = det.observe(st)
        assert p < 0.2

    def test_large_visual_change_high_boundary_prob(self):
        grid0 = np.zeros((8, 8), dtype=np.uint8)
        grid1 = np.full((8, 8), 7, dtype=np.uint8)
        st0 = GridCursorState(grid=grid0, cursor_y=0, cursor_x=0)
        st1 = GridCursorState(grid=grid1, cursor_y=0, cursor_x=0)
        det = HeuristicEventBoundaryDetector(background=0)
        det.reset(st0)
        det.observe(st0)
        p = det.observe(st1)
        assert p > 0.6

    def test_reset_like_transition_detected(self):
        init = np.zeros((8, 8), dtype=np.uint8)
        mid = init.copy()
        mid[3:5, 3:5] = 2
        st_init = GridCursorState(grid=init, cursor_y=0, cursor_x=0)
        st_mid = GridCursorState(grid=mid, cursor_y=0, cursor_x=0)
        det = HeuristicEventBoundaryDetector(background=0, reset_bonus=4.0)
        det.reset(st_init)
        det.observe(st_init)
        det.observe(st_mid)
        p_back = det.observe(st_init)
        assert p_back > 0.4



