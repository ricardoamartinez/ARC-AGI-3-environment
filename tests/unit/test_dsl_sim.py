import numpy as np
import pytest

from agents.ppo.bayes.dsl import Action, GridCursorState
from agents.ppo.bayes.rules import IdentityRule, MoveColorOnKeyRule, PaintOnClickRule, ToggleOnClickRule
from agents.ppo.bayes.simulator import mismatch_fraction, transition_log_likelihood


@pytest.mark.unit
class TestDslSimulator:
    def test_identity_rule_keeps_grid(self):
        grid = np.zeros((4, 4), dtype=np.uint8)
        st = GridCursorState(grid=grid, cursor_y=1, cursor_x=2)
        nxt = IdentityRule().apply(st, Action.noop())
        assert np.array_equal(nxt.grid, grid)
        assert (nxt.cursor_y, nxt.cursor_x) == (1, 2)

    def test_paint_on_click(self):
        grid = np.zeros((4, 4), dtype=np.uint8)
        st = GridCursorState(grid=grid, cursor_y=1, cursor_x=2)
        rule = PaintOnClickRule(color=7)
        nxt = rule.apply(st, Action.click())
        assert nxt.grid[1, 2] == 7

    def test_toggle_on_click(self):
        grid = np.zeros((4, 4), dtype=np.uint8)
        grid[1, 2] = 3
        st = GridCursorState(grid=grid, cursor_y=1, cursor_x=2)
        rule = ToggleOnClickRule(a=3, b=9)
        nxt = rule.apply(st, Action.click())
        assert nxt.grid[1, 2] == 9
        nxt2 = rule.apply(nxt, Action.click())
        assert nxt2.grid[1, 2] == 3

    def test_move_color_on_key(self):
        grid = np.zeros((4, 4), dtype=np.uint8)
        grid[2, 1] = 5
        grid[2, 2] = 5
        st = GridCursorState(grid=grid, cursor_y=0, cursor_x=0)
        rule = MoveColorOnKeyRule(color=5, background=0)
        nxt = rule.apply(st, Action.keypress("RIGHT"))
        assert nxt.grid[2, 2] == 5
        assert nxt.grid[2, 3] == 5
        assert nxt.grid[2, 1] == 0

    def test_mismatch_fraction_and_ll(self):
        a = np.zeros((2, 2), dtype=np.uint8)
        b = a.copy()
        b[0, 0] = 1
        frac = mismatch_fraction(a, b)
        assert frac == 0.25

        st0 = GridCursorState(grid=a, cursor_y=0, cursor_x=0)
        st1 = GridCursorState(grid=b, cursor_y=0, cursor_x=0)
        ll = transition_log_likelihood(IdentityRule(), st0, Action.noop(), st1, beta=100.0)
        assert ll < 0.0



