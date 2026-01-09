import numpy as np
import pytest

from agents.ppo.bayes.belief import BeliefFilter
from agents.ppo.bayes.dsl import Action, GridCursorState
from agents.ppo.bayes.planner import BeliefPlanner
from agents.ppo.bayes.rules import PaintOnClickRule
from agents.structs import GameAction


@pytest.mark.unit
class TestBeliefPlanner:
    def test_planner_prefers_informative_click(self):
        grid = np.zeros((4, 4), dtype=np.uint8)
        st = GridCursorState(grid=grid, cursor_y=1, cursor_x=1)
        rules = [PaintOnClickRule(color=1), PaintOnClickRule(color=2)]
        belief = BeliefFilter(rules, beam_size=2, beta=50.0)
        planner = BeliefPlanner(belief, horizon=1, w_progress=0.0, w_info=10.0, top_k_rules=2, max_sequences=16)

        res = planner.plan(st, candidate_actions=[Action.noop(), Action.click()])
        assert res.action.kind.value == "click"

    def test_to_game_action_uses_action_processor(self):
        grid = np.zeros((4, 4), dtype=np.uint8)
        st = GridCursorState(grid=grid, cursor_y=1, cursor_x=1)
        belief = BeliefFilter([PaintOnClickRule(color=1)], beam_size=1)
        planner = BeliefPlanner(belief, horizon=1, max_sequences=1)
        ga = planner.to_game_action(Action.click(), st, game_id="test-game")
        assert ga is not None
        assert ga == GameAction.ACTION6



