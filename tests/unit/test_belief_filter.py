import numpy as np
import pytest

from agents.ppo.bayes.belief import BeliefFilter, RuleProposalNetwork
from agents.ppo.bayes.dsl import Action, GridCursorState
from agents.ppo.bayes.rules import IdentityRule, MoveColorOnKeyRule, PaintOnClickRule


@pytest.mark.unit
class TestBeliefFilter:
    def test_belief_concentrates_on_paint_rule(self):
        grid0 = np.zeros((4, 4), dtype=np.uint8)
        st0 = GridCursorState(grid=grid0, cursor_y=1, cursor_x=2)
        grid1 = grid0.copy()
        grid1[1, 2] = 7
        st1 = GridCursorState(grid=grid1, cursor_y=1, cursor_x=2)

        rules = [IdentityRule(), PaintOnClickRule(color=7), PaintOnClickRule(color=1)]
        bf = BeliefFilter(rules, beam_size=3, beta=200.0, proposal_net=None)
        bf.update(st0, Action.click(), st1)
        best = bf.best()
        assert best.rule.signature().startswith("paint_on_click")
        assert "color=7" in best.rule.signature()

    def test_belief_concentrates_on_move_color_rule(self):
        grid0 = np.zeros((4, 4), dtype=np.uint8)
        grid0[2, 1] = 5
        grid0[2, 2] = 5
        st0 = GridCursorState(grid=grid0, cursor_y=0, cursor_x=0)

        grid1 = np.zeros((4, 4), dtype=np.uint8)
        grid1[2, 2] = 5
        grid1[2, 3] = 5
        st1 = GridCursorState(grid=grid1, cursor_y=0, cursor_x=0)

        rules = [IdentityRule(), MoveColorOnKeyRule(color=5, background=0), PaintOnClickRule(color=5)]
        bf = BeliefFilter(rules, beam_size=3, beta=200.0, proposal_net=None)
        bf.update(st0, Action.keypress("RIGHT"), st1)
        best = bf.best()
        assert best.rule.signature().startswith("move_color_on_key")

    def test_proposal_network_scores_shape(self):
        net = RuleProposalNetwork(hidden_dim=32)
        # two-step dummy feature sequence (B=1,T=2,D)
        feat_seq = np.zeros((1, 2, 2 + 4 + 4), dtype=np.float32)
        # three rules
        rule_specs = np.array([[0, 0, 0], [1, 7, 0], [3, 5, 0]], dtype=np.int64)
        import torch

        scores = net.score_rules(torch.from_numpy(feat_seq), torch.from_numpy(rule_specs))
        assert tuple(scores.shape) == (1, 3)



