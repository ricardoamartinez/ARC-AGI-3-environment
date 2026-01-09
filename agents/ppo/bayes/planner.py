from __future__ import annotations

from dataclasses import dataclass
from itertools import islice, product
from typing import Optional, Sequence

import numpy as np

from ...structs import GameAction
from ..control.actions import ActionProcessor
from .belief import BeliefFilter, Particle
from .dsl import Action, ActionKind, GridCursorState
from .event_boundary import HeuristicEventBoundaryDetector, heuristic_boundary_probability
from .simulator import mismatch_fraction


def _action_space_around_cursor(
    state: GridCursorState,
    *,
    step: int = 4,
    max_clicks: int = 9,
) -> list[Action]:
    """Small, cheap candidate action set."""
    h, w = state.grid.shape
    cy, cx = int(state.cursor_y), int(state.cursor_x)

    pts: list[tuple[int, int]] = [(cx, cy)]
    for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
        pts.append((cx + dx, cy + dy))
    # corners (occasionally useful)
    pts.extend([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])

    uniq: list[tuple[int, int]] = []
    for x, y in pts:
        xx = int(np.clip(x, 0, w - 1))
        yy = int(np.clip(y, 0, h - 1))
        if (xx, yy) not in uniq:
            uniq.append((xx, yy))
    uniq = uniq[: int(max_clicks)]

    actions: list[Action] = [Action.noop()]
    actions.extend([Action.keypress(k) for k in ("UP", "DOWN", "LEFT", "RIGHT")])
    actions.extend([Action.click(x=x, y=y) for (x, y) in uniq])
    return actions


def disagreement_score(particles: Sequence[tuple[Particle, np.ndarray]]) -> float:
    """Weighted pairwise mismatch among predicted grids (proxy for information gain)."""
    if len(particles) < 2:
        return 0.0
    dis = 0.0
    for i in range(len(particles)):
        pi = float(np.exp(particles[i][0].log_w))
        gi = particles[i][1]
        for j in range(i + 1, len(particles)):
            pj = float(np.exp(particles[j][0].log_w))
            gj = particles[j][1]
            dis += pi * pj * mismatch_fraction(gi, gj)
    return float(dis)


@dataclass(frozen=True, slots=True)
class PlanResult:
    action: Action
    score: float
    expected_boundary: float
    disagreement: float
    seq: tuple[Action, ...]


class BeliefPlanner:
    """Short-horizon planner under rule uncertainty."""

    def __init__(
        self,
        belief: BeliefFilter,
        boundary: Optional[HeuristicEventBoundaryDetector] = None,
        *,
        horizon: int = 2,
        max_sequences: int = 512,
        top_k_rules: int = 8,
        w_progress: float = 1.0,
        w_info: float = 1.0,
    ) -> None:
        self.belief = belief
        self.boundary = boundary or HeuristicEventBoundaryDetector()
        self.horizon = int(max(1, horizon))
        self.max_sequences = int(max(1, max_sequences))
        self.top_k_rules = int(max(1, top_k_rules))
        self.w_progress = float(w_progress)
        self.w_info = float(w_info)
        self._action_processor = ActionProcessor()

    def plan(self, state: GridCursorState, *, candidate_actions: Optional[Sequence[Action]] = None) -> PlanResult:
        actions = list(candidate_actions) if candidate_actions is not None else _action_space_around_cursor(state)
        particles = self.belief.topk(self.top_k_rules)
        init_grid = getattr(self.boundary, "_initial_grid", None)

        best: Optional[PlanResult] = None

        seq_iter = islice(product(actions, repeat=self.horizon), self.max_sequences)
        for seq in seq_iter:
            # per-rule rollout state
            rule_states: list[GridCursorState] = [state for _ in particles]
            total_score = 0.0

            for a in seq:
                # Predict next grids under each rule hypothesis
                preds: list[tuple[Particle, np.ndarray]] = []
                next_states: list[GridCursorState] = []
                for i, p in enumerate(particles):
                    nxt = p.rule.apply(rule_states[i], a)
                    next_states.append(nxt)
                    preds.append((p, nxt.grid))

                # Expected boundary (progress)
                exp_b = 0.0
                for i, p in enumerate(particles):
                    w = float(np.exp(p.log_w))
                    exp_b += w * heuristic_boundary_probability(
                        rule_states[i],
                        next_states[i],
                        initial_grid=init_grid,
                        background=int(getattr(self.boundary, "background", 0)),
                        delta_hi=float(getattr(self.boundary, "delta_hi", 0.25)),
                        reset_bonus=float(getattr(self.boundary, "reset_bonus", 2.0)),
                        bias=float(getattr(self.boundary, "bias", 3.0)),
                    )

                dis = disagreement_score(preds)
                total_score += self.w_progress * float(exp_b) + self.w_info * float(dis)

                rule_states = next_states

            res = PlanResult(
                action=seq[0],
                score=float(total_score),
                expected_boundary=float(exp_b),
                disagreement=float(dis),
                seq=tuple(seq),
            )
            if best is None or res.score > best.score:
                best = res

        if best is None:
            return PlanResult(action=Action.noop(), score=0.0, expected_boundary=0.0, disagreement=0.0, seq=(Action.noop(),))
        return best

    def to_game_action(self, action: Action, state: GridCursorState, *, game_id: str) -> Optional[GameAction]:
        """Convert an abstract action into a GameAction using the existing ActionProcessor."""
        # Determine the cursor position used for actions that require coordinates.
        cx = int(state.cursor_x)
        cy = int(state.cursor_y)
        if action.kind in (ActionKind.SET_CURSOR, ActionKind.CLICK) and action.x is not None and action.y is not None:
            cx = int(action.x)
            cy = int(action.y)

        final_action_idx = -1
        if action.kind == ActionKind.CLICK:
            final_action_idx = 0  # click bucket (<=3)
        elif action.kind == ActionKind.KEY and action.key:
            k = action.key.upper()
            final_action_idx = {"UP": 4, "DOWN": 5, "LEFT": 6, "RIGHT": 7, "SPACE": 8, "ENTER": 9}.get(k, -1)

        return self._action_processor.get_game_action(final_action_idx, cx, cy, game_id)



