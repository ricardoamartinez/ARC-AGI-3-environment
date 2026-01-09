from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .belief import BeliefFilter
from .dsl import Action, ActionKind, GridCursorState
from .event_boundary import HeuristicEventBoundaryDetector
from .planner import BeliefPlanner
from .simulator import enumerate_candidate_rules


def _normalize_weights(log_ws: np.ndarray) -> np.ndarray:
    if log_ws.size == 0:
        return log_ws
    x = log_ws.astype(np.float64)
    x = x - float(np.max(x))
    w = np.exp(x)
    s = float(np.sum(w))
    if s <= 0:
        return np.full_like(w, 1.0 / float(len(w)))
    return (w / s).astype(np.float64)


def _entropy_from_probs(p: np.ndarray, *, eps: float = 1e-12) -> float:
    pp = np.clip(p.astype(np.float64), eps, 1.0)
    return float(-np.sum(pp * np.log(pp)))


def _action_from_last_action_viz(
    last_action_viz: Optional[dict[str, Any]],
    prev: Optional[GridCursorState],
    cur: GridCursorState,
) -> Action:
    """Best-effort mapping from UI action payload to our abstract Action."""
    if last_action_viz and isinstance(last_action_viz, dict):
        aid = last_action_viz.get("id")
        if isinstance(aid, int):
            if aid == 6:
                data = last_action_viz.get("data") or {}
                try:
                    x = int(data.get("x", cur.cursor_x))
                    y = int(data.get("y", cur.cursor_y))
                    return Action.click(x=x, y=y)
                except Exception:
                    return Action.click()
            if aid in (1, 2, 3, 4, 5, 7):
                key = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACE", 7: "ENTER"}[aid]
                return Action.keypress(key)

    # Fallback: treat cursor motion as a SET_CURSOR command.
    if prev is not None and (prev.cursor_x != cur.cursor_x or prev.cursor_y != cur.cursor_y):
        return Action.set_cursor(x=cur.cursor_x, y=cur.cursor_y)
    return Action.noop()


@dataclass
class BayesUIMappers:
    """Maintains a small belief+planner state and produces UI heatmaps."""

    belief: BeliefFilter
    boundary: HeuristicEventBoundaryDetector
    planner: BeliefPlanner
    prev: Optional[GridCursorState] = None
    last_transition: Optional[tuple[GridCursorState, Action, GridCursorState]] = None

    @staticmethod
    def from_initial_state(
        initial: GridCursorState,
        *,
        max_rules: int = 96,
        beam_size: int = 32,
        top_k_rules: int = 8,
        horizon: int = 2,
    ) -> "BayesUIMappers":
        palette = set(int(x) for x in np.unique(initial.grid))
        rules = enumerate_candidate_rules(palette, background=0, max_rules=max_rules)
        belief = BeliefFilter(rules, beam_size=beam_size, beta=80.0)
        boundary = HeuristicEventBoundaryDetector(background=0)
        boundary.reset(initial)
        planner = BeliefPlanner(
            belief,
            boundary=boundary,
            horizon=horizon,
            top_k_rules=top_k_rules,
            max_sequences=256,
            w_progress=1.0,
            w_info=1.0,
        )
        return BayesUIMappers(belief=belief, boundary=boundary, planner=planner, prev=None, last_transition=None)

    def update(self, cur: GridCursorState, last_action_viz: Optional[dict[str, Any]]) -> None:
        if self.prev is None:
            self.prev = cur
            return
        a = _action_from_last_action_viz(last_action_viz, self.prev, cur)
        self.last_transition = (self.prev, a, cur)
        self.belief.update(self.prev, a, cur)
        self.prev = cur
        # Update boundary detector state (used for planning objective).
        _ = self.boundary.observe(cur)

    def controllability_map(self, cur: GridCursorState, *, plan_seq: tuple[Action, ...]) -> np.ndarray:
        """E_z[ mean_a I(pred_z(a) != pred_z(noop)) ] per cell."""
        parts = self.belief.topk(self.planner.top_k_rules)
        if not parts:
            return np.zeros_like(cur.grid, dtype=np.float32)
        logw = np.array([p.log_w for p in parts], dtype=np.float64)
        w = _normalize_weights(logw)

        base_grids = [p.rule.apply(cur, Action.noop()).grid for p in parts]
        acts = plan_seq  # use plan actions as representative set (cheap)
        cand = [Action.noop(), *acts, Action.click(), Action.keypress("UP"), Action.keypress("RIGHT")]
        # Dedup by (kind,x,y,key)
        seen: set[tuple] = set()
        uniq: list[Action] = []
        for a in cand:
            k = (a.kind.value, a.x, a.y, a.key)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(a)

        denom = max(1, len(uniq) - 1)
        out = np.zeros_like(cur.grid, dtype=np.float32)
        for a in uniq:
            if a.kind == ActionKind.NOOP:
                continue
            for i, p in enumerate(parts):
                pred = p.rule.apply(cur, a).grid
                out += float(w[i]) * (pred != base_grids[i]).astype(np.float32)
        out /= float(denom)
        return np.clip(out, 0.0, 1.0)

    def disagreement_map(self, cur: GridCursorState, action: Action) -> np.ndarray:
        """Per-cell entropy of predicted values under top-k hypotheses."""
        parts = self.belief.topk(self.planner.top_k_rules)
        if not parts:
            return np.zeros_like(cur.grid, dtype=np.float32)
        logw = np.array([p.log_w for p in parts], dtype=np.float64)
        w = _normalize_weights(logw)

        preds = [p.rule.apply(cur, action).grid for p in parts]
        h, w0 = cur.grid.shape
        out = np.zeros((h, w0), dtype=np.float32)

        # For speed: compute entropy over a small set of actually predicted values.
        for y in range(h):
            for x in range(w0):
                vals = np.array([int(g[y, x]) for g in preds], dtype=np.int32)
                # compress
                uniq, inv = np.unique(vals, return_inverse=True)
                probs = np.zeros((len(uniq),), dtype=np.float64)
                for i in range(len(vals)):
                    probs[int(inv[i])] += float(w[i])
                ent = _entropy_from_probs(probs)
                # normalize by log(K)
                denom = np.log(max(2, len(uniq)))
                out[y, x] = float(ent / denom) if denom > 0 else 0.0
        return np.clip(out, 0.0, 1.0)

    def surprise_map(self, cur: GridCursorState) -> np.ndarray:
        """Per-cell 1 - P(pred==obs) under top-k hypotheses for the last transition."""
        if self.last_transition is None:
            return np.zeros_like(cur.grid, dtype=np.float32)
        prev, a, cur = self.last_transition
        parts = self.belief.topk(self.planner.top_k_rules)
        if not parts:
            return np.zeros_like(cur.grid, dtype=np.float32)
        logw = np.array([p.log_w for p in parts], dtype=np.float64)
        w = _normalize_weights(logw)

        preds = [p.rule.apply(prev, a).grid for p in parts]
        obs = cur.grid
        h, w0 = obs.shape
        out = np.zeros((h, w0), dtype=np.float32)
        for y in range(h):
            for x in range(w0):
                p_match = 0.0
                ov = int(obs[y, x])
                for i, g in enumerate(preds):
                    p_match += float(w[i]) * float(int(g[y, x]) == ov)
                out[y, x] = float(1.0 - p_match)
        return np.clip(out, 0.0, 1.0)

    def visit_map(self, cur: GridCursorState, *, plan_seq: tuple[Action, ...]) -> np.ndarray:
        """Cursor visitation along the planner's selected sequence under the MAP rule."""
        best_rule = self.belief.best().rule
        st = cur
        h, w0 = cur.grid.shape
        visit = np.zeros((h, w0), dtype=np.float32)
        # Mark current position
        visit[int(np.clip(st.cursor_y, 0, h - 1)), int(np.clip(st.cursor_x, 0, w0 - 1))] += 1.0
        for a in plan_seq:
            st = best_rule.apply(st, a)
            visit[int(np.clip(st.cursor_y, 0, h - 1)), int(np.clip(st.cursor_x, 0, w0 - 1))] += 1.0
        return visit

    def compute_maps(self, cur: GridCursorState) -> dict[str, list[list[float]]]:
        plan = self.planner.plan(cur)
        action0 = plan.action
        plan_seq = tuple(plan.seq)
        maps = {
            "controllability": self.controllability_map(cur, plan_seq=plan_seq).astype(float).tolist(),
            "disagreement": self.disagreement_map(cur, action0).astype(float).tolist(),
            "surprise": self.surprise_map(cur).astype(float).tolist(),
            "visit": self.visit_map(cur, plan_seq=plan_seq).astype(float).tolist(),
        }
        return maps


