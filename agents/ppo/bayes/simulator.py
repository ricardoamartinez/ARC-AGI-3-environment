from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Iterable, Sequence

import numpy as np

from .dsl import Action, GridCursorState
from .rules import IdentityRule, MoveColorOnKeyRule, PaintOnClickRule, RuleProgram, ToggleOnClickRule


def mismatch_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of cells that differ between two grids."""
    if a.shape != b.shape:
        raise ValueError(f"Grid shapes differ: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 0.0
    return float(np.mean(a != b))


def log_likelihood_from_mismatch(frac: float, *, beta: float = 50.0) -> float:
    """Convert mismatch fraction into a simple log-likelihood proxy.

    Higher `beta` makes the likelihood sharper (penalizes mismatches more).
    """
    f = float(np.clip(frac, 0.0, 1.0))
    return -beta * f


def transition_log_likelihood(
    rule: RuleProgram,
    prev: GridCursorState,
    action: Action,
    observed_next: GridCursorState,
    *,
    beta: float = 50.0,
) -> float:
    pred = rule.apply(prev, action)
    frac = mismatch_fraction(pred.grid, observed_next.grid)
    return log_likelihood_from_mismatch(frac, beta=beta)


def enumerate_candidate_rules(
    palette: Iterable[int],
    *,
    background: int = 0,
    max_rules: int = 128,
) -> list[RuleProgram]:
    """Generate a compact library of candidate rule programs.

    This is intentionally small; the belief filter can expand it over time.
    """
    colors = [int(c) for c in sorted(set(int(x) & 0xFF for x in palette))]
    colors = [c for c in colors if c != int(background)]

    rules: list[RuleProgram] = [IdentityRule()]

    for c in colors:
        rules.append(PaintOnClickRule(color=c))
    # Toggle pairs (limited to keep branching manageable)
    for i, a in enumerate(colors[:8]):
        for b in colors[i + 1 : 8]:
            rules.append(ToggleOnClickRule(a=a, b=b))

    for c in colors[:16]:
        rules.append(MoveColorOnKeyRule(color=c, background=background))

    # Hard cap for safety
    return rules[: int(max_rules)]


@dataclass(frozen=True, slots=True)
class SimResult:
    pred: GridCursorState
    ll: float


def simulate_and_score(
    rule: RuleProgram,
    prev: GridCursorState,
    action: Action,
    observed_next: GridCursorState,
    *,
    beta: float = 50.0,
) -> SimResult:
    pred = rule.apply(prev, action)
    ll = transition_log_likelihood(rule, prev, action, observed_next, beta=beta)
    return SimResult(pred=pred, ll=ll)



