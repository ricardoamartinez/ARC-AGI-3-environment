from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dsl import Action, GridCursorState
from .features import TransitionFeatures, feature_dim, transition_features
from .rules import IdentityRule, MoveColorOnKeyRule, PaintOnClickRule, RuleProgram, ToggleOnClickRule
from .simulator import transition_log_likelihood


def _logsumexp(xs: np.ndarray) -> float:
    x = xs.astype(np.float64)
    m = float(np.max(x)) if x.size else 0.0
    return float(m + np.log(np.sum(np.exp(x - m)))) if x.size else float("-inf")


@dataclass(frozen=True, slots=True)
class Particle:
    rule: RuleProgram
    log_w: float


def _rule_spec(rule: RuleProgram) -> tuple[int, int, int]:
    """Convert a rule into a small integer spec (type_id, a, b)."""
    # type_id mapping is stable and must match RuleProposalNetwork.type_count.
    if isinstance(rule, IdentityRule):
        return (0, 0, 0)
    if isinstance(rule, PaintOnClickRule):
        return (1, int(rule.color) & 0xFF, 0)
    if isinstance(rule, ToggleOnClickRule):
        return (2, int(rule.a) & 0xFF, int(rule.b) & 0xFF)
    if isinstance(rule, MoveColorOnKeyRule):
        return (3, int(rule.color) & 0xFF, int(rule.background) & 0xFF)
    # Unknown rule types can be mapped to a reserved bucket.
    return (0, 0, 0)


class RuleProposalNetwork(nn.Module):
    """Lightweight amortized proposal q_phi(z | last_k_transitions).

    This network is intentionally tiny and generic. It outputs unnormalized
    scores over a provided candidate rule list.
    """

    def __init__(self, *, hidden_dim: int = 64, color_vocab: int = 256, type_count: int = 4):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.color_vocab = int(color_vocab)
        self.type_count = int(type_count)

        d = int(feature_dim())
        self.rnn = nn.GRU(input_size=d, hidden_size=self.hidden_dim, batch_first=True)
        self.type_emb = nn.Embedding(self.type_count, self.hidden_dim)
        self.color_emb = nn.Embedding(self.color_vocab, self.hidden_dim)
        self.rule_ln = nn.LayerNorm(self.hidden_dim)
        self.ctx_ln = nn.LayerNorm(self.hidden_dim)

    def score_rules(self, feat_seq: torch.Tensor, rule_specs: torch.Tensor) -> torch.Tensor:
        """Return scores (B,R) given feat_seq (B,T,D) and rule_specs (R,3) int64."""
        if feat_seq.ndim != 3:
            raise ValueError("feat_seq must be (B,T,D)")
        if rule_specs.ndim != 2 or rule_specs.shape[1] != 3:
            raise ValueError("rule_specs must be (R,3)")

        _, h_last = self.rnn(feat_seq)  # h_last: (1,B,H)
        ctx = h_last.squeeze(0)  # (B,H)
        ctx = self.ctx_ln(ctx)

        type_id = rule_specs[:, 0].clamp(0, self.type_count - 1)
        a = rule_specs[:, 1].clamp(0, self.color_vocab - 1)
        b = rule_specs[:, 2].clamp(0, self.color_vocab - 1)
        rule_emb = self.type_emb(type_id) + self.color_emb(a) + self.color_emb(b)  # (R,H)
        rule_emb = self.rule_ln(rule_emb)

        # Dot-product scoring
        scores = torch.matmul(ctx, rule_emb.t()) / float(np.sqrt(self.hidden_dim))
        return scores


class BeliefFilter:
    """Particle/beam belief over discrete rule programs z."""

    def __init__(
        self,
        rules: Sequence[RuleProgram],
        *,
        beam_size: int = 32,
        beta: float = 50.0,
        proposal_net: Optional[RuleProposalNetwork] = None,
        proposal_mix: float = 0.1,
        context_len: int = 4,
        device: str | torch.device = "cpu",
    ) -> None:
        if not rules:
            raise ValueError("rules must be non-empty")
        if beam_size < 1:
            raise ValueError("beam_size must be >= 1")
        if not (0.0 <= float(proposal_mix) <= 1.0):
            raise ValueError("proposal_mix must be in [0,1]")

        self.rules: list[RuleProgram] = list(rules)
        self.beam_size = int(beam_size)
        self.beta = float(beta)
        self.proposal_net = proposal_net
        self.proposal_mix = float(proposal_mix)
        self.device = torch.device(device)

        r = float(len(self.rules))
        init_log_w = -float(np.log(r))
        # Maintain a full categorical distribution over the candidate rule library.
        # Beam behavior is provided by `topk()` / `best()`.
        self._log_w: np.ndarray = np.full((len(self.rules),), init_log_w, dtype=np.float64)

        self._ctx: Deque[TransitionFeatures] = deque(maxlen=int(max(1, context_len)))

    def reset_uniform(self) -> None:
        r = float(len(self.rules))
        init_log_w = -float(np.log(r))
        self._log_w[:] = init_log_w
        self._ctx.clear()

    def append_context(self, feats: TransitionFeatures) -> None:
        self._ctx.append(feats)

    def _proposal_logprobs(self) -> Optional[np.ndarray]:
        if self.proposal_net is None:
            return None
        if not self._ctx:
            return None

        feat_seq = np.stack([f.vec for f in self._ctx], axis=0).astype(np.float32)  # (T,D)
        feat_t = torch.from_numpy(feat_seq[None, :, :]).to(self.device)  # (1,T,D)
        rule_specs = np.array([_rule_spec(r) for r in self.rules], dtype=np.int64)  # (R,3)
        rule_t = torch.from_numpy(rule_specs).to(self.device)
        with torch.no_grad():
            logits = self.proposal_net.score_rules(feat_t, rule_t)  # (1,R)
            logp = F.log_softmax(logits, dim=-1).squeeze(0)  # (R,)
        return logp.detach().cpu().numpy().astype(np.float64)

    def update(self, prev: GridCursorState, action: Action, observed_next: GridCursorState) -> None:
        """Update belief with one transition."""
        self.append_context(transition_features(prev, action, observed_next))

        # Exact update over the candidate rule library (library is intentionally small).
        lls = np.array(
            [
                transition_log_likelihood(r, prev, action, observed_next, beta=self.beta)
                for r in self.rules
            ],
            dtype=np.float64,
        )
        self._log_w = self._log_w + lls

        # Normalize
        lse = _logsumexp(self._log_w)
        if np.isfinite(lse):
            self._log_w = self._log_w - lse

        # Optional amortized proposal mixing (rejuvenation + stability)
        proposal_logp = self._proposal_logprobs()
        if proposal_logp is not None and self.proposal_mix > 0.0:
            # Convert current posterior and proposal to probs, mix, then go back to log-space.
            post = np.exp(self._log_w - float(np.max(self._log_w)))
            post = post / float(np.sum(post)) if post.size else post

            prop = np.exp(proposal_logp - float(np.max(proposal_logp)))
            prop = prop / float(np.sum(prop)) if prop.size else prop

            alpha = float(self.proposal_mix)
            mixed = (1.0 - alpha) * post + alpha * prop
            mixed = np.clip(mixed, 1e-12, 1.0)
            mixed = mixed / float(np.sum(mixed))
            self._log_w = np.log(mixed).astype(np.float64)

    def best(self) -> Particle:
        i = int(np.argmax(self._log_w))
        return Particle(rule=self.rules[i], log_w=float(self._log_w[i]))

    def topk(self, k: Optional[int] = None) -> list[Particle]:
        kk = int(self.beam_size if k is None else k)
        kk = max(1, min(int(kk), len(self.rules)))
        idx = np.argsort(-self._log_w)[:kk]
        return [Particle(rule=self.rules[int(i)], log_w=float(self._log_w[int(i)])) for i in idx]

    def normalized(self) -> list[tuple[RuleProgram, float]]:
        ws = self._log_w.astype(np.float64)
        w = np.exp(ws - float(np.max(ws))) if ws.size else ws
        s = float(np.sum(w)) if w.size else 1.0
        probs = w / s
        idx = np.argsort(-probs)[: min(self.beam_size, len(self.rules))]
        return [(self.rules[int(i)], float(probs[int(i)])) for i in idx]


