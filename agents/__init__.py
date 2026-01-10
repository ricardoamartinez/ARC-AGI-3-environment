from __future__ import annotations

from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm

# Core agents (best-effort). These imports register subclasses for AVAILABLE_AGENTS,
# but should never prevent importing `agents` in minimal environments.
# Note: Manual agent has been consolidated into JEPAAgent - use JEPAAgent for both
# RL training and interactive control.

JEPAAgent: type[Agent] | None
try:  # pragma: no cover
    from .jepa import JEPAAgent as _JEPAAgent
except ModuleNotFoundError:  # pragma: no cover
    JEPAAgent = None
else:
    JEPAAgent = _JEPAAgent

from .templates.random_agent import Random

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

__all__ = [
    "Swarm",
    "Random",
    "JEPAAgent",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
]
