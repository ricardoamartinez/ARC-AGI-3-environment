from __future__ import annotations

from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm

# Core agents (best-effort). These imports register subclasses for AVAILABLE_AGENTS,
# but should never prevent importing `agents` in minimal environments.
Manual: type[Agent] | None
try:  # pragma: no cover
    from .manual.agent import Manual as _Manual
except ModuleNotFoundError:  # pragma: no cover
    Manual = None
else:
    Manual = _Manual

PPOAgent: type[Agent] | None
try:  # pragma: no cover
    from .ppo import PPOAgent as _PPOAgent
except ModuleNotFoundError:  # pragma: no cover
    PPOAgent = None
else:
    PPOAgent = _PPOAgent

from .templates.random_agent import Random

ReasoningAgent: type[Agent] | None
try:  # pragma: no cover
    from .templates.reasoning_agent import ReasoningAgent as _ReasoningAgent
except ModuleNotFoundError:  # pragma: no cover
    ReasoningAgent = None
else:
    ReasoningAgent = _ReasoningAgent

# Optional / heavyweight agent stacks. These are best-effort imports so that
# importing `agents` (and by extension `agents.structs`) works in minimal
# environments where some ML/observability deps may not be installed.
try:  # pragma: no cover
    from .templates.langgraph_functional_agent import LangGraphFunc, LangGraphTextOnly
    from .templates.langgraph_random_agent import LangGraphRandom
    from .templates.langgraph_thinking import LangGraphThinking
except ModuleNotFoundError:  # pragma: no cover
    LangGraphFunc = None  # type: ignore[assignment]
    LangGraphTextOnly = None  # type: ignore[assignment]
    LangGraphRandom = None  # type: ignore[assignment]
    LangGraphThinking = None  # type: ignore[assignment]

try:  # pragma: no cover
    from .templates.llm_agents import LLM, FastLLM, GuidedLLM, ReasoningLLM
except ModuleNotFoundError:  # pragma: no cover
    LLM = None  # type: ignore[assignment]
    FastLLM = None  # type: ignore[assignment]
    GuidedLLM = None  # type: ignore[assignment]
    ReasoningLLM = None  # type: ignore[assignment]

try:  # pragma: no cover
    from .templates.smolagents import SmolCodingAgent, SmolVisionAgent
except ModuleNotFoundError:  # pragma: no cover
    SmolCodingAgent = None  # type: ignore[assignment]
    SmolVisionAgent = None  # type: ignore[assignment]

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

# update the agent dictionary to include subclasses of LLM class
if ReasoningAgent is not None:
    AVAILABLE_AGENTS["reasoningagent"] = ReasoningAgent

__all__ = [
    "Swarm",
    "Random",
    "LangGraphFunc",
    "LangGraphTextOnly",
    "LangGraphThinking",
    "LangGraphRandom",
    "LLM",
    "FastLLM",
    "ReasoningLLM",
    "GuidedLLM",
    "ReasoningAgent",
    "Manual",
    "SmolCodingAgent",
    "SmolVisionAgent",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
]
