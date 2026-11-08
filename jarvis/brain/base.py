"""
The Brain interface.

Every LLM provider (local Ollama, free-tier cloud API, ...) implements this
same interface. The rest of Jarvis only talks to `Brain`, never to a provider
directly. That is what makes the brain swappable via the config.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class BrainResponse:
    """What a brain returns after one request."""

    text: str                                       # the final answer
    thinking: str = ""                              # reasoning, if thinking mode was on
    tool_calls: list = field(default_factory=list)  # tools the model wants to call (Phase 2)
    stats: dict = field(default_factory=dict)       # timing and token statistics


class Brain(ABC):
    """Base class for all brains. Every provider must implement `chat`."""

    def __init__(self, settings: dict):
        self.settings = settings

    @property
    def name(self) -> str:
        """Short description of this brain, e.g. 'ollama/qwen3:4b'."""
        return self.__class__.__name__

    @abstractmethod
    def chat(self, messages: list[dict], tools: list | None = None) -> BrainResponse:
        """
        Send a conversation to the model and return its answer.

        messages: list of {"role": "system" | "user" | "assistant", "content": "..."}
        tools:    optional list of tool definitions (used from Phase 2)
        """
