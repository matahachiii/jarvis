"""
Brain interface: the abstraction every LLM provider implements.

Every LLM provider (local Ollama, free-tier cloud API, ...) implements this
same interface. The rest of Jarvis only talks to `Brain`, never to a provider
directly. That is what makes the brain swappable via the config.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class BrainResponse:
    """What a brain returns after one request."""

    text: str                                       # the final answer
    thinking: str = ""                              # reasoning, if thinking mode was on
    tool_calls: list = field(default_factory=list)  # tools the model wants to call (Phase 2)
    stats: dict = field(default_factory=dict)       # timing and token statistics


class Brain(ABC):
    """Base class for all brains. Every provider must implement `chat_stream`."""

    def __init__(self, settings: dict):
        self.settings = settings
        self.last_response: BrainResponse | None = None  # filled in after every request

    @property
    def name(self) -> str:
        """Short description of this brain, e.g. 'ollama/qwen3:4b'."""
        return self.__class__.__name__

    @abstractmethod
    def chat_stream(self, messages: list[dict], tools: list | None = None) -> Iterator[str]:
        """
        Send a conversation to the model and yield the answer piece by piece.

        messages: list of {"role": "system" | "user" | "assistant", "content": "..."}
        tools:    optional list of tool definitions (used from Phase 2)

        Yields text chunks of the answer as soon as they arrive. When the stream
        is finished, self.last_response holds the complete BrainResponse
        (full text, thinking, tool calls, stats).
        """

    def chat(self, messages: list[dict], tools: list | None = None) -> BrainResponse:
        """Send a conversation and return the complete answer at once (non-streaming)."""
        for _ in self.chat_stream(messages, tools):
            pass  # consume the whole stream; we only want the final result
        return self.last_response
