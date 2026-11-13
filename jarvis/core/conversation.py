"""
Conversation memory for one session: the system prompt, fixed example exchanges,
and the message history.

The history is a flat list of messages, grouped into TURNS. A turn starts with
a user message and contains everything until the next user message:

    user -> [assistant with tool_calls -> tool results]* -> final assistant answer

Trimming drops whole turns (oldest first), so a tool result is never left
without the request that caused it. The system prompt, the examples and the
current turn are never dropped.
"""

import json

CHARS_PER_TOKEN = 4    # rough average for English text
MESSAGE_OVERHEAD = 4   # chat-template tokens around each message (role markers etc.)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for one message, used until Ollama reports real numbers."""
    return len(text) // CHARS_PER_TOKEN + MESSAGE_OVERHEAD


def estimate_message(message: dict) -> int:
    """Estimate for a whole message, including any tool calls in it."""
    text = message.get("content", "")
    if message.get("tool_calls"):
        text += json.dumps(message["tool_calls"], default=str)
    return estimate_tokens(text)


class Conversation:
    """Holds the system prompt, the examples and the message history (grouped in turns) of one session."""

    def __init__(self, system_prompt: str, context_length: int = 4096,
                 max_tokens: int = 1024, safety_margin: int = 64, examples: list[dict] | None = None):
        self.system = {"role": "system", "content": system_prompt,
                       "tokens": estimate_tokens(system_prompt)}
        # Fixed example exchanges (with real tool calls) that show the model how to behave.
        # They come right after the system prompt and are never trimmed or reset.
        self.examples = [{**m, "tokens": estimate_message(m)} for m in (examples or [])]
        self.history: list[dict] = []  # all messages after the examples, oldest first
        # Room for system prompt + examples + history + new user message;
        # max_tokens stays free for the answer, safety_margin covers estimation errors.
        self.budget = context_length - max_tokens - safety_margin
        self.dropped_turns = 0  # how many whole turns were trimmed this session

    @property
    def messages(self) -> list[dict]:
        """The conversation in the format the brain expects (without our internal token counts)."""
        return [{k: v for k, v in m.items() if k != "tokens"}
                for m in [self.system, *self.examples, *self.history]]

    def _fixed_tokens(self) -> int:
        """System prompt + examples: the part that is always sent."""
        return self.system["tokens"] + sum(m["tokens"] for m in self.examples)

    def total_tokens(self) -> int:
        """Current size of everything that would be sent to the model."""
        return self._fixed_tokens() + sum(m["tokens"] for m in self.history)

    # ------------------------------------------------------------ adding messages
    def add_user(self, text: str) -> None:
        """Start a new turn with a user message, then trim so the next request fits the budget."""
        self.history.append({"role": "user", "content": text, "tokens": estimate_tokens(text)})
        self.trim()

    def add_tool_call(self, text: str, tool_calls: list[dict], stats: dict | None = None) -> None:
        """
        Add an assistant message that asks for tools.
        tool_calls: [{"name": "open_app", "arguments": {"app_name": "Spotify"}}, ...]
        """
        calls = [{"function": {"name": c["name"], "arguments": c.get("arguments") or {}}}
                 for c in tool_calls]
        message = {"role": "assistant", "content": text, "tool_calls": calls}
        self.history.append({**message, "tokens": estimate_message(message)})
        if stats:
            self._calibrate(stats)

    def add_tool_result(self, name: str, content: str) -> None:
        """Add the result of one tool call, then trim so the next request fits."""
        self.history.append({"role": "tool", "tool_name": name, "content": content,
                             "tokens": estimate_tokens(content)})
        self.trim()

    def add_assistant(self, text: str, stats: dict | None = None) -> None:
        """Add the final answer of a turn and, if available, calibrate with Ollama's real numbers."""
        self.history.append({"role": "assistant", "content": text, "tokens": estimate_tokens(text)})
        if stats:
            self._calibrate(stats)

    # ------------------------------------------------------------ trimming and undo
    def _turn_starts(self) -> list[int]:
        """Positions of the user messages: every turn starts at one of these."""
        return [i for i, m in enumerate(self.history) if m["role"] == "user"]

    def trim(self) -> None:
        """Drop the oldest whole turn until we fit. Never drops the current (newest) turn."""
        while self.total_tokens() > self.budget:
            starts = self._turn_starts()
            if len(starts) < 2:
                break  # only the current turn is left
            del self.history[:starts[1]]
            self.dropped_turns += 1

    def undo_last_user(self) -> None:
        """
        Remove the current turn if it has no final answer yet (answer failed or was
        interrupted): the user message plus any tool calls and tool results after it.
        A finished turn is left alone.
        """
        starts = self._turn_starts()
        if not starts:
            return
        last = self.history[-1]
        finished = last["role"] == "assistant" and not last.get("tool_calls")
        if not finished:
            del self.history[starts[-1]:]

    def reset(self) -> None:
        """Forget the conversation, keep the system prompt and the examples."""
        self.history.clear()
        self.dropped_turns = 0

    # ------------------------------------------------------------ token calibration
    def _calibrate(self, stats: dict) -> None:
        """
        Use Ollama's real token counts after any assistant message (tool call or final
        answer), so total_tokens() matches reality.

        prompt_tokens = everything we sent (system + examples + history before this message)
        output_tokens = this assistant message itself (exact)
        Our estimation error on what we sent is parked on the system prompt's count:
        the system prompt is never dropped, so the correction stays valid.
        """
        output_tokens = stats.get("output_tokens")
        if output_tokens:
            self.history[-1]["tokens"] = output_tokens + MESSAGE_OVERHEAD

        prompt_tokens = stats.get("prompt_tokens")
        if prompt_tokens:
            sent_estimate = self._fixed_tokens() + sum(m["tokens"] for m in self.history[:-1])
            self.system["tokens"] = max(1, self.system["tokens"] + prompt_tokens - sent_estimate)
