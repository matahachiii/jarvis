"""
Conversation memory for one session: the system prompt plus the message history.

Keeps the conversation within the model's context window by dropping the
oldest user/assistant turns (in pairs). The system prompt is never dropped.
"""

CHARS_PER_TOKEN = 4    # rough average for English text
MESSAGE_OVERHEAD = 4   # chat-template tokens around each message (role markers etc.)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for one message, used until Ollama reports real numbers."""
    return len(text) // CHARS_PER_TOKEN + MESSAGE_OVERHEAD


class Conversation:
    """Holds the system prompt and the user/assistant history of one session."""

    def __init__(self, system_prompt: str, context_length: int = 4096,
                 max_tokens: int = 1024, safety_margin: int = 64):
        self.system = {"role": "system", "content": system_prompt,
                       "tokens": estimate_tokens(system_prompt)}
        self.history: list[dict] = []  # user/assistant messages, oldest first, strictly alternating
        # Room for system prompt + history + new user message;
        # max_tokens stays free for the answer, safety_margin covers estimation errors.
        self.budget = context_length - max_tokens - safety_margin
        self.dropped_turns = 0  # how many user/assistant pairs were trimmed this session

    @property
    def messages(self) -> list[dict]:
        """The conversation in the format the brain expects (without our internal token counts)."""
        return [{"role": m["role"], "content": m["content"]} for m in [self.system, *self.history]]

    def total_tokens(self) -> int:
        """Current size of everything that would be sent to the model."""
        return self.system["tokens"] + sum(m["tokens"] for m in self.history)

    def add_user(self, text: str) -> None:
        """Add a user message, then trim so the next request fits the budget."""
        self.history.append({"role": "user", "content": text, "tokens": estimate_tokens(text)})
        self.trim()

    def add_assistant(self, text: str, stats: dict | None = None) -> None:
        """Add the assistant's answer and, if available, calibrate token counts with Ollama's real numbers."""
        self.history.append({"role": "assistant", "content": text, "tokens": estimate_tokens(text)})
        if stats:
            self._calibrate(stats)

    def trim(self) -> None:
        """Drop the oldest user/assistant pair until we fit. Never drops the newest user message."""
        while self.total_tokens() > self.budget and len(self.history) > 1:
            del self.history[:2]
            self.dropped_turns += 1

    def reset(self) -> None:
        """Forget the conversation, keep the system prompt."""
        self.history.clear()
        self.dropped_turns = 0

    def _calibrate(self, stats: dict) -> None:
        """
        Use Ollama's real token counts after an answer, so total_tokens() matches reality.

        prompt_tokens = everything we sent (system + history up to the last user message)
        output_tokens = the answer itself (exact)
        Our estimation error on what we sent is parked on the system prompt's count:
        the system prompt is never dropped, so the correction stays valid.
        """
        output_tokens = stats.get("output_tokens")
        if output_tokens:
            self.history[-1]["tokens"] = output_tokens + MESSAGE_OVERHEAD

        prompt_tokens = stats.get("prompt_tokens")
        if prompt_tokens:
            sent_estimate = self.system["tokens"] + sum(m["tokens"] for m in self.history[:-1])
            self.system["tokens"] = max(1, self.system["tokens"] + prompt_tokens - sent_estimate)
