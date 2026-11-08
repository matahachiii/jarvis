"""
Brain provider for local models running in Ollama.
"""

import time

from ollama import Client

from jarvis.brain.base import Brain, BrainResponse


class OllamaBrain(Brain):
    """Talks to a local model through the Ollama server."""

    def __init__(self, settings: dict):
        super().__init__(settings)
        ollama_settings = settings["ollama"]
        self.model = ollama_settings["model"]
        self.keep_alive = ollama_settings.get("keep_alive", "10m")
        self.client = Client(host=ollama_settings.get("host", "http://localhost:11434"))

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    def chat(self, messages: list[dict], tools: list | None = None) -> BrainResponse:
        start = time.perf_counter()
        first_token_s = None   # first token of any kind (thinking or answer)
        first_answer_s = None  # first token of the actual answer
        text_parts, thinking_parts, tool_calls = [], [], []
        final_chunk = None

        # stream=True: the answer arrives piece by piece, so we can measure
        # how fast the first word appears (crucial for voice later)
        stream = self.client.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            think=self.settings.get("think", False),
            stream=True,
            keep_alive=self.keep_alive,
            options={
                "temperature": self.settings.get("temperature", 0.6),
                "num_ctx": self.settings.get("context_length", 4096),
            },
        )

        for chunk in stream:
            elapsed = time.perf_counter() - start
            message = chunk.message

            if message.thinking:
                thinking_parts.append(message.thinking)
                if first_token_s is None:
                    first_token_s = elapsed

            if message.content:
                text_parts.append(message.content)
                if first_token_s is None:
                    first_token_s = elapsed
                if first_answer_s is None:
                    first_answer_s = elapsed

            if message.tool_calls:
                tool_calls.extend(message.tool_calls)
                if first_answer_s is None:
                    first_answer_s = elapsed

            if chunk.done:
                final_chunk = chunk

        stats = {
            "total_s": time.perf_counter() - start,
            "first_token_s": first_token_s,
            "first_answer_s": first_answer_s,
        }

        # Ollama reports durations in nanoseconds
        if final_chunk is not None:
            ns = 1e9
            eval_count = final_chunk.eval_count or 0
            eval_duration = (final_chunk.eval_duration or 0) / ns
            stats.update({
                "load_s": (final_chunk.load_duration or 0) / ns,
                "prompt_tokens": final_chunk.prompt_eval_count,
                "output_tokens": eval_count,
                "tokens_per_s": eval_count / eval_duration if eval_duration else None,
            })

        return BrainResponse(
            text="".join(text_parts).strip(),
            thinking="".join(thinking_parts).strip(),
            tool_calls=tool_calls,
            stats=stats,
        )
