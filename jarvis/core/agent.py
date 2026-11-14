"""
The agent loop: answer one user turn, calling tools when the model asks for them.

    brain (with tools) -> tool calls? -> execute them, add the results -> brain again ...
    -> final answer, streamed through StreamCleaner

Used by the chat loop and by the tests, so both run exactly the same code.
How text, tool calls and confirmations are shown is up to the caller (callbacks),
so this file knows nothing about the terminal (or, in Phase 3, the voice).

Safety-critical outcomes are not left to the model's wording: when sir declines every
action of a round, the answer is a fixed sentence (the model once invented a reason).
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from jarvis.core.conversation import Conversation
from jarvis.core.text_cleaner import StreamCleaner, clean_for_speech
from jarvis.tools.registry import Tool, ToolRegistry, ToolResult

MAX_TOOL_ROUNDS = 5   # tool rounds per user turn; after that the model must answer
CAP_MESSAGE = "I'm afraid I couldn't finish that, sir."
DECLINED_MESSAGE = "Very well, sir. Nothing was done."


def _noop(*args, **kwargs) -> None:
    """Default callback: do nothing."""


@dataclass
class ToolRun:
    """One executed (or refused) tool call."""

    name: str
    arguments: dict
    result: ToolResult


@dataclass
class TurnResult:
    """Everything that happened in one user turn."""

    text: str                                   # final answer, cleaned for speech
    raw: str                                    # final answer as the model wrote it
    stats: dict                                 # stats of the last brain call
    tool_runs: list[ToolRun] = field(default_factory=list)
    rounds: int = 0                             # brain calls in this turn
    capped: bool = False                        # True if the round cap stopped the model
    declined: bool = False                      # True if sir declined every action of a round


def format_tool_result(result: ToolResult) -> str:
    """What the model reads back: an explicit status, so it can never mistake a failure for success."""
    return f"{'SUCCESS' if result.success else 'FAILED'}: {result.message}"


def _get(obj, key):
    """Read a field from a dict or from an object (Ollama returns objects, tests use dicts)."""
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def _parse_calls(tool_calls) -> list[dict]:
    """Turn the brain's tool calls into plain {"name", "arguments"} dicts."""
    calls = []
    for tool_call in tool_calls:
        function = _get(tool_call, "function")
        arguments = _get(function, "arguments") or {}
        if isinstance(arguments, str):          # some APIs send the arguments as a JSON string
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass                            # registry.execute() will reject it with a clear message
        calls.append({"name": _get(function, "name"),
                      "arguments": dict(arguments) if isinstance(arguments, dict) else arguments})
    return calls


def run_turn(brain, conversation: Conversation, registry: ToolRegistry | None,
             confirm: Callable[[Tool, dict], bool] | None = None,
             on_text: Callable[[str], None] = _noop,
             on_tool_call: Callable[[str, dict], None] = _noop,
             on_tool_result: Callable[[str, dict, ToolResult], None] = _noop,
             max_tool_rounds: int = MAX_TOOL_ROUNDS) -> TurnResult:
    """
    Answer the user message that was just added to `conversation`.

    Errors from the brain (and Ctrl+C) are NOT caught here: the caller catches them
    and calls conversation.undo_last_user(), which also removes a half-finished tool turn.
    """
    schemas = registry.schemas() if registry is not None else []
    tools = schemas or None   # no tools registered: plain chat, exactly as in Phase 1
    runs: list[ToolRun] = []
    rounds = 0

    while True:
        rounds += 1
        cleaner = StreamCleaner()
        for piece in brain.chat_stream(conversation.messages, tools=tools):
            on_text(cleaner.feed(piece))
        on_text(cleaner.flush())

        reply = brain.last_response
        text = clean_for_speech(reply.text).strip()

        # No tool calls: this is the final answer
        if not reply.tool_calls:
            conversation.add_assistant(text, reply.stats)
            return TurnResult(text, reply.text, reply.stats, runs, rounds)

        # Round cap: a confused model asking for tools again and again gets stopped here
        if rounds > max_tool_rounds:
            on_text((" " if text else "") + CAP_MESSAGE)
            conversation.add_assistant(CAP_MESSAGE)   # no stats: they describe the refused tool call
            return TurnResult(CAP_MESSAGE, reply.text, reply.stats, runs, rounds, capped=True)

        # Execute every tool call of this round, in order, and feed the results back
        calls = _parse_calls(reply.tool_calls)
        conversation.add_tool_call(text, calls, reply.stats)
        round_runs = []
        for call in calls:
            on_tool_call(call["name"], call["arguments"])
            result = registry.execute(call["name"], call["arguments"], confirm=confirm)
            on_tool_result(call["name"], call["arguments"], result)
            run = ToolRun(call["name"], call["arguments"], result)
            round_runs.append(run)
            runs.append(run)
            conversation.add_tool_result(call["name"], format_tool_result(result))

        # Sir said no to everything this round: a fixed, honest answer instead of the model's wording
        if all(r.result.confirmed is False for r in round_runs):
            on_text((" " if text else "") + DECLINED_MESSAGE)
            conversation.add_assistant(DECLINED_MESSAGE)
            return TurnResult(DECLINED_MESSAGE, DECLINED_MESSAGE, reply.stats, runs, rounds, declined=True)
