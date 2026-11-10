"""
Text Jarvis: a terminal chat loop.

Type a message and Jarvis streams his answer back, word by word. When a request
needs a tool, the call is shown dimmed; "confirm" tools ask y/n first (default no).
Commands start with "/" (type /help). Ctrl+C stops an answer; at the prompt it quits.
Every session is saved as a JSONL transcript in logs/.
"""

import readline  # noqa: F401  (importing it is enough: input() gets arrow-key history and line editing)

from jarvis.brain import create_brain
from jarvis.core.agent import run_turn
from jarvis.core.config import load_config
from jarvis.core.conversation import Conversation
from jarvis.core.personality import load_system_prompt
from jarvis.core.session_log import SessionLog
from jarvis.tools.registry import ToolRegistry, registry

HELP = """  Commands:
    /help    show this help
    /tools   list the tools Jarvis can use
    /reset   forget the conversation (the personality stays)
    /stats   timing and tokens of the last answer, current history size
    /exit    quit (also /quit, Ctrl+D, or Ctrl+C at the prompt)
  Ctrl+C while Jarvis is answering stops the answer."""

OLLAMA_HINT = "Is Ollama running? Start it with:  brew services start ollama"
DIM, RESET = "\033[2m", "\033[0m"   # terminal codes: dimmed text, back to normal


def _fmt(value, spec: str = ".2f") -> str:
    """Format a number, or 'n/a' if it is missing."""
    return "n/a" if value is None else format(value, spec)


def describe_call(name: str, args) -> str:
    """A tool call as readable text, e.g. quit_app(app_name='Spotify')."""
    if isinstance(args, dict):
        args = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"{name}({args})"


def ask_confirmation(tool, args: dict) -> bool:
    """
    Ask sir before a "confirm" tool runs. Default is NO: only 'y' or 'yes' counts.
    One small function on purpose, so Phase 3 can swap it for a voice confirmation.
    """
    print(f"  About to run: {describe_call(tool.name, args)}")
    try:
        reply = input("  Allow? [y/N] ").strip().lower()
    except EOFError:  # Ctrl+D counts as no
        print()
        return False
    return reply in ("y", "yes")


class TerminalOutput:
    """Prints streamed text and dimmed tool lines, keeping the 'Jarvis: ' prefix and line breaks tidy."""

    def __init__(self, name: str):
        self.name = name
        self.mid_line = False  # are we in the middle of a "Jarvis: ..." line?

    def text(self, piece: str) -> None:
        if not piece:
            return
        if not self.mid_line:
            print(f"{self.name}: ", end="")
            self.mid_line = True
            piece = piece.lstrip()  # a fresh line never starts with a space
        print(piece, end="", flush=True)

    def end_line(self) -> None:
        if self.mid_line:
            print()
            self.mid_line = False

    def tool_call(self, name: str, args) -> None:
        self.end_line()
        print(f"{DIM}  [tool] {describe_call(name, args)}{RESET}", flush=True)

    def tool_result(self, name: str, args, result) -> None:
        status = "ok" if result.success else "FAILED"
        print(f"{DIM}  [tool] -> {status}: {result.message} ({result.duration_s:.2f} s){RESET}", flush=True)


def print_stats(brain, conversation: Conversation, tools: ToolRegistry) -> None:
    """Show timing and tokens of the last answer, plus the current history size."""
    reply = brain.last_response
    if reply is None:
        print("  No completed answer yet.")
    else:
        s = reply.stats
        print(f"  last answer: first word {_fmt(s.get('first_answer_s'))} s, total {_fmt(s.get('total_s'))} s, "
              f"{s.get('output_tokens', 'n/a')} tokens at {_fmt(s.get('tokens_per_s'), '.1f')} tok/s, "
              f"prompt {s.get('prompt_tokens', 'n/a')} tokens, done: {s.get('done_reason', 'n/a')}")
    print(f"  history: {len(conversation.history)} messages, ~{conversation.total_tokens()} of "
          f"{conversation.budget} tokens, trimmed turns: {conversation.dropped_turns}")
    print(f"  tools: {len(tools.names)} registered")


def print_tools(tools: ToolRegistry) -> None:
    if not tools.names:
        print("  No tools registered yet.")
        return
    for tool_name in tools.names:
        t = tools.get(tool_name)
        print(f"  {tool_name:<14} [{t.risk}]  {t.description}")


def answer(brain, conversation: Conversation, tools: ToolRegistry, name: str, log: SessionLog) -> None:
    """Run one turn (tool rounds + streamed final answer), store it and log it."""
    out = TerminalOutput(name)
    try:
        turn = run_turn(brain, conversation, tools, confirm=ask_confirmation,
                        on_text=out.text, on_tool_call=out.tool_call, on_tool_result=out.tool_result)
        out.end_line()
    except KeyboardInterrupt:
        out.text(" [interrupted]")
        out.end_line()
        conversation.undo_last_user()  # forget the unfinished turn, including half-done tool calls
        log.log("event", "answer interrupted")
        return
    except ConnectionError:
        out.end_line()
        print(f"  Lost the connection to the brain. {OLLAMA_HINT}")
        conversation.undo_last_user()
        log.log("event", "brain connection lost")
        return
    except Exception as error:
        out.end_line()
        print(f"  Brain error: {error}")
        conversation.undo_last_user()
        log.log("event", f"brain error: {error}")
        return

    if turn.stats.get("done_reason") == "length":
        print("  [cut off: the answer reached max_tokens]")
    if turn.capped:
        log.log("event", "tool round cap reached")
    extra = {"stats": turn.stats}
    if turn.text != turn.raw:
        extra["raw"] = turn.raw  # keep the original when the safety net had to step in
    log.log("assistant", turn.text, **extra)


def main() -> None:
    config = load_config()
    name = config["jarvis"]["name"]
    brain_cfg = config["brain"]
    brain = create_brain(config)
    conversation = Conversation(
        load_system_prompt(config),
        context_length=brain_cfg.get("context_length", 4096),
        max_tokens=brain_cfg.get("max_tokens", 512),
    )

    print(f"Starting {name} ({brain.name}, {len(registry.names)} tools)...")
    try:
        brain.warm_up(conversation.messages)
    except ConnectionError:
        print(f"Can't reach the brain. {OLLAMA_HINT}")
        return
    except KeyboardInterrupt:
        print()
        return
    except Exception as error:
        print(f"Brain error during start-up: {error}")
        return

    log = SessionLog()
    log.log("system", conversation.messages[0]["content"], brain=brain.name, tools=registry.names)
    print(f"{name} is ready. Type /help for commands.  (log: logs/{log.path.name})\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()  # Ctrl+C / Ctrl+D at the prompt: move to a fresh line, then quit
            break

        if not user_text:
            continue
        if user_text.startswith("/"):
            command = user_text.lower()
            if command in ("/exit", "/quit"):
                break
            elif command == "/help":
                print(HELP)
            elif command == "/tools":
                print_tools(registry)
            elif command == "/reset":
                conversation.reset()
                log.log("event", "reset")
                print("  Conversation cleared.")
            elif command == "/stats":
                print_stats(brain, conversation, registry)
            else:
                print(f"  Unknown command: {user_text}. Type /help.")
            continue

        conversation.add_user(user_text)
        log.log("user", user_text)
        answer(brain, conversation, registry, name, log)
        print()  # blank line between turns

    log.log("event", "session end")
    print(f"{name}: Goodbye, sir.")
