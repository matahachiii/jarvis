"""
Text Jarvis: a terminal chat loop.

Type a message and Jarvis streams his answer back, word by word.
Commands start with "/" (type /help). Ctrl+C stops an answer; at the prompt it quits.
"""

import readline  # noqa: F401  (importing it is enough: input() gets arrow-key history and line editing)

from jarvis.brain import create_brain
from jarvis.core.config import load_config
from jarvis.core.conversation import Conversation
from jarvis.core.personality import load_system_prompt
from jarvis.core.text_cleaner import StreamCleaner, clean_for_speech

HELP = """  Commands:
    /help    show this help
    /reset   forget the conversation (the personality stays)
    /stats   timing and tokens of the last answer, current history size
    /exit    quit (also /quit, Ctrl+D, or Ctrl+C at the prompt)
  Ctrl+C while Jarvis is answering stops the answer."""

OLLAMA_HINT = "Is Ollama running? Start it with:  brew services start ollama"


def _fmt(value, spec: str = ".2f") -> str:
    """Format a number, or 'n/a' if it is missing."""
    return "n/a" if value is None else format(value, spec)


def print_stats(brain, conversation: Conversation) -> None:
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


def answer(brain, conversation: Conversation, name: str) -> None:
    """Stream one answer to the terminal and store it in the conversation."""
    cleaner = StreamCleaner()
    print(f"{name}: ", end="", flush=True)
    try:
        for piece in brain.chat_stream(conversation.messages):
            print(cleaner.feed(piece), end="", flush=True)
        print(cleaner.flush())
    except KeyboardInterrupt:
        print(" [interrupted]")
        conversation.undo_last_user()  # forget the question, so history stays in user/assistant pairs
        return
    except ConnectionError:
        print(f"\n  Lost the connection to the brain. {OLLAMA_HINT}")
        conversation.undo_last_user()
        return
    except Exception as error:
        print(f"\n  Brain error: {error}")
        conversation.undo_last_user()
        return

    reply = brain.last_response
    if reply.stats.get("done_reason") == "length":
        print("  [cut off: the answer reached max_tokens]")
    # Store the cleaned text, so the model sees its own earlier replies without markdown/emojis
    conversation.add_assistant(clean_for_speech(reply.text).strip(), reply.stats)


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

    print(f"Starting {name} ({brain.name})...")
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
    print(f"{name} is ready. Type /help for commands.\n")

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
            elif command == "/reset":
                conversation.reset()
                print("  Conversation cleared.")
            elif command == "/stats":
                print_stats(brain, conversation)
            else:
                print(f"  Unknown command: {user_text}. Type /help.")
            continue

        conversation.add_user(user_text)
        answer(brain, conversation, name)
        print()  # blank line between turns

    print(f"{name}: Goodbye, sir.")
