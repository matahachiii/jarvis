"""
Load Jarvis' personality (system prompt) and his example exchanges.

The system prompt comes from a text file with placeholders. The example exchanges are
short conversations with REAL tool calls: small models learn far more from examples
than from rules (lesson from Phase 1). They are deliberately different from the cases
in test_tools.py, so the test measures whether the pattern is learned, not memorised.
Keep the example answers short and plain: the model sometimes copies their wording.
"""

from datetime import datetime
from pathlib import Path

from jarvis.core.config import PROJECT_ROOT


def tool_names() -> str:
    """The names of all registered tools, e.g. 'calculator, get_time, open_app, ...'."""
    import jarvis.tools  # noqa: F401  (imported here, not at the top: importing it registers every tool)
    from jarvis.tools.registry import registry
    return ", ".join(registry.names) or "none"


def load_system_prompt(config: dict, now: datetime | None = None) -> str:
    """Read the personality file and fill in {name}, {user_name} and {tools} ({date} and {time} if used)."""
    jarvis_cfg = config["jarvis"]
    path = Path(PROJECT_ROOT) / jarvis_cfg.get("personality_file", "config/personality.md")
    template = path.read_text(encoding="utf-8").strip()

    # 'now' can be passed in for testing; otherwise use the current time
    now = now or datetime.now()
    values = {
        "name": jarvis_cfg["name"],
        "user_name": jarvis_cfg["user_name"],
        "tools": tool_names(),
        # Still available, but the personality no longer uses them: get_time is always up to date
        "date": f"{now:%A} {now.day} {now:%B %Y}",
        "time": now.strftime("%H:%M"),
    }
    return template.format_map(values)


# ------------------------------------------------------------ example exchanges
def _user(text):
    return {"role": "user", "content": text}


def _call(name, **arguments):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


def _result(name, text):
    return {"role": "tool", "tool_name": name, "content": text}


def _say(text):
    return {"role": "assistant", "content": text}


def example_turns() -> list[dict]:
    """Example exchanges placed right after the system prompt in every conversation."""
    return [
        # The same request twice: do it again with the tool, never answer from memory
        _user("Quit Notes."),
        _call("quit_app", app_name="Notes"),
        _result("quit_app", "SUCCESS: Closed Notes."),
        _say("Notes is closed, sir."),
        _user("Quit Notes."),
        _call("quit_app", app_name="Notes"),
        _result("quit_app", "FAILED: Notes is not running, so there was nothing to quit."),
        _say("Notes was not running, sir, so there was nothing to close."),
        # Every calculation goes through the calculator
        _user("What is 18 percent of 50?"),
        _call("calculator", expression="18% * 50"),
        _result("calculator", "SUCCESS: 18% * 50 = 9"),
        _say("Nine, sir."),
        # The same question twice: ask the tool again, the answer may have changed meanwhile
        _user("What's the volume?"),
        _call("volume", action="get"),
        _result("volume", "SUCCESS: The volume is 40 percent."),
        _say("Forty percent, sir."),
        _user("And what is it now?"),
        _call("volume", action="get"),
        _result("volume", "SUCCESS: The volume is 65 percent."),
        _say("Sixty-five percent now, sir."),
        # Vague requests still use the tool, and the answer repeats what the tool reported
        _user("Make it louder."),
        _call("volume", action="up"),
        _result("volume", "SUCCESS: The volume is now 75 percent (it was 65)."),
        _say("Seventy-five percent now, sir."),
        # Websites are opened with open_item
        _user("Open the BBC website."),
        _call("open_item", target="https://www.bbc.co.uk"),
        _result("open_item", "SUCCESS: Opened https://www.bbc.co.uk in the browser."),
        _say("The BBC is open in your browser, sir."),
    ]
