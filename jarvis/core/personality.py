"""Load Jarvis' personality (system prompt) from a text file and fill in placeholders."""

from datetime import datetime
from pathlib import Path

from jarvis.core.config import PROJECT_ROOT


def load_system_prompt(config: dict, now: datetime | None = None) -> str:
    """Read the personality file and fill in {name}, {user_name}, {date} and {time}."""
    jarvis_cfg = config["jarvis"]
    path = Path(PROJECT_ROOT) / jarvis_cfg.get("personality_file", "config/personality.md")
    template = path.read_text(encoding="utf-8").strip()

    # 'now' can be passed in for testing; otherwise use the current time
    now = now or datetime.now()
    values = {
        "name": jarvis_cfg["name"],
        "user_name": jarvis_cfg["user_name"],
        # Spoken-friendly date, e.g. "Monday 9 November 2026"
        "date": f"{now:%A} {now.day} {now:%B %Y}",
        "time": now.strftime("%H:%M"),
    }
    return template.format_map(values)
