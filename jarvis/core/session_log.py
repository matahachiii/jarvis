"""
Session log: every chat session is saved as a JSONL transcript in logs/ (git-ignored).

JSONL = one JSON object per line. Each line is written immediately, so the log
survives a crash. Useful for debugging now and for long-term memory in Phase 5.
"""

import json
from datetime import datetime
from pathlib import Path

from jarvis.core.config import PROJECT_ROOT


class SessionLog:
    """Appends one JSON line per message or event to logs/session_<start time>.jsonl."""

    def __init__(self, log_dir: str = "logs"):
        folder = Path(PROJECT_ROOT) / log_dir
        folder.mkdir(exist_ok=True)
        started = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = folder / f"session_{started}.jsonl"

    def log(self, role: str, text: str, **extra) -> None:
        """role: "system", "user", "assistant", "tool" or "event". Extra fields (stats, raw, ...) are added as-is."""
        record = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                  "role": role, "text": text, **extra}
        with self.path.open("a", encoding="utf-8") as f:
            # ensure_ascii=False keeps characters like é readable; default=str turns anything unusual into text
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
