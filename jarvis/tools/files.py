"""
File tools: search sir's files with Spotlight, and open a file, folder or web page.

search_files: only the home folder is searched; Library, hidden folders (.venv, .git, ...)
and other noise are skipped. Results are kept short: the most recently modified matches.

open_item: opening a script or program would RUN it (e.g. a .command file runs in
Terminal). So programs, scripts and executable files are never opened, files must be
inside the home folder, and web pages must be http or https.

No shell anywhere: every value is passed as a separate argument.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from jarvis.tools.mac_apps import _run
from jarvis.tools.registry import ToolResult, tool

SEARCH_ROOT = Path.home()
MAX_RESULTS = 5
MAX_CHECKED = 1000   # never check the dates of more paths than this (a search for "a" finds thousands)
SKIPPED_FOLDERS = {"Library", "node_modules", "site-packages", "__pycache__"}

# Opening these would run code instead of showing a document
BLOCKED_SUFFIXES = {
    ".app", ".command", ".sh", ".bash", ".zsh", ".tool", ".terminal", ".pkg", ".mpkg", ".dmg",
    ".scpt", ".scptd", ".applescript", ".workflow", ".action", ".jar", ".exe", ".py", ".rb", ".pl",
}
WEB_DOMAINS = {"com", "org", "net", "be", "nl", "fr", "de", "eu", "uk", "io", "dev", "ai",
               "edu", "gov", "info", "tv", "me", "co"}
BARE_DOMAIN = re.compile(r"^(www\.)?([a-z0-9-]+\.)+([a-z]{2,})(/\S*)?$", re.IGNORECASE)


# ------------------------------------------------------------ search helpers
def is_visible(path: Path, root: Path) -> bool:
    """True for normal files inside root; False for Library, hidden folders, noise, or outside root."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False
    return not any(part.startswith(".") or part in SKIPPED_FOLDERS for part in parts)


def _modified(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:   # no permission (e.g. Documents not allowed) or file gone
        return 0.0


def _is_folder(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def describe(path: Path, root: Path) -> str:
    """'~/Documents/thesis.pdf, modified 2 November 2026' (date left out when unknown)."""
    text = "~/" + str(path.relative_to(root))
    if _is_folder(path):
        text += " (folder)"
    mtime = _modified(path)
    if mtime:
        moment = datetime.fromtimestamp(mtime)
        text += f", modified {moment.day} {moment:%B %Y}"
    return text


# ------------------------------------------------------------ open helpers
def as_url(target: str) -> str | None:
    """'https://x.com' or 'www.x.com' or 'x.com' -> a https URL; anything else -> None."""
    parsed = urlparse(target)
    if parsed.scheme in ("http", "https"):
        return target if parsed.netloc else None
    if parsed.scheme or target.startswith(("~", "/", ".")):
        return None
    match = BARE_DOMAIN.match(target)
    if match and (target.lower().startswith("www.") or match.group(3).lower() in WEB_DOMAINS):
        return "https://" + target
    return None


def resolve_in_home(target: str, root: Path) -> Path | None:
    """The real path inside root (symlinks followed), or None if it points outside root."""
    if target == "~":
        path = root
    elif target.startswith("~/"):
        path = root / target[2:]
    else:
        path = Path(target)
        if not path.is_absolute():
            path = root / path
    path = path.resolve()   # follows symlinks and "..": a link cannot escape the home folder
    real_root = root.resolve()
    if path != real_root and real_root not in path.parents:
        return None
    return path


def is_program(path: Path) -> bool:
    """True for apps, scripts, installers and executable files: opening them would run code."""
    if any(part.lower().endswith(tuple(BLOCKED_SUFFIXES)) for part in path.parts):
        return True
    return path.is_file() and os.access(path, os.X_OK)


def _display(path: Path, root: Path) -> str:
    real_root = root.resolve()
    return "~" if path == real_root else "~/" + str(path.relative_to(real_root))


# ------------------------------------------------------------ tools
@tool("Search sir's files and folders by name with Spotlight, in his home folder. "
      "Returns the most recently changed matches with their paths.",
      {"query": ("string", "Part of the file or folder name, e.g. thesis or invoice")})
def search_files(query):
    if query.startswith("-"):
        return ToolResult(False, "A search term cannot start with a dash.")
    root = SEARCH_ROOT
    result = _run(["mdfind", "-onlyin", str(root), "-name", query])
    if result.returncode != 0:
        return ToolResult(False, f"Spotlight search failed: {result.stderr.strip()}")

    paths = [Path(line) for line in result.stdout.splitlines() if line.strip()]
    paths = [p for p in paths if is_visible(p, root)]
    if not paths:
        return ToolResult(True, f"Found no files or folders with '{query}' in their name.")

    total = len(paths)
    newest = sorted(paths[:MAX_CHECKED], key=_modified, reverse=True)[:MAX_RESULTS]
    header = f"Found {total} match{'es' if total != 1 else ''}"
    if total > MAX_RESULTS:
        header += f", the {MAX_RESULTS} most recent"
    return ToolResult(True, header + ": " + "; ".join(describe(p, root) for p in newest) + ".")


@tool("Open a file or folder from sir's home folder in its default app, or a web page in the browser. "
      "Programs and scripts are never opened; use open_app for apps.",
      {"target": ("string", "A path like ~/Documents/report.pdf (e.g. from search_files) "
                            "or a web address like https://www.vub.be")})
def open_item(target):
    target = target.strip()

    url = as_url(target)
    if url:
        result = _run(["open", url])
        if result.returncode != 0:
            return ToolResult(False, f"macOS could not open {url}: {result.stderr.strip()}")
        return ToolResult(True, f"Opened {url} in the browser.")
    if urlparse(target).scheme and not target.startswith(("~", "/")):
        return ToolResult(False, "Only web pages starting with http or https can be opened.")

    root = SEARCH_ROOT
    path = resolve_in_home(target, root)
    if path is None:
        return ToolResult(False, "Only files and folders inside sir's home folder can be opened.")
    if not path.exists():
        return ToolResult(False, f"There is no file or folder at {target}. search_files can find it.")
    if is_program(path):
        return ToolResult(False, f"{path.name} is a program or script, and those are never opened "
                                 "for safety. Apps can be opened with open_app.")

    result = _run(["open", str(path)])
    if result.returncode != 0:
        return ToolResult(False, f"macOS could not open {_display(path, root)}: {result.stderr.strip()}")
    where = " in Finder" if _is_folder(path) else ""
    return ToolResult(True, f"Opened {_display(path, root)}{where}.")
