"""
File tools: search sir's files with Spotlight (mdfind), limited to his home folder.
(Opening a file or URL follows in step 4.7.)

Only the home folder is searched; Library, hidden folders (.venv, .git, ...) and other
noise are skipped. Results are kept short: the most recently modified matches.
No shell: the search term is passed as a separate argument.
"""

from datetime import datetime
from pathlib import Path

from jarvis.tools.mac_apps import _run
from jarvis.tools.registry import ToolResult, tool

SEARCH_ROOT = Path.home()
MAX_RESULTS = 5
MAX_CHECKED = 1000   # never check the dates of more paths than this (a search for "a" finds thousands)
SKIPPED_FOLDERS = {"Library", "node_modules", "site-packages", "__pycache__"}


# ------------------------------------------------------------ helpers
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
