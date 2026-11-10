"""
Tools that open and quit applications on this Mac.

Safety:
- Only apps that are really installed can be opened or quit (we look them up first).
- No shell: subprocess gets an argument list. AppleScript gets the app name as a
  separate argument (argv), so a name is never pasted into script text.
- Quitting is a "confirm" tool and always a polite quit, never a force-kill.
- Some apps are protected: quitting them would break the Mac or stop Jarvis himself.
  That check happens FIRST, by name, so it also covers system apps outside /Applications.
"""

import difflib
import subprocess
import time
from pathlib import Path

from jarvis.tools.registry import ToolResult, tool

APP_FOLDERS = [
    Path("/Applications"), Path("/Applications/Utilities"),
    Path("/System/Applications"), Path("/System/Applications/Utilities"),
    Path.home() / "Applications",
]
ALIASES = {
    "vs code": "visual studio code", "vscode": "visual studio code", "code": "visual studio code",
    "chrome": "google chrome", "settings": "system settings", "system preferences": "system settings",
}
PROTECTED = {"Finder", "Dock", "Terminal", "Visual Studio Code", "iTerm", "Ollama"}
COMMAND_TIMEOUT_S = 10    # subprocess kills the command after this
MAX_SUGGESTIONS = 5
QUIT_WAIT_S = 3.0         # an app needs a moment to really close after "quit"


# ------------------------------------------------------------ helpers (pure, easy to test)
def _normalise_name(name: str) -> str:
    """'  Safari.app ' -> 'safari', 'VS Code' -> 'visual studio code'."""
    wanted = name.strip().lower().removesuffix(".app").strip()
    return ALIASES.get(wanted, wanted)


def protected_name(name: str) -> str | None:
    """The protected app this name refers to (e.g. 'finder' -> 'Finder'), or None."""
    wanted = _normalise_name(name)
    for app in PROTECTED:
        if app.lower() == wanted:
            return app
    return None


def installed_apps(folders=APP_FOLDERS) -> dict[str, Path]:
    """{'spotify': Path('/Applications/Spotify.app'), ...}: lowercase name -> app bundle."""
    apps = {}
    for folder in folders:
        if folder.is_dir():
            for bundle in sorted(folder.glob("*.app")):
                apps.setdefault(bundle.stem.lower(), bundle)
    return apps


def find_app(name: str, apps: dict[str, Path]) -> tuple[Path | None, list[str]]:
    """Find an installed app by name. Returns (bundle, []) or (None, suggestions)."""
    wanted = _normalise_name(name)
    if not wanted:
        return None, []
    if wanted in apps:
        return apps[wanted], []
    matches = sorted(key for key in apps if wanted in key)
    if len(matches) == 1:
        return apps[matches[0]], []
    if not matches:   # no partial match: look for typos, e.g. "texteditor" -> "textedit"
        matches = difflib.get_close_matches(wanted, list(apps), n=MAX_SUGGESTIONS, cutoff=0.6)
    return None, [apps[key].stem for key in matches[:MAX_SUGGESTIONS]]


def _not_found(app_name: str, suggestions: list[str]) -> ToolResult:
    if suggestions:
        return ToolResult(False, f"No app called '{app_name}' is installed. "
                                 f"Did you mean: {', '.join(suggestions)}?")
    return ToolResult(False, f"No app called '{app_name}' is installed on this Mac.")


def _protected(name: str) -> ToolResult:
    return ToolResult(False, f"{name} is protected: quitting it would disrupt the Mac or Jarvis himself.")


def _run(args: list[str]) -> subprocess.CompletedProcess:
    """Run a command WITHOUT a shell and with a timeout (raises subprocess.TimeoutExpired)."""
    return subprocess.run(args, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S)


def _osascript(*lines: str, args: tuple = ()) -> subprocess.CompletedProcess:
    """Run AppleScript lines; values in `args` arrive as argv, never inside the script text."""
    command = ["osascript"]
    for line in lines:
        command += ["-e", line]
    return _run(command + list(args))


def is_running(app_name: str) -> bool:
    result = _osascript("on run argv",
                        "return application (item 1 of argv) is running",
                        "end run", args=(app_name,))
    return result.stdout.strip() == "true"


def _wait_until_closed(app_name: str) -> bool:
    """Check a few times, up to QUIT_WAIT_S seconds, whether the app has really closed."""
    deadline = time.monotonic() + QUIT_WAIT_S
    while time.monotonic() < deadline:
        if not is_running(app_name):
            return True
        time.sleep(0.25)
    return not is_running(app_name)


# ------------------------------------------------------------ tools
@tool("Open an application on this Mac by its name.",
      {"app_name": ("string", "The application's name, e.g. Spotify or Safari")})
def open_app(app_name):
    bundle, suggestions = find_app(app_name, installed_apps())
    if bundle is None:
        return _not_found(app_name, suggestions)
    result = _run(["open", "-a", str(bundle)])
    if result.returncode != 0:
        return ToolResult(False, f"macOS could not open {bundle.stem}: {result.stderr.strip()}")
    return ToolResult(True, f"Opened {bundle.stem}.")


@tool("Quit an application on this Mac. Always use this tool to quit an app: it checks by itself "
      "whether the app is running or protected. Sir is asked to confirm first.",
      {"app_name": ("string", "The application's name, e.g. Spotify")}, risk="confirm", timeout_s=15)
def quit_app(app_name):
    protected = protected_name(app_name)       # first, so system apps outside /Applications are covered
    if protected:
        return _protected(protected)
    bundle, suggestions = find_app(app_name, installed_apps())
    if bundle is None:
        return _not_found(app_name, suggestions)
    name = bundle.stem
    if name in PROTECTED:                      # e.g. an alias or partial name that led to a protected app
        return _protected(name)
    if not is_running(name):
        return ToolResult(False, f"{name} is not running, so there was nothing to quit.")

    result = _osascript("on run argv",
                        "with timeout of 5 seconds",
                        "tell application (item 1 of argv) to quit",
                        "end timeout",
                        "end run", args=(name,))
    if result.returncode != 0:
        if "-1743" in result.stderr:   # macOS Automation permission was refused
            return ToolResult(False, f"macOS did not allow controlling {name}. Sir can allow it in "
                                     "System Settings, Privacy and Security, Automation.")
        if "-1712" in result.stderr:   # timed out: usually a "save changes?" dialog
            return ToolResult(False, f"{name} is still open: it is probably asking whether to save something.")
        return ToolResult(False, f"{name} did not quit: {result.stderr.strip()}")

    if not _wait_until_closed(name):   # never claim success without checking
        return ToolResult(False, f"{name} was asked to quit but is still running.")
    return ToolResult(True, f"Closed {name}.")
