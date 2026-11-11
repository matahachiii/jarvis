"""
Media tools: Spotify control via AppleScript (volume follows in step 4.5).

What AppleScript CAN do with Spotify: play, pause, next, previous, report what is playing.
What it CANNOT do: search a song, artist or playlist by name. (Possible later with
Spotify's free Web API: search -> track URI -> AppleScript 'play track'.)

Safety: the AppleScript text only contains fixed commands; `action` comes from a fixed
list, so no user text ever reaches a script. The tool never starts Spotify by itself.
After every command we check what really happened before reporting success.
"""

import time

from jarvis.tools.mac_apps import _osascript, installed_apps, is_running
from jarvis.tools.registry import ToolResult, tool

SPOTIFY_ACTIONS = ["play", "pause", "next", "previous", "now_playing"]
SPOTIFY_COMMANDS = {"play": "play", "pause": "pause", "next": "next track", "previous": "previous track"}
STATE_WAIT_S = 3.0   # max wait until Spotify really plays/pauses (it may still be loading)
SETTLE_S = 0.6       # pause before reading the new track after next/previous
SEPARATOR = " || "   # between title and artist in the AppleScript output


# ------------------------------------------------------------ Spotify helpers
def _tell_spotify(command: str):
    return _osascript(f'tell application "Spotify" to {command}')


def player_state() -> str:
    """'playing', 'paused' or 'stopped' ('' if Spotify did not answer)."""
    return _tell_spotify("player state as string").stdout.strip()


def _wait_for_state(want_playing: bool) -> str:
    """Check the player state a few times until it matches, or time runs out. Returns the last state."""
    deadline = time.monotonic() + STATE_WAIT_S
    state = player_state()
    while (state == "playing") != want_playing and time.monotonic() < deadline:
        time.sleep(0.25)
        state = player_state()
    return state


def current_track() -> str | None:
    """'Title by Artist', or None if there is no current track."""
    result = _tell_spotify(f'(name of current track) & "{SEPARATOR}" & (artist of current track)')
    if result.returncode != 0 or SEPARATOR not in result.stdout:
        return None
    title, artist = result.stdout.strip().split(SEPARATOR, 1)
    return f"{title} by {artist}"


def describe_player() -> str:
    state = player_state()
    track = current_track()
    if state == "stopped" or not track:
        return "Nothing is playing on Spotify."
    return f"{'Playing' if state == 'playing' else 'Paused'}: {track}."


# ------------------------------------------------------------ tools
@tool("Control Spotify: play, pause, next or previous track, or tell what is playing now. "
      "It cannot search for a specific song, artist or playlist.",
      {"action": ("string", "What to do", SPOTIFY_ACTIONS)})
def spotify(action):
    if "spotify" not in installed_apps():
        return ToolResult(False, "Spotify is not installed on this Mac.")
    if not is_running("Spotify"):
        return ToolResult(False, "Spotify is not running. It can be opened first with open_app.")
    if action == "now_playing":
        return ToolResult(True, describe_player())

    result = _tell_spotify(SPOTIFY_COMMANDS[action])
    if result.returncode != 0:
        if "-1743" in result.stderr:
            return ToolResult(False, "macOS did not allow controlling Spotify. Sir can allow it in "
                                     "System Settings, Privacy and Security, Automation.")
        return ToolResult(False, f"Spotify did not respond: {result.stderr.strip()}")

    # Check what really happened, never assume
    if action == "play":
        state = _wait_for_state(want_playing=True)
        if state != "playing":
            return ToolResult(False, f"Spotify did not start playing (it is {state or 'not answering'}).")
    elif action == "pause":
        if _wait_for_state(want_playing=False) == "playing":
            return ToolResult(False, "Spotify is still playing.")
    else:
        time.sleep(SETTLE_S)
    return ToolResult(True, describe_player())
