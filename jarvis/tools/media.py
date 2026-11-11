"""
Media tools: Spotify control and the Mac's sound volume, via AppleScript.

What AppleScript CAN do with Spotify: play, pause, next, previous, report what is playing.
What it CANNOT do: search a song, artist or playlist by name. (Possible later with
Spotify's free Web API: search -> track URI -> AppleScript 'play track'.)

Safety: AppleScript text only contains fixed commands; values (like a volume level)
arrive as argv. The Spotify tool never starts Spotify by itself. After every command
we check what really happened before reporting success.
"""

import time

from jarvis.tools.mac_apps import _osascript, installed_apps, is_running
from jarvis.tools.registry import ToolResult, tool

SPOTIFY_ACTIONS = ["play", "pause", "next", "previous", "now_playing"]
SPOTIFY_COMMANDS = {"play": "play", "pause": "pause", "next": "next track", "previous": "previous track"}
STATE_WAIT_S = 3.0   # max wait until Spotify really plays/pauses (it may still be loading)
SETTLE_S = 0.6       # pause before reading the new track after next/previous
SEPARATOR = " || "   # between two values in AppleScript output

VOLUME_ACTIONS = ["get", "set", "up", "down", "mute", "unmute"]
VOLUME_STEP = 10     # how much "up" / "down" change the volume
VOLUME_TOLERANCE = 3  # macOS may read back 31 after setting 30


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


# ------------------------------------------------------------ volume helpers
def volume_settings() -> tuple[int, bool]:
    """(output volume 0-100, muted?) of the Mac's current sound output."""
    result = _osascript("set s to get volume settings",
                        f'return ((output volume of s) as string) & "{SEPARATOR}" & ((output muted of s) as string)')
    if result.returncode != 0 or SEPARATOR not in result.stdout:
        raise RuntimeError("macOS did not report the volume")
    level, muted = result.stdout.strip().split(SEPARATOR, 1)
    if not level.isdigit():   # e.g. "missing value" for some HDMI or Bluetooth outputs
        raise RuntimeError("this sound output has no adjustable volume")
    return int(level), muted == "true"


def _set_level(level: int) -> None:
    _osascript("on run argv", "set volume output volume ((item 1 of argv) as integer)", "end run",
               args=(str(level),))


def _set_muted(muted: bool) -> None:
    _osascript(f"set volume output muted {'true' if muted else 'false'}")


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


@tool("Get or change the Mac's sound volume from 0 to 100: get, set to a level, up, down, mute or unmute.",
      {"action": ("string", "What to do", VOLUME_ACTIONS),
       "level": ("integer", "Volume from 0 to 100, only for action 'set'")},
      optional=("level",))
def volume(action, level=None):
    before, muted = volume_settings()

    if action == "get":
        return ToolResult(True, f"The volume is {before} percent" + (", but sound is muted." if muted else "."))

    if action in ("mute", "unmute"):
        _set_muted(action == "mute")
        now_level, now_muted = volume_settings()   # read again: some Macs restore an older level on unmute
        if now_muted != (action == "mute"):
            return ToolResult(False, "macOS did not change the mute setting.")
        return ToolResult(True, "Sound is muted." if now_muted else f"Sound is on, at {now_level} percent.")

    if action == "set":
        if level is None:
            return ToolResult(False, "Action 'set' needs a level from 0 to 100.")
        if not 0 <= level <= 100:
            return ToolResult(False, f"{level} is not a valid volume: use a level from 0 to 100.")
        target = level
    else:   # up / down, kept within 0-100
        target = min(100, max(0, before + (VOLUME_STEP if action == "up" else -VOLUME_STEP)))

    _set_level(target)
    if muted:
        _set_muted(False)   # whoever changes the volume wants to hear something
    after, _ = volume_settings()
    if abs(after - target) > VOLUME_TOLERANCE:
        return ToolResult(False, f"The volume is {after} percent, not {target} as asked.")
    return ToolResult(True, f"The volume is now {after} percent (it was {before}).")
