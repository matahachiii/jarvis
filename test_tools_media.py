"""
Unit tests for the media tools (Phase 2, steps 4.4 and 4.5). Nothing touches the real Mac:
fakes answer like Spotify / the Mac's sound settings would, and record every script they got.

Usage:  python -m unittest test_tools_media -v
"""

import subprocess
import unittest
from pathlib import Path
from unittest import mock

from jarvis.tools import media
from jarvis.tools.registry import registry


def done(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# ============================================================ Spotify
class FakeSpotify:
    """Stands in for subprocess.run: answers like Spotify, records every script line."""

    def __init__(self, state="playing", running=True, track="Bohemian Rhapsody || Queen", fail=None):
        self.state, self.running, self.track, self.fail = state, running, track, fail
        self.scripts = []

    def __call__(self, args, **kwargs):
        self.scripts += args[1:]
        script = " ".join(args)
        if "is running" in script:
            return done("true\n" if self.running else "false\n")
        if "player state" in script:
            return done(self.state + "\n")
        if "current track" in script:
            return done(self.track + "\n")
        if self.fail:
            return done(returncode=1, stderr=self.fail)
        return done()

    def sent(self, command):
        return f'tell application "Spotify" to {command}' in self.scripts


class SpotifyTestCase(unittest.TestCase):

    def setUp(self):
        patches = [
            mock.patch.object(media, "installed_apps",
                              mock.Mock(return_value={"spotify": Path("/Applications/Spotify.app")})),
            mock.patch.object(media, "SETTLE_S", 0),
            mock.patch.object(media, "STATE_WAIT_S", 0.3),   # don't make failure tests wait 3 seconds
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_tool(self, action, **fake_args):
        self.fake = FakeSpotify(**fake_args)
        with mock.patch("jarvis.tools.mac_apps.subprocess.run", self.fake):
            return media.spotify(action)


class TestSpotify(SpotifyTestCase):

    def test_now_playing(self):
        result = self.run_tool("now_playing")
        self.assertTrue(result.success)
        self.assertEqual(result.message, "Playing: Bohemian Rhapsody by Queen.")

    def test_nothing_playing(self):
        self.assertEqual(self.run_tool("now_playing", state="stopped").message, "Nothing is playing on Spotify.")

    def test_play(self):
        result = self.run_tool("play", state="playing")
        self.assertTrue(result.success)
        self.assertTrue(self.fake.sent("play"))
        self.assertIn("Bohemian Rhapsody", result.message)

    def test_play_that_does_not_start_is_a_failure(self):
        result = self.run_tool("play", state="paused")
        self.assertFalse(result.success)
        self.assertIn("did not start", result.message)

    def test_pause(self):
        result = self.run_tool("pause", state="paused")
        self.assertTrue(result.success)
        self.assertEqual(result.message, "Paused: Bohemian Rhapsody by Queen.")

    def test_pause_that_keeps_playing_is_a_failure(self):
        self.assertFalse(self.run_tool("pause", state="playing").success)

    def test_next_and_previous(self):
        self.run_tool("next")
        self.assertTrue(self.fake.sent("next track"))
        self.run_tool("previous")
        self.assertTrue(self.fake.sent("previous track"))

    def test_not_running_never_starts_spotify(self):
        result = self.run_tool("pause", running=False)
        self.assertFalse(result.success)
        self.assertIn("open_app", result.message)
        # only the "is it running?" check was sent, never a command to Spotify itself
        self.assertFalse(any(s.startswith('tell application "Spotify"') for s in self.fake.scripts))

    def test_permission_refused(self):
        result = self.run_tool("play", fail="Not authorized (-1743)")
        self.assertFalse(result.success)
        self.assertIn("Automation", result.message)

    def test_not_installed(self):
        with mock.patch.object(media, "installed_apps", mock.Mock(return_value={})):
            self.assertFalse(media.spotify("play").success)


class TestSpotifyRegistration(unittest.TestCase):

    def test_safe_and_in_registry(self):
        self.assertEqual(registry.get("spotify").risk, "safe")

    def test_unknown_action_rejected_before_anything_runs(self):
        with mock.patch("jarvis.tools.mac_apps.subprocess.run") as run:
            result = registry.execute("spotify", {"action": "play Bohemian Rhapsody"})
        self.assertFalse(result.success)
        self.assertIn("must be one of", result.message)
        run.assert_not_called()


# ============================================================ Volume
class FakeAudio:
    """Stands in for subprocess.run: behaves like the Mac's sound settings."""

    def __init__(self, start=40, muted=False, stuck=False, report="normal"):
        self.level, self.muted, self.stuck, self.report = start, muted, stuck, report
        self.scripts = []

    def __call__(self, args, **kwargs):
        self.scripts.append(args)
        script = " ".join(args)
        if "get volume settings" in script:
            if self.report == "missing":
                return done(f"missing value{media.SEPARATOR}false\n")
            return done(f"{self.level}{media.SEPARATOR}{'true' if self.muted else 'false'}\n")
        if "output volume ((item 1 of argv)" in script:
            if not self.stuck:
                self.level = int(args[-1])
            return done()
        if "output muted true" in script:
            self.muted = True
        elif "output muted false" in script:
            self.muted = False
        return done()


class TestVolume(unittest.TestCase):

    def run_tool(self, action, level=None, **fake_args):
        """Run the volume tool against a fake Mac (fake_args: start, muted, stuck, report)."""
        self.audio = FakeAudio(**fake_args)
        with mock.patch("jarvis.tools.mac_apps.subprocess.run", self.audio):
            return media.volume(action, level)

    def test_get(self):
        self.assertEqual(self.run_tool("get").message, "The volume is 40 percent.")

    def test_get_while_muted(self):
        self.assertIn("muted", self.run_tool("get", muted=True).message)

    def test_set(self):
        result = self.run_tool("set", 30)
        self.assertTrue(result.success)
        self.assertEqual(result.message, "The volume is now 30 percent (it was 40).")
        set_command = next(a for a in self.audio.scripts if "argv" in " ".join(a))
        self.assertEqual(set_command[-1], "30")                    # the level arrives as argv ...
        self.assertNotIn("30", " ".join(set_command[:-1]))         # ... never inside the script

    def test_set_needs_a_level(self):
        self.assertFalse(self.run_tool("set").success)

    def test_set_out_of_range_changes_nothing(self):
        result = self.run_tool("set", 150)
        self.assertFalse(result.success)
        self.assertEqual(self.audio.level, 40)

    def test_up(self):
        self.assertEqual(self.run_tool("up").message, "The volume is now 50 percent (it was 40).")

    def test_up_never_above_100(self):
        self.assertEqual(self.run_tool("up", start=95).message, "The volume is now 100 percent (it was 95).")

    def test_down(self):
        self.assertEqual(self.run_tool("down").message, "The volume is now 30 percent (it was 40).")

    def test_down_never_below_zero(self):
        self.assertEqual(self.run_tool("down", start=5).message, "The volume is now 0 percent (it was 5).")

    def test_changing_volume_unmutes(self):
        result = self.run_tool("set", 25, muted=True)
        self.assertTrue(result.success)
        self.assertFalse(self.audio.muted)

    def test_mute_and_unmute(self):
        self.assertEqual(self.run_tool("mute").message, "Sound is muted.")
        self.assertTrue(self.audio.muted)
        self.assertEqual(self.run_tool("unmute", muted=True).message, "Sound is on, at 40 percent.")

    def test_volume_that_does_not_change_is_a_failure(self):
        result = self.run_tool("set", 70, stuck=True)
        self.assertFalse(result.success)
        self.assertIn("not 70", result.message)

    def test_output_without_volume_fails_gracefully(self):
        with mock.patch("jarvis.tools.mac_apps.subprocess.run", FakeAudio(report="missing")):
            result = registry.execute("volume", {"action": "get"})
        self.assertFalse(result.success)
        self.assertIn("no adjustable volume", result.message)

    def test_through_registry_with_text_level(self):
        audio = FakeAudio()
        with mock.patch("jarvis.tools.mac_apps.subprocess.run", audio):
            result = registry.execute("volume", {"action": "set", "level": "30"})   # small models send "30"
        self.assertTrue(result.success)
        self.assertEqual(audio.level, 30)
        self.assertEqual(registry.get("volume").risk, "safe")


if __name__ == "__main__":
    unittest.main(verbosity=2)
