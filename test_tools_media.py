"""
Unit tests for the media tools (Phase 2, step 4.4). Nothing touches the real Mac:
a fake osascript answers like Spotify would, and records every script it got.

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
        for target, value in [("installed_apps", {"spotify": Path("/Applications/Spotify.app")}),
                              ("SETTLE_S", 0)]:
            patcher = mock.patch.object(media, target, value if target == "SETTLE_S" else mock.Mock(return_value=value))
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
