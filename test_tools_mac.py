"""
Unit tests for the app tools (Phase 2, step 4.3).

Nothing touches the real Mac: fake .app folders live in a temporary directory,
and subprocess.run is replaced by a mock that only records the command.

Usage:  python -m unittest test_tools_mac -v
"""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jarvis.tools import mac_apps
from jarvis.tools.registry import registry

FAKE_APPS = ["Spotify", "Safari", "Visual Studio Code", "Xcode", "Google Chrome",
             "Microsoft Word", "Calculator", "Finder"]


def done(stdout="", returncode=0, stderr=""):
    """A fake finished command."""
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class FakeAppsTestCase(unittest.TestCase):
    """Builds fake apps in a temp folder and points the tools at them."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        folder = Path(self.tmp.name)
        for name in FAKE_APPS:
            (folder / f"{name}.app").mkdir()
        (folder / "notes.txt").write_text("not an app")
        self.apps = mac_apps.installed_apps([folder, folder / "missing"])
        patcher = mock.patch.object(mac_apps, "installed_apps", return_value=self.apps)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)


class TestFindApp(FakeAppsTestCase):

    def find(self, name):
        bundle, suggestions = mac_apps.find_app(name, self.apps)
        return (bundle.stem if bundle else None), suggestions

    def test_index(self):
        self.assertEqual(len(self.apps), len(FAKE_APPS))   # notes.txt and the missing folder are ignored

    def test_matching(self):
        for name, expected in [("spotify", "Spotify"), ("SAFARI", "Safari"), ("Safari.app", "Safari"),
                               ("vs code", "Visual Studio Code"), ("chrome", "Google Chrome"),
                               ("word", "Microsoft Word")]:
            with self.subTest(name=name):
                self.assertEqual(self.find(name)[0], expected)

    def test_ambiguous_gives_suggestions(self):
        bundle, suggestions = self.find("co")   # Visual Studio Code and Xcode
        self.assertIsNone(bundle)
        self.assertEqual(sorted(suggestions), ["Visual Studio Code", "Xcode"])

    def test_not_installed(self):
        self.assertEqual(self.find("Photoshop"), (None, []))
        self.assertEqual(self.find(""), (None, []))


class TestOpenApp(FakeAppsTestCase):

    @mock.patch("jarvis.tools.mac_apps.subprocess.run", return_value=done())
    def test_opens_with_argument_list_and_no_shell(self, run):
        result = mac_apps.open_app("spotify")
        self.assertTrue(result.success)
        self.assertEqual(result.message, "Opened Spotify.")
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["open", "-a", str(self.apps["spotify"])])
        self.assertNotIn("shell", kwargs)
        self.assertIn("timeout", kwargs)

    @mock.patch("jarvis.tools.mac_apps.subprocess.run")
    def test_not_installed_runs_nothing(self, run):
        result = mac_apps.open_app("Photoshop")
        self.assertFalse(result.success)
        self.assertIn("No app called", result.message)
        run.assert_not_called()

    @mock.patch("jarvis.tools.mac_apps.subprocess.run", return_value=done(returncode=1, stderr="boom"))
    def test_open_failure_is_reported(self, run):
        self.assertFalse(mac_apps.open_app("Safari").success)


class TestQuitApp(FakeAppsTestCase):

    @mock.patch("jarvis.tools.mac_apps.subprocess.run")
    def test_protected_app(self, run):
        result = mac_apps.quit_app("Finder")
        self.assertFalse(result.success)
        self.assertIn("protected", result.message)
        run.assert_not_called()

    @mock.patch("jarvis.tools.mac_apps.subprocess.run", return_value=done("false\n"))
    def test_not_running(self, run):
        result = mac_apps.quit_app("Calculator")
        self.assertFalse(result.success)
        self.assertIn("not running", result.message)
        self.assertEqual(run.call_count, 1)   # only the "is it running?" check

    @mock.patch("jarvis.tools.mac_apps.subprocess.run",
                side_effect=[done("true\n"), done(), done("false\n")])
    def test_quit_success_is_verified(self, run):
        result = mac_apps.quit_app("calculator")
        self.assertTrue(result.success)
        self.assertEqual(result.message, "Closed Calculator.")
        quit_command = run.call_args_list[1][0][0]
        self.assertEqual(quit_command[0], "osascript")
        self.assertEqual(quit_command[-1], "Calculator")   # the name arrives as argv ...
        self.assertNotIn("Calculator", " ".join(quit_command[:-1]))   # ... never inside the script

    @mock.patch.object(mac_apps, "QUIT_WAIT_S", 0.3)   # don't make the test wait 3 seconds
    @mock.patch("jarvis.tools.mac_apps.subprocess.run", return_value=done("true\n"))
    def test_still_running_after_quit(self, run):
        self.assertFalse(mac_apps.quit_app("Calculator").success)
        self.assertGreater(run.call_count, 3)   # it really checked several times before giving up

    @mock.patch("jarvis.tools.mac_apps.subprocess.run",
                side_effect=[done("true\n"), done(returncode=1, stderr="Not authorized (-1743)")])
    def test_permission_refused(self, run):
        result = mac_apps.quit_app("Calculator")
        self.assertFalse(result.success)
        self.assertIn("Automation", result.message)

    @mock.patch("jarvis.tools.mac_apps.subprocess.run",
                side_effect=[done("true\n"), done(returncode=1, stderr="AppleEvent timed out (-1712)")])
    def test_save_dialog(self, run):
        self.assertIn("save", mac_apps.quit_app("Calculator").message)


class TestRegistration(FakeAppsTestCase):

    def test_risk_levels(self):
        self.assertEqual(registry.get("open_app").risk, "safe")
        self.assertEqual(registry.get("quit_app").risk, "confirm")

    @mock.patch("jarvis.tools.mac_apps.subprocess.run")
    def test_quit_without_yes_runs_nothing(self, run):
        result = registry.execute("quit_app", {"app_name": "Calculator"})   # no confirm callback = no
        self.assertFalse(result.success)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
