"""
Unit tests for the file tools (Phase 2, steps 4.6 and 4.7).

Nothing touches the real Mac: real files are made in a temporary folder that plays
"home", and mdfind / open are replaced by fakes that only record the command.

Usage:  python -m unittest test_tools_files -v
"""

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from jarvis.tools import files
from jarvis.tools.registry import registry


def done(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class FakeHomeTestCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        # (relative path, days ago it was modified)
        self.make("Documents/thesis_draft.docx", days_ago=30)
        self.make("Desktop/thesis_final.pdf", days_ago=1)
        self.make("Documents/Thesis", days_ago=10, folder=True)
        self.make("Library/Caches/thesis_cache.db", days_ago=0)   # noise: must be skipped
        self.make(".hidden/thesis.txt", days_ago=0)               # hidden: skipped
        self.make("Desktop/Jarvis/.venv/thesis.py", days_ago=0)   # inside .venv: skipped
        patcher = mock.patch.object(files, "SEARCH_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make(self, relative, days_ago=0, folder=False):
        path = self.root / relative
        if folder:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")
        moment = time.time() - days_ago * 86400
        os.utime(path, (moment, moment))
        return path

    def fake_mdfind(self, *extra_lines, returncode=0, stderr=""):
        """A fake subprocess.run that lists every file under root (plus extra lines) as Spotlight would."""
        listed = [str(p) for p in self.root.rglob("*thesis*")] + [str(p) for p in self.root.rglob("*Thesis*")]
        lines = sorted(set(listed)) + list(extra_lines)
        return mock.Mock(return_value=done("\n".join(lines) + "\n", returncode, stderr))

    def search(self, query="thesis", run=None):
        self.run = run or self.fake_mdfind("/tmp/elsewhere/thesis.txt")   # outside home: skipped
        with mock.patch("jarvis.tools.mac_apps.subprocess.run", self.run):
            return files.search_files(query)

    def open(self, target, run=None):
        self.run = run or mock.Mock(return_value=done())
        with mock.patch("jarvis.tools.mac_apps.subprocess.run", self.run):
            return files.open_item(target)


# ============================================================ search_files
class TestSearchFiles(FakeHomeTestCase):

    def test_command_is_safe(self):
        self.search("thesis")
        args, kwargs = self.run.call_args
        self.assertEqual(args[0], ["mdfind", "-onlyin", str(self.root), "-name", "thesis"])
        self.assertNotIn("shell", kwargs)
        self.assertIn("timeout", kwargs)

    def test_noise_is_skipped(self):
        message = self.search().message
        self.assertTrue(message.startswith("Found 3 matches"), message)
        for noise in ["Library", ".hidden", ".venv", "elsewhere"]:
            self.assertNotIn(noise, message)

    def test_most_recent_first(self):
        message = self.search().message
        self.assertLess(message.index("thesis_final.pdf"), message.index("Thesis (folder)"))
        self.assertLess(message.index("Thesis (folder)"), message.index("thesis_draft.docx"))

    def test_paths_and_dates(self):
        message = self.search().message
        self.assertIn("~/Desktop/thesis_final.pdf, modified", message)
        self.assertIn("~/Documents/Thesis (folder)", message)

    def test_only_the_newest_are_shown(self):
        for i in range(6):   # 6 old copies: 100 to 105 days old
            self.make(f"Downloads/thesis_copy_{i}.txt", days_ago=100 + i)
        message = self.search().message
        self.assertIn("Found 9 matches, the 5 most recent", message)
        self.assertIn("thesis_final.pdf", message)        # newest
        self.assertIn("thesis_copy_1.txt", message)       # 5th newest
        for old in range(2, 6):                           # the 4 oldest copies are left out
            self.assertNotIn(f"thesis_copy_{old}.txt", message)

    def test_no_matches_is_not_a_failure(self):
        result = self.search("zzz", run=mock.Mock(return_value=done("")))
        self.assertTrue(result.success)
        self.assertIn("Found no files", result.message)

    def test_dash_is_rejected_before_running(self):
        run = mock.Mock()
        result = self.search("-onlyin", run=run)
        self.assertFalse(result.success)
        run.assert_not_called()

    def test_spotlight_failure_is_reported(self):
        result = self.search(run=mock.Mock(return_value=done(returncode=1, stderr="Spotlight is off")))
        self.assertFalse(result.success)
        self.assertIn("Spotlight", result.message)


class TestSearchRegistration(FakeHomeTestCase):

    def test_safe_and_validated(self):
        self.assertEqual(registry.get("search_files").risk, "safe")
        with mock.patch("jarvis.tools.mac_apps.subprocess.run") as run:
            self.assertFalse(registry.execute("search_files", {"query": "   "}).success)
            self.assertFalse(registry.execute("search_files", {"query": "x" * 1000}).success)
        run.assert_not_called()


# ============================================================ open_item: web pages
class TestOpenWeb(FakeHomeTestCase):

    def test_https_url(self):
        result = self.open("https://www.vub.be")
        self.assertTrue(result.success)
        self.assertEqual(self.run.call_args[0][0], ["open", "https://www.vub.be"])

    def test_bare_domains_become_https(self):
        for target, url in [("www.youtube.com", "https://www.youtube.com"), ("vub.be", "https://vub.be"),
                            ("github.com/matahachiii", "https://github.com/matahachiii")]:
            with self.subTest(target=target):
                self.open(target)
                self.assertEqual(self.run.call_args[0][0], ["open", url])

    def test_file_names_are_not_mistaken_for_websites(self):
        self.assertIsNone(files.as_url("report.pdf"))
        self.assertIsNone(files.as_url("notes.txt"))

    def test_other_schemes_are_refused(self):
        for target in ["javascript:alert(1)", "file:///etc/passwd", "ftp://example.com", "ssh://server"]:
            with self.subTest(target=target):
                run = mock.Mock()
                self.assertFalse(self.open(target, run=run).success)
                run.assert_not_called()


# ============================================================ open_item: files and folders
class TestOpenFiles(FakeHomeTestCase):

    def test_file_from_search_result(self):
        result = self.open("~/Desktop/thesis_final.pdf")
        self.assertTrue(result.success)
        self.assertEqual(result.message, "Opened ~/Desktop/thesis_final.pdf.")
        self.assertEqual(self.run.call_args[0][0], ["open", str(self.root / "Desktop/thesis_final.pdf")])

    def test_relative_path_is_inside_home(self):
        self.assertTrue(self.open("Documents/thesis_draft.docx").success)

    def test_folder_opens_in_finder(self):
        self.assertEqual(self.open("~/Documents/Thesis").message, "Opened ~/Documents/Thesis in Finder.")

    def test_missing_file(self):
        result = self.open("~/Documents/nope.pdf")
        self.assertFalse(result.success)
        self.assertIn("search_files", result.message)
        self.run.assert_not_called()

    def test_outside_home_is_refused(self):
        for target in ["/etc/hosts", "~/../../etc/hosts", "../../../etc/hosts"]:
            with self.subTest(target=target):
                run = mock.Mock()
                self.assertFalse(self.open(target, run=run).success)
                run.assert_not_called()

    def test_symlink_out_of_home_is_refused(self):
        (self.root / "sneaky_link").symlink_to("/etc")
        run = mock.Mock()
        self.assertFalse(self.open("~/sneaky_link/hosts", run=run).success)
        run.assert_not_called()

    def test_programs_and_scripts_are_never_opened(self):
        self.make("Desktop/run_me.command")
        self.make("Desktop/install.sh")
        self.make("Desktop/script.py")
        self.make("Applications/Evil.app", folder=True)
        executable = self.make("Desktop/no_extension_tool")
        executable.chmod(0o755)
        for target in ["~/Desktop/run_me.command", "~/Desktop/install.sh", "~/Desktop/script.py",
                       "~/Applications/Evil.app", "~/Desktop/no_extension_tool"]:
            with self.subTest(target=target):
                run = mock.Mock()
                result = self.open(target, run=run)
                self.assertFalse(result.success)
                self.assertIn("never opened", result.message)
                run.assert_not_called()

    def test_open_failure_is_reported(self):
        result = self.open("~/Desktop/thesis_final.pdf", run=mock.Mock(return_value=done(returncode=1, stderr="boom")))
        self.assertFalse(result.success)

    def test_safe_in_registry(self):
        self.assertEqual(registry.get("open_item").risk, "safe")


if __name__ == "__main__":
    unittest.main(verbosity=2)
