"""
Unit tests for the file search tool (Phase 2, step 4.6).

Nothing touches the real Mac: real files are made in a temporary folder that plays
"home", and mdfind is replaced by a fake that lists them (plus some noise).

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
        self.root = Path(self.tmp.name)
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

    def make(self, relative, days_ago, folder=False):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
