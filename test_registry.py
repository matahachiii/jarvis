"""
Unit tests for the tool registry (Phase 2, step 2; duration fix in step 5c.1).

No LLM and no real Mac actions: only dummy tools. The `calls` list records
every time a tool function really runs, so we can prove that a "confirm"
tool never runs without a yes.

Usage:  python -m unittest test_registry -v
"""

import time
import unittest

from jarvis.tools.registry import MAX_RESULT_CHARS, ToolRegistry, ToolResult


def make_registry():
    """A fresh registry with dummy tools, plus the list of real executions."""
    reg = ToolRegistry()
    calls = []

    @reg.tool("Set volume.", {"level": ("integer", "Volume 0-100")})
    def set_volume(level):
        calls.append(("set_volume", level))
        return ToolResult(True, f"Volume set to {level} ({type(level).__name__}).")

    @reg.tool("Media control.", {"action": ("string", "What to do", ["play", "pause"])})
    def media(action):
        calls.append(("media", action))
        return ToolResult(True, f"{action} done")

    @reg.tool("Greet someone.", {"name": ("string", "Who"), "loud": ("boolean", "Shout?")},
              optional=("loud",))
    def greet(name, loud=False):
        calls.append(("greet", name, loud))
        text = f"Hello {name}"
        return ToolResult(True, text.upper() if loud else text)

    @reg.tool("Always crashes.")
    def crash():
        return 1 / 0

    @reg.tool("Very slow.", timeout_s=0.2)
    def slow():
        time.sleep(0.5)
        return ToolResult(True, "finally")

    @reg.tool("Returns the wrong type.")
    def broken():
        return "just a string"

    @reg.tool("Long output.")
    def chatty():
        return ToolResult(True, "word " * 200)

    @reg.tool("Quit an app.", {"app_name": ("string", "App name")}, risk="confirm")
    def quit_app(app_name):
        calls.append(("quit_app", app_name))
        return ToolResult(True, f"Quit {app_name}.")

    return reg, calls, set_volume


class TestRegistration(unittest.TestCase):

    def test_decorator_returns_function_unchanged(self):
        _, calls, set_volume = make_registry()
        self.assertTrue(set_volume(10).success)
        self.assertEqual(calls, [("set_volume", 10)])

    def test_names(self):
        reg, _, _ = make_registry()
        self.assertEqual(len(reg.names), 8)
        self.assertIn("quit_app", reg.names)

    def test_schema_format(self):
        reg, _, _ = make_registry()
        schema = reg.get("greet").schema()
        self.assertEqual(schema["type"], "function")
        params = schema["function"]["parameters"]
        self.assertEqual(params["required"], ["name"])          # optional 'loud' is not required
        self.assertEqual(params["properties"]["loud"]["type"], "boolean")
        media = reg.get("media").schema()["function"]["parameters"]["properties"]["action"]
        self.assertEqual(media["enum"], ["play", "pause"])
        self.assertEqual(len(reg.schemas()), 8)

    def test_bad_definitions_are_rejected(self):
        reg = ToolRegistry()
        with self.assertRaises(ValueError):
            reg.tool("x", risk="yolo")
        with self.assertRaises(ValueError):
            reg.tool("x", {"a": ("text", "?")})
        with self.assertRaises(ValueError):
            reg.tool("x", optional=("ghost",))
        with self.assertRaises(ValueError):   # argument name does not match
            reg.tool("x", {"a": ("string", "?")})(lambda b: None)
        with self.assertRaises(ValueError):   # optional parameter without a default
            reg.tool("x", {"a": ("string", "?")}, optional=("a",))(lambda a: None)

    def test_duplicate_name_rejected(self):
        reg = ToolRegistry()
        reg.tool("x", name="dup")(lambda: None)
        with self.assertRaises(ValueError):
            reg.tool("y", name="dup")(lambda: None)


class TestValidation(unittest.TestCase):

    def setUp(self):
        self.reg, self.calls, _ = make_registry()

    def test_success(self):
        r = self.reg.execute("set_volume", {"level": 30})
        self.assertTrue(r.success)
        self.assertIsNone(r.confirmed)

    def test_number_as_text_is_converted(self):
        r = self.reg.execute("set_volume", {"level": "30"})
        self.assertTrue(r.success)
        self.assertEqual(self.calls, [("set_volume", 30)])
        self.assertIn("(int)", r.message)

    def test_wrong_types_rejected(self):
        for bad in ["loud", 30.5, True, [30]]:
            with self.subTest(level=bad):
                self.assertFalse(self.reg.execute("set_volume", {"level": bad}).success)
        self.assertEqual(self.calls, [])

    def test_missing_and_unknown_arguments(self):
        self.assertIn("Missing", self.reg.execute("set_volume", {}).message)
        self.assertIn("Missing", self.reg.execute("set_volume", {"level": None}).message)
        self.assertIn("Unknown", self.reg.execute("set_volume", {"level": 1, "mute": True}).message)
        self.assertFalse(self.reg.execute("set_volume", "loud").success)   # not a dict at all
        self.assertEqual(self.calls, [])

    def test_optional_argument(self):
        self.assertEqual(self.reg.execute("greet", {"name": "sir"}).message, "Hello sir")
        self.assertEqual(self.reg.execute("greet", {"name": "sir", "loud": None}).message, "Hello sir")
        self.assertEqual(self.reg.execute("greet", {"name": "sir", "loud": "true"}).message, "HELLO SIR")

    def test_enum(self):
        self.assertTrue(self.reg.execute("media", {"action": "play"}).success)
        self.assertFalse(self.reg.execute("media", {"action": "rewind"}).success)

    def test_strings(self):
        self.assertFalse(self.reg.execute("greet", {"name": "   "}).success)
        self.assertFalse(self.reg.execute("greet", {"name": "x" * 1000}).success)
        self.assertEqual(self.reg.execute("greet", {"name": "  sir  "}).message, "Hello sir")

    def test_unknown_tool(self):
        r = self.reg.execute("rm_rf", {})
        self.assertFalse(r.success)
        self.assertIn("no tool", r.message)


class TestConfirmation(unittest.TestCase):

    def setUp(self):
        self.reg, self.calls, _ = make_registry()

    def test_no_callback_means_no(self):
        r = self.reg.execute("quit_app", {"app_name": "Spotify"})
        self.assertFalse(r.success)
        self.assertFalse(r.confirmed)
        self.assertEqual(self.calls, [])

    def test_callback_says_no(self):
        r = self.reg.execute("quit_app", {"app_name": "Spotify"}, confirm=lambda tool, args: False)
        self.assertFalse(r.success)
        self.assertFalse(r.confirmed)
        self.assertEqual(self.calls, [])

    def test_crashing_callback_means_no(self):
        def broken_confirm(tool, args):
            raise RuntimeError("microphone died")
        r = self.reg.execute("quit_app", {"app_name": "Spotify"}, confirm=broken_confirm)
        self.assertFalse(r.success)
        self.assertEqual(self.calls, [])

    def test_callback_says_yes(self):
        seen = []

        def yes(tool, args):
            seen.append((tool.name, args))
            return True

        r = self.reg.execute("quit_app", {"app_name": " Spotify "}, confirm=yes)
        self.assertTrue(r.success)
        self.assertTrue(r.confirmed)
        self.assertEqual(seen, [("quit_app", {"app_name": "Spotify"})])   # sees the CLEANED args
        self.assertEqual(self.calls, [("quit_app", "Spotify")])

    def test_invalid_args_never_reach_confirmation(self):
        asked = []
        r = self.reg.execute("quit_app", {}, confirm=lambda tool, args: asked.append(1) or True)
        self.assertFalse(r.success)
        self.assertEqual(asked, [])
        self.assertEqual(self.calls, [])

    def test_safe_tool_never_asks(self):
        asked = []
        r = self.reg.execute("set_volume", {"level": 5},
                             confirm=lambda tool, args: asked.append(1) or False)
        self.assertTrue(r.success)
        self.assertEqual(asked, [])


class TestRobustness(unittest.TestCase):

    def setUp(self):
        self.reg, self.calls, _ = make_registry()

    def test_exception_becomes_failure(self):
        r = self.reg.execute("crash")
        self.assertFalse(r.success)
        self.assertIn("division by zero", r.message)

    def test_timeout(self):
        r = self.reg.execute("slow")
        self.assertFalse(r.success)
        self.assertIn("0.2 seconds", r.message)
        self.assertLess(r.duration_s, 0.45)   # we stopped waiting, we did not wait the full 0.5 s

    def test_wrong_return_type(self):
        r = self.reg.execute("broken")
        self.assertFalse(r.success)
        self.assertIn("unexpected", r.message)

    def test_long_result_truncated(self):
        r = self.reg.execute("chatty")
        self.assertLessEqual(len(r.message), MAX_RESULT_CHARS)
        self.assertTrue(r.message.endswith("(truncated)"))

    def test_duration_recorded(self):
        r = self.reg.execute("set_volume", {"level": 5})
        self.assertGreaterEqual(r.duration_s, 0)


class TestDuration(unittest.TestCase):
    """duration_s = time the tool itself ran. Sir's y/N thinking time must NOT count."""

    def setUp(self):
        self.reg, self.calls, _ = make_registry()

    @staticmethod
    def slow_yes(tool, args):
        time.sleep(0.3)   # sir thinks for 0.3 s, then says yes
        return True

    def test_confirmation_time_not_counted(self):
        r = self.reg.execute("quit_app", {"app_name": "Notes"}, confirm=self.slow_yes)
        self.assertTrue(r.success)
        self.assertTrue(r.confirmed)
        self.assertLess(r.duration_s, 0.1)   # quit_app is instant; the 0.3 s was sir's

    def test_tool_time_counted_but_not_confirmation(self):
        reg = ToolRegistry()

        @reg.tool("Slow confirm tool.", risk="confirm")
        def nap():
            time.sleep(0.2)
            return ToolResult(True, "rested")

        r = reg.execute("nap", {}, confirm=self.slow_yes)
        self.assertTrue(r.success)
        self.assertGreaterEqual(r.duration_s, 0.19)   # the tool's own 0.2 s is measured...
        self.assertLess(r.duration_s, 0.4)            # ...but not the 0.3 s before it

    def test_never_ran_is_zero(self):
        declined = self.reg.execute("quit_app", {"app_name": "Notes"}, confirm=lambda tool, args: False)
        invalid = self.reg.execute("set_volume", {"level": "loud"})
        unknown = self.reg.execute("rm_rf", {})
        for r in (declined, invalid, unknown):
            with self.subTest(message=r.message):
                self.assertFalse(r.success)
                self.assertEqual(r.duration_s, 0.0)
        self.assertFalse(declined.confirmed)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
