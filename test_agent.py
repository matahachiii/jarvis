"""
Unit tests for the agent loop (Phase 2, step 3b).

No LLM and no real Mac actions: a FakeBrain plays back scripted replies,
and the tools are dummies that only record that they ran.

Usage:  python -m unittest test_agent -v
"""

import unittest

from jarvis.brain.base import BrainResponse
from jarvis.core.agent import CAP_MESSAGE, run_turn
from jarvis.core.conversation import Conversation
from jarvis.tools.registry import ToolRegistry, ToolResult

INTERRUPT = "INTERRUPT"   # script entry that simulates Ctrl+C during a brain call


def call(name, **arguments):
    """A tool call in the shape Ollama returns it."""
    return {"function": {"name": name, "arguments": arguments}}


class FakeBrain:
    """Plays back scripted replies (text, tool_calls) and records what it was sent."""

    def __init__(self, script):
        self.script = list(script)
        self.sent = []
        self.last_response = None

    def chat_stream(self, messages, tools=None):
        self.sent.append({"messages": messages, "tools": tools})
        self.last_response = None
        step = self.script.pop(0)
        if step == INTERRUPT:
            raise KeyboardInterrupt
        text, tool_calls = step
        for i, word in enumerate(text.split(" ")):
            yield word if i == 0 else " " + word
        self.last_response = BrainResponse(text=text, tool_calls=tool_calls, stats={})


def make_registry():
    """Dummy tools; `executed` records every real execution."""
    reg = ToolRegistry()
    executed = []

    @reg.tool("Get the current time.")
    def get_time():
        executed.append("get_time")
        return ToolResult(True, "It is 22:40.")

    @reg.tool("Open an app.", {"app_name": ("string", "App name")})
    def open_app(app_name):
        executed.append(("open_app", app_name))
        return ToolResult(True, f"Opened {app_name}.")

    @reg.tool("Set the volume.", {"level": ("integer", "0-100")})
    def set_volume(level):
        executed.append(("set_volume", level))
        return ToolResult(True, f"Volume set to {level}.")

    @reg.tool("Quit an app.", {"app_name": ("string", "App name")}, risk="confirm")
    def quit_app(app_name):
        executed.append(("quit_app", app_name))
        return ToolResult(True, f"Quit {app_name}.")

    return reg, executed


class TestAgentLoop(unittest.TestCase):

    def setUp(self):
        self.reg, self.executed = make_registry()
        self.conv = Conversation("You are Jarvis.")

    def run_script(self, script, user="Hello", **kwargs):
        self.brain = FakeBrain(script)
        self.conv.add_user(user)
        return run_turn(self.brain, self.conv, self.reg, **kwargs)

    def roles(self):
        return [m["role"] for m in self.conv.history]

    def test_no_tool_needed(self):
        result = self.run_script([("Good evening, sir.", [])])
        self.assertEqual(result.text, "Good evening, sir.")
        self.assertEqual(result.rounds, 1)
        self.assertEqual(result.tool_runs, [])
        self.assertEqual(self.roles(), ["user", "assistant"])

    def test_tools_are_offered(self):
        self.run_script([("Hi.", [])])
        names = sorted(t["function"]["name"] for t in self.brain.sent[0]["tools"])
        self.assertEqual(names, ["get_time", "open_app", "quit_app", "set_volume"])

    def test_empty_registry_sends_no_tools(self):
        self.reg = ToolRegistry()
        self.run_script([("Hi.", [])])
        self.assertIsNone(self.brain.sent[0]["tools"])

    def test_one_tool(self):
        result = self.run_script([("", [call("get_time")]), ("Twenty to eleven, sir.", [])],
                                 user="What time is it?")
        self.assertEqual(self.executed, ["get_time"])
        self.assertEqual(result.text, "Twenty to eleven, sir.")
        self.assertEqual(result.rounds, 2)
        self.assertEqual(self.roles(), ["user", "assistant", "tool", "assistant"])
        tool_msg = self.brain.sent[1]["messages"][-1]   # what the model saw in round 2
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["content"], "SUCCESS: It is 22:40.")

    def test_several_tools_in_one_request(self):
        result = self.run_script(
            [("", [call("open_app", app_name="Spotify"), call("set_volume", level=30)]),
             ("Done, sir.", [])],
            user="Open Spotify and set the volume to 30")
        self.assertEqual(self.executed, [("open_app", "Spotify"), ("set_volume", 30)])
        self.assertEqual(len(result.tool_runs), 2)
        self.assertEqual(self.roles(), ["user", "assistant", "tool", "tool", "assistant"])

    def test_confirm_tool_denied_is_not_executed(self):
        result = self.run_script([("", [call("quit_app", app_name="Spotify")]),
                                  ("Very well, sir, Spotify stays open.", [])],
                                 user="Quit Spotify", confirm=lambda tool, args: False)
        self.assertEqual(self.executed, [])
        self.assertFalse(result.tool_runs[0].result.confirmed)
        self.assertTrue(self.brain.sent[1]["messages"][-1]["content"].startswith("FAILED: Sir declined"))

    def test_confirm_tool_without_callback_is_not_executed(self):
        self.run_script([("", [call("quit_app", app_name="Spotify")]), ("Understood.", [])])
        self.assertEqual(self.executed, [])

    def test_confirm_tool_accepted(self):
        self.run_script([("", [call("quit_app", app_name="Spotify")]), ("Spotify is closed, sir.", [])],
                        confirm=lambda tool, args: True)
        self.assertEqual(self.executed, [("quit_app", "Spotify")])

    def test_unknown_tool_is_reported_back(self):
        result = self.run_script([("", [call("delete_everything")]), ("I cannot do that, sir.", [])])
        self.assertFalse(result.tool_runs[0].result.success)
        self.assertIn("no tool", self.brain.sent[1]["messages"][-1]["content"])
        self.assertEqual(result.text, "I cannot do that, sir.")

    def test_round_cap(self):
        stuck = [("", [call("get_time")])] * 10   # a confused model that never stops asking
        result = self.run_script(stuck, max_tool_rounds=3)
        self.assertTrue(result.capped)
        self.assertEqual(result.rounds, 4)        # 3 tool rounds + the refused 4th
        self.assertEqual(len(self.executed), 3)
        self.assertEqual(result.text, CAP_MESSAGE)
        self.assertEqual(self.conv.history[-1]["content"], CAP_MESSAGE)

    def test_callbacks(self):
        shown, events = [], []
        self.run_script(
            [("One moment.", [call("get_time")]), ("It is late, sir.", [])],
            on_text=shown.append,
            on_tool_call=lambda name, args: events.append(("call", name)),
            on_tool_result=lambda name, args, result: events.append(("result", name, result.success)))
        self.assertEqual("".join(shown), "One moment.It is late, sir.")
        self.assertEqual(events, [("call", "get_time"), ("result", "get_time", True)])

    def test_interrupt_then_undo_leaves_clean_history(self):
        self.conv.add_user("Hi")
        self.conv.add_assistant("Good evening, sir.")
        with self.assertRaises(KeyboardInterrupt):
            self.run_script([("", [call("get_time")]), INTERRUPT], user="What time is it?")
        self.conv.undo_last_user()   # what the chat loop does after Ctrl+C
        self.assertEqual(self.roles(), ["user", "assistant"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
