"""
Unit tests for Conversation with tool turns (Phase 2, step 3a). No LLM.

Usage:  python -m unittest test_conversation_unit -v
"""

import unittest

from jarvis.core.conversation import MESSAGE_OVERHEAD, Conversation

CALL = [{"name": "get_time", "arguments": {}}]


def tool_turn(conv, user="What time is it?", result="OK: It is 22:15.",
              answer="A quarter past ten, sir."):
    """Add one complete tool turn: user -> tool call -> tool result -> final answer."""
    conv.add_user(user)
    conv.add_tool_call("", CALL)
    conv.add_tool_result("get_time", result)
    conv.add_assistant(answer)


class TestMessages(unittest.TestCase):

    def test_plain_chat_format_unchanged(self):
        conv = Conversation("You are Jarvis.")
        conv.add_user("Hello")
        conv.add_assistant("Good evening, sir.")
        self.assertEqual(conv.messages, [
            {"role": "system", "content": "You are Jarvis."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Good evening, sir."},
        ])

    def test_tool_turn_format(self):
        conv = Conversation("You are Jarvis.")
        tool_turn(conv)
        self.assertEqual([m["role"] for m in conv.messages],
                         ["system", "user", "assistant", "tool", "assistant"])
        call_msg, tool_msg = conv.messages[2], conv.messages[3]
        self.assertEqual(call_msg["tool_calls"], [{"function": {"name": "get_time", "arguments": {}}}])
        self.assertEqual(tool_msg["tool_name"], "get_time")
        self.assertTrue(all("tokens" not in m for m in conv.messages))

    def test_reset(self):
        conv = Conversation("sys")
        tool_turn(conv)
        conv.reset()
        self.assertEqual(conv.history, [])
        self.assertEqual(conv.dropped_turns, 0)


class TestTrimming(unittest.TestCase):

    def small(self):
        # system prompt ~104 tokens, budget = 600 - 100 - 0 = 500
        return Conversation("x" * 400, context_length=600, max_tokens=100, safety_margin=0)

    def test_drops_whole_tool_turn(self):
        conv = self.small()
        tool_turn(conv, result="r" * 400)
        conv.add_user("u2")
        conv.add_assistant("a2")
        conv.add_user("z" * 1200)
        self.assertEqual(conv.dropped_turns, 1)
        self.assertEqual(conv.history[0]["content"], "u2")
        self.assertNotIn("tool", [m["role"] for m in conv.history])   # no orphan tool results
        self.assertLessEqual(conv.total_tokens(), conv.budget)

    def test_never_drops_current_turn(self):
        conv = self.small()
        tool_turn(conv)
        conv.add_user("z" * 4000)   # far over budget on its own
        self.assertEqual([m["role"] for m in conv.history], ["user"])
        self.assertEqual(conv.history[0]["content"], "z" * 4000)
        self.assertEqual(conv.dropped_turns, 1)

    def test_trim_during_tool_turn_keeps_current_turn(self):
        conv = self.small()
        conv.add_user("u1")
        conv.add_assistant("a1")
        conv.add_user("Search my files")
        conv.add_tool_call("", [{"name": "search_files", "arguments": {"query": "thesis"}}])
        conv.add_tool_result("search_files", "f" * 1600)   # big result pushes us over budget
        self.assertEqual([m["role"] for m in conv.history], ["user", "assistant", "tool"])
        self.assertEqual(conv.history[0]["content"], "Search my files")


class TestUndo(unittest.TestCase):

    def test_removes_half_finished_tool_turn(self):
        conv = Conversation("sys")
        tool_turn(conv)
        conv.add_user("Open Spotify")
        conv.add_tool_call("", [{"name": "open_app", "arguments": {"app_name": "Spotify"}}])
        conv.add_tool_result("open_app", "OK: Opened Spotify.")
        conv.undo_last_user()   # e.g. Ctrl+C before the final answer
        self.assertEqual([m["role"] for m in conv.history], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(conv.history[-1]["content"], "A quarter past ten, sir.")

    def test_removes_plain_unanswered_question(self):
        conv = Conversation("sys")
        conv.add_user("Hello")
        conv.undo_last_user()
        self.assertEqual(conv.history, [])

    def test_leaves_finished_turn_alone(self):
        conv = Conversation("sys")
        tool_turn(conv)
        conv.undo_last_user()
        self.assertEqual(len(conv.history), 4)


class TestCalibration(unittest.TestCase):

    def test_total_matches_ollama_after_tool_call_and_answer(self):
        conv = Conversation("You are Jarvis. " * 20)
        conv.add_user("What time is it?")
        conv.add_tool_call("", CALL, {"prompt_tokens": 900, "output_tokens": 20})
        self.assertEqual(conv.total_tokens(), 900 + 20 + MESSAGE_OVERHEAD)
        conv.add_tool_result("get_time", "OK: It is 22:15.")
        conv.add_assistant("A quarter past ten, sir.", {"prompt_tokens": 950, "output_tokens": 9})
        self.assertEqual(conv.total_tokens(), 950 + 9 + MESSAGE_OVERHEAD)

class TestExamples(unittest.TestCase):

    EXAMPLES = [
        {"role": "user", "content": "Quit Notes."},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "quit_app", "arguments": {"app_name": "Notes"}}}]},
        {"role": "tool", "tool_name": "quit_app", "content": "SUCCESS: Closed Notes."},
        {"role": "assistant", "content": "Notes is closed, sir."},
    ]

    def test_examples_come_right_after_the_system_prompt(self):
        conv = Conversation("sys", examples=self.EXAMPLES)
        conv.add_user("Hello")
        self.assertEqual([m["role"] for m in conv.messages],
                         ["system", "user", "assistant", "tool", "assistant", "user"])
        self.assertEqual(conv.messages[1:5], self.EXAMPLES)

    def test_examples_are_never_trimmed_or_reset(self):
        conv = Conversation("x" * 400, context_length=600, max_tokens=100, safety_margin=0,
                            examples=self.EXAMPLES)
        for _ in range(5):
            conv.add_user("z" * 400)
            conv.add_assistant("a")
        self.assertGreater(conv.dropped_turns, 0)
        self.assertEqual(conv.messages[1:5], self.EXAMPLES)
        conv.reset()
        self.assertEqual(conv.messages[1:5], self.EXAMPLES)
        self.assertEqual(conv.history, [])

    def test_calibration_includes_examples(self):
        conv = Conversation("sys", examples=self.EXAMPLES)
        conv.add_user("Hello")
        conv.add_assistant("Good evening, sir.", {"prompt_tokens": 500, "output_tokens": 5})
        self.assertEqual(conv.total_tokens(), 500 + 5 + MESSAGE_OVERHEAD)
if __name__ == "__main__":
    unittest.main(verbosity=2)
