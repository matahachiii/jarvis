"""
Unit tests for the basic tools (Phase 2, step 4). No LLM, nothing touches the Mac.

Usage:  python -m unittest test_tools_basic -v
"""

import unittest

from datetime import datetime

from jarvis.tools.basic import calculator, describe_moment, spoken_time
from jarvis.tools.registry import registry


class TestCalculator(unittest.TestCase):

    def check(self, expression, expected):
        result = calculator(expression)
        self.assertTrue(result.success, result.message)
        self.assertTrue(result.message.endswith(f"= {expected}"), result.message)

    def test_basic_arithmetic(self):
        for expression, expected in [("0.15 * 240", "36"), ("1234 * 56", "69104"), ("900 / 12", "75"),
                                     ("(900 / 12) - 2**3", "67"), ("-5 + 2", "-3"), ("7 // 2", "3")]:
            with self.subTest(expression=expression):
                self.check(expression, expected)

    def test_spoken_notations(self):
        for expression, expected in [("15% of 240", "36"), ("15% * 240", "36"), ("2 ^ 10", "1024"),
                                     ("2 × 3", "6"), ("2 x 3", "6"), ("8 ÷ 2", "4"), ("1,234 * 2", "2468")]:
            with self.subTest(expression=expression):
                self.check(expression, expected)

    def test_formatting(self):
        self.check("10 / 3", "3.333333333")
        self.check("0.1 + 0.2", "0.3")   # no 0.30000000000000004

    def test_rejected(self):
        for expression in ["1 / 0", "__import__('os').system('ls')", "open('x')", "2 ** 1000",
                           "9 ** 9 ** 9", "(-8) ** 0.5", "abc", "", "2 +"]:
            with self.subTest(expression=expression):
                self.assertFalse(calculator(expression).success)

    def test_through_registry(self):
        self.assertIn("calculator", registry.names)
        result = registry.execute("calculator", {"expression": "15% of 240"})
        self.assertTrue(result.success)
        self.assertEqual(result.message, "15% of 240 = 36")

class TestTime(unittest.TestCase):

    def test_spoken_time(self):
        for (hour, minute), expected in [
            ((0, 0), "midnight"), ((12, 0), "noon"),
            ((9, 0), "nine o'clock in the morning"),
            ((9, 1), "one minute past nine in the morning"),
            ((15, 15), "a quarter past three in the afternoon"),
            ((18, 30), "half past six in the evening"),
            ((12, 20), "twenty past twelve in the afternoon"),
            ((7, 45), "a quarter to eight in the morning"),
            ((22, 52), "eight minutes to eleven at night"),
            ((11, 45), "a quarter to noon"),
            ((23, 52), "eight minutes to midnight"),
            ((0, 10), "ten past midnight"),
        ]:
            with self.subTest(time=f"{hour:02d}:{minute:02d}"):
                self.assertEqual(spoken_time(hour, minute), expected)

    def test_every_minute_of_the_day_works(self):
        for hour in range(24):
            for minute in range(60):
                self.assertTrue(spoken_time(hour, minute))   # no crash, never empty

    def test_describe_moment(self):
        self.assertEqual(describe_moment(datetime(2026, 11, 10, 22, 52)),
                         "It is Tuesday 10 November 2026, 22:52, which is eight minutes to eleven at night.")

    def test_get_time_through_registry(self):
        self.assertIn("get_time", registry.names)
        result = registry.execute("get_time")
        self.assertTrue(result.success)
        self.assertIn(str(datetime.now().year), result.message)
if __name__ == "__main__":
    unittest.main(verbosity=2)
