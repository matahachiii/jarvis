"""
Unit tests for the basic tools (Phase 2, step 4). No LLM, nothing touches the Mac.

Usage:  python -m unittest test_tools_basic -v
"""

import unittest

from jarvis.tools.basic import calculator
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
