"""
Basic tools that never touch the Mac: a safe calculator (date/time follows).

The calculator never uses eval(): the expression is parsed into a tree (ast)
and only numbers and + - * / ** ( ) are evaluated. "%" always means percent.
"""

import ast
import math
import operator
import re

from jarvis.tools.registry import ToolResult, tool

MAX_EXPONENT = 100   # 9 ** 9 ** 9 would freeze the Mac; this keeps every result small and fast

_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _power(base, exponent):
    if abs(exponent) > MAX_EXPONENT:
        raise ValueError(f"exponents above {MAX_EXPONENT} are not allowed")
    return operator.pow(base, exponent)


def safe_eval(expression: str):
    """Evaluate plain arithmetic without eval(). Raises ValueError for anything else."""
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                return _power(walk(node.left), walk(node.right))
            if type(node.op) in _OPERATORS:
                return _OPERATORS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](walk(node.operand))
        raise ValueError(f"'{type(node).__name__}' is not allowed")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ValueError("this is not an expression I can read") from None
    return walk(tree)


def normalise(expression: str) -> str:
    """Turn the ways people (and models) write maths into plain Python arithmetic."""
    text = expression.strip().lower()
    text = text.replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**")
    text = re.sub(r"(?<=[\d)\s])x(?=[\s\d(])", "*", text)          # 2 x 3 -> 2 * 3
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)           # 1,234 -> 1234
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", text)      # 15% -> (15/100)
    text = re.sub(r"\bof\b", "*", text)                           # 15% of 240 -> (15/100) * 240
    return text


def format_number(value) -> str:
    """36.0 -> '36', 10/3 -> '3.333333333'."""
    if isinstance(value, complex):
        raise ValueError("the result is not a real number")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OverflowError
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


@tool("Calculate an arithmetic expression exactly. Use it for every sum, percentage or "
      "other calculation instead of doing the maths yourself.",
      {"expression": ("string", "Arithmetic with numbers and + - * / ** ( ) %, e.g. 15% * 240 or (900 / 12) - 8")})
def calculator(expression):
    try:
        result = format_number(safe_eval(normalise(expression)))
    except ZeroDivisionError:
        return ToolResult(False, "Division by zero is not possible.")
    except OverflowError:
        return ToolResult(False, "The result is too large to calculate.")
    except ValueError as error:
        return ToolResult(False, f"Could not calculate '{expression}': {error}.")
    return ToolResult(True, f"{expression} = {result}")
