"""
Phase 2, step 1: can the local 4B model do tool calling?

Gives the model three DUMMY tools (calculator, get_time, open_app) and a fixed
set of prompts, about half of which need NO tool. Nothing is executed on the
Mac: we only record which tool the model picks and with which arguments.

Per run:  tool choice, arguments, false tool calls, decision latency.
Once:     token overhead of the tool definitions, RAM of the loaded model.
At the end the decision rule (agreed BEFORE the first run) gives a verdict.

Touches the real system: no (no tool is ever executed).
Usage:  python test_toolcalling.py
"""

import ast
import operator
import re
import statistics
from collections import Counter

import psutil

from jarvis.brain import create_brain
from jarvis.core.config import load_config
from jarvis.core.personality import load_system_prompt

# ============================================================ 0. Parameters
RUNS = 3

# Decision rule, agreed BEFORE looking at the results
MIN_CHOICE = 0.90            # tool choice correct in >= 90% of all trials
MIN_ARGS = 0.90              # arguments correct in >= 90% of the correct tool calls
MAX_FALSE_CHITCHAT = 0       # false tool calls on chit-chat, over all runs
MAX_FALSE_KNOWLEDGE = 1      # false tool calls on knowledge/trap prompts, over all runs
MAX_MEDIAN_LATENCY_S = 1.5   # median warm time until the model's decision
CLOUD_BELOW_CHOICE = 0.75    # below this choice accuracy: cloud/hybrid
CLOUD_FROM_FALSE_CHITCHAT = 3

# Replaces the "What you cannot do yet" paragraph, IN MEMORY only
TOOLS_PARAGRAPH = (
    "What you can do on this Mac: you have a few tools. Use a tool only when sir's "
    "request actually needs one: a calculation, the current date or time, or opening "
    "an app. For conversation, opinions and general knowledge, answer directly without "
    "any tool. You cannot do any other action yet, such as playing music, sending "
    "messages, browsing the internet or reading files. If sir asks for one, say plainly "
    "that you cannot do that yet, and never pretend you did."
)

# ============================================================ 1. Dummy tools
TOOLS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Calculate the result of an arithmetic expression, such as a sum or a percentage.",
        "parameters": {"type": "object",
                       "properties": {"expression": {
                           "type": "string",
                           "description": "Plain arithmetic with numbers and + - * / ** ( ), e.g. 0.15 * 240"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_time",
        "description": "Get the current date and time.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "open_app",
        "description": "Open an application on sir's Mac by its name.",
        "parameters": {"type": "object",
                       "properties": {"app_name": {
                           "type": "string",
                           "description": "The exact application name, e.g. Spotify or Safari"}},
                       "required": ["app_name"]}}},
]

# ============================================================ 2. Prompts
# (prompt, expected tool or None, expected argument, category)
PROMPTS = [
    ("What's 15% of 240?",                           "calculator", 36,     "tool"),
    ("How much is 1234 times 56?",                   "calculator", 69104,  "tool"),
    ("Can you divide 900 by 12 for me?",             "calculator", 75,     "tool"),
    ("What time is it?",                             "get_time",   None,   "tool"),
    ("What's today's date?",                         "get_time",   None,   "tool"),
    ("Is it already past noon?",                     "get_time",   None,   "tool"),
    ("Open Spotify.",                                "open_app",   "Spotify", "tool"),
    ("Could you launch Safari, please?",             "open_app",   "Safari",  "tool"),
    ("Fire up Visual Studio Code.",                  "open_app",   "Visual Studio Code", "tool"),
    ("Good evening, Jarvis.",                        None, None, "chitchat"),
    ("How are you today?",                           None, None, "chitchat"),
    ("Tell me a joke.",                              None, None, "chitchat"),
    ("Who wrote Pride and Prejudice?",               None, None, "knowledge"),
    ("What's the capital of Kenya?",                 None, None, "knowledge"),
    ("Explain what an ETF is in one sentence.",      None, None, "knowledge"),
    ("What's the time complexity of binary search?", None, None, "knowledge"),
    ("Should I open a savings account?",             None, None, "knowledge"),
]

# Info only (not counted): two tool calls in one request
MULTI_PROMPT = "Open Spotify and Safari."
MULTI_EXPECTED = {"spotify", "safari"}

# ============================================================ 3. Helpers
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    """Evaluate plain arithmetic without eval(): only numbers and + - * / ** % ( )."""
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError(f"not allowed: {type(node).__name__}")
    return walk(ast.parse(expression, mode="eval"))


def build_test_prompt(config: dict) -> str:
    """Real personality, minus the date/time sentence and with the NO TOOLS paragraph replaced."""
    prompt = load_system_prompt(config)

    prompt, n = re.subn(r" Today is [^.]*\. The time at startup was [^.]*\.", "", prompt)
    assert n == 1, "Date/time sentence not found in personality.md"

    paragraphs = prompt.split("\n\n")
    hits = [i for i, p in enumerate(paragraphs) if p.startswith("What you cannot do yet:")]
    assert len(hits) == 1, "'What you cannot do yet' paragraph not found in personality.md"
    paragraphs[hits[0]] = TOOLS_PARAGRAPH
    return "\n\n".join(paragraphs)


def get_calls(response) -> list[tuple[str, dict]]:
    """Tool calls as simple (name, arguments) pairs."""
    return [(c.function.name, dict(c.function.arguments or {})) for c in response.tool_calls]


def fmt_calls(calls) -> str:
    parts = []
    for name, args in calls:
        arg_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
        parts.append(f"{name}({arg_text})")
    return ", ".join(parts)


def args_correct(name: str, args: dict, expected_arg) -> bool:
    if name == "calculator":
        try:
            return abs(safe_eval(str(args.get("expression", ""))) - expected_arg) < 1e-6
        except Exception:
            return False
    if name == "open_app":
        app = str(args.get("app_name", "")).strip().lower().removesuffix(".app")
        return app == expected_arg.lower()
    return True  # get_time takes no arguments


def score(expected_tool, expected_arg, calls) -> tuple[bool, bool | None]:
    """Returns (choice_ok, args_ok). args_ok is None when there are no arguments to check."""
    if expected_tool is None:
        return (not calls), None
    if {name for name, _ in calls} != {expected_tool}:
        return False, None
    return True, all(args_correct(n, a, expected_arg) for n, a in calls)


def label(expected_tool, calls, choice_ok, args_ok) -> str:
    if choice_ok:
        return "OK   " if args_ok is not False else "ARGS "
    if expected_tool is None:
        return "FALSE"
    return "MISS " if not calls else "WRONG"


def ask(brain, system_prompt, prompt, tools=TOOLS):
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
    response = brain.chat(messages, tools=tools)
    stats = response.stats
    latency = stats.get("first_answer_s") or stats.get("total_s")
    return response, get_calls(response), latency


def print_ram(brain, moment: str) -> None:
    vm, swap = psutil.virtual_memory(), psutil.swap_memory()
    print(f"RAM {moment}: available {vm.available / 1e9:.2f} GB, swap used {swap.used / 1e9:.2f} GB")
    try:
        for m in brain.client.ps().models:
            print(f"  loaded model: {m.model}  size {m.size / 1e9:.2f} GB")
    except Exception as e:
        print(f"  (could not read ollama ps: {e})")


# ============================================================ 4. Main
def main() -> None:
    config = load_config()
    brain = create_brain(config)
    system_prompt = build_test_prompt(config)
    print(f"Tool-calling test: {brain.name}, {RUNS} runs x {len(PROMPTS)} prompts\n")

    # 4a. Token overhead (also loads the model and warms the cache WITH tools)
    try:
        r_without, _, _ = ask(brain, system_prompt, "Good evening, Jarvis.", tools=None)
        r_with, _, _ = ask(brain, system_prompt, "Good evening, Jarvis.")
    except ConnectionError:
        print("Ollama is not reachable. Start it with: brew services start ollama")
        return
    t_without = r_without.stats.get("prompt_tokens") or 0
    t_with = r_with.stats.get("prompt_tokens") or 0
    extra = t_with - t_without
    print(f"Prompt tokens: without tools {t_without}, with 3 tools {t_with} "
          f"-> tools add {extra} (~{extra / 3:.0f} per tool, 9 tools ~{extra * 3})")
    print_ram(brain, "with model loaded")

    # 4b. Runs
    results, multi_ok = [], []
    for run in range(1, RUNS + 1):
        print(f"\n=== Run {run}/{RUNS} ===")
        for prompt, tool, arg, category in PROMPTS:
            response, calls, latency = ask(brain, system_prompt, prompt)
            choice_ok, args_ok = score(tool, arg, calls)
            what = fmt_calls(calls) if calls else f"(answer) {response.text[:45]}"
            print(f"  {label(tool, calls, choice_ok, args_ok)} {prompt:<46} -> {what}  [{latency:.2f} s]")
            results.append({"run": run, "prompt": prompt, "category": category,
                            "choice_ok": choice_ok, "args_ok": args_ok,
                            "calls": calls, "latency": latency})

        _, calls, latency = ask(brain, system_prompt, MULTI_PROMPT)
        apps = {str(a.get("app_name", "")).strip().lower() for n, a in calls if n == "open_app"}
        ok = len(calls) == 2 and apps == MULTI_EXPECTED
        multi_ok.append(ok)
        print(f"  info  {MULTI_PROMPT:<46} -> {fmt_calls(calls) or '(no tool)'}  "
              f"[{latency:.2f} s] {'OK' if ok else 'not ok'}")

    print()
    print_ram(brain, "after all runs")

    # 4c. Summary
    n = len(results)
    choice = sum(r["choice_ok"] for r in results) / n
    tool_hits = [r for r in results if r["category"] == "tool" and r["choice_ok"]]
    args = sum(bool(r["args_ok"]) for r in tool_hits) / len(tool_hits) if tool_hits else 0.0
    false_chit = sum(1 for r in results if r["category"] == "chitchat" and r["calls"])
    false_know = sum(1 for r in results if r["category"] == "knowledge" and r["calls"])
    med_tool = statistics.median(r["latency"] for r in results if r["category"] == "tool")
    med_none = statistics.median(r["latency"] for r in results if r["category"] != "tool")
    med_all = statistics.median(r["latency"] for r in results)

    print("\n=== Summary ===")
    for run in range(1, RUNS + 1):
        rr = [r for r in results if r["run"] == run]
        print(f"  run {run}: choice {sum(r['choice_ok'] for r in rr)}/{len(rr)}")
    fails = Counter(r["prompt"] for r in results
                    if not r["choice_ok"] or r["args_ok"] is False)
    for prompt, count in fails.most_common():
        print(f"  failed {count}x: {prompt}")
    print(f"  multi-call (info): {sum(multi_ok)}/{RUNS}")
    print(f"  median latency: tool {med_tool:.2f} s, no-tool {med_none:.2f} s\n")

    checks = [
        (f"Tool choice >= {MIN_CHOICE:.0%}", choice >= MIN_CHOICE,
         f"{sum(r['choice_ok'] for r in results)}/{n} = {choice:.0%}"),
        (f"Arguments >= {MIN_ARGS:.0%}", args >= MIN_ARGS,
         f"{sum(bool(r['args_ok']) for r in tool_hits)}/{len(tool_hits)} = {args:.0%}"),
        (f"False calls chit-chat <= {MAX_FALSE_CHITCHAT}", false_chit <= MAX_FALSE_CHITCHAT,
         str(false_chit)),
        (f"False calls knowledge <= {MAX_FALSE_KNOWLEDGE}", false_know <= MAX_FALSE_KNOWLEDGE,
         str(false_know)),
        (f"Median latency <= {MAX_MEDIAN_LATENCY_S} s", med_all <= MAX_MEDIAN_LATENCY_S,
         f"{med_all:.2f} s"),
    ]
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    if all(ok for _, ok, _ in checks):
        verdict = "LOCAL (stay on the 4B model)"
    elif choice < CLOUD_BELOW_CHOICE or false_chit >= CLOUD_FROM_FALSE_CHITCHAT:
        verdict = "CLOUD / HYBRID"
    else:
        verdict = "TUNE (one change to the tool descriptions, then re-measure 3x)"
    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
