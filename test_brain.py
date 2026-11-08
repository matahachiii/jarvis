"""
===============================================================
 test_brain.py - Phase 0 "done" test for Jarvis
===============================================================
 Sends one question to the brain (through the swappable
 interface) and reports:
   - the answer
   - the response time (cold start vs warm model)
   - the RAM usage (system, swap, model loaded in Ollama)

 Phase 0 is done when the local model answers and the warm
 response time is acceptable for a voice assistant.

 Usage:
   python test_brain.py             -> test the brain from config.toml
   python test_brain.py --compare   -> also compare the candidate models
===============================================================
"""

import copy
import sys
import time

import psutil
from ollama import Client

from jarvis.brain import create_brain
from jarvis.core.config import load_config

# ---------------------------------------------------------------
# 0. Parameters
# ---------------------------------------------------------------
QUESTION = "Hello Jarvis, who are you?"

# Temporary personality (the real one comes in Phase 1)
SYSTEM_PROMPT = (
    "You are Jarvis, a witty and loyal AI assistant running locally on Anwar's Mac. "
    "Answer in English, in at most two sentences."
)

# Candidates for --compare: (model, think)
CANDIDATES = [
    ("qwen3:4b-instruct-2507-q4_K_M", False),
    ("qwen3.5:4b", False),
    ("qwen3.5:4b", True),
]

# Voice assistant thresholds (warm model)
TARGET_FIRST_WORD_S = 1.5  # feels instant
MAX_FIRST_WORD_S = 3.0     # acceptable upper limit
MIN_TOKENS_PER_S = 8.0     # must generate faster than Jarvis can speak

GB = 1024 ** 3


# ---------------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------------
def fmt_s(seconds) -> str:
    """Format seconds for printing ('n/a' if missing)."""
    return "n/a" if seconds is None else f"{seconds:.2f} s"


def unload_all_models(host: str) -> None:
    """Free the RAM: ask Ollama to unload every model that is loaded."""
    client = Client(host=host)
    for model in client.ps().models:
        client.generate(model=model.model, keep_alive=0)


def ram_snapshot(host: str | None) -> dict:
    """Measure system RAM, swap and the size of the models loaded in Ollama."""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    models = []
    if host:
        models = [(m.model, m.size / GB) for m in Client(host=host).ps().models]
    return {
        "total_gb": memory.total / GB,
        "available_gb": memory.available / GB,
        "used_pct": memory.percent,
        "swap_gb": swap.used / GB,
        "models": models,
    }


def ask(brain, question: str):
    """Send one question (with the system prompt) to the brain."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return brain.chat(messages)


def verdict(result: dict) -> str:
    """Judge whether this brain is fast enough for a voice assistant."""
    first = result["warm_first_word_s"]
    tps = result["tokens_per_s"]
    if not result["answer"] or first is None:
        return "FAIL (no answer)"
    if tps is not None and tps < MIN_TOKENS_PER_S:
        return "FAIL (generates slower than speech)"
    if first <= TARGET_FIRST_WORD_S:
        return "PASS (feels instant)"
    if first <= MAX_FIRST_WORD_S:
        return "PASS (acceptable)"
    return "FAIL (too slow for voice)"


# ---------------------------------------------------------------
# 2. One test run: cold start + warm run + RAM
# ---------------------------------------------------------------
def run_test(config: dict) -> dict:
    settings = config["brain"]
    is_ollama = settings["provider"] == "ollama"
    host = settings["ollama"]["host"] if is_ollama else None

    # Start from an empty RAM so every model gets a fair cold start
    if is_ollama:
        unload_all_models(host)
        time.sleep(1)

    brain = create_brain(config)
    think = settings.get("think", False)
    print(f"\n--- {brain.name} (think={think}) ---")

    cold = ask(brain, QUESTION)   # includes loading the model into RAM
    warm = ask(brain, QUESTION)   # model already loaded: the realistic case
    ram = ram_snapshot(host)

    # Answer
    print(f"Answer     : {warm.text}")
    if warm.thinking:
        print(f"             (thought {len(warm.thinking.split())} words before answering)")
    if "<think>" in warm.text or "</think>" in warm.text:
        print("WARNING    : the model leaked its thinking into the answer")

    # Timing
    print(
        f"Cold start : first word after {fmt_s(cold.stats.get('first_answer_s'))} "
        f"(of which loading {fmt_s(cold.stats.get('load_s'))}), "
        f"total {fmt_s(cold.stats.get('total_s'))}"
    )
    tps = warm.stats.get("tokens_per_s")
    print(
        f"Warm       : first word after {fmt_s(warm.stats.get('first_answer_s'))}, "
        f"total {fmt_s(warm.stats.get('total_s'))}, "
        f"{warm.stats.get('output_tokens')} tokens at {tps or 0:.1f} tokens/s"
    )

    # RAM
    print(
        f"RAM        : {ram['available_gb']:.1f} GB free of {ram['total_gb']:.0f} GB "
        f"({ram['used_pct']:.0f}% used), swap in use {ram['swap_gb']:.1f} GB"
    )
    for name, size in ram["models"]:
        print(f"             loaded in Ollama: {name} = {size:.1f} GB")

    result = {
        "name": brain.name,
        "think": think,
        "answer": warm.text,
        "warm_first_word_s": warm.stats.get("first_answer_s"),
        "tokens_per_s": tps,
        "model_gb": sum(size for _, size in ram["models"]),
    }
    result["verdict"] = verdict(result)
    print(f"Verdict    : {result['verdict']}")
    return result


# ---------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------
def main() -> None:
    config = load_config()
    results = [run_test(config)]  # the brain from config.toml

    # Optional: compare the other candidates
    if "--compare" in sys.argv and config["brain"]["provider"] == "ollama":
        configured = (config["brain"]["ollama"]["model"], config["brain"].get("think", False))
        for model, think in CANDIDATES:
            if (model, think) == configured:
                continue  # already tested
            candidate = copy.deepcopy(config)  # a real copy, so config itself stays untouched
            candidate["brain"]["ollama"]["model"] = model
            candidate["brain"]["think"] = think
            results.append(run_test(candidate))

        print("\n=================== SUMMARY ===================")
        print(f"{'Brain':<42}{'think':>6}{'1st word':>10}{'tok/s':>8}{'RAM':>8}  Verdict")
        for r in results:
            first = r["warm_first_word_s"]
            print(
                f"{r['name']:<42}{str(r['think']):>6}"
                f"{(f'{first:.2f}s' if first is not None else 'n/a'):>10}"
                f"{(r['tokens_per_s'] or 0):>8.1f}"
                f"{r['model_gb']:>7.1f}G  {r['verdict']}"
            )

    # Phase 0 check, on the brain from config.toml
    print("\n===============================================")
    if results[0]["verdict"].startswith("PASS"):
        print(f"PHASE 0 DONE: {results[0]['name']} answers fast enough for voice.")
    else:
        print(f"PHASE 0 NOT DONE: {results[0]['name']} -> {results[0]['verdict']}")
    print("===============================================")


if __name__ == "__main__":
    main()
