"""
STT benchmark (Phase 3 step 1): one candidate per run, on the fixed test set in data/testset/.

Measures, WITH the qwen3 model loaded in Ollama (like real use):
- RAM:     footprint growth of this Python process + of macOS speech services (for engines that
           run outside Python), swap growth; the system-wide drop is shown for information
- speed:   seconds of STT per phrase, and seconds of STT per second of audio
- quality: word error rate (WER) against the expected text, and the yes/no clips

Usage (Terminal.app; VS Code and Safari closed):
    python -u bench_stt.py fw-tiny | tee logs/bench_stt_fw-tiny.txt
    python bench_stt.py --list            show the engines this script knows
Results are also saved as logs/bench_stt_<engine>.json for the comparison at the end.

Decision rule, agreed BEFORE the first run (see RULE_* below). A candidate passes if ALL hold:
  1. WER <= 10% (median of 3 runs), phrase 08 (Buizingen) left out; it is reported separately
  2. all 5 confirmation clips exactly right in every run (a "no" must never become something else)
  3. median STT time for phrases of 2-4 s <= 0.6 s
  4. extra RAM <= 400 MB and swap growth <= 250 MB (plus: no stutter or freeze, judged by sir)
     extra RAM = the worst of: drop in system-wide available memory, growth of this process'
     footprint (the "Memory" column of Activity Monitor). Added after the fw-tiny run, which
     showed +78 MB system vs +309 MB RSS (RSS counts shared library files, footprint does not).
Choice: the lightest candidate that passes; within 100 MB of each other, the lowest WER wins.
Change after the first 5 runs (2026-11-19): no candidate passed criterion 2, and the two best
(fw-base, apple) only failed on clip 18 because they leave out the filler word ("Hmm, I'm not
sure" -> "I'm not sure"). Filler words are now dropped in normalize(), like Whisper's own
normalizer does. fw-tiny (clip 17) and Moonshine (WER, RAM) fail regardless; fw-base and apple re-run.
Second change (same day): the system-wide drop in available memory proved too noisy to decide
(same engine: fw-base +69 vs +382 MB, apple +201 vs +451 MB, while their own footprint fell).
RAM cost is now measured per process: footprint growth of this process + of every process of this
user whose name matches SERVICE_PATTERN (the macOS speech services). Limit unchanged: 400 MB.
"""

import argparse
import ctypes
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import jiwer
import numpy as np
import psutil
import soundfile as sf
from ollama import Client

from jarvis.core.config import load_config

TESTSET_DIR = Path("data/testset")
LOG_DIR = Path("logs")
SAMPLE_RATE = 16_000
MB = 1024 * 1024

RUNS = 3
RULE_WER_MAX = 0.10            # 10%
RULE_SPEED_MAX_S = 0.6         # median STT time for phrases of 2-4 s
RULE_SPEED_RANGE_S = (2.0, 4.0)
RULE_EXTRA_RAM_MAX_MB = 400
RULE_SWAP_GROWTH_MAX_MB = 250
NAME_WORD = "buizingen"        # phrase with this word: reported separately, not in the rule WER
SERVICE_PATTERN = re.compile(r"speech|siri|assistant|dictation", re.IGNORECASE)  # macOS speech services


# ---------------------------------------------------------------- engines
# Every engine has: description, load() and transcribe(audio) -> text.
# audio = numpy float32 vector, 16 kHz mono. Imports happen inside load(), so an engine
# that is not installed never breaks the others.

class FasterWhisperEngine:
    """faster-whisper (CTranslate2) on the CPU, int8, greedy decoding (beam_size=1) for speed."""

    def __init__(self, size: str):
        self.size = size
        self.model = None
        self.description = f"faster-whisper {size} (int8, CPU, greedy)"

    def load(self) -> None:
        from faster_whisper import WhisperModel
        self.model = WhisperModel(self.size, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self.model.transcribe(
            audio, language="en", beam_size=1,
            condition_on_previous_text=False, without_timestamps=True,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class MoonshineEngine:
    """Moonshine (Useful Sensors) via its small ONNX package, CPU, float weights, greedy decoding."""

    def __init__(self, size: str):
        self.size = size
        self.model = None
        self.tokenizer = None
        self.description = f"Moonshine {size} (ONNX, float, CPU, greedy)"

    def load(self) -> None:
        from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
        self.model = MoonshineOnnxModel(model_name=f"moonshine/{self.size}")
        self.tokenizer = load_tokenizer()

    def transcribe(self, audio: np.ndarray) -> str:
        tokens = self.model.generate(audio[np.newaxis, :].astype(np.float32))  # shape [1, samples]
        return self.tokenizer.decode_batch(tokens)[0].strip()


class AppleSpeechEngine:
    """
    macOS built-in speech recognition (SFSpeechRecognizer) via pyobjc, en-US, ON-DEVICE ONLY.
    requiresOnDeviceRecognition=True keeps the audio on this Mac; if on-device is not supported,
    load() refuses (the audio would otherwise go to Apple). The recognition runs in a macOS
    service, so its RAM shows up under "speech services", not in this process.
    """

    NOT_DETERMINED, AUTHORIZED = 0, 3
    NO_SPEECH_ERROR = 1110          # "No speech detected": counts as an empty transcript
    TIMEOUT_S = 15

    def __init__(self):
        self.recognizer = None
        self.wav_file = None
        self.errors_shown = set()
        self.description = "macOS SFSpeechRecognizer (on-device only, en-US)"

    def _spin(self, done, timeout_s: float) -> None:
        """Let macOS deliver its callbacks: run the main run loop in small steps until done() or timeout."""
        import Foundation
        deadline = time.perf_counter() + timeout_s
        run_loop = Foundation.NSRunLoop.currentRunLoop()
        while not done() and time.perf_counter() < deadline:
            run_loop.runUntilDate_(Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.01))

    def load(self) -> None:
        import tempfile

        import Foundation
        import Speech

        if Speech.SFSpeechRecognizer.authorizationStatus() != self.AUTHORIZED:
            answer = {}

            def on_answer(status):          # a block that returns nothing, as macOS expects
                answer["status"] = status

            Speech.SFSpeechRecognizer.requestAuthorization_(on_answer)
            self._spin(lambda: "status" in answer, 60)
            if answer.get("status") != self.AUTHORIZED:
                raise RuntimeError("Speech Recognition not allowed. System Settings > Privacy & Security > "
                                   "Speech Recognition > switch on Terminal, then run again.")

        locale = Foundation.NSLocale.alloc().initWithLocaleIdentifier_("en-US")
        self.recognizer = Speech.SFSpeechRecognizer.alloc().initWithLocale_(locale)
        if self.recognizer is None or not self.recognizer.supportsOnDeviceRecognition():
            raise RuntimeError("On-device recognition is not available for en-US on this Mac: "
                               "candidate dropped (we never send audio to Apple).")
        self.wav_file = Path(tempfile.gettempdir()) / "jarvis_bench_apple.wav"

    def transcribe(self, audio: np.ndarray) -> str:
        import Foundation
        import Speech

        sf.write(self.wav_file, audio, SAMPLE_RATE, subtype="PCM_16")
        url = Foundation.NSURL.fileURLWithPath_(str(self.wav_file))
        request = Speech.SFSpeechURLRecognitionRequest.alloc().initWithURL_(url)
        request.setRequiresOnDeviceRecognition_(True)   # the audio never leaves this Mac
        request.setShouldReportPartialResults_(False)

        state = {"done": False, "text": "", "error": None}

        def handler(result, error):
            if error is not None:
                state["error"] = (int(error.code()), str(error.localizedDescription()))
                state["done"] = True
            elif result is not None and result.isFinal():
                state["text"] = str(result.bestTranscription().formattedString())
                state["done"] = True

        task = self.recognizer.recognitionTaskWithRequest_resultHandler_(request, handler)
        self._spin(lambda: state["done"], self.TIMEOUT_S)
        if not state["done"]:
            task.cancel()
            print(f"    (apple: no answer within {self.TIMEOUT_S} s)")
            return ""
        if state["error"] is not None:
            code, message = state["error"]
            if code != self.NO_SPEECH_ERROR and code not in self.errors_shown:
                self.errors_shown.add(code)
                print(f"    (apple error {code}: {message})")
            return ""
        return state["text"].strip()


ENGINES = {
    "fw-tiny": lambda: FasterWhisperEngine("tiny.en"),
    "fw-base": lambda: FasterWhisperEngine("base.en"),
    "moon-tiny": lambda: MoonshineEngine("tiny"),
    "moon-base": lambda: MoonshineEngine("base"),
    "apple": lambda: AppleSpeechEngine(),
}


# ---------------------------------------------------------------- text normalisation for WER
CONTRACTIONS = {
    "what's": "what is", "that's": "that is", "it's": "it is", "i'm": "i am",
    "don't": "do not", "can't": "can not", "let's": "let us", "you're": "you are",
}
FILLERS = {"hm": "hmm", "hmm": "hmm", "hmmm": "hmm", "mm": "hmm", "mmm": "hmm",
           "um": "hmm", "umm": "hmm", "uh": "hmm", "erm": "hmm"}
ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_to_words(n: int) -> str:
    """0-999 as English words (enough for the test set): 40 -> 'forty', 65 -> 'sixty five'."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + number_to_words(rest))
    return str(n)


def normalize(text: str) -> str:
    """
    Lowercase, no punctuation, '40%' == 'forty percent', "what's" == 'what is'.
    Filler words (hmm, um, uh, ...) are dropped, as in Whisper's own text normalizer:
    many recognisers leave them out on purpose, and they carry no meaning for Jarvis.
    """
    t = text.lower().replace("\u2019", "'").replace("%", " percent ").replace("-", " ")
    for short, full in CONTRACTIONS.items():
        t = re.sub(rf"\b{re.escape(short)}\b", full, t)
    t = re.sub(r"\d+", lambda m: f" {number_to_words(int(m.group()))} ", t)
    t = re.sub(r"[^a-z ]", " ", t)  # drop punctuation, apostrophes and anything else
    return " ".join(word for word in t.split() if word not in FILLERS)


def wer(refs: list[str], hyps: list[str]) -> float:
    """Word error rate over a list of phrases (all words together), on normalised text."""
    return jiwer.wer([normalize(r) for r in refs], [normalize(h) for h in hyps])


# ---------------------------------------------------------------- measurements
def footprint_mb(pid: int) -> float | None:
    """
    macOS 'physical footprint' of a process in MB: the number Activity Monitor shows as "Memory".
    Unlike RSS it does not count shared library files that macOS can drop for free.
    Read via proc_pid_rusage (RUSAGE_INFO_V2: 16-byte uuid, then uint64 fields; phys_footprint = index 9).
    Returns None if it cannot be read (then the report falls back to RSS).
    """
    try:
        libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        buffer = (ctypes.c_uint64 * 40)()
        if libsystem.proc_pid_rusage(pid, 2, ctypes.byref(buffer)) != 0:
            return None
        return buffer[9] / MB
    except (OSError, AttributeError):
        return None


def other_footprints() -> dict:
    """Footprint of every OTHER process of this user: {pid: (name, MB)}. Catches engines that run in a macOS service."""
    me, my_pid = os.getuid(), os.getpid()
    result = {}
    for proc in psutil.process_iter(["name", "uids"]):
        try:
            uids = proc.info["uids"]
            if proc.pid == my_pid or uids is None or uids.real != me:
                continue
            value = footprint_mb(proc.pid)
            if value is not None:
                result[proc.pid] = (proc.info["name"] or "?", value)
        except (psutil.Error, OSError):
            continue
    return result


def memory_snapshot() -> dict:
    """Swap, system-wide available RAM, this process (footprint + RSS) and all other processes of this user, in MB."""
    process = psutil.Process()
    return {
        "available_mb": psutil.virtual_memory().available / MB,
        "swap_mb": psutil.swap_memory().used / MB,
        "footprint_mb": footprint_mb(process.pid),
        "rss_mb": process.memory_info().rss / MB,
        "others": other_footprints(),
    }


def other_growth(samples: list[dict], baseline: dict) -> list[tuple[str, int, float]]:
    """Per other process: largest footprint during the run minus its baseline (0 if it did not exist yet)."""
    peaks = {}
    for sample in samples:
        for pid, (name, value) in sample["others"].items():
            if pid not in peaks or value > peaks[pid][1]:
                peaks[pid] = (name, value)
    growths = []
    for pid, (name, peak) in peaks.items():
        start = baseline["others"].get(pid, (name, 0.0))[1]
        growths.append((name, pid, peak - start))
    return sorted(growths, key=lambda item: item[2], reverse=True)


def growth(samples: list[dict], baseline: dict, key: str) -> float | None:
    """Largest increase of one memory field compared to the baseline (None if not measurable)."""
    values = [s[key] for s in samples if s[key] is not None]
    if baseline[key] is None or not values:
        return None
    return max(values) - baseline[key]


def load_brain_model(config: dict) -> tuple[str, bool]:
    """Make sure qwen3 is loaded in Ollama with the same num_ctx as Jarvis (same RAM use)."""
    brain = config["brain"]
    ollama_cfg = brain["ollama"]
    client = Client(host=ollama_cfg.get("host", "http://localhost:11434"))
    model = ollama_cfg["model"]
    client.generate(model=model, prompt="Hi", keep_alive="15m",
                    options={"num_ctx": brain.get("context_length", 4096), "num_predict": 1})
    loaded = [getattr(m, "model", None) or getattr(m, "name", "") for m in client.ps().models]
    return model, model in loaded


def load_testset() -> list[dict]:
    index_file = TESTSET_DIR / "testset.json"
    if not index_file.exists():
        sys.exit("No test set found. Record it first: python record_testset.py")
    items = json.loads(index_file.read_text(encoding="utf-8"))
    for item in items:
        path = TESTSET_DIR / item["file"]
        if not path.exists():
            sys.exit(f"Missing {path}. Record it first: python record_testset.py")
        audio, rate = sf.read(path, dtype="float32")
        if rate != SAMPLE_RATE:
            sys.exit(f"{path} is {rate} Hz, expected {SAMPLE_RATE} Hz.")
        item["audio"] = audio
        item["duration_s"] = len(audio) / SAMPLE_RATE
    return items


def run_once(engine, items: list[dict], run_nr: int, memory_samples: list[dict]) -> list[dict]:
    """Transcribe every phrase once; print one line per phrase."""
    results = []
    for item in items:
        start = time.perf_counter()
        text = engine.transcribe(item["audio"])
        seconds = time.perf_counter() - start
        memory_samples.append(memory_snapshot())
        exact = normalize(text) == normalize(item["text"])
        results.append({"id": item["id"], "text": text, "stt_s": seconds, "exact": exact})
        mark = "ok " if exact else "-- "
        print(f"  [run {run_nr}] {item['id']} {seconds:5.2f} s  {mark} {text!r}")
    return results


def summarize_run(items: list[dict], results: list[dict]) -> dict:
    refs = [item["text"] for item in items]
    hyps = [r["text"] for r in results]
    rule_idx = [i for i, item in enumerate(items) if NAME_WORD not in normalize(item["text"])]
    confirm_idx = [i for i, item in enumerate(items) if item["category"] == "confirm"]
    low, high = RULE_SPEED_RANGE_S
    speed_idx = [i for i, item in enumerate(items) if low <= item["duration_s"] <= high]
    name_idx = [i for i in range(len(items)) if i not in rule_idx]
    return {
        "wer_rule": wer([refs[i] for i in rule_idx], [hyps[i] for i in rule_idx]),
        "wer_all": wer(refs, hyps),
        "confirm_ok": sum(results[i]["exact"] for i in confirm_idx),
        "confirm_total": len(confirm_idx),
        "median_stt_s_2_4": statistics.median(results[i]["stt_s"] for i in speed_idx) if speed_idx else None,
        "stt_s_per_audio_s": sum(r["stt_s"] for r in results) / sum(item["duration_s"] for item in items),
        "name_heard": [NAME_WORD in normalize(hyps[i]) for i in name_idx],
    }


def fmt_mb(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.0f} MB"


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one STT engine on the fixed test set.")
    parser.add_argument("engine", nargs="?", help="engine name, see --list")
    parser.add_argument("--list", action="store_true", help="show the known engines")
    parser.add_argument("--runs", type=int, default=RUNS, help=f"number of runs (default {RUNS})")
    args = parser.parse_args()

    if args.list or not args.engine:
        for name, make in ENGINES.items():
            print(f"  {name:<10} {make().description}")
        return
    if args.engine not in ENGINES:
        sys.exit(f"Unknown engine '{args.engine}'. Known: {', '.join(ENGINES)}")

    engine = ENGINES[args.engine]()
    print(f"STT benchmark: {args.engine} = {engine.description}, {args.runs} runs")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    # 1. test set + brain model loaded, then the baseline (everything as in real use, minus the STT)
    items = load_testset()
    total_audio = sum(item["duration_s"] for item in items)
    print(f"Test set: {len(items)} phrases, {total_audio:.1f} s of audio")
    model, loaded = load_brain_model(load_config())
    print(f"Brain model {model} loaded in Ollama: {'yes' if loaded else 'NO (RAM numbers are not comparable!)'}")
    time.sleep(3)  # let memory settle after the load
    baseline = memory_snapshot()
    footprint = "n/a" if baseline["footprint_mb"] is None else f"{baseline['footprint_mb']:.0f} MB"
    print(f"Baseline: available {baseline['available_mb']:.0f} MB, swap {baseline['swap_mb']:.0f} MB, "
          f"this process: footprint {footprint}, RSS {baseline['rss_mb']:.0f} MB\n")

    # 2. load the engine
    start = time.perf_counter()
    engine.load()
    load_s = time.perf_counter() - start
    memory_samples = [memory_snapshot()]
    print(f"Engine loaded in {load_s:.2f} s")

    # 3. one warm-up call (not counted: the first call often allocates memory)
    start = time.perf_counter()
    engine.transcribe(items[0]["audio"])
    warmup_s = time.perf_counter() - start
    memory_samples.append(memory_snapshot())
    print(f"Warm-up call: {warmup_s:.2f} s (not counted)\n")

    # 4. the runs
    runs = []
    for run_nr in range(1, args.runs + 1):
        results = run_once(engine, items, run_nr, memory_samples)
        summary = summarize_run(items, results)
        runs.append({"summary": summary, "results": results})
        print(f"  run {run_nr}: WER {summary['wer_rule']:.1%} (all 18: {summary['wer_all']:.1%}), "
              f"confirm {summary['confirm_ok']}/{summary['confirm_total']}, "
              f"median 2-4 s phrase {summary['median_stt_s_2_4']:.2f} s, "
              f"{summary['stt_s_per_audio_s']:.3f} s per s of audio\n")

    # 5. memory: worst moment compared to the baseline.
    #    RAM cost for the rule = footprint growth of this process + of the macOS speech services.
    #    The system-wide drop is only shown for information (too noisy: other apps move it too).
    extra_ram = baseline["available_mb"] - min(s["available_mb"] for s in memory_samples)
    swap_growth = growth(memory_samples, baseline, "swap_mb")
    footprint_growth = growth(memory_samples, baseline, "footprint_mb")
    rss_growth = growth(memory_samples, baseline, "rss_mb")
    process_growth = footprint_growth if footprint_growth is not None else rss_growth
    others = other_growth(memory_samples, baseline)
    services = [(name, pid, mb) for name, pid, mb in others if SERVICE_PATTERN.search(name) and mb > 0]
    service_growth = sum(mb for _, _, mb in services)
    ram_cost = process_growth + service_growth

    # 6. the decision rule
    summaries = [run["summary"] for run in runs]
    median_wer = statistics.median(s["wer_rule"] for s in summaries)
    median_wer_all = statistics.median(s["wer_all"] for s in summaries)
    median_speed = statistics.median(s["median_stt_s_2_4"] for s in summaries)
    median_rtf = statistics.median(s["stt_s_per_audio_s"] for s in summaries)
    confirm_every_run = all(s["confirm_ok"] == s["confirm_total"] for s in summaries)
    name_heard = sum(sum(s["name_heard"]) for s in summaries)
    name_total = sum(len(s["name_heard"]) for s in summaries)

    checks = {
        f"WER <= {RULE_WER_MAX:.0%} (without phrase 08)": median_wer <= RULE_WER_MAX,
        "confirmation clips 5/5 in every run": confirm_every_run,
        f"median STT time 2-4 s phrase <= {RULE_SPEED_MAX_S} s": median_speed <= RULE_SPEED_MAX_S,
        f"extra RAM <= {RULE_EXTRA_RAM_MAX_MB} MB (this process + speech services)": ram_cost <= RULE_EXTRA_RAM_MAX_MB,
        f"swap growth <= {RULE_SWAP_GROWTH_MAX_MB} MB": swap_growth <= RULE_SWAP_GROWTH_MAX_MB,
    }

    print("=" * 72)
    print(f"RESULT {args.engine} ({engine.description})")
    print(f"  WER (rule, without 08): {median_wer:.1%}   WER all 18: {median_wer_all:.1%}   (median of {args.runs} runs)")
    print(f"  confirmation clips: {[s['confirm_ok'] for s in summaries]} of {summaries[0]['confirm_total']} per run")
    print(f"  speed: median 2-4 s phrase {median_speed:.2f} s, {median_rtf:.3f} s per s of audio, "
          f"load {load_s:.2f} s, warm-up call {warmup_s:.2f} s")
    print(f"  memory: RAM cost {fmt_mb(ram_cost)} = this process {fmt_mb(process_growth)} + speech services "
          f"{fmt_mb(service_growth)}; swap {fmt_mb(swap_growth)}")
    print(f"          for info: system-wide drop {fmt_mb(extra_ram)}, RSS {fmt_mb(rss_growth)}")
    for name, pid, mb in services:
        print(f"          speech service {name} (pid {pid}): {fmt_mb(mb)}")
    print("          biggest growers among your other processes: "
          + ", ".join(f"{name} {fmt_mb(mb)}" for name, _, mb in others[:5]))
    print(f"  'Buizingen' recognised: {name_heard} of {name_total} (not part of the rule)")
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    verdict = "PASS" if all(checks.values()) else "FAIL"
    print(f"  ==> {verdict}  (plus your own check: did the Mac stutter or freeze during the run?)")
    print("=" * 72)

    # 7. save everything for the comparison at the end
    LOG_DIR.mkdir(exist_ok=True)
    out_file = LOG_DIR / f"bench_stt_{args.engine}.json"
    out_file.write_text(json.dumps({
        "engine": args.engine,
        "description": engine.description,
        "date": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "brain_model": model,
        "brain_loaded": loaded,
        "runs": args.runs,
        "load_s": load_s,
        "warmup_s": warmup_s,
        "baseline": {k: v for k, v in baseline.items() if k != "others"},
        "ram_cost_mb": ram_cost,
        "process_growth_mb": process_growth,
        "service_growth_mb": service_growth,
        "services": services,
        "top_other_growers": others[:10],
        "extra_ram_system_mb": extra_ram,
        "footprint_growth_mb": footprint_growth,
        "rss_growth_mb": rss_growth,
        "swap_growth_mb": swap_growth,
        "median_wer_rule": median_wer,
        "median_wer_all": median_wer_all,
        "median_stt_s_2_4": median_speed,
        "median_stt_s_per_audio_s": median_rtf,
        "confirm_every_run": confirm_every_run,
        "name_heard": f"{name_heard}/{name_total}",
        "checks": checks,
        "verdict": verdict,
        "run_details": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
