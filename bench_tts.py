"""
TTS benchmark (Phase 3 step 1.4): one candidate per run, WITH the qwen3 model loaded in Ollama.

Measures:
- time to first audio: seconds to synthesize one sentence of ~13 words (Jarvis will speak
  sentence by sentence, so "first sentence ready" = "first sound")
- RAM: the largest TOTAL at one moment of: footprint growth of this process + of the macOS
  speech/TTS services (sampled every 0.1 s during synthesis, so short-lived processes like `say`
  are caught; `say` starts new helper processes for every sentence, which do not run at the
  same time, so their peaks must not be added up), swap growth
- quality: sir scores 3 sentences by ear (1-5); the WAV files are kept in data/tts_samples/

Usage (Terminal.app; VS Code and Safari closed):
    python -u bench_tts.py say | tee logs/bench_tts_say.txt
    python bench_tts.py --list                  show the engines and their default voice
    python -u bench_tts.py kokoro --voice bm_daniel   try another voice of the same engine
Results are also saved as logs/bench_tts_<engine>[-<voice>].json.

Decision rule, agreed BEFORE the first run. A candidate passes if ALL hold:
  1. time to first audio <= 0.5 s (median of 9 syntheses of the speed sentence)
  2. extra RAM <= 300 MB (this process + TTS services) and swap growth <= 250 MB
  3. by ear: average >= 3.5 and no sentence below 3
Choice: highest ear score; within 0.5 of each other, the lightest wins.
STT + TTS together <= 700 MB (STT = faster-whisper base.en, +74 MB in step 1.3).
Outcome (2026-11-27): no candidate passed. say (Daniel): too robotic by ear. Piper alan: 0.14 s,
+171 MB, ear 3.0 (FAIL). Kokoro int8: ear 4.0 but 0.75 s and +366 MB. Tuning (max 2 attempts):
1) no memory arena + 4 threads: +153 MB, 1.00 s; 2) no memory arena only: +114 MB, 1.00 s, ear 5.0.
Decision by sir: kokoro-lean, with a recorded exception on criterion 1 (time to first audio), on the
condition that step 6 measures key release -> first audio <= 1.5 s for a normal question; otherwise
fall back to Piper (one line in config.toml). Piper stays installed until then.
"""


import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from bench_stt import fmt_mb, growth, load_brain_model, memory_snapshot
from jarvis.core.config import load_config

LOG_DIR = Path("logs")
SAMPLES_DIR = Path("data/tts_samples")
MODELS_DIR = Path("data/models")

SPEED_SENTENCE = "Certainly, sir. Spotify is paused, and the volume is now at forty percent."
EAR_SENTENCES = [
    "Good evening, sir. It is a quarter past seven, and twelve degrees in Buizingen.",
    "Very well, sir. I have closed Notes. Is there anything else?",
    "Eighteen percent of fifty is nine. Shall I open your thesis folder?",
]
RUNS, REPEATS = 3, 3                 # 3 runs x 3 syntheses of the speed sentence = 9 timings

RULE_FIRST_AUDIO_MAX_S = 0.5
RULE_RAM_MAX_MB = 300
RULE_SWAP_GROWTH_MAX_MB = 250
RULE_EAR_AVERAGE_MIN = 3.5
RULE_EAR_EACH_MIN = 3
STT_RAM_MB = 74                      # faster-whisper base.en, measured in step 1.3
RULE_TOTAL_MAX_MB = 700
SERVICE_PATTERN = re.compile(r"^say$|speech|synth|tts|kona|maui|ausp", re.IGNORECASE)


# ---------------------------------------------------------------- engines
# Every engine has: description, load() and synthesize(text) -> (float32 audio, sample_rate).
# Imports happen inside load(), so an engine that is not installed never breaks the others.

class SayEngine:
    """macOS `say` (built in). Synthesizes to a temporary AIFF file, which we read back."""

    def __init__(self, voice: str):
        self.voice = voice
        self.description = f"macOS say, voice {voice}"
        self.out_file = Path(tempfile.gettempdir()) / "jarvis_bench_say.aiff"

    def load(self) -> None:
        voices = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
        if not any(line.split("  ")[0].strip() == self.voice for line in voices.splitlines()):
            raise RuntimeError(f"Voice '{self.voice}' is not installed (see: say -v '?').")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        subprocess.run(["say", "-v", self.voice, "-o", str(self.out_file), text], check=True, timeout=30)
        audio, rate = sf.read(self.out_file, dtype="float32")
        return audio, rate


class PiperEngine:
    """Piper (VITS, ONNX Runtime on the CPU). The voice comes from the rhasspy/piper-voices repo."""

    def __init__(self, voice: str):
        self.voice_name = voice
        self.voice = None
        self.description = f"Piper, voice {voice}"

    def load(self) -> None:
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice

        lang, speaker, quality = self.voice_name.split("-")          # e.g. en_GB, alan, medium
        folder = f"{lang.split('_')[0]}/{lang}/{speaker}/{quality}"
        model = hf_hub_download("rhasspy/piper-voices", f"{folder}/{self.voice_name}.onnx")
        hf_hub_download("rhasspy/piper-voices", f"{folder}/{self.voice_name}.onnx.json")  # next to it
        self.voice = PiperVoice.load(model)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        parts, rate = [], 22050
        for chunk in self.voice.synthesize(text):
            parts.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).astype(np.float32) / 32768)
            rate = chunk.sample_rate
        return np.concatenate(parts), rate


class KokoroEngine:
    """Kokoro 82M, int8 ONNX (kokoro-onnx package), British English."""

    MODEL = MODELS_DIR / "kokoro" / "kokoro-v1.0.int8.onnx"
    VOICES = MODELS_DIR / "kokoro" / "voices-v1.0.bin"

    def __init__(self, voice: str, lean: bool = False):
        self.voice = voice
        self.lean = lean
        self.kokoro = None
        self.description = f"Kokoro v1.0 int8, voice {voice}" + (", lean ONNX Runtime settings" if lean else "")

    def load(self) -> None:
        from kokoro_onnx import Kokoro
        for path in (self.MODEL, self.VOICES):
            if not path.exists():
                raise RuntimeError(f"Missing {path}: download the model files first (see the chat).")
        if not self.lean:
            self.kokoro = Kokoro(str(self.MODEL), str(self.VOICES))
            return
        # Tuning (step 1.4): ONNX Runtime settings for a small Mac.
        # Attempt 1: no memory arena (RAM 366 -> 153 MB) + 4 threads (slower: 0.75 -> 1.00 s).
        # Attempt 2: no memory arena only; threads back to the default (all cores).
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        session = ort.InferenceSession(str(self.MODEL), sess_options=options, providers=["CPUExecutionProvider"])
        self.kokoro = Kokoro.from_session(session, str(self.VOICES))

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        audio, rate = self.kokoro.create(text, voice=self.voice, speed=1.0, lang="en-gb")
        return np.asarray(audio, dtype=np.float32), rate


ENGINES = {
    "say": (SayEngine, "Daniel"),
    "piper": (PiperEngine, "en_GB-alan-medium"),
    "kokoro": (KokoroEngine, "bm_george"),
    "kokoro-lean": (lambda voice: KokoroEngine(voice, lean=True), "bm_george"),
}


# ---------------------------------------------------------------- measurements
class MemorySampler:
    """Takes a memory snapshot every `interval` seconds in a background thread (use with `with`)."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(memory_snapshot())
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.samples.append(memory_snapshot())


def ram_peak(samples: list[dict], baseline: dict) -> tuple[float, float, dict]:
    """
    The worst moment: max over all samples of (growth of this process + growth of all TTS services
    running at that moment). Returns (total, this process, {service name: MB}) at that moment.
    """
    key = "footprint_mb" if baseline["footprint_mb"] is not None else "rss_mb"
    base_others = {pid: value for pid, (_, value) in baseline["others"].items()}
    worst = (0.0, 0.0, {})
    for sample in samples:
        own = (sample[key] if sample[key] is not None else baseline[key]) - baseline[key]
        parts = {}
        for pid, (name, value) in sample["others"].items():
            extra = value - base_others.get(pid, 0.0)
            if SERVICE_PATTERN.search(name) and extra > 0:
                parts[name] = parts.get(name, 0.0) + extra
        total = own + sum(parts.values())
        if total > worst[0]:
            worst = (total, own, parts)
    return worst


def timed_synthesis(engine, text: str) -> tuple[float, np.ndarray, int]:
    start = time.perf_counter()
    audio, rate = engine.synthesize(text)
    return time.perf_counter() - start, audio, rate


def ear_test(engine_label: str, samples: list[tuple[np.ndarray, int]]) -> list[int]:
    """Play each sentence; sir gives a score from 1 to 5 (r = play again)."""
    import sounddevice as sd

    print("\nBy ear: listen to each sentence and give a score:")
    print("  5 = natural, like a real person   4 = good, small glitches   3 = acceptable every day")
    print("  2 = clearly robotic               1 = unbearable")
    scores = []
    for nr, ((audio, rate), text) in enumerate(zip(samples, EAR_SENTENCES), start=1):
        print(f'\n  [{nr}] "{text}"')
        while True:
            sd.play(audio, rate)
            sd.wait()
            answer = input("  Score 1-5 (r = play again): ").strip().lower()
            if answer in {"1", "2", "3", "4", "5"}:
                scores.append(int(answer))
                break
            if answer != "r":
                print("  Type 1, 2, 3, 4, 5 or r.")
    return scores


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one TTS engine.")
    parser.add_argument("engine", nargs="?", help="engine name, see --list")
    parser.add_argument("--voice", help="another voice than the default of this engine")
    parser.add_argument("--list", action="store_true", help="show the known engines")
    parser.add_argument("--no-listen", action="store_true", help="skip the ear test (rule check 3 = FAIL)")
    args = parser.parse_args()

    if args.list or not args.engine:
        for name, (cls, voice) in ENGINES.items():
            print(f"  {name:<8} {cls(voice).description}")
        return
    if args.engine not in ENGINES:
        sys.exit(f"Unknown engine '{args.engine}'. Known: {', '.join(ENGINES)}")

    cls, default_voice = ENGINES[args.engine]
    voice = args.voice or default_voice
    engine = cls(voice)
    label = args.engine if voice == default_voice else f"{args.engine}-{voice}"
    print(f"TTS benchmark: {label} = {engine.description}")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    # 1. brain model loaded, then the baseline
    model, loaded = load_brain_model(load_config())
    print(f"Brain model {model} loaded in Ollama: {'yes' if loaded else 'NO (RAM numbers are not comparable!)'}")
    time.sleep(3)
    baseline = memory_snapshot()
    print(f"Baseline: available {baseline['available_mb']:.0f} MB, swap {baseline['swap_mb']:.0f} MB\n")

    # 2. load + one warm-up synthesis (not counted), with memory sampling
    with MemorySampler() as load_sampler:
        start = time.perf_counter()
        engine.load()
        load_s = time.perf_counter() - start
        warmup_s, _, _ = timed_synthesis(engine, "Hello, sir.")
    print(f"Engine loaded in {load_s:.2f} s, warm-up synthesis {warmup_s:.2f} s (not counted)\n")

    # 3. speed: WITHOUT the sampler, so the measuring does not slow the synthesis down
    timings = []
    for run in range(1, RUNS + 1):
        run_timings = [timed_synthesis(engine, SPEED_SENTENCE)[0] for _ in range(REPEATS)]
        timings.extend(run_timings)
        print(f"  run {run}: " + ", ".join(f"{t:.2f} s" for t in run_timings))
    first_audio_s = statistics.median(timings)

    # 4. memory: synthesize every sentence twice WITH the sampler
    with MemorySampler() as ram_sampler:
        ear_samples, ear_times = [], []
        for _ in range(2):
            ear_samples, ear_times = [], []
            for text in EAR_SENTENCES:
                seconds, audio, rate = timed_synthesis(engine, text)
                ear_samples.append((audio, rate))
                ear_times.append(seconds)
    samples = load_sampler.samples + ram_sampler.samples
    swap_growth = growth(samples, baseline, "swap_mb")
    rss_growth = growth(samples, baseline, "rss_mb")
    ram_cost, process_growth, services = ram_peak(samples, baseline)
    service_growth = sum(services.values())
    extra_ram = baseline["available_mb"] - min(s["available_mb"] for s in samples)

    audio_s = [len(audio) / rate for audio, rate in ear_samples]
    rtf = sum(ear_times) / sum(audio_s)

    # 5. keep the samples, then the ear test
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for nr, (audio, rate) in enumerate(ear_samples, start=1):
        sf.write(SAMPLES_DIR / f"{label}_{nr}.wav", audio, rate, subtype="PCM_16")
    scores = [] if args.no_listen else ear_test(label, ear_samples)
    ear_average = statistics.mean(scores) if scores else None

    # 6. the decision rule
    checks = {
        f"time to first audio <= {RULE_FIRST_AUDIO_MAX_S} s": first_audio_s <= RULE_FIRST_AUDIO_MAX_S,
        f"extra RAM <= {RULE_RAM_MAX_MB} MB (this process + TTS services)": ram_cost <= RULE_RAM_MAX_MB,
        f"swap growth <= {RULE_SWAP_GROWTH_MAX_MB} MB": swap_growth <= RULE_SWAP_GROWTH_MAX_MB,
        f"by ear: average >= {RULE_EAR_AVERAGE_MIN}, none below {RULE_EAR_EACH_MIN}":
            bool(scores) and ear_average >= RULE_EAR_AVERAGE_MIN and min(scores) >= RULE_EAR_EACH_MIN,
        f"STT + TTS <= {RULE_TOTAL_MAX_MB} MB (STT {STT_RAM_MB} MB)": STT_RAM_MB + ram_cost <= RULE_TOTAL_MAX_MB,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"

    print("\n" + "=" * 72)
    print(f"RESULT {label} ({engine.description})")
    print(f"  time to first audio: {first_audio_s:.2f} s (median of {len(timings)}; "
          f"min {min(timings):.2f}, max {max(timings):.2f}), {rtf:.3f} s per s of audio, load {load_s:.2f} s")
    print(f"  memory: RAM cost {fmt_mb(ram_cost)} (worst moment) = this process {fmt_mb(process_growth)} "
          f"+ TTS services {fmt_mb(service_growth)}; swap {fmt_mb(swap_growth)}")
    print(f"          for info: system-wide drop {fmt_mb(extra_ram)}, RSS {fmt_mb(rss_growth)}")
    for name, mb in sorted(services.items(), key=lambda item: -item[1]):
        print(f"          TTS service at that moment: {name} {fmt_mb(mb)}")
    print(f"  by ear: {scores if scores else 'not tested'}"
          + (f", average {ear_average:.2f}" if scores else ""))
    print(f"  samples saved in {SAMPLES_DIR}/{label}_1..3.wav")
    for check, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {check}")
    print(f"  ==> {verdict}  (plus your own check: did the Mac stutter or freeze during the run?)")
    print("=" * 72)

    # 7. save everything for the comparison at the end
    LOG_DIR.mkdir(exist_ok=True)
    out_file = LOG_DIR / f"bench_tts_{label}.json"
    out_file.write_text(json.dumps({
        "engine": label,
        "description": engine.description,
        "date": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "brain_model": model,
        "brain_loaded": loaded,
        "load_s": load_s,
        "warmup_s": warmup_s,
        "timings_s": timings,
        "first_audio_s": first_audio_s,
        "s_per_audio_s": rtf,
        "ram_cost_mb": ram_cost,
        "process_growth_mb": process_growth,
        "service_growth_mb": service_growth,
        "services_at_worst_moment": services,
        "extra_ram_system_mb": extra_ram,
        "swap_growth_mb": swap_growth,
        "ear_scores": scores,
        "ear_average": ear_average,
        "checks": checks,
        "verdict": verdict,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
