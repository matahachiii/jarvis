"""
Text-to-speech: one interface, several engines (like the brain and the STT).

Use `create_tts(config)` to get the engine chosen in config.toml [voice.tts].
Both engines were tested in the Phase 3 step 1 benchmark (bench_tts.py):
  "kokoro" (default): Kokoro v1.0 int8, voice bm_george, lean ONNX Runtime settings:
           ear 5/5, +114 MB, ~1.0 s for a 13-word sentence (~0.2 s per second of speech)
  "piper"  (plan B): Piper en_GB-alan-medium: ear 3/5, +171 MB, 0.14 s for the same sentence
Kokoro was chosen with a recorded exception: step 6 must measure key release -> first audio
<= 1.5 s (median) for a normal question, otherwise switch to "piper" in config.toml.

synthesize() does NOT normalize: the Speaker runs normalize_for_speech() first, sentence by sentence.

Every synthesis runs on ONE thread that lives as long as the engine, whoever calls synthesize().
Measured in step 3.4c: the first synthesis in a NEW thread costs ~1.1 s extra (1.37 s instead of
0.28 s for "Certainly, sir."), so a fresh thread per answer made every answer slow. warm_up()
warms this one thread at start-up.

Checkpoint (speaks out loud):  python -m jarvis.voice.tts
                               python -m jarvis.voice.tts "It's 12°C at 3 pm." --engine piper
"""

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]   # jarvis/voice/tts.py -> the project folder
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"


@dataclass
class TTSResult:
    """What a TTS engine returns for one piece of text."""

    audio: np.ndarray                           # 1-D float32 in [-1, 1] (empty for empty text)
    sample_rate: int
    stats: dict = field(default_factory=dict)   # tts_s (time spent), audio_s (length of the audio)


class TTS(ABC):
    """Base class for all TTS engines. Every engine implements `_synthesize`; `synthesize` adds timing."""

    def __init__(self, settings: dict):
        self.settings = settings
        self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")   # see the docstring

    @property
    def name(self) -> str:
        """Short description of this engine, e.g. 'kokoro/bm_george'."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Samples per second of this engine's audio (known before the first synthesis)."""

    def synthesize(self, text: str) -> TTSResult:
        """
        Turn text into audio, with timing. Empty or blank text gives empty audio (the engine is not called).
        Runs on the engine's own thread and waits for the result; engine errors reach the caller.
        """
        text = text.strip()
        start = time.perf_counter()
        if text:
            audio = self._thread.submit(self._synthesize, text).result()
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        else:
            audio = np.zeros(0, dtype=np.float32)
        return TTSResult(
            audio=audio,
            sample_rate=self.sample_rate,
            stats={"tts_s": time.perf_counter() - start, "audio_s": len(audio) / self.sample_rate},
        )

    def warm_up(self) -> None:
        """One short synthesis at start-up (on the engine's thread), so the first real one is fast."""
        self.synthesize("Hello, sir.")

    @abstractmethod
    def _synthesize(self, text: str) -> np.ndarray:
        """Engine-specific: text -> 1-D float32 audio at self.sample_rate."""


def _project_path(path: str) -> Path:
    """Paths in config.toml are relative to the project folder (works from any working directory)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_DIR / p


class KokoroTTS(TTS):
    """
    Kokoro 82M, int8 ONNX (kokoro-onnx), on the CPU. The step 1 choice (ear 5/5).
    lean = true: ONNX Runtime without its memory arena (366 -> 114 MB extra RAM, a bit slower).
    """

    SAMPLE_RATE = 24_000

    def __init__(self, settings: dict):
        super().__init__(settings)
        self.model_path = _project_path(settings.get("model", "data/models/kokoro/kokoro-v1.0.int8.onnx"))
        self.voices_path = _project_path(settings.get("voices", "data/models/kokoro/voices-v1.0.bin"))
        self.voice = settings.get("voice", "bm_george")
        self.lang = settings.get("lang", "en-gb")
        self.speed = float(settings.get("speed", 1.0))
        self.lean = bool(settings.get("lean", True))
        for path in (self.model_path, self.voices_path):
            if not path.exists():
                raise RuntimeError(f"Missing {path}. Download it with:  curl -L -o {path} {KOKORO_URL}{path.name}")

        from kokoro_onnx import Kokoro  # imported here: only needed when this engine is chosen

        if self.lean:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.enable_cpu_mem_arena = False   # the big RAM saver (step 1.4, attempt 2)
            options.enable_mem_pattern = False
            session = ort.InferenceSession(str(self.model_path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
            self.kokoro = Kokoro.from_session(session, str(self.voices_path))
        else:
            self.kokoro = Kokoro(str(self.model_path), str(self.voices_path))

    @property
    def name(self) -> str:
        return f"kokoro/{self.voice}" + (" (lean)" if self.lean else "")

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    def _synthesize(self, text: str) -> np.ndarray:
        audio, rate = self.kokoro.create(text, voice=self.voice, speed=self.speed, lang=self.lang)
        if rate != self.SAMPLE_RATE:
            raise RuntimeError(f"Kokoro returned {rate} Hz, expected {self.SAMPLE_RATE} Hz")
        return audio


class PiperTTS(TTS):
    """Piper (VITS, ONNX Runtime on the CPU). Plan B: faster, lighter, but sounds less natural (ear 3/5)."""

    def __init__(self, settings: dict):
        super().__init__(settings)
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice  # imported here: only needed when this engine is chosen

        self.voice_name = settings.get("voice", "en_GB-alan-medium")
        lang, speaker, quality = self.voice_name.split("-")          # e.g. en_GB, alan, medium
        folder = f"{lang.split('_')[0]}/{lang}/{speaker}/{quality}"
        files = [f"{folder}/{self.voice_name}.onnx", f"{folder}/{self.voice_name}.onnx.json"]
        try:    # use the cached copy without asking the internet; download only the first time
            model = [hf_hub_download("rhasspy/piper-voices", f, local_files_only=True) for f in files][0]
        except Exception:
            model = [hf_hub_download("rhasspy/piper-voices", f) for f in files][0]
        self.voice = PiperVoice.load(model)

    @property
    def name(self) -> str:
        return f"piper/{self.voice_name}"

    @property
    def sample_rate(self) -> int:
        return self.voice.config.sample_rate

    def _synthesize(self, text: str) -> np.ndarray:
        parts = [np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).astype(np.float32) / 32768
                 for chunk in self.voice.synthesize(text)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


ENGINES = {"kokoro": KokoroTTS, "piper": PiperTTS}


def create_tts(config: dict, engine: str | None = None) -> TTS:
    """Build the TTS engine chosen in config['voice']['tts']['engine'] (or `engine`, to override)."""
    tts_config = config.get("voice", {}).get("tts", {})
    engine = engine or tts_config.get("engine", "kokoro")
    if engine not in ENGINES:
        raise ValueError(f"Unknown TTS engine in config.toml: '{engine}' (use \"kokoro\" or \"piper\")")
    return ENGINES[engine](tts_config.get(engine, {}))


CHECKPOINT_SENTENCES = [
    "Certainly, sir.",
    "Certainly, sir. Spotify is paused, and the volume is now at 40%.",
    "It's 12°C in Buizingen, and the match ended 4-2.",
    "Your meeting is at 3 pm, and it is now 14:30.",
]


def main() -> None:
    """Checkpoint: normalize + synthesize + play a few sentences, with timing per sentence."""
    import argparse

    import sounddevice as sd

    from jarvis.core.config import load_config
    from jarvis.voice.normalize import normalize_for_speech

    parser = argparse.ArgumentParser(description="Speak a few sentences with the configured TTS engine.")
    parser.add_argument("text", nargs="*", help="what to say (default: the checkpoint sentences)")
    parser.add_argument("--engine", choices=list(ENGINES), help="override the engine in config.toml")
    args = parser.parse_args()
    sentences = [" ".join(args.text)] if args.text else CHECKPOINT_SENTENCES

    start = time.perf_counter()
    tts = create_tts(load_config(), args.engine)
    load_s = time.perf_counter() - start
    start = time.perf_counter()
    tts.warm_up()
    print(f"TTS ready: {tts.name}, {tts.sample_rate} Hz (load {load_s:.2f} s, warm-up {time.perf_counter() - start:.2f} s)")

    try:
        for text in sentences:
            spoken = normalize_for_speech(text)
            result = tts.synthesize(spoken)
            s = result.stats
            print(f'\n  text:   "{text}"')
            if spoken != text:
                print(f'  spoken: "{spoken}"')
            print(f"  TTS {s['tts_s']:.2f} s for {s['audio_s']:.1f} s of audio "
                  f"({s['tts_s'] / max(s['audio_s'], 1e-9):.2f} s per s of speech)")
            sd.play(result.audio, result.sample_rate)
            sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
