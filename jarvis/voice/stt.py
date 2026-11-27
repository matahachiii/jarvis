"""
Speech-to-text: one interface, several engines (like the brain).

Use `create_stt(config)` to get the engine chosen in config.toml [voice.stt].
Both engines were tested in the Phase 3 step 1 benchmark (bench_stt.py):
  "faster-whisper" (default): base.en, WER 4.6%, 0.30 s per phrase, +74 MB
  "apple": macOS on-device recognition, WER 1.5%, 0.11 s per phrase, +212 MB (in a macOS service)

Microphone checkpoint:  python -m jarvis.voice.stt
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000  # what every STT model expects: 16 kHz, mono, float32 in [-1, 1]


@dataclass
class STTResult:
    """What an STT engine returns for one recording."""

    text: str                                   # the transcript ("" if nothing was heard)
    stats: dict = field(default_factory=dict)   # stt_s (time spent), audio_s (length of the recording)


class STT(ABC):
    """Base class for all STT engines. Every engine implements `_transcribe`; `transcribe` adds timing."""

    def __init__(self, settings: dict):
        self.settings = settings

    @property
    def name(self) -> str:
        """Short description of this engine, e.g. 'faster-whisper/base.en'."""
        return self.__class__.__name__

    def transcribe(self, audio: np.ndarray) -> STTResult:
        """Turn a recording (16 kHz mono) into text, with timing. Empty audio gives empty text."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)   # (n, 1) from the microphone -> (n,)
        start = time.perf_counter()
        text = self._transcribe(audio) if len(audio) else ""
        return STTResult(
            text=text.strip(),
            stats={"stt_s": time.perf_counter() - start, "audio_s": len(audio) / SAMPLE_RATE},
        )

    def warm_up(self) -> None:
        """One tiny transcription at start-up, so the first real one is fast (like brain.warm_up)."""
        self.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))

    @abstractmethod
    def _transcribe(self, audio: np.ndarray) -> str:
        """Engine-specific: 1-D float32 audio at 16 kHz -> text."""


class FasterWhisperSTT(STT):
    """faster-whisper (CTranslate2) on the CPU, int8, greedy decoding. The step 1 winner."""

    def __init__(self, settings: dict):
        super().__init__(settings)
        from faster_whisper import WhisperModel  # imported here: only needed when this engine is chosen

        self.model_name = settings.get("model", "base.en")
        self.language = settings.get("language", "en")
        self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")

    @property
    def name(self) -> str:
        return f"faster-whisper/{self.model_name}"

    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self.model.transcribe(
            audio, language=self.language, beam_size=1,
            condition_on_previous_text=False, without_timestamps=True,
        )
        return " ".join(segment.text.strip() for segment in segments)


class AppleSTT(STT):
    """
    macOS built-in speech recognition (SFSpeechRecognizer via pyobjc), ON-DEVICE ONLY.
    requiresOnDeviceRecognition=True keeps the audio on this Mac. If on-device recognition is not
    available, the engine refuses to start: audio is never sent to Apple.
    Needs the Speech Recognition permission for the app that runs Jarvis (Terminal).
    """

    AUTHORIZED = 3
    NO_SPEECH_ERROR = 1110      # "No speech detected": an empty transcript, not an error
    TIMEOUT_S = 15

    def __init__(self, settings: dict):
        super().__init__(settings)
        import tempfile

        import Foundation
        import Speech

        self._Foundation, self._Speech = Foundation, Speech
        self.locale = settings.get("apple_locale", "en-US")
        if Speech.SFSpeechRecognizer.authorizationStatus() != self.AUTHORIZED:
            answer = {}

            def on_answer(status):          # a block that returns nothing, as macOS expects
                answer["status"] = status

            Speech.SFSpeechRecognizer.requestAuthorization_(on_answer)
            self._spin(lambda: "status" in answer, 60)
            if answer.get("status") != self.AUTHORIZED:
                raise RuntimeError("Speech Recognition is not allowed. System Settings > Privacy & Security > "
                                   "Speech Recognition > switch on Terminal.")
        ns_locale = Foundation.NSLocale.alloc().initWithLocaleIdentifier_(self.locale)
        self.recognizer = Speech.SFSpeechRecognizer.alloc().initWithLocale_(ns_locale)
        if self.recognizer is None or not self.recognizer.supportsOnDeviceRecognition():
            raise RuntimeError(f"On-device recognition is not available for {self.locale} on this Mac "
                               "(Jarvis never sends audio to Apple). Use engine = \"faster-whisper\".")
        self.wav_file = Path(tempfile.gettempdir()) / "jarvis_stt_apple.wav"

    @property
    def name(self) -> str:
        return f"apple/{self.locale} (on-device)"

    def _spin(self, done, timeout_s: float) -> None:
        """Let macOS deliver its callbacks: run the run loop in small steps until done() or timeout."""
        deadline = time.perf_counter() + timeout_s
        run_loop = self._Foundation.NSRunLoop.currentRunLoop()
        while not done() and time.perf_counter() < deadline:
            run_loop.runUntilDate_(self._Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.01))

    def _transcribe(self, audio: np.ndarray) -> str:
        import soundfile as sf

        sf.write(self.wav_file, audio, SAMPLE_RATE, subtype="PCM_16")
        url = self._Foundation.NSURL.fileURLWithPath_(str(self.wav_file))
        request = self._Speech.SFSpeechURLRecognitionRequest.alloc().initWithURL_(url)
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
            return ""
        if state["error"] is not None and state["error"][0] != self.NO_SPEECH_ERROR:
            raise RuntimeError(f"Apple speech recognition error {state['error'][0]}: {state['error'][1]}")
        return state["text"]


def create_stt(config: dict) -> STT:
    """Build the STT engine chosen in config['voice']['stt']['engine']."""
    settings = config.get("voice", {}).get("stt", {})
    engine = settings.get("engine", "faster-whisper")
    if engine == "faster-whisper":
        return FasterWhisperSTT(settings)
    if engine == "apple":
        return AppleSTT(settings)
    raise ValueError(f"Unknown STT engine in config.toml: '{engine}' (use \"faster-whisper\" or \"apple\")")


def main() -> None:
    """Checkpoint: record from the microphone (Enter = start/stop) and print the transcript + timing."""
    from jarvis.core.config import load_config
    from jarvis.voice.microphone import Microphone

    config = load_config()
    start = time.perf_counter()
    stt = create_stt(config)
    stt.warm_up()
    print(f"STT ready: {stt.name} (loaded + warmed up in {time.perf_counter() - start:.2f} s)")
    print("Enter = start speaking, Enter = stop. Control+C to quit.")

    mic = Microphone(SAMPLE_RATE)
    while True:
        try:
            input("\n[Enter] start ")
            mic.start()
            input("  recording... [Enter] stop ")
            audio = mic.stop()
        except (KeyboardInterrupt, EOFError):
            mic.stop()
            print()
            break
        result = stt.transcribe(audio)
        print(f'  heard: "{result.text}"')
        print(f"  {result.stats['audio_s']:.1f} s of audio, STT {result.stats['stt_s']:.2f} s")


if __name__ == "__main__":
    main()
