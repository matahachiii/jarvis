"""
Unit tests for jarvis/voice/tts.py (TTS interface + create_tts). Fake engine only:
no Kokoro, no Piper, no sound.

    python -m unittest test_voice_tts
"""

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from jarvis.voice import tts as tts_module
from jarvis.voice.tts import PROJECT_DIR, TTS, KokoroTTS, _project_path, create_tts


class FakeTTS(TTS):
    """Returns 0.1 s of a fixed signal per word, and remembers what it was asked to say."""

    def __init__(self, settings: dict | None = None, shape: str = "flat"):
        super().__init__(settings or {})
        self.calls = []
        self.shape = shape

    @property
    def sample_rate(self) -> int:
        return 1000

    def _synthesize(self, text: str) -> np.ndarray:
        self.calls.append(text)
        audio = np.full(100 * len(text.split()), 0.5, dtype=np.float64)   # float64 on purpose
        return audio.reshape(-1, 1) if self.shape == "column" else audio


class TestTTSInterface(unittest.TestCase):
    def test_synthesize_returns_float32_audio_and_stats(self):
        engine = FakeTTS()
        result = engine.synthesize("Certainly, sir.")
        self.assertEqual(result.audio.dtype, np.float32)
        self.assertEqual(result.audio.shape, (200,))
        self.assertEqual(result.sample_rate, 1000)
        self.assertAlmostEqual(result.stats["audio_s"], 0.2)
        self.assertGreaterEqual(result.stats["tts_s"], 0.0)

    def test_text_is_stripped_before_the_engine(self):
        engine = FakeTTS()
        engine.synthesize("  Hello, sir.  \n")
        self.assertEqual(engine.calls, ["Hello, sir."])

    def test_empty_text_does_not_call_the_engine(self):
        engine = FakeTTS()
        for text in ["", "   ", "\n"]:
            result = engine.synthesize(text)
            self.assertEqual(len(result.audio), 0)
            self.assertEqual(result.stats["audio_s"], 0.0)
        self.assertEqual(engine.calls, [])

    def test_column_audio_is_flattened(self):
        result = FakeTTS(shape="column").synthesize("one two three")
        self.assertEqual(result.audio.shape, (300,))

    def test_warm_up_synthesizes_once(self):
        engine = FakeTTS()
        engine.warm_up()
        self.assertEqual(len(engine.calls), 1)


class TestCreateTTS(unittest.TestCase):
    def test_unknown_engine(self):
        with self.assertRaises(ValueError):
            create_tts({"voice": {"tts": {"engine": "espeak"}}})

    def test_engine_gets_its_own_settings_table(self):
        config = {"voice": {"tts": {"engine": "fake", "fake": {"voice": "x"}, "other": {"voice": "y"}}}}
        with mock.patch.dict(tts_module.ENGINES, {"fake": FakeTTS}):
            engine = create_tts(config)
        self.assertIsInstance(engine, FakeTTS)
        self.assertEqual(engine.settings, {"voice": "x"})

    def test_engine_override(self):
        config = {"voice": {"tts": {"engine": "kokoro", "fake": {"voice": "x"}}}}
        with mock.patch.dict(tts_module.ENGINES, {"fake": FakeTTS}):
            engine = create_tts(config, engine="fake")
        self.assertEqual(engine.settings, {"voice": "x"})

    def test_kokoro_missing_model_gives_download_hint(self):
        settings = {"model": "/nonexistent/kokoro.onnx", "voices": "/nonexistent/voices.bin"}
        with self.assertRaises(RuntimeError) as caught:
            KokoroTTS(settings)
        self.assertIn("curl -L", str(caught.exception))

    def test_project_paths(self):
        self.assertEqual(_project_path("data/models/x.onnx"), PROJECT_DIR / "data/models/x.onnx")
        self.assertEqual(_project_path("/tmp/x.onnx"), Path("/tmp/x.onnx"))
        self.assertTrue((PROJECT_DIR / "jarvis" / "voice" / "tts.py").exists())


if __name__ == "__main__":
    unittest.main()
