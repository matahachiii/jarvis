"""
Unit tests for the STT interface and the microphone (Phase 3 step 2).
No real model, no microphone: fake engines and a fake audio stream. Runs in a second.

    python -m unittest test_voice_stt
"""

import unittest
from unittest import mock

import numpy as np

from jarvis.voice import stt as stt_module
from jarvis.voice.microphone import Microphone
from jarvis.voice.stt import SAMPLE_RATE, STT, STTResult, create_stt


class FakeSTT(STT):
    """Returns a fixed text and remembers what it was given."""

    def __init__(self, text=" Hello, sir. "):
        super().__init__({})
        self.text = text
        self.calls = []

    def _transcribe(self, audio):
        self.calls.append(audio)
        return self.text


class TestSTTInterface(unittest.TestCase):
    def test_transcribe_returns_stripped_text_and_stats(self):
        result = FakeSTT().transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
        self.assertIsInstance(result, STTResult)
        self.assertEqual(result.text, "Hello, sir.")
        self.assertAlmostEqual(result.stats["audio_s"], 1.0)
        self.assertGreaterEqual(result.stats["stt_s"], 0.0)

    def test_empty_audio_gives_empty_text_without_calling_the_engine(self):
        engine = FakeSTT()
        result = engine.transcribe(np.zeros(0, dtype=np.float32))
        self.assertEqual(result.text, "")
        self.assertEqual(engine.calls, [])

    def test_microphone_shape_is_flattened_to_float32(self):
        engine = FakeSTT()
        engine.transcribe(np.zeros((800, 1), dtype=np.float64))   # (n, 1) as delivered by sounddevice
        self.assertEqual(engine.calls[0].shape, (800,))
        self.assertEqual(engine.calls[0].dtype, np.float32)

    def test_warm_up_runs_one_small_transcription(self):
        engine = FakeSTT()
        engine.warm_up()
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(len(engine.calls[0]), SAMPLE_RATE // 2)

    def test_text_of_only_spaces_becomes_empty(self):
        self.assertEqual(FakeSTT(text="   ").transcribe(np.ones(100)).text, "")


class TestCreateSTT(unittest.TestCase):
    def test_default_engine_is_faster_whisper(self):
        with mock.patch.object(stt_module, "FasterWhisperSTT") as fake:
            create_stt({})
            fake.assert_called_once_with({})

    def test_engine_from_config_with_settings(self):
        config = {"voice": {"stt": {"engine": "faster-whisper", "model": "base.en"}}}
        with mock.patch.object(stt_module, "FasterWhisperSTT") as fake:
            create_stt(config)
            fake.assert_called_once_with({"engine": "faster-whisper", "model": "base.en"})

    def test_apple_engine(self):
        config = {"voice": {"stt": {"engine": "apple"}}}
        with mock.patch.object(stt_module, "AppleSTT") as fake:
            create_stt(config)
            fake.assert_called_once_with({"engine": "apple"})

    def test_unknown_engine_is_refused(self):
        with self.assertRaises(ValueError):
            create_stt({"voice": {"stt": {"engine": "cloud-whisper"}}})


class FakeStream:
    def __init__(self):
        self.started = self.stopped = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class TestMicrophone(unittest.TestCase):
    def test_stop_without_start_returns_empty_audio(self):
        audio = Microphone().stop()
        self.assertEqual(len(audio), 0)
        self.assertEqual(audio.dtype, np.float32)

    def test_blocks_are_glued_together_and_stream_is_closed(self):
        mic = Microphone()
        stream = FakeStream()
        with mock.patch("jarvis.voice.microphone.sd.InputStream", return_value=stream) as opened:
            mic.start()
            opened.assert_called_once()
            self.assertTrue(mic.recording)
            mic._callback(np.full((160, 1), 0.5, dtype=np.float32), 160, None, None)
            mic._callback(np.full((80, 1), -0.5, dtype=np.float32), 80, None, None)
            audio = mic.stop()
        self.assertEqual(audio.shape, (240,))
        self.assertTrue(stream.started and stream.stopped and stream.closed)
        self.assertFalse(mic.recording)

    def test_start_twice_opens_the_microphone_once(self):
        mic = Microphone()
        with mock.patch("jarvis.voice.microphone.sd.InputStream", return_value=FakeStream()) as opened:
            mic.start()
            mic.start()
            mic.stop()
        opened.assert_called_once()

    def test_a_new_recording_starts_empty(self):
        mic = Microphone()
        with mock.patch("jarvis.voice.microphone.sd.InputStream", side_effect=lambda **kw: FakeStream()):
            mic.start()
            mic._callback(np.ones((100, 1), dtype=np.float32), 100, None, None)
            first = mic.stop()
            mic.start()
            second = mic.stop()
        self.assertEqual(len(first), 100)
        self.assertEqual(len(second), 0)


if __name__ == "__main__":
    unittest.main()
