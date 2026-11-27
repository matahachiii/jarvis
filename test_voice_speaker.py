"""
Unit tests for jarvis/voice/speaker.py (sentence splitting + sentence-by-sentence speaking).
Fake TTS and fake audio output only: no Kokoro, no sound card, no sound.

    python -m unittest test_voice_speaker
"""

import threading
import time
import unittest

import numpy as np

from jarvis.voice.speaker import SentenceSplitter, Speaker
from jarvis.voice.tts import TTS


def split_stream(chunks, **kwargs):
    """Feed chunks one by one (like the model streams) and return every sentence, flush included."""
    splitter = SentenceSplitter(**kwargs)
    sentences = []
    for chunk in chunks:
        sentences += splitter.feed(chunk)
    return sentences + splitter.flush()


class TestSentenceSplitter(unittest.TestCase):
    def test_sentences_across_chunks(self):
        splitter = SentenceSplitter()
        self.assertEqual(splitter.feed("Certainly, "), [])
        self.assertEqual(splitter.feed("sir. Spotify "), ["Certainly, sir."])
        self.assertEqual(splitter.feed("is paused. "), ["Spotify is paused."])
        self.assertEqual(splitter.flush(), [])

    def test_last_sentence_comes_with_flush(self):
        self.assertEqual(split_stream(["Hello, sir. ", "Anything ", "else"]),
                         ["Hello, sir.", "Anything else"])

    def test_question_exclamation_and_new_line(self):
        self.assertEqual(split_stream(["Shall I? ", "Yes! ", "Done\n", "Next line"]),
                         ["Shall I?", "Yes!", "Done", "Next line"])

    def test_no_cut_inside_numbers_or_after_abbreviations(self):
        self.assertEqual(split_stream(["Pi is 3.14 ", "and Mr. Smith ", "agrees. ", "Fine."]),
                         ["Pi is 3.14 and Mr. Smith agrees.", "Fine."])
        self.assertEqual(split_stream(["Fruit, e.g. apples. ", "Good."]),
                         ["Fruit, e.g. apples.", "Good."])

    def test_closing_quote_stays_with_its_sentence(self):
        self.assertEqual(split_stream(['He said "Hello." ', "Then he left."]),
                         ['He said "Hello."', "Then he left."])

    def test_first_piece_may_end_at_a_comma(self):
        sentences = split_stream(["Spotify is now paused, ", "and the volume is at forty, ", "sir. ", "Done."])
        self.assertEqual(sentences, ["Spotify is now paused,", "and the volume is at forty, sir.", "Done."])

    def test_short_first_comma_is_not_cut(self):
        self.assertEqual(split_stream(["Certainly, sir. ", "Done."]), ["Certainly, sir.", "Done."])

    def test_first_comma_rule_can_be_switched_off(self):
        self.assertEqual(split_stream(["Spotify is now paused, and done."], first_comma_words=10**6),
                         ["Spotify is now paused, and done."])

    def test_run_on_text_is_cut(self):
        sentences = split_stream(["word " * 100], max_chars=50)
        self.assertGreater(len(sentences), 1)
        self.assertTrue(all(len(s) <= 50 for s in sentences))
        self.assertEqual(" ".join(sentences).split(), ["word"] * 100)

    def test_blank_pieces_are_skipped(self):
        self.assertEqual(split_stream(["\n\n", "Hello.\n\n", "  "]), ["Hello."])


class FakeTTS(TTS):
    """Every sentence becomes 50 samples of its own value (0.1, 0.2, ...), so the play order is visible."""

    def __init__(self, delay_s: float = 0.0, fail_on: str | None = None):
        super().__init__({})
        self.delay_s = delay_s
        self.fail_on = fail_on
        self.calls = []

    @property
    def sample_rate(self) -> int:
        return 1000

    def _synthesize(self, text: str) -> np.ndarray:
        if self.fail_on and self.fail_on in text:
            raise ValueError("engine broke")
        time.sleep(self.delay_s)
        self.calls.append(text)
        return np.full(50, len(self.calls) / 10, dtype=np.float32)


class FakeStream:
    """Stands in for sd.OutputStream: a thread that calls the callback block after block and keeps the output."""

    def __init__(self, samplerate, channels, dtype, blocksize, callback, **kwargs):
        self.samplerate, self.channels, self.dtype, self.blocksize = samplerate, channels, dtype, blocksize
        self.callback = callback
        self.played = []
        self.latency = 0.0
        self.started = self.closed = self.aborted = False
        self._running = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.started = self._running = True
        self._thread.start()

    def _run(self):
        while self._running:
            out = np.ones((self.blocksize, 1), dtype=np.float32)   # garbage: the callback must overwrite it
            self.callback(out, self.blocksize, None, None)
            self.played.append(out[:, 0].copy())
            time.sleep(0.001)

    def stop(self):
        self._running = False
        self._thread.join()

    def abort(self):
        self.aborted = True
        self.stop()

    def close(self):
        self.closed = True

    def output(self) -> np.ndarray:
        return np.concatenate(self.played) if self.played else np.zeros(0)


def make_speaker(tts, **kwargs):
    streams = []

    def factory(**stream_kwargs):
        streams.append(FakeStream(**stream_kwargs))
        return streams[-1]

    return Speaker(tts, stream_factory=factory, **kwargs), streams


def feed_words(speaker, text):
    for word in text.split(" "):
        speaker.feed(word + " ")


class TestSpeaker(unittest.TestCase):
    def test_speaks_every_sentence_in_order(self):
        tts = FakeTTS(delay_s=0.005)
        speaker, streams = make_speaker(tts)
        feed_words(speaker, "Certainly, sir. Spotify is paused. Anything else")
        stats = speaker.finish()
        self.assertEqual(tts.calls, ["Certainly, sir.", "Spotify is paused.", "Anything else"])
        played = streams[0].output()
        expected = np.concatenate([np.full(50, v, dtype=np.float32) for v in (0.1, 0.2, 0.3)])
        np.testing.assert_allclose(played[played != 0], expected)   # silence only between and after
        self.assertEqual(stats["sentences"], 3)
        self.assertIsNotNone(stats["first_audio_s"])
        self.assertFalse(stats["stopped"])
        self.assertTrue(streams[0].closed)
        self.assertFalse(streams[0].aborted)

    def test_sentences_are_normalized_before_synthesis(self):
        tts = FakeTTS()
        speaker, _ = make_speaker(tts)
        feed_words(speaker, "The volume is at 40%.")
        speaker.finish()
        self.assertEqual(tts.calls, ["The volume is at forty percent."])
        self.assertEqual(speaker.sentences[0]["text"], "The volume is at 40%.")

    def test_stream_uses_the_engine_sample_rate(self):
        speaker, streams = make_speaker(FakeTTS())
        speaker.feed("Hello. ")
        speaker.finish()
        self.assertEqual((streams[0].samplerate, streams[0].channels, streams[0].dtype), (1000, 1, "float32"))

    def test_no_text_means_no_stream(self):
        speaker, streams = make_speaker(FakeTTS())
        speaker.feed("")
        speaker.feed("   ")
        stats = speaker.finish()
        self.assertEqual(streams, [])
        self.assertIsNone(stats["first_audio_s"])
        self.assertEqual(stats["sentences"], 0)

    def test_stop_silences_at_once_and_drops_the_rest(self):
        tts = FakeTTS(delay_s=0.02)
        speaker, streams = make_speaker(tts)
        feed_words(speaker, " ".join(f"Sentence number {i}." for i in range(10)))
        deadline = time.perf_counter() + 2
        while speaker.stats["first_audio_s"] is None and time.perf_counter() < deadline:
            time.sleep(0.001)
        speaker.stop()
        blocks_at_stop = len(streams[0].played)
        time.sleep(0.1)
        self.assertEqual(len(streams[0].played), blocks_at_stop)   # the stream really stopped
        self.assertTrue(streams[0].aborted and streams[0].closed)
        self.assertLess(len(tts.calls), 10)                        # the rest was never synthesized
        self.assertTrue(speaker.stats["stopped"])
        speaker._worker.join(timeout=1)
        self.assertFalse(speaker._worker.is_alive())               # the synthesis thread ended

    def test_feed_after_stop_is_ignored(self):
        tts = FakeTTS()
        speaker, _ = make_speaker(tts)
        speaker.stop()
        speaker.feed("Hello there. ")
        self.assertEqual(speaker.finish()["sentences"], 0)
        self.assertEqual(tts.calls, [])

    def test_tts_error_is_reported_by_finish(self):
        speaker, streams = make_speaker(FakeTTS(fail_on="boom"))
        feed_words(speaker, "Fine. Then boom.")
        with self.assertRaises(RuntimeError):
            speaker.finish()
        self.assertTrue(streams[0].closed)

    def test_finish_twice_and_stop_after_finish_are_harmless(self):
        speaker, _ = make_speaker(FakeTTS())
        speaker.feed("Hello. ")
        first = speaker.finish()
        self.assertEqual(speaker.finish()["sentences"], first["sentences"])
        speaker.stop()

    def test_overlap_next_sentence_is_ready_before_the_first_ends(self):
        # Sentence audio lasts 50 ms at 1000 Hz; synthesis takes 5 ms -> no gap once speaking has started
        speaker, _ = make_speaker(FakeTTS(delay_s=0.005), block_s=0.01)
        feed_words(speaker, "One. Two. Three. Four.")
        stats = speaker.finish()
        self.assertEqual(stats["sentences"], 4)
        self.assertLess(stats["gap_s"], 0.05)


if __name__ == "__main__":
    unittest.main()
