"""
Speaker: speak a streamed answer sentence by sentence, WHILE the model is still writing.

    run_turn --on_text(chunk)--> Speaker.feed()
      SentenceSplitter: chunks -> complete sentences (the first one may be cut at a comma)
      -> text queue -> synthesis thread: normalize_for_speech() + tts.synthesize()
      -> audio buffer -> OutputStream callback (sounddevice asks for ~20 ms of audio at a time)

Sentence 2 is synthesized while sentence 1 is playing. One Speaker per answer (like TerminalOutput):

    speaker = Speaker(tts)
    try:
        run_turn(..., on_text=speaker.feed)
        stats = speaker.finish()      # speak the rest, wait until everything has been played
    except KeyboardInterrupt:
        speaker.stop()                # silence at once, drop everything not played yet

Checkpoint (speaks out loud, no LLM):  python -m jarvis.voice.speaker
"""

import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable

from jarvis.voice.normalize import normalize_for_speech

# A full stop after these words is not the end of a sentence
ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "e.g.", "i.e.", "approx.", "no."}
_SENTENCE_END = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s)|\n")   # ". " / "?! " / '." ' / a new line
_COMMA = re.compile(r",(?=\s)")
_CLOSERS = "\"'”’)]"


class SentenceSplitter:
    """
    Turns streamed text chunks into complete sentences (like StreamCleaner turns them into complete words).

    A sentence ends at . ! or ? followed by a space, or at a new line. Not after an abbreviation
    ("Mr. Smith") and not inside a number ("3.5": no space after the point).
    Latency trick: only the FIRST piece of an answer may also end at a comma, if at least
    `first_comma_words` words come before it ("Spotify is now paused," is spoken while the
    model is still writing the rest of the sentence; "Certainly, sir." is not cut).
    A run-on text without any punctuation is cut after `max_chars`, so speech never waits forever.
    """

    def __init__(self, first_comma_words: int = 4, max_chars: int = 300):
        self.first_comma_words = first_comma_words
        self.max_chars = max_chars
        self._buffer = ""
        self._first = True

    def feed(self, chunk: str) -> list[str]:
        """Add a chunk; return the sentences that are complete now (often none, sometimes several)."""
        self._buffer += chunk
        sentences = []
        while True:
            cut = self._next_cut()
            if cut is None:
                return sentences
            piece, self._buffer = self._buffer[:cut].strip(), self._buffer[cut:]
            if piece:
                sentences.append(piece)
                self._first = False

    def flush(self) -> list[str]:
        """Return whatever is left at the end of the answer (a last sentence without a full stop)."""
        piece, self._buffer = self._buffer.strip(), ""
        return [piece] if piece else []

    def _next_cut(self) -> int | None:
        """Position right after the earliest place where the buffer may be cut, or None (keep waiting)."""
        cuts = []
        for match in _SENTENCE_END.finditer(self._buffer):
            if match.group() != "\n":
                word = self._buffer[:match.end()].split()[-1].rstrip(_CLOSERS).lower()
                if word in ABBREVIATIONS:
                    continue
            cuts.append(match.end())
            break
        if self._first:
            for match in _COMMA.finditer(self._buffer):
                if len(self._buffer[:match.start()].split()) >= self.first_comma_words:
                    cuts.append(match.end())
                    break
        if cuts:
            return min(cuts)
        if len(self._buffer) > self.max_chars:
            head = self._buffer[:self.max_chars]
            cut = max(head.rfind(", "), head.rfind(" "))
            return cut + 1 if cut > 0 else self.max_chars
        return None


def _sounddevice_stream(**kwargs):
    """The real audio output (imported here, so the unit tests never need a sound card)."""
    import sounddevice as sd

    return sd.OutputStream(**kwargs)


class Speaker:
    """Speaks one streamed answer. See the module docstring for the flow."""

    def __init__(self, tts, stream_factory: Callable | None = None,
                 normalize: Callable[[str], str] = normalize_for_speech,
                 first_comma_words: int = 4, block_s: float = 0.02):
        self.tts = tts
        self.sample_rate = tts.sample_rate
        self._normalize = normalize
        self._stream_factory = stream_factory or _sounddevice_stream
        self._blocksize = int(self.sample_rate * block_s)
        self._splitter = SentenceSplitter(first_comma_words)

        self._texts = queue.Queue()      # sentences waiting for synthesis (None = the thread must end)
        self._lock = threading.Lock()    # protects everything below that two threads touch
        self._chunks = deque()           # synthesized audio waiting to be played, in order
        self._pos = 0                    # how far the first chunk has been played
        self._pending = 0                # sentences queued or being synthesized
        self._finishing = False          # finish() was called: no more text will come
        self._stopped = False
        self._closed = False
        self._error = None
        self._stream = None

        self.sentences = []              # one dict per spoken sentence: text, spoken, tts_s, audio_s
        self._t_first_text = None        # moments (time.perf_counter) for the latency stats
        self._t_first_sentence = None
        self._t_first_audio = None
        self._gap_s = 0.0                # silence while waiting for audio in the middle of an answer

        self._worker = threading.Thread(target=self._synthesis_loop, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------ called by the main thread
    def feed(self, chunk: str) -> None:
        """on_text callback for run_turn: add a streamed chunk; complete sentences go to synthesis."""
        if self._stopped or self._closed or not chunk:
            return
        if self._t_first_text is None and chunk.strip():
            self._t_first_text = time.perf_counter()
            self._open_stream()          # opening overlaps with the first synthesis
        for sentence in self._splitter.feed(chunk):
            self._queue(sentence)

    def finish(self, timeout_s: float = 120) -> dict:
        """Speak the rest of the answer and wait until all of it has been played. Returns the stats."""
        if not (self._stopped or self._closed):
            for sentence in self._splitter.flush():
                self._queue(sentence)
            self._finishing = True
            deadline = time.perf_counter() + timeout_s
            while self._error is None and not self._done() and time.perf_counter() < deadline:
                time.sleep(0.01)         # small sleeps: Control+C still works while we wait
            if self._stream is not None and self._error is None:
                time.sleep(getattr(self._stream, "latency", 0) or 0)   # let the last block reach the speaker
            self._close(abort=self._error is not None)
        if self._error is not None:
            raise RuntimeError(f"TTS failed: {self._error}") from self._error
        return self.stats

    def stop(self) -> None:
        """Interrupt: silence at once, forget every sentence that was not played yet."""
        self._stopped = True
        with self._lock:
            self._chunks.clear()
            self._pos = 0
        while True:                      # empty the text queue
            try:
                self._texts.get_nowait()
            except queue.Empty:
                break
        self._close(abort=True)

    @property
    def stats(self) -> dict:
        """Latency and synthesis numbers of this answer (None where it did not happen)."""
        def since_first_text(moment):
            if moment is None or self._t_first_text is None:
                return None
            return moment - self._t_first_text

        return {
            "first_audio_s": since_first_text(self._t_first_audio),        # first text chunk -> first sound
            "first_sentence_s": since_first_text(self._t_first_sentence),  # waiting for the model's first sentence
            "first_tts_s": self.sentences[0]["tts_s"] if self.sentences else None,
            "gap_s": self._gap_s,
            "sentences": len(self.sentences),
            "tts_s": sum(s["tts_s"] for s in self.sentences),
            "audio_s": sum(s["audio_s"] for s in self.sentences),
            "stopped": self._stopped,
        }

    # ------------------------------------------------------------ internals
    def _queue(self, sentence: str) -> None:
        with self._lock:
            self._pending += 1
            if self._t_first_sentence is None:
                self._t_first_sentence = time.perf_counter()
        self._texts.put(sentence)

    def _done(self) -> bool:
        with self._lock:
            return self._pending == 0 and not self._chunks

    def _open_stream(self) -> None:
        self._stream = self._stream_factory(samplerate=self.sample_rate, channels=1, dtype="float32",
                                            blocksize=self._blocksize, latency="low", callback=self._callback)
        self._stream.start()

    def _close(self, abort: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        self._texts.put(None)            # the synthesis thread ends after its current sentence
        stream, self._stream = self._stream, None
        if stream is not None:
            if abort:
                stream.abort()           # stop now, drop what is still in the sound card's buffer
            else:
                stream.stop()
            stream.close()

    def _synthesis_loop(self) -> None:
        """Synthesis thread: text queue -> normalize -> TTS -> audio buffer, one sentence at a time."""
        while True:
            sentence = self._texts.get()
            if sentence is None:
                return
            if self._stopped:
                continue
            try:
                spoken = self._normalize(sentence)
                result = self.tts.synthesize(spoken)
            except Exception as error:   # reported by finish(), never lost silently
                with self._lock:
                    self._error = self._error or error
                    self._pending -= 1
                continue
            with self._lock:
                if self._stopped:        # stop() came while this sentence was being synthesized
                    continue
                if len(result.audio):
                    self._chunks.append(result.audio)
                self._pending -= 1
                self.sentences.append({"text": sentence, "spoken": spoken, **result.stats})

    def _callback(self, outdata, frames, time_info, status) -> None:
        """Called by sounddevice (in its own audio thread) every ~20 ms: fill `outdata` with the next samples."""
        out = outdata[:, 0]
        filled = 0
        with self._lock:
            while filled < frames and self._chunks:
                chunk = self._chunks[0]
                take = min(frames - filled, len(chunk) - self._pos)
                out[filled:filled + take] = chunk[self._pos:self._pos + take]
                filled += take
                self._pos += take
                if self._pos >= len(chunk):
                    self._chunks.popleft()
                    self._pos = 0
            if filled and self._t_first_audio is None:
                self._t_first_audio = time.perf_counter()
            answer_busy = not (self._finishing and self._pending == 0 and not self._chunks)
            if filled < frames and self._t_first_audio is not None and answer_busy and not self._stopped:
                self._gap_s += (frames - filled) / self.sample_rate
        out[filled:] = 0.0               # silence for the rest of this block


DEMO_ANSWER = ("Certainly, sir. Spotify is paused, and the volume is now at 40%. "
               "It's 12°C in Buizingen this afternoon, with wind at 15 km/h. "
               "Your meeting is at 3 pm, so you have about two hours. Is there anything else?")


def main() -> None:
    """Checkpoint without LLM: stream a fixed answer word by word (like the model does) and speak it."""
    import argparse

    from jarvis.core.config import load_config
    from jarvis.voice.tts import ENGINES, create_tts

    parser = argparse.ArgumentParser(description="Speak a fake streamed answer, 3 times, with timing.")
    parser.add_argument("--engine", choices=list(ENGINES), help="override the engine in config.toml")
    parser.add_argument("--words-per-s", type=float, default=15.0, help="streaming speed of the fake model")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    tts = create_tts(load_config(), args.engine)
    tts.warm_up()
    print(f"TTS ready: {tts.name}. Streaming {args.words_per_s:.0f} words/s. Control+C = interrupt.\n")

    for run in range(1, args.runs + 1):
        speaker = Speaker(tts)
        try:
            for word in DEMO_ANSWER.split(" "):
                speaker.feed(word + " ")
                time.sleep(1 / args.words_per_s)
            s = speaker.finish()
        except KeyboardInterrupt:
            start = time.perf_counter()
            speaker.stop()
            print(f"\n  run {run}: interrupted, stop() took {time.perf_counter() - start:.3f} s")
            break
        print(f"  run {run}: first text -> first audio {s['first_audio_s']:.2f} s "
              f"(waiting for sentence 1: {s['first_sentence_s']:.2f} s, its TTS {s['first_tts_s']:.2f} s), "
              f"gaps {s['gap_s']:.2f} s, {s['sentences']} sentences, "
              f"TTS {s['tts_s']:.2f} s for {s['audio_s']:.1f} s of speech")
    for sentence in speaker.sentences:
        print(f'    "{sentence["spoken"]}"')


if __name__ == "__main__":
    main()
