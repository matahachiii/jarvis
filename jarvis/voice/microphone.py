"""
Microphone: record from the default input device between start() and stop().

The audio arrives in small blocks in a separate audio thread (a callback from the sound driver);
stop() glues them together into one numpy vector (16 kHz, mono, float32 in [-1, 1]).
Needs the Microphone permission for the app that runs Jarvis
(System Settings > Privacy & Security > Microphone > Terminal).
"""

import numpy as np
import sounddevice as sd


class Microphone:
    """Push-to-talk recorder: start() opens the microphone, stop() closes it and returns the audio."""

    def __init__(self, sample_rate: int = 16_000):
        self.sample_rate = sample_rate
        self._chunks: list[np.ndarray] = []
        self._stream = None

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def _callback(self, indata, frames, time_info, status) -> None:
        self._chunks.append(indata.copy())   # called by the audio thread every few milliseconds

    def start(self) -> None:
        """Open the microphone and start collecting audio. Calling it twice does nothing."""
        if self.recording:
            return
        self._chunks = []
        self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32",
                                      callback=self._callback)
        self._stream.start()

    def stop(self) -> np.ndarray:
        """Close the microphone and return everything recorded since start() (empty if not recording)."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._chunks)[:, 0]   # list of (n, 1) blocks -> one flat vector
        self._chunks = []
        return audio
