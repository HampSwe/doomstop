"""Microphone features: loudness, spectral centroid, zero-crossing rate, reel score.

Only summary numbers are kept; no audio is saved to disk.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback

import numpy as np

from . import teammates
from .tracks import NumericTrack


def block_features(x: np.ndarray, sr: int) -> dict[str, float]:
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-10)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    centroid = float((spec * freqs).sum() / (spec.sum() + 1e-10))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.int8)))))
    return {"rms_db": 20 * np.log10(rms), "centroid_hz": centroid, "zcr": zcr}


class AudioWorker(threading.Thread):
    def __init__(self, tracks: dict[str, NumericTrack], device=None, sample_rate: int = 16000,
                 block_s: float = 0.25, clock=time.monotonic):
        super().__init__(daemon=True, name="audio")
        self.tracks, self.device, self.sr = tracks, device, sample_rate
        self.block = int(sample_rate * block_s)
        self.clock = clock
        self.q: queue.Queue = queue.Queue(maxsize=50)
        self.stop_event = threading.Event()
        self.error: str | None = None
        self._warned = False

    def _callback(self, indata, frames, time_info, status):
        try:
            self.q.put_nowait((self.clock(), indata[:, 0].copy()))
        except queue.Full:
            pass  # analysis fell behind; drop the block rather than block the audio thread

    def run(self) -> None:
        import sounddevice as sd

        try:
            stream = sd.InputStream(device=self.device, channels=1, samplerate=self.sr,
                                    blocksize=self.block, dtype="float32", callback=self._callback)
            stream.start()
        except Exception as e:
            self.error = f"could not open microphone: {e}"
            return
        try:
            while not self.stop_event.is_set():
                try:
                    t, x = self.q.get(timeout=0.5)
                except queue.Empty:
                    continue
                for k, v in block_features(x, self.sr).items():
                    self.tracks[k].add(t, v)
                self.tracks["reel_score"].add(t, self._reel(x))
        finally:
            stream.stop()
            stream.close()

    def _reel(self, x):
        try:
            return teammates.audio_reel_score(x, self.sr)
        except Exception:
            if not self._warned:
                self._warned = True
                print(f"\n[audio] audio_reel_score raised; logging it as missing:\n{traceback.format_exc()}")
            return None

    def stop(self) -> None:
        self.stop_event.set()
