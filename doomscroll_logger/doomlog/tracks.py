"""Thread-safe rolling buffers that turn raw signal samples into features."""

from __future__ import annotations

import math
import threading
from collections import deque

NAN = float("nan")


class BinaryTrack:
    """A True/False/None signal sampled at irregular times (e.g. per camera frame)."""

    def __init__(self, horizon: float = 30.0):
        self._buf: deque[tuple[float, bool | None]] = deque()
        self._lock = threading.Lock()
        self._horizon = horizon
        self._true_since: float | None = None
        self.latest: bool | None = None

    def add(self, t: float, value: bool | None) -> None:
        with self._lock:
            self._buf.append((t, value))
            while self._buf and self._buf[0][0] < t - self._horizon:
                self._buf.popleft()
            if value is True and self._true_since is None:
                self._true_since = t
            elif value is not True:
                self._true_since = None
            self.latest = value

    def frac(self, now: float, window: float) -> float:
        """Share of known samples in the last `window` seconds that were True."""
        with self._lock:
            vals = [v for t, v in self._buf if t >= now - window and v is not None]
        return sum(vals) / len(vals) if vals else NAN

    def streak(self, now: float) -> float:
        """Seconds the signal has been continuously True (0 if it is not True now)."""
        with self._lock:
            return now - self._true_since if self._true_since is not None else 0.0

    def rate(self, now: float, window: float) -> float:
        """Samples per second over the window (used as a health check: camera fps)."""
        with self._lock:
            n = sum(1 for t, _ in self._buf if t >= now - window)
        return n / window


class EventTrack:
    """Timestamps of discrete events (keystrokes)."""

    def __init__(self, horizon: float = 30.0, start: float = 0.0):
        self._buf: deque[float] = deque()
        self._lock = threading.Lock()
        self._horizon = horizon
        self._last: float = start  # "since last key" counts from start until the first key

    def add(self, t: float) -> None:
        with self._lock:
            self._buf.append(t)
            while self._buf and self._buf[0] < t - self._horizon:
                self._buf.popleft()
            self._last = t

    def rate(self, now: float, window: float) -> float:
        with self._lock:
            n = sum(1 for t in self._buf if t >= now - window)
        return n / window

    def since_last(self, now: float) -> float:
        with self._lock:
            return now - self._last


class NumericTrack:
    """A float signal (audio loudness etc.); NaN/None samples are ignored."""

    def __init__(self, horizon: float = 30.0):
        self._buf: deque[tuple[float, float]] = deque()
        self._lock = threading.Lock()
        self._horizon = horizon

    def add(self, t: float, value: float | None) -> None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return
        with self._lock:
            self._buf.append((t, float(value)))
            while self._buf and self._buf[0][0] < t - self._horizon:
                self._buf.popleft()

    def mean(self, now: float, window: float) -> float:
        with self._lock:
            vals = [v for t, v in self._buf if t >= now - window]
        return sum(vals) / len(vals) if vals else NAN
