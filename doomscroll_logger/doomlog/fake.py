"""Synthetic signals that depend on the current label.

Lets the whole pipeline (recording -> CSV -> logistic regression) run before
any teammate model or hardware is ready. The numbers are invented.
"""

from __future__ import annotations

import random
import threading
import time


class FakeWorld(threading.Thread):
    def __init__(self, get_label, head, phone, keys, audio, rate_hz: float = 10.0,
                 clock=time.monotonic, seed: int | None = None):
        super().__init__(daemon=True, name="fake")
        self.get_label = get_label
        self.head, self.phone, self.keys, self.audio = head, phone, keys, audio
        self.period, self.clock = 1.0 / rate_hz, clock
        self.rng = random.Random(seed)
        self.stop_event = threading.Event()
        self.error = None

    def _sticky(self, prev: bool, p_true: float, stickiness: float = 0.85) -> bool:
        # Markov chain so values come in runs, like a real head pose would
        if self.rng.random() < stickiness:
            return prev
        return self.rng.random() < p_true

    def run(self) -> None:
        down = phone = False
        r = self.rng
        while not self.stop_event.is_set():
            t = self.clock()
            doom = self.get_label() == 1
            down = self._sticky(down, 0.8 if doom else 0.2)
            phone = self._sticky(phone, 0.55 if doom else 0.08)
            self.head.add(t, down)
            self.phone.add(t, phone)
            if r.random() < (0.4 if doom else 2.5) * self.period:  # keystroke rate per second
                self.keys.add(t)
            self.audio["rms_db"].add(t, r.gauss(-44 if doom else -48, 14))
            self.audio["centroid_hz"].add(t, r.gauss(1900 if doom else 1700, 1500))
            self.audio["zcr"].add(t, max(0.0, r.gauss(0.11 if doom else 0.10, 0.06)))
            self.audio["reel_score"].add(t, min(1.0, max(0.0, r.gauss(0.45 if doom else 0.35, 0.4))))
            self.stop_event.wait(self.period)

    def stop(self) -> None:
        self.stop_event.set()
