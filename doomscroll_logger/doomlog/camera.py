"""Owns the webcam and feeds every frame to both camera classifiers."""

from __future__ import annotations

import threading
import time
import traceback

from . import teammates
from .tracks import BinaryTrack


class CameraWorker(threading.Thread):
    def __init__(self, index: int, fps: float, head: BinaryTrack, phone: BinaryTrack,
                 clock=time.monotonic):
        super().__init__(daemon=True, name="camera")
        self.index, self.period = index, 1.0 / fps
        self.head, self.phone = head, phone
        self.clock = clock
        self.stop_event = threading.Event()
        self.error: str | None = None
        self._warned: set[str] = set()

    def _call(self, name, fn, frame):
        try:
            out = fn(frame)
            return None if out is None else bool(out)
        except Exception:
            if name not in self._warned:  # report once, keep recording
                self._warned.add(name)
                print(f"\n[camera] {name} raised; logging it as missing:\n{traceback.format_exc()}")
            return None

    def run(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            self.error = (f"could not open camera {self.index} "
                          "(wrong --camera index, or Terminal lacks camera permission)")
            return
        try:
            next_t = self.clock()
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                t = self.clock()
                if not ok:
                    time.sleep(0.05)
                    continue
                if t < next_t:  # throttle classifier calls to --cam-fps
                    continue
                next_t = t + self.period
                self.head.add(t, self._call("head_is_down", teammates.head_is_down, frame))
                self.phone.add(t, self._call("phone_in_frame", teammates.phone_in_frame, frame))
        finally:
            cap.release()

    def stop(self) -> None:
        self.stop_event.set()
