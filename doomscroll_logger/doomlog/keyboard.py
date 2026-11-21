"""Global keyboard listener: counts keystrokes and catches the label key.

Only timestamps are recorded, never which key was pressed. The label key
(space by default) is excluded from the keystroke counts, otherwise the
label would leak into the features.
"""

from __future__ import annotations

import time

from .tracks import EventTrack


class KeyboardWorker:
    def __init__(self, events: EventTrack, on_label_key, label_key: str = "space",
                 clock=time.monotonic):
        from pynput import keyboard

        self._kb = keyboard
        self.events, self.on_label_key, self.clock = events, on_label_key, clock
        self.label_key = getattr(keyboard.Key, label_key, None) or keyboard.KeyCode.from_char(label_key)
        self._label_held = False  # ignore key-repeat while the label key is held down
        self.listener = keyboard.Listener(on_press=self._press, on_release=self._release)

    def _press(self, key):
        if key == self.label_key:
            if not self._label_held:
                self._label_held = True
                self.on_label_key()
        else:
            self.events.add(self.clock())

    def _release(self, key):
        if key == self.label_key:
            self._label_held = False

    def start(self) -> None:
        self.listener.start()
        self.listener.wait()
        if getattr(self.listener, "IS_TRUSTED", True) is False:  # macOS only
            print("[warning] keyboard: this terminal is not allowed to monitor input, so SPACE "
                  "and keystrokes will NOT be seen.\n  Fix: System Settings > Privacy & Security > "
                  "Accessibility AND Input Monitoring > enable your terminal app, then restart it.\n")

    def stop(self) -> None:
        self.listener.stop()
