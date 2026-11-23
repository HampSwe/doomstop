"""Pop up the friendly "stop scrolling" card when the attention score stays low.

    nudger = Nudger()
    nudger.update(now, attention)      # every frame; True when a card was just started

The card is message/popup.py (Alexander's), started in its own process so the
camera loop never waits for it. A card appears when

  - the attention score has been below `below` for `after_s` seconds without a break,
  - no card is open, and
  - the last card was started at least `cooldown_s` seconds ago.

Every card says the same: TITLE as the headline and TEXT under it. With the
sound on, the card plays the spoken clip VOICE ("doomscroll detected") and no
other sound; if that file is missing, the card's own chime.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # the repository folder
MESSAGE_DIR = ROOT / "message"
VOICE = ROOT / "sound" / "doomscroll_detected.mp3"     # the card's only sound

BELOW = 0.5          # attention below this counts as distracted
AFTER_S = 2.0        # ... for this long before a card appears (the score itself already lags a few seconds)
COOLDOWN_S = 30.0    # from one card's start to the next
CARD_S = 12.0        # the card closes itself after this long
TITLE = "DOOMSCROLL DETECTED"
TEXT = "Get back to work."


def launch_card(title, text, seconds=CARD_S, position="top-right", sound=True):
    """Start the card in its own process and return it."""
    cmd = [sys.executable, str(MESSAGE_DIR / "popup.py"), "--title", title, "--text", text,
           "--seconds", f"{seconds:g}", "--position", position]
    if not sound:
        cmd.append("--no-sound")
    elif VOICE.exists():
        cmd += ["--voice", str(VOICE)]
    return subprocess.Popen(cmd)


class Nudger:
    def __init__(self, below=BELOW, after_s=AFTER_S, cooldown_s=COOLDOWN_S, seconds=CARD_S,
                 position="top-right", sound=True, launch=launch_card):
        self.below, self.after_s, self.cooldown_s = below, after_s, cooldown_s
        self.seconds, self.position, self.sound, self.launch = seconds, position, sound, launch
        self.shown = 0                   # cards started so far
        self._low_since = None           # when the score last dropped below `below`
        self._last_start = None
        self._card = None                # the open card's process

    @property
    def card_open(self):
        return self._card is not None and self._card.poll() is None

    def update(self, now, attention):
        """Call with every new score. Returns True when a card was just started."""
        if attention is None or attention >= self.below:
            self._low_since = None
            return False
        if self._low_since is None:
            self._low_since = now
        if now - self._low_since < self.after_s or self.card_open:
            return False
        if self._last_start is not None and now - self._last_start < self.cooldown_s:
            return False
        self._card = self.launch(TITLE, TEXT, seconds=self.seconds, position=self.position, sound=self.sound)
        self._last_start = now
        self.shown += 1
        return True

    def close(self):
        """Close a card that is still open (when the program quits)."""
        if self.card_open:
            self._card.terminate()
