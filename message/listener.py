#!/usr/bin/env python3
"""
listener.py - turn incoming signals into friendly pop-ups.

Other parts of the project (doomscroll detector, look-down / phone detector ...) send a short text signal
to this listener over a local UDP socket. Known signals pop up a card on screen (see popup.py):

    "stop scrolling"  ->  a friendly retro "look up for a sec" card in the middle of the screen

Run it once, in its own terminal:
    python message/listener.py                  # pop-up card (default)
    python message/listener.py --style alert    # plain macOS alert window instead
    python message/listener.py --style banner   # macOS notification banner instead
    python message/listener.py --dry-run        # only print, no pop-ups (for testing)

Send a signal from another terminal or any script:
    python message/send.py "stop scrolling"
or, from Python, without importing anything from this folder:
    import socket
    socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"stop scrolling", ("127.0.0.1", 8766))

Only signals listed in SIGNALS do anything, and the pop-up text comes from MESSAGES in popup.py, never from
the message itself. The socket listens on 127.0.0.1 only, so nothing outside this computer can trigger it.
"""
import argparse
import os
import random
import re
import socket
import subprocess
import sys
import time
from datetime import datetime

from popup import MESSAGES

DEFAULT_PORT = 8766
HERE = os.path.dirname(os.path.abspath(__file__))

# normalised signal -> kind of message (a key of MESSAGES in popup.py). Add new signals here.
SIGNALS = {
    "stop scrolling": "scrolling",
    "doomscrolling": "scrolling",
}


def normalise(raw):
    """b'  Stop Scrolling!\\n' -> 'stop scrolling' (case, punctuation and extra spaces don't matter)."""
    text = raw.decode("utf-8", errors="replace").lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def pick(kind, last):
    """A random message of this kind, never the same one twice in a row (when there is a choice)."""
    options = [m for m in MESSAGES[kind] if m != last.get(kind)] or MESSAGES[kind]
    last[kind] = random.choice(options)
    return last[kind]


def applescript_string(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def popup(message, style="card", seconds=12, sound=True):
    """Show message = (title, text) without blocking the listener. Returns the pop-up's process."""
    title, text = message
    if style == "card":
        cmd = [sys.executable, os.path.join(HERE, "popup.py"), "--title", title, "--text", text,
               "--seconds", str(seconds)]
        return subprocess.Popen(cmd + ([] if sound else ["--no-sound"]))
    if sys.platform == "darwin":
        t, m = applescript_string(title), applescript_string(text)
        if style == "banner":
            script = f"display notification {m} with title {t}" + (' sound name "Sosumi"' if sound else "")
        else:
            script = f"display alert {t} message {m} giving up after {int(seconds)}"
        return subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Windows / Linux plain alert: a small always-on-top Tk dialog in its own process, closed after `seconds`
    code = ("import sys, tkinter as tk\nfrom tkinter import messagebox\n"
            "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
            "r.after(int(sys.argv[3]) * 1000, r.destroy)\n"
            "messagebox.showinfo(sys.argv[1], sys.argv[2], parent=r)\n")
    return subprocess.Popen([sys.executable, "-c", code, title, text, str(int(seconds))])


def main(argv=None):
    p = argparse.ArgumentParser(description="Show a friendly pop-up when another program sends a signal.")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"UDP port to listen on (default {DEFAULT_PORT})")
    p.add_argument("--style", choices=["card", "alert", "banner"], default="card",
                   help="card = friendly pop-up card (default), alert = plain alert window, banner = macOS "
                        "notification banner")
    p.add_argument("--seconds", type=float, default=12, help="close the pop-up after this long (default 12)")
    p.add_argument("--cooldown", type=float, default=20,
                   help="minimum seconds between two pop-ups of the same kind, so repeated signals don't spam")
    p.add_argument("--no-sound", action="store_true", help="no chime")
    p.add_argument("--dry-run", action="store_true", help="print what would pop up instead of showing it")
    a = p.parse_args(argv)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", a.port))
    except OSError as e:
        sys.exit(f"cannot listen on port {a.port} ({e}) - is another listener already running?")
    print(f"listening on udp://127.0.0.1:{a.port} - signals: {', '.join(repr(s) for s in SIGNALS)}"
          " (Ctrl+C to stop)", flush=True)

    last_shown, last_message = {}, {}  # kind -> time it last popped up / the message it showed
    current = None  # the pop-up that is open right now
    try:
        while True:
            raw, _ = sock.recvfrom(1024)
            signal = normalise(raw)
            stamp = datetime.now().strftime("%H:%M:%S")
            if signal not in SIGNALS:
                print(f"[{stamp}] ignored unknown signal {signal!r}", flush=True)
                continue
            kind, now = SIGNALS[signal], time.monotonic()
            if current is not None and current.poll() is None:
                print(f"[{stamp}] {signal!r}: a pop-up is already open", flush=True)
                continue
            if now - last_shown.get(kind, -1e9) < a.cooldown:
                print(f"[{stamp}] {signal!r}: skipped, last pop-up was {now - last_shown[kind]:.0f} s ago", flush=True)
                continue
            last_shown[kind] = now
            title, text = message = pick(kind, last_message)
            if a.dry_run:
                print(f"[{stamp}] {signal!r} -> would pop up: {title} - {text}", flush=True)
            else:
                current = popup(message, a.style, a.seconds, not a.no_sound)
                print(f"[{stamp}] {signal!r} -> pop-up: {title}", flush=True)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
