#!/usr/bin/env python3
"""
listener.py - turn incoming signals into pop-ups.

Other parts of the project (doomscroll detector, look-down / phone detector ...) send a short text signal
to this listener over a local UDP socket. Known signals pop up a window on screen:

    "stop scrolling"  ->  pop-up "Scrolling detected"

Run it once, in its own terminal:
    python message/listener.py                  # pop-up window (default)
    python message/listener.py --style banner   # macOS notification banner instead
    python message/listener.py --dry-run        # only print, no pop-ups (for testing)

Send a signal from another terminal or any script:
    python message/send.py "stop scrolling"
or, from Python, without importing anything from this folder:
    import socket
    socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"stop scrolling", ("127.0.0.1", 8766))

Only signals listed in SIGNALS do anything, and the pop-up text comes from that table, never from the
message itself. The socket listens on 127.0.0.1 only, so nothing outside this computer can trigger it.
"""
import argparse
import re
import socket
import subprocess
import sys
import time
from datetime import datetime

DEFAULT_PORT = 8766

# normalised signal -> (pop-up title, pop-up text). Add new signals here.
SCROLLING = ("Scrolling detected", "Put the phone down and get back to the meeting.")
SIGNALS = {
    "stop scrolling": SCROLLING,
    "doomscrolling": SCROLLING,
}


def normalise(raw):
    """b'  Stop Scrolling!\\n' -> 'stop scrolling' (case, punctuation and extra spaces don't matter)."""
    text = raw.decode("utf-8", errors="replace").lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def applescript_string(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def popup(title, text, style="alert", seconds=10):
    """Show a pop-up without blocking the listener. Returns the pop-up's process."""
    if sys.platform == "darwin":
        t, m = applescript_string(title), applescript_string(text)
        if style == "banner":
            script = f'display notification {m} with title {t} sound name "Funk"'
        else:
            script = f"display alert {t} message {m} as warning giving up after {int(seconds)}"
        return subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Windows / Linux: a small always-on-top Tk dialog in its own process, closed after `seconds`
    code = ("import sys, tkinter as tk\nfrom tkinter import messagebox\n"
            "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
            "r.after(int(sys.argv[3]) * 1000, r.destroy)\n"
            "messagebox.showwarning(sys.argv[1], sys.argv[2], parent=r)\n")
    return subprocess.Popen([sys.executable, "-c", code, title, text, str(int(seconds))])


def main(argv=None):
    p = argparse.ArgumentParser(description="Show a pop-up when another program sends a signal.")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"UDP port to listen on (default {DEFAULT_PORT})")
    p.add_argument("--style", choices=["alert", "banner"], default="alert",
                   help="alert = pop-up window (default), banner = macOS notification banner")
    p.add_argument("--seconds", type=float, default=10, help="close the pop-up window after this long")
    p.add_argument("--cooldown", type=float, default=20,
                   help="minimum seconds between two pop-ups of the same kind, so repeated signals don't spam")
    p.add_argument("--dry-run", action="store_true", help="print what would pop up instead of showing it")
    a = p.parse_args(argv)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", a.port))
    except OSError as e:
        sys.exit(f"cannot listen on port {a.port} ({e}) - is another listener already running?")
    print(f"listening on udp://127.0.0.1:{a.port} - signals: {', '.join(repr(s) for s in SIGNALS)}"
          " (Ctrl+C to stop)", flush=True)

    last_shown = {}  # pop-up title -> time it was last shown
    current = None  # the pop-up that is open right now
    try:
        while True:
            raw, _ = sock.recvfrom(1024)
            signal = normalise(raw)
            stamp = datetime.now().strftime("%H:%M:%S")
            if signal not in SIGNALS:
                print(f"[{stamp}] ignored unknown signal {signal!r}", flush=True)
                continue
            title, text = SIGNALS[signal]
            now = time.monotonic()
            if current is not None and current.poll() is None:
                print(f"[{stamp}] {signal!r}: a pop-up is already open", flush=True)
                continue
            if now - last_shown.get(title, -1e9) < a.cooldown:
                print(f"[{stamp}] {signal!r}: skipped, last pop-up was {now - last_shown[title]:.0f} s ago", flush=True)
                continue
            last_shown[title] = now
            if a.dry_run:
                print(f"[{stamp}] {signal!r} -> would pop up: {title} - {text}", flush=True)
            else:
                current = popup(title, text, a.style, a.seconds)
                print(f"[{stamp}] {signal!r} -> pop-up: {title}", flush=True)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
