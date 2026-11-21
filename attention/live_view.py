#!/usr/bin/env python3
"""
live_view.py - a live window with the attention graph while the scores come in.

    python attention/simulate.py | python attention/live_view.py   # test with generated data
    my_detector | python attention/live_view.py                    # real data: "time,score[,topic]" lines
    python attention/live_view.py --udp 8767                       # or: UDP datagrams "time,score[,topic]"

Twice a second the window redraws the full analysis of everything received so far (attention_profile.py):
raw score (grey), smoothed attention (blue), the typical level, topic boundaries, the most interesting part
so far (shaded) and the highest ▲ / lowest ▼ attention events so far. When the input ends, the window keeps
the final picture and the full report is printed. Close the window to quit.

Time can be seconds, mm:ss / h:mm:ss or ISO timestamps; a non-empty topic starts a new topic there.
"""
import argparse
import socket
import sys
import threading
from datetime import datetime

from attention_profile import (SURFACE, TEXT, TEXT2, AttentionProfile, chart_style, draw_timeline, hms,
                               parse_float, parse_time, print_report, save_report, style_axis)


class Feed:
    """Thread-safe collector: reader threads add lines, the window thread analyses snapshots."""

    def __init__(self):
        self.prof, self.lock = AttentionProfile(), threading.Lock()
        self.version, self.done, self.t0 = 0, False, None

    def add_line(self, line):
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) < 2:
            return
        t, v = parse_time(parts[0]), parse_float(parts[1])
        if t is None or v is None:  # header line or junk
            return
        if isinstance(t, datetime):
            self.t0 = self.t0 or t
            t = (t - self.t0).total_seconds()
        with self.lock:
            self.prof.add(t, v, parts[2] if len(parts) > 2 and parts[2] else None)
            self.version += 1


def read_stdin(feed):
    try:
        for line in sys.stdin:
            feed.add_line(line)
    except (OSError, ValueError):
        pass
    feed.done = True


def read_udp(feed, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))  # localhost only
    while True:
        data, _ = sock.recvfrom(65536)
        for line in data.decode("utf-8", errors="replace").splitlines():
            feed.add_line(line)


def render(ax, res, latest, done, n):
    """One frame of the live window on ax (also used to test the layout off-screen)."""
    ax.clear()
    if res is None:
        style_axis(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, f"waiting for data ... ({n} samples)", ha="center", va="center", transform=ax.transAxes,
                color=TEXT2, fontsize=11)
        return
    topic = next((d["topic"] for d in sorted(res["topics"], key=lambda d: -d["start"]) if d["start"] <= latest[0]),
                 None)
    state = "finished" if done else f"now {latest[1]:.3g}" + (f" in “{topic}”" if topic else "")
    draw_timeline(ax, res, min_span=0 if done else 120)
    ax.set_title(f"Live attention · {hms(latest[0])} · {state}", loc="left", fontsize=12, color=TEXT,
                 weight="bold", pad=24)
    b = res["best_part"]
    best = "most interesting part" if done else "most interesting so far"
    ax.text(0, 1.03, (f"{best}: {hms(b['start'])}-{hms(b['end'])}   ·   " if b else "")
            + f"▲ {len(res['peaks'])} peaks   ▼ {len(res['dips'])} dips   ·   {n} samples",
            transform=ax.transAxes, fontsize=8.5, color=TEXT2, va="bottom")


def main(argv=None):
    p = argparse.ArgumentParser(description="Live window with the attention graph.")
    p.add_argument("--udp", type=int, metavar="PORT", help="receive UDP datagrams on 127.0.0.1:PORT instead of stdin")
    p.add_argument("--smooth", type=float, default=30, help="smoothing window in seconds (default 30)")
    p.add_argument("--window", type=float, default=120, help="length of the most interesting part in s")
    p.add_argument("--report", help="when the input ends, also save the PNG report here")
    a = p.parse_args(argv)

    import matplotlib.pyplot as plt  # GUI backend (not Agg): this one opens a window

    feed = Feed()
    if a.udp:
        threading.Thread(target=read_udp, args=(feed, a.udp), daemon=True).start()
        source = f"udp://127.0.0.1:{a.udp}"
    else:
        threading.Thread(target=read_stdin, args=(feed,), daemon=True).start()
        source = "stdin"
    print(f"live view: reading 'time,score[,topic]' from {source} - close the window to quit", file=sys.stderr)

    chart_style()
    plt.ion()
    fig, ax = plt.subplots(figsize=(10.5, 4.8), facecolor=SURFACE)
    fig.canvas.manager.set_window_title("Live attention")
    fig.subplots_adjust(left=0.08, right=0.93, top=0.84, bottom=0.12)
    seen, res, finished, latest, n = -1, None, False, None, 0
    while plt.fignum_exists(fig.number):
        with feed.lock:
            version, n = feed.version, len(feed.prof)
            if version != seen and n >= 5:
                res = feed.prof.analyse(smooth=a.smooth, window=a.window)
                latest = (feed.prof.t[-1], feed.prof.s[-1])
        done = feed.done
        if version != seen or (done and not finished):
            render(ax, res, latest, done, n)
            fig.canvas.draw_idle()
            seen = version
        if done and not finished:
            finished = True
            if res:
                print_report(res)
                if a.report and save_report(res, a.report):
                    print(f"\nSaved: {a.report}")
            print("\ninput ended - close the window to quit", file=sys.stderr)
        plt.pause(0.5)


if __name__ == "__main__":
    main()
