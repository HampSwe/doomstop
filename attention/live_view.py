#!/usr/bin/env python3
"""
live_view.py - a live window with the attention graph while the scores come in.

    python attention/simulate.py | python attention/live_view.py   # test with generated data
    my_detector | python attention/live_view.py                    # real data: "time,score[,topic]" lines
    python attention/live_view.py --udp 8767                       # or: UDP datagrams "time,score[,topic]"
    python attention/live_view.py --udp 8767 --bind 0.0.0.0 --short
        # the demo: scores from call/score_sender.py on the camera computer, sized for a few-minute talk

Twice a second the window redraws the full analysis of everything received so far (attention_profile.py):
raw score (grey), smoothed attention (blue), the typical level, topic boundaries, the most interesting part
so far (shaded), the highest ▲ / lowest ▼ attention events so far, and the max and min of the smoothed
attention so far (dotted lines, labelled on the right). If no score has arrived for 3 s, the title says so.
When the input ends, or the window is closed with UDP input, the full report is printed (and saved with
--report). Close the window to quit.

Time can be seconds, mm:ss / h:mm:ss or ISO timestamps; a non-empty topic starts a new topic there.
"""
import argparse
import socket
import sys
import threading
import time
from datetime import datetime

import numpy as np

from attention_profile import (BLUE, RED, SURFACE, TEXT, TEXT2, AttentionProfile, chart_style, draw_timeline,
                               hms, parse_float, parse_time, print_report, save_report, style_axis)

STALE_S = 3.0  # no score for this long: the title says so


class Feed:
    """Thread-safe collector: reader threads add lines, the window thread analyses snapshots."""

    def __init__(self):
        self.prof, self.lock = AttentionProfile(), threading.Lock()
        self.version, self.done, self.t0 = 0, False, None
        self.last_rx = None  # time.monotonic() of the last sample

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
            self.last_rx = time.monotonic()


def read_stdin(feed):
    try:
        for line in sys.stdin:
            feed.add_line(line)
    except (OSError, ValueError):
        pass
    feed.done = True


def read_udp(feed, port, bind="127.0.0.1"):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind, port))  # 127.0.0.1: this computer only; 0.0.0.0: also from other computers
    while True:
        data, _ = sock.recvfrom(65536)
        for line in data.decode("utf-8", errors="replace").splitlines():
            feed.add_line(line)


def draw_extremes(ax, res):
    """The max and min of the smoothed attention so far: a dotted line each, a marker, a label on the right.
    Returns ((t_max, max), (t_min, min)), or None if there is nothing to mark."""
    t, sm = np.asarray(res["profile"]["t"]), np.asarray(res["profile"]["smoothed"], float)
    if np.all(np.isnan(sm)):
        return None
    out = []
    for name, i, color, va in (("max", int(np.nanargmax(sm)), BLUE, "bottom"), ("min", int(np.nanargmin(sm)), RED,
                                                                               "top")):
        ax.axhline(sm[i], color=color, linewidth=0.9, linestyle=":", alpha=0.8)
        ax.plot(t[i], sm[i], "D", ms=6, color=color, mec=SURFACE, mew=1.2, zorder=4)
        ax.text(1.005, sm[i], f"{name} {sm[i]:.2f}\nat {hms(t[i])}", transform=ax.get_yaxis_transform(),
                fontsize=8, color=TEXT2, va=va, ha="left", clip_on=False)
        out.append((float(t[i]), float(sm[i])))
    return tuple(out)


def render(ax, res, latest, done, n, stale=None, min_span=120):
    """One frame of the live window on ax (also used to test the layout off-screen).
    stale: seconds since the last sample, when it is long enough to warn about."""
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
    if stale is not None:
        state = f"no score for {stale:.0f} s - is the sender running?"
    draw_timeline(ax, res, min_span=0 if done else min_span)
    ext = draw_extremes(ax, res)
    ax.set_title(f"Live attention · {hms(latest[0])} · {state}", loc="left", fontsize=12,
                 color=RED if stale is not None else TEXT, weight="bold", pad=24)
    b = res["best_part"]
    best = "most interesting part" if done else "most interesting so far"
    ax.text(0, 1.03, (f"{best}: {hms(b['start'])}-{hms(b['end'])}   ·   " if b else "")
            + (f"max {ext[0][1]:.2f} at {hms(ext[0][0])}   min {ext[1][1]:.2f} at {hms(ext[1][0])}   ·   "
               if ext else "")
            + f"▲ {len(res['peaks'])} peaks   ▼ {len(res['dips'])} dips   ·   {n} samples",
            transform=ax.transAxes, fontsize=8.5, color=TEXT2, va="bottom")


def main(argv=None):
    p = argparse.ArgumentParser(description="Live window with the attention graph.")
    p.add_argument("--udp", type=int, metavar="PORT", help="receive UDP datagrams on 127.0.0.1:PORT instead of stdin")
    p.add_argument("--bind", default="127.0.0.1",
                   help="with --udp: the address to listen on; 0.0.0.0 also takes scores from other computers")
    p.add_argument("--smooth", type=float, default=30, help="smoothing window in seconds (default 30)")
    p.add_argument("--window", type=float, default=120, help="length of the most interesting part in s")
    p.add_argument("--min-gap", type=float, default=60, help="seconds between listed peaks or dips (default 60)")
    p.add_argument("--short", action="store_true",
                   help="settings for a talk of a few minutes: --smooth 5 --window 30 --min-gap 15")
    p.add_argument("--report", help="when the input ends (or the window is closed), also save the PNG report here")
    a = p.parse_args(argv)
    if a.short:
        a.smooth, a.window, a.min_gap = 5.0, 30.0, 15.0
    min_span = 60 if a.short else 120

    import matplotlib.pyplot as plt  # GUI backend (not Agg): this one opens a window

    feed = Feed()
    if a.udp:
        threading.Thread(target=read_udp, args=(feed, a.udp, a.bind), daemon=True).start()
        source = f"udp://{a.bind}:{a.udp}"
        if a.bind == "0.0.0.0":
            source += f" (this computer's address on the network: {own_address()})"
    else:
        threading.Thread(target=read_stdin, args=(feed,), daemon=True).start()
        source = "stdin"
    print(f"live view: reading 'time,score[,topic]' from {source} - close the window to quit", file=sys.stderr)

    chart_style()
    plt.ion()
    fig, ax = plt.subplots(figsize=(10.5, 4.8), facecolor=SURFACE)
    fig.canvas.manager.set_window_title("Live attention")
    fig.subplots_adjust(left=0.08, right=0.93, top=0.84, bottom=0.12)
    seen, res, finished, latest, n, was_stale = -1, None, False, None, 0, False

    def final_report():
        if res:
            print_report(res)
            if a.report and save_report(res, a.report):
                print(f"\nSaved: {a.report}")

    try:
        while plt.fignum_exists(fig.number):
            with feed.lock:
                version, n = feed.version, len(feed.prof)
                if version != seen and n >= 5:
                    res = feed.prof.analyse(smooth=a.smooth, window=a.window, min_gap=a.min_gap)
                    latest = (feed.prof.t[-1], feed.prof.s[-1])
                last_rx = feed.last_rx
            done = feed.done
            quiet = time.monotonic() - last_rx if last_rx is not None and not done else 0.0
            stale = quiet if quiet >= STALE_S else None
            if version != seen or (done and not finished) or stale is not None or was_stale:
                render(ax, res, latest, done, n, stale, min_span)
                fig.canvas.draw_idle()
                seen, was_stale = version, stale is not None
            if done and not finished:
                finished = True
                final_report()
                print("\ninput ended - close the window to quit", file=sys.stderr)
            plt.pause(0.5)
    except KeyboardInterrupt:
        pass
    if not finished:  # UDP never "ends": the report comes when the window is closed (or Ctrl+C)
        final_report()


def own_address():
    """This computer's address on the network that has the default route (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "unknown (macOS: ipconfig getifaddr en0)"


if __name__ == "__main__":
    main()
