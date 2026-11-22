#!/usr/bin/env python3
"""
live_insights.py - presenter feedback: a small attention meter during the talk, the full picture right after.

Joins the live attention stream (attention_profile.py) with attention_insights (transcript + LLM summary):
  - while attention samples and transcript words come in, only a small meter is shown (presenter_ui.py): an
    LED eye that lights up and turns from red to green with the audience's attention, so the presenter isn't
    distracted;
  - when the stream ends, a feedback card pops up with the attention over the whole talk; hovering over the
    graph shows the topic at that moment (named by a local LLM via Ollama, else keywords).
    attention_insights also writes report.md, timeline.png and insights.json into --out-dir.

    python attention/simulate.py --demo-talk | python attention/live_insights.py    # test: recorded 30 s talk
    my_feed | python attention/live_insights.py                    # attention + words from your own pipeline
    my_feed | python attention/live_insights.py --words words.json # transcript known in advance (e.g. a script)
    python attention/live_insights.py --udp 8767                   # receive UDP datagrams instead of stdin
    ... --no-window                                                # terminal only: print moments as found

Closing the meter early (its × box) ends the talk there and shows the feedback on what came in.

Input lines (stdin or UDP, one per line / datagram):
    <time>,<attention>[,<topic>]     an attention sample (seconds or mm:ss, any scale), e.g. 12.5,0.73
    word,<start>,<end>,<word>        a transcript word once it has been recognised, e.g. word,12.1,12.4,surprise
"""
import argparse
import json
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from attention_profile import AttentionProfile, hms, parse_float, parse_time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "attention_insights"))
import insights  # noqa: E402  attention_insights/insights.py

OUT_DIR = Path(__file__).resolve().parent.parent / "attention_insights" / "data" / "live"


class Feed:
    """Thread-safe collector: reader threads add lines, the main thread reads them."""

    def __init__(self):
        self.prof, self.words, self.lock = AttentionProfile(), [], threading.Lock()
        self.version, self.done, self.t0 = 0, False, None

    def add_line(self, line):
        line = line.strip()
        if line.lower().startswith("word,"):
            parts = line.split(",", 3)
            if len(parts) == 4:
                a, b, word = parse_time(parts[1]), parse_time(parts[2]), parts[3].strip()
                if isinstance(a, float) and isinstance(b, float) and word:
                    with self.lock:
                        self.words.append({"start": a, "end": b, "word": word})
                        self.version += 1
            return
        parts = [p.strip() for p in line.split(",")]
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


def scaled(duration):
    """Settings that fit the talk so far: a 30 s talk and an hour-long one need different windows."""
    return {"smooth": float(np.clip(duration / 12, 2, 30)), "window": float(np.clip(duration / 6, 4, 120)),
            "min_gap": float(np.clip(duration / 10, 2, 60))}


def said_around(words, t, half=4.0, max_words=18):
    """The words spoken within `half` seconds of time t, as a short quote ('' if nothing was said)."""
    near = [w for w in words if w["end"] > t - half and w["start"] < t + half]
    sel = [w["word"] for w in near]
    if len(sel) > max_words:  # keep the words closest to t
        i = int(np.argmin([abs((w["start"] + w["end"]) / 2 - t) for w in near]))
        lo = max(0, min(i - max_words // 2, len(sel) - max_words))
        sel = (["…"] if lo > 0 else []) + sel[lo:lo + max_words] + (["…"] if lo + max_words < len(sel) else [])
    return " ".join(sel)


def snapshot(feed):
    with feed.lock:
        prof = AttentionProfile()
        prof.t, prof.s, prof.markers = list(feed.prof.t), list(feed.prof.s), list(feed.prof.markers)
        words = sorted(feed.words, key=lambda w: w["start"])
    return prof, words


def analyse(prof, words, lag):
    """Attention profile of the data so far, with what was said at each ▲ / ▼ moment.
    Returns (profile result, (t, attention) on an even grid without gaps)."""
    t = np.asarray(prof.t)
    duration = float(t.max() - t.min())
    res = prof.analyse(**scaled(duration))
    half = float(np.clip(duration / 14, 1.5, 6))  # quote window: ~3 s around the moment in a 40 s talk
    for ev in res["peaks"] + res["dips"]:
        ev["said"] = said_around(words, ev["time"] - lag, half)
    grid = np.asarray(res["profile"]["t"])
    y = np.asarray(res["profile"]["score"], float)
    ok = ~np.isnan(y)
    return res, (grid[ok], y[ok])


def finish(feed, lag, out_dir, model):
    """End of the talk: print the ▲ / ▼ moments, let attention_insights write its report, and return what the
    feedback card shows (None if there is too little attention data)."""
    prof, words = snapshot(feed)
    if len(prof) < 5:
        print("not enough attention data for feedback", file=sys.stderr)
        return None
    res, (t, y) = analyse(prof, words, lag)
    print(f"\n=== Live insights ({hms(t[-1] - t[0])} of attention, {len(words)} words) ===")
    for key, title in (("peaks", "Highest attention"), ("dips", "Lowest attention")):
        print(f"\n{title}")
        for k, ev in enumerate(res[key], 1):
            print(f"  {k}. {hms(ev['time'])}  ({ev['relative']:+.1f} spread)  said: “{ev['said'] or '-'}”")
        if not res[key]:
            print("  none stand out")
    result = {"res": res, "episodes": [], "chunks": [], "tips": [], "report": None, "n_words": len(words),
              "lag": lag}
    if len(words) < 6:
        print("\nToo few transcript words for subjects (send 'word,<start>,<end>,<word>' lines or use --words).")
        return result
    if y.min() < 0 or y.max() > 1:  # attention_insights works on 0..1: rescale other score scales
        y = (y - y.min()) / ((y.max() - y.min()) or 1.0)
        print("(attention rescaled to 0..1 for the insights report)")
    print()
    out_dir = Path(out_dir)
    lines = insights.write_report(out_dir, words, t, y, lag, model=model)
    print("\n".join(lines[:lines.index("## All chunks")]))
    print(f"\nWrote {out_dir / 'report.md'}, {out_dir / 'timeline.png'}, {out_dir / 'insights.json'}")
    data = json.loads((out_dir / "insights.json").read_text())
    llm = any(e.get("why") for e in data["episodes"])  # the keyword fallback writes no explanations or tips
    result.update(episodes=data["episodes"], chunks=data["chunks"], tips=data["insights"] if llm else [],
                  report=out_dir / "report.md")
    return result


def run_terminal(feed, lag):
    """Print every new ▲ / ▼ moment with its quote once it has settled, until the input ends."""
    seen, version = set(), -1
    while True:
        done = feed.done
        if feed.version != version:
            version = feed.version
            prof, words = snapshot(feed)
            duration = max(prof.t) - min(prof.t) if len(prof) else 0.0
            if len(prof) >= 5 and duration >= 8:
                res, _ = analyse(prof, words, lag)
                settled = max(prof.t) - max(2.0, scaled(duration)["min_gap"])  # newer moments may still move
                for key, mark in (("peaks", "▲ high"), ("dips", "▼ low")):
                    for ev in [ev for ev in res[key] if ev["time"] <= settled]:
                        k = (key, round(ev["time"]))
                        if k not in seen and not any((key, k[1] + d) in seen for d in (-2, -1, 1, 2)):
                            seen.add(k)
                            print(f"[{hms(prof.t[-1])}] {mark} attention at {hms(ev['time'])}, said: "
                                  f"“{ev['said'] or '-'}”", flush=True)
        if done:
            return
        time.sleep(0.5)


def main(argv=None):
    p = argparse.ArgumentParser(description="Presenter feedback: live attention meter, feedback card at the end.")
    p.add_argument("--udp", type=int, metavar="PORT", help="receive UDP datagrams on 127.0.0.1:PORT instead of stdin")
    p.add_argument("--words", help="transcript known in advance (words.json from attention_insights/transcribe.py)")
    p.add_argument("--lag", type=float, default=2.0, help="audience reaction lag in seconds (default 2)")
    p.add_argument("--out-dir", default=str(OUT_DIR), help="where the attention_insights report goes")
    p.add_argument("--model", default=insights.MODEL, help="Ollama model for the subject labels and tips")
    p.add_argument("--no-llm", action="store_true", help="keyword labels only, don't ask a local LLM")
    p.add_argument("--no-window", action="store_true", help="terminal only: print moments as they are found")
    p.add_argument("--no-sound", action="store_true", help="no chime when the feedback card appears")
    a = p.parse_args(argv)
    model = None if a.no_llm else a.model

    feed = Feed()
    if a.words:
        data = json.loads(Path(a.words).read_text())
        feed.words = [dict(w) for s in data["segments"] for w in s["words"]]
    if a.udp:
        threading.Thread(target=read_udp, args=(feed, a.udp), daemon=True).start()
        source = f"udp://127.0.0.1:{a.udp}"
    else:
        threading.Thread(target=read_stdin, args=(feed,), daemon=True).start()
        source = "stdin"
    print(f"live insights: reading attention + words from {source}", file=sys.stderr)

    if a.no_window:
        run_terminal(feed, a.lag)
        finish(feed, a.lag, a.out_dir, model)
    else:
        import presenter_ui
        presenter_ui.run(feed, lambda: finish(feed, a.lag, a.out_dir, model), sound=not a.no_sound)


if __name__ == "__main__":
    main()
