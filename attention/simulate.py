#!/usr/bin/env python3
"""
simulate.py - stream attention scores (and optionally transcript words) for testing, as a live feed would.

Default: a synthetic 25-minute talk (one sample per 5 s on a 1-4 scale) with a strong live demo, a joke, an
audience question, a technical problem and a pricing section that slowly loses the room. Or replay a recording:
an attention CSV plus a transcript (words.json from attention_insights/transcribe.py), each word sent when it
has been spoken.

    python attention/simulate.py | python attention/live_view.py      # synthetic talk (10x speed: ~2.5 min)
    python attention/simulate.py --demo-talk | python attention/live_insights.py   # the recorded 30 s talk
    python attention/simulate.py --csv attention.csv --words words.json --speed 1  # any recording
    python attention/simulate.py --speed 0 > talk.csv                  # all at once, into a file
    python attention/simulate.py --udp 8767                            # send to: live_view.py --udp 8767

Lines are flushed one by one, so any program reading stdin sees them arrive as from a live detector:
    <time>,<score>,<topic>       an attention sample (topic only on a topic's first sample)
    word,<start>,<end>,<word>    a transcript word, sent at its end time
"""
import argparse
import csv
import json
import os
import socket
import sys
import time
from pathlib import Path

from attention_profile import demo_stream

DEMO_TALK = Path(__file__).resolve().parent.parent / "attention_insights" / "data"


def recorded_events(csv_path, words_path):
    """(time, line) events from an attention CSV (columns t / time and attention / score) and a words.json."""
    events = []
    if csv_path:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        keys = {k.lower(): k for k in rows[0]} if rows else {}
        tk = next((keys[k] for k in ("t", "t_s", "time") if k in keys), None)
        sk = next((keys[k] for k in ("attention", "score", "engagement", "level") if k in keys), None)
        if tk is None or sk is None:
            sys.exit(f"{csv_path}: need a time column (t / time) and an attention column (attention / score)")
        events += [(float(r[tk]), f"{float(r[tk]):g},{float(r[sk]):.4f},") for r in rows]
    if words_path:
        data = json.loads(Path(words_path).read_text())
        for w in (w for s in data["segments"] for w in s["words"]):
            text = w["word"].replace(",", " ").strip()  # one field: commas would split the line
            events.append((float(w["end"]), f"word,{w['start']:g},{w['end']:g},{text}"))
    return sorted(events, key=lambda e: e[0])


def main(argv=None):
    p = argparse.ArgumentParser(description="Stream attention scores (and transcript words) for testing.")
    p.add_argument("--speed", type=float, help="times faster than real time, 0 = instant "
                                               "(default 10 for the synthetic talk, 1 for recordings)")
    p.add_argument("--seed", type=int, help="noise seed of the synthetic talk (default: different every run)")
    p.add_argument("--csv", help="replay this attention CSV instead of the synthetic talk")
    p.add_argument("--words", help="also send the words of this transcript (words.json) as they are spoken")
    p.add_argument("--demo-talk", action="store_true",
                   help="replay the recorded 30 s talk in attention_insights/data (attention + words)")
    p.add_argument("--udp", type=int, metavar="PORT", help="send each line as a UDP datagram to 127.0.0.1:PORT")
    a = p.parse_args(argv)
    if a.demo_talk:
        a.csv, a.words = a.csv or DEMO_TALK / "attention.csv", a.words or DEMO_TALK / "words.json"
    if a.csv or a.words:
        events, speed = recorded_events(a.csv, a.words), 1.0 if a.speed is None else a.speed
    else:
        events = [(t, f"{t:g},{score:.3f},{topic or ''}") for t, score, topic in demo_stream(a.seed)]
        speed = 10.0 if a.speed is None else a.speed

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if a.udp else None
    start, n, t0 = time.monotonic(), 0, events[0][0] if events else 0.0
    try:
        for t, line in events:
            if speed > 0:  # wait until this line is "due"
                delay = start + (t - t0) / speed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            if sock:
                sock.sendto(line.encode(), ("127.0.0.1", a.udp))
            else:
                print(line, flush=True)
            n += 1
    except BrokenPipeError:  # the reader (e.g. the live window) was closed: stop quietly
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())  # no second error at exit
    except KeyboardInterrupt:
        pass
    finally:
        print(f"simulate: sent {n} lines", file=sys.stderr)


if __name__ == "__main__":
    main()
