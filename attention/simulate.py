#!/usr/bin/env python3
"""
simulate.py - generate a stream of attention scores for testing: a synthetic 25-minute talk (one sample per
5 s on a 1-4 scale) with a strong live demo, a joke, an audience question, a technical problem and a
pricing section that slowly loses the room.

    python attention/simulate.py | python attention/live_view.py   # watch it live (10x speed: ~2.5 min)
    python attention/simulate.py --speed 1                         # real time
    python attention/simulate.py --speed 0 > talk.csv              # all at once, into a file
    python attention/simulate.py --udp 8767                        # send to: live_view.py --udp 8767

Prints "time,score,topic" lines (topic only on a topic's first sample), flushed one by one, so any program
reading stdin sees the samples arrive as they would from a live detector.
"""
import argparse
import os
import socket
import sys
import time

from attention_profile import demo_stream


def main(argv=None):
    p = argparse.ArgumentParser(description="Stream synthetic attention scores for testing.")
    p.add_argument("--speed", type=float, default=10, help="how many times faster than real time (0 = instant)")
    p.add_argument("--seed", type=int, help="noise seed (default: different every run)")
    p.add_argument("--udp", type=int, metavar="PORT", help="send each sample as a UDP datagram to 127.0.0.1:PORT")
    a = p.parse_args(argv)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if a.udp else None
    start, n = time.monotonic(), 0
    try:
        for t, score, topic in demo_stream(a.seed):
            if a.speed > 0:  # wait until this sample is "due"
                delay = start + t / a.speed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            line = f"{t:g},{score:.3f},{topic or ''}"
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
        print(f"simulate: sent {n} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
