#!/usr/bin/env python3
"""
send.py - send a signal to listener.py.

    python message/send.py "stop scrolling"

From Python:
    from send import send_signal      # with message/ on the path, or copy the 3-line function
    send_signal("stop scrolling")

Fire-and-forget: it returns immediately and never fails just because no listener is running.
"""
import socket
import sys

DEFAULT_PORT = 8766


def send_signal(text, port=DEFAULT_PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(text.encode("utf-8"), ("127.0.0.1", port))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python message/send.py "stop scrolling" [port]')
    send_signal(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT)
    print(f"sent {sys.argv[1]!r}")
