#!/usr/bin/env python3
"""Record a labelled session: one feature row every 1/--hz seconds -> data/session_*.csv

Press SPACE to switch between "attentive" and "doomscrolling". Ctrl+C to stop.

    python record.py                 # real camera + mic + keyboard
    python record.py --fake          # synthetic signals, still switch with SPACE
    python record.py --fake --auto-switch 20 --duration 120 --no-keyboard   # fully hands-off
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from doomlog.tracks import BinaryTrack, EventTrack, NumericTrack

LABELS = {0: "attentive", 1: "doomscrolling"}
AUDIO_KEYS = ("rms_db", "centroid_hz", "zcr", "reel_score")

COLUMNS = [
    # meta (not model features)
    "session", "wall_time", "t", "label", "label_name", "secs_since_switch", "cam_fps",
    # camera: head pose
    "head_down", "head_down_frac_1s", "head_down_frac_5s", "head_down_streak_s",
    # camera: phone detector
    "phone_seen", "phone_frac_1s", "phone_frac_5s", "phone_streak_s",
    # keyboard
    "keys_per_s_1s", "keys_per_s_5s", "secs_since_key",
    # audio
    "audio_rms_db_1s", "audio_rms_db_5s", "audio_centroid_hz_1s", "audio_zcr_1s", "reel_score_1s",
]


class Label:
    def __init__(self, clock):
        self.clock = clock
        self.value = 0
        self.switched_at = clock()
        self.n_switches = 0
        self._lock = threading.Lock()

    def toggle(self):
        with self._lock:
            self.value = 1 - self.value
            self.switched_at = self.clock()
            self.n_switches += 1

    def get(self):
        return self.value


def fmt(x):
    if x is None:
        return ""
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, float):
        return "" if math.isnan(x) else round(x, 4)
    return x


def build_row(now, t0, session, label, head, phone, keys, audio):
    return {
        "session": session,
        "wall_time": datetime.now().isoformat(timespec="milliseconds"),
        "t": now - t0,
        "label": label.value,
        "label_name": LABELS[label.value],
        "secs_since_switch": now - label.switched_at,
        "cam_fps": head.rate(now, 2.0),
        "head_down": head.latest,
        "head_down_frac_1s": head.frac(now, 1.0),
        "head_down_frac_5s": head.frac(now, 5.0),
        "head_down_streak_s": head.streak(now),
        "phone_seen": phone.latest,
        "phone_frac_1s": phone.frac(now, 1.0),
        "phone_frac_5s": phone.frac(now, 5.0),
        "phone_streak_s": phone.streak(now),
        "keys_per_s_1s": keys.rate(now, 1.0),
        "keys_per_s_5s": keys.rate(now, 5.0),
        "secs_since_key": keys.since_last(now),
        "audio_rms_db_1s": audio["rms_db"].mean(now, 1.0),
        "audio_rms_db_5s": audio["rms_db"].mean(now, 5.0),
        "audio_centroid_hz_1s": audio["centroid_hz"].mean(now, 1.0),
        "audio_zcr_1s": audio["zcr"].mean(now, 1.0),
        "reel_score_1s": audio["reel_score"].mean(now, 1.0),
    }


def status_line(row, n_rows, n_switches):
    color = "\033[41;97m" if row["label"] == 1 else "\033[42;30m"

    def f(v, spec):
        return "  -  " if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)

    return (f"\r{color} {row['label_name'].upper():^15} \033[0m"
            f" {row['secs_since_switch']:5.1f}s | rows {n_rows:5d} sw {n_switches:3d}"
            f" | down5s {f(row['head_down_frac_5s'], '.2f')} phone5s {f(row['phone_frac_5s'], '.2f')}"
            f" | keys/s {row['keys_per_s_5s']:.1f} | mic {f(row['audio_rms_db_1s'], '.0f')}dB"
            f" | cam {row['cam_fps']:.0f}fps   ")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data", help="output folder (default: data)")
    p.add_argument("--hz", type=float, default=4.0, help="rows per second (default: 4)")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index (see --list-cameras)")
    p.add_argument("--cam-fps", type=float, default=10.0, help="classifier calls per second")
    p.add_argument("--mic", default=None, help="sounddevice input device name or index")
    p.add_argument("--label-key", default="space", help="key that switches the label (default: space)")
    p.add_argument("--start-label", type=int, choices=[0, 1], default=0)
    p.add_argument("--fake", action="store_true", help="synthetic signals instead of camera/mic")
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--no-keyboard", action="store_true", help="no global key listener (use with --auto-switch)")
    p.add_argument("--auto-switch", type=float, default=0, help="switch label every N s (testing)")
    p.add_argument("--duration", type=float, default=0, help="stop after N s (0 = until Ctrl+C)")
    p.add_argument("--list-cameras", action="store_true", help="probe camera indices 0-5 and exit")
    p.add_argument("--quiet", action="store_true", help="no live status line")
    a = p.parse_args(argv)

    if a.list_cameras:
        import cv2
        for i in range(6):
            cap = cv2.VideoCapture(i)
            ok = cap.isOpened() and cap.read()[0]
            if ok:
                print(f"camera {i}: {int(cap.get(3))}x{int(cap.get(4))}")
            cap.release()
        return 0

    clock = time.monotonic
    t0 = clock()
    session = datetime.now().strftime("%Y%m%d_%H%M%S") + ("_fake" if a.fake else "")
    label = Label(clock)
    label.value = a.start_label

    head, phone = BinaryTrack(), BinaryTrack()
    keys = EventTrack(start=t0)
    audio = {k: NumericTrack() for k in AUDIO_KEYS}

    workers = []
    if a.fake:
        from doomlog.fake import FakeWorld
        workers.append(FakeWorld(label.get, head, phone, keys, audio, clock=clock))
    else:
        if not a.no_camera:
            from doomlog.camera import CameraWorker
            workers.append(CameraWorker(a.camera, a.cam_fps, head, phone, clock=clock))
        if not a.no_audio:
            from doomlog.audio import AudioWorker
            mic = int(a.mic) if a.mic and a.mic.isdigit() else a.mic
            workers.append(AudioWorker(audio, device=mic, clock=clock))

    kb = None
    if not a.no_keyboard:
        from doomlog.keyboard import KeyboardWorker
        # in fake mode real keystrokes would pollute the fake key signal, so count into a dummy
        kb = KeyboardWorker(EventTrack() if a.fake else keys, label.toggle, a.label_key, clock=clock)
        kb.start()
    for w in workers:
        w.start()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"session_{session}.csv"
    print(f"recording -> {path}\n"
          f"press {a.label_key.upper()} to switch attentive <-> doomscrolling, Ctrl+C to stop\n")

    period = 1.0 / a.hz
    next_tick = clock() + period
    next_auto = t0 + a.auto_switch if a.auto_switch else math.inf
    n = 0
    try:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            while True:
                time.sleep(max(0.0, next_tick - clock()))
                now = clock()
                next_tick += period
                for wk in workers:
                    if wk.error:
                        print(f"\n[error] {wk.name}: {wk.error}", file=sys.stderr)
                        wk.error = None  # report once
                if now >= next_auto:
                    label.toggle()
                    next_auto += a.auto_switch
                row = build_row(now, t0, session, label, head, phone, keys, audio)
                w.writerow({k: fmt(v) for k, v in row.items()})
                fh.flush()  # a crash or Ctrl+C never loses more than one row
                n += 1
                if not a.quiet:
                    print(status_line(row, n, label.n_switches), end="", flush=True)
                if a.duration and now - t0 >= a.duration:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        for wk in workers:
            wk.stop()
        if kb:
            kb.stop()
    print(f"\n\nsaved {n} rows, {label.n_switches} label switches -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
