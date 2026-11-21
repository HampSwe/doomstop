"""macOS: open the LogiInsight COLOUR stream straight over USB (pyuvc), bypassing
macOS's camera list, which only shows the infrared stream.

Run with sudo (macOS must release the camera to libusb):

    sudo .venv/bin/python mac/probe_colour.py

Close every other camera window (phone_detect, look_down, Zoom) first.
Reports which stream opened, its frame rate, saves one frame to
mac/probe_colour.jpg, and checks whether the camera's face AI runs meanwhile.
"""
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "logiinsight_api"))

import cv2  # noqa: E402
import uvc  # noqa: E402

devs = uvc.device_list()
print("UVC devices:", [(d["name"], d["uid"]) for d in devs])
dev = next((d for d in devs if d["name"].startswith("LVNext")), None)
if dev is None:
    sys.exit("LogiInsight camera not found on USB")

cap = uvc.Capture(dev["uid"])
print("opened:", cap.name)
modes = list(cap.available_modes)
print("modes:", modes[:20])


def fmt(m):
    return str(getattr(m, "format_name", "") or (m[4] if len(m) > 4 else ""))


def size(m):
    return (getattr(m, "width", m[0]), getattr(m, "height", m[1]))


def fps(m):
    return getattr(m, "fps", m[2])


pick = [m for m in modes if "MJPG" in fmt(m).upper() and size(m) == (1280, 720)] \
    or [m for m in modes if "MJPG" in fmt(m).upper()] or modes
pick.sort(key=lambda m: (fps(m) != 30, -fps(m) if fps(m) <= 30 else fps(m)))  # 30 fps first
cap.frame_mode = pick[0]
print("using:", cap.frame_mode)


def to_bgr(f):
    for attr in ("bgr", "img"):
        try:
            img = getattr(f, attr)
            if img is not None:
                return img
        except Exception:
            pass
    import numpy as np
    return cv2.imdecode(np.frombuffer(bytes(f.jpeg_buffer), np.uint8), cv2.IMREAD_COLOR)

import logging  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(message)s")
SECONDS = 20
ai = {"distinct": set(), "faces": 0, "error": None, "client": None}


def watch_ai(stop):
    try:
        from logiwebcam import LogiWebcamClient
        c = LogiWebcamClient(ip="192.168.1.10")
        c.connect()
        ai["client"] = c
        print("AI: connected, config set:",
              c.set_ai_config(enable_face_detection=True, enable_face_mesh=True))
        last_print = 0
        while not stop.is_set():
            d = c.get_latest_ai_data()
            if d:
                ai["distinct"].add(d.get("timestamp_ms"))
                ai["faces"] = max(ai["faces"], len(d.get("faces") or []))
            if time.time() - last_print >= 1:
                last_print = time.time()
                print("AI:", "no file/empty" if not d else
                      f"ts={d.get('timestamp_ms')} plugin_loaded={d.get('plugin_loaded')} "
                      f"faces={len(d.get('faces') or [])}")
            time.sleep(0.25)
    except Exception as e:
        ai["error"] = f"{type(e).__name__}: {e}"


stop = threading.Event()
t = threading.Thread(target=watch_ai, args=(stop,), daemon=True)
t.start()

n, t0, saved, diag_done = 0, time.time(), None, False
while time.time() - t0 < SECONDS:
    try:
        f = cap.get_frame(timeout=1.0)
    except uvc.StreamError as e:
        print("stream error:", e)
        continue
    n += 1
    if saved is None and n > 10:
        saved = ROOT / "mac" / "probe_colour.jpg"
        cv2.imwrite(str(saved), to_bgr(f))
stop.set()
t.join(timeout=3)
if ai["client"] is not None:  # health check while the stream is still open
    try:
        ai["client"].diagnose()
    except Exception as e:
        print("diagnose failed:", e)
    ai["client"].disconnect()
cap.close()

print(f"\nframes: {n} in {SECONDS} s ({n / SECONDS:.1f} fps)")
print("saved frame:", saved)
print("camera AI:", "LIVE" if len(ai["distinct"]) >= 4 else "not updating",
      f"({len(ai['distinct'])} distinct results, max faces {ai['faces']})",
      ai["error"] or "")
