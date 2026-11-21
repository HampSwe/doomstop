"""Live view of the logistic-regression inputs for the attention score, each in [0, 1].

One value per frame, computed only from the current (lightly smoothed) measurements:

  eyes                   upper-eyelid drop relative to the "looking at the screen" pose
                         (LookDownDetector.eyes, clipped to [0, 1]; about 1 = clearly down)
  no_face                1 when no face has been seen for FACE_LOST_S, else 0
  phone_conf             best phone-detector confidence over the last PHONE_HOLD_S
  hand_on_phone          how much of the closest hand is on the held phone box (see below)
  phone_conf x eyes      a phone is there AND you look down
  phone_conf x hand      a phone is there AND a hand is on it

hand_on_phone, per hand result:
  1. distance of each of the hand's 21 landmarks to the phone box (0 inside it)
  2. divided by the hand's size (wrist -> middle-finger base), so it works near and far
  3. near = exp(-dist / (HAND_DECAY * hand_size)) per landmark, mean over the 21 landmarks
  4. max over the hands (one hand on the phone is enough)
  5. EMA with HAND_SMOOTH_S

eyes and no_face come from the camera's face mesh, which the camera computes from its
colour stream. Phone and hands run on this PC, on either stream:

    python logistic_vars.py               # phone + hands on the colour picture
    python logistic_vars.py --ir          # phone + hands on the near-infrared picture
                                          #   (portrait, its tall FOV reaches the desk)
    python logistic_vars.py --device 1    # name the colour camera index

Keys: c re-measure the screen pose   m mesh points   q / Esc quit
"""

import argparse
import math
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from look_down import (FACE_LOST_S, PHONE_HOLD_S, LookDownDetector, draw_face, features,
                       put, put_centred)
from face_stream import connect_device_ai, pick_camera   # also puts the SDK on sys.path
from camera_utils import CameraNotFoundError, backend_id, describe, find_camera, open_camera
from logiwebcam import AiStalenessMonitor
from phone_detect import IR_WARMUP_FRAMES, PhoneDetector

ROOT = Path(__file__).resolve().parent
HAND_MODEL = ROOT / "models" / "hand_landmarker.task"
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")

HAND_DECAY = 0.5         # landmark closeness falls to 1/e at this many hand sizes from the box
HAND_SMOOTH_S = 0.25     # EMA time constant for hand_on_phone (same as SMOOTH_S for the face)
HAND_STALE_S = 0.5       # no hand result for this long: hand_on_phone decays to 0
HISTORY_S = 10.0         # length of the trace behind each bar
PANEL_W = 480
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

WINDOW = "logistic_vars (c: screen pose, q: quit)"
RED, GREEN, GREY, WHITE, CYAN, YELLOW, MAGENTA = ((0, 0, 255), (0, 200, 0), (150, 150, 150),
                                                  (255, 255, 255), (255, 200, 0), (0, 255, 255),
                                                  (255, 0, 255))
VARIABLES = ("eyes", "no_face", "phone_conf", "hand_on_phone", "phone_conf x eyes",
             "phone_conf x hand")


# --------------------------------------------------------------------------- #
# Background detectors
# --------------------------------------------------------------------------- #

class Worker(threading.Thread):
    """Runs `fn(frame)` on the newest submitted frame; older unprocessed frames are dropped."""

    def __init__(self, name, fn):
        super().__init__(daemon=True, name=name)
        self.fn = fn
        self.result, self.result_t = None, -1e9
        self._frame = None
        self._wake = threading.Event()
        self._quit = False
        self._warned = False

    def submit(self, frame):
        self._frame = frame
        self._wake.set()

    def run(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._quit:
                return
            frame, self._frame = self._frame, None
            if frame is None:
                continue
            try:
                result = self.fn(frame)
            except Exception as e:  # keep everything else running if one detector breaks
                result = None
                if not self._warned:
                    self._warned = True
                    print(f"WARNING: {self.name} detection failed: {e}")
            self.result, self.result_t = result, time.time()

    def stop(self):
        self._quit = True
        self._wake.set()


class HandLandmarker:
    """MediaPipe Hand Landmarker: up to two hands, each a (21, 2) array in frame pixels."""

    def __init__(self, min_conf=0.4):
        from mediapipe.tasks.python import BaseOptions, vision
        if not HAND_MODEL.exists():
            print(f"Downloading the MediaPipe hand model to {HAND_MODEL} ...")
            HAND_MODEL.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL)
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf)
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._last_ts = -1

    def __call__(self, frame):
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB if frame.ndim == 2 else cv2.COLOR_BGR2RGB)
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)   # must increase strictly
        self._last_ts = ts
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                 data=np.ascontiguousarray(rgb)), ts)
        h, w = frame.shape[:2]
        return [np.array([(p.x * w, p.y * h) for p in lms], dtype=np.float32)
                for lms in res.hand_landmarks]


# --------------------------------------------------------------------------- #
# Variables
# --------------------------------------------------------------------------- #

def hand_on_phone_raw(hands, box):
    """0..1 for one frame: closeness of the best hand's landmarks to the phone box."""
    if box is None or not hands:
        return 0.0
    x, y, w, h = box
    best = 0.0
    for lm in hands:
        dx = np.maximum(np.maximum(x - lm[:, 0], lm[:, 0] - (x + w)), 0.0)
        dy = np.maximum(np.maximum(y - lm[:, 1], lm[:, 1] - (y + h)), 0.0)
        dist = np.hypot(dx, dy)
        hand_size = max(float(np.linalg.norm(lm[0] - lm[9])), 1.0)
        best = max(best, float(np.exp(-dist / (HAND_DECAY * hand_size)).mean()))
    return best


class Variables:
    """Turns the latest detector results into the six [0, 1] inputs."""

    def __init__(self):
        self.sightings = deque()       # (time, score, box) of each phone result with a phone
        self.hand = 0.0                # smoothed hand_on_phone
        self._hand_t = None            # time of the last hand result folded into the EMA
        self.history = {k: deque() for k in VARIABLES}

    def add_phones(self, phones, t):
        if phones:
            x, y, w, h, score = phones[0]
            self.sightings.append((t, score, (x, y, w, h)))

    def held_phone(self, now):
        """(best confidence, most recent box) over the last PHONE_HOLD_S."""
        while self.sightings and now - self.sightings[0][0] > PHONE_HOLD_S:
            self.sightings.popleft()
        if not self.sightings:
            return 0.0, None
        return max(s[1] for s in self.sightings), self.sightings[-1][2]

    def add_hands(self, hands, t, box):
        raw = hand_on_phone_raw(hands, box)
        dt = None if self._hand_t is None else t - self._hand_t
        self._hand_t = t
        blend = 1.0 if dt is None or dt <= 0 or dt > 1.0 else 1.0 - math.exp(-dt / HAND_SMOOTH_S)
        self.hand += blend * (raw - self.hand)

    def compute(self, det, face_lost, now, hands_t):
        phone_conf, _ = self.held_phone(now)
        if now - hands_t > HAND_STALE_S:              # hand detector stalled: let it fade
            self.add_hands([], now, None)
        eyes = 0.0 if det.base is None else min(max(det.eyes, 0.0), 1.0)
        v = {
            "eyes": eyes,
            "no_face": 1.0 if face_lost else 0.0,
            "phone_conf": phone_conf,
            "hand_on_phone": self.hand,
            "phone_conf x eyes": phone_conf * eyes,
            "phone_conf x hand": phone_conf * self.hand,
        }
        for k, val in v.items():
            hist = self.history[k]
            hist.append((now, val))
            while hist and now - hist[0][0] > HISTORY_S:
                hist.popleft()
        return v


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

def to_bgr(frame):
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame


def draw_hands(frame, hands):
    for lm in hands:
        pts = lm.astype(int)
        for a, b in HAND_EDGES:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), MAGENTA, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p), 3, WHITE, -1)


def draw_phone(frame, box, conf):
    if box is None:
        return
    x, y, w, h = (int(round(v)) for v in box)
    cv2.rectangle(frame, (x, y), (x + w, y + h), RED, 3)
    cv2.putText(frame, f"phone {conf:.2f}", (x, max(y - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, RED, 2, cv2.LINE_AA)


def colour_for(v):
    """Green at 0, through yellow, to red at 1."""
    v = min(max(v, 0.0), 1.0)
    return (0, int(200 * min(1.0, 2 * (1 - v))), int(255 * min(1.0, 2 * v)))


def draw_panel(height, values, history, now, status):
    panel = np.full((height, PANEL_W, 3), 30, np.uint8)
    put(panel, "logistic regression inputs", (16, 34), 0.7, WHITE, 2, backing=False)
    if status:
        put(panel, status, (16, 62), 0.55, YELLOW, 1, backing=False)
    top, bottom = 80, height - 40
    row_h = (bottom - top) // len(VARIABLES)
    bar_x0, bar_x1 = 16, PANEL_W - 16
    for i, name in enumerate(VARIABLES):
        y0 = top + i * row_h
        v = values[name]
        put(panel, name, (bar_x0, y0 + 20), 0.6, WHITE, 1, backing=False)
        put(panel, f"{v:.2f}", (bar_x1 - 52, y0 + 20), 0.6, colour_for(v), 2, backing=False)
        # bar
        by0, by1 = y0 + 30, y0 + 46
        cv2.rectangle(panel, (bar_x0, by0), (bar_x1, by1), GREY, 1)
        fill = bar_x0 + int((bar_x1 - bar_x0) * v)
        if fill > bar_x0:
            cv2.rectangle(panel, (bar_x0 + 1, by0 + 1), (fill, by1 - 1), colour_for(v), -1)
        # trace of the last HISTORY_S seconds
        ty0, ty1 = by1 + 6, y0 + row_h - 10
        if ty1 - ty0 < 10:
            continue
        cv2.rectangle(panel, (bar_x0, ty0), (bar_x1, ty1), (60, 60, 60), 1)
        pts = [(bar_x1 - int((now - t) / HISTORY_S * (bar_x1 - bar_x0)),
                ty1 - int(val * (ty1 - ty0))) for t, val in history[name]]
        if len(pts) > 1:
            cv2.polylines(panel, [np.array(pts, np.int32)], False, CYAN, 1, cv2.LINE_AA)
    put(panel, f"traces: last {HISTORY_S:.0f} s", (16, height - 16), 0.5, GREY, 1, backing=False)
    return panel


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def open_streams(args, client):
    """(colour capture, IR capture or None, description)."""
    index = args.device if args.device is not None else pick_camera(client)
    # With IR open too, MJPG colour starves the IR stream (they share the JPEG encoder).
    fourcc = "YUY2" if args.ir else "MJPG"
    rgb, info = open_camera(index, width=args.width, height=args.height, fourcc=fourcc, fps=30)
    text = describe(info)
    ir = None
    if args.ir:
        if args.ir_device is not None:
            ir_index, backend = args.ir_device, None
        else:
            print("Looking for the near-infrared (CPD) camera ...")
            ir_index, name = find_camera("ir")
            backend = backend_id(name)
        ir, ir_info = open_camera(ir_index, backend, 512, 640, "MJPG", 30)
        text += f"  +  IR {describe(ir_info)}"
    return rgb, ir, text


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", type=int, help="colour camera index (default: the one feeding the camera AI)")
    p.add_argument("--ir", action="store_true", help="run phone and hand detection on the near-infrared stream")
    p.add_argument("--ir-device", type=int, help="IR camera index (default: find it)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--phone-score", type=float, default=0.2,
                   help="lowest phone confidence reported (default 0.2; lower keeps phone_conf more continuous)")
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the picture")
    args = p.parse_args()

    client, reason = connect_device_ai(mesh=True)
    if client is None:
        sys.exit(f"ERROR: this needs the camera's face mesh, which is unavailable: {reason}")
    try:
        rgb, ir, text = open_streams(args, client)
    except CameraNotFoundError as e:
        client.disconnect()
        sys.exit(f"ERROR: {e}")
    print(f"Opened {text}")
    print("Look at the screen for a second while it measures your normal pose.")

    det = LookDownDetector()
    monitor = AiStalenessMonitor()
    variables = Variables()
    phones_w = Worker("phone", PhoneDetector(min_score=args.phone_score).detect)
    hands_w = Worker("hands", HandLandmarker())
    phones_w.start()
    hands_w.start()
    seen_phone_t = seen_hand_t = -1e9
    hands = []
    face, last_ts, last_face_t = None, None, time.time()
    show_mesh = False
    ir_skip = IR_WARMUP_FRAMES
    try:
        while True:
            ok, frame = rgb.read()
            if not ok or frame is None:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            view = frame                                   # what phone + hands run on and get drawn
            if ir is not None:
                ok_ir, ir_frame = ir.read()
                if not ok_ir or ir_frame is None:
                    continue
                if ir_skip > 0:                            # IR auto-exposure ramp
                    ir_skip -= 1
                    continue
                view = to_bgr(ir_frame)
            now = time.time()
            phones_w.submit(view.copy())
            hands_w.submit(view.copy())

            # new detector results
            if phones_w.result_t > seen_phone_t:
                seen_phone_t = phones_w.result_t
                variables.add_phones(phones_w.result or [], seen_phone_t)
            _, box = variables.held_phone(now)
            if hands_w.result_t > seen_hand_t:
                seen_hand_t = hands_w.result_t
                hands = hands_w.result or []
                variables.add_hands(hands, seen_hand_t, box)

            # face mesh from the camera
            data = client.get_latest_ai_data()
            if monitor.update(data):
                print("WARNING: the camera AI is not updating; try another --device index.")
            if data and data.get("timestamp_ms") != last_ts:
                last_ts = data.get("timestamp_ms")
                faces = data.get("faces") or []
                face = faces[0] if faces else None
                if face:
                    last_face_t = now
                    f = features(face.get("landmarks"))
                    if f is not None:
                        det.update(f, now)
            face_lost = now - last_face_t > FACE_LOST_S

            values = variables.compute(det, face_lost, now, seen_hand_t)
            phone_conf, box = variables.held_phone(now)

            # draw (before mirroring: every overlay is in the picture's own coordinates)
            if ir is None:
                draw_face(view, None if face_lost else face, show_mesh)
            draw_phone(view, box, phone_conf)
            if now - seen_hand_t <= HAND_STALE_S:
                draw_hands(view, hands)
            if not args.no_mirror:
                view = cv2.flip(view, 1)
            if det.calibrating:
                put_centred(view, f"LOOK AT THE SCREEN  {int(det.progress * 100)} %",
                            view.shape[0] // 2, 1.0, YELLOW, 2)
            status = ("measuring the screen pose" if det.calibrating
                      else "waiting for the face mesh" if det.base is None else "")
            panel = draw_panel(view.shape[0], values, variables.history, now, status)
            out = np.hstack([view, panel])
            put(out, "c: screen pose   m: mesh   q: quit", (16, out.shape[0] - 16), 0.5, WHITE, 1)
            cv2.imshow(WINDOW, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                det.calibrate("screen")
            elif key == ord("m"):
                show_mesh = not show_mesh
    except KeyboardInterrupt:
        pass
    finally:
        phones_w.stop()
        hands_w.stop()
        rgb.release()
        if ir is not None:
            ir.release()
        cv2.destroyAllWindows()
        client.disconnect()


if __name__ == "__main__":
    main()
