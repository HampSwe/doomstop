"""Live view of the three doomscroll signals: FACE down, GAZE down and PHONE seen.

PHONE comes from an object detector running on this PC on the video picture
(see phone_detect.py). It runs in the background, since one pass takes about
90 ms, and counts as seen for a second after each sighting.

FACE and GAZE come from the LogiInsight camera's 478 face landmarks:

  pitch   head tilt from the landmarks' depth: chin further away than the forehead
  lid     how far the upper eyelid sits below the eye corners (it follows the eye down)

They are compared with your "looking at the screen" pose, which is measured in
the first second after start, and turned into two separate checks:

  FACE down   the head is tilted down                 (head: pitch)
  GAZE down   you are looking below the screen        (eyes: upper eyelid drop)

So eyes lowered with the head still gives GAZE only, and a bowed head with the
eyes still on the screen gives FACE only. Blinks are skipped and both scores
are smoothed.

The design comes from recordings of one person in four poses (captures/, keys
1 - 4):

  - The irises of this camera's mesh barely move vertically, so the eyelid is
    used instead.
  - Head tilt is not part of the gaze score. The camera sits by the screen, so
    the eyelid as the camera sees it already says whether you look towards the
    screen or below it, whatever the head does: with the head bowed 23 degrees
    and the eyes on the screen the lid hardly moved, and adding the tilt made
    GAZE fire wrongly.
  - The nose tip's position below the eye line was tried as a second head-tilt
    measure and dropped: lowered eyelids disturb it, and its noise follows the
    pitch's.

    python look_down.py               # finds the LogiInsight colour camera
    python look_down.py --device 1    # or name the camera index
    python look_down.py --no-phone    # FACE and GAZE only
    python look_down.py --ir          # plus a second window: the infrared stream with its own phone
                                      # detection and hands (PHONE OUT / PHONE IN HAND)

With --ir the colour stream is opened as YUY2 instead of MJPG (MJPG would slow
the infrared stream to about 1 frame per second), and the camera's face
results arrive at about half their usual rate.

Keys in the window:
    c  re-measure the "looking at the screen" pose
    d  measure a "looking down" pose (e.g. at your phone) and set both thresholds from it
    r  reset the thresholds to the defaults
    m  show / hide all mesh points
    q  or Esc: quit

    1 - 4  record the raw landmarks of a held pose into captures/, for tuning:
           1 looking at the screen, 2 eyes down with the head still,
           3 head and eyes down (at your phone), 4 head down with the eyes on the screen
"""

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from face_stream import connect_device_ai, pick_camera  # also puts the SDK on sys.path
from camera_utils import CameraNotFoundError, backend_id, describe, find_camera, open_camera
from logiwebcam import AiStalenessMonitor
from phone_detect import (IR_SIZES, IR_WARMUP_FRAMES, PHONE, PhoneDetector, Recent, Stream, banner,
                          enhance_grey, phone_score)
from phone_detect import label as put_label

# MediaPipe face-mesh indexes (the camera's mesh uses the same layout).
FOREHEAD, CHIN = 10, 152
EYES = (((33, 133), (160, 159, 158), (144, 145, 153)),     # corners, upper lid, lower lid
        ((263, 362), (387, 386, 385), (373, 374, 380)))
KEY_POINTS = (FOREHEAD, CHIN, *(i for eye in EYES for part in eye for i in part))

# Change of each measurement that counts as 1.0, taken from the recorded poses.
PITCH_SCALE = 11.5   # head: degrees of tilt, phone-in-lap pose (two recordings: 10 and 13)
LID_SCALE = 0.075    # eyes: upper lid drop in eye widths, eyes-down pose with the head still
BLINK_RATIO = 0.45   # eye opening below this fraction of normal is treated as a blink ...
BLINK_HOLD_S = 0.5   # ... for this long; after that it counts as eyes down

FACE_THRESHOLD = 0.6     # on `head`
GAZE_THRESHOLD = 0.65    # on `eyes`: midway between the recorded head-down-eyes-up and eyes-down poses
RELEASE = 0.7            # a check switches off below this fraction of its threshold
SMOOTH_S = 0.25          # smoothing time constant
CALIB_SAMPLES = 20       # mesh results per calibration (under a second)
FACE_LOST_S = 1.0        # no face for this long counts as "no face"
AI_WARN_S = 8 * 60       # the camera can overheat after about 10 minutes of AI
PHONE_HOLD_S = 1.0       # PHONE stays on this long after the last sighting (detection flickers)

RECORD_SAMPLES = 30      # mesh results per recorded pose (about a second)
RECORD_POSES = {ord("1"): ("screen", "LOOK AT THE SCREEN"),
                ord("2"): ("eyes_down", "EYES DOWN, HEAD STILL"),
                ord("3"): ("head_and_eyes_down", "LOOK DOWN AT YOUR PHONE"),
                ord("4"): ("head_down_eyes_up", "HEAD DOWN, EYES ON THE SCREEN")}
CAPTURE_DIR = Path(__file__).resolve().parent / "captures"

WINDOW = "look_down (c / d: calibrate, q: quit)"
IR_WINDOW = "look_down: infrared"
GREEN, RED, YELLOW, WHITE, GREY, CYAN = ((0, 200, 0), (0, 0, 255), (0, 255, 255),
                                         (255, 255, 255), (150, 150, 150), (255, 200, 0))


def features(landmarks):
    """(pitch, lid, opening) from one mesh, or None if the mesh is unusable."""
    if not landmarks or len(landmarks) < 478:
        return None
    full = np.asarray(landmarks, dtype=float)
    pts = full[:, :2]
    axis = pts[CHIN] - pts[FOREHEAD]           # points down the face, so head roll cancels out
    face_h = np.linalg.norm(axis)
    if face_h < 1:
        return None
    axis /= face_h
    pitch = math.degrees(math.atan2(full[CHIN, 2] - full[FOREHEAD, 2], face_h))

    lid, opening = [], []
    for (outer, inner), upper, lower in EYES:
        width = np.linalg.norm(pts[outer] - pts[inner])
        if width < 1:
            return None
        corners = (pts[outer] + pts[inner]) / 2
        up, low = pts[list(upper)].mean(axis=0), pts[list(lower)].mean(axis=0)
        lid.append(np.dot(up - corners, axis) / width)
        opening.append(np.dot(low - up, axis) / width)
    return np.array([pitch, np.mean(lid), np.mean(opening)])


def score_parts(f, base):
    """(head, eyes, blink): head and eyes are about 1.0 for a clear look down."""
    head = (f[0] - base[0]) / PITCH_SCALE
    eyes = (f[1] - base[1]) / LID_SCALE
    blink = f[2] < BLINK_RATIO * base[2]
    return float(np.clip(head, -1.5, 3.0)), float(np.clip(eyes, -1.0, 1.5)), bool(blink)


class LookDownDetector:
    def __init__(self):
        self.base = None                 # features of the "looking at the screen" pose
        self.face_threshold = FACE_THRESHOLD
        self.gaze_threshold = GAZE_THRESHOLD
        self.head = self.eyes = 0.0      # smoothed
        self.face_down = self.gaze_down = False
        self.message = ""
        self._collect = ("screen", [])   # calibrate on the screen pose at start
        self._last_t = None
        self._blink_since = None

    @property
    def is_down(self):
        return self.face_down or self.gaze_down

    @property
    def face_score(self):
        """Head tilt as 0..1: 0.5 at the FACE threshold, 1 at twice that tilt."""
        return float(np.clip(self.head / (2 * self.face_threshold), 0.0, 1.0))

    @property
    def gaze_score(self):
        """Eyelid drop as 0..1: 0.5 at the GAZE threshold, 1 at twice that drop."""
        return float(np.clip(self.eyes / (2 * self.gaze_threshold), 0.0, 1.0))

    @property
    def calibrating(self):
        return self._collect[0] if self._collect else None

    @property
    def progress(self):
        return len(self._collect[1]) / CALIB_SAMPLES if self._collect else 1.0

    def calibrate(self, pose):
        if pose == "down" and self.base is None:
            return
        self._collect = (pose, [])
        self.message = ""

    def reset_thresholds(self):
        self.face_threshold, self.gaze_threshold = FACE_THRESHOLD, GAZE_THRESHOLD
        self.message = "thresholds reset"

    def _finish_calibration(self, pose, median):
        if pose == "screen":
            self.base = median
            self.head = self.eyes = 0.0
            self.face_down = self.gaze_down = False
            self._last_t = None
            self.message = "screen pose saved"
            return
        head, eyes, _ = score_parts(median, self.base)
        set_ = []                        # a part that did not move says nothing about its threshold
        if eyes >= 0.3:
            self.gaze_threshold = 0.6 * eyes
            set_.append(f"gaze threshold {self.gaze_threshold:.2f}")
        if head >= 0.3:
            self.face_threshold = 0.6 * head
            set_.append(f"face threshold {self.face_threshold:.2f}")
        self.message = ("down pose saved: " + ", ".join(set_) if set_
                        else "that looked like the screen pose; thresholds unchanged")

    def update(self, f, now):
        if self._collect:
            pose, samples = self._collect
            samples.append(f)
            if len(samples) >= CALIB_SAMPLES:
                self._collect = None
                self._finish_calibration(pose, np.median(samples, axis=0))
            return

        head, eyes, blink = score_parts(f, self.base)
        dt = 0.0 if self._last_t is None else now - self._last_t
        self._last_t = now
        blend = 1.0 if dt <= 0 or dt > 1.0 else 1.0 - math.exp(-dt / SMOOTH_S)
        self.head += blend * (head - self.head)
        if not blink:
            self._blink_since = None
        elif self._blink_since is None:
            self._blink_since = now
        if not blink or now - self._blink_since >= BLINK_HOLD_S:   # a short blink leaves `eyes` alone
            self.eyes += blend * (eyes - self.eyes)
        self.face_down = self.head >= self.face_threshold * (RELEASE if self.face_down else 1.0)
        self.gaze_down = self.eyes >= self.gaze_threshold * (RELEASE if self.gaze_down else 1.0)


class PoseRecorder:
    """Saves the raw landmarks of a held pose, so the detector can be tuned on real data."""

    def __init__(self, directory=CAPTURE_DIR):
        self.directory = Path(directory)
        self.label = self.prompt = None
        self.samples = []

    @property
    def progress(self):
        return len(self.samples) / RECORD_SAMPLES

    def start(self, label, prompt):
        self.label, self.prompt, self.samples = label, prompt, []

    def add(self, landmarks, frame_size):
        """Returns a message once the pose is saved, else None."""
        if self.label is None or not landmarks or len(landmarks) < 478:
            return None
        self.samples.append([[round(float(v), 1) for v in point[:3]] for point in landmarks])
        if len(self.samples) < RECORD_SAMPLES:
            return None
        self.directory.mkdir(exist_ok=True)
        path = self.directory / f"pose_{self.label}.json"
        path.write_text(json.dumps({"label": self.label, "frame_size": list(frame_size),
                                    "samples": self.samples}), encoding="utf-8")
        self.label = self.prompt = None
        return f"saved {path.relative_to(self.directory.parent)}"


def put(frame, text, org, scale=0.6, colour=WHITE, thick=2, backing=True):
    if backing:  # dark box behind the text so it stays readable on any picture
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(frame, (org[0] - 4, org[1] - th - 5), (org[0] + tw + 4, org[1] + base + 3),
                      (25, 25, 25), -1)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def put_centred(frame, text, y, scale, colour, thick, backing=True):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    put(frame, text, ((frame.shape[1] - tw) // 2, y), scale, colour, thick, backing)


def draw_face(frame, face, show_mesh):
    """Drawn before mirroring, so it uses the camera's own coordinates."""
    if not face:
        return
    x, y, w, h = (int(round(v)) for v in face["bbox"])
    cv2.rectangle(frame, (x, y), (x + w, y + h), GREY, 1)
    landmarks = face.get("landmarks") or []
    if len(landmarks) < 478:
        return
    if show_mesh:
        for px, py, *_ in landmarks:
            cv2.circle(frame, (int(px), int(py)), 1, GREY, -1)
    for i in KEY_POINTS:
        cv2.circle(frame, (int(landmarks[i][0]), int(landmarks[i][1])), 3, CYAN, -1)


class PhoneWorker(threading.Thread):
    """Runs the phone detector in the background, on the newest frame of each stream in turn."""

    def __init__(self, detector, names=("rgb",), ir_hands=None):
        """`ir_hands`: a hand_detect.HandDetector to also track hands on the "ir" stream."""
        super().__init__(daemon=True, name="phone")
        self.detector = detector
        self.ir_hands = ir_hands
        self.hand_points = []                         # hands in the latest infrared frame
        self.use = None                               # hand_detect.PhoneUse for the infrared stream
        if ir_hands:
            from hand_detect import PhoneUse
            self.use = PhoneUse()
        self.phones = {n: [] for n in names}          # latest result per stream: [(x, y, w, h, score)]
        self.scores = {n: Recent(PHONE_HOLD_S) for n in names}   # phone confidence per stream
        self._frames = {n: None for n in names}
        self._wake = threading.Event()
        self._quit = False
        self._warned = False

    def submit(self, frame, name="rgb"):
        """Hand over a frame the caller will not draw on; an unprocessed older one is dropped."""
        self._frames[name] = frame
        self._wake.set()

    def score(self, now, name="rgb"):
        """0..1: the detector's best phone confidence in that stream over the last PHONE_HOLD_S."""
        return self.scores[name].max(now)

    def is_on(self, now, name="rgb"):
        return self.score(now, name) >= self.detector.min_score

    def run(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._quit:
                return
            for name in self._frames:
                frame, self._frames[name] = self._frames[name], None
                if frame is None:
                    continue
                try:
                    candidates = self.detector.candidates(frame)
                    self.scores[name].add(time.time(), phone_score(candidates))
                    found = [d[:5] for d in candidates if d[5] == PHONE and d[4] >= self.detector.min_score]
                except Exception as e:  # keep the face signals running if the detector breaks
                    found = []
                    if not self._warned:
                        self._warned = True
                        print(f"WARNING: phone detection failed and is off: {e}")
                self.phones[name] = found
                if name == "ir" and self.ir_hands:
                    try:
                        self.hand_points = self.ir_hands.detect(
                            frame, time.time(), found[0][:4] if found else self.use.box)
                    except Exception as e:
                        self.hand_points = []
                        if not self._warned:
                            self._warned = True
                            print(f"WARNING: hand detection failed: {e}")
                    self.use.update(found, self.hand_points, time.time())

    def stop(self):
        self._quit = True
        self._wake.set()
        self.join(timeout=2)                          # let a detection in progress finish
        if self.ir_hands:
            self.ir_hands.close()


def signals(det, worker, now):
    """Every signal as one dict. Scores are 0..1; None means that signal is switched off.

      face_down, gaze_down          0 or 1
      face_down_score               head tilt: 0.5 at the FACE threshold
      gaze_down_score               eyelid drop: 0.5 at the GAZE threshold
      color_phone                   phone confidence in the colour stream
      phone                         phone confidence in the infrared stream
      phone_in_hand                 how steadily a hand has been at that phone
      gaze_down_x_phone             gaze_down * phone: the infrared phone confidence while looking down, else 0
    """
    ir = worker is not None and "ir" in worker.phones
    phone = worker.score(now, "ir") if ir else None
    return {
        "face_down": int(det.face_down),
        "face_down_score": det.face_score,
        "gaze_down": int(det.gaze_down),
        "gaze_down_score": det.gaze_score,
        "color_phone": worker.score(now, "rgb") if worker else None,
        "phone": phone,
        "phone_in_hand": worker.use.in_hand if ir and worker.use else None,
        "gaze_down_x_phone": int(det.gaze_down) * phone if ir else None,
    }


def draw_signals(frame, values, x, y):
    """One row per signal: name, value, and a bar that turns red from 0.5."""
    for i, (name, value) in enumerate(values.items()):
        row = y + 27 * i
        put(frame, name, (x, row), 0.5, WHITE, 1)
        text = "-" if value is None else str(value) if isinstance(value, int) else f"{value:.2f}"
        put(frame, text, (x + 170, row), 0.5, WHITE, 1)
        x0, x1 = x + 225, x + 365
        cv2.rectangle(frame, (x0, row - 13), (x1, row + 3), (25, 25, 25), -1)
        if value is not None:
            cv2.rectangle(frame, (x0, row - 13), (x0 + int((x1 - x0) * value), row + 3),
                          RED if value >= 0.5 else GREEN, -1)
        cv2.rectangle(frame, (x0, row - 13), (x1, row + 3), GREY, 1)


def ir_view(frame, phones, score, needed, mirror, hands=(), use=None):
    """The infrared picture as the detector sees it (contrast-stretched), with its phone and hand result."""
    view = enhance_grey(frame)
    for x, y, w, h, _ in phones:
        cv2.rectangle(view, (int(x), int(y)), (int(x + w), int(y + h)), RED, 3)
    if use is not None:
        from hand_detect import IN_HAND_ON, draw_hands
        draw_hands(view, hands)
    if mirror:
        view = cv2.flip(view, 1)
    if view.shape[0] < 700:              # the infrared frame is small on screen
        k = 800 / view.shape[0]
        view = cv2.resize(view, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
    banner(view, use, score >= needed)
    status = f"infrared: phone {score:.2f} / {needed:.2f}"
    if use is not None:
        status += f"   in hand {use.in_hand:.2f} / {IN_HAND_ON:.2f}"
    put_label(view, status, (28, 48), 0.65, WHITE)
    return view


def draw_phones(frame, phones):
    """Drawn before mirroring, like the face."""
    for x, y, w, h, score in phones:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), RED, 3)


def draw_tile(frame, x0, x1, y0, y1, text, active, scale):
    """One signal lamp: filled red when on, a grey outline when off."""
    cv2.rectangle(frame, (x0, y0), (x1, y1), RED if active else GREY, -1 if active else 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 4)
    if tw > 0.9 * (x1 - x0):                          # a long label: shrink it to fit the lamp
        scale *= 0.9 * (x1 - x0) / tw
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 4)
    cv2.putText(frame, text, ((x0 + x1 - tw) // 2, (y0 + y1 + th) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, WHITE if active else GREY, 4, cv2.LINE_AA)


def draw_meter(frame, x, label, value, threshold, active, bottom):
    """Vertical bar: fill = value, white line = threshold (at half height)."""
    top, width = 30, 24
    span = bottom - top
    fill = int(span * min(max(value, 0.0) / (2 * threshold), 1.0))
    cv2.rectangle(frame, (x, top), (x + width, bottom), GREY, 1)
    cv2.rectangle(frame, (x + 1, bottom - fill), (x + width - 1, bottom - 1), RED if active else GREEN, -1)
    line_y = bottom - span // 2
    cv2.line(frame, (x - 6, line_y), (x + width + 6, line_y), WHITE, 2)
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    put(frame, label, (x + (width - tw) // 2, bottom + 24), 0.5, RED if active else WHITE, 1)


def draw_hud(frame, det, state, ai_seconds, recorder=None, phone=None, values=None):
    """`phone` is None when the phone detector is off, else (on, score, needed score).
    `values` is the signals() dict, drawn as a panel."""
    h, w = frame.shape[:2]
    big = w / 1280                       # the large texts scale with the picture
    tiles = [("FACE DOWN", det.face_down), ("GAZE DOWN", det.gaze_down)]
    if phone is not None:
        tiles.append(("COLOR PHONE", phone[0]))       # the phone seen in this, the colour, stream
    any_on = any(on for _, on in tiles)
    if any_on:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 24)
    tile_bottom = h - 62
    tile_top = tile_bottom - int(110 * big)
    gap = 16
    tile_w = (w - 56 - gap * (len(tiles) - 1)) // len(tiles)
    for i, (text, on) in enumerate(tiles):
        x0 = 28 + i * (tile_w + gap)
        draw_tile(frame, x0, x0 + tile_w, tile_top, tile_bottom, text, on, 1.9 * big)

    if recorder is not None and recorder.label:
        put_centred(frame, recorder.prompt, h // 2, 1.4 * big, CYAN, 3)
        put_centred(frame, f"recording {int(recorder.progress * 100)} %: hold the pose",
                    h // 2 + int(45 * big), 0.8 * big, CYAN, 2)
    elif det.calibrating:
        target = "LOOK AT THE SCREEN" if det.calibrating == "screen" else "LOOK DOWN (at your phone)"
        put_centred(frame, target, h // 2, 1.4 * big, YELLOW, 3)
        put_centred(frame, f"measuring {int(det.progress * 100)} %", h // 2 + int(45 * big), 0.8 * big, YELLOW, 2)
    elif state == "no face":
        put_centred(frame, "NO FACE", h // 2, 2.0 * big, GREY, 4)

    if state in ("down", "screen"):
        label = (f"face: {'DOWN' if det.face_down else 'up'}   "
                 f"gaze: {'DOWN' if det.gaze_down else 'on the screen'}")
    else:
        label = {"no face": "no face", "wait": "waiting for the face mesh"}[state]
    numbers = (f"head {det.head:+.2f} / {det.face_threshold:.2f}   "
               f"eyes {det.eyes:+.2f} / {det.gaze_threshold:.2f}")
    if phone is not None:
        label += f"   color phone: {'SEEN' if phone[0] else 'no'}"
        numbers += f"   color_phone {phone[1]:.2f} / {phone[2]:.2f}"
    put(frame, label, (28, 48), 0.8, RED if any_on else GREEN)
    put(frame, numbers, (28, 80))
    if det.message:
        put(frame, det.message, (28, 110), 0.6, YELLOW)
    if ai_seconds > AI_WARN_S:
        put(frame, f"camera AI on for {int(ai_seconds // 60)} min: quit soon, it can overheat",
            (28, 140), 0.6, YELLOW)
    put(frame, "c: screen pose   d: down pose   r: reset   m: mesh   1-4: record poses   q: quit", (28, h - 30), 0.55, WHITE, 1)

    if values:
        draw_signals(frame, values, 28, 182)
    draw_meter(frame, w - 120, "FACE", det.head, det.face_threshold, det.face_down, tile_top - 50)
    draw_meter(frame, w - 60, "GAZE", det.eyes, det.gaze_threshold, det.gaze_down, tile_top - 50)


class FaceFeed:
    """Reads the camera's face results and keeps a LookDownDetector up to date."""

    def __init__(self, client, camera_index):
        self.client, self.camera_index = client, camera_index
        self.det = LookDownDetector()
        self.face = None                 # the latest face the camera reported, or None
        self.n_mesh = 0
        self._monitor = AiStalenessMonitor()
        self._last_ts = None
        self._last_face_t = time.time()

    def poll(self, now):
        """Fetch the latest result. Returns its landmarks if it is a new result with a face, else None."""
        data = self.client.get_latest_ai_data()
        if self._monitor.update(data):
            print(f"WARNING: the camera AI is not updating, so camera index {self.camera_index} is probably "
                  "not the LogiInsight colour camera. Quit and try another --device index.")
        if not data or data.get("timestamp_ms") == self._last_ts:
            return None
        self._last_ts = data.get("timestamp_ms")
        faces = data.get("faces") or []
        self.face = faces[0] if faces else None
        if not self.face:
            return None
        self._last_face_t = now
        landmarks = self.face.get("landmarks")
        f = features(landmarks)
        if f is not None:                # the mesh is missing from some results
            self.n_mesh += 1
            self.det.update(f, now)
        return landmarks

    def face_lost(self, now):
        return now - self._last_face_t > FACE_LOST_S


def open_cameras(client, device=None, with_ir=False, ir_device=None, width=1280, height=720):
    """Open the colour stream and, if asked, the infrared one. Returns (capture, index, infrared Stream or None).

    Raises CameraNotFoundError with nothing left open.
    """
    index = device if device is not None else pick_camera(client)
    ir_index, ir_backend = ir_device, None
    if with_ir and ir_index is None:                  # look before the colour stream is open
        print("Looking for the near-infrared (CPD) camera ...")
        ir_index, ir_backend_name = find_camera("ir")
        ir_backend = backend_id(ir_backend_name)
    # With infrared open too, MJPG on the colour side would slow infrared to about 1 fps.
    cap, info = open_camera(index, width=width, height=height, fourcc="YUY2" if with_ir else "MJPG", fps=30)
    print(f"Opened {describe(info)}")
    if not with_ir:
        return cap, index, None
    try:
        for _ in range(5):                            # colour first, as the API reference advises
            cap.read()
        ir_cap, ir_info = open_camera(ir_index, ir_backend, *IR_SIZES["512x640"], "MJPG", 30)
    except CameraNotFoundError:
        cap.release()
        raise
    ir = Stream("ir", ir_cap, describe(ir_info), IR_WARMUP_FRAMES)
    print(f"Opened infrared: {ir.description}")
    return cap, index, ir


def start_phone_worker(phone_score, with_ir):
    """A running PhoneWorker for the colour stream, plus the infrared one (with hands) if asked."""
    ir_hands = None
    if with_ir:
        try:
            from hand_detect import HandDetector
            ir_hands = HandDetector()
        except ImportError:
            print("NOTE: no hand detection, because the mediapipe package is missing "
                  "(pip install -r requirements.txt).")
    worker = PhoneWorker(PhoneDetector(min_score=phone_score), ("rgb", "ir") if with_ir else ("rgb",), ir_hands)
    worker.start()
    return worker


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", type=int, help="camera index (default: find the LogiInsight camera)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the picture")
    p.add_argument("--no-phone", action="store_true", help="skip the phone detector (FACE and GAZE only)")
    p.add_argument("--phone-score", type=float, default=0.35,
                   help="confidence the phone detector needs (default 0.35; lower finds more, with more false alarms)")
    p.add_argument("--ir", action="store_true",
                   help="also open the infrared stream in a second window, with phone detection on it")
    p.add_argument("--ir-device", type=int, help="infrared camera index (default: auto-detect)")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until quit)")
    p.add_argument("--no-display", action="store_true", help="no window; print a summary")
    args = p.parse_args()
    if args.ir and args.no_phone:
        sys.exit("ERROR: --ir adds phone detection on the infrared stream, so it cannot go with --no-phone.")

    client, reason = connect_device_ai(mesh=True)
    if client is None:
        sys.exit(f"ERROR: this needs the camera's face mesh, which is unavailable: {reason}\n"
                 "Run logiinsight_api\\check_connection.py to see what is wrong.")
    try:
        cap, index, ir = open_cameras(client, args.device, args.ir, args.ir_device, args.width, args.height)
    except CameraNotFoundError as e:
        client.disconnect()
        sys.exit(f"ERROR: {e}")
    print("Look at the screen for a second while it measures your normal pose.")

    feed = FaceFeed(client, index)
    det = feed.det
    recorder = PoseRecorder()
    phone_worker = None if args.no_phone else start_phone_worker(args.phone_score, ir is not None)
    show_mesh = False
    started = time.time()
    n_frames = n_face_down = n_gaze_down = n_phone = n_ir = n_phone_ir = n_in_hand = 0
    try:
        while not args.frames or n_frames < args.frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                if not args.no_display and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            now = time.time()
            n_frames += 1
            if phone_worker:
                phone_worker.submit(frame.copy())     # a copy: this frame gets drawn on below
            ir_frame = ir.take() if ir else None
            if ir_frame is not None:
                phone_worker.submit(ir_frame, "ir")   # never drawn on: its window shows a copy
                n_ir += 1
                n_phone_ir += phone_worker.is_on(now, "ir")
                n_in_hand += bool(phone_worker.use and phone_worker.use.state == "phone in hand")

            landmarks = feed.poll(now)
            if landmarks:
                saved = recorder.add(landmarks, frame.shape[1::-1])
                if saved:
                    det.message = saved
                    print(saved)

            face_lost = feed.face_lost(now)
            if det.base is None:
                state = "wait"
            elif det.is_down:
                state = "down"       # kept while the face is lost: a bowed head often loses it
            elif face_lost:
                state = "no face"
            else:
                state = "screen"
            n_face_down += det.face_down
            n_gaze_down += det.gaze_down
            phone = None
            if phone_worker:
                phones = phone_worker.phones["rgb"]
                phone = (phone_worker.is_on(now), phone_worker.score(now), args.phone_score)
                n_phone += phone[0]
            values = signals(det, phone_worker, now)

            if args.no_display:
                continue
            draw_face(frame, None if face_lost else feed.face, show_mesh)
            if phone_worker:
                draw_phones(frame, phones)
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            draw_hud(frame, det, state, now - started, recorder, phone, values)
            cv2.imshow(WINDOW, frame)
            if ir_frame is not None:
                cv2.imshow(IR_WINDOW, ir_view(ir_frame, phone_worker.phones["ir"], phone_worker.score(now, "ir"),
                                              args.phone_score, not args.no_mirror,
                                              phone_worker.hand_points, phone_worker.use))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                det.calibrate("screen")
            elif key == ord("d"):
                det.calibrate("down")
            elif key == ord("r"):
                det.reset_thresholds()
            elif key == ord("m"):
                show_mesh = not show_mesh
            elif key in RECORD_POSES:
                recorder.start(*RECORD_POSES[key])
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.disconnect()
        if phone_worker:
            phone_worker.stop()
        if ir:
            ir.close()
    print(f"{n_frames} frames, {feed.n_mesh} mesh results, face down in {n_face_down} frames, "
          f"gaze down in {n_gaze_down} frames"
          + (f", phone seen in {n_phone} frames" if phone_worker else "")
          + (f"; infrared: {n_ir} frames, phone seen in {n_phone_ir}, in hand in {n_in_hand}" if ir else ""))


if __name__ == "__main__":
    main()
