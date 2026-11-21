"""Shows when your face and your gaze point down, using the LogiInsight face mesh.

These are read from the camera's 478 face landmarks:

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
import time
from pathlib import Path

import cv2
import numpy as np

from face_stream import connect_device_ai, pick_camera  # also puts the SDK on sys.path
from camera_utils import CameraNotFoundError, describe, open_camera
from logiwebcam import AiStalenessMonitor

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

RECORD_SAMPLES = 30      # mesh results per recorded pose (about a second)
RECORD_POSES = {ord("1"): ("screen", "LOOK AT THE SCREEN"),
                ord("2"): ("eyes_down", "EYES DOWN, HEAD STILL"),
                ord("3"): ("head_and_eyes_down", "LOOK DOWN AT YOUR PHONE"),
                ord("4"): ("head_down_eyes_up", "HEAD DOWN, EYES ON THE SCREEN")}
CAPTURE_DIR = Path(__file__).resolve().parent / "captures"

WINDOW = "look_down (c / d: calibrate, q: quit)"
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


def draw_meter(frame, x, label, value, threshold, active):
    """Vertical bar: fill = value, white line = threshold (at half height)."""
    h = frame.shape[0]
    top, bottom, width = 30, h - 70, 24
    span = bottom - top
    fill = int(span * min(max(value, 0.0) / (2 * threshold), 1.0))
    cv2.rectangle(frame, (x, top), (x + width, bottom), GREY, 1)
    cv2.rectangle(frame, (x + 1, bottom - fill), (x + width - 1, bottom - 1), RED if active else GREEN, -1)
    line_y = bottom - span // 2
    cv2.line(frame, (x - 6, line_y), (x + width + 6, line_y), WHITE, 2)
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    put(frame, label, (x + (width - tw) // 2, bottom + 24), 0.5, RED if active else WHITE, 1)


def draw_hud(frame, det, state, ai_seconds, recorder=None):
    h, w = frame.shape[:2]
    big = w / 1280                       # the large texts scale with the picture
    if recorder is not None and recorder.label:
        put_centred(frame, recorder.prompt, h // 2, 1.4 * big, CYAN, 3)
        put_centred(frame, f"recording {int(recorder.progress * 100)} %: hold the pose",
                    h // 2 + int(45 * big), 0.8 * big, CYAN, 2)
    elif det.calibrating:
        target = "LOOK AT THE SCREEN" if det.calibrating == "screen" else "LOOK DOWN (at your phone)"
        put_centred(frame, target, h // 2, 1.4 * big, YELLOW, 3)
        put_centred(frame, f"measuring {int(det.progress * 100)} %", h // 2 + int(45 * big), 0.8 * big, YELLOW, 2)
    elif state == "down":
        what = " + ".join(n for n, on in (("FACE", det.face_down), ("GAZE", det.gaze_down)) if on)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 24)
        put_centred(frame, what, h // 2 - int(100 * big), 3.0 * big, RED, 8, backing=False)
        put_centred(frame, "DOWN", h // 2 + int(110 * big), 7.0 * big, RED, 18, backing=False)
    elif state == "no face":
        put_centred(frame, "NO FACE", h // 2, 2.0 * big, GREY, 4)

    if state in ("down", "screen"):
        label = (f"face: {'DOWN' if det.face_down else 'up'}   "
                 f"gaze: {'DOWN' if det.gaze_down else 'on the screen'}")
    else:
        label = {"no face": "no face", "wait": "waiting for the face mesh"}[state]
    put(frame, label, (28, 48), 0.8, RED if state == "down" else GREEN)
    put(frame, f"head {det.head:+.2f} / {det.face_threshold:.2f}   "
               f"eyes {det.eyes:+.2f} / {det.gaze_threshold:.2f}", (28, 80))
    if det.message:
        put(frame, det.message, (28, 110), 0.6, YELLOW)
    if ai_seconds > AI_WARN_S:
        put(frame, f"camera AI on for {int(ai_seconds // 60)} min: quit soon, it can overheat",
            (28, 140), 0.6, YELLOW)
    put(frame, "c: screen pose   d: down pose   r: reset   m: mesh   1-4: record poses   q: quit", (28, h - 30), 0.55, WHITE, 1)

    draw_meter(frame, w - 120, "FACE", det.head, det.face_threshold, det.face_down)
    draw_meter(frame, w - 60, "GAZE", det.eyes, det.gaze_threshold, det.gaze_down)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", type=int, help="camera index (default: find the LogiInsight camera)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the picture")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until quit)")
    p.add_argument("--no-display", action="store_true", help="no window; print a summary")
    args = p.parse_args()

    client, reason = connect_device_ai(mesh=True)
    if client is None:
        sys.exit(f"ERROR: this needs the camera's face mesh, which is unavailable: {reason}\n"
                 "Run logiinsight_api\\check_connection.py to see what is wrong.")
    try:
        index = args.device if args.device is not None else pick_camera(client)
        cap, info = open_camera(index, width=args.width, height=args.height, fourcc="MJPG", fps=30)
    except CameraNotFoundError as e:
        client.disconnect()
        sys.exit(f"ERROR: {e}")
    print(f"Opened {describe(info)}")
    print("Look at the screen for a second while it measures your normal pose.")

    det = LookDownDetector()
    recorder = PoseRecorder()
    monitor = AiStalenessMonitor()
    show_mesh = False
    face, last_ts, last_face_t = None, None, time.time()
    started = time.time()
    n_frames = n_mesh = n_face_down = n_gaze_down = 0
    try:
        while not args.frames or n_frames < args.frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                if not args.no_display and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            now = time.time()
            n_frames += 1

            data = client.get_latest_ai_data()
            if monitor.update(data):
                print(f"WARNING: the camera AI is not updating, so camera index {index} is probably "
                      "not the LogiInsight colour camera. Quit and try another --device index.")
            if data and data.get("timestamp_ms") != last_ts:   # a new AI result
                last_ts = data.get("timestamp_ms")
                faces = data.get("faces") or []
                face = faces[0] if faces else None
                if face:
                    last_face_t = now
                    saved = recorder.add(face.get("landmarks"), frame.shape[1::-1])
                    if saved:
                        det.message = saved
                        print(saved)
                    f = features(face.get("landmarks"))
                    if f is not None:                           # the mesh is missing from some results
                        n_mesh += 1
                        det.update(f, now)

            face_lost = now - last_face_t > FACE_LOST_S
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

            if args.no_display:
                continue
            draw_face(frame, None if face_lost else face, show_mesh)
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            draw_hud(frame, det, state, now - started, recorder)
            cv2.imshow(WINDOW, frame)
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
    print(f"{n_frames} frames, {n_mesh} mesh results, "
          f"face down in {n_face_down} frames, gaze down in {n_gaze_down} frames")


if __name__ == "__main__":
    main()
