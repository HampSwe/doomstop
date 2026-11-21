"""Shows DOWN in big letters when you look down, using the LogiInsight face mesh.

Three numbers are read from the camera's 478 face landmarks:

  nose   how far the nose tip sits below the eye line (grows as the head tilts down)
  lids   how open the eyes are (shrinks when the eyes look down)
  iris   how far the irises sit below the eye corners (grows when the eyes look down)

They are compared with your "looking at the screen" pose, which is measured in
the first second after start, and turned into one score: head tilt + eye
rotation. The score is smoothed, so a blink does not trigger it.

    python look_down.py               # finds the LogiInsight colour camera
    python look_down.py --device 1    # or name the camera index

Keys in the window:
    c  re-measure the "looking at the screen" pose
    d  measure a "looking down" pose (e.g. at your phone) and set the threshold from it
    r  reset the threshold to the default
    m  show / hide all mesh points
    q  or Esc: quit
"""

import argparse
import math
import sys
import time

import cv2
import numpy as np

from face_stream import connect_device_ai, pick_camera  # also puts the SDK on sys.path
from camera_utils import CameraNotFoundError, describe, open_camera
from logiwebcam import AiStalenessMonitor

# MediaPipe face-mesh indexes (the camera's mesh uses the same layout).
FOREHEAD, CHIN, NOSE_TIP = 10, 152, 1
EYES = ((33, 133, 159, 145), (263, 362, 386, 374))  # outer corner, inner corner, upper lid, lower lid
IRIS_CENTRES = (468, 473)
KEY_POINTS = (FOREHEAD, CHIN, NOSE_TIP, *IRIS_CENTRES, *(i for eye in EYES for i in eye))

# How much each number changes for a clear look down; the score is in these units.
NOSE_SCALE = 0.08   # about 25 degrees of head tilt
LIDS_SCALE = 0.35   # eyes 35 % less open
IRIS_SCALE = 0.10   # iris a tenth of an eye width lower

DEFAULT_THRESHOLD = 1.0
RELEASE = 0.7            # DOWN switches off below this fraction of the threshold
SMOOTH_S = 0.25          # smoothing time constant
CALIB_SAMPLES = 20       # mesh results per calibration (under a second)
FACE_LOST_S = 1.0        # no face for this long counts as "no face"
AI_WARN_S = 8 * 60       # the camera can overheat after about 10 minutes of AI

WINDOW = "look_down (c / d: calibrate, q: quit)"
GREEN, RED, YELLOW, WHITE, GREY, CYAN = ((0, 200, 0), (0, 0, 255), (0, 255, 255),
                                         (255, 255, 255), (150, 150, 150), (255, 200, 0))


def features(landmarks):
    """(nose, lids, iris) from one mesh, or None if the mesh is unusable."""
    if not landmarks or len(landmarks) < 478:
        return None
    pts = np.asarray(landmarks, dtype=float)[:, :2]
    axis = pts[CHIN] - pts[FOREHEAD]           # points down the face, so head roll cancels out
    face_h = np.linalg.norm(axis)
    if face_h < 1:
        return None
    axis /= face_h
    eye_line = (pts[EYES[0][0]] + pts[EYES[1][0]]) / 2
    nose = np.dot(pts[NOSE_TIP] - eye_line, axis) / face_h

    lids, iris = [], []
    for outer, inner, upper, lower in EYES:
        width = np.linalg.norm(pts[outer] - pts[inner])
        if width < 1:
            return None
        centre = (pts[outer] + pts[inner]) / 2
        iris_pt = min((pts[i] for i in IRIS_CENTRES), key=lambda p: np.linalg.norm(p - centre))
        lids.append(np.linalg.norm(pts[upper] - pts[lower]) / width)
        iris.append(np.dot(iris_pt - centre, axis) / width)
    return np.array([nose, np.mean(lids), np.mean(iris)])


def score_parts(f, base):
    """(head, eyes): how far below the screen pose, each about 1.0 for a clear look down."""
    head = np.clip((f[0] - base[0]) / NOSE_SCALE, -1.0, 3.0)
    lids = np.clip((base[1] - f[1]) / (LIDS_SCALE * max(base[1], 1e-6)), -0.5, 1.5)  # capped: a blink is not a look down
    iris = np.clip((f[2] - base[2]) / IRIS_SCALE, -1.0, 1.5)
    return float(head), float((lids + iris) / 2)


class LookDownDetector:
    def __init__(self):
        self.base = None                 # features of the "looking at the screen" pose
        self.threshold = DEFAULT_THRESHOLD
        self.head = self.eyes = self.score = 0.0
        self.is_down = False
        self.message = ""
        self._collect = ("screen", [])   # calibrate on the screen pose at start
        self._last_t = None

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

    def reset_threshold(self):
        self.threshold = DEFAULT_THRESHOLD
        self.message = "threshold reset"

    def update(self, f, now):
        if self._collect:
            pose, samples = self._collect
            samples.append(f)
            if len(samples) < CALIB_SAMPLES:
                return
            median = np.median(samples, axis=0)
            self._collect = None
            if pose == "screen":
                self.base = median
                self.score, self.is_down, self._last_t = 0.0, False, None
                self.message = "screen pose saved"
            else:
                raw = sum(score_parts(median, self.base))
                if raw < 0.3:
                    self.message = "that looked like the screen pose; threshold unchanged"
                else:
                    self.threshold = 0.6 * raw
                    self.message = f"down pose saved, threshold {self.threshold:.2f}"
            return

        self.head, self.eyes = score_parts(f, self.base)
        raw = self.head + self.eyes
        dt = 0.0 if self._last_t is None else now - self._last_t
        self._last_t = now
        blend = 1.0 if dt <= 0 or dt > 1.0 else 1.0 - math.exp(-dt / SMOOTH_S)
        self.score += blend * (raw - self.score)
        limit = self.threshold * (RELEASE if self.is_down else 1.0)
        self.is_down = self.score >= limit


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


def draw_hud(frame, det, state, ai_seconds):
    h, w = frame.shape[:2]
    if det.calibrating:
        target = "LOOK AT THE SCREEN" if det.calibrating == "screen" else "LOOK DOWN (at your phone)"
        put_centred(frame, target, h // 2, 1.4, YELLOW, 3)
        put_centred(frame, f"measuring {int(det.progress * 100)} %", h // 2 + 45, 0.8, YELLOW, 2)
    elif state == "down":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 24)
        put_centred(frame, "DOWN", h // 2 + 60, 7.0, RED, 18, backing=False)
    elif state == "no face":
        put_centred(frame, "NO FACE", h // 2, 2.0, GREY, 4)

    label = {"down": "looking DOWN", "screen": "looking at the screen",
             "no face": "no face", "wait": "waiting for the face mesh"}[state]
    put(frame, label, (28, 48), 0.8, RED if state == "down" else GREEN)
    put(frame, f"head {det.head:+.2f}   eyes {det.eyes:+.2f}   score {det.score:.2f} / {det.threshold:.2f}",
        (28, 80))
    if det.message:
        put(frame, det.message, (28, 110), 0.6, YELLOW)
    if ai_seconds > AI_WARN_S:
        put(frame, f"camera AI on for {int(ai_seconds // 60)} min: quit soon, it can overheat",
            (28, 140), 0.6, YELLOW)
    put(frame, "c: screen pose   d: down pose   r: reset   m: mesh   q: quit", (28, h - 30), 0.55, WHITE, 1)

    # Meter on the right edge: fill = score, line = threshold.
    top, bottom, x0, x1 = 30, h - 30, w - 56, w - 32
    span = bottom - top
    full = 2 * det.threshold
    fill = int(span * min(max(det.score, 0.0) / full, 1.0))
    cv2.rectangle(frame, (x0, top), (x1, bottom), GREY, 1)
    cv2.rectangle(frame, (x0 + 1, bottom - fill), (x1 - 1, bottom - 1), RED if state == "down" else GREEN, -1)
    line_y = bottom - span // 2
    cv2.line(frame, (x0 - 6, line_y), (x1 + 6, line_y), WHITE, 2)


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
    monitor = AiStalenessMonitor()
    show_mesh = False
    face, last_ts, last_face_t = None, None, time.time()
    started = time.time()
    n_frames = n_mesh = n_down = 0
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
            n_down += state == "down"

            if args.no_display:
                continue
            draw_face(frame, None if face_lost else face, show_mesh)
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            draw_hud(frame, det, state, now - started)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                det.calibrate("screen")
            elif key == ord("d"):
                det.calibrate("down")
            elif key == ord("r"):
                det.reset_threshold()
            elif key == ord("m"):
                show_mesh = not show_mesh
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.disconnect()
    print(f"{n_frames} frames, {n_mesh} mesh results, DOWN in {n_down} frames")


if __name__ == "__main__":
    main()
