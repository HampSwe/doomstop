"""Interactive hands-on-desk detector for the LogiInsight webcam.

Why the IR stream: the near-infrared "CPD" camera delivers PORTRAIT frames
(512x640, 1024x1280), so with the camera on top of a monitor its tall vertical
field of view reaches down to the desk in front of the keyboard -- the RGB
stream (landscape) usually cuts it off. The camera's own NIR illuminator also
makes skin bright and evenly lit regardless of room light. The official
"Logi Hands Tracker" app uses this same stream (`LVNext CPD Camera`) with
MediaPipe Hand Landmarker; this script does the same locally so you get the
full 21-point skeleton, a configurable desk zone and an IR blob fallback.

Two detectors (cycle with 'd'):
  mediapipe  MediaPipe Hand Landmarker run on the desk zone crop (upscaled)
  blob       IR foreground blobs in the desk zone: difference to a captured
             empty-desk background ('b'), or bright-pixel Otsu if no background
  both       MediaPipe hands, plus blobs no MediaPipe hand explains

    python hand_desk.py                        # auto-detect the IR camera
    python hand_desk.py --size 1024x1280       # more pixels for small hands
    python hand_desk.py --source rgb           # compare with the RGB stream
    python hand_desk.py --device 2 --rotate 90
    python hand_desk.py --video desk.mp4       # replay a recording

Mouse: drag a rectangle in the window to set the desk zone.
Keys:  d detector   c CLAHE on/off   b capture empty-desk background
       x clear background   + / - blob threshold   o rotate 90 degrees
       r reset desk zone   s save snapshot   h help   q / Esc quit

Stop the Logi Hands Tracker app first: only one program can open the CPD
camera at a time.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # hide the per-index probe warnings

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "logiinsight_api"))

from camera_utils import (  # noqa: E402
    CameraNotFoundError,
    backend_id,
    describe,
    find_camera,
    install_stop_flag,
    open_camera,
    parse_device_arg,
)

MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
WINDOW = "hand_desk (h = help, q = quit)"
DETECTORS = ["both", "mediapipe", "blob"]
IR_SIZES = {"256x320": (256, 320), "512x640": (512, 640), "1024x1280": (1024, 1280)}
WARMUP_FRAMES = 20          # IR auto-exposure ramp after opening (see API.md section 3)
BG_FRAMES = 30              # frames averaged into the empty-desk background
MP_MIN_SIDE = 384           # upscale the desk crop so MediaPipe sees enough pixels
PALM_IDS = (0, 5, 9, 13, 17)
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
GREEN, YELLOW, CYAN, MAGENTA, RED, WHITE = ((0, 255, 0), (0, 255, 255), (255, 200, 0),
                                            (255, 0, 255), (0, 0, 255), (255, 255, 255))
ROTATIONS = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #

class MediaPipeHands:
    """MediaPipe Hand Landmarker (Tasks API) in VIDEO mode."""

    def __init__(self, max_hands, min_conf):
        from mediapipe.tasks.python import BaseOptions, vision

        if not MODEL_PATH.exists():
            print(f"Downloading the MediaPipe hand model to {MODEL_PATH} ...")
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._last_ts = -1

    def detect(self, gray, roi):
        """Hands inside `roi` (x0, y0, x1, y1 pixels) of a grey frame, full-frame coords."""
        import mediapipe as mp

        x0, y0, x1, y1 = roi
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            return []
        scale = max(1.0, MP_MIN_SIDE / min(crop.shape[:2]))
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB))
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)  # must increase strictly
        self._last_ts = ts
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        hands = []
        ch, cw = crop.shape[:2]
        for i, lms in enumerate(res.hand_landmarks):
            pts = np.array([(x0 + p.x * cw / scale, y0 + p.y * ch / scale) for p in lms],
                           dtype=np.float32)
            cat = res.handedness[i][0] if i < len(res.handedness) and res.handedness[i] else None
            hands.append({
                "source": "mediapipe",
                "points": pts,
                "center": pts[list(PALM_IDS)].mean(axis=0),
                "bbox": cv2.boundingRect(pts.astype(np.int32)),
                "label": cat.category_name if cat else "?",
                "score": float(cat.score) if cat else 0.0,
            })
        return hands

    def close(self):
        self._lm.close()


class IrBlobHands:
    """Foreground blobs in the desk zone of the IR image.

    With a background (an average of the empty desk) a pixel is foreground when
    it differs from it by more than `thresh`. Without one, NIR-lit skin is
    assumed to be brighter than the desk and Otsu picks the cut.
    """

    def __init__(self, thresh=25, min_area_frac=0.015):
        self.thresh = thresh
        self.min_area_frac = min_area_frac
        self.background = None
        self._acc = []
        self.mask = None

    @property
    def capturing(self):
        return bool(self._acc)

    def start_capture(self):
        self._acc = [None]  # sentinel: next frames are accumulated

    def feed(self, gray):
        if not self._acc:
            return
        if self._acc[0] is None:
            self._acc = []
        self._acc.append(gray.astype(np.float32))
        if len(self._acc) >= BG_FRAMES:
            self.background = np.mean(self._acc, axis=0)
            self._acc = []
            print("  background captured: blob detector now uses frame differencing")

    def detect(self, gray, roi):
        x0, y0, x1, y1 = roi
        crop = cv2.GaussianBlur(gray[y0:y1, x0:x1], (7, 7), 0)
        if crop.size == 0:
            return []
        if self.background is not None and self.background.shape == gray.shape:
            bg = cv2.GaussianBlur(self.background[y0:y1, x0:x1], (7, 7), 0)
            diff = cv2.absdiff(crop.astype(np.float32), bg)
            mask = (diff > self.thresh).astype(np.uint8) * 255
        else:
            _, mask = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        self.mask = mask

        min_area = self.min_area_frac * mask.size
        hands = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
            area = cv2.contourArea(c)
            # Reject blobs that fill the whole zone (lighting change, moved camera).
            if area < min_area or area > 0.8 * mask.size:
                continue
            c = c + np.array([[x0, y0]])
            m = cv2.moments(c)
            hands.append({
                "source": "blob",
                "contour": c,
                "center": np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], np.float32),
                "bbox": cv2.boundingRect(c),
                "label": "blob",
                "score": area / mask.size,
            })
        return hands


def merge(mp_hands, blob_hands):
    """MediaPipe hands, plus blobs that do not contain a MediaPipe palm."""
    extra = []
    for b in blob_hands:
        x, y, w, h = b["bbox"]
        if not any(x <= hd["center"][0] <= x + w and y <= hd["center"][1] <= y + h
                   for hd in mp_hands):
            extra.append(b)
    return mp_hands + extra


# --------------------------------------------------------------------------- #
# Interactive app
# --------------------------------------------------------------------------- #

class DeskZone:
    """Desk zone kept in normalised coords so it survives rotation / resize."""

    def __init__(self, norm):
        self.default = norm
        self.norm = norm
        self.drag_start = None
        self.drag_now = None

    def pixels(self, w, h):
        x0, y0, x1, y1 = self.norm
        return (int(x0 * w), int(y0 * h), max(int(x0 * w) + 2, int(x1 * w)),
                max(int(y0 * h) + 2, int(y1 * h)))

    def on_mouse(self, event, x, y, view_w, view_h):
        nx, ny = min(max(x / view_w, 0.0), 1.0), min(max(y / view_h, 0.0), 1.0)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = self.drag_now = (nx, ny)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start:
            self.drag_now = (nx, ny)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            (ax, ay), (bx, by) = self.drag_start, (nx, ny)
            if abs(bx - ax) > 0.03 and abs(by - ay) > 0.03:
                self.norm = (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
                print("  desk zone: --zone " + ",".join(f"{v:.2f}" for v in self.norm))
            self.drag_start = self.drag_now = None


class Debouncer:
    """Majority vote over the last N frames so the count does not flicker."""

    def __init__(self, n=7):
        self.hist = deque(maxlen=n)
        self.value = 0

    def update(self, count):
        self.hist.append(count)
        votes = np.bincount(list(self.hist))
        best = int(votes.argmax())
        if votes[best] > len(self.hist) // 2:
            self.value = best
        return self.value


def draw(view, hands, zone_px, zone, stable, state, fps):
    x0, y0, x1, y1 = zone_px
    active = stable > 0
    cv2.rectangle(view, (x0, y0), (x1, y1), GREEN if active else YELLOW, 2)
    cv2.putText(view, "desk zone", (x0 + 6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                GREEN if active else YELLOW, 1, cv2.LINE_AA)
    if zone.drag_start and zone.drag_now:
        h, w = view.shape[:2]
        a = (int(zone.drag_start[0] * w), int(zone.drag_start[1] * h))
        b = (int(zone.drag_now[0] * w), int(zone.drag_now[1] * h))
        cv2.rectangle(view, a, b, MAGENTA, 1)

    for hd in hands:
        if hd["source"] == "mediapipe":
            pts = hd["points"].astype(int)
            for a, b in HAND_EDGES:
                cv2.line(view, tuple(pts[a]), tuple(pts[b]), CYAN, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(view, tuple(p), 3, WHITE, -1, cv2.LINE_AA)
            text = f"{hd['label']} {hd['score']:.2f}"
        else:
            cv2.drawContours(view, [hd["contour"]], -1, MAGENTA, 2)
            text = "IR blob"
        x, y, _, _ = hd["bbox"]
        cv2.putText(view, text, (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    YELLOW, 1, cv2.LINE_AA)
        c = tuple(hd["center"].astype(int))
        cv2.circle(view, c, 6, GREEN, -1, cv2.LINE_AA)

    status = f"HANDS ON DESK: {stable}"
    cv2.rectangle(view, (0, 0), (view.shape[1], 54), (0, 0, 0), -1)
    cv2.putText(view, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                GREEN if active else WHITE, 2, cv2.LINE_AA)
    sub = (f"{state['detector']}  clahe={'on' if state['clahe'] else 'off'}  "
           f"bg={state['bg']}  t={state['thresh']}  {fps:.0f} fps")
    cv2.putText(view, sub, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)


HELP_LINES = [
    "drag mouse  set desk zone",
    "d  detector (both/mediapipe/blob)",
    "c  CLAHE contrast on/off",
    "b  capture empty-desk background",
    "x  clear background",
    "+/-  blob threshold",
    "o  rotate 90   r  reset zone",
    "s  save snapshot   q/Esc  quit",
]


def draw_help(view):
    h = 22 * len(HELP_LINES) + 14
    overlay = view.copy()
    cv2.rectangle(overlay, (10, 64), (330, 64 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, view, 0.25, 0, view)
    for i, line in enumerate(HELP_LINES):
        cv2.putText(view, line, (20, 86 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1,
                    cv2.LINE_AA)


def open_source(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise CameraNotFoundError(f"Cannot open video {args.video}")
        return cap, f"file {args.video}", 0
    if args.source == "ir":
        w, h, fourcc = *IR_SIZES[args.size], "MJPG"
    else:
        w, h, fourcc = 1280, 720, "MJPG"
    if args.device is not None:
        index, backend = parse_device_arg(args.device), None
    else:
        print(f"Looking for the {'near-infrared (CPD)' if args.source == 'ir' else 'RGB'} camera ...")
        index, backend_name = find_camera(args.source)
        backend = backend_id(backend_name)
    cap, info = open_camera(index, backend, w, h, fourcc, args.fps)
    return cap, describe(info), WARMUP_FRAMES if args.source == "ir" else 0


def run(args):
    stop = install_stop_flag()
    try:
        cap, desc, warmup = open_source(args)
    except CameraNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"Source: {desc}")

    detector = args.detector
    mp_hands = MediaPipeHands(args.max_hands, args.min_conf)  # kept so 'd' can switch to it
    blobs = IrBlobHands(args.blob_threshold)
    zone = DeskZone(tuple(args.zone))
    debounce = Debouncer()
    clahe_on = not args.no_clahe
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    rot = {0: 0, 90: 1, 180: 2, 270: 3}[args.rotate]
    show_help = False
    prev_gray = None
    frames = failures = saved = 0
    fps, t_prev = 0.0, time.time()
    last_reported = None
    view_size = [1, 1]

    if not args.no_display:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, lambda e, x, y, f, p: zone.on_mouse(e, x, y, *view_size))
        print("Drag a rectangle over the desk. Press h for help, q to quit.")

    try:
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                if args.video:
                    break
                failures += 1
                if failures > 90:
                    print("ERROR: the camera stopped delivering frames.", file=sys.stderr)
                    return 3
                time.sleep(0.03)
                continue
            failures = 0
            frames += 1
            if frames <= warmup:
                continue

            if ROTATIONS[rot] is not None:
                frame = cv2.rotate(frame, ROTATIONS[rot])
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if args.flicker_filter:
                src = gray if prev_gray is None or prev_gray.shape != gray.shape \
                    else np.maximum(gray, prev_gray)
                prev_gray = gray
                gray = src
            work = clahe.apply(gray) if clahe_on else gray

            blobs.feed(gray)
            h, w = gray.shape
            zpx = zone.pixels(w, h)
            found_mp = mp_hands.detect(work, zpx) if detector in ("both", "mediapipe") else []
            found_blob = (blobs.detect(gray, zpx)
                          if detector in ("both", "blob") and not blobs.capturing else [])
            hands = merge(found_mp, found_blob) if detector == "both" else found_mp + found_blob
            stable = debounce.update(min(len(hands), 9))

            if stable != last_reported:
                event = {"t": round(time.time(), 3), "hands_on_desk": stable,
                         "hands": [{"source": hd["source"], "label": hd["label"],
                                    "center": [round(float(v) / s, 3)
                                               for v, s in zip(hd["center"], (w, h))]}
                                   for hd in hands]}
                print(json.dumps(event) if args.json else
                      f"[{time.strftime('%H:%M:%S')}] hands on desk: {stable}"
                      + "".join(f"  {e['label']}@{e['center']}" for e in event["hands"]))
                last_reported = stable

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if fps else 1.0 / max(now - t_prev, 1e-6)
            t_prev = now

            if args.no_display:
                if args.frames and frames - warmup >= args.frames:
                    break
                continue

            view = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
            if blobs.mask is not None and detector != "mediapipe":
                x0, y0, x1, y1 = zpx
                tint = view[y0:y1, x0:x1]
                tint[blobs.mask > 0] = (0.6 * tint[blobs.mask > 0] + [100, 0, 100]).astype(np.uint8)
            state = {"detector": detector, "clahe": clahe_on, "thresh": blobs.thresh,
                     "bg": "capturing" if blobs.capturing else
                           ("set" if blobs.background is not None else "none")}
            draw(view, hands, zpx, zone, stable, state, fps)
            if view.shape[0] > args.max_height:
                s = args.max_height / view.shape[0]
                view = cv2.resize(view, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            if show_help:
                draw_help(view)
            view_size[:] = [view.shape[1], view.shape[0]]
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("d"):
                detector = DETECTORS[(DETECTORS.index(detector) + 1) % len(DETECTORS)]
                blobs.mask = None
                print(f"  detector: {detector}")
            elif key == ord("c"):
                clahe_on = not clahe_on
            elif key == ord("b"):
                print("  capturing background: keep your hands off the desk ...")
                blobs.start_capture()
            elif key == ord("x"):
                blobs.background = None
                print("  background cleared (blob detector back to bright-pixel Otsu)")
            elif key in (ord("+"), ord("=")):
                blobs.thresh = min(150, blobs.thresh + 5)
            elif key in (ord("-"), ord("_")):
                blobs.thresh = max(5, blobs.thresh - 5)
            elif key == ord("o"):
                rot = (rot + 1) % 4
                blobs.background = None  # shape changed
                print(f"  rotate: {rot * 90} (use --rotate {rot * 90})")
            elif key == ord("r"):
                zone.norm = zone.default
            elif key == ord("s"):
                out = ROOT / "captures"
                out.mkdir(exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(out / f"desk_{stamp}_{saved:03d}_raw.png"), gray)
                cv2.imwrite(str(out / f"desk_{stamp}_{saved:03d}_view.png"), view)
                saved += 1
                print(f"  saved to {out}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if mp_hands:
            mp_hands.close()
        cv2.destroyAllWindows()
        print(f"Released. {frames} frames.")
    return 0


def parse_zone(text):
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 4 or not (0 <= vals[0] < vals[2] <= 1 and 0 <= vals[1] < vals[3] <= 1):
        raise argparse.ArgumentTypeError("zone is x0,y0,x1,y1 with 0 <= x0 < x1 <= 1 (same for y)")
    return vals


def main():
    p = argparse.ArgumentParser(description="Detect hands on the desk with the LogiInsight IR stream")
    p.add_argument("--source", choices=["ir", "rgb"], default="ir",
                   help="ir = near-infrared CPD stream (default, tall FOV); rgb for comparison")
    p.add_argument("--device", default=None, help="camera index; default: auto-detect")
    p.add_argument("--video", default=None, help="read frames from a video file instead")
    p.add_argument("--size", choices=list(IR_SIZES), default="512x640", help="IR frame size")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0)
    p.add_argument("--zone", type=parse_zone, default=[0.0, 0.45, 1.0, 1.0],
                   help="desk zone as normalised x0,y0,x1,y1 (default: lower 55%% of the frame)")
    p.add_argument("--detector", choices=DETECTORS, default="both")
    p.add_argument("--max-hands", type=int, default=2)
    p.add_argument("--min-conf", type=float, default=0.4,
                   help="MediaPipe confidence; IR images often need lower than the 0.5 default")
    p.add_argument("--blob-threshold", type=int, default=25)
    p.add_argument("--no-clahe", action="store_true", help="start with CLAHE contrast off")
    p.add_argument("--flicker-filter", action="store_true",
                   help="per-pixel max of two frames, if the IR illuminator flickers")
    p.add_argument("--json", action="store_true", help="print state changes as JSON lines")
    p.add_argument("--max-height", type=int, default=820, help="max window height in pixels")
    p.add_argument("--no-display", action="store_true", help="headless; print changes only")
    p.add_argument("--frames", type=int, default=0, help="with --no-display: stop after N frames")
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
