"""Detects a phone in the picture with a ready-made object detector.

The model is YOLOX-S from the OpenCV model zoo, trained on the 80 COCO object
classes, one of which is "cell phone". It runs on this PC through OpenCV, on
the plain video frame, so it works with any webcam and needs no camera login.
The model file (34 MB) is downloaded into models/ on first use.

    python phone_detect.py                 # pick a camera, show PHONE when one is seen
    python phone_detect.py --device 1      # skip the camera prompt
    python phone_detect.py --source ir     # the LogiInsight near-infrared stream instead

    python phone_detect.py --source both   # colour and infrared together, one window each

With both streams the colour one is opened as YUY2, not MJPG: MJPG on the
colour side slows the infrared stream to about 1 frame per second.

    python phone_detect.py --all           # also draw every other object it recognises
    python phone_detect.py --image a.jpg   # run on a picture file instead of a camera

Infrared: the picture is grey and often low in contrast. Grey input by itself
does not bother the model, but flat contrast does (on grey test photos the
score fell from about 0.83 to 0.53 and came back with a contrast stretch), so
grey frames are stretched before detection (--ir-enhance). How well the model
copes with real infrared pictures has to be measured on recordings:

    p / n in the window    save the next frames as "phone" / "no phone" examples
                           (with --source both: the same moments from both streams)
    python phone_detect.py --eval captures/phone_ir    # score the saved examples

From other code:

    from phone_detect import phone_in_frame
    phone_in_frame(frame)                  # True / False for one BGR frame, colour or infrared

Press q or Esc in the window to quit.
"""

import argparse
import math
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # hide the per-index camera probe warnings

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "object_detection_yolox_2022nov.onnx"
MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "object_detection_yolox/object_detection_yolox_2022nov.onnx")

COCO = (  # the 80 classes, in the model's output order
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush")
PHONE = COCO.index("cell phone")

IR_SIZES = {"256x320": (256, 320), "512x640": (512, 640), "1024x1280": (1024, 1280)}
IR_WARMUP_FRAMES = 20        # the infrared stream's auto-exposure settles (API reference, section 3)
ENHANCE_MODES = ("stretch", "clahe", "none")
RECORD_FRAMES = 20           # frames saved per press of p / n
RECORD_KEYS = {ord("p"): "phone", ord("n"): "nophone"}
CAPTURE_DIR = ROOT / "captures"

WINDOW = "phone_detect (q / Esc to quit)"
RED, GREEN, GREY, WHITE, CYAN = (0, 0, 255), (0, 200, 0), (150, 150, 150), (255, 255, 255), (255, 200, 0)


def is_grey(frame):
    """True for a frame whose three channels are the same picture, as the infrared stream's are."""
    if frame.ndim == 2:
        return True
    sample = frame[::16, ::16].astype(np.int16)
    return bool(np.abs(sample[..., 0] - sample[..., 1]).max() <= 2
                and np.abs(sample[..., 1] - sample[..., 2]).max() <= 2)


def enhance_grey(frame, mode="stretch"):
    """Raise the contrast of a grey frame; returns a 3-channel frame the detector accepts."""
    grey = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mode == "stretch":      # 1st..99th percentile to full range
        low, high = np.percentile(grey, (1, 99))
        grey = np.clip((grey - low) * (255.0 / max(high - low, 1.0)), 0, 255).astype(np.uint8)
    elif mode == "clahe":      # local contrast
        grey = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(grey)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


class PhoneDetector:
    """YOLOX-S through OpenCV. `detect()` returns phones, `detect_all()` every class.

    Grey (infrared) frames are contrast-enhanced first with `grey_enhance`:
    "stretch", "clahe" or "none".
    """

    def __init__(self, input_width=640, min_score=0.35, grey_enhance="stretch"):
        if not MODEL_PATH.exists():
            print(f"Downloading the object detector (34 MB) to {MODEL_PATH} ...")
            MODEL_PATH.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        self._net = cv2.dnn.readNet(str(MODEL_PATH))
        self.input_width = input_width
        self.min_score = min_score
        self.grey_enhance = grey_enhance
        self._grid_for = {}

    def _grid(self, in_h, in_w):
        """Cell offsets and strides for the three output scales, in the model's output order."""
        if (in_h, in_w) not in self._grid_for:
            offsets, strides = [], []
            for stride in (8, 16, 32):
                xs, ys = np.meshgrid(np.arange(in_w // stride), np.arange(in_h // stride))
                offsets.append(np.stack((xs, ys), axis=2).reshape(-1, 2))
                strides.append(np.full((offsets[-1].shape[0], 1), stride))
            self._grid_for[(in_h, in_w)] = (np.concatenate(offsets), np.concatenate(strides))
        return self._grid_for[(in_h, in_w)]

    def detect_all(self, frame):
        """[(x, y, w, h, score, class_index)] in frame pixels, best first."""
        if self.grey_enhance != "none" and is_grey(frame):
            frame = enhance_grey(frame, self.grey_enhance)
        elif frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        h, w = frame.shape[:2]
        scale = self.input_width / w
        in_w = self.input_width
        in_h = math.ceil(h * scale / 32) * 32          # keeps the frame's shape: no wasted padding
        resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (in_w, round(h * scale)))
        padded = np.full((in_h, in_w, 3), 114.0, np.float32)
        padded[:resized.shape[0]] = resized
        self._net.setInput(padded.transpose(2, 0, 1)[np.newaxis])
        out = self._net.forward()[0]                   # rows: cx, cy, w, h, objectness, 80 class scores

        offsets, strides = self._grid(in_h, in_w)
        scores = out[:, 4:5] * out[:, 5:]
        classes = scores.argmax(axis=1)
        best = scores[np.arange(len(classes)), classes]
        keep = best >= self.min_score
        if not keep.any():
            return []
        centre = (out[keep, :2] + offsets[keep]) * strides[keep] / scale
        size = np.exp(out[keep, 2:4]) * strides[keep] / scale
        boxes = np.concatenate((centre - size / 2, size), axis=1)
        best, classes = best[keep], classes[keep]
        picked = cv2.dnn.NMSBoxesBatched(boxes.tolist(), best.tolist(), classes.tolist(),
                                         self.min_score, 0.5)
        found = [(*boxes[i].tolist(), float(best[i]), int(classes[i])) for i in np.array(picked).flatten()]
        return sorted(found, key=lambda d: -d[4])

    def detect(self, frame):
        """[(x, y, w, h, score)] for every phone in the frame, best first."""
        return [d[:5] for d in self.detect_all(frame) if d[5] == PHONE]


_shared = None


def phone_in_frame(frame):
    """True if a phone is visible in this frame (colour or infrared). Loads the model on the first call."""
    global _shared
    if _shared is None:
        _shared = PhoneDetector()
    return bool(_shared.detect(frame))


def draw(frame, found, show_all):
    for x, y, w, h, score, cls in found:
        if cls != PHONE and not show_all:
            continue
        colour = RED if cls == PHONE else GREY
        p0, p1 = (int(x), int(y)), (int(x + w), int(y + h))
        cv2.rectangle(frame, p0, p1, colour, 3 if cls == PHONE else 1)
        cv2.putText(frame, f"{COCO[cls]} {score:.2f}", (p0[0], max(p0[1] - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


def label(frame, text, org, scale, colour, thick=2):
    """Text on a dark box, so it reads on any picture."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(frame, (org[0] - 6, org[1] - th - 8), (org[0] + tw + 6, org[1] + base + 4), (25, 25, 25), -1)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def run_on_image(detector, path, show_all, out_path):
    frame = cv2.imread(path)
    if frame is None:
        sys.exit(f"ERROR: cannot read {path}")
    t0 = time.time()
    found = detector.detect_all(frame)
    ms = (time.time() - t0) * 1000
    print(f"{path}: {ms:.0f} ms, " + (", ".join(f"{COCO[c]} {s:.2f}" for *_, s, c in found) or "nothing"))
    if out_path:
        draw(frame, found, show_all)
        cv2.imwrite(out_path, frame)


def evaluate(folder, input_width):
    """Score the frames saved with p / n: how often is the phone found, how often falsely?"""
    folder = Path(folder)
    sets = {name: sorted(folder.glob(f"{name}_*.png")) for name in ("phone", "nophone")}
    if not sets["phone"] and not sets["nophone"]:
        sys.exit(f"ERROR: no phone_*.png or nophone_*.png in {folder}. Record some with p / n first.")
    print(f"{folder}: {len(sets['phone'])} frames with a phone, {len(sets['nophone'])} without")
    thresholds = (0.15, 0.25, 0.35, 0.50)
    frames = {name: [cv2.imread(str(f)) for f in files] for name, files in sets.items()}
    grey = all(is_grey(f) for fs in frames.values() for f in fs)
    print("\nenhance   " + "".join(f"  found@{t:.2f}" for t in thresholds)
          + "  |" + "".join(f"  false@{t:.2f}" for t in thresholds) + "  | median phone score")
    for mode in (ENHANCE_MODES if grey else ("none",)):
        detector = PhoneDetector(input_width, min_score=0.05, grey_enhance=mode)
        best = {name: np.array([max((p[4] for p in detector.detect(f)), default=0.0) for f in fs])
                for name, fs in frames.items()}
        cells = ["".join(f"  {100 * (best[name] >= t).mean():9.0f}%" if len(best[name]) else f"  {'-':>10s}"
                         for t in thresholds) for name in ("phone", "nophone")]
        median = f"{np.median(best['phone']):.2f}" if len(best["phone"]) else "-"
        print(f"{mode:10s}{cells[0]}  |{cells[1]}  | {median}")
    print("\nfound = share of phone frames where a phone was detected at that confidence;"
          "\nfalse = share of no-phone frames where one was detected anyway.")


class Stream:
    """One camera. A background reader keeps only the newest frame, so a slow
    detector never falls behind the live picture and the camera is always read."""

    def __init__(self, name, cap, description, skip=0):
        self.name, self.cap, self.description = name, cap, description
        self.window = f"phone_detect: {name} (q / Esc to quit)"
        self.record_dir = CAPTURE_DIR / f"phone_{name}"
        self.last_seen = -1e9          # time a phone was last found
        self.last_frame_t = time.time()
        self.n_frames = self.n_phone = 0
        self.ms = 0.0
        self.warned = False
        self._skip, self._frame, self._fresh, self._quit = skip, None, False, False
        self._thread = threading.Thread(target=self._read, daemon=True, name=f"read-{name}")
        self._thread.start()

    def _read(self):
        while not self._quit:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
            elif self._skip:
                self._skip -= 1
            else:
                self._frame, self._fresh = frame, True

    def take(self):
        """The newest frame, once; None if nothing new has arrived."""
        if not self._fresh:
            return None
        self._fresh = False
        self.last_frame_t = time.time()
        return self._frame

    def close(self):
        self._quit = True
        self._thread.join(timeout=2)
        self.cap.release()


def open_streams(args):
    """The streams the --source option asks for, colour first."""
    from face_stream import pick_camera               # also puts the SDK on sys.path
    from camera_utils import (CameraNotFoundError, backend_id, describe, find_camera,
                              list_cameras, open_camera)

    def colour(index, fmt):
        cap, info = open_camera(index, width=args.width, height=args.height, fourcc=fmt, fps=30)
        return Stream("rgb", cap, describe(info))

    def infrared(index, backend=None):
        cap, info = open_camera(index, backend, *IR_SIZES[args.size], "MJPG", 30)
        return Stream("ir", cap, describe(info), IR_WARMUP_FRAMES)

    if args.source == "rgb":
        return [colour(args.device if args.device is not None else pick_camera(), args.format)]
    ir_index = args.ir_device if args.ir_device is not None else (args.device if args.source == "ir" else None)
    if args.source == "ir":
        if ir_index is not None:
            return [infrared(ir_index)]
        print("Looking for the near-infrared (CPD) camera ...")
        index, backend_name = find_camera("ir")
        return [infrared(index, backend_id(backend_name))]

    # both: look at what is connected before anything is open
    print("Scanning for cameras ...")
    cams = list_cameras()
    for c in cams:
        c["ir"] = c["gray"] or c["height"] > c["width"]
        print(f"  [{c['index']}] {c['width']}x{c['height']}  ({'infrared' if c['ir'] else 'colour'})")
    if ir_index is None:
        ir_index = next((c["index"] for c in cams if c["ir"]), None)
        if ir_index is None:
            raise CameraNotFoundError("No near-infrared (CPD) camera found. Is the LogiInsight plugged in?")
    rgb_index = args.device
    if rgb_index is None:
        choices = [c["index"] for c in cams if not c["ir"]]
        if not choices:
            raise CameraNotFoundError("No colour camera found.")
        rgb_index = choices[0]
        if len(choices) > 1:
            nearest = min(choices, key=lambda i: abs(i - ir_index))   # the camera's streams sit together
            try:
                answer = input(f"Colour camera index {choices} [{nearest}]: ").strip()
            except EOFError:
                answer = ""
            rgb_index = int(answer) if answer.isdigit() and int(answer) in choices else nearest
    fmt = args.format
    if fmt == "MJPG":
        fmt = "YUY2"
        print("Opening the colour stream as YUY2 (MJPG would slow the infrared stream to about 1 fps).")
    first = colour(rgb_index, fmt)                    # colour first, as the API reference advises
    time.sleep(0.5)
    try:
        return [first, infrared(ir_index)]
    except CameraNotFoundError:
        first.close()
        raise


def show(stream, frame, found, phones, args, now, recording, record_left, message):
    view = enhance_grey(frame, args.ir_enhance) if is_grey(frame) else frame   # what the model saw
    if view is frame:
        view = frame.copy()
    draw(view, found, args.all)
    if view.shape[0] < 700:                           # the infrared sizes are small on screen
        k = 800 / view.shape[0]
        view = cv2.resize(view, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
    h, w = view.shape[:2]
    big = w / 1280
    if recording:
        label(view, f"recording '{recording}': {record_left} left", (20, h // 2), 0.8, CYAN)
    elif now - stream.last_seen <= args.hold:
        cv2.rectangle(view, (0, 0), (w - 1, h - 1), RED, 24)
        (tw, _), _ = cv2.getTextSize("PHONE", cv2.FONT_HERSHEY_SIMPLEX, 5.0 * big, 14)
        cv2.putText(view, "PHONE", ((w - tw) // 2, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 5.0 * big, RED, 14)
    best = f"{phones[0][4]:.2f}" if phones else "-"
    label(view, f"{stream.name}: phone score {best} (needs {args.min_score:.2f})   {stream.ms:.0f} ms",
          (28, 48), 0.65, WHITE)
    label(view, message or "p: record phone   n: record no phone   q: quit", (28, h - 30), 0.5, WHITE, 1)
    cv2.imshow(stream.window, view)


def run(streams, detector, args):
    """Detect on each stream's newest frame in turn until quit. Returns False if a stream died."""
    recording, record_left, message = None, 0, ""
    alive = True
    try:
        while not args.frames or min(s.n_frames for s in streams) < args.frames:
            saved_any = False
            for s in streams:
                frame = s.take()
                now = time.time()
                if frame is None:
                    if now - s.last_frame_t > 5 and not s.warned:
                        s.warned = True
                        print(f"WARNING: no frames from the {s.name} stream for 5 s "
                              f"(camera unplugged, or in use by another program?)")
                    if now - s.last_frame_t > 12 and (args.frames or args.no_display):
                        alive = False
                    continue
                if recording:                         # the frame as the camera sent it
                    s.record_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(s.record_dir / f"{recording}_{stamp}_{record_left:02d}.png"), frame)
                    saved_any = True
                t0 = time.time()
                found = detector.detect_all(frame)
                took = (time.time() - t0) * 1000
                s.ms = 0.9 * s.ms + 0.1 * took if s.n_frames else took
                s.n_frames += 1
                phones = [d for d in found if d[5] == PHONE]
                s.n_phone += bool(phones)
                if phones:
                    s.last_seen = t0
                if not args.no_display:
                    show(s, frame, found, phones, args, t0, recording, record_left, message)
            if not alive:
                break
            if saved_any:
                record_left -= 1
                if not record_left:
                    where = ", ".join(f"{s.record_dir.parent.name}/{s.record_dir.name}" for s in streams)
                    message = f"saved {RECORD_FRAMES} '{recording}' frames to {where}"
                    print(message)
                    recording = None
            if args.no_display:
                time.sleep(0.002)
                continue
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in RECORD_KEYS and not recording:
                recording, record_left, message = RECORD_KEYS[key], RECORD_FRAMES, ""
                stamp = time.strftime("%H%M%S")
    except KeyboardInterrupt:
        pass
    finally:
        for s in streams:
            s.close()
        cv2.destroyAllWindows()
    for s in streams:
        print(f"{s.name}: {s.n_frames} frames, phone seen in {s.n_phone}, {s.ms:.0f} ms per detection")
    return alive


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", choices=("rgb", "ir", "both"), default="rgb",
                   help="rgb = a colour camera (default); ir = the LogiInsight near-infrared stream; "
                        "both = the two together, one window each")
    p.add_argument("--device", type=int, help="colour camera index (default: ask); with --source ir, the infrared index")
    p.add_argument("--ir-device", type=int, help="infrared camera index (default: auto-detect)")
    p.add_argument("--width", type=int, default=1280, help="colour stream width")
    p.add_argument("--height", type=int, default=720, help="colour stream height")
    p.add_argument("--format", choices=("MJPG", "YUY2", "NV12"), default="MJPG",
                   help="colour stream format (with --source both, MJPG is replaced by YUY2)")
    p.add_argument("--size", choices=list(IR_SIZES), default="512x640", help="infrared frame size")
    p.add_argument("--ir-enhance", choices=ENHANCE_MODES, default="stretch",
                   help="contrast step for grey frames before detection (default: stretch)")
    p.add_argument("--input-width", type=int, default=640,
                   help="width the picture is scaled to for the model (multiple of 32; larger is slower)")
    p.add_argument("--min-score", type=float, default=0.35, help="confidence needed to count (default 0.35)")
    p.add_argument("--hold", type=float, default=1.0, help="keep PHONE shown this long after the last sighting")
    p.add_argument("--all", action="store_true", help="also draw the other object classes")
    p.add_argument("--image", help="run on this picture file instead of a camera")
    p.add_argument("--save", help="with --image: write the annotated picture here")
    p.add_argument("--eval", metavar="DIR", help="score the frames saved with p / n in this folder, then exit")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames per stream (0 = run until quit)")
    p.add_argument("--no-display", action="store_true", help="no window; print a summary")
    args = p.parse_args()

    if args.eval:
        evaluate(args.eval, args.input_width)
        return
    detector = PhoneDetector(args.input_width, args.min_score, args.ir_enhance)
    if args.image:
        run_on_image(detector, args.image, args.all, args.save)
        return

    try:
        streams = open_streams(args)
    except Exception as e:                            # CameraNotFoundError, defined in the SDK
        if type(e).__name__ != "CameraNotFoundError":
            raise
        sys.exit(f"ERROR: {e}")
    for s in streams:
        print(f"Opened {s.name}: {s.description}")
    if not run(streams, detector, args):
        sys.exit("ERROR: a stream stopped delivering frames.")


if __name__ == "__main__":
    main()
