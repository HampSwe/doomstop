"""Detects a phone in the picture with a ready-made object detector.

The model is YOLOX-S from the OpenCV model zoo, trained on the 80 COCO object
classes, one of which is "cell phone". It runs on this PC through OpenCV, on
the plain video frame, so it works with any webcam and needs no camera login.
The model file (34 MB) is downloaded into models/ on first use.

    python phone_detect.py                 # pick a camera, show PHONE when one is seen
    python phone_detect.py --device 1      # skip the camera prompt
    python phone_detect.py --source ir     # the LogiInsight near-infrared stream instead

Colour and infrared side by side, in two terminals (colour first, as YUY2):

    python phone_detect.py --device 1 --format YUY2
    python phone_detect.py --source ir --device 2

    python phone_detect.py --all           # also draw every other object it recognises
    python phone_detect.py --image a.jpg   # run on a picture file instead of a camera

Infrared: the picture is grey and often low in contrast. Grey input by itself
does not bother the model, but flat contrast does (on grey test photos the
score fell from about 0.83 to 0.53 and came back with a contrast stretch), so
grey frames are stretched before detection (--ir-enhance). How well the model
copes with real infrared pictures has to be measured on recordings:

    p / n in the window    save the next frames as "phone" / "no phone" examples
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


def open_stream(args):
    """Returns (capture, description, frames to skip)."""
    from face_stream import pick_camera               # also puts the SDK on sys.path
    from camera_utils import backend_id, describe, find_camera, open_camera
    if args.source == "ir":
        width, height = IR_SIZES[args.size]
        if args.device is not None:
            index, backend = args.device, None
        else:
            print("Looking for the near-infrared (CPD) camera ...")
            index, backend_name = find_camera("ir")
            backend = backend_id(backend_name)
        cap, info = open_camera(index, backend, width, height, "MJPG", 30)
        return cap, describe(info), IR_WARMUP_FRAMES
    index = args.device if args.device is not None else pick_camera()
    cap, info = open_camera(index, width=args.width, height=args.height, fourcc=args.format, fps=30)
    return cap, describe(info), 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", choices=("rgb", "ir"), default="rgb",
                   help="rgb = a colour camera (default); ir = the LogiInsight near-infrared stream")
    p.add_argument("--device", type=int, help="camera index (default: ask for rgb, auto-detect for ir)")
    p.add_argument("--width", type=int, default=1280, help="colour stream width")
    p.add_argument("--height", type=int, default=720, help="colour stream height")
    p.add_argument("--format", choices=("MJPG", "YUY2", "NV12"), default="MJPG",
                   help="colour stream format; use YUY2 when the infrared stream is open too, "
                        "since MJPG on the colour side slows infrared to about 1 fps")
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
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until quit)")
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
        cap, description, skip = open_stream(args)
    except Exception as e:                            # CameraNotFoundError, defined in the SDK
        if type(e).__name__ != "CameraNotFoundError":
            raise
        sys.exit(f"ERROR: {e}")
    print(f"Opened {description}")
    record_dir = CAPTURE_DIR / f"phone_{args.source}"

    last_seen, n_frames, n_phone, ms = -1e9, 0, 0, 0.0
    recording, record_left, message = None, 0, ""
    try:
        while not args.frames or n_frames < args.frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                if not args.no_display and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            if skip:
                skip -= 1
                continue
            if recording:                             # the frame as the camera sent it
                record_dir.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%H%M%S")
                cv2.imwrite(str(record_dir / f"{recording}_{stamp}_{record_left:02d}.png"), frame)
                record_left -= 1
                if not record_left:
                    message = f"saved {RECORD_FRAMES} '{recording}' frames to {record_dir.relative_to(ROOT)}"
                    print(message)
                    recording = None

            t0 = time.time()
            found = detector.detect_all(frame)
            ms = 0.9 * ms + 0.1 * (time.time() - t0) * 1000 if n_frames else (time.time() - t0) * 1000
            n_frames += 1
            phones = [d for d in found if d[5] == PHONE]
            n_phone += bool(phones)
            if phones:
                last_seen = t0
            if args.no_display:
                continue

            view = enhance_grey(frame, args.ir_enhance) if is_grey(frame) else frame   # show what the model saw
            draw(view, found, args.all)
            if view.shape[0] < 700:                   # the infrared sizes are small on screen
                k = 800 / view.shape[0]
                view = cv2.resize(view, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
            h, w = view.shape[:2]
            big = w / 1280
            if recording:
                label(view, f"recording '{recording}': {record_left} frames left", (20, h // 2), 0.8, CYAN)
            elif t0 - last_seen <= args.hold:
                cv2.rectangle(view, (0, 0), (w - 1, h - 1), RED, 24)
                (tw, _), _ = cv2.getTextSize("PHONE", cv2.FONT_HERSHEY_SIMPLEX, 5.0 * big, 14)
                cv2.putText(view, "PHONE", ((w - tw) // 2, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 5.0 * big, RED, 14)
            best = f"{phones[0][4]:.2f}" if phones else "-"
            label(view, f"phone score {best} (needs {args.min_score:.2f})   {ms:.0f} ms", (28, 48), 0.65, WHITE)
            label(view, message or "p: record phone   n: record no phone   q: quit", (28, h - 30), 0.5, WHITE, 1)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in RECORD_KEYS and not recording:
                recording, record_left, message = RECORD_KEYS[key], RECORD_FRAMES, ""
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"{n_frames} frames, phone seen in {n_phone}, {ms:.0f} ms per frame")


if __name__ == "__main__":
    main()
