"""Detects a phone in the picture with a ready-made object detector.

The model is YOLOX-S from the OpenCV model zoo, trained on the 80 COCO object
classes, one of which is "cell phone". It runs on this PC through OpenCV, on
the plain video frame, so it works with any webcam and needs no camera login.
The model file (34 MB) is downloaded into models/ on first use.

    python phone_detect.py                 # pick a camera, show PHONE when one is seen
    python phone_detect.py --device 1      # skip the camera prompt
    python phone_detect.py --all           # also draw every other object it recognises
    python phone_detect.py --image a.jpg   # run on a picture file instead of a camera

From other code:

    from phone_detect import phone_in_frame
    phone_in_frame(frame)                  # True / False for one BGR frame

Press q or Esc in the window to quit.
"""

import argparse
import math
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

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

WINDOW = "phone_detect (q / Esc to quit)"
RED, GREEN, GREY, WHITE = (0, 0, 255), (0, 200, 0), (150, 150, 150), (255, 255, 255)


class PhoneDetector:
    """YOLOX-S through OpenCV. `detect()` returns phones, `detect_all()` every class."""

    def __init__(self, input_width=640, min_score=0.35):
        if not MODEL_PATH.exists():
            print(f"Downloading the object detector (34 MB) to {MODEL_PATH} ...")
            MODEL_PATH.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        self._net = cv2.dnn.readNet(str(MODEL_PATH))
        self.input_width = input_width
        self.min_score = min_score
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
    """True if a phone is visible in this BGR frame. Loads the model on the first call."""
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


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", type=int, help="camera index (default: list and ask)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--min-score", type=float, default=0.35, help="confidence needed to count (default 0.35)")
    p.add_argument("--hold", type=float, default=1.0, help="keep PHONE shown this long after the last sighting")
    p.add_argument("--all", action="store_true", help="also draw the other object classes")
    p.add_argument("--image", help="run on this picture file instead of a camera")
    p.add_argument("--save", help="with --image: write the annotated picture here")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until quit)")
    p.add_argument("--no-display", action="store_true", help="no window; print a summary")
    args = p.parse_args()

    detector = PhoneDetector(min_score=args.min_score)
    if args.image:
        run_on_image(detector, args.image, args.all, args.save)
        return

    from face_stream import pick_camera               # also puts the SDK on sys.path
    from camera_utils import CameraNotFoundError, describe, open_camera
    try:
        index = args.device if args.device is not None else pick_camera()
        cap, info = open_camera(index, width=args.width, height=args.height, fourcc="MJPG", fps=30)
    except CameraNotFoundError as e:
        sys.exit(f"ERROR: {e}")
    print(f"Opened {describe(info)}")

    last_seen, n_frames, n_phone, ms = -1e9, 0, 0, 0.0
    try:
        while not args.frames or n_frames < args.frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                if not args.no_display and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
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

            draw(frame, found, args.all)
            h, w = frame.shape[:2]
            if t0 - last_seen <= args.hold:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 24)
                (tw, _), _ = cv2.getTextSize("PHONE", cv2.FONT_HERSHEY_SIMPLEX, 5.0, 14)
                cv2.putText(frame, "PHONE", ((w - tw) // 2, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 5.0, RED, 14)
            best = f"{phones[0][4]:.2f}" if phones else "-"
            status = f"phone score {best} (needs {args.min_score:.2f})   {ms:.0f} ms per frame"
            cv2.rectangle(frame, (20, 20), (620, 60), (25, 25, 25), -1)
            cv2.putText(frame, status, (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2)
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"{n_frames} frames, phone seen in {n_phone}, {ms:.0f} ms per frame")


if __name__ == "__main__":
    main()
