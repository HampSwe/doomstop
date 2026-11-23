"""Hand positions, and whether a detected phone is in a hand or just lying there.

Hands come from MediaPipe's Hand Landmarker: 21 points per hand (wrist,
knuckles, fingertips). The model file is downloaded into models/ on first use.

The phone box (from phone_detect.py) and the hands must come from the SAME
picture, since the colour and infrared cameras see different views. From them:

  distance   from the phone box to the nearest hand point, measured in phone
             sizes (the box diagonal), so it does not depend on how far away
             the phone is. 0 means a hand point is inside the box.

  closeness  the distance turned into 0..1: 1 with a hand point inside the
             box, 0.5 at NEAR phone sizes away, 0 from twice that or with no
             hand in view.

  in_hand    0..1: the closeness averaged over the last moments (a running
             average with a time constant of IN_HAND_SMOOTH_S). 0 without a
             phone.

  state      "no phone"        no phone box
             "phone out"       a phone, with in_hand low
             "phone in hand"   in_hand at IN_HAND_ON or above (it switches
                               back below IN_HAND_OFF, so it does not flicker)

The averaging over time is there because the camera has no depth: a hand
passing in front of a phone on the table looks like touching it for a moment.
The phone's last position is kept briefly when the detector loses it, because
the hand holding a phone can hide it.
"""

import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent      # the repository folder
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

NEAR = 0.25            # a hand point this close to the phone box (in phone sizes) counts as at the phone
IN_HAND_SMOOTH_S = 0.7 # time constant of the in-hand score: about half a second of holding reaches 0.5
IN_HAND_ON = 0.5       # "phone in hand" from this in-hand score ...
IN_HAND_OFF = 0.35     # ... until it falls below this
PHONE_HOLD_S = 1.0     # keep the phone's last position this long after the detector loses it
MIN_SIDE = 384         # small pictures are enlarged to this, so the hand model sees enough pixels
CLOSE_UP_PAD = 1.5     # the second look covers the phone plus this many phone lengths on each side ...
CLOSE_UP_CONFIDENCE = 0.2   # ... with a lower bar: a hand gripping a phone is half hidden by it

HAND_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17))
STATES = ("no phone", "phone out", "phone in hand")


class HandDetector:
    """MediaPipe Hand Landmarker. Use one instance per camera stream: it tracks between frames.

    Two looks: the whole frame, and, when a phone box is given and no hand was
    found at it, a close-up around the phone with a lower confidence bar.
    """

    def __init__(self, max_hands=2, min_confidence=0.4):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if not MODEL_PATH.exists():
            print(f"Downloading the hand model to {MODEL_PATH} ...")
            MODEL_PATH.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        def landmarker(mode, confidence):
            return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=mode,
                num_hands=max_hands,
                min_hand_detection_confidence=confidence,
                min_hand_presence_confidence=confidence,
                min_tracking_confidence=confidence,
            ))

        self._mp = mp
        # 0.4: infrared pictures need less than the 0.5 default (taken from hand_desk.py)
        self._whole = landmarker(vision.RunningMode.VIDEO, min_confidence)
        self._close_up = landmarker(vision.RunningMode.IMAGE, CLOSE_UP_CONFIDENCE)   # the crop moves: no tracking
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self._timestamp_ms = 0

    def _image(self, picture):
        """(MediaPipe image, enlargement) for a BGR picture, enlarged if it is small."""
        scale = max(1.0, MIN_SIDE / min(picture.shape[:2]))
        if scale > 1.0:
            picture = cv2.resize(picture, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        rgb = np.ascontiguousarray(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB))
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

    def detect(self, frame, now, phone_box=None):
        """One (21, 2) array of pixel positions per hand found in this BGR frame.

        `phone_box` (x, y, w, h): where a phone is, to take the second look there.
        """
        sample = frame[::16, ::16].astype(np.int16)
        if np.abs(sample[..., 0] - sample[..., 1]).max() <= 2 and np.abs(sample[..., 1] - sample[..., 2]).max() <= 2:
            # a grey (infrared) frame: raise its local contrast first, as hand_desk.py does
            frame = cv2.cvtColor(self._clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)
        h, w = frame.shape[:2]
        self._timestamp_ms = max(int(now * 1000), self._timestamp_ms + 1)   # must strictly increase
        result = self._whole.detect_for_video(self._image(frame), self._timestamp_ms)
        hands = [np.array([(p.x * w, p.y * h) for p in hand], dtype=np.float32)
                 for hand in result.hand_landmarks]

        if phone_box is not None:
            distance = hand_phone_distance(phone_box, hands)
            if distance is None or distance > NEAR:
                bx, by, bw, bh = phone_box[:4]
                pad = CLOSE_UP_PAD * max(bw, bh)
                x0, y0 = int(max(bx - pad, 0)), int(max(by - pad, 0))
                x1, y1 = int(min(bx + bw + pad, w)), int(min(by + bh + pad, h))
                if x1 - x0 > 8 and y1 - y0 > 8:
                    result = self._close_up.detect(self._image(frame[y0:y1, x0:x1]))
                    hands += [np.array([(x0 + p.x * (x1 - x0), y0 + p.y * (y1 - y0)) for p in hand],
                                       dtype=np.float32) for hand in result.hand_landmarks]
        return hands

    def close(self):
        self._whole.close()
        self._close_up.close()


def hand_phone_distance(box, hands):
    """Distance from a phone box (x, y, w, h) to the nearest hand point, in phone sizes.

    0 if a hand point lies inside the box; None if there are no hands.
    """
    if not hands:
        return None
    x, y, w, h = box[:4]
    points = np.concatenate(hands)
    dx = np.maximum(np.maximum(x - points[:, 0], points[:, 0] - (x + w)), 0)
    dy = np.maximum(np.maximum(y - points[:, 1], points[:, 1] - (y + h)), 0)
    return float(np.hypot(dx, dy).min() / max(math.hypot(w, h), 1.0))


def closeness(distance):
    """A hand-to-phone distance as 0..1: 1 touching, 0.5 at NEAR, 0 from 2 * NEAR or with no hand."""
    if distance is None:
        return 0.0
    return float(min(max(1.0 - distance / (2 * NEAR), 0.0), 1.0))


class PhoneUse:
    """Turns one stream's phone boxes and hands into an in-hand score and
    "no phone" / "phone out" / "phone in hand"."""

    def __init__(self):
        self.state = "no phone"
        self.in_hand = 0.0         # 0..1: how steadily a hand has been at the phone
        self.distance = None       # hand-to-phone distance of the latest frame, in phone sizes
        self.box = None            # the phone's last known box
        self._box_t = -1e9
        self._last_t = None

    def update(self, phones, hands, now):
        """`phones`: [(x, y, w, h, score)], best first. `hands`: HandDetector.detect() output."""
        if phones:
            self.box, self._box_t = phones[0][:4], now
        elif now - self._box_t > PHONE_HOLD_S:
            self.box = None
        if self.box is None:
            self.distance, self.in_hand, self.state, self._last_t = None, 0.0, "no phone", None
            return self.state
        self.distance = hand_phone_distance(self.box, hands)
        dt = 0.0 if self._last_t is None else min(now - self._last_t, 1.0)
        self._last_t = now
        self.in_hand += (1.0 - math.exp(-dt / IN_HAND_SMOOTH_S)) * (closeness(self.distance) - self.in_hand)
        held = self.in_hand >= (IN_HAND_OFF if self.state == "phone in hand" else IN_HAND_ON)
        self.state = "phone in hand" if held else "phone out"
        return self.state


def draw_hands(frame, hands, colour=(255, 200, 0)):
    for points in hands:
        pts = points.astype(int)
        for a, b in HAND_EDGES:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), colour, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p), 3, (255, 255, 255), -1, cv2.LINE_AA)
