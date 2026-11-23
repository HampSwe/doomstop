"""The live signal source shared by dashboard.py and collect.py.

Opens the LogiInsight colour and infrared streams, reads the camera's face
data and runs the phone and hand detectors, so that each step() gives the two
newest pictures and every signal (see look_down.signals).
"""

import time
from collections import namedtuple

from face_stream import connect_device_ai
from camera_utils import CameraNotFoundError
from look_down import FaceFeed, open_cameras, signals, start_phone_worker

# ir_frame is None until the first infrared frame. `raw` holds the measurements behind the signals.
Step = namedtuple("Step", "now frame ir_frame values raw", defaults=({},))
RAW = ("color_phone_seen", "phone_seen", "phone_held",
       "face_seen", "head", "eyes", "pitch_deg", "lid", "eye_opening", "pitch_base", "lid_base",
       "eye_opening_base", "color_phone_now", "phone_now", "hands", "hand_distance")


class EngineError(Exception):
    """The cameras or the camera's face data are not available; the message says why."""


class Engine:
    def __init__(self, device=None, ir_device=None, phone_score=0.35):
        client, reason = connect_device_ai(mesh=True)
        if client is None:
            raise EngineError(f"this needs the camera's face mesh, which is unavailable: {reason}\n"
                              "Run logiinsight_api\\check_connection.py to see what is wrong.")
        try:
            self.cap, index, self.ir = open_cameras(client, device, True, ir_device)
        except CameraNotFoundError as e:
            client.disconnect()
            raise EngineError(str(e)) from None
        self.client = client
        self.feed = FaceFeed(client, index)
        self.det = self.feed.det                      # the look-down detector: calibrate() it from the UI
        self.worker = start_phone_worker(phone_score, True)
        self.phone_score = phone_score
        self.started = time.time()
        self._ir_frame = None

    def step(self):
        """Wait for the next colour frame. Returns a Step, or None if the camera gave none."""
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        now = time.time()
        self.worker.submit(frame.copy())              # a copy: callers draw on `frame`
        fresh = self.ir.take()
        if fresh is not None:
            self._ir_frame = fresh
            self.worker.submit(fresh, "ir")           # never drawn on: views copy it
        self.feed.poll(now)
        return Step(now, frame, self._ir_frame, signals(self.det, self.worker, now), self.raw(now))

    def raw(self, now):
        """The yes/no versions of the phone signals, and the measurements behind all signals (names in RAW).

        color_phone_seen, phone_seen   0 or 1: the phone score of that stream is at the threshold or above
        phone_held        0 or 1: the infrared stream's state is "phone in hand"
        head, eyes        the look-down scores (about 1.0 at the poses they were tuned on), smoothed
        pitch_deg, lid, eye_opening and their *_base   latest face-mesh measurements and the
                          "looking at the screen" values they are compared with
        color_phone_now, phone_now   phone confidence in the latest frame alone (the signals hold
                          the best of the last second)
        hands, hand_distance   hands found in the latest infrared frame, and the nearest hand
                          point's distance to the phone in phone sizes (None without both)
        """
        det, feed, worker = self.det, self.feed, self.worker
        mesh = feed.features if feed.features is not None else (None, None, None)
        base = det.base if det.base is not None else (None, None, None)
        return {
            "color_phone_seen": int(worker.is_on(now, "rgb")), "phone_seen": int(worker.is_on(now, "ir")),
            "phone_held": int(worker.use.state == "phone in hand") if worker.use else None,
            "face_seen": int(not feed.face_lost(now)), "head": det.head, "eyes": det.eyes,
            "pitch_deg": mesh[0], "lid": mesh[1], "eye_opening": mesh[2],
            "pitch_base": base[0], "lid_base": base[1], "eye_opening_base": base[2],
            "color_phone_now": worker.latest["rgb"], "phone_now": worker.latest["ir"],
            "hands": len(worker.hand_points),
            "hand_distance": worker.use.distance if worker.use else None,
        }

    def face(self, now):
        """The face the camera last reported, or None once it has been lost for a moment."""
        return None if self.feed.face_lost(now) else self.feed.face

    def close(self):
        self.cap.release()
        self.ir.close()
        self.worker.stop()
        self.client.disconnect()
