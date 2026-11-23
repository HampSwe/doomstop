"""MAC ONLY. engine.Engine for macOS, where the LogiInsight colour camera is not listed.

macOS's camera list only offers the camera's infrared stream ("LVNext CPD
Camera"), so cv2.VideoCapture can never open the colour one. This engine reads
the colour stream straight over USB with pyuvc instead, which needs sudo. While
it holds the camera, macOS lets go of the whole device, including its USB
network link, so the camera's own face AI can't be reached either. The
478-point face mesh is therefore computed on this Mac with MediaPipe's
FaceLandmarker, which uses the same landmark layout, and fed to
look_down.LookDownDetector exactly as the camera's mesh is.

MacEngine has the same interface as engine.Engine (det, feed, worker,
started, step(), raw(), face(), close()), so dashboard.py and collect.py run
unchanged through mac/run_mac.py. On Windows and Linux, use engine.Engine.

Colour OR infrared, never both, on a Mac (tested 2026-11-21 with
mac/diag_ir_first.py): taking the colour stream over USB takes the whole
device away from macOS, so infrared delivers no frames afterwards; and with
infrared open first, the colour stream over USB times out. So this engine runs
colour only: the infrared signals (phone, phone_in_hand, gaze_down_x_phone)
are off (None) and the panel says "waiting for infrared". Both streams
together need Windows or Linux. (open_infrared() is kept for with_ir=True.)
"""

import math
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT, ROOT / "detector"):      # detector/ (engine.py, look_down.py ...) ends up first
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import engine  # noqa: E402
from look_down import FACE_LOST_S, LookDownDetector, features, signals, start_phone_worker  # noqa: E402

MESH_MODEL = ROOT / "models" / "face_landmarker.task"
MESH_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task")


class LocalMesh:
    """MediaPipe FaceLandmarker on this Mac, returning a face shaped like the camera's JSON."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if not MESH_MODEL.exists():
            print(f"Downloading the face mesh model to {MESH_MODEL} ...")
            MESH_MODEL.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(MESH_URL, MESH_MODEL)
        self._mp = mp
        self._lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MESH_MODEL)),
            running_mode=vision.RunningMode.VIDEO, num_faces=1))
        self._last_ms = -1

    def face(self, frame_bgr, now):
        ms = int(now * 1000)
        if ms <= self._last_ms:          # VIDEO mode needs strictly increasing timestamps
            ms = self._last_ms + 1
        self._last_ms = ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._lm.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ms)
        if not result.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        # pixels, with depth on the x scale (MediaPipe's z uses roughly the same scale as x)
        pts = [[p.x * w, p.y * h, p.z * w] for p in result.face_landmarks[0]]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return {"landmarks": pts, "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]}

    def close(self):
        self._lm.close()


class LocalFaceFeed:
    """look_down.FaceFeed, but fed by LocalMesh on each colour frame instead of the camera's AI."""

    def __init__(self):
        self.mesh = LocalMesh()
        self.det = LookDownDetector()
        self.face = None
        self.features = None
        self.n_mesh = 0
        self._last_face_t = time.time()

    def poll(self, now, frame):
        self.face = self.mesh.face(frame, now)
        if not self.face:
            return None
        self._last_face_t = now
        landmarks = self.face["landmarks"]
        f = features(landmarks)
        if f is not None:
            self.n_mesh += 1
            self.features = f
            self.det.update(f, now)
        return landmarks

    def face_lost(self, now):
        return now - self._last_face_t > FACE_LOST_S


def to_bgr(f):
    """A writable BGR copy of a pyuvc frame (pyuvc reuses its buffers)."""
    try:
        img = f.bgr
        if img is not None:
            return img.copy()
    except Exception:
        pass
    return cv2.imdecode(np.frombuffer(bytes(f.jpeg_buffer), np.uint8), cv2.IMREAD_COLOR)


class ColourUSB:
    """The LogiInsight colour stream over USB (pyuvc), with a cv2.VideoCapture-like read()."""

    def __init__(self, width=1280, height=720, fps=30):
        import uvc

        self._uvc = uvc
        devs = [d for d in uvc.device_list() if d["name"].startswith("LVNext")]
        if not devs:
            raise engine.EngineError("the LogiInsight camera is not on USB. Plug it in directly "
                                     "(no hub) and wait about 20 s.")
        try:
            self.cap = uvc.Capture(devs[0]["uid"])
        except uvc.OpenError as e:
            raise engine.EngineError(f"could not open the camera over USB ({e}). Run with sudo, and close "
                                     "other camera apps (phone_detect, Zoom, FaceTime, browser).") from None
        modes = [m for m in self.cap.available_modes if m.format_name == "MJPG"] or list(self.cap.available_modes)
        if not modes:
            self.cap.close()
            raise engine.EngineError("the camera offered no video modes")
        best = min(modes, key=lambda m: (abs(m.width * m.height - width * height), abs(m.fps - fps)))
        self.cap.frame_mode = best
        if best.width < best.height:
            print("WARNING: this looks like the infrared stream, not the colour one")
        self.description = (f"LogiInsight colour over USB (pyuvc)  {best.width}x{best.height} "
                            f"@ {best.fps}fps {best.format_name}")
        self._errors = 0

    def read(self):
        try:
            frame = to_bgr(self.cap.get_frame(timeout=1.0))
            self._errors = 0
            return frame is not None, frame
        except (self._uvc.StreamError, TimeoutError) as e:   # get_frame raises TimeoutError on a stall
            self._errors += 1
            if self._errors >= 5:
                raise engine.EngineError(f"the colour stream stopped (no frame for 5 s{': ' + str(e) if str(e) else ''}). "
                                         "Unplug and replug the camera, wait about 20 s, and start again.") from None
            return False, None

    def release(self):
        self.cap.close()


def cpd_listed():
    """True if macOS currently lists the LogiInsight infrared camera ("LVNext CPD Camera")."""
    import subprocess

    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType"], capture_output=True,
                             text=True, timeout=15).stdout
    except Exception:
        return False
    return "CPD Camera" in out


def looks_like_ir(frame):
    """The LogiInsight infrared picture is portrait AND grey. An iPhone (Continuity Camera)
    can be portrait too, but it is in colour, so both are required."""
    if frame is None or frame.ndim != 3 or frame.shape[0] <= frame.shape[1]:
        return False
    small = frame[::8, ::8].astype(np.int16)
    spread = np.abs(small[..., 0] - small[..., 1]).mean() + np.abs(small[..., 1] - small[..., 2]).mean()
    return spread < 6.0


def open_infrared(ir_device=None):
    """The LogiInsight infrared stream through macOS, if it is still offered while pyuvc holds
    the camera; else None. Never falls back to another portrait camera such as an iPhone."""
    from camera_utils import CameraNotFoundError, backend_id, describe, open_camera
    from phone_detect import IR_SIZES, IR_WARMUP_FRAMES, Stream

    def off(why):
        print(f"NOTE: no LogiInsight infrared stream ({why}); the infrared signals are off.")
        return None

    if not cpd_listed():
        return off("macOS stopped listing 'LVNext CPD Camera' once the colour stream was opened over USB")
    backend = None
    try:
        if ir_device is None:
            # find_camera("ir") takes the first camera that is portrait OR grey, which on a Mac
            # is often an iPhone (Continuity Camera). Probe them all and require portrait AND grey.
            from camera_utils import list_cameras
            print("Looking for the near-infrared (CPD) camera ...")
            cams = list_cameras()
            for c in cams:
                print(f"  [{c['index']}] {c['width']}x{c['height']}  {'grey' if c['gray'] else 'colour'}")
            ir_cams = [c for c in cams if c["gray"] and c["height"] > c["width"]]
            if not ir_cams:
                return off("no camera gives a grey portrait picture")
            ir_device = ir_cams[0]["index"]
            backend = ir_cams[0].get("backend")
            if isinstance(backend, str):
                backend = backend_id(backend)
        cap, info = open_camera(ir_device, backend, *IR_SIZES["512x640"], "MJPG", 30)
    except (CameraNotFoundError, Exception) as e:  # noqa: BLE001 - any failure means "no infrared"
        return off(str(e))
    frame = None
    for _ in range(15):                               # skip warm-up frames, keep the last good one
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    if not looks_like_ir(frame):
        cap.release()
        return off(f"camera index {ir_device} is not the LogiInsight infrared (a colour picture, "
                   "e.g. an iPhone's Continuity Camera)")
    ir = Stream("ir", cap, describe(info), IR_WARMUP_FRAMES)
    print(f"Opened infrared: {ir.description}")
    return ir


class MacEngine(engine.Engine):
    def __init__(self, device=None, ir_device=None, phone_score=0.35, with_ir=False):
        # `device` is ignored: the colour stream is found on USB, not by camera index.
        self.cap = ColourUSB()
        print(f"Opened {self.cap.description}")
        if not with_ir:
            print("NOTE: on a Mac the infrared stream can't run next to the colour one (see mac/README.md); "
                  "the infrared signals are off.")
        self.ir = open_infrared(ir_device) if with_ir else None
        try:
            self.feed = LocalFaceFeed()
        except Exception:
            self.cap.release()
            if self.ir:
                self.ir.close()
            raise
        self.client = None
        self.det = self.feed.det
        self.worker = start_phone_worker(phone_score, self.ir is not None)
        self.phone_score = phone_score
        # the camera's AI is not running (the mesh runs on this Mac), so its overheating warning never applies
        self.started = math.inf
        self._ir_frame = None

    def step(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        now = time.time()
        self.worker.submit(frame.copy())
        if self.ir is not None:
            fresh = self.ir.take()
            if fresh is not None:
                self._ir_frame = fresh
                self.worker.submit(fresh, "ir")
        self.feed.poll(now, frame)
        return engine.Step(now, frame, self._ir_frame, signals(self.det, self.worker, now), self.raw(now))

    def raw(self, now):
        if self.ir is not None:
            return super().raw(now)
        det, feed, worker = self.det, self.feed, self.worker
        mesh = feed.features if feed.features is not None else (None, None, None)
        base = det.base if det.base is not None else (None, None, None)
        return {
            "color_phone_seen": int(worker.is_on(now, "rgb")), "phone_seen": None, "phone_held": None,
            "face_seen": int(not feed.face_lost(now)), "head": det.head, "eyes": det.eyes,
            "pitch_deg": mesh[0], "lid": mesh[1], "eye_opening": mesh[2],
            "pitch_base": base[0], "lid_base": base[1], "eye_opening_base": base[2],
            "color_phone_now": worker.latest["rgb"], "phone_now": None,
            "hands": None, "hand_distance": None,
        }

    def close(self):
        self.cap.release()
        if self.ir is not None:
            self.ir.close()
        self.worker.stop()
        self.feed.mesh.close()
