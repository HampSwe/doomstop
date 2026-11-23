"""Live video with face detection, for the LogiInsight webcam (or any webcam).

Face boxes come from one of two sources:

  device  the camera's own AI, read as JSON over SSH/SFTP (LogiInsight only).
          Needs the USB network link (PC address 192.168.1.20/24) and the kit
          card's login in LOGIWEBCAM_USER / LOGIWEBCAM_PASSWORD.
  local   OpenCV's YuNet detector running on this PC. Works with any webcam.

The default (--source auto) uses the camera's AI when it is reachable and
falls back to the local detector otherwise.

    python detector/face_stream.py                 # pick a camera, auto source
    python detector/face_stream.py --device 1      # skip the camera prompt
    python detector/face_stream.py --mesh          # + 478-point face mesh (device AI only)
    python detector/face_stream.py --source local  # never touch the network link

Press q or Esc in the window to quit.
"""

import argparse
import os
import socket
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # hide the per-index probe warnings

import cv2  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent      # the repository folder
sys.path.insert(0, str(ROOT / "logiinsight_api"))

from camera_utils import CameraNotFoundError, describe, list_cameras, open_camera  # noqa: E402
from logiwebcam import AiStalenessMonitor, LogiWebcamClient, LogiWebcamConnectionError  # noqa: E402

CAMERA_IP = "192.168.1.10"
YUNET_MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
WINDOW = "face_stream (q / Esc to quit)"
GREEN, YELLOW, CYAN = (0, 255, 0), (0, 255, 255), (255, 200, 0)


def drives_camera_ai(index, client, seconds=3.0):
    """True if reading camera `index` makes the camera AI's timestamp advance.

    The LogiInsight AI only produces results while its own colour stream is
    being read, so this tells it apart from a laptop's built-in webcam.
    """
    try:
        cap, _ = open_camera(index, width=640, height=480, fourcc="MJPG")
    except CameraNotFoundError:
        return False
    seen = set()
    deadline = time.time() + seconds
    try:
        while time.time() < deadline and len(seen) < 4:
            cap.read()
            ts = (client.get_latest_ai_data() or {}).get("timestamp_ms")
            if ts is not None:
                seen.add(ts)
    finally:
        cap.release()
    return len(seen) >= 4  # one stale value is expected; several means it is live


def pick_camera(client=None):
    """Choose a camera: the one feeding the camera AI if connected, else ask."""
    print("Scanning for cameras ...")
    cams = list_cameras()
    if not cams:
        raise CameraNotFoundError("No camera delivered a frame. Is it plugged in, and not "
                                  "in use by another app (Teams, Zoom, browser)?")
    for c in cams:
        c["ir"] = c["gray"] or c["height"] > c["width"]
        kind = "portrait/grey: the LogiInsight IR stream" if c["ir"] else "colour"
        print(f"  [{c['index']}] {c['width']}x{c['height']}  ({kind})")
    if len(cams) == 1:
        return cams[0]["index"]
    if client:
        print("Finding the LogiInsight colour camera ...")
        for c in cams:
            if not c["ir"] and drives_camera_ai(c["index"], client):
                print(f"  camera index {c['index']} feeds the camera AI, using it.")
                return c["index"]
        print("  none of them made the camera AI update; choose one yourself.")
    valid = {c["index"] for c in cams}
    while True:
        try:
            choice = input("Camera index to open: ").strip()
        except EOFError:
            return cams[0]["index"]
        if choice.isdigit() and int(choice) in valid:
            return int(choice)


def connect_device_ai(mesh):
    """Return a connected LogiWebcamClient, or (None, reason) if unavailable."""
    if not os.environ.get("LOGIWEBCAM_USER") or os.environ.get("LOGIWEBCAM_PASSWORD") is None:
        return None, "LOGIWEBCAM_USER / LOGIWEBCAM_PASSWORD are not set"
    try:  # fail fast instead of waiting out the SSH timeout
        socket.create_connection((CAMERA_IP, 22), timeout=2).close()
    except OSError:
        return None, f"{CAMERA_IP}:22 is unreachable (is the USB adapter set to 192.168.1.20/24?)"
    try:
        client = LogiWebcamClient(ip=CAMERA_IP)
        client.connect()
    except LogiWebcamConnectionError as e:
        return None, str(e)
    client.set_ai_config(enable_face_detection=True, enable_face_mesh=mesh)
    return client, None


class LocalDetector:
    """YuNet on the PC, returning faces in the same shape as the camera's JSON."""

    def __init__(self):
        if not YUNET_MODEL.exists():
            raise FileNotFoundError(f"Missing {YUNET_MODEL}")
        self._net = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320), score_threshold=0.7)
        self._size = None

    def faces(self, frame):
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._net.setInputSize((w, h))
            self._size = (w, h)
        _, found = self._net.detect(frame)
        if found is None:
            return []
        return [{"bbox": [float(v) for v in f[:4]], "confidence": float(f[-1]), "landmarks": []}
                for f in found]


def draw_faces(frame, faces, mesh):
    for face in faces:
        x, y, w, h = (int(round(v)) for v in face["bbox"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
        cv2.putText(frame, f"face {face.get('confidence', 0):.2f}", (x, max(y - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
        if mesh:
            for px, py, *_ in face.get("landmarks") or []:
                cv2.circle(frame, (int(px), int(py)), 1, CYAN, -1)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", type=int, help="camera index (default: list and ask)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--source", choices=("auto", "device", "local"), default="auto",
                   help="where face boxes come from (default: auto)")
    p.add_argument("--mesh", action="store_true", help="478-point face mesh (device AI only)")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until quit)")
    p.add_argument("--no-display", action="store_true", help="no window; print a summary")
    args = p.parse_args()

    client, local = None, None
    if args.source != "local":
        client, reason = connect_device_ai(args.mesh)
        if client is None and args.source == "device":
            sys.exit(f"ERROR: camera AI unavailable: {reason}")
        if client is None:
            print(f"Camera AI unavailable ({reason}); using the local detector.")

    try:
        index = args.device if args.device is not None else pick_camera(client)
        cap, info = open_camera(index, width=args.width, height=args.height, fourcc="MJPG", fps=30)
    except CameraNotFoundError as e:
        if client:
            client.disconnect()
        sys.exit(f"ERROR: {e}")
    print(f"Opened {describe(info)}")

    if client is None:
        local = LocalDetector()
    source = "camera AI" if client else "local YuNet"
    print(f"Face source: {source}. Press q or Esc in the window to quit.")

    monitor = AiStalenessMonitor()
    fps, t_prev, n_frames, n_with_face = 0.0, time.time(), 0, 0
    try:
        while not args.frames or n_frames < args.frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                if not args.no_display and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

            if client:
                data = client.get_latest_ai_data()
                warning = monitor.update(data)
                if warning:
                    print(f"WARNING: the camera AI is not updating, so camera index {index} is "
                          "probably not the LogiInsight colour camera. Quit and try another "
                          "--device index.")
                faces = (data or {}).get("faces", [])
            else:
                faces = local.faces(frame)

            n_frames += 1
            n_with_face += bool(faces)
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            if args.no_display:
                continue
            draw_faces(frame, faces, args.mesh)
            cv2.putText(frame, f"{source} | faces: {len(faces)} | {fps:.0f} fps", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if client:
            client.disconnect()
    print(f"{n_frames} frames, face found in {n_with_face}, ~{fps:.0f} fps")


if __name__ == "__main__":
    main()
