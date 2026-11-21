# macOS only: LogiInsight colour camera on a Mac

> **Use this folder only on a Mac.** On Windows and Linux the colour camera is
> listed normally: run `dashboard.py` / `collect.py` directly and ignore `mac/`.

## The problem

macOS only lists the LogiInsight's **infrared** stream (`LVNext CPD Camera`).
The colour stream (`LVNext Camera`) reaches the Mac over USB, and macOS even
loads its driver, but it never appears in the camera list. So
`cv2.VideoCapture` can't open it. `dashboard.py` and `collect.py` then pick
the MacBook's own webcam, and the camera's face AI never starts.

Check whether your Mac has this problem:
```bash
system_profiler SPCameraDataType | grep -E "^\s{4}\S"
```
If you see `LVNext CPD Camera` but no `LVNext Camera`, you have it.

## The fix

Read the colour stream **straight over USB** with `pyuvc` (libusb), bypassing
macOS's camera list. That needs `sudo`. While it holds the camera, macOS lets
go of the whole device, including its USB network link, so the camera's own
face AI can't be reached either. The **478-point face mesh is computed on the
Mac** with MediaPipe's FaceLandmarker instead. It uses the same landmark layout,
and is fed to `look_down.LookDownDetector` exactly as the camera's mesh is.

`mac/engine_mac.py` is a drop-in `engine.Engine` that does this, and
`mac/run_mac.py` swaps it in, so `dashboard.py` and `collect.py` run unchanged.

## Setup (once)

From the repo root:
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r mac/requirements-mac.txt
```

## Run

Close every other camera app first (Zoom, FaceTime, browser tabs, other
scripts), and plug the camera in directly, not through a hub.

```bash
sudo .venv/bin/python mac/run_mac.py dashboard
```
```bash
sudo .venv/bin/python mac/run_mac.py collect --person NAME
```

Any other options are passed straight to `dashboard.py` / `collect.py`.
`--device` is ignored on the Mac, because the colour stream is found on USB.

**First check** that the colour stream works at all. This saves one frame to
`mac/probe_colour.jpg`:
```bash
sudo .venv/bin/python mac/probe_colour.py
```

## What's different from Windows/Linux

| | Windows / Linux | Mac (this folder) |
|---|---|---|
| colour stream | `cv2.VideoCapture` | `pyuvc` over USB, needs `sudo` |
| face mesh | the camera's own AI over SSH | MediaPipe on the Mac (no login needed) |
| infrared stream + its signals (`phone`, `phone_in_hand`, `gaze_down_x_phone`) | yes | only if macOS still offers it while `pyuvc` runs; otherwise off (`None`) and the panel says "waiting for infrared" |
| overheating warning | after 8 min of camera AI | off (the camera's AI isn't running) |

**Known caveats:**
- **Untested thresholds.** The look-down thresholds were tuned on the
  camera's mesh, and MediaPipe's depth scale may differ a little. Use `d`
  (calibrate on a looking-down pose) if FACE/GAZE fire too easily or not at all.
- **Possibly flipped head tilt.** If FACE switches on when you look *up*,
  MediaPipe's depth runs the other way: say so and flip it in
  `LocalMesh.face` (`p.z * w` -> `-p.z * w`).
- **No network link while it runs.** The camera's USB network link and
  `check_connection.py` don't work while `sudo` holds the camera. They come
  back when it quits.
- **The colour stream sometimes stalls.** Quit, unplug and replug the camera,
  wait about 20 s, and start again.
