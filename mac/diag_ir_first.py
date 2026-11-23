"""MAC ONLY diagnostic: can the infrared stream keep running if it is opened BEFORE the
colour stream is taken over USB?

    sudo .venv/bin/python mac/diag_ir_first.py

Close the dashboard and every other camera app first. Prints frame counts for
both streams over 8 seconds.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "detector"), str(ROOT), str(ROOT / "logiinsight_api")]

from camera_utils import list_cameras, open_camera  # noqa: E402
from mac.engine_mac import ColourUSB, looks_like_ir  # noqa: E402
from phone_detect import IR_SIZES, Stream  # noqa: E402

print("1) cameras macOS offers BEFORE the colour stream is taken:")
cams = list_cameras()
for c in cams:
    print(f"   [{c['index']}] {c['width']}x{c['height']}  {'grey' if c['gray'] else 'colour'}")
ir_cams = [c for c in cams if c["gray"] and c["height"] > c["width"]]
if not ir_cams:
    sys.exit("   no grey portrait camera: is the LogiInsight plugged in and nothing else using it?")
idx = ir_cams[0]["index"]

cap, info = open_camera(idx, None, *IR_SIZES["512x640"], "MJPG", 30)
ir = Stream("ir", cap, "ir", 0)
time.sleep(1.5)
n = 0
t_end = time.time() + 2
while time.time() < t_end:
    f = ir.take()
    if f is not None:
        n += 1
        last = f
    time.sleep(0.005)
print(f"2) infrared alone: {n} frames in 2 s, looks like IR: {looks_like_ir(last) if n else '-'}")

print("3) now taking the colour stream over USB ...")
try:
    colour = ColourUSB()
    print(f"   {colour.description}")
except Exception as e:
    ir.close()
    sys.exit(f"   colour failed: {e}")

n_ir = n_col = 0
t_end = time.time() + 8
while time.time() < t_end:
    ok, _ = colour.read()
    n_col += bool(ok)
    if ir.take() is not None:
        n_ir += 1
print(f"4) together for 8 s: colour {n_col} frames, infrared {n_ir} frames")
print("   RESULT:", "BOTH WORK -> open infrared first" if n_ir > 20 and n_col > 20 else
      "infrared stops once the colour stream is taken" if n_col > 20 else "colour failed")
colour.release()
ir.close()
