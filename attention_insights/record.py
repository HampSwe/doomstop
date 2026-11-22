"""Record the manuscript from the default microphone to a WAV file.

    python attention_insights/record.py [out.wav] [--max 60] [--silence 4]

Starts after a 3 s countdown. Stops on Ctrl+C, after --max seconds, or after
--silence seconds of quiet once you've started talking.
"""

import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

RATE = 16000  # Whisper works at 16 kHz anyway


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="attention_insights/data/talk.wav")
    ap.add_argument("--max", type=float, default=60.0)
    ap.add_argument("--silence", type=float, default=4.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Microphone: {sd.query_devices(kind='input')['name']}")
    for i in (3, 2, 1):
        print(f"Starting in {i}...", flush=True)
        time.sleep(1)
    print("RECORDING - read the manuscript. Ctrl+C or stay quiet to stop.", flush=True)

    q: queue.Queue[np.ndarray] = queue.Queue()
    stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                            callback=lambda d, f, t, s: q.put(d.copy()))
    frames, spoke, quiet, noise_floor = 0, False, 0.0, None
    with sf.SoundFile(out, mode="w", samplerate=RATE, channels=1, subtype="PCM_16") as f, stream:
        try:
            while frames / RATE < a.max:
                block = q.get()
                f.write(block)
                frames += len(block)
                rms = float(np.sqrt(np.mean(block ** 2)))
                if noise_floor is None and frames > RATE // 2:
                    noise_floor = max(rms, 1e-4)
                if noise_floor is None:
                    continue
                if rms > noise_floor * 4:
                    spoke, quiet = True, 0.0
                elif spoke:
                    quiet += len(block) / RATE
                    if quiet >= a.silence:
                        print(f"\n{a.silence:.0f} s of silence - stopping.")
                        break
                if frames % (RATE * 10) < len(block):
                    print(f"  {frames / RATE:5.0f} s", flush=True)
        except KeyboardInterrupt:
            print("\nStopped.")
    print(f"Saved {out} ({frames / RATE:.1f} s)")


if __name__ == "__main__":
    main()
