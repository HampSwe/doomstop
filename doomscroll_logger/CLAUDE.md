# CLAUDE.md: doomscroll_logger

Instructions for a Claude session working in this folder. Human docs are in
`README.md`. Read it for the column list and recording protocol.

## What this is

A deliberately over-engineered LauzHack joke. We detect whether a user in a
Zoom meeting is doomscrolling on their phone, using **logistic regression**
over hand-made features. This folder is the **data-collection and training
tool**:

- `record.py`: records a continuous session at 4 rows/s to
  `data/session_<timestamp>.csv`. The test user presses **SPACE** to toggle
  the label: 0 = attentive, 1 = doomscrolling.
- `train.py`: loads the sessions, fits logistic regression with
  group-wise validation, prints metrics and coefficients, and writes
  `reports/coefficients.png`, `reports/timeline.png` and
  `models/doomscroll_lr.joblib`.

Signals are split across teammates. Their models are **not in this folder
yet**. They plug in through `doomlog/teammates.py`.

## Layout

```
record.py            CLI entry: threads + 4 Hz loop + CSV writer + status line
train.py             CLI entry: load -> filter -> CV -> fit -> plots/model
doomlog/
  teammates.py       THE plug-in point: head_is_down, phone_in_frame, audio_reel_score
  camera.py          owns the webcam (OpenCV), calls both camera models per frame
  audio.py           mic via sounddevice; RMS dB, spectral centroid, ZCR, reel score
  keyboard.py        pynput global listener; counts keys, SPACE toggles label
  tracks.py          thread-safe rolling buffers -> fractions / streaks / rates / means
  fake.py            synthetic label-dependent signals for testing without hardware
```

Data flow: workers (threads) push timestamped samples into tracks (`tracks.py`).
The main loop in `record.py` reads the tracks every 1/hz s and builds one row
(`build_row`). `COLUMNS` in `record.py` is the CSV schema.

## Setup and running

```bash
cd doomscroll_logger
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**Your shell almost certainly cannot use the camera, microphone or global
keyboard.** macOS grants those per terminal app, and Claude's sandboxed shell
doesn't have them. You'll see `not authorized to capture video`,
`PortAudio error -9986` and `This process is not trusted!`. That is expected,
not a bug. Verify changes in fake mode, hands-free:

```bash
for i in 1 2 3; do .venv/bin/python record.py --fake --no-keyboard --auto-switch 7 --duration 45 --quiet > /dev/null & .venv/bin/python -c "import time; time.sleep(1.2)"; done; wait
.venv/bin/python train.py --include-fake
```

Three sessions are needed for leave-one-session-out validation. They are
started 1.2 s apart because the session name has 1 s resolution. Fake mode
gives roughly 0.85-0.97 balanced accuracy. Its numbers are invented, so it
only proves the pipeline runs. Delete `data/*_fake.csv` afterwards, or leave
it: `train.py` skips `_fake` sessions unless you pass `--include-fake`.

For real recordings, ask the user to run `record.py` in their own terminal,
after granting Camera, Microphone, Accessibility and Input Monitoring in
System Settings > Privacy & Security. Use `record.py --list-cameras` to find
the Logitech's index.

## Plugging in a teammate's model (the most likely task)

Edit only the function bodies in `doomlog/teammates.py`, keeping these signatures:

| function | input | return |
|---|---|---|
| `head_is_down(frame)` | BGR `np.ndarray` (H, W, 3) | `True`/`False`/`None` |
| `phone_in_frame(frame)` | the same frame | `True`/`False`/`None` |
| `audio_reel_score(samples, sr)` | mono float32, ~0.25 s, sr=16000 | float 0..1 or `None` |

- `None` means "no opinion" (no face, model not loaded). It is logged as
  missing, never as False.
- An exception is printed once and logged as missing. Recording continues.
- Both camera functions run in the camera thread at `--cam-fps` (default
  10). A slow model just lowers `cam_fps` (logged per row as a health
  check). Load models once at import time, not per call.
- If a teammate's model runs as a separate process, have the function return
  the latest value received from it (socket, shared file, etc.). Don't open
  the camera a second time: the recorder owns it.

## Adding a new signal

1. Add a track in `record.py:main` (`BinaryTrack` / `EventTrack` / `NumericTrack`).
2. Add a worker that feeds it (pattern: `camera.py`; it needs `start`, `stop`,
   `.name` and `.error`). Add a fake version in `fake.py` so tests still cover it.
3. Add the feature columns to `COLUMNS` and `build_row` in `record.py`.
4. `train.py` picks up every non-meta numeric column automatically. Only
   touch `META` if the new column is not a model feature.

## Invariants: do not break these

- **The label key never counts as a keystroke** (`keyboard.py`). Otherwise
  the label leaks into `keys_per_s_*`.
- **Never record key identities, audio or video.** Only counts, timestamps
  and summary numbers. The test users are real people in real meetings.
- **Never validate with a random row-level split.** Neighbouring rows are
  near-duplicates, so a random split gives fake 99% accuracy. Keep
  leave-one-session-out (3+ sessions) or held-out label segments.
- Rows less than `--guard` s (1 s) after a switch and the first `--warmup` s
  (5 s) are dropped in training. The label lags the behaviour, and the 5 s
  windows need time to fill.
- The CSV is flushed every row, so a crash loses at most one row. Keep it
  that way.
- The 1 s and 5 s versions of a feature are correlated, so their coefficients
  can have opposite signs. Explain that; don't "fix" it.

## Style

Plain Python 3.10+, standard library plus numpy, pandas, scikit-learn,
matplotlib, opencv, sounddevice and pynput. Keep it small and readable. The
joke is the over-engineering of the *idea*, not of the code.
