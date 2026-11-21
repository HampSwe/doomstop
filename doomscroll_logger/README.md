# Doomscroll detector (data logger + logistic regression)

Records a continuous stream of feature rows (4 per second) from the camera,
microphone and keyboard while the test user presses **SPACE** to mark
whether they are currently *attentive* or *doomscrolling*. Then `train.py`
fits a logistic regression and tells you which signal gives a doomscroller away.

## Setup (once)

```bash
cd doomscroll_logger      # from the repo root
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

macOS permissions: run it from your own Terminal/iTerm and allow, for **that
terminal app**:
- System Settings > Privacy & Security > **Camera**
- ... > **Microphone**
- ... > **Accessibility** and **Input Monitoring** (for SPACE + keystroke counts)

Restart the terminal after granting. The recorder prints a warning for any of
these that is missing.

## Recording

```bash
.venv/bin/python record.py --list-cameras     # find the Logitech's index
.venv/bin/python record.py --camera 1         # record; SPACE switches label, Ctrl+C stops
```

The status bar shows the current label (green = attentive, red = doomscrolling),
seconds since the last switch and the live signals. Each session is saved to
`data/session_<timestamp>.csv`, flushed every row.

**Protocol tips**
- Press SPACE *as* you pick up / put down the phone. The first 1 s after
  each switch is dropped in training anyway (`--guard`).
- Switch every 20-60 s, and vary it: sometimes type during attention,
  sometimes sit still; scroll with sound on and off. Otherwise the model
  learns your script, not doomscrolling.
- Record several sessions (3+), ideally different people and lighting.
  Validation then holds out whole sessions.

Testing without hardware or teammate models:
```bash
.venv/bin/python record.py --fake                       # invented signals, switch with SPACE
.venv/bin/python record.py --fake --no-keyboard --auto-switch 10 --duration 60
```

## Training

```bash
.venv/bin/python train.py                  # all real sessions in data/
.venv/bin/python train.py --include-fake   # also the --fake ones
```

Prints held-out balanced accuracy, ROC AUC, the confusion matrix and the
coefficients; writes `reports/coefficients.png` (the slide),
`reports/timeline.png` (model probability vs. your SPACE label) and
`models/doomscroll_lr.joblib`.

Validation never tests on rows next to training rows (they are near-copies):
leave-one-session-out with 3+ sessions, otherwise held-out label segments.

Coefficients are per +1 standard deviation. The 1 s and 5 s versions of a
signal are strongly correlated, so they can get opposite signs. Read them
as a pair.

## Plugging in teammates' models

Edit **`doomlog/teammates.py`** only:

| function | input | return |
|---|---|---|
| `head_is_down(frame)` | BGR frame (numpy) | `True` / `False` / `None` |
| `phone_in_frame(frame)` | same frame | `True` / `False` / `None` |
| `audio_reel_score(samples, sr)` | ~0.25 s mono float32 | float 0..1 or `None` |

The recorder owns the camera and passes each frame to both camera models
(`--cam-fps`, default 10 calls/s), so they never fight over the device.
`None` means "no opinion" and is logged as missing. If a function raises,
the error is printed once and recording continues.

If a teammate's model must run as a separate process instead, have it
write into the same functions (e.g. read the latest value from a socket);
the rest stays unchanged.

## Logged columns

| column | meaning |
|---|---|
| `label`, `label_name` | 0 attentive / 1 doomscrolling (from SPACE) |
| `secs_since_switch` | time since the last SPACE |
| `head_down`, `head_down_frac_1s/5s`, `head_down_streak_s` | latest, share of frames down in window, seconds continuously down |
| `phone_seen`, `phone_frac_1s/5s`, `phone_streak_s` | same for the phone detector |
| `keys_per_s_1s/5s`, `secs_since_key` | keystroke rate (SPACE excluded) and idle time; key identities are never recorded |
| `audio_rms_db_1s/5s`, `audio_centroid_hz_1s`, `audio_zcr_1s` | loudness, brightness, noisiness; no audio is saved |
| `reel_score_1s` | teammate audio model, averaged |
| `cam_fps` | health check: classifier calls per second actually achieved |
