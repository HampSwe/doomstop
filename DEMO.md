# The demo: live audience attention for the presenter

```
CAMERA COMPUTER (Windows, LogiInsight, the audience)     PRESENTER MAC
 engine -> attention model -> score, 10 a second  ──▶     attention/live_insights.py
 call/score_sender.py                   iPhone hotspot      during the talk: the LED eye only
                                         (UDP)              records the presenter's microphone
                                                            close the eye = the talk is over:
                                                              transcribe -> topics
                                                              feedback card: attention graph,
                                                              what got / lost attention
```

The camera computer should be the **Windows** laptop: the model uses the infrared phone signals, which a
Mac can't read next to the colour camera (see `mac/README.md`).

## Once, before the day (on normal Wi-Fi, not the hotspot)

Both computers: clone the repo, `git switch demo`, make the venv at the repo root.

**Presenter Mac:**
```bash
.venv/bin/pip install -r requirements.txt matplotlib faster-whisper soundfile sounddevice
```
- Run the demo command below once with `--record`: it downloads the speech model (about 150 MB).
- Optional, for topic names written by an LLM instead of keywords (the card then waits for the LLM, likely tens of seconds on
  a laptop; not measured here):
  `brew install ollama`, `ollama serve`, `ollama pull qwen2.5:3b` (about 2 GB).
- The first time, macOS asks whether Terminal may use the microphone: allow it.

**Camera computer (Windows):** `pip install -r requirements.txt` and the camera login (`LOGIWEBCAM_USER`,
`LOGIWEBCAM_PASSWORD`), as for `dashboard.py`. Check the camera works: `python dashboard.py`.

## On the day

1. Both computers join the iPhone hotspot ("Maximise Compatibility" off, phone plugged in).
2. **Presenter Mac**, start the eye:
   ```bash
   .venv/bin/python attention/live_insights.py --udp 8767 --bind 0.0.0.0 --record
   ```
   It prints this Mac's address, e.g. `172.20.10.3`.
3. **Camera computer**, start the score:
   ```bash
   .venv\Scripts\python.exe call\score_sender.py --to 172.20.10.3
   ```
   The person in front of the camera looks at the screen for the first second (it measures their normal pose).
   Sending then starts by itself. With `--wait` it prints `DONE. PRESS ENTER TO START` instead and sends
   nothing until Enter is pressed.
4. Give the talk. The eye turns from red to green with the audience's attention; it says **no signal** if the
   scores stop for 3 s (it carries on by itself when they come back).
5. **End of the talk: close the eye** (its × box, or Esc). The talk is transcribed while you speak, so the
   feedback card comes up in about a second: the attention over the talk, the moments that got and lost
   attention with their topic and what was said, and OPEN REPORT for the full report.

The talk counts from the **first score that arrives** to closing the eye: whatever is said before the camera
computer starts sending (setting up, "can you hear me") is left out of the feedback, and the card starts at 0:00.

Test without the camera: `.venv\Scripts\python.exe call\score_sender.py --to 172.20.10.3 --simulate` sends a made-up score.

## Options worth knowing

| | |
|---|---|
| `score_sender.py --model models/attention_model_face_down.json` | another fitted model (default: Hampus's latest) |
| `score_sender.py --dashboard --popup` | on the camera computer: also show the dashboard, and pop up the DOOMSCROLL DETECTED card with its voice |
| `live_insights.py --whisper small.en` | better transcription, about 3x slower than `base.en` |
| `live_insights.py --no-llm` | keyword topics, don't ask Ollama |
| `live_insights.py --no-highlights` | the feedback card with the graph only |
| `score_sender.py --mac` (with `sudo`) | camera on a Mac: colour only, infrared signals off |

Every score is also saved on the camera computer in `call/scores/`, and the recording, transcript and
report on the presenter Mac in `attention_insights/data/live/`.
