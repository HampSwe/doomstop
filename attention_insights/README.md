# attention_insights

Find which topics of a talk got particularly high or low audience attention.

```
audio ──transcribe.py──▶ words.json ─┐
                                     ├─insights.py─▶ report.md, timeline.png, insights.json
attention.csv (t, attention∈[0,1]) ──┘
```

1. **transcribe.py**: faster-whisper with word-level timestamps.
2. **insights.py**:
   - splits the words into 12–30 s chunks at sentence ends or pauses;
   - smooths and linearly detrends the attention signal to remove fatigue drift;
   - averages attention per chunk, shifted by the reaction `--lag`, and z-scores it;
   - picks and merges the high and low episodes, dropping anything under 10 s;
   - asks Claude (`claude-opus-5-5`) for topic labels and insights, or falls back to keywords;
   - plots the timeline.

For a demo without real attention data:

- **manuscript.md**: a ~3-minute script whose sections are engaging or dull on purpose.
- **record.py**: records the microphone to a WAV file. It stops on Ctrl+C or after 6 s of silence.
- **simulate_attention.py**: builds a plausible attention curve for the recording. It knows how engaging each manuscript section is, and adds reaction delay, sluggish response, fatigue, noise and one spurious spike.

```bash
pip install faster-whisper anthropic soundfile sounddevice numpy matplotlib
python attention_insights/record.py                       # -> data/talk.wav
python attention_insights/transcribe.py attention_insights/data/talk.wav attention_insights/data/words.json
python attention_insights/simulate_attention.py attention_insights/data/words.json
python attention_insights/insights.py attention_insights/data/words.json attention_insights/data/attention.csv
```

The LLM summaries need `ANTHROPIC_API_KEY`. Without it you get keyword labels.
